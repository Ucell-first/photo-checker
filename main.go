// @title Image Recognition API
// @version 1.1
// @description API service for comparing images
// @termsOfService http://swagger.io/terms/
// @contact.name API Support
// @contact.email support@imageapi.com
// @license.name Apache 2.0
// @license.url http://www.apache.org/licenses/LICENSE-2.0.html
// @host localhost:8080
// @BasePath /
// @securityDefinitions.apikey ApiKeyAuth
// @in header
// @name Authorization
package main

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	_ "photot/docs"

	"github.com/disintegration/imaging"
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/patrickmn/go-cache"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
)

// ImageDatabase holds all known image hashes and provides thread-safe access
type ImageDatabase struct {
	hashes map[string]imageInfo
	mutex  sync.RWMutex
	cache  *cache.Cache
}

// imageInfo stores metadata about an image
type imageInfo struct {
	Filename  string    `json:"filename"`
	Hash      string    `json:"hash"`
	AddedAt   time.Time `json:"added_at"`
	Thumbnail string    `json:"thumbnail,omitempty"`
}

// RecognizeResponse is the API response for image recognition
type RecognizeResponse struct {
	Result           string  `json:"result"`     // "OK" or "NOT OK"
	Similarity       float64 `json:"similarity"` // Similarity percentage
	MatchedImage     string  `json:"matched_image,omitempty"`
	ProcessingTimeMs int64   `json:"processing_time_ms"`
}

// NewImageDatabase creates a new image database instance
func NewImageDatabase() *ImageDatabase {
	return &ImageDatabase{
		hashes: make(map[string]imageInfo),
		cache:  cache.New(5*time.Minute, 10*time.Minute),
	}
}

// computeDCTHash computes a perceptual hash using DCT (Discrete Cosine Transform)
// This is a simplified version simulating DCT with average gradients
// computeDCTHash rasmning perceptual hash qiymatini hisoblaydigan funksiya
func computeDCTHash(img image.Image) string {
	// Rasmni 32x32 o'lchamga keltirish
	resized := imaging.Resize(img, 32, 32, imaging.Lanczos)
	gray := imaging.Grayscale(resized)

	// Rasmni 8x8 bloklarga ajratib, har bir blok qiymatlarini saqlash
	const blockSize = 8
	const numBlocks = 16 // 4x4 grid of blocks
	blockValues := make([]float64, numBlocks)

	// Har bir blok uchun o'rtacha qiymatni hisoblash
	blockIndex := 0
	for by := 0; by < 4; by++ {
		for bx := 0; bx < 4; bx++ {
			var sum float64
			var count int

			// Blok ichidagi piksellarni yig'ish
			for y := by * blockSize; y < (by+1)*blockSize && y < 32; y++ {
				for x := bx * blockSize; x < (bx+1)*blockSize && x < 32; x++ {
					c := color.GrayModel.Convert(gray.At(x, y)).(color.Gray)
					sum += float64(c.Y)
					count++
				}
			}

			if count > 0 {
				blockValues[blockIndex] = sum / float64(count)
				blockIndex++
			}
		}
	}

	// O'rtacha qiymatni hisoblash
	var sum float64
	for _, val := range blockValues {
		sum += val
	}
	avg := sum / float64(len(blockValues))

	// Hash yaratish: har bir blok qiymati o'rtachadan katta bo'lsa 1, aks holda 0
	var hash strings.Builder
	for _, val := range blockValues {
		if val >= avg {
			hash.WriteString("1")
		} else {
			hash.WriteString("0")
		}
	}

	// Qo'shimcha detallar qo'shamiz:
	// Gorzontal va vertikal farqlarni ham qo'shish
	for y := 0; y < 8; y++ {
		for x := 0; x < 7; x++ {
			c1 := color.GrayModel.Convert(gray.At(x*4, y*4)).(color.Gray)
			c2 := color.GrayModel.Convert(gray.At((x+1)*4, y*4)).(color.Gray)

			if c1.Y > c2.Y {
				hash.WriteString("1")
			} else {
				hash.WriteString("0")
			}
		}
	}

	// Diagonal farqlar
	for i := 0; i < 7; i++ {
		c1 := color.GrayModel.Convert(gray.At(i*4, i*4)).(color.Gray)
		c2 := color.GrayModel.Convert(gray.At((i+1)*4, (i+1)*4)).(color.Gray)

		if c1.Y > c2.Y {
			hash.WriteString("1")
		} else {
			hash.WriteString("0")
		}
	}

	return hash.String()
}

// generateThumbnail creates a small thumbnail version of the image
func generateThumbnail(img image.Image) string {
	thumbnail := imaging.Resize(img, 100, 0, imaging.Lanczos)
	var buf bytes.Buffer
	// Use imaging.JPEG format with default quality (no options parameter)
	err := imaging.Encode(&buf, thumbnail, imaging.JPEG)
	if err != nil {
		return ""
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

// hammingDistance calculates bit differences between two binary strings
func hammingDistance(hash1, hash2 string) (int, error) {
	if len(hash1) != len(hash2) {
		return 0, fmt.Errorf("hash lengths do not match: %d vs %d", len(hash1), len(hash2))
	}
	distance := 0
	for i := 0; i < len(hash1); i++ {
		if hash1[i] != hash2[i] {
			distance++
		}
	}
	return distance, nil
}

// LoadImages loads all images from the specified directory
func (db *ImageDatabase) LoadImages(imageDir string) error {
	if _, err := os.Stat(imageDir); os.IsNotExist(err) {
		return fmt.Errorf("images directory does not exist: %s", imageDir)
	}

	files, err := os.ReadDir(imageDir)
	if err != nil {
		return fmt.Errorf("could not read directory: %s", err)
	}

	// We'll use a wait group to load images in parallel
	var wg sync.WaitGroup
	threadLimit := make(chan struct{}, 4) // Limit concurrent threads

	for _, file := range files {
		if file.IsDir() {
			continue
		}

		// Skip non-image files based on extension
		ext := strings.ToLower(filepath.Ext(file.Name()))
		if !isImageFile(ext) {
			continue
		}

		wg.Add(1)
		threadLimit <- struct{}{} // Acquire thread slot

		go func(fileName string) {
			defer wg.Done()
			defer func() { <-threadLimit }() // Release thread slot

			path := filepath.Join(imageDir, fileName)
			img, err := imaging.Open(path)
			if err != nil {
				log.Printf("Could not open file %s: %v", path, err)
				return
			}

			hash := computeDCTHash(img)
			thumbnail := generateThumbnail(img)

			db.mutex.Lock()
			db.hashes[hash] = imageInfo{
				Filename:  fileName,
				Hash:      hash,
				AddedAt:   time.Now(),
				Thumbnail: thumbnail,
			}
			db.mutex.Unlock()

			log.Printf("Loaded image: %s", fileName)
		}(file.Name())
	}

	wg.Wait()
	log.Printf("Loaded %d images into the database", len(db.hashes))
	return nil
}

// isImageFile checks if a file extension belongs to a supported image format
func isImageFile(ext string) bool {
	supportedExts := map[string]bool{
		".jpg":  true,
		".jpeg": true,
		".png":  true,
		".gif":  true,
		".bmp":  true,
		".tiff": true,
		".webp": true,
	}
	return supportedExts[ext]
}

// FindMatch searches for a matching image in the database
func (db *ImageDatabase) FindMatch(img image.Image, similarityThreshold float64) (bool, string, float64) {
	uploadedHash := computeDCTHash(img)

	db.mutex.RLock()
	defer db.mutex.RUnlock()

	// Agar database bo'sh bo'lsa, hech qanday o'xshashlik yo'q
	if len(db.hashes) == 0 {
		return false, "", 0.0
	}

	bestMatch := ""
	minDistance := len(uploadedHash) // Maksimal masofa

	for hash, info := range db.hashes {
		distance, err := hammingDistance(uploadedHash, hash)
		if err != nil {
			continue
		}

		if distance < minDistance {
			minDistance = distance
			bestMatch = info.Filename
		}
	}

	// O'xshashlik foizini hisoblash
	maxDistance := len(uploadedHash)
	similarity := 100.0 - (float64(minDistance) / float64(maxDistance) * 100.0)

	// DEBUG: Konsol orqali qiymatlarni ko'rib chiqish
	log.Printf("Uploaded hash: %s", uploadedHash[:20]+"...")
	log.Printf("Best match: %s, similarity: %.2f%%, threshold: %.2f%%", bestMatch, similarity, similarityThreshold)

	// Eng muhim qism: similarity ostonasidan oshganligini tekshirish
	isMatch := similarity >= similarityThreshold

	return isMatch, bestMatch, similarity
}

// AddImage adds a new image to the database
func (db *ImageDatabase) AddImage(img image.Image, filename string) (string, error) {
	hash := computeDCTHash(img)
	thumbnail := generateThumbnail(img)

	db.mutex.Lock()
	defer db.mutex.Unlock()

	// Check if this hash already exists
	for _, info := range db.hashes {
		if info.Hash == hash {
			return "", fmt.Errorf("image already exists in database as: %s", info.Filename)
		}
	}

	db.hashes[hash] = imageInfo{
		Filename:  filename,
		Hash:      hash,
		AddedAt:   time.Now(),
		Thumbnail: thumbnail,
	}

	return hash, nil
}

// ListImages returns a list of all images in the database
func (db *ImageDatabase) ListImages() []imageInfo {
	db.mutex.RLock()
	defer db.mutex.RUnlock()

	images := make([]imageInfo, 0, len(db.hashes))
	for _, info := range db.hashes {
		// Create a copy without the thumbnail to reduce payload size
		images = append(images, imageInfo{
			Filename: info.Filename,
			Hash:     info.Hash,
			AddedAt:  info.AddedAt,
		})
	}

	return images
}

// @Summary Recognize image
// @Description Compare uploaded image against database
// @Tags Image Recognition
// @Accept multipart/form-data
// @Produce json
// @Param image formData file true "Image to check"
// @Param threshold formData float64 false "Similarity threshold (0-100, default 80)"
// @Success 200 {object} RecognizeResponse
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /recognize [post]
func recognizeHandler(c *gin.Context, db *ImageDatabase) {
	startTime := time.Now()

	// Parse request
	file, header, err := c.Request.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Rasm fayli topilmadi"})
		return
	}
	defer file.Close()

	// Validate file size
	if header.Size > 10<<20 { // 10MB limit
		c.JSON(http.StatusBadRequest, gin.H{"error": "Fayl hajmi 10MB dan oshib ketdi"})
		return
	}

	// Get threshold parameter (optional)
	var similarityThreshold float64 = 85.0 // Juda qattiq default qiymat (85%)
	thresholdStr := c.DefaultPostForm("threshold", "")
	if thresholdStr != "" {
		parsedThreshold, err := strconv.ParseFloat(thresholdStr, 64)
		if err == nil && parsedThreshold >= 0 && parsedThreshold <= 100 {
			similarityThreshold = parsedThreshold
		}
	}

	// Filedan rasmni o'qish
	fileBytes, err := io.ReadAll(file)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Faylni o'qib bo'lmadi"})
		return
	}

	// Decode image
	img, err := imaging.Decode(bytes.NewReader(fileBytes))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Noto'g'ri rasm formati"})
		return
	}

	// Find match - bu yerda similarityThreshold parametrini yuboramiz
	isMatch, matchedImage, similarity := db.FindMatch(img, similarityThreshold)

	// Create response
	response := RecognizeResponse{
		ProcessingTimeMs: time.Since(startTime).Milliseconds(),
		Similarity:       similarity,
	}

	if isMatch {
		response.Result = "OK"
		response.MatchedImage = matchedImage
	} else {
		response.Result = "NOT OK"
		// Agar o'xshashlik topilmasa, matchedImage ni bo'sh qoldirish kerak
		response.MatchedImage = ""
	}

	// Response bo'yicha log yozish
	log.Printf("Response: %+v", response)

	c.JSON(http.StatusOK, response)
}

// @Summary Add new image to database
// @Description Add a new reference image to the database
// @Tags Image Database Management
// @Accept multipart/form-data
// @Produce json
// @Param image formData file true "Image to add"
// @Param name formData string false "Custom name for the image"
// @Success 200 {object} map[string]string
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /admin/add [post]
func addImageHandler(c *gin.Context, db *ImageDatabase, imageDir string) {
	// Parse request
	file, header, err := c.Request.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "No image file found"})
		return
	}
	defer file.Close()

	// Validate file size
	if header.Size > 10<<20 { // 10MB limit
		c.JSON(http.StatusBadRequest, gin.H{"error": "File size exceeds 10MB limit"})
		return
	}

	// Validate file type
	ext := strings.ToLower(filepath.Ext(header.Filename))
	if !isImageFile(ext) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Unsupported file format. Please upload a valid image."})
		return
	}

	// Custom name if provided
	filename := header.Filename
	customName := c.PostForm("name")
	if customName != "" {
		// Ensure we keep the original extension
		filename = customName + ext
	}

	// Generate a unique filename to avoid overwriting
	uniqueFilename := fmt.Sprintf("%d_%s", time.Now().UnixNano(), filename)
	savePath := filepath.Join(imageDir, uniqueFilename)

	// Read file
	fileBytes, err := io.ReadAll(file)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not read file"})
		return
	}

	// Decode image
	img, err := imaging.Decode(bytes.NewReader(fileBytes))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid image format"})
		return
	}

	// Save the image to disk
	err = imaging.Save(img, savePath)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not save image"})
		return
	}

	// Add to database
	hash, err := db.AddImage(img, uniqueFilename)
	if err != nil {
		// If there was an error, try to delete the saved file
		os.Remove(savePath)
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message":  "Image added successfully",
		"filename": uniqueFilename,
		"hash":     hash,
	})
}

// @Summary List all images in database
// @Description Get a list of all reference images
// @Tags Image Database Management
// @Produce json
// @Success 200 {array} imageInfo
// @Router /admin/images [get]
func listImagesHandler(c *gin.Context, db *ImageDatabase) {
	images := db.ListImages()
	c.JSON(http.StatusOK, images)
}

// @Summary Get database stats
// @Description Get statistics about the image database
// @Tags Image Database Management
// @Produce json
// @Success 200 {object} map[string]interface{}
// @Router /admin/stats [get]
func statsHandler(c *gin.Context, db *ImageDatabase) {
	db.mutex.RLock()
	defer db.mutex.RUnlock()

	c.JSON(http.StatusOK, gin.H{
		"total_images": len(db.hashes),
		"cache_items":  db.cache.ItemCount(),
	})
}

// @Summary Health check
// @Description API health check endpoint
// @Tags System
// @Produce json
// @Success 200 {object} map[string]string
// @Router /health [get]
func healthHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status": "running",
		"time":   time.Now().Format(time.RFC3339),
	})
}

func main() {
	// Debug loglarni yoqish
	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)
	log.Println("Application starting...")

	// Image directory configuration
	imageDir := "./images"
	if _, err := os.Stat(imageDir); os.IsNotExist(err) {
		// Create directory if it doesn't exist
		err = os.MkdirAll(imageDir, 0755)
		if err != nil {
			log.Fatalf("Could not create images directory: %v", err)
		}
		log.Printf("Created images directory: %s", imageDir)
	}

	// Initialize database
	db := NewImageDatabase()
	if err := db.LoadImages(imageDir); err != nil {
		log.Fatalf("Failed to load images: %v", err)
	}

	log.Printf("Loaded %d images into database", len(db.hashes))

	// Set up Gin
	gin.SetMode(gin.DebugMode) // Debug rejimini yoqish
	r := gin.Default()

	// CORS ni yoqish
	r.Use(cors.Default())

	// Main API endpoints
	r.POST("/recognize", func(c *gin.Context) {
		recognizeHandler(c, db)
	})

	// Admin API endpoints
	admin := r.Group("/admin")
	{
		admin.POST("/add", func(c *gin.Context) {
			addImageHandler(c, db, imageDir)
		})
		admin.GET("/images", func(c *gin.Context) {
			listImagesHandler(c, db)
		})
	}

	// Swagger UI
	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	// Server portini yoqish
	port := ":8080"
	log.Printf("Server started on port %s...", port)
	if err := r.Run(port); err != nil {
		log.Fatalf("Could not start server: %v", err)
	}
}
