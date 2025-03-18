// @securityDefinitions.apikey ApiKeyAuth
// @in header
// @name Authorization
// @title User
// @version 1.0
// @description API Gateway
// BasePath: /
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

type ImageDatabase struct {
	hashes map[string]imageInfo
	mutex  sync.RWMutex
	cache  *cache.Cache
}

type imageInfo struct {
	Filename  string    `json:"filename"`
	Hash      string    `json:"hash"`
	AddedAt   time.Time `json:"added_at"`
	Thumbnail string    `json:"thumbnail,omitempty"`
}

type RecognizeResponse struct {
	Result           string  `json:"result"`
	Similarity       float64 `json:"similarity"`
	MatchedImage     string  `json:"matched_image,omitempty"`
	ProcessingTimeMs int64   `json:"processing_time_ms"`
}

func NewImageDatabase() *ImageDatabase {
	return &ImageDatabase{
		hashes: make(map[string]imageInfo),
		cache:  cache.New(5*time.Minute, 10*time.Minute),
	}
}

func computeDCTHash(img image.Image) string {
	resized := imaging.Resize(img, 32, 32, imaging.Lanczos)
	gray := imaging.Grayscale(resized)
	const blockSize = 8
	const numBlocks = 16
	blockValues := make([]float64, numBlocks)

	blockIndex := 0
	for by := 0; by < 4; by++ {
		for bx := 0; bx < 4; bx++ {
			var sum float64
			var count int
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
	var sum float64
	for _, val := range blockValues {
		sum += val
	}
	avg := sum / float64(len(blockValues))
	var hash strings.Builder
	for _, val := range blockValues {
		if val >= avg {
			hash.WriteString("1")
		} else {
			hash.WriteString("0")
		}
	}
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

func generateThumbnail(img image.Image) string {
	thumbnail := imaging.Resize(img, 100, 0, imaging.Lanczos)
	var buf bytes.Buffer
	err := imaging.Encode(&buf, thumbnail, imaging.JPEG)
	if err != nil {
		return ""
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

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

func (db *ImageDatabase) LoadImages(imageDir string) error {
	if _, err := os.Stat(imageDir); os.IsNotExist(err) {
		return fmt.Errorf("images directory does not exist: %s", imageDir)
	}

	files, err := os.ReadDir(imageDir)
	if err != nil {
		return fmt.Errorf("could not read directory: %s", err)
	}

	var wg sync.WaitGroup
	threadLimit := make(chan struct{}, 4)

	for _, file := range files {
		if file.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(file.Name()))
		if !isImageFile(ext) {
			continue
		}

		wg.Add(1)
		threadLimit <- struct{}{}

		go func(fileName string) {
			defer wg.Done()
			defer func() { <-threadLimit }()

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

func (db *ImageDatabase) FindMatch(img image.Image, similarityThreshold float64) (bool, string, float64) {
	uploadedHash := computeDCTHash(img)

	db.mutex.RLock()
	defer db.mutex.RUnlock()

	if len(db.hashes) == 0 {
		return false, "", 0.0
	}

	bestMatch := ""
	minDistance := len(uploadedHash)

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

	maxDistance := len(uploadedHash)
	similarity := 100.0 - (float64(minDistance) / float64(maxDistance) * 100.0)

	log.Printf("Uploaded hash: %s", uploadedHash[:20]+"...")
	log.Printf("Best match: %s, similarity: %.2f%%, threshold: %.2f%%", bestMatch, similarity, similarityThreshold)

	isMatch := similarity >= similarityThreshold

	return isMatch, bestMatch, similarity
}

func (db *ImageDatabase) AddImage(img image.Image, filename string) (string, error) {
	hash := computeDCTHash(img)
	thumbnail := generateThumbnail(img)

	db.mutex.Lock()
	defer db.mutex.Unlock()

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

func (db *ImageDatabase) ListImages() []imageInfo {
	db.mutex.RLock()
	defer db.mutex.RUnlock()

	images := make([]imageInfo, 0, len(db.hashes))
	for _, info := range db.hashes {
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
	file, header, err := c.Request.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Rasm fayli topilmadi"})
		return
	}
	defer file.Close()

	if header.Size > 10<<20 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Fayl hajmi 10MB dan oshib ketdi"})
		return
	}

	var similarityThreshold float64 = 85.0
	thresholdStr := c.DefaultPostForm("threshold", "")
	if thresholdStr != "" {
		parsedThreshold, err := strconv.ParseFloat(thresholdStr, 64)
		if err == nil && parsedThreshold >= 0 && parsedThreshold <= 100 {
			similarityThreshold = parsedThreshold
		}
	}
	fileBytes, err := io.ReadAll(file)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Faylni o'qib bo'lmadi"})
		return
	}

	img, err := imaging.Decode(bytes.NewReader(fileBytes))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Noto'g'ri rasm formati"})
		return
	}

	isMatch, matchedImage, similarity := db.FindMatch(img, similarityThreshold)

	response := RecognizeResponse{
		ProcessingTimeMs: time.Since(startTime).Milliseconds(),
		Similarity:       similarity,
	}

	if isMatch {
		response.Result = "OK"
		response.MatchedImage = matchedImage
	} else {
		response.Result = "NOT OK"
		response.MatchedImage = ""
	}
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
	file, header, err := c.Request.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "No image file found"})
		return
	}
	defer file.Close()

	if header.Size > 10<<20 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "File size exceeds 10MB limit"})
		return
	}
	ext := strings.ToLower(filepath.Ext(header.Filename))
	if !isImageFile(ext) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Unsupported file format. Please upload a valid image."})
		return
	}
	filename := header.Filename
	customName := c.PostForm("name")
	if customName != "" {
		filename = customName + ext
	}
	uniqueFilename := fmt.Sprintf("%d_%s", time.Now().UnixNano(), filename)
	savePath := filepath.Join(imageDir, uniqueFilename)
	fileBytes, err := io.ReadAll(file)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not read file"})
		return
	}

	img, err := imaging.Decode(bytes.NewReader(fileBytes))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid image format"})
		return
	}
	err = imaging.Save(img, savePath)
	if err != nil {
		log.Printf("Error saving image to %s: %v", savePath, err)
		if os.IsPermission(err) {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Permission denied when saving image. Check container volume permissions."})
		} else {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not save image"})
		}
		return
	}

	hash, err := db.AddImage(img, uniqueFilename)
	if err != nil {
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

func main() {
	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)
	log.Println("Application starting...")

	imageDir := "./images"
	if _, err := os.Stat(imageDir); os.IsNotExist(err) {
		err = os.MkdirAll(imageDir, 0755)
		if err != nil {
			log.Fatalf("Could not create images directory: %v", err)
		}
		log.Printf("Created images directory: %s", imageDir)
	}

	db := NewImageDatabase()
	if err := db.LoadImages(imageDir); err != nil {
		log.Fatalf("Failed to load images: %v", err)
	}

	log.Printf("Loaded %d images into database", len(db.hashes))

	gin.SetMode(gin.DebugMode)
	r := gin.Default()

	r.Use(cors.Default())

	r.POST("/recognize", func(c *gin.Context) {
		recognizeHandler(c, db)
	})

	admin := r.Group("/admin")
	{
		admin.POST("/add", func(c *gin.Context) {
			addImageHandler(c, db, imageDir)
		})
		admin.GET("/hello", func(c *gin.Context) {
			Hello(c)
		})
	}

	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	port := ":8080"
	log.Printf("Server started on port %s...", port)
	if err := r.Run(port); err != nil {
		log.Fatalf("Could not start server: %v", err)
	}
}

// @Summary Hello
// @Description hello
// @Tags Image Database Management
// @Success 200 {object} string
// @Router /admin/hello [get]
func Hello(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"message": "Hello, world",
	})
}
