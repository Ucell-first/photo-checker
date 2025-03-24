package handler

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/disintegration/imaging"
	"github.com/gin-gonic/gin"
	"github.com/patrickmn/go-cache"
)

type Handler struct {
	DB       *ImageDatabase
	ImageDir string
}

// ImageDatabase stores image hashes and features for recognition
type ImageDatabase struct {
	hashes map[string]imageInfo
	mutex  sync.RWMutex
	cache  *cache.Cache
	useML  bool // Switch between ML or hash-based comparison
}

// imageInfo contains metadata for stored images
type imageInfo struct {
	Filename  string    `json:"filename"`
	Hash      string    `json:"hash"`
	Features  []float64 `json:"features,omitempty"` // ML feature vector
	AddedAt   time.Time `json:"added_at"`
	Thumbnail string    `json:"thumbnail,omitempty"`
}

// RecognizeResponse structure for API responses
type RecognizeResponse struct {
	Result           string  `json:"result"`
	Similarity       float64 `json:"similarity"`
	MatchedImage     string  `json:"matched_image,omitempty"`
	ProcessingTimeMs int64   `json:"processing_time_ms"`
	Method           string  `json:"method"` // "ml" or "hash"
}

// NewImageDatabase creates a new image database instance
func NewImageDatabase() *ImageDatabase {
	db := &ImageDatabase{
		hashes: make(map[string]imageInfo),
		cache:  cache.New(5*time.Minute, 10*time.Minute),
		useML:  true,
	}
	return db
}

// computeDCTHash calculates perceptual hash using Discrete Cosine Transform
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

	return hash.String()
}

// extractImageFeatures extracts HOG (Histogram of Oriented Gradients) features
func extractImageFeatures(img image.Image) []float64 {
	// Resize image to 64x64
	resized := imaging.Resize(img, 64, 64, imaging.Lanczos)
	gray := imaging.Grayscale(resized)

	// Simple HOG implementation
	features := make([]float64, 0, 144) // 3x3 blocks with 16 orientations

	// Calculate HOG features
	for by := 0; by < 3; by++ {
		for bx := 0; bx < 3; bx++ {
			histogram := make([]float64, 16) // 16 orientation bins

			for y := by*20 + 2; y < (by+1)*20-2 && y < 64; y++ {
				for x := bx*20 + 2; x < (bx+1)*20-2 && x < 64; x++ {
					// Calculate gradients
					gx := int(color.GrayModel.Convert(gray.At(x+1, y)).(color.Gray).Y) -
						int(color.GrayModel.Convert(gray.At(x-1, y)).(color.Gray).Y)
					gy := int(color.GrayModel.Convert(gray.At(x, y+1)).(color.Gray).Y) -
						int(color.GrayModel.Convert(gray.At(x, y-1)).(color.Gray).Y)

					// Calculate magnitude and angle
					magnitude := math.Sqrt(float64(gx*gx + gy*gy))
					angle := math.Atan2(float64(gy), float64(gx))

					// Convert angle to bin index
					binIndex := int((angle + math.Pi) * 8 / math.Pi)
					if binIndex == 16 {
						binIndex = 0
					}

					histogram[binIndex] += magnitude
				}
			}

			// Normalize histogram
			var sum float64
			for _, val := range histogram {
				sum += val * val
			}
			norm := math.Sqrt(sum + 1e-6)

			for i, val := range histogram {
				if norm > 0 {
					histogram[i] = val / norm
				}
			}

			features = append(features, histogram...)
		}
	}

	return features
}

// cosineSimilarity calculates similarity between two feature vectors
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA <= 0 || normB <= 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB)) * 100.0
}

// generateThumbnail creates base64 encoded thumbnail
func generateThumbnail(img image.Image) string {
	thumbnail := imaging.Resize(img, 100, 0, imaging.Lanczos)
	var buf bytes.Buffer
	err := imaging.Encode(&buf, thumbnail, imaging.JPEG)
	if err != nil {
		return ""
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

// hammingDistance calculates difference between two hashes
func hammingDistance(hash1, hash2 string) (int, error) {
	if len(hash1) != len(hash2) {
		return 0, fmt.Errorf("hash length mismatch: %d vs %d", len(hash1), len(hash2))
	}
	distance := 0
	for i := 0; i < len(hash1); i++ {
		if hash1[i] != hash2[i] {
			distance++
		}
	}
	return distance, nil
}

// LoadImages loads images from directory and extracts features
func (db *ImageDatabase) LoadImages(imageDir string) error {
	if _, err := os.Stat(imageDir); os.IsNotExist(err) {
		return fmt.Errorf("image directory not found: %s", imageDir)
	}

	files, err := os.ReadDir(imageDir)
	if err != nil {
		return fmt.Errorf("failed to read directory: %s", err)
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
				log.Printf("Failed to open file %s: %v", path, err)
				return
			}

			hash := computeDCTHash(img)
			thumbnail := generateThumbnail(img)

			info := imageInfo{
				Filename:  fileName,
				Hash:      hash,
				AddedAt:   time.Now(),
				Thumbnail: thumbnail,
			}

			// Extract ML features
			features := extractImageFeatures(img)
			info.Features = features

			db.mutex.Lock()
			db.hashes[hash] = info
			db.mutex.Unlock()

			log.Printf("Loaded image: %s", fileName)
		}(file.Name())
	}

	wg.Wait()
	log.Printf("Loaded %d images into database", len(db.hashes))
	return nil
}

// isImageFile checks if extension is supported
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

// FindMatch searches for similar images using combined ML and hash methods
func (db *ImageDatabase) FindMatch(img image.Image, similarityThreshold float64) (bool, string, float64, string) {
	method := "hash"

	// First try ML-based matching
	if db.useML {
		method = "ml"
		features := extractImageFeatures(img)
		isMatch, matchedImage, similarity := db.findMatchByFeatures(features, similarityThreshold)

		if isMatch {
			return isMatch, matchedImage, similarity, method
		}
	}

	// Fallback to hash-based matching
	uploadedHash := computeDCTHash(img)

	db.mutex.RLock()
	defer db.mutex.RUnlock()

	if len(db.hashes) == 0 {
		return false, "", 0.0, method
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
	similarity := 100.0 - (float64(minDistance)/float64(maxDistance))*100.0

	isMatch := similarity >= similarityThreshold
	method = "hash"

	return isMatch, bestMatch, similarity, method
}

// findMatchByFeatures performs ML-based similarity search
func (db *ImageDatabase) findMatchByFeatures(features []float64, similarityThreshold float64) (bool, string, float64) {
	db.mutex.RLock()
	defer db.mutex.RUnlock()

	bestMatch := ""
	maxSimilarity := 0.0

	for _, info := range db.hashes {
		if info.Features == nil {
			continue
		}

		similarity := cosineSimilarity(features, info.Features)
		if similarity > maxSimilarity {
			maxSimilarity = similarity
			bestMatch = info.Filename
		}
	}

	isMatch := maxSimilarity >= similarityThreshold
	return isMatch, bestMatch, maxSimilarity
}

// AddImage adds new image to the database
func (db *ImageDatabase) AddImage(img image.Image, filename string) (string, error) {
	hash := computeDCTHash(img)
	thumbnail := generateThumbnail(img)

	info := imageInfo{
		Filename:  filename,
		Hash:      hash,
		AddedAt:   time.Now(),
		Thumbnail: thumbnail,
		Features:  extractImageFeatures(img),
	}

	db.mutex.Lock()
	defer db.mutex.Unlock()

	for _, existingInfo := range db.hashes {
		if existingInfo.Hash == hash {
			return "", fmt.Errorf("image already exists: %s", existingInfo.Filename)
		}
	}

	db.hashes[hash] = info
	return hash, nil
}

// @Summary Recognize image
// @Description Compare uploaded image against database using ML or hashing
// @Tags Image Recognition
// @Accept multipart/form-data
// @Produce json
// @Param image formData file true "Image file to check"
// @Param threshold formData number false "Similarity threshold (0-100)"
// @Success 200 {object} RecognizeResponse
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /recognize [post]
func (h *Handler) RecognizeHandler(c *gin.Context) {
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

	isMatch, matchedImage, similarity, method := h.DB.FindMatch(img, similarityThreshold)

	response := RecognizeResponse{
		ProcessingTimeMs: time.Since(startTime).Milliseconds(),
		Similarity:       similarity,
		Method:           method,
	}

	if isMatch {
		response.Result = "OK"
		response.MatchedImage = matchedImage
	} else {
		response.Result = "NOT OK"
		response.MatchedImage = matchedImage
	}

	c.JSON(http.StatusOK, response)
}

// @Summary Add new image
// @Description Add reference image to database
// @Tags Image Database Management
// @Accept multipart/form-data
// @Produce json
// @Param image formData file true "Image file to upload"
// @Param name formData string false "Custom image name"
// @Success 200 {object} map[string]string
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /admin/add [post]
func (h *Handler) AddImageHandler(c *gin.Context) {
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
	ext := strings.ToLower(filepath.Ext(header.Filename))
	if !isImageFile(ext) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Qo'llab-quvvatlanmaydigan fayl formati. Iltimos, to'g'ri rasm yuklang."})
		return
	}
	filename := header.Filename
	customName := c.PostForm("name")
	if customName != "" {
		filename = customName + ext
	}
	uniqueFilename := fmt.Sprintf("%d_%s", time.Now().UnixNano(), filename)
	savePath := filepath.Join(h.ImageDir, uniqueFilename)
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
	err = imaging.Save(img, savePath)
	if err != nil {
		log.Printf("Error saving image to %s: %v", savePath, err)
		if os.IsPermission(err) {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Rasm saqlashda ruxsat rad etildi"})
		} else {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Rasmni saqlashda xatolik"})
		}
		return
	}

	hash, err := h.DB.AddImage(img, uniqueFilename)
	if err != nil {
		os.Remove(savePath)
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message":  "Rasm muvaffaqiyatli qo'shildi",
		"filename": uniqueFilename,
		"hash":     hash,
	})
}

// @Summary Toggle ML mode
// @Description Enable/disable ML-based recognition
// @Tags Image Database Management
// @Accept multipart/form-data
// @Produce json
// @Param enable formData string false "Set to 'true' or 'false'"
// @Success 200 {object} map[string]interface{}
// @Router /admin/toggle-ml [post]
func (h *Handler) ToggleMLHandler(c *gin.Context) {
	enable := c.DefaultPostForm("enable", "")
	if enable == "true" {
		h.DB.useML = true
		c.JSON(http.StatusOK, gin.H{"message": "ML enabled", "status": "enabled"})
	} else if enable == "false" {
		h.DB.useML = false
		c.JSON(http.StatusOK, gin.H{"message": "ML disabled", "status": "disabled"})
	} else {
		c.JSON(http.StatusOK, gin.H{"message": "ML status", "status": h.DB.useML})
	}
}

// @Summary Hello endpoint
// @Description Test connection endpoint
// @Tags Image Database Management
// @Produce json
// @Success 200 {object} map[string]string
// @Router /admin/hello [get]
func (h *Handler) Hello(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"message": "Hello, world"})
}
