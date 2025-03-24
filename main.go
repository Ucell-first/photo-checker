package main

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
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/patrickmn/go-cache"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
	tf "github.com/wamuir/graft/tensorflow"
)

// Update imageInfo to store an embedding vector (as a []float32)
type imageInfo struct {
	Filename  string    `json:"filename"`
	Hash      string    `json:"hash"` // still keeping the hash for legacy or quick checks
	AddedAt   time.Time `json:"added_at"`
	Thumbnail string    `json:"thumbnail,omitempty"`
	Embedding []float32 `json:"-"` // not sent in JSON responses
}

type ImageDatabase struct {
	hashes map[string]imageInfo
	mutex  sync.RWMutex
	cache  *cache.Cache
	// Hold the TensorFlow model once loaded
	tfModel *tf.SavedModel
}

type RecognizeResponse struct {
	Result           string  `json:"result"`
	Similarity       float64 `json:"similarity"`
	MatchedImage     string  `json:"matched_image,omitempty"`
	ProcessingTimeMs int64   `json:"processing_time_ms"`
}

func NewImageDatabase(modelDir string) *ImageDatabase {
	// Load the TensorFlow SavedModel.
	// Adjust the tags and options as needed.
	model, err := tf.LoadSavedModel(modelDir, []string{"serve"}, nil)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	return &ImageDatabase{
		hashes:  make(map[string]imageInfo),
		cache:   cache.New(5*time.Minute, 10*time.Minute),
		tfModel: model,
	}
}

// computeDCTHash remains unchanged (if you want to keep it)
func computeDCTHash(img image.Image) string {
	// ... your original DCT hash code ...
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
	// Additional gradient bits (unchanged)
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

// computeMLEmbedding extracts a feature vector using the ML model.
// Adjust input preprocessing and output extraction based on your model.
func (db *ImageDatabase) computeMLEmbedding(img image.Image) ([]float32, error) {
	// Resize and normalize the image as required by the model.
	// Here we assume the model requires 224x224 images (like many CNNs).
	resized := imaging.Resize(img, 224, 224, imaging.Lanczos)
	// Convert the image to a [][][]float32 tensor (batch size 1).
	// This is a simplified example. Real preprocessing may require mean subtraction, scaling, etc.
	bounds := resized.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	// Create a slice to hold the pixel data in [height][width][channels] format.
	data := make([][][]float32, height)
	for y := 0; y < height; y++ {
		row := make([][]float32, width)
		for x := 0; x < width; x++ {
			// Get RGB values (we assume the model expects three channels)
			r, g, b, _ := resized.At(x, y).RGBA()
			// Convert from uint32 (0-65535) to float32 (0-1)
			row[x] = []float32{float32(r) / 65535.0, float32(g) / 65535.0, float32(b) / 65535.0}
		}
		data[y] = row
	}
	// Create a tensor from the data. The shape will be [1, height, width, 3]
	tensor, err := tf.NewTensor([][][][]float32{data})
	if err != nil {
		return nil, fmt.Errorf("failed to create tensor: %v", err)
	}

	// Run the model (adjust input and output node names accordingly)
	// For example, if your model input node is "input" and output node is "embedding":
	result, err := db.tfModel.Session.Run(
		map[tf.Output]*tf.Tensor{
			db.tfModel.Graph.Operation("input").Output(0): tensor,
		},
		[]tf.Output{
			db.tfModel.Graph.Operation("embedding").Output(0),
		},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to run session: %v", err)
	}

	// The result is expected to be a 2D slice with shape [1, embedding_size]
	embeddingRaw, ok := result[0].Value().([][]float32)
	if !ok || len(embeddingRaw) == 0 {
		return nil, fmt.Errorf("unexpected tensor output type")
	}
	return embeddingRaw[0], nil
}

// cosineSimilarity calculates the cosine similarity between two vectors.
func cosineSimilarity(a, b []float32) (float64, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("vector length mismatch")
	}
	var dot, normA, normB float64
	for i := 0; i < len(a); i++ {
		dot += float64(a[i] * b[i])
		normA += float64(a[i] * a[i])
		normB += float64(b[i] * b[i])
	}
	if normA == 0 || normB == 0 {
		return 0, fmt.Errorf("zero vector")
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB)) * 100.0, nil // percentage similarity
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

			// Compute the legacy hash if you still want to use it.
			hash := computeDCTHash(img)
			thumbnail := generateThumbnail(img)

			// Compute the ML embedding.
			embedding, err := db.computeMLEmbedding(img)
			if err != nil {
				log.Printf("Could not compute ML embedding for %s: %v", path, err)
				return
			}

			db.mutex.Lock()
			db.hashes[hash] = imageInfo{
				Filename:  fileName,
				Hash:      hash,
				AddedAt:   time.Now(),
				Thumbnail: thumbnail,
				Embedding: embedding,
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

// FindMatch now uses the ML embedding for similarity measurement.
func (db *ImageDatabase) FindMatch(img image.Image, similarityThreshold float64) (bool, string, float64) {
	// Compute both the legacy hash and the ML embedding for the uploaded image.
	uploadedHash := computeDCTHash(img)
	uploadedEmbedding, err := db.computeMLEmbedding(img)
	if err != nil {
		log.Printf("Error computing embedding: %v", err)
		return false, "", 0.0
	}

	db.mutex.RLock()
	defer db.mutex.RUnlock()

	if len(db.hashes) == 0 {
		return false, "", 0.0
	}

	bestMatch := ""
	bestSimilarity := 0.0

	for _, info := range db.hashes {
		// First, you might filter using the hash difference as a fast pre-check.
		distance, err := hammingDistance(uploadedHash, info.Hash)
		if err != nil {
			continue
		}
		if distance > 10 { // for example, skip if the perceptual hash is too different
			continue
		}

		// Then compute the cosine similarity on ML embeddings.
		sim, err := cosineSimilarity(uploadedEmbedding, info.Embedding)
		if err != nil {
			continue
		}
		if sim > bestSimilarity {
			bestSimilarity = sim
			bestMatch = info.Filename
		}
	}

	log.Printf("Best ML similarity: %.2f%%, threshold: %.2f%%", bestSimilarity, similarityThreshold)

	isMatch := bestSimilarity >= similarityThreshold

	return isMatch, bestMatch, bestSimilarity
}

// hammingDistance remains unchanged.
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

func (db *ImageDatabase) AddImage(img image.Image, filename string) (string, error) {
	hash := computeDCTHash(img)
	thumbnail := generateThumbnail(img)
	embedding, err := db.computeMLEmbedding(img)
	if err != nil {
		return "", fmt.Errorf("failed to compute ML embedding: %v", err)
	}

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
		Embedding: embedding,
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

// recognizeHandler and addImageHandler remain largely unchanged except that they now use the updated FindMatch and AddImage.
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

func Hello(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"message": "Hello, world",
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

	// Pass the path to your SavedModel directory
	modelDir := "./saved_model"
	db := NewImageDatabase(modelDir)
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
