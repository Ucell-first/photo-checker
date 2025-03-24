// Package main is the entry point for the photot application
package main

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/disintegration/imaging"
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"

	_ "photot/docs" // Generated Swagger doc
	"photot/internal/database"
	"photot/internal/imageprocessing"
)

// RecognizeResponse represents the response for an image recognition request
// @Summary Recognize image
// @Description Compare uploaded image against database
// @Tags Image Recognition
// @Accept multipart/form-data
// @Produce json
// @Param image formData file true "Image to check"
// @Param threshold formData float64 false "Similarity threshold (0-100, default 85)"
// @Success 200 {object} RecognizeResponse
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /recognize [post]
type RecognizeResponse struct {
	Result           string  `json:"result"`
	Similarity       float64 `json:"similarity"`
	MatchedImage     string  `json:"matched_image,omitempty"`
	ProcessingTimeMs int64   `json:"processing_time_ms"`
}

// recognizeHandler handles image recognition requests
func recognizeHandler(c *gin.Context, db *database.ImageDatabase) {
	startTime := time.Now()

	// Read uploaded file
	file, header, err := c.Request.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Rasm fayli topilmadi"})
		return
	}
	defer file.Close()

	// Check file size
	if header.Size > 10<<20 { // 10MB limit
		c.JSON(http.StatusBadRequest, gin.H{"error": "Fayl hajmi 10MB dan oshib ketdi"})
		return
	}

	// Parse similarity threshold
	var similarityThreshold float64 = 85.0
	thresholdStr := c.DefaultPostForm("threshold", "")
	if thresholdStr != "" {
		parsedThreshold, err := strconv.ParseFloat(thresholdStr, 64)
		if err == nil && parsedThreshold >= 0 && parsedThreshold <= 100 {
			similarityThreshold = parsedThreshold
		}
	}

	// Read and decode the image
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

	// Find match
	isMatch, matchedImage, similarity := db.FindMatch(img, similarityThreshold)

	// Prepare response
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

// addImageHandler handles requests to add new images to the database
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
func addImageHandler(c *gin.Context, db *database.ImageDatabase, imageDir string) {
	// Read uploaded file
	file, header, err := c.Request.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "No image file found"})
		return
	}
	defer file.Close()

	// Check file size
	if header.Size > 10<<20 { // 10MB limit
		c.JSON(http.StatusBadRequest, gin.H{"error": "File size exceeds 10MB limit"})
		return
	}

	// Validate file type
	ext := strings.ToLower(filepath.Ext(header.Filename))
	if !imageprocessing.IsImageFile(ext) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Unsupported file format. Please upload a valid image."})
		return
	}

	// Handle custom name
	filename := header.Filename
	customName := c.PostForm("name")
	if customName != "" {
		filename = customName + ext
	}

	// Create unique filename
	uniqueFilename := fmt.Sprintf("%d_%s", time.Now().UnixNano(), filename)
	savePath := filepath.Join(imageDir, uniqueFilename)

	// Read and decode the image
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

	// Save the image
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

	// Add to database
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

// Hello is a simple endpoint to check if the server is running
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

type CompareResponse struct {
	Similarity   float64 `json:"similarity"`
	Match        bool    `json:"match"`
	ProcessingMs int64   `json:"processing_time_ms"`
}

// CompareRequest represents the request for image comparison
// @Summary Compare two images
// @Description Compare two uploaded images and return similarity
// @Tags Image Comparison
// @Accept multipart/form-data
// @Produce json
// @Param image1 formData file true "First image to compare"
// @Param image2 formData file true "Second image to compare"
// @Param threshold formData float64 false "Similarity threshold (0-100, default 85)"
// @Success 200 {object} CompareResponse
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /compare [post]
func compareHandler(c *gin.Context, mlRecognizer *imageprocessing.MLImageRecognizer) {
	startTime := time.Now()

	// Read first image
	file1, header1, err := c.Request.FormFile("image1")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Birinchi rasm fayli topilmadi"})
		return
	}
	defer file1.Close()

	// Read second image
	file2, header2, err := c.Request.FormFile("image2")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Ikkinchi rasm fayli topilmadi"})
		return
	}
	defer file2.Close()

	// Validate file sizes
	if header1.Size > 10<<20 || header2.Size > 10<<20 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Fayl hajmi 10MB dan oshmasligi kerak"})
		return
	}

	// Parse threshold
	threshold := 85.0
	if thresholdStr := c.PostForm("threshold"); thresholdStr != "" {
		if parsed, err := strconv.ParseFloat(thresholdStr, 64); err == nil {
			threshold = parsed
		}
	}

	// Decode images
	img1, err := imaging.Decode(file1)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Birinchi rasm formati noto'g'ri"})
		return
	}

	img2, err := imaging.Decode(file2)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Ikkinchi rasm formati noto'g'ri"})
		return
	}

	// Compare images
	similarity, err := imageprocessing.RecognizeImage(img1, img2, mlRecognizer)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Rasmlarni solishtirishda xato"})
		return
	}

	c.JSON(http.StatusOK, CompareResponse{
		Similarity:   similarity,
		Match:        similarity >= threshold,
		ProcessingMs: time.Since(startTime).Milliseconds(),
	})
}

func main() {
	// Configure logging
	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)
	log.Println("Application starting...")

	// Set up images directory
	imageDir := "./images"
	if _, err := os.Stat(imageDir); os.IsNotExist(err) {
		err = os.MkdirAll(imageDir, 0755)
		if err != nil {
			log.Fatalf("Could not create images directory: %v", err)
		}
		log.Printf("Created images directory: %s", imageDir)
	}

	// Set up ML recognizer
	mlRecognizer := imageprocessing.NewMLImageRecognizer("./ml_model")
	if err := mlRecognizer.Load(); err != nil {
		log.Printf("Warning: Could not load ML model: %v", err)
		log.Printf("Continuing with DCT hash-based recognition only")
	} else {
		log.Printf("ML image recognition model loaded successfully")
		defer mlRecognizer.Close()
	}

	// Initialize database
	db := database.NewImageDatabase()
	db.SetMLRecognizer(mlRecognizer)

	// Load existing images
	if err := db.LoadImages(imageDir); err != nil {
		log.Fatalf("Failed to load images: %v", err)
	}
	log.Printf("Loaded %d images into database", len(db.ListImages()))

	// Configure Gin router
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()

	// CORS configuration
	r.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"*"},
		AllowMethods:     []string{"GET", "POST"},
		AllowHeaders:     []string{"Origin", "Content-Type"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

	// API endpoints
	r.POST("/recognize", func(c *gin.Context) {
		recognizeHandler(c, db)
	})

	r.POST("/compare", func(c *gin.Context) {
		compareHandler(c, mlRecognizer)
	})

	admin := r.Group("/admin")
	{
		admin.POST("/add", func(c *gin.Context) {
			addImageHandler(c, db, imageDir)
		})
		admin.GET("/hello", Hello)
	}

	// Swagger documentation
	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	// Start server
	port := ":8080"
	log.Printf("Server started on port %s...", port)
	if err := r.Run(port); err != nil {
		log.Fatalf("Could not start server: %v", err)
	}
}
