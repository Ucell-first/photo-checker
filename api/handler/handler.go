package handler

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"photot/helper/database"
	"strconv"
	"strings"
	"time"

	"github.com/disintegration/imaging"
	"github.com/gin-gonic/gin"
)

type Handler struct {
	DB       *database.ImageDatabase
	ImageDir string
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

// @Summary Recognize image
// @Description Compare uploaded image against database using ML or hashing
// @Tags Image Recognition
// @Accept multipart/form-data
// @Produce json
// @Param image formData file true "Image file to check"
// @Param threshold formData number false "Similarity threshold (0-100)"
// @Success 200 {object} database.RecognizeResponse
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

	response := database.RecognizeResponse{
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
		"message":  "immage added succesfully",
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
		h.DB.UseML = true
		c.JSON(http.StatusOK, gin.H{"message": "ML enabled", "status": "enabled"})
	} else if enable == "false" {
		h.DB.UseML = false
		c.JSON(http.StatusOK, gin.H{"message": "ML disabled", "status": "disabled"})
	} else {
		c.JSON(http.StatusOK, gin.H{"message": "ML status", "status": h.DB.UseML})
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
