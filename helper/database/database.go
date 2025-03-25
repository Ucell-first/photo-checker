package database

import (
	"fmt"
	"image"
	"log"
	"os"
	"path/filepath"
	im "photot/helper/image"
	"strings"
	"sync"
	"time"

	"github.com/disintegration/imaging"
	"github.com/patrickmn/go-cache"
)

// ImageDatabase stores image hashes and features for recognition
type ImageDatabase struct {
	Hashes map[string]imageInfo
	Mutex  sync.RWMutex
	Cache  *cache.Cache
	UseML  bool // Switch between ML or hash-based comparison
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
		Hashes: make(map[string]imageInfo),
		Cache:  cache.New(5*time.Minute, 10*time.Minute),
		UseML:  true,
	}
	return db
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

			hash := im.ComputeDCTHash(img)
			thumbnail := im.GenerateThumbnail(img)

			info := imageInfo{
				Filename:  fileName,
				Hash:      hash,
				AddedAt:   time.Now(),
				Thumbnail: thumbnail,
			}

			// Extract ML features
			features := im.ExtractImageFeatures(img)
			info.Features = features

			db.Mutex.Lock()
			db.Hashes[hash] = info
			db.Mutex.Unlock()

			log.Printf("Loaded image: %s", fileName)
		}(file.Name())
	}

	wg.Wait()
	log.Printf("Loaded %d images into database", len(db.Hashes))
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
	if db.UseML {
		method = "ml"
		features := im.ExtractImageFeatures(img)
		isMatch, matchedImage, similarity := db.findMatchByFeatures(features, similarityThreshold)

		if isMatch {
			return isMatch, matchedImage, similarity, method
		}
	}

	// Fallback to hash-based matching
	uploadedHash := im.ComputeDCTHash(img)

	db.Mutex.RLock()
	defer db.Mutex.RUnlock()

	if len(db.Hashes) == 0 {
		return false, "", 0.0, method
	}

	bestMatch := ""
	minDistance := len(uploadedHash)

	for hash, info := range db.Hashes {
		distance, err := im.HammingDistance(uploadedHash, hash)
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
	db.Mutex.RLock()
	defer db.Mutex.RUnlock()

	bestMatch := ""
	maxSimilarity := 0.0

	for _, info := range db.Hashes {
		if info.Features == nil {
			continue
		}

		similarity := im.CosineSimilarity(features, info.Features)
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
	hash := im.ComputeDCTHash(img)
	thumbnail := im.GenerateThumbnail(img)

	info := imageInfo{
		Filename:  filename,
		Hash:      hash,
		AddedAt:   time.Now(),
		Thumbnail: thumbnail,
		Features:  im.ExtractImageFeatures(img),
	}

	db.Mutex.Lock()
	defer db.Mutex.Unlock()

	for _, existingInfo := range db.Hashes {
		if existingInfo.Hash == hash {
			return "", fmt.Errorf("image already exists: %s", existingInfo.Filename)
		}
	}

	db.Hashes[hash] = info
	return hash, nil
}
