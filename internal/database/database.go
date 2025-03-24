// Package database provides image database management functionality
package database

import (
	"fmt"
	"image"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/disintegration/imaging"
	"github.com/patrickmn/go-cache"
	"github.com/pkg/errors"

	"photot/internal/imageprocessing"
)

// ImageInfo represents metadata about an image in the database
type ImageInfo struct {
	Filename  string    `json:"filename"`
	Hash      string    `json:"hash"`
	AddedAt   time.Time `json:"added_at"`
	Thumbnail string    `json:"thumbnail,omitempty"`
	Features  []float32 `json:"-"` // ML features are not exposed in JSON
}

// ImageDatabase manages a collection of images and their hashes
type ImageDatabase struct {
	hashes       map[string]ImageInfo
	mutex        sync.RWMutex
	cache        *cache.Cache
	mlRecognizer *imageprocessing.MLImageRecognizer
}

// NewImageDatabase creates a new image database
func NewImageDatabase() *ImageDatabase {
	return &ImageDatabase{
		hashes: make(map[string]ImageInfo),
		cache:  cache.New(5*time.Minute, 10*time.Minute),
	}
}

// SetMLRecognizer assigns a machine learning recognizer to the database
func (db *ImageDatabase) SetMLRecognizer(recognizer *imageprocessing.MLImageRecognizer) {
	db.mlRecognizer = recognizer
}

// LoadImages loads all images from the specified directory into the database
func (db *ImageDatabase) LoadImages(imageDir string) error {
	// Check if the directory exists
	if _, err := os.Stat(imageDir); os.IsNotExist(err) {
		return fmt.Errorf("images directory does not exist: %s", imageDir)
	}

	// Read all files in the directory
	files, err := os.ReadDir(imageDir)
	if err != nil {
		return fmt.Errorf("could not read directory: %s", err)
	}

	// Process files in parallel with a limit on concurrent operations
	var wg sync.WaitGroup
	threadLimit := make(chan struct{}, 4)

	for _, file := range files {
		if file.IsDir() {
			continue
		}

		ext := filepath.Ext(file.Name())
		if !imageprocessing.IsImageFile(ext) {
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

			// Compute hash and thumbnail
			hash := imageprocessing.ComputeDCTHash(img)
			thumbnail := imageprocessing.GenerateThumbnail(img, 100)

			// Extract ML features if available
			var features []float32
			if db.mlRecognizer != nil {
				features, err = db.mlRecognizer.ExtractFeatures(img)
				if err != nil {
					log.Printf("Warning: Could not extract ML features from %s: %v", path, err)
				}
			}

			// Store image info
			db.mutex.Lock()
			db.hashes[hash] = ImageInfo{
				Filename:  fileName,
				Hash:      hash,
				AddedAt:   time.Now(),
				Thumbnail: thumbnail,
				Features:  features,
			}
			db.mutex.Unlock()

			log.Printf("Loaded image: %s", fileName)
		}(file.Name())
	}

	wg.Wait()
	log.Printf("Loaded %d images into the database", len(db.hashes))
	return nil
}

// FindMatch searches for a matching image in the database
func (db *ImageDatabase) FindMatch(img image.Image, similarityThreshold float64) (bool, string, float64) {
	// Compute hash for the uploaded image
	uploadedHash := imageprocessing.ComputeDCTHash(img)

	db.mutex.RLock()
	defer db.mutex.RUnlock()

	if len(db.hashes) == 0 {
		return false, "", 0.0
	}

	// Extract features from the uploaded image for ML comparison
	var uploadedFeatures []float32
	var err error
	if db.mlRecognizer != nil {
		uploadedFeatures, err = db.mlRecognizer.ExtractFeatures(img)
		if err != nil {
			log.Printf("Warning: Could not extract ML features from uploaded image: %v", err)
			// Continue with hash-based comparison only
		}
	}

	bestMatch := ""
	bestSimilarity := 0.0

	// Compare with all images in the database
	for hash, info := range db.hashes {
		var similarity float64

		// If ML features are available for both images, use combined comparison
		if db.mlRecognizer != nil && uploadedFeatures != nil && info.Features != nil {
			// Get hash similarity
			hashSimilarity, err := imageprocessing.HashSimilarity(uploadedHash, hash)
			if err != nil {
				continue
			}

			// Get ML feature similarity
			mlSimilarity, err := db.mlRecognizer.CompareFeatures(uploadedFeatures, info.Features)
			if err != nil {
				similarity = hashSimilarity // Fall back to hash similarity
			} else {
				// Combine both similarities with more weight on ML
				similarity = 0.3*hashSimilarity + 0.7*mlSimilarity
			}
		} else {
			// Fall back to hash-based comparison
			hashSimilarity, err := imageprocessing.HashSimilarity(uploadedHash, hash)
			if err != nil {
				continue
			}
			similarity = hashSimilarity
		}

		if similarity > bestSimilarity {
			bestSimilarity = similarity
			bestMatch = info.Filename
		}
	}

	log.Printf("Best match: %s, similarity: %.2f%%, threshold: %.2f%%", bestMatch, bestSimilarity, similarityThreshold)

	isMatch := bestSimilarity >= similarityThreshold

	return isMatch, bestMatch, bestSimilarity
}

// AddImage adds a new image to the database
func (db *ImageDatabase) AddImage(img image.Image, filename string) (string, error) {
	hash := imageprocessing.ComputeDCTHash(img)
	thumbnail := imageprocessing.GenerateThumbnail(img, 100)

	// Extract ML features if available
	var features []float32
	var err error
	if db.mlRecognizer != nil {
		features, err = db.mlRecognizer.ExtractFeatures(img)
		if err != nil {
			log.Printf("Warning: Could not extract ML features from new image: %v", err)
		}
	}

	db.mutex.Lock()
	defer db.mutex.Unlock()

	// Check if the image already exists in the database
	for _, info := range db.hashes {
		if info.Hash == hash {
			return "", errors.New("image already exists in database as: " + info.Filename)
		}
	}

	// Store the new image info
	db.hashes[hash] = ImageInfo{
		Filename:  filename,
		Hash:      hash,
		AddedAt:   time.Now(),
		Thumbnail: thumbnail,
		Features:  features,
	}

	return hash, nil
}

// ListImages returns a list of all images in the database
func (db *ImageDatabase) ListImages() []ImageInfo {
	db.mutex.RLock()
	defer db.mutex.RUnlock()

	images := make([]ImageInfo, 0, len(db.hashes))
	for _, info := range db.hashes {
		// Create a copy without the Features field
		images = append(images, ImageInfo{
			Filename:  info.Filename,
			Hash:      info.Hash,
			AddedAt:   info.AddedAt,
			Thumbnail: info.Thumbnail,
		})
	}

	return images
}
