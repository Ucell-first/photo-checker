// Package imageprocessing provides image processing utilities for the photot application.
// It includes functions for image hashing, thumbnail generation, and machine learning
// based image recognition.
package imageprocessing

import (
	"bytes"
	"encoding/base64"
	"image"
	"path/filepath"
	"strings"

	"github.com/disintegration/imaging"
)

// SupportedImageFormats is a map of supported image file extensions
var SupportedImageFormats = map[string]bool{
	".jpg":  true,
	".jpeg": true,
	".png":  true,
	".gif":  true,
	".bmp":  true,
	".tiff": true,
	".webp": true,
}

// IsImageFile checks if the file extension is a supported image format
func IsImageFile(filename string) bool {
	ext := strings.ToLower(filepath.Ext(filename))
	return SupportedImageFormats[ext]
}

// GenerateThumbnail creates a smaller version of the image
// and returns it as a base64-encoded string
func GenerateThumbnail(img image.Image, size int) string {
	// Resize the image while maintaining aspect ratio
	thumbnail := imaging.Resize(img, size, 0, imaging.Lanczos)

	// Encode the thumbnail to JPEG format
	var buf bytes.Buffer
	err := imaging.Encode(&buf, thumbnail, imaging.JPEG)
	if err != nil {
		return ""
	}

	// Convert to base64 for easy transmission
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

// HammingDistance calculates the number of positions at which the corresponding bits are different
func HammingDistance(hash1, hash2 string) (int, error) {
	if len(hash1) != len(hash2) {
		return 0, ErrHashLengthMismatch
	}

	distance := 0
	for i := 0; i < len(hash1); i++ {
		if hash1[i] != hash2[i] {
			distance++
		}
	}
	return distance, nil
}

// HashSimilarity calculates how similar two hashes are as a percentage
func HashSimilarity(hash1, hash2 string) (float64, error) {
	dist, err := HammingDistance(hash1, hash2)
	if err != nil {
		return 0, err
	}

	maxDistance := len(hash1)
	similarity := 100.0 - (float64(dist) / float64(maxDistance) * 100.0)
	return similarity, nil
}
