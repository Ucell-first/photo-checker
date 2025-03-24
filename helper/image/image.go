package image

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"math"
	"strings"

	"github.com/disintegration/imaging"
)

// computeDCTHash calculates perceptual hash using Discrete Cosine Transform
func ComputeDCTHash(img image.Image) string {
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
func ExtractImageFeatures(img image.Image) []float64 {
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
func CosineSimilarity(a, b []float64) float64 {
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
func GenerateThumbnail(img image.Image) string {
	thumbnail := imaging.Resize(img, 100, 0, imaging.Lanczos)
	var buf bytes.Buffer
	err := imaging.Encode(&buf, thumbnail, imaging.JPEG)
	if err != nil {
		return ""
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

// hammingDistance calculates difference between two hashes
func HammingDistance(hash1, hash2 string) (int, error) {
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
