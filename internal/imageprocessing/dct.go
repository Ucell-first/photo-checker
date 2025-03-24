package imageprocessing

import (
	"errors"
	"image"
	"image/color"
	"strings"

	"github.com/disintegration/imaging"
)

// Error definitions
var (
	ErrHashLengthMismatch = errors.New("hash lengths do not match")
)

// ComputeDCTHash generates a perceptual hash of an image using DCT-like technique
// The hash is a binary string that represents the image content
// This is useful for finding similar images regardless of size or minor changes
func ComputeDCTHash(img image.Image) string {
	// Step 1: Resize the image to 32x32 to normalize size
	resized := imaging.Resize(img, 32, 32, imaging.Lanczos)

	// Step 2: Convert to grayscale to remove color information
	gray := imaging.Grayscale(resized)

	// Step 3: Compute the hash using frequency components
	const blockSize = 8
	const numBlocks = 16
	blockValues := make([]float64, numBlocks)

	// Compute average pixel value for each block
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

	// Calculate the average of all blocks
	var sum float64
	for _, val := range blockValues {
		sum += val
	}
	avg := sum / float64(len(blockValues))

	// Build the hash string based on whether each block is above or below average
	var hash strings.Builder
	for _, val := range blockValues {
		if val >= avg {
			hash.WriteString("1")
		} else {
			hash.WriteString("0")
		}
	}

	// Add directional gradients for better differentiation
	// Horizontal gradient
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

	// Diagonal gradient
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
