package imageprocessing

import (
	"image"
	"math"

	"github.com/nfnt/resize"
	"github.com/pkg/errors"
	tf "github.com/wamuir/graft/tensorflow"
)

// MLImageRecognizer implements machine learning-based image recognition
type MLImageRecognizer struct {
	model     *tf.SavedModel
	modelPath string
	loaded    bool
}

// NewMLImageRecognizer creates a new ML-based image recognizer
func NewMLImageRecognizer(modelPath string) *MLImageRecognizer {
	return &MLImageRecognizer{
		modelPath: modelPath,
		loaded:    false,
	}
}

// Load loads the pre-trained TensorFlow model
func (r *MLImageRecognizer) Load() error {
	model, err := tf.LoadSavedModel(r.modelPath, []string{"serve"}, nil)
	if err != nil {
		return errors.Wrap(err, "failed to load ML model")
	}

	r.model = model
	r.loaded = true
	return nil
}

// Close releases resources associated with the ML model
func (r *MLImageRecognizer) Close() {
	if r.loaded && r.model != nil {
		r.model.Session.Close()
		r.loaded = false
	}
}

// ExtractFeatures extracts ML features from an image
// The features are a high-dimensional vector representing the image content
func (r *MLImageRecognizer) ExtractFeatures(img image.Image) ([]float32, error) {
	if !r.loaded {
		return nil, errors.New("ML model not loaded")
	}

	// Preprocess the image for the model
	resizedImg := resize.Resize(224, 224, img, resize.Lanczos3)
	normalizedImg := normalizeImage(resizedImg)

	// Create tensor for the input image
	tensor, err := tf.NewTensor(normalizedImg)
	if err != nil {
		return nil, errors.Wrap(err, "failed to create input tensor")
	}

	// Run the model
	output, err := r.model.Session.Run(
		map[tf.Output]*tf.Tensor{
			r.model.Graph.Operation("input").Output(0): tensor,
		},
		[]tf.Output{
			r.model.Graph.Operation("features").Output(0),
		},
		nil,
	)
	if err != nil {
		return nil, errors.Wrap(err, "failed to run inference")
	}

	// Extract features from the output tensor
	features, ok := output[0].Value().([][]float32)
	if !ok {
		return nil, errors.New("unexpected output format")
	}

	return features[0], nil
}

// CompareFeatures compares two feature vectors and returns a similarity score
func (r *MLImageRecognizer) CompareFeatures(features1, features2 []float32) (float64, error) {
	if len(features1) != len(features2) {
		return 0, errors.New("feature vectors have different dimensions")
	}

	// Calculate cosine similarity
	var dotProduct float64
	var norm1 float64
	var norm2 float64

	for i := 0; i < len(features1); i++ {
		dotProduct += float64(features1[i] * features2[i])
		norm1 += float64(features1[i] * features1[i])
		norm2 += float64(features2[i] * features2[i])
	}

	// Prevent division by zero
	if norm1 == 0 || norm2 == 0 {
		return 0, nil
	}

	similarity := dotProduct / (math.Sqrt(norm1) * math.Sqrt(norm2))

	// Convert to percentage
	return similarity * 100, nil
}

// normalizeImage preprocesses an image for the ML model
func normalizeImage(img image.Image) [][][]float32 {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()

	// Create a 3D array for the image (height, width, 3 channels)
	result := make([][][]float32, height)
	for y := 0; y < height; y++ {
		result[y] = make([][]float32, width)
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()

			// Convert 16-bit color to float32 and normalize to [0,1]
			result[y][x] = []float32{
				float32(r) / 65535.0,
				float32(g) / 65535.0,
				float32(b) / 65535.0,
			}
		}
	}

	return result
}

// RecognizeImage performs image recognition using both DCT hash and ML features
// It returns a combined similarity score
func RecognizeImage(img1, img2 image.Image, mlRecognizer *MLImageRecognizer) (float64, error) {
	// Get DCT hash similarity
	hash1 := ComputeDCTHash(img1)
	hash2 := ComputeDCTHash(img2)
	hashSimilarity, err := HashSimilarity(hash1, hash2)
	if err != nil {
		return 0, err
	}

	// If ML recognizer is available, use it to improve recognition
	if mlRecognizer != nil && mlRecognizer.loaded {
		// Get ML feature similarity
		features1, err := mlRecognizer.ExtractFeatures(img1)
		if err != nil {
			return hashSimilarity, nil // Fall back to hash similarity
		}

		features2, err := mlRecognizer.ExtractFeatures(img2)
		if err != nil {
			return hashSimilarity, nil // Fall back to hash similarity
		}

		mlSimilarity, err := mlRecognizer.CompareFeatures(features1, features2)
		if err != nil {
			return hashSimilarity, nil // Fall back to hash similarity
		}

		// Combine both similarities with more weight on ML
		return 0.3*hashSimilarity + 0.7*mlSimilarity, nil
	}

	// If ML is not available, return hash similarity
	return hashSimilarity, nil
}
