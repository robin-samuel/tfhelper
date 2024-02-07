package tfhelper

import (
	"image"

	tf "github.com/galeone/tensorflow/tensorflow/go"
)

// ImageToBGRTensor converts an image.Image to a TensorFlow tensor with shape [1, height, width, 3]
//   - pixel values in the range [0, 255]
//   - the pixel order is BGR (the standard for OpenCV)
func ImageToBGRTensor(img image.Image) (*tf.Tensor, error) {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()

	tensorData := make([][][][]float32, 1)
	tensorData[0] = make([][][]float32, height)

	for y := 0; y < height; y++ {
		row := make([][]float32, width)
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			row[x] = []float32{float32(b >> 8), float32(g >> 8), float32(r >> 8)}
		}
		tensorData[0][y] = row
	}

	tensor, err := tf.NewTensor(tensorData)
	if err != nil {
		return nil, err
	}

	return tensor, nil
}

// ImageToRGBTensor converts an image.Image to a TensorFlow tensor with shape [1, height, width, 3]
//   - pixel values in the range [0, 255]
//   - the pixel order is RGB
func ImageToRGBTensor(img image.Image) (*tf.Tensor, error) {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()

	tensorData := make([][][][]float32, 1)
	tensorData[0] = make([][][]float32, height)

	for y := 0; y < height; y++ {
		row := make([][]float32, width)
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			row[x] = []float32{float32(r >> 8), float32(g >> 8), float32(b >> 8)}
		}
		tensorData[0][y] = row
	}

	tensor, err := tf.NewTensor(tensorData)
	if err != nil {
		return nil, err
	}

	return tensor, nil
}

// ImageToBGRNormalizedTensor converts an image.Image to a TensorFlow tensor with shape [1, height, width, 3]
//   - pixel values in the range [0, 1].
//   - the pixel order is BGR (the standard for OpenCV).
func ImageToBGRNormalizedTensor(img image.Image) (*tf.Tensor, error) {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()

	tensorData := make([][][][]float32, 1)
	tensorData[0] = make([][][]float32, height)

	for y := 0; y < height; y++ {
		row := make([][]float32, width)
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			row[x] = []float32{float32(b>>8) / 255, float32(g>>8) / 255, float32(r>>8) / 255}
		}
		tensorData[0][y] = row
	}

	tensor, err := tf.NewTensor(tensorData)
	if err != nil {
		return nil, err
	}

	return tensor, nil
}

// ImageToRGBNormalizedTensor converts an image.Image to a TensorFlow tensor with shape [1, height, width, 3]
//   - pixel values in the range [0, 1].
//   - the pixel order is RGB.
func ImageToRGBNormalizedTensor(img image.Image) (*tf.Tensor, error) {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()

	tensorData := make([][][][]float32, 1)
	tensorData[0] = make([][][]float32, height)

	for y := 0; y < height; y++ {
		row := make([][]float32, width)
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			row[x] = []float32{float32(r>>8) / 255, float32(g>>8) / 255, float32(b>>8) / 255}
		}
		tensorData[0][y] = row
	}

	tensor, err := tf.NewTensor(tensorData)
	if err != nil {
		return nil, err
	}

	return tensor, nil
}
