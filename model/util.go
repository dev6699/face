package model

import (
	"bytes"
	"encoding/binary"
	"image"
	"io"
	"math"

	"gocv.io/x/gocv"
)

func BytesToFloat32Slice(data []byte) ([]float32, error) {
	t := []float32{}

	// Create a buffer from the input data
	buffer := bytes.NewReader(data)
	for {
		// Read the binary data from the buffer
		var binaryValue uint32
		err := binary.Read(buffer, binary.LittleEndian, &binaryValue)
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}

		t = append(t, math.Float32frombits(binaryValue))

	}

	return t, nil
}

// Argmax return the index of the maximum value in a slice
func Argmax(slice []float32) int {
	maxIndex := 0
	maxValue := slice[0]

	for i, value := range slice {
		if value > maxValue {
			maxIndex = i
			maxValue = value
		}
	}
	return maxIndex
}

// Translation represents the translation vector.
type Translation [2]float64

func WarpFaceByTranslation(visionFrame gocv.Mat, translation Translation, scale float64, cropSize image.Point) (gocv.Mat, gocv.Mat) {
	affineMatrix := gocv.NewMatWithSize(2, 3, gocv.MatTypeCV64F)
	affineMatrix.SetDoubleAt(0, 0, scale)
	affineMatrix.SetDoubleAt(0, 1, 0.0)
	affineMatrix.SetDoubleAt(0, 2, translation[0])
	affineMatrix.SetDoubleAt(1, 0, 0.0)
	affineMatrix.SetDoubleAt(1, 1, scale)
	affineMatrix.SetDoubleAt(1, 2, translation[1])

	cropVisionFrame := gocv.NewMat()
	gocv.WarpAffine(visionFrame, &cropVisionFrame, affineMatrix, cropSize)

	return cropVisionFrame, affineMatrix
}
