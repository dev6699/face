package model

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/color"
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

func InvertAffineMatrix(affineMatrix gocv.Mat) gocv.Mat {
	invertAffineMatrix := gocv.NewMatWithSize(2, 3, gocv.MatTypeCV64F)
	gocv.InvertAffineTransform(affineMatrix, &invertAffineMatrix)
	return invertAffineMatrix
}

// CalculateMean2f calculates the mean of a slice of gocv.Point2f
func CalculateMean2f(points []gocv.Point2f) gocv.Point2f {
	if len(points) == 0 {
		return gocv.Point2f{}
	}

	var sumX, sumY float32
	for _, point := range points {
		sumX += point.X
		sumY += point.Y
	}

	meanX := sumX / float32(len(points))
	meanY := sumY / float32(len(points))

	return gocv.Point2f{X: meanX, Y: meanY}
}

// Size represents the width and height dimensions.
type Size struct {
	Width, Height int
}

func WarpFaceByFaceLandmark5(visionFrame gocv.Mat, faceLandmark5 []gocv.Point2f, warpTemplate []gocv.Point2f, cropSize Size) (gocv.Mat, gocv.Mat) {
	affineMatrix := estimateMatrixByFaceLandmark5(faceLandmark5, warpTemplate, cropSize)
	cropVisionFrame := gocv.NewMat()

	gocv.WarpAffineWithParams(
		visionFrame,
		&cropVisionFrame,
		affineMatrix,
		image.Pt(cropSize.Width, cropSize.Height),
		gocv.InterpolationArea,
		gocv.BorderReplicate,
		color.RGBA{},
	)

	return cropVisionFrame, affineMatrix
}

func estimateMatrixByFaceLandmark5(faceLandmark5 []gocv.Point2f, warpTemplate []gocv.Point2f, cropSize Size) gocv.Mat {
	normedWarpTemplate := normalizeWarpTemplate(warpTemplate, cropSize)
	pvsrc := gocv.NewPoint2fVectorFromPoints(faceLandmark5)
	pvdst := gocv.NewPoint2fVectorFromPoints(normedWarpTemplate)
	inliers := gocv.NewMat()
	defer inliers.Close()
	method := 8
	ransacProjThreshold := 100.0
	maxiters := uint(2000)
	confidence := 0.99
	refineIters := uint(10)
	// https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gad767faff73e9cbd8b9d92b955b50062d
	affineMatrix := gocv.EstimateAffinePartial2DWithParams(pvsrc, pvdst, inliers, method, ransacProjThreshold, maxiters, confidence, refineIters)
	return affineMatrix
}

// normalizeWarpTemplate scales the warp template according to the crop size.
func normalizeWarpTemplate(warpTemplate []gocv.Point2f, cropSize Size) []gocv.Point2f {
	normedWarpTemplate := make([]gocv.Point2f, len(warpTemplate))
	for i, pt := range warpTemplate {
		normedWarpTemplate[i] = gocv.Point2f{
			X: pt.X * float32(cropSize.Width),
			Y: pt.Y * float32(cropSize.Height),
		}
	}
	return normedWarpTemplate
}

// MatSubtract subtract value v fom mat
func MatSubtract(mat gocv.Mat, v float64) {
	constantValue := float64(v)
	constantMat := gocv.NewMatWithSizeFromScalar(gocv.NewScalar(constantValue, constantValue, constantValue, 0), mat.Rows(), mat.Cols(), mat.Type())
	defer constantMat.Close()
	gocv.Subtract(mat, constantMat, &mat)
}

// ClipMat clips the values of a gocv.Mat within a specified range.
// mat need to be float32 type
func ClipMat(mat gocv.Mat, minVal, maxVal float32) {
	for row := 0; row < mat.Rows(); row++ {
		for col := 0; col < mat.Cols(); col++ {
			value := mat.GetFloatAt(row, col)
			if value < minVal {
				value = minVal
			} else if value > maxVal {
				value = maxVal
			}
			mat.SetFloatAt(row, col, value)
		}
	}
}
