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

// Padding represents the padding values for the mask.
type Padding struct {
	Top, Right, Bottom, Left float64
}

// CreateStaticBoxMask create a static box mask with specified size, blur, and padding.
func CreateStaticBoxMask(cropSize Size, faceMaskBlur float64, faceMaskPadding Padding) gocv.Mat {
	blurAmount := int(float64(cropSize.Width) * 0.5 * faceMaskBlur)
	blurArea := int(math.Max(float64(blurAmount/2), 1))

	// Create a box mask initialized to ones.
	boxMask := gocv.NewMatWithSize(cropSize.Height, cropSize.Width, gocv.MatTypeCV32F)
	boxMask.SetTo(gocv.NewScalar(1.0, 1.0, 1.0, 1.0)) // Fill the entire matrix with ones.

	// Calculate padding values.
	padTop := int(math.Max(float64(blurArea), float64(cropSize.Height)*faceMaskPadding.Top/100))
	padBottom := int(math.Max(float64(blurArea), float64(cropSize.Height)*faceMaskPadding.Bottom/100))
	padLeft := int(math.Max(float64(blurArea), float64(cropSize.Width)*faceMaskPadding.Left/100))
	padRight := int(math.Max(float64(blurArea), float64(cropSize.Width)*faceMaskPadding.Right/100))

	// Set padding areas to zero.
	topRegion := boxMask.Region(image.Rect(0, 0, cropSize.Width, padTop))
	defer topRegion.Close()
	topRegion.SetTo(gocv.NewScalar(0, 0, 0, 0))

	bottomRegion := boxMask.Region(image.Rect(0, cropSize.Height-padBottom, cropSize.Width, cropSize.Height))
	defer bottomRegion.Close()
	bottomRegion.SetTo(gocv.NewScalar(0, 0, 0, 0))

	leftRegion := boxMask.Region(image.Rect(0, 0, padLeft, cropSize.Height))
	defer leftRegion.Close()
	leftRegion.SetTo(gocv.NewScalar(0, 0, 0, 0))

	rightRegion := boxMask.Region(image.Rect(cropSize.Width-padRight, 0, cropSize.Width, cropSize.Height))
	defer rightRegion.Close()
	rightRegion.SetTo(gocv.NewScalar(0, 0, 0, 0))

	// Apply Gaussian blur if required.
	if blurAmount > 0 {
		gocv.GaussianBlur(boxMask, &boxMask, image.Point{0, 0}, float64(blurAmount)*0.25, 0, gocv.BorderDefault)
	}

	return boxMask
}

// ReduceMinimum finds the element-wise minimum of a list of gocv.Mat
func ReduceMinimum(mats []gocv.Mat) gocv.Mat {
	if len(mats) == 0 {
		return gocv.NewMat()
	}

	// Start with the first matrix as the initial minimum
	minMat := mats[0].Clone()
	rows, cols := minMat.Rows(), minMat.Cols()

	// Iterate over the remaining matrices
	for i := 1; i < len(mats); i++ {
		for row := 0; row < rows; row++ {
			for col := 0; col < cols; col++ {
				currentMin := minMat.GetFloatAt(row, col)
				newValue := mats[i].GetFloatAt(row, col)

				// Update the minimum value
				if newValue < currentMin {
					minMat.SetFloatAt(row, col, newValue)
				}
			}
		}
	}

	return minMat
}

// PasteBack paste cropVisionFrame back to targetVisionFrame
func PasteBack(targetVisionFrame gocv.Mat, cropVisionFrame gocv.Mat, cropMask gocv.Mat, affineMatrix gocv.Mat) gocv.Mat {
	inverseMatrix := InvertAffineMatrix(affineMatrix)
	defer inverseMatrix.Close()
	tempSize := image.Pt(targetVisionFrame.Cols(), targetVisionFrame.Rows())
	cropVisionFrame.ConvertTo(&cropVisionFrame, gocv.MatTypeCV64F)
	inverseVisionFrame := getInverseVisionFrame(cropVisionFrame, inverseMatrix, tempSize)
	defer inverseVisionFrame.Close()

	inverseMask := getInverseMask(inverseMatrix, cropMask, tempSize)
	defer inverseMask.Close()
	inverseMaskData, _ := inverseMask.DataPtrFloat64()
	inverseVisionFrameData, _ := inverseVisionFrame.DataPtrFloat64()

	data, _ := targetVisionFrame.DataPtrUint8()
	d := make([]uint8, len(data))
	j := 0
	for i := 0; i < len(data); i += 3 {
		inverseM := inverseMaskData[j]
		d[i] = uint8(inverseM*inverseVisionFrameData[i] + (1-inverseM)*float64(data[i]))
		d[i+1] = uint8(inverseM*inverseVisionFrameData[i+1] + (1-inverseM)*float64(data[i+1]))
		d[i+2] = uint8(inverseM*inverseVisionFrameData[i+2] + (1-inverseM)*float64(data[i+2]))
		j++
	}

	mat, _ := gocv.NewMatFromBytes(targetVisionFrame.Rows(), targetVisionFrame.Cols(), gocv.MatTypeCV8UC3, d)
	return mat
}

func getInverseMask(inverseMatrix gocv.Mat, cropMask gocv.Mat, tempSize image.Point) gocv.Mat {
	inverseMask := gocv.NewMat()
	gocv.WarpAffine(
		cropMask,
		&inverseMask,
		inverseMatrix,
		tempSize,
	)
	inverseMask.ConvertTo(&inverseMask, gocv.MatTypeCV32F)
	ClipMat(inverseMask, 0, 1)
	inverseMask.ConvertTo(&inverseMask, gocv.MatTypeCV64F)
	return inverseMask
}

func getInverseVisionFrame(cropVisionFrame gocv.Mat, inverseMatrix gocv.Mat, tempSize image.Point) gocv.Mat {
	inverseVisionFrame := gocv.NewMat()
	cropVisionFrame.ConvertTo(&cropVisionFrame, gocv.MatTypeCV64F)
	gocv.WarpAffineWithParams(cropVisionFrame, &inverseVisionFrame, inverseMatrix, tempSize, gocv.InterpolationLinear, gocv.BorderReplicate, color.RGBA{})
	inverseVisionFrame.ConvertTo(&inverseVisionFrame, gocv.MatTypeCV64F)
	return inverseVisionFrame
}
