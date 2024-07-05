package faceoccluder

import (
	"image"

	"github.com/dev6699/face/model"
	"gocv.io/x/gocv"
)

func (m *Model) PostProcess(rawOutputContents [][]byte) (*Output, error) {
	// "outputs": [
	// 	{
	// 		"name": "out_mask:0",
	// 		"datatype": "FP32",
	// 		"shape": [
	// 			-1,
	// 			256,
	// 			256,
	// 			1
	// 		]
	// 	}
	// ]

	outMask, err := model.BytesToFloat32Slice(rawOutputContents[0])
	if err != nil {
		return nil, err
	}

	rows := 256
	cols := 256
	maskMat := gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV32F)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			idx := i*cols + j
			maskMat.SetFloatAt(i, j, outMask[idx])
		}
	}

	model.ClipMat(maskMat, 0, 1)
	gocv.Resize(maskMat, &maskMat, image.Point{X: m.cropSize.Width, Y: m.cropSize.Height}, 0, 0, gocv.InterpolationDefault)
	gocv.GaussianBlur(maskMat, &maskMat, image.Point{0, 0}, 5.0, 0, gocv.BorderDefault)

	model.ClipMat(maskMat, 0.5, 1)
	model.MatSubtract(maskMat, 0.5)
	maskMat.MultiplyFloat(2)
	cropMask := reduceMinimum([]gocv.Mat{m.boxMask, maskMat})
	model.ClipMat(cropMask, 0, 1)

	defer m.boxMask.Close()
	defer maskMat.Close()

	return &Output{
		CropVisionFrame: m.cropVisionFrame,
		AffineMatrix:    m.affineMatrix,
		CropMask:        cropMask,
	}, nil
}

// reduceMinimum finds the element-wise minimum of a list of gocv.Mat
func reduceMinimum(mats []gocv.Mat) gocv.Mat {
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
