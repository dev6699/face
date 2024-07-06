package inswapper

import (
	"math"

	"github.com/dev6699/face/model"
	"gocv.io/x/gocv"
)

func (m *Model) PostProcess(rawOutputContents [][]byte) (*Output, error) {
	// "outputs": [
	// 	{
	// 		"name": "output",
	// 		"datatype": "FP32",
	// 		"shape": [
	// 			1,
	// 			3,
	// 			128,
	// 			128
	// 		]
	// 	}
	// ]
	output, err := model.BytesToFloat32Slice(rawOutputContents[0])
	if err != nil {
		return nil, err
	}

	d := make([]uint8, len(output))
	j := 0
	for i := 0; i < len(output); i += 3 {
		d[i+2] = uint8(math.Round(float64(output[j]) * 255.0))
		d[i+1] = uint8(math.Round(float64(output[len(d)/3+j]) * 255.0))
		d[i] = uint8(math.Round(float64(output[len(d)/3*2+j]) * 255.0))
		j++
	}

	width := 128
	height := 128
	imgType := gocv.MatTypeCV8UC3

	mat, err := gocv.NewMatFromBytes(height, width, imgType, d)
	if err != nil {
		return nil, err
	}

	return &Output{
		CropVisionFrame: mat,
	}, nil
}
