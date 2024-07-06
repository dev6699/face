package gfpgan

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
	// 			512,
	// 			512
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
		d[i+2] = uint8(math.Round(255.0 * float64(clip(output[j], -1, 1)+1.0) / 2.0))
		d[i+1] = uint8(math.Round(255.0 * float64(clip(output[len(d)/3+j], -1, 1)+1.0) / 2.0))
		d[i] = uint8(math.Round(255.0 * float64(clip(output[len(d)/3*2+j], -1, 1)+1.0) / 2.0))
		j++
	}

	width := 512
	height := 512
	imgType := gocv.MatTypeCV8UC3

	mat, err := gocv.NewMatFromBytes(height, width, imgType, d)
	if err != nil {
		return nil, err
	}
	defer mat.Close()

	boxMask := model.CreateStaticBoxMask(m.cropSize, 0.3, model.Padding{Top: 0, Right: 0, Bottom: 0, Left: 0})
	defer boxMask.Close()

	cropMask := model.ReduceMinimum([]gocv.Mat{boxMask})
	defer cropMask.Close()
	model.ClipMat(cropMask, 0, 1)

	outMat := model.PasteBack(m.img, mat, cropMask, m.affineMatrix)
	defer outMat.Close()
	defer m.affineMatrix.Close()

	return &Output{
		OutFrame: m.blendFrame(m.img, outMat),
	}, nil
}

// blendFrame blends two frame (gocv.Mat) images with a specified blending factor.
func (m *Model) blendFrame(tempVisionFrame, pasteVisionFrame gocv.Mat) gocv.Mat {
	faceEnhancerBlendRatio := 1.0 - (m.faceEnhancerBlend / 100.0)
	outputFrame := gocv.NewMat()
	gocv.AddWeighted(tempVisionFrame, faceEnhancerBlendRatio, pasteVisionFrame, 1-faceEnhancerBlendRatio, 0, &outputFrame)
	return outputFrame
}

func clip(v float32, min float32, max float32) float32 {
	if v < min {
		return min
	}

	if v > max {
		return max
	}

	return v
}
