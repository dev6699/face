package yoloface

import (
	"image"
	"math"

	"github.com/dev6699/face/protobuf"
	"gocv.io/x/gocv"
)

func (m *Model) PreProcess(i *Input) ([]*protobuf.InferTensorContents, error) {
	img := i.Img
	width := img.Cols()
	height := img.Rows()

	faceDetectorSize := Resolution{Width: 640, Height: 640}
	resizedVisionFrame, newWidth, newHeight := resizeFrameResolution(img.Clone(), faceDetectorSize)
	defer resizedVisionFrame.Close()

	ratioHeight := float32(height) / float32(newHeight)
	ratioWidth := float32(width) / float32(newWidth)
	m.ratioHeight = ratioHeight
	m.ratioWidth = ratioWidth

	contents := &protobuf.InferTensorContents{
		Fp32Contents: prepareDetectFrame(resizedVisionFrame, faceDetectorSize),
	}
	return []*protobuf.InferTensorContents{contents}, nil
}

type Resolution struct {
	Width  uint
	Height uint
}

// resizeFrameResolution resize visionFrame where its resolution will be capped at maxResolution.
func resizeFrameResolution(visionFrame gocv.Mat, maxResolution Resolution) (gocv.Mat, uint, uint) {
	width := visionFrame.Cols()
	height := visionFrame.Rows()

	maxHeight := int(maxResolution.Height)
	maxWidth := int(maxResolution.Width)

	if height > maxHeight || width > maxWidth {
		scale := math.Min(float64(maxHeight)/float64(height), float64(maxWidth)/float64(width))
		newWidth := int(float64(width) * scale)
		newHeight := int(float64(height) * scale)

		gocv.Resize(visionFrame, &visionFrame, image.Point{X: newWidth, Y: newHeight}, 0, 0, gocv.InterpolationDefault)
		return visionFrame, uint(newWidth), uint(newHeight)
	}

	return visionFrame, uint(width), uint(height)
}

func prepareDetectFrame(visionFrame gocv.Mat, faceDetectorSize Resolution) []float32 {
	faceDetectorWidth := int(faceDetectorSize.Width)
	faceDetectorHeight := int(faceDetectorSize.Height)

	detectVisionFrame := gocv.NewMatWithSize(faceDetectorHeight, faceDetectorWidth, gocv.MatTypeCV8UC3)
	defer detectVisionFrame.Close()

	roi := detectVisionFrame.Region(image.Rect(0, 0, visionFrame.Cols(), visionFrame.Rows()))
	defer roi.Close()
	visionFrame.CopyTo(&roi)

	output := make([]float32, 3*faceDetectorHeight*faceDetectorWidth)
	idx := 0
	for y := 0; y < faceDetectorHeight; y++ {
		for x := 0; x < faceDetectorWidth; x++ {
			pixel := detectVisionFrame.GetVecbAt(y, x)

			output[idx] = (float32(pixel[0]) - 127.5) / 128.0
			output[faceDetectorHeight*faceDetectorWidth+idx] = (float32(pixel[1]) - 127.5) / 128.0
			output[2*faceDetectorHeight*faceDetectorWidth+idx] = (float32(pixel[2]) - 127.5) / 128.0
			idx++
		}
	}

	return output
}
