package faceoccluder

import (
	"image"
	"math"

	"github.com/dev6699/face/model"
	"github.com/dev6699/face/protobuf"
	"gocv.io/x/gocv"
)

func (m *Model) PreProcess(i *Input) ([]*protobuf.InferTensorContents, error) {
	cropSize := model.Size{Width: 128, Height: 128}
	m.cropSize = cropSize
	cropVisionFrame, affineMatrix := model.WarpFaceByFaceLandmark5(i.Img, i.FaceLandmark5, arcface_128_v2, cropSize)
	m.cropVisionFrame = cropVisionFrame
	m.affineMatrix = affineMatrix

	boxMask := createStaticBoxMask(model.Size{Width: 128, Height: 128}, 0.3, Padding{Top: 0, Right: 0, Bottom: 0, Left: 0})
	m.boxMask = boxMask

	resizedFrame := gocv.NewMat()
	defer resizedFrame.Close()
	gocv.Resize(cropVisionFrame, &resizedFrame, image.Point{X: 256, Y: 256}, 0, 0, gocv.InterpolationDefault)
	data, _ := resizedFrame.DataPtrUint8()

	d := make([]float32, len(data))
	for i, a := range data {
		d[i] = float32(a) / 255.0
	}

	contents := &protobuf.InferTensorContents{
		Fp32Contents: d,
	}
	return []*protobuf.InferTensorContents{contents}, nil
}

// Padding represents the padding values for the mask.
type Padding struct {
	Top, Right, Bottom, Left float64
}

// Create a static box mask with specified size, blur, and padding.
func createStaticBoxMask(cropSize model.Size, faceMaskBlur float64, faceMaskPadding Padding) gocv.Mat {
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
