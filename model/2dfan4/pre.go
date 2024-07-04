package _2dfan4

import (
	"image"
	"math"

	"github.com/dev6699/face/model"
	"github.com/dev6699/face/protobuf"
	"gocv.io/x/gocv"
)

func (m *Model) PreProcess(i *Input) ([]*protobuf.InferTensorContents, error) {
	scale := calculateScale(i.BoundingBox, 195.0)
	targetSize := 256.0
	translation := calculateTranslation(i.BoundingBox, scale, targetSize)

	cropVisionFrame, affineMatrix := model.WarpFaceByTranslation(i.Img, translation, scale, image.Point{X: 256, Y: 256})
	defer cropVisionFrame.Close()
	m.affineMatrix = affineMatrix
	gocv.CvtColor(cropVisionFrame, &cropVisionFrame, gocv.ColorRGBToLab)

	labChannels := gocv.Split(cropVisionFrame)
	defer labChannels[0].Close()
	defer labChannels[1].Close()
	defer labChannels[2].Close()

	meanValue := calculateMean(labChannels[0])
	threshold := 30.0
	clipLimit := float32(2.0)

	if meanValue < threshold {
		applyCLAHE(labChannels[0], clipLimit)
	}

	d := []float32{}

	gocv.CvtColor(cropVisionFrame, &cropVisionFrame, gocv.ColorLabToRGB)

	cropVisionFrame.ConvertTo(&cropVisionFrame, gocv.MatTypeCV32F)
	cropVisionFrame.DivideFloat(255.0)
	rgbChannels := gocv.Split(cropVisionFrame)
	r := rgbChannels[0]
	defer r.Close()
	rd, _ := r.DataPtrFloat32()
	d = append(d, rd...)

	g := rgbChannels[1]
	defer g.Close()
	gd, _ := g.DataPtrFloat32()
	d = append(d, gd...)

	b := rgbChannels[2]
	defer b.Close()
	bd, _ := b.DataPtrFloat32()
	d = append(d, bd...)

	contents := &protobuf.InferTensorContents{
		Fp32Contents: d,
	}
	return []*protobuf.InferTensorContents{contents}, nil
}

// calculateScale calculates the scale factor based on the bounding box dimensions.
func calculateScale(boundingBox model.BoundingBox, factor float64) float64 {
	width := boundingBox.X2 - boundingBox.X1
	height := boundingBox.Y2 - boundingBox.Y1

	maxDim := math.Max(float64(width), float64(height))

	scale := factor / float64(maxDim)
	return scale
}

// calculateTranslation computes the translation vector based on the bounding box and scale.
func calculateTranslation(boundingBox model.BoundingBox, scale float64, targetSize float64) model.Translation {
	translationX := (targetSize - ((boundingBox.X1 + boundingBox.X2) * scale)) * 0.5
	translationY := (targetSize - ((boundingBox.Y1 + boundingBox.Y2) * scale)) * 0.5

	return model.Translation{translationX, translationY}
}

// calculateMean calculates the mean of the first channel of an image.
func calculateMean(mat gocv.Mat) float64 {
	totalSum := 0.0
	pixelCount := mat.Rows() * mat.Cols()
	for i := 0; i < mat.Rows(); i++ {
		for j := 0; j < mat.Cols(); j++ {
			value := mat.GetUCharAt(i, j)
			totalSum += float64(value)
		}
	}
	return totalSum / float64(pixelCount)
}

// applyCLAHE applies CLAHE to a single-channel image.
func applyCLAHE(src gocv.Mat, clipLimit float32) {
	// https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html#gad3b7f72da85b821fda2bc41687573974
	clahe := gocv.NewCLAHEWithParams(float64(clipLimit), image.Point{X: 8, Y: 8})
	clahe.Apply(src, &src)
	defer clahe.Close()
}
