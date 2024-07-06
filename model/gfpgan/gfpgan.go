package gfpgan

import (
	"github.com/dev6699/face/model"
	"gocv.io/x/gocv"
)

type Model struct {
	faceEnhancerBlend float64

	img          gocv.Mat
	cropSize     model.Size
	affineMatrix gocv.Mat
}

type Input struct {
	Img           gocv.Mat
	FaceLandmark5 []gocv.Point2f
}

type Output struct {
	OutFrame gocv.Mat
}

type ModelT = model.Model[*Input, *Output]

var _ ModelT = &Model{}

func NewFactory(faceEnhancerBlend float64) func() ModelT {
	return func() ModelT {
		return New(faceEnhancerBlend)
	}
}

func New(faceEnhancerBlend float64) *Model {
	return &Model{
		faceEnhancerBlend: faceEnhancerBlend,
	}
}

func (m *Model) ModelName() string {
	return "gfpgan_1.4"
}

func (m *Model) ModelVersion() string {
	return "1"
}
