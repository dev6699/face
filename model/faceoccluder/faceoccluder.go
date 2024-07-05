package faceoccluder

import (
	"github.com/dev6699/face/model"
	"gocv.io/x/gocv"
)

type Model struct {
	cropSize        model.Size
	cropVisionFrame gocv.Mat
	affineMatrix    gocv.Mat
	boxMask         gocv.Mat
}

type Input struct {
	Img           gocv.Mat
	FaceLandmark5 []gocv.Point2f
}

type Output struct {
	CropVisionFrame gocv.Mat
	AffineMatrix    gocv.Mat
	CropMask        gocv.Mat
}

type TModel = model.Model[*Input, *Output]

var _ TModel = &Model{}

func NewFactory() func() TModel {
	return func() TModel {
		return New()
	}
}

func New() *Model {
	return &Model{}
}

func (m *Model) ModelName() string {
	return "face_occluder"
}

func (m *Model) ModelVersion() string {
	return "1"
}
