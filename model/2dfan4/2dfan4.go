package _2dfan4

import (
	"github.com/dev6699/face/model"
	"gocv.io/x/gocv"
)

type Model struct {
	affineMatrix gocv.Mat
}

type Input struct {
	Img         gocv.Mat
	BoundingBox model.BoundingBox
}

type Output struct {
	FaceLandmark68      FaceLandmark68
	FaceLandmark68Score float64
}

type FaceLandmark68 struct {
	Data []gocv.Point2f
}

func (f *FaceLandmark68) ToLandmark5() []gocv.Point2f {
	return []gocv.Point2f{
		model.CalculateMean2f(f.Data[36:42]),
		model.CalculateMean2f(f.Data[42:48]),
		f.Data[30],
		f.Data[48],
		f.Data[54],
	}
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
	return "2dfan4"
}

func (m *Model) ModelVersion() string {
	return "1"
}
