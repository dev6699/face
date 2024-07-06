package inswapper

import (
	"github.com/dev6699/face/model"
	"gocv.io/x/gocv"
)

type Model struct {
	affineMatrix gocv.Mat
}

type Input struct {
	CropVisionFrame gocv.Mat
	Embedding       []float32
}

type Output struct {
	CropVisionFrame gocv.Mat
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
	return "inswapper_128_fp16"
}

func (m *Model) ModelVersion() string {
	return "1"
}
