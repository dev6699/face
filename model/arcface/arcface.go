package arcface

import (
	"github.com/dev6699/face/model"
	"gocv.io/x/gocv"
)

type Model struct{}

type Input struct {
	Img           gocv.Mat
	FaceLandmark5 []gocv.Point2f
}

type Output struct {
	Embedding       []float32
	NormedEmbedding []float32
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
	return "arcface_w600k_r50"
}

func (m *Model) ModelVersion() string {
	return "1"
}
