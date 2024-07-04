package genderage

import (
	"github.com/dev6699/face/model"
	"gocv.io/x/gocv"
)

type Model struct{}

type Input struct {
	Img         gocv.Mat
	BoundingBox model.BoundingBox
}

type Output struct {
	Age int
	// Gender: female=0, male=1
	Gender int
}

type ModelT = model.Model[*Input, *Output]

var _ ModelT = &Model{}

func NewFactory() func() ModelT {
	return func() ModelT {
		return New()
	}
}

func New() *Model {
	return &Model{}
}

func (m *Model) ModelName() string {
	return "gender_age"
}

func (m *Model) ModelVersion() string {
	return "1"
}
