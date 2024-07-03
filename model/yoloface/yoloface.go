package yoloface

import (
	"github.com/dev6699/face/model"
	"gocv.io/x/gocv"
)

type Model struct {
	faceDetectorScore float32
	iouThreshold      float64
	ratioHeight       float32
	ratioWidth        float32
}

type Input struct {
	Img gocv.Mat
}

type Output struct {
	Detections []Detection
}

type ModelT = model.Model[*Input, *Output]

var _ ModelT = &Model{}

func NewFactory(faceDetectorScore float32, iouThreshold float64) func() ModelT {
	return func() ModelT {
		return New(faceDetectorScore, iouThreshold)
	}
}

func New(faceDetectorScore float32, iouThreshold float64) *Model {
	return &Model{
		faceDetectorScore: faceDetectorScore,
		iouThreshold:      iouThreshold,
	}
}

func (m *Model) ModelName() string {
	return "yoloface"
}

func (m *Model) ModelVersion() string {
	return "1"
}
