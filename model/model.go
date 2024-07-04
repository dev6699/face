package model

import (
	"github.com/dev6699/face/protobuf"
)

type Model[I any, O any] interface {
	ModelMeta
	PreProcess(input I) ([]*protobuf.InferTensorContents, error)
	PostProcess(rawOutputContents [][]byte) (O, error)
}

type ModelMeta interface {
	ModelName() string
	ModelVersion() string
}

type BoundingBox struct {
	X1, Y1, X2, Y2 float64
}
