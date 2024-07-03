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
