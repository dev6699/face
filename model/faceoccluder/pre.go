package faceoccluder

import (
	"image"

	"github.com/dev6699/face/model"
	"github.com/dev6699/face/protobuf"
	"gocv.io/x/gocv"
)

func (m *Model) PreProcess(i *Input) ([]*protobuf.InferTensorContents, error) {
	cropSize := model.Size{Width: 128, Height: 128}
	m.cropSize = cropSize
	cropVisionFrame, affineMatrix := model.WarpFaceByFaceLandmark5(i.Img, i.FaceLandmark5, arcface_128_v2, cropSize)
	m.cropVisionFrame = cropVisionFrame
	m.affineMatrix = affineMatrix

	boxMask := model.CreateStaticBoxMask(cropSize, 0.3, model.Padding{Top: 0, Right: 0, Bottom: 0, Left: 0})
	m.boxMask = boxMask

	resizedFrame := gocv.NewMat()
	defer resizedFrame.Close()
	gocv.Resize(cropVisionFrame, &resizedFrame, image.Point{X: 256, Y: 256}, 0, 0, gocv.InterpolationDefault)
	data, _ := resizedFrame.DataPtrUint8()

	d := make([]float32, len(data))
	for i, a := range data {
		d[i] = float32(a) / 255.0
	}

	contents := &protobuf.InferTensorContents{
		Fp32Contents: d,
	}
	return []*protobuf.InferTensorContents{contents}, nil
}
