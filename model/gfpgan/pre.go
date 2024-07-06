package gfpgan

import (
	"github.com/dev6699/face/model"
	"github.com/dev6699/face/protobuf"
	"gocv.io/x/gocv"
)

func (m *Model) PreProcess(i *Input) ([]*protobuf.InferTensorContents, error) {
	m.img = i.Img
	cropSize := model.Size{Width: 512, Height: 512}
	m.cropSize = cropSize

	cropVisionFrame, affineMatrix := model.WarpFaceByFaceLandmark5(i.Img, i.FaceLandmark5, ffhq_512, cropSize)
	m.affineMatrix = affineMatrix
	defer cropVisionFrame.Close()

	d := []float32{}
	cropVisionFrame.ConvertTo(&cropVisionFrame, gocv.MatTypeCV32F)
	cropVisionFrame.DivideFloat(255.0)
	model.MatSubtract(cropVisionFrame, 0.5)
	cropVisionFrame.DivideFloat(0.5)

	rgbChannels := gocv.Split(cropVisionFrame)
	b := rgbChannels[2]
	defer b.Close()
	bd, _ := b.DataPtrFloat32()
	d = append(d, bd...)

	g := rgbChannels[1]
	defer g.Close()
	gd, _ := g.DataPtrFloat32()
	d = append(d, gd...)

	r := rgbChannels[0]
	defer r.Close()
	rd, _ := r.DataPtrFloat32()
	d = append(d, rd...)

	contents := &protobuf.InferTensorContents{
		Fp32Contents: d,
	}
	return []*protobuf.InferTensorContents{contents}, nil
}
