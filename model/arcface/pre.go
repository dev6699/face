package arcface

import (
	"github.com/dev6699/face/model"
	"github.com/dev6699/face/protobuf"
	"gocv.io/x/gocv"
)

func (m *Model) PreProcess(i *Input) ([]*protobuf.InferTensorContents, error) {
	cropVisionFrame, affineMatrix := model.WarpFaceByFaceLandmark5(
		i.Img,
		i.FaceLandmark5,
		arcface_112_v2,
		model.Size{Width: 112, Height: 112},
	)
	defer cropVisionFrame.Close()
	defer affineMatrix.Close()

	d := []float32{}
	cropVisionFrame.ConvertTo(&cropVisionFrame, gocv.MatTypeCV32F)
	cropVisionFrame.DivideFloat(127.5)
	model.MatSubtract(cropVisionFrame, 1.0)

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
