package genderage

import (
	"image"
	"math"

	"github.com/dev6699/face/model"
	"github.com/dev6699/face/protobuf"
)

func (m *Model) PreProcess(i *Input) ([]*protobuf.InferTensorContents, error) {
	boundingBox := i.BoundingBox

	scale := 64.0 / max(math.Abs(boundingBox.X2-boundingBox.X1), math.Abs(boundingBox.Y2-boundingBox.Y1))
	translation := model.Translation{
		48.0 - (boundingBox.X1+boundingBox.X2)*scale*0.5,
		48.0 - (boundingBox.Y1+boundingBox.Y2)*scale*0.5,
	}

	cropVisionFrame, affineMatrix := model.WarpFaceByTranslation(i.Img, translation, scale, image.Point{X: 96, Y: 96})
	defer cropVisionFrame.Close()
	defer affineMatrix.Close()

	di, _ := cropVisionFrame.DataPtrUint8()
	d := make([]float32, len(di))

	idx := 0
	for i := 0; i < len(di); i += 3 {
		d[idx] = float32(di[i+2])
		d[cropVisionFrame.Cols()*cropVisionFrame.Rows()+idx] = float32(di[i+1])
		d[2*cropVisionFrame.Cols()*cropVisionFrame.Rows()+idx] = float32(di[i])
		idx++
	}

	contents := &protobuf.InferTensorContents{
		Fp32Contents: d,
	}
	return []*protobuf.InferTensorContents{contents}, nil
}
