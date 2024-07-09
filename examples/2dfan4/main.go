package main

import (
	"log"

	"github.com/dev6699/face/client"
	"github.com/dev6699/face/examples"
	"github.com/dev6699/face/model"
	_2dfan4 "github.com/dev6699/face/model/2dfan4"
	"github.com/dev6699/face/model/yoloface"
	"gocv.io/x/gocv"
)

func main() {
	faceDetectorScore := float32(0.5)
	iouThreshold := 0.4
	yolofaceFactory := yoloface.NewFactory(faceDetectorScore, iouThreshold)
	_2dfan4Factory := _2dfan4.NewFactory()
	err := client.Init(
		"tritonserver:8001",
		[]model.ModelMeta{
			yolofaceFactory(),
			_2dfan4Factory(),
		},
	)
	if err != nil {
		log.Fatal(err)
	}

	img := gocv.IMRead("../image.jpg", gocv.IMReadColor)
	yoloFaceOutput, err := client.Infer(yolofaceFactory, &yoloface.Input{Img: img})
	if err != nil {
		log.Fatal(err)
	}

	for _, d := range yoloFaceOutput.Detections {
		fanOutput, err := client.Infer(_2dfan4Factory, &_2dfan4.Input{Img: img, BoundingBox: d.BoundingBox})
		if err != nil {
			log.Fatal(err)
		}

		examples.DrawPoints(&img, fanOutput.FaceLandmark68.Data, examples.Red, 2)
	}

	gocv.IMWrite("output.jpg", img)
}
