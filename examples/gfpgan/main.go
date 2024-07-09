package main

import (
	"fmt"
	"log"

	"github.com/dev6699/face/client"
	"github.com/dev6699/face/model"
	"github.com/dev6699/face/model/gfpgan"
	"github.com/dev6699/face/model/yoloface"
	"gocv.io/x/gocv"
)

func main() {
	faceDetectorScore := float32(0.5)
	iouThreshold := 0.4
	yolofaceFactory := yoloface.NewFactory(faceDetectorScore, iouThreshold)
	gfpganFactory := gfpgan.NewFactory(80.0)
	err := client.Init(
		"tritonserver:8001",
		[]model.ModelMeta{
			yolofaceFactory(),
			gfpganFactory(),
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

	for i, d := range yoloFaceOutput.Detections {
		gfpganOutput, err := client.Infer(gfpganFactory, &gfpgan.Input{Img: img, FaceLandmark5: d.FaceLandmark5})
		if err != nil {
			log.Fatal(err)
		}
		gocv.IMWrite(fmt.Sprintf("output%d.jpg", i+1), gfpganOutput.OutFrame)
	}
}
