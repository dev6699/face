package main

import (
	"fmt"
	"log"

	"github.com/dev6699/face/client"
	"github.com/dev6699/face/examples"
	"github.com/dev6699/face/model"
	"github.com/dev6699/face/model/genderage"
	"github.com/dev6699/face/model/yoloface"
	"gocv.io/x/gocv"
)

func main() {
	faceDetectorScore := float32(0.5)
	iouThreshold := 0.4
	yolofaceFactory := yoloface.NewFactory(faceDetectorScore, iouThreshold)
	genderAgeFactory := genderage.NewFactory()
	err := client.Init(
		"tritonserver:8001",
		[]model.ModelMeta{
			yolofaceFactory(),
			genderAgeFactory(),
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
		genderAgeOutput, err := client.Infer(genderAgeFactory, &genderage.Input{Img: img, BoundingBox: d.BoundingBox})
		if err != nil {
			log.Fatal(err)
		}

		genderString := "M"
		if genderAgeOutput.Gender == 0 {
			genderString = "F"
		}
		examples.DrawBoundingBoxes(&img, d.BoundingBox, fmt.Sprintf("%s %d", genderString, genderAgeOutput.Age), examples.Green, examples.Red)
	}

	gocv.IMWrite("output.jpg", img)
}
