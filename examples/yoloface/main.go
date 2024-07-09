package main

import (
	"fmt"
	"log"

	"github.com/dev6699/face/client"
	"github.com/dev6699/face/examples"
	"github.com/dev6699/face/model"
	"github.com/dev6699/face/model/yoloface"
	"gocv.io/x/gocv"
)

func main() {
	faceDetectorScore := float32(0.5)
	iouThreshold := 0.4
	yolofaceFactory := yoloface.NewFactory(faceDetectorScore, iouThreshold)
	err := client.Init(
		"tritonserver:8001",
		[]model.ModelMeta{
			yolofaceFactory(),
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
		examples.DrawBoundingBoxes(&img, d.BoundingBox, fmt.Sprintf("Score: %.2f", d.Confidence), examples.Green, examples.Green)
		examples.DrawPoints(&img, d.FaceLandmark5, examples.Red, 3)
	}

	gocv.IMWrite("output.jpg", img)
}
