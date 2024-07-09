package main

import (
	"fmt"
	"log"

	"github.com/dev6699/face/client"
	"github.com/dev6699/face/model"
	"github.com/dev6699/face/model/arcface"
	"github.com/dev6699/face/model/yoloface"
	"gocv.io/x/gocv"
)

func main() {
	faceDetectorScore := float32(0.5)
	iouThreshold := 0.4
	yolofaceFactory := yoloface.NewFactory(faceDetectorScore, iouThreshold)
	arcfaceFactory := arcface.NewFactory()
	err := client.Init(
		"tritonserver:8001",
		[]model.ModelMeta{
			yolofaceFactory(),
			arcfaceFactory(),
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

	faceEmbeddings := [][]float32{}
	for _, d := range yoloFaceOutput.Detections {
		arcfaceOutput, err := client.Infer(arcfaceFactory, &arcface.Input{Img: img, FaceLandmark5: d.FaceLandmark5})
		if err != nil {
			log.Fatal(err)
		}
		faceEmbeddings = append(faceEmbeddings, arcfaceOutput.NormedEmbedding)
	}

	similarDistance := 0.6
	for i, f1 := range faceEmbeddings {
		for j, f2 := range faceEmbeddings {
			faceDistance := model.CalcFaceDistance(f1, f2)
			isSimilar := faceDistance < similarDistance
			fmt.Printf("Face %d & %d: %.2f %v\n", i, j, faceDistance, isSimilar)
		}
	}
}
