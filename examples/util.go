package examples

import (
	"image"
	"image/color"

	"github.com/dev6699/face/model"
	"gocv.io/x/gocv"
)

var (
	Red   = color.RGBA{R: 255, G: 0, B: 0, A: 255}
	Green = color.RGBA{R: 0, G: 255, B: 0, A: 255}
	Blue  = color.RGBA{R: 0, G: 0, B: 255, A: 255}
)

func DrawBoundingBoxes(img *gocv.Mat, box model.BoundingBox, text string, boxColor, textColor color.RGBA) {
	gocv.Rectangle(img, image.Rectangle{Min: image.Point{X: int(box.X1), Y: int(box.Y1)}, Max: image.Point{X: int(box.X2), Y: int(box.Y2)}}, boxColor, 2)
	gocv.PutText(img, text, image.Point{X: int(box.X1), Y: int(box.Y1) - 5}, gocv.FontHersheySimplex, 0.5, textColor, 2)
}

func DrawPoints(img *gocv.Mat, points []gocv.Point2f, col color.RGBA, radius int) {
	for _, pt := range points {
		gocv.Circle(img, image.Pt(int(pt.X), int(pt.Y)), radius, col, -1)
	}
}
