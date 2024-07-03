package yoloface

import (
	"math"
	"sort"

	"github.com/dev6699/face/model"
	"gocv.io/x/gocv"
)

type Detection struct {
	BoundingBox   model.BoundingBox
	FaceLandmark5 []gocv.Point2f
	Confidence    float32
}

func (m *Model) PostProcess(rawOutputContents [][]byte) (*Output, error) {
	// outputs": [
	// 	{
	// 	"name": "output0",
	// 	"datatype": "FP32",
	// 	"shape": [
	// 		1,
	// 		20,
	// 		8400
	// 	]
	// 	}
	// ]
	outputCount := 8400
	rawDetections, err := model.BytesToFloat32Slice(rawOutputContents[0])
	if err != nil {
		return nil, err
	}
	ratioWidth := m.ratioWidth
	ratioHeight := m.ratioHeight

	var detections []Detection

	boundingBoxRaw := rawDetections[:4*outputCount]
	scoreRaw := rawDetections[4*outputCount : 5*outputCount]
	faceLandmark5Raw := rawDetections[5*outputCount:]

	for i := 0; i < outputCount; i++ {
		score := scoreRaw[i]
		if score < m.faceDetectorScore {
			continue
		}

		d := Detection{
			Confidence: score,
		}

		bboxRaw := []float32{
			boundingBoxRaw[i],
			boundingBoxRaw[i+outputCount],
			boundingBoxRaw[i+outputCount*2],
			boundingBoxRaw[i+outputCount*3],
		}

		d.BoundingBox = model.BoundingBox{
			X1: float64(bboxRaw[0]-bboxRaw[2]/2) * float64(ratioWidth),
			Y1: float64(bboxRaw[1]-bboxRaw[3]/2) * float64(ratioHeight),
			X2: float64(bboxRaw[0]+bboxRaw[2]/2) * float64(ratioWidth),
			Y2: float64(bboxRaw[1]+bboxRaw[3]/2) * float64(ratioHeight),
		}

		faceLandmark5Extract := []float32{}
		for j := 0; j < 15; j++ {
			if (j-2)%3 == 0 {
				continue
			}

			idx := j*outputCount + i
			fl := faceLandmark5Raw[idx]
			if j%3 == 0 {
				fl *= ratioWidth
			}
			if (j-1)%3 == 0 {
				fl *= ratioHeight
			}

			faceLandmark5Extract = append(faceLandmark5Extract, fl)
		}

		faceLandmark5 := []gocv.Point2f{}
		for j := 0; j < len(faceLandmark5Extract); j += 2 {
			faceLandmark5 = append(faceLandmark5,
				gocv.Point2f{
					X: faceLandmark5Extract[j],
					Y: faceLandmark5Extract[j+1],
				})
		}
		d.FaceLandmark5 = faceLandmark5
		detections = append(detections, d)
	}

	keepIndices := applyNMS(detections, m.iouThreshold)
	keepDetections := make([]Detection, len(keepIndices))
	for i, idx := range keepIndices {
		keepDetections[i] = detections[idx]
	}

	sort.Slice(keepDetections, func(i, j int) bool {
		return keepDetections[i].Confidence > keepDetections[j].Confidence
	})

	return &Output{
		Detections: keepDetections,
	}, nil
}

// applyNMS performs non-maximum suppression to eliminate duplicate detections.
func applyNMS(detections []Detection, iouThreshold float64) []int {
	boundingBoxList := []model.BoundingBox{}
	for _, d := range detections {
		boundingBoxList = append(boundingBoxList, d.BoundingBox)
	}

	var keepIndices []int
	indices := make([]int, len(boundingBoxList))
	for i := range boundingBoxList {
		indices[i] = i
	}

	areas := make([]float64, len(boundingBoxList))
	for i, box := range boundingBoxList {
		areas[i] = (box.X2 - box.X1 + 1) * (box.Y2 - box.Y1 + 1)
	}

	for len(indices) > 0 {
		index := indices[0]
		keepIndices = append(keepIndices, index)
		var remainIndices []int

		for _, i := range indices[1:] {
			xx1 := math.Max(boundingBoxList[index].X1, boundingBoxList[i].X1)
			yy1 := math.Max(boundingBoxList[index].Y1, boundingBoxList[i].Y1)
			xx2 := math.Min(boundingBoxList[index].X2, boundingBoxList[i].X2)
			yy2 := math.Min(boundingBoxList[index].Y2, boundingBoxList[i].Y2)

			width := math.Max(0, xx2-xx1+1)
			height := math.Max(0, yy2-yy1+1)
			intersection := width * height
			union := areas[index] + areas[i] - intersection
			iou := intersection / union

			if iou <= iouThreshold {
				remainIndices = append(remainIndices, i)
			}
		}
		indices = remainIndices
	}

	return keepIndices
}
