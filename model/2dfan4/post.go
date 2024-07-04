package _2dfan4

import (
	"github.com/dev6699/face/model"
	"gocv.io/x/gocv"
)

func (m *Model) PostProcess(rawOutputContents [][]byte) (*Output, error) {
	// "outputs": [
	// 	{
	// 		"name": "heatmaps",
	// 		"datatype": "FP32",
	// 		"shape": [
	// 			1,
	// 			68,
	// 			64,
	// 			64
	// 		]
	// 	},
	// 	{
	// 		"name": "landmarks_xyscore",
	// 		"datatype": "FP32",
	// 		"shape": [
	// 			1,
	// 			68,
	// 			3
	// 		]
	// 	}
	// ]

	heatmaps, err := model.BytesToFloat32Slice(rawOutputContents[0])
	if err != nil {
		return nil, err
	}

	landmarkRaw, err := model.BytesToFloat32Slice(rawOutputContents[1])
	if err != nil {
		return nil, err
	}

	landmark := processLandmark(landmarkRaw, 64.0, 256.0)
	invertAffineMatrix := model.InvertAffineMatrix(m.affineMatrix)
	defer m.affineMatrix.Close()
	defer invertAffineMatrix.Close()

	faceLandmark68 := transformLandmark68(landmark, invertAffineMatrix)
	faceLandmark68Score := calculateHeatmapScores(heatmaps)

	return &Output{
		FaceLandmark68: FaceLandmark68{
			Data: faceLandmark68,
		},
		FaceLandmark68Score: faceLandmark68Score,
	}, nil
}

func processLandmark(landmark []float32, normalizationFactor, scale float64) [][]float32 {
	numLandmark := len(landmark) / 3
	scaledLandmark := make([][]float32, numLandmark)

	for i := 0; i < numLandmark; i++ {
		normalizedX := float64(landmark[3*i]) / normalizationFactor
		normalizedY := float64(landmark[3*i+1]) / normalizationFactor

		scaledLandmark[i] = []float32{float32(normalizedX * scale), float32(normalizedY * scale)}
	}
	return scaledLandmark
}

// transformLandmark68 applies an affine transformation to the landmark.
func transformLandmark68(landmark [][]float32, affineMatrix gocv.Mat) []gocv.Point2f {
	input := gocv.NewMatWithSize(len(landmark), 1, gocv.MatTypeCV32FC2)
	defer input.Close()

	for i, landmark := range landmark {
		input.SetFloatAt(i, 0, landmark[0])
		input.SetFloatAt(i, 1, landmark[1])
	}

	output := gocv.NewMat()
	defer output.Close()

	gocv.Transform(input, &output, affineMatrix)

	transformedLandmark := make([]gocv.Point2f, len(landmark))
	for i := range transformedLandmark {
		transformedLandmark[i] = gocv.Point2f{
			X: output.GetFloatAt(i, 0),
			Y: output.GetFloatAt(i, 1),
		}
	}
	return transformedLandmark
}

// calculateHeatmapScores calculates the scores from the heatmap and returns the mean value.
func calculateHeatmapScores(heatmap []float32) float64 {
	scores := make([]float64, 68)

	for c := 0; c < 68; c++ {
		maxVal := 0.0
		for y := 0; y < 64; y++ {
			for x := 0; x < 64; x++ {
				idx := (c * 64 * 64) + (64 * y) + x
				value := float64(heatmap[idx])
				if value > maxVal {
					maxVal = value
				}
			}
		}
		scores[c] = maxVal
	}

	meanScore := 0.0
	for _, score := range scores {
		meanScore += score
	}
	return meanScore / float64(len(scores))
}
