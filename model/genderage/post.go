package genderage

import (
	"math"

	"github.com/dev6699/face/model"
)

func (m *Model) PostProcess(rawOutputContents [][]byte) (*Output, error) {
	// "outputs": [
	// 	{
	// 		"name": "fc1",
	// 		"datatype": "FP32",
	// 		"shape": [
	// 			1,
	// 			3
	// 		]
	// 	}
	// ]
	prediction, err := model.BytesToFloat32Slice(rawOutputContents[0])
	if err != nil {
		return nil, err
	}

	gender := model.Argmax(prediction[:2])
	age := int(math.Round(float64(prediction[2] * 100)))

	return &Output{
		Gender: gender,
		Age:    age,
	}, nil
}
