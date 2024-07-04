package arcface

import (
	"math"

	"github.com/dev6699/face/model"
)

func (m *Model) PostProcess(rawOutputContents [][]byte) (*Output, error) {
	// "outputs": [
	// 	{
	// 		"name": "683",
	// 		"datatype": "FP32",
	// 		"shape": [
	// 			1,
	// 			512
	// 		]
	// 	}
	// ]
	embedding, err := model.BytesToFloat32Slice(rawOutputContents[0])
	if err != nil {
		return nil, err
	}

	normedEmbedding := normalizeEmbedding(embedding)

	return &Output{
		Embedding:       embedding,
		NormedEmbedding: normedEmbedding,
	}, nil
}

// Normalize the embedding
func normalizeEmbedding(embedding []float32) []float32 {
	normalizeEmbedding := make([]float32, len(embedding))
	norm := float32(l2Norm(embedding))
	if norm == 0 {
		normalizeEmbedding = append(normalizeEmbedding, embedding...)
		return normalizeEmbedding // Avoid division by zero
	}

	for i, val := range embedding {
		normalizeEmbedding[i] = (val / norm)
	}
	return normalizeEmbedding
}

// Compute the L2 norm of a vector
func l2Norm(vector []float32) float64 {
	var sum float64
	for _, val := range vector {
		sum += float64(val * val)
	}
	return math.Sqrt(sum)
}
