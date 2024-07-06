package inswapper

import (
	"math"

	"github.com/dev6699/face/protobuf"
)

func (m *Model) PreProcess(i *Input) ([]*protobuf.InferTensorContents, error) {
	data, _ := i.CropVisionFrame.DataPtrUint8()

	d := make([]float32, len(data))
	j := 0
	for i := 0; i < len(data); i += 3 {
		d[j] = float32(data[i+2]) / 255.0
		d[len(d)/3+j] = float32(data[i+1]) / 255.0
		d[len(d)/3*2+j] = float32(data[i]) / 255.0
		j++
	}

	sourceEmbedding := prepareSourceEmbedding(i.Embedding)
	return []*protobuf.InferTensorContents{
		{
			Fp32Contents: sourceEmbedding,
		},
		{
			Fp32Contents: d,
		},
	}, nil
}

func prepareSourceEmbedding(embedding []float32) []float32 {
	sourceEmbedding := embedding
	dot := matrixMultiply([][]float32{sourceEmbedding}, initializer)
	normSource := norm(sourceEmbedding)
	normalized := scale(dot[0], 1/normSource)

	return normalized
}

func matrixMultiply(A, B [][]float32) [][]float32 {
	m := len(A)
	n := len(A[0])
	p := len(B[0])

	C := make([][]float32, m)
	for i := range C {
		C[i] = make([]float32, p)
	}

	// Perform matrix multiplication
	for i := 0; i < m; i++ {
		for k := 0; k < p; k++ {
			sum := float32(0.0)
			for j := 0; j < n; j++ {
				sum += A[i][j] * B[j][k]
			}
			C[i][k] = sum
		}
	}

	return C
}

func norm(vector []float32) float32 {
	var norm float32
	for _, v := range vector {
		norm += v * v
	}
	return float32(math.Sqrt(float64(norm)))
}

func scale(vector []float32, scalar float32) []float32 {
	result := make([]float32, len(vector))
	for i, v := range vector {
		result[i] = v * scalar
	}
	return result
}
