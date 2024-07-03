package client

import (
	"github.com/dev6699/face/model"
	"github.com/dev6699/face/protobuf"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

var (
	conn           *grpc.ClientConn
	client         protobuf.GRPCInferenceServiceClient
	modelsMetadata = make(map[string]*modelMetadata)
)

// Init initializes grpc connection and fetch all models metadata from grpc server.
func Init(url string, models []model.ModelMeta) error {
	var err error
	conn, err = grpc.NewClient(url, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return err
	}

	client = protobuf.NewGRPCInferenceServiceClient(conn)
	for _, m := range models {

		meta, err := newModelMetadata(client, m.ModelName(), m.ModelVersion())
		if err != nil {
			return err
		}

		modelsMetadata[m.ModelName()] = meta
	}
	return nil
}

// Close tears down underlying grpc connection.
func Close() error {
	return conn.Close()
}

// Infer is a generic function takes in modelFactory to create model.Model and input for model.PreProcess(),
// and performs infer request based on model metadata automatically.
func Infer[I, O any](modelFactory func() model.Model[I, O], input I) (O, error) {
	var zeroOutput O
	model := modelFactory()
	contents, err := model.PreProcess(input)
	if err != nil {
		return zeroOutput, err
	}

	modelInferRequest := modelsMetadata[model.ModelName()].formInferRequest(contents)

	inferResponse, err := ModelInferRequest(client, modelInferRequest)
	if err != nil {
		return zeroOutput, err
	}

	return model.PostProcess(inferResponse.RawOutputContents)
}

type modelMetadata struct {
	modelName    string
	modelVersion string
	*protobuf.ModelMetadataResponse
}

func newModelMetadata(client protobuf.GRPCInferenceServiceClient, modelName string, modelVersion string) (*modelMetadata, error) {
	metaResponse, err := ModelMetadataRequest(client, modelName, modelVersion)
	if err != nil {
		return nil, err
	}

	return &modelMetadata{
		modelName:             modelName,
		modelVersion:          modelVersion,
		ModelMetadataResponse: metaResponse,
	}, nil
}

func (m *modelMetadata) formInferRequest(contents []*protobuf.InferTensorContents) *protobuf.ModelInferRequest {

	inputs := []*protobuf.ModelInferRequest_InferInputTensor{}
	for i, c := range contents {
		input := m.Inputs[i]
		shape := input.Shape
		if shape[0] == -1 {
			shape[0] = 1
		}
		inputs = append(inputs, &protobuf.ModelInferRequest_InferInputTensor{
			Name:     input.Name,
			Datatype: input.Datatype,
			Shape:    shape,
			Contents: c,
		})
	}

	outputs := make([]*protobuf.ModelInferRequest_InferRequestedOutputTensor, len(m.Outputs))
	for i, o := range m.Outputs {
		outputs[i] = &protobuf.ModelInferRequest_InferRequestedOutputTensor{
			Name: o.Name,
		}
	}

	return &protobuf.ModelInferRequest{
		ModelName:    m.modelName,
		ModelVersion: m.modelVersion,
		Inputs:       inputs,
		Outputs:      outputs,
	}
}
