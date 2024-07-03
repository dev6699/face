package client

import (
	"context"
	"time"

	"github.com/dev6699/face/protobuf"
)

var requestTimeout = 10 * time.Second

func ServerLiveRequest(client protobuf.GRPCInferenceServiceClient) (*protobuf.ServerLiveResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), requestTimeout)
	defer cancel()

	serverLiveRequest := protobuf.ServerLiveRequest{}
	serverLiveResponse, err := client.ServerLive(ctx, &serverLiveRequest)
	if err != nil {
		return nil, err
	}
	return serverLiveResponse, nil
}

func ServerReadyRequest(client protobuf.GRPCInferenceServiceClient) (*protobuf.ServerReadyResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), requestTimeout)
	defer cancel()

	serverReadyRequest := protobuf.ServerReadyRequest{}
	serverReadyResponse, err := client.ServerReady(ctx, &serverReadyRequest)
	if err != nil {
		return nil, err
	}
	return serverReadyResponse, nil
}

func ModelMetadataRequest(client protobuf.GRPCInferenceServiceClient, modelName string, modelVersion string) (*protobuf.ModelMetadataResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), requestTimeout)
	defer cancel()

	modelMetadataRequest := protobuf.ModelMetadataRequest{
		Name:    modelName,
		Version: modelVersion,
	}
	modelMetadataResponse, err := client.ModelMetadata(ctx, &modelMetadataRequest)
	if err != nil {
		return nil, err
	}
	return modelMetadataResponse, nil
}

func ModelInferRequest(client protobuf.GRPCInferenceServiceClient, modelInferRequest *protobuf.ModelInferRequest) (*protobuf.ModelInferResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), requestTimeout)
	defer cancel()

	modelInferResponse, err := client.ModelInfer(ctx, modelInferRequest)
	if err != nil {
		return nil, err
	}

	return modelInferResponse, nil
}
