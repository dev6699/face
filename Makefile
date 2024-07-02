PROTOC_VERSION=27.2

.PHONY: protoc
protoc:
	wget https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-x86_64.zip
	unzip protoc-${PROTOC_VERSION}-linux-x86_64.zip -d protoc-${PROTOC_VERSION}
	mv protoc-${PROTOC_VERSION}/bin/protoc /usr/local/bin/
	go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
	go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

.PHONY: gen-proto
gen-proto:
	protoc --proto_path=protobuf/proto --go_out=protobuf --go-grpc_out=protobuf --go_opt=paths=source_relative --go-grpc_opt=paths=source_relative protobuf/proto/*.proto