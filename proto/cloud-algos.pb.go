// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.27.1-devel
// 	protoc        v3.19.1
// source: proto/cloud-algos.proto

package proto

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

var File_proto_cloud_algos_proto protoreflect.FileDescriptor

var file_proto_cloud_algos_proto_rawDesc = []byte{
	0x0a, 0x17, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x2f, 0x63, 0x6c, 0x6f, 0x75, 0x64, 0x2d, 0x61, 0x6c,
	0x67, 0x6f, 0x73, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x05, 0x70, 0x72, 0x6f, 0x74, 0x6f,
	0x1a, 0x1c, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x2f, 0x63, 0x6f, 0x6d, 0x70, 0x75, 0x74, 0x61, 0x74,
	0x69, 0x6f, 0x6e, 0x2d, 0x6d, 0x73, 0x67, 0x73, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x32, 0x47,
	0x0a, 0x09, 0x43, 0x6c, 0x6f, 0x75, 0x64, 0x41, 0x6c, 0x67, 0x6f, 0x12, 0x3a, 0x0a, 0x07, 0x43,
	0x6f, 0x6d, 0x70, 0x75, 0x74, 0x65, 0x12, 0x15, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x2e, 0x43,
	0x6f, 0x6d, 0x70, 0x75, 0x74, 0x65, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x16, 0x2e,
	0x70, 0x72, 0x6f, 0x74, 0x6f, 0x2e, 0x43, 0x6f, 0x6d, 0x70, 0x75, 0x74, 0x65, 0x52, 0x65, 0x73,
	0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x00, 0x42, 0x09, 0x5a, 0x07, 0x2e, 0x3b, 0x70, 0x72, 0x6f,
	0x74, 0x6f, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var file_proto_cloud_algos_proto_goTypes = []interface{}{
	(*ComputeRequest)(nil),  // 0: proto.ComputeRequest
	(*ComputeResponse)(nil), // 1: proto.ComputeResponse
}
var file_proto_cloud_algos_proto_depIdxs = []int32{
	0, // 0: proto.CloudAlgo.Compute:input_type -> proto.ComputeRequest
	1, // 1: proto.CloudAlgo.Compute:output_type -> proto.ComputeResponse
	1, // [1:2] is the sub-list for method output_type
	0, // [0:1] is the sub-list for method input_type
	0, // [0:0] is the sub-list for extension type_name
	0, // [0:0] is the sub-list for extension extendee
	0, // [0:0] is the sub-list for field type_name
}

func init() { file_proto_cloud_algos_proto_init() }
func file_proto_cloud_algos_proto_init() {
	if File_proto_cloud_algos_proto != nil {
		return
	}
	file_proto_computation_msgs_proto_init()
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_proto_cloud_algos_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   0,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_proto_cloud_algos_proto_goTypes,
		DependencyIndexes: file_proto_cloud_algos_proto_depIdxs,
	}.Build()
	File_proto_cloud_algos_proto = out.File
	file_proto_cloud_algos_proto_rawDesc = nil
	file_proto_cloud_algos_proto_goTypes = nil
	file_proto_cloud_algos_proto_depIdxs = nil
}

// Reference imports to suppress errors if they are not otherwise used.
var _ context.Context
var _ grpc.ClientConnInterface

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
const _ = grpc.SupportPackageIsVersion6

// CloudAlgoClient is the client API for CloudAlgo service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://godoc.org/google.golang.org/grpc#ClientConn.NewStream.
type CloudAlgoClient interface {
	// Performs the appropriate computation at the host algo and returns the result.
	Compute(ctx context.Context, in *ComputeRequest, opts ...grpc.CallOption) (*ComputeResponse, error)
}

type cloudAlgoClient struct {
	cc grpc.ClientConnInterface
}

func NewCloudAlgoClient(cc grpc.ClientConnInterface) CloudAlgoClient {
	return &cloudAlgoClient{cc}
}

func (c *cloudAlgoClient) Compute(ctx context.Context, in *ComputeRequest, opts ...grpc.CallOption) (*ComputeResponse, error) {
	out := new(ComputeResponse)
	err := c.cc.Invoke(ctx, "/proto.CloudAlgo/Compute", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// CloudAlgoServer is the server API for CloudAlgo service.
type CloudAlgoServer interface {
	// Performs the appropriate computation at the host algo and returns the result.
	Compute(context.Context, *ComputeRequest) (*ComputeResponse, error)
}

// UnimplementedCloudAlgoServer can be embedded to have forward compatible implementations.
type UnimplementedCloudAlgoServer struct {
}

func (*UnimplementedCloudAlgoServer) Compute(context.Context, *ComputeRequest) (*ComputeResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Compute not implemented")
}

func RegisterCloudAlgoServer(s *grpc.Server, srv CloudAlgoServer) {
	s.RegisterService(&_CloudAlgo_serviceDesc, srv)
}

func _CloudAlgo_Compute_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ComputeRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(CloudAlgoServer).Compute(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/proto.CloudAlgo/Compute",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(CloudAlgoServer).Compute(ctx, req.(*ComputeRequest))
	}
	return interceptor(ctx, in, info, handler)
}

var _CloudAlgo_serviceDesc = grpc.ServiceDesc{
	ServiceName: "proto.CloudAlgo",
	HandlerType: (*CloudAlgoServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "Compute",
			Handler:    _CloudAlgo_Compute_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "proto/cloud-algos.proto",
}
