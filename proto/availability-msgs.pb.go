// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.27.1-devel
// 	protoc        v3.19.1
// source: proto/availability-msgs.proto

package proto

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

// Message sent to a site to determine wheter it's available
type SiteAvailableReq struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	SiteId int64 `protobuf:"varint,1,opt,name=site_id,json=siteId,proto3" json:"site_id,omitempty"`
}

func (x *SiteAvailableReq) Reset() {
	*x = SiteAvailableReq{}
	if protoimpl.UnsafeEnabled {
		mi := &file_proto_availability_msgs_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *SiteAvailableReq) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SiteAvailableReq) ProtoMessage() {}

func (x *SiteAvailableReq) ProtoReflect() protoreflect.Message {
	mi := &file_proto_availability_msgs_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use SiteAvailableReq.ProtoReflect.Descriptor instead.
func (*SiteAvailableReq) Descriptor() ([]byte, []int) {
	return file_proto_availability_msgs_proto_rawDescGZIP(), []int{0}
}

func (x *SiteAvailableReq) GetSiteId() int64 {
	if x != nil {
		return x.SiteId
	}
	return 0
}

// Message returned from a site indicating whether it is available
type SiteAvailableRes struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Site *Site `protobuf:"bytes,1,opt,name=site,proto3" json:"site,omitempty"`
}

func (x *SiteAvailableRes) Reset() {
	*x = SiteAvailableRes{}
	if protoimpl.UnsafeEnabled {
		mi := &file_proto_availability_msgs_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *SiteAvailableRes) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SiteAvailableRes) ProtoMessage() {}

func (x *SiteAvailableRes) ProtoReflect() protoreflect.Message {
	mi := &file_proto_availability_msgs_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use SiteAvailableRes.ProtoReflect.Descriptor instead.
func (*SiteAvailableRes) Descriptor() ([]byte, []int) {
	return file_proto_availability_msgs_proto_rawDescGZIP(), []int{1}
}

func (x *SiteAvailableRes) GetSite() *Site {
	if x != nil {
		return x.Site
	}
	return nil
}

// Message sent to find out all the sites that are available at the moment
type SitesAvailableReq struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields
}

func (x *SitesAvailableReq) Reset() {
	*x = SitesAvailableReq{}
	if protoimpl.UnsafeEnabled {
		mi := &file_proto_availability_msgs_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *SitesAvailableReq) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SitesAvailableReq) ProtoMessage() {}

func (x *SitesAvailableReq) ProtoReflect() protoreflect.Message {
	mi := &file_proto_availability_msgs_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use SitesAvailableReq.ProtoReflect.Descriptor instead.
func (*SitesAvailableReq) Descriptor() ([]byte, []int) {
	return file_proto_availability_msgs_proto_rawDescGZIP(), []int{2}
}

// Message returned indicating all the sites registered at the moment and their status
type SitesAvailableRes struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Responses []*SiteAvailableRes `protobuf:"bytes,1,rep,name=responses,proto3" json:"responses,omitempty"`
}

func (x *SitesAvailableRes) Reset() {
	*x = SitesAvailableRes{}
	if protoimpl.UnsafeEnabled {
		mi := &file_proto_availability_msgs_proto_msgTypes[3]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *SitesAvailableRes) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*SitesAvailableRes) ProtoMessage() {}

func (x *SitesAvailableRes) ProtoReflect() protoreflect.Message {
	mi := &file_proto_availability_msgs_proto_msgTypes[3]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use SitesAvailableRes.ProtoReflect.Descriptor instead.
func (*SitesAvailableRes) Descriptor() ([]byte, []int) {
	return file_proto_availability_msgs_proto_rawDescGZIP(), []int{3}
}

func (x *SitesAvailableRes) GetResponses() []*SiteAvailableRes {
	if x != nil {
		return x.Responses
	}
	return nil
}

// Structure holding information on a site
type Site struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	SiteId    int64 `protobuf:"varint,1,opt,name=site_id,json=siteId,proto3" json:"site_id,omitempty"`
	Available bool  `protobuf:"varint,2,opt,name=available,proto3" json:"available,omitempty"`
}

func (x *Site) Reset() {
	*x = Site{}
	if protoimpl.UnsafeEnabled {
		mi := &file_proto_availability_msgs_proto_msgTypes[4]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Site) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Site) ProtoMessage() {}

func (x *Site) ProtoReflect() protoreflect.Message {
	mi := &file_proto_availability_msgs_proto_msgTypes[4]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Site.ProtoReflect.Descriptor instead.
func (*Site) Descriptor() ([]byte, []int) {
	return file_proto_availability_msgs_proto_rawDescGZIP(), []int{4}
}

func (x *Site) GetSiteId() int64 {
	if x != nil {
		return x.SiteId
	}
	return 0
}

func (x *Site) GetAvailable() bool {
	if x != nil {
		return x.Available
	}
	return false
}

var File_proto_availability_msgs_proto protoreflect.FileDescriptor

var file_proto_availability_msgs_proto_rawDesc = []byte{
	0x0a, 0x1d, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x2f, 0x61, 0x76, 0x61, 0x69, 0x6c, 0x61, 0x62, 0x69,
	0x6c, 0x69, 0x74, 0x79, 0x2d, 0x6d, 0x73, 0x67, 0x73, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12,
	0x05, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x22, 0x2b, 0x0a, 0x10, 0x53, 0x69, 0x74, 0x65, 0x41, 0x76,
	0x61, 0x69, 0x6c, 0x61, 0x62, 0x6c, 0x65, 0x52, 0x65, 0x71, 0x12, 0x17, 0x0a, 0x07, 0x73, 0x69,
	0x74, 0x65, 0x5f, 0x69, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x03, 0x52, 0x06, 0x73, 0x69, 0x74,
	0x65, 0x49, 0x64, 0x22, 0x33, 0x0a, 0x10, 0x53, 0x69, 0x74, 0x65, 0x41, 0x76, 0x61, 0x69, 0x6c,
	0x61, 0x62, 0x6c, 0x65, 0x52, 0x65, 0x73, 0x12, 0x1f, 0x0a, 0x04, 0x73, 0x69, 0x74, 0x65, 0x18,
	0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x0b, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x2e, 0x53, 0x69,
	0x74, 0x65, 0x52, 0x04, 0x73, 0x69, 0x74, 0x65, 0x22, 0x13, 0x0a, 0x11, 0x53, 0x69, 0x74, 0x65,
	0x73, 0x41, 0x76, 0x61, 0x69, 0x6c, 0x61, 0x62, 0x6c, 0x65, 0x52, 0x65, 0x71, 0x22, 0x4a, 0x0a,
	0x11, 0x53, 0x69, 0x74, 0x65, 0x73, 0x41, 0x76, 0x61, 0x69, 0x6c, 0x61, 0x62, 0x6c, 0x65, 0x52,
	0x65, 0x73, 0x12, 0x35, 0x0a, 0x09, 0x72, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x73, 0x18,
	0x01, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x17, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x2e, 0x53, 0x69,
	0x74, 0x65, 0x41, 0x76, 0x61, 0x69, 0x6c, 0x61, 0x62, 0x6c, 0x65, 0x52, 0x65, 0x73, 0x52, 0x09,
	0x72, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x73, 0x22, 0x3d, 0x0a, 0x04, 0x53, 0x69, 0x74,
	0x65, 0x12, 0x17, 0x0a, 0x07, 0x73, 0x69, 0x74, 0x65, 0x5f, 0x69, 0x64, 0x18, 0x01, 0x20, 0x01,
	0x28, 0x03, 0x52, 0x06, 0x73, 0x69, 0x74, 0x65, 0x49, 0x64, 0x12, 0x1c, 0x0a, 0x09, 0x61, 0x76,
	0x61, 0x69, 0x6c, 0x61, 0x62, 0x6c, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x08, 0x52, 0x09, 0x61,
	0x76, 0x61, 0x69, 0x6c, 0x61, 0x62, 0x6c, 0x65, 0x42, 0x09, 0x5a, 0x07, 0x2e, 0x3b, 0x70, 0x72,
	0x6f, 0x74, 0x6f, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_proto_availability_msgs_proto_rawDescOnce sync.Once
	file_proto_availability_msgs_proto_rawDescData = file_proto_availability_msgs_proto_rawDesc
)

func file_proto_availability_msgs_proto_rawDescGZIP() []byte {
	file_proto_availability_msgs_proto_rawDescOnce.Do(func() {
		file_proto_availability_msgs_proto_rawDescData = protoimpl.X.CompressGZIP(file_proto_availability_msgs_proto_rawDescData)
	})
	return file_proto_availability_msgs_proto_rawDescData
}

var file_proto_availability_msgs_proto_msgTypes = make([]protoimpl.MessageInfo, 5)
var file_proto_availability_msgs_proto_goTypes = []interface{}{
	(*SiteAvailableReq)(nil),  // 0: proto.SiteAvailableReq
	(*SiteAvailableRes)(nil),  // 1: proto.SiteAvailableRes
	(*SitesAvailableReq)(nil), // 2: proto.SitesAvailableReq
	(*SitesAvailableRes)(nil), // 3: proto.SitesAvailableRes
	(*Site)(nil),              // 4: proto.Site
}
var file_proto_availability_msgs_proto_depIdxs = []int32{
	4, // 0: proto.SiteAvailableRes.site:type_name -> proto.Site
	1, // 1: proto.SitesAvailableRes.responses:type_name -> proto.SiteAvailableRes
	2, // [2:2] is the sub-list for method output_type
	2, // [2:2] is the sub-list for method input_type
	2, // [2:2] is the sub-list for extension type_name
	2, // [2:2] is the sub-list for extension extendee
	0, // [0:2] is the sub-list for field type_name
}

func init() { file_proto_availability_msgs_proto_init() }
func file_proto_availability_msgs_proto_init() {
	if File_proto_availability_msgs_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_proto_availability_msgs_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*SiteAvailableReq); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_proto_availability_msgs_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*SiteAvailableRes); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_proto_availability_msgs_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*SitesAvailableReq); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_proto_availability_msgs_proto_msgTypes[3].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*SitesAvailableRes); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_proto_availability_msgs_proto_msgTypes[4].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Site); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_proto_availability_msgs_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   5,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_proto_availability_msgs_proto_goTypes,
		DependencyIndexes: file_proto_availability_msgs_proto_depIdxs,
		MessageInfos:      file_proto_availability_msgs_proto_msgTypes,
	}.Build()
	File_proto_availability_msgs_proto = out.File
	file_proto_availability_msgs_proto_rawDesc = nil
	file_proto_availability_msgs_proto_goTypes = nil
	file_proto_availability_msgs_proto_depIdxs = nil
}
