// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: KNearestNeighborsClassifier.proto

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "KNearestNeighborsClassifier.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
// @@protoc_insertion_point(includes)

namespace CoreML {
namespace Specification {
class KNearestNeighborsClassifier_FloatVectorDefaultTypeInternal : public ::google::protobuf::internal::ExplicitlyConstructed<KNearestNeighborsClassifier_FloatVector> {
} _KNearestNeighborsClassifier_FloatVector_default_instance_;
class KNearestNeighborsClassifierDefaultTypeInternal : public ::google::protobuf::internal::ExplicitlyConstructed<KNearestNeighborsClassifier> {
  public:
  const ::CoreML::Specification::StringVector* stringclasslabels_;
  const ::CoreML::Specification::Int64Vector* int64classlabels_;
} _KNearestNeighborsClassifier_default_instance_;

namespace protobuf_KNearestNeighborsClassifier_2eproto {

PROTOBUF_CONSTEXPR_VAR ::google::protobuf::internal::ParseTableField
    const TableStruct::entries[] = {
  {0, 0, 0, ::google::protobuf::internal::kInvalidMask, 0, 0},
};

PROTOBUF_CONSTEXPR_VAR ::google::protobuf::internal::AuxillaryParseTableField
    const TableStruct::aux[] = {
  ::google::protobuf::internal::AuxillaryParseTableField(),
};
PROTOBUF_CONSTEXPR_VAR ::google::protobuf::internal::ParseTable const
    TableStruct::schema[] = {
  { NULL, NULL, 0, -1, -1, false },
  { NULL, NULL, 0, -1, -1, false },
};


void TableStruct::Shutdown() {
  _KNearestNeighborsClassifier_FloatVector_default_instance_.Shutdown();
  _KNearestNeighborsClassifier_default_instance_.Shutdown();
}

void TableStruct::InitDefaultsImpl() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  ::google::protobuf::internal::InitProtobufDefaults();
  ::CoreML::Specification::protobuf_DataStructures_2eproto::InitDefaults();
  _KNearestNeighborsClassifier_FloatVector_default_instance_.DefaultConstruct();
  _KNearestNeighborsClassifier_default_instance_.DefaultConstruct();
}

void InitDefaults() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &TableStruct::InitDefaultsImpl);
}
void AddDescriptorsImpl() {
  InitDefaults();
  ::CoreML::Specification::protobuf_DataStructures_2eproto::AddDescriptors();
  ::google::protobuf::internal::OnShutdown(&TableStruct::Shutdown);
}

void AddDescriptors() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &AddDescriptorsImpl);
}
#ifdef GOOGLE_PROTOBUF_NO_STATIC_INITIALIZER
// Force AddDescriptors() to be called at static initialization time.
struct StaticDescriptorInitializer {
  StaticDescriptorInitializer() {
    AddDescriptors();
  }
} static_descriptor_initializer;
#endif  // GOOGLE_PROTOBUF_NO_STATIC_INITIALIZER

}  // namespace protobuf_KNearestNeighborsClassifier_2eproto


// ===================================================================

#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int KNearestNeighborsClassifier_FloatVector::kVectorFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

KNearestNeighborsClassifier_FloatVector::KNearestNeighborsClassifier_FloatVector()
  : ::google::protobuf::MessageLite(), _internal_metadata_(NULL) {
  if (GOOGLE_PREDICT_TRUE(this != internal_default_instance())) {
    protobuf_KNearestNeighborsClassifier_2eproto::InitDefaults();
  }
  SharedCtor();
  // @@protoc_insertion_point(constructor:CoreML.Specification.KNearestNeighborsClassifier.FloatVector)
}
KNearestNeighborsClassifier_FloatVector::KNearestNeighborsClassifier_FloatVector(const KNearestNeighborsClassifier_FloatVector& from)
  : ::google::protobuf::MessageLite(),
      _internal_metadata_(NULL),
      vector_(from.vector_),
      _cached_size_(0) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:CoreML.Specification.KNearestNeighborsClassifier.FloatVector)
}

void KNearestNeighborsClassifier_FloatVector::SharedCtor() {
  _cached_size_ = 0;
}

KNearestNeighborsClassifier_FloatVector::~KNearestNeighborsClassifier_FloatVector() {
  // @@protoc_insertion_point(destructor:CoreML.Specification.KNearestNeighborsClassifier.FloatVector)
  SharedDtor();
}

void KNearestNeighborsClassifier_FloatVector::SharedDtor() {
}

void KNearestNeighborsClassifier_FloatVector::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const KNearestNeighborsClassifier_FloatVector& KNearestNeighborsClassifier_FloatVector::default_instance() {
  protobuf_KNearestNeighborsClassifier_2eproto::InitDefaults();
  return *internal_default_instance();
}

KNearestNeighborsClassifier_FloatVector* KNearestNeighborsClassifier_FloatVector::New(::google::protobuf::Arena* arena) const {
  KNearestNeighborsClassifier_FloatVector* n = new KNearestNeighborsClassifier_FloatVector;
  if (arena != NULL) {
    arena->Own(n);
  }
  return n;
}

void KNearestNeighborsClassifier_FloatVector::Clear() {
// @@protoc_insertion_point(message_clear_start:CoreML.Specification.KNearestNeighborsClassifier.FloatVector)
  vector_.Clear();
}

bool KNearestNeighborsClassifier_FloatVector::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:CoreML.Specification.KNearestNeighborsClassifier.FloatVector)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // repeated float vector = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(10u)) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadPackedPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, this->mutable_vector())));
        } else if (static_cast< ::google::protobuf::uint8>(tag) ==
                   static_cast< ::google::protobuf::uint8>(13u)) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadRepeatedPrimitiveNoInline<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 1, 10u, input, this->mutable_vector())));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0 ||
            ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormatLite::SkipField(input, tag));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:CoreML.Specification.KNearestNeighborsClassifier.FloatVector)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:CoreML.Specification.KNearestNeighborsClassifier.FloatVector)
  return false;
#undef DO_
}

void KNearestNeighborsClassifier_FloatVector::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:CoreML.Specification.KNearestNeighborsClassifier.FloatVector)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated float vector = 1;
  if (this->vector_size() > 0) {
    ::google::protobuf::internal::WireFormatLite::WriteTag(1, ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED, output);
    output->WriteVarint32(_vector_cached_byte_size_);
    ::google::protobuf::internal::WireFormatLite::WriteFloatArray(
      this->vector().data(), this->vector_size(), output);
  }

  // @@protoc_insertion_point(serialize_end:CoreML.Specification.KNearestNeighborsClassifier.FloatVector)
}

size_t KNearestNeighborsClassifier_FloatVector::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:CoreML.Specification.KNearestNeighborsClassifier.FloatVector)
  size_t total_size = 0;

  // repeated float vector = 1;
  {
    unsigned int count = this->vector_size();
    size_t data_size = 4UL * count;
    if (data_size > 0) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(data_size);
    }
    int cached_size = ::google::protobuf::internal::ToCachedSize(data_size);
    GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
    _vector_cached_byte_size_ = cached_size;
    GOOGLE_SAFE_CONCURRENT_WRITES_END();
    total_size += data_size;
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = cached_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void KNearestNeighborsClassifier_FloatVector::CheckTypeAndMergeFrom(
    const ::google::protobuf::MessageLite& from) {
  MergeFrom(*::google::protobuf::down_cast<const KNearestNeighborsClassifier_FloatVector*>(&from));
}

void KNearestNeighborsClassifier_FloatVector::MergeFrom(const KNearestNeighborsClassifier_FloatVector& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:CoreML.Specification.KNearestNeighborsClassifier.FloatVector)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  vector_.MergeFrom(from.vector_);
}

void KNearestNeighborsClassifier_FloatVector::CopyFrom(const KNearestNeighborsClassifier_FloatVector& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:CoreML.Specification.KNearestNeighborsClassifier.FloatVector)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool KNearestNeighborsClassifier_FloatVector::IsInitialized() const {
  return true;
}

void KNearestNeighborsClassifier_FloatVector::Swap(KNearestNeighborsClassifier_FloatVector* other) {
  if (other == this) return;
  InternalSwap(other);
}
void KNearestNeighborsClassifier_FloatVector::InternalSwap(KNearestNeighborsClassifier_FloatVector* other) {
  vector_.InternalSwap(&other->vector_);
  std::swap(_cached_size_, other->_cached_size_);
}

::std::string KNearestNeighborsClassifier_FloatVector::GetTypeName() const {
  return "CoreML.Specification.KNearestNeighborsClassifier.FloatVector";
}

#if PROTOBUF_INLINE_NOT_IN_HEADERS
// KNearestNeighborsClassifier_FloatVector

// repeated float vector = 1;
int KNearestNeighborsClassifier_FloatVector::vector_size() const {
  return vector_.size();
}
void KNearestNeighborsClassifier_FloatVector::clear_vector() {
  vector_.Clear();
}
float KNearestNeighborsClassifier_FloatVector::vector(int index) const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.KNearestNeighborsClassifier.FloatVector.vector)
  return vector_.Get(index);
}
void KNearestNeighborsClassifier_FloatVector::set_vector(int index, float value) {
  vector_.Set(index, value);
  // @@protoc_insertion_point(field_set:CoreML.Specification.KNearestNeighborsClassifier.FloatVector.vector)
}
void KNearestNeighborsClassifier_FloatVector::add_vector(float value) {
  vector_.Add(value);
  // @@protoc_insertion_point(field_add:CoreML.Specification.KNearestNeighborsClassifier.FloatVector.vector)
}
const ::google::protobuf::RepeatedField< float >&
KNearestNeighborsClassifier_FloatVector::vector() const {
  // @@protoc_insertion_point(field_list:CoreML.Specification.KNearestNeighborsClassifier.FloatVector.vector)
  return vector_;
}
::google::protobuf::RepeatedField< float >*
KNearestNeighborsClassifier_FloatVector::mutable_vector() {
  // @@protoc_insertion_point(field_mutable_list:CoreML.Specification.KNearestNeighborsClassifier.FloatVector.vector)
  return &vector_;
}

#endif  // PROTOBUF_INLINE_NOT_IN_HEADERS

// ===================================================================

#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int KNearestNeighborsClassifier::kDimensionalityFieldNumber;
const int KNearestNeighborsClassifier::kFloatSamplesFieldNumber;
const int KNearestNeighborsClassifier::kKFieldNumber;
const int KNearestNeighborsClassifier::kStringClassLabelsFieldNumber;
const int KNearestNeighborsClassifier::kInt64ClassLabelsFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

KNearestNeighborsClassifier::KNearestNeighborsClassifier()
  : ::google::protobuf::MessageLite(), _internal_metadata_(NULL) {
  if (GOOGLE_PREDICT_TRUE(this != internal_default_instance())) {
    protobuf_KNearestNeighborsClassifier_2eproto::InitDefaults();
  }
  SharedCtor();
  // @@protoc_insertion_point(constructor:CoreML.Specification.KNearestNeighborsClassifier)
}
KNearestNeighborsClassifier::KNearestNeighborsClassifier(const KNearestNeighborsClassifier& from)
  : ::google::protobuf::MessageLite(),
      _internal_metadata_(NULL),
      floatsamples_(from.floatsamples_),
      _cached_size_(0) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::memcpy(&dimensionality_, &from.dimensionality_,
    reinterpret_cast<char*>(&k_) -
    reinterpret_cast<char*>(&dimensionality_) + sizeof(k_));
  clear_has_ClassLabels();
  switch (from.ClassLabels_case()) {
    case kStringClassLabels: {
      mutable_stringclasslabels()->::CoreML::Specification::StringVector::MergeFrom(from.stringclasslabels());
      break;
    }
    case kInt64ClassLabels: {
      mutable_int64classlabels()->::CoreML::Specification::Int64Vector::MergeFrom(from.int64classlabels());
      break;
    }
    case CLASSLABELS_NOT_SET: {
      break;
    }
  }
  // @@protoc_insertion_point(copy_constructor:CoreML.Specification.KNearestNeighborsClassifier)
}

void KNearestNeighborsClassifier::SharedCtor() {
  ::memset(&dimensionality_, 0, reinterpret_cast<char*>(&k_) -
    reinterpret_cast<char*>(&dimensionality_) + sizeof(k_));
  clear_has_ClassLabels();
  _cached_size_ = 0;
}

KNearestNeighborsClassifier::~KNearestNeighborsClassifier() {
  // @@protoc_insertion_point(destructor:CoreML.Specification.KNearestNeighborsClassifier)
  SharedDtor();
}

void KNearestNeighborsClassifier::SharedDtor() {
  if (has_ClassLabels()) {
    clear_ClassLabels();
  }
}

void KNearestNeighborsClassifier::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const KNearestNeighborsClassifier& KNearestNeighborsClassifier::default_instance() {
  protobuf_KNearestNeighborsClassifier_2eproto::InitDefaults();
  return *internal_default_instance();
}

KNearestNeighborsClassifier* KNearestNeighborsClassifier::New(::google::protobuf::Arena* arena) const {
  KNearestNeighborsClassifier* n = new KNearestNeighborsClassifier;
  if (arena != NULL) {
    arena->Own(n);
  }
  return n;
}

void KNearestNeighborsClassifier::clear_ClassLabels() {
// @@protoc_insertion_point(one_of_clear_start:CoreML.Specification.KNearestNeighborsClassifier)
  switch (ClassLabels_case()) {
    case kStringClassLabels: {
      delete ClassLabels_.stringclasslabels_;
      break;
    }
    case kInt64ClassLabels: {
      delete ClassLabels_.int64classlabels_;
      break;
    }
    case CLASSLABELS_NOT_SET: {
      break;
    }
  }
  _oneof_case_[0] = CLASSLABELS_NOT_SET;
}


void KNearestNeighborsClassifier::Clear() {
// @@protoc_insertion_point(message_clear_start:CoreML.Specification.KNearestNeighborsClassifier)
  floatsamples_.Clear();
  ::memset(&dimensionality_, 0, reinterpret_cast<char*>(&k_) -
    reinterpret_cast<char*>(&dimensionality_) + sizeof(k_));
  clear_ClassLabels();
}

bool KNearestNeighborsClassifier::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:CoreML.Specification.KNearestNeighborsClassifier)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(16383u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // int32 dimensionality = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(8u)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &dimensionality_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // repeated .CoreML.Specification.KNearestNeighborsClassifier.FloatVector floatSamples = 4;
      case 4: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(34u)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
                input, add_floatsamples()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // int32 k = 10;
      case 10: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(80u)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &k_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // .CoreML.Specification.StringVector stringClassLabels = 100;
      case 100: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(802u)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
               input, mutable_stringclasslabels()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // .CoreML.Specification.Int64Vector int64ClassLabels = 101;
      case 101: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(810u)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
               input, mutable_int64classlabels()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0 ||
            ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormatLite::SkipField(input, tag));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:CoreML.Specification.KNearestNeighborsClassifier)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:CoreML.Specification.KNearestNeighborsClassifier)
  return false;
#undef DO_
}

void KNearestNeighborsClassifier::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:CoreML.Specification.KNearestNeighborsClassifier)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // int32 dimensionality = 1;
  if (this->dimensionality() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(1, this->dimensionality(), output);
  }

  // repeated .CoreML.Specification.KNearestNeighborsClassifier.FloatVector floatSamples = 4;
  for (unsigned int i = 0, n = this->floatsamples_size(); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::WriteMessage(
      4, this->floatsamples(i), output);
  }

  // int32 k = 10;
  if (this->k() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(10, this->k(), output);
  }

  // .CoreML.Specification.StringVector stringClassLabels = 100;
  if (has_stringclasslabels()) {
    ::google::protobuf::internal::WireFormatLite::WriteMessage(
      100, *ClassLabels_.stringclasslabels_, output);
  }

  // .CoreML.Specification.Int64Vector int64ClassLabels = 101;
  if (has_int64classlabels()) {
    ::google::protobuf::internal::WireFormatLite::WriteMessage(
      101, *ClassLabels_.int64classlabels_, output);
  }

  // @@protoc_insertion_point(serialize_end:CoreML.Specification.KNearestNeighborsClassifier)
}

size_t KNearestNeighborsClassifier::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:CoreML.Specification.KNearestNeighborsClassifier)
  size_t total_size = 0;

  // repeated .CoreML.Specification.KNearestNeighborsClassifier.FloatVector floatSamples = 4;
  {
    unsigned int count = this->floatsamples_size();
    total_size += 1UL * count;
    for (unsigned int i = 0; i < count; i++) {
      total_size +=
        ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
          this->floatsamples(i));
    }
  }

  // int32 dimensionality = 1;
  if (this->dimensionality() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::Int32Size(
        this->dimensionality());
  }

  // int32 k = 10;
  if (this->k() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::Int32Size(
        this->k());
  }

  switch (ClassLabels_case()) {
    // .CoreML.Specification.StringVector stringClassLabels = 100;
    case kStringClassLabels: {
      total_size += 2 +
        ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
          *ClassLabels_.stringclasslabels_);
      break;
    }
    // .CoreML.Specification.Int64Vector int64ClassLabels = 101;
    case kInt64ClassLabels: {
      total_size += 2 +
        ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
          *ClassLabels_.int64classlabels_);
      break;
    }
    case CLASSLABELS_NOT_SET: {
      break;
    }
  }
  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = cached_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void KNearestNeighborsClassifier::CheckTypeAndMergeFrom(
    const ::google::protobuf::MessageLite& from) {
  MergeFrom(*::google::protobuf::down_cast<const KNearestNeighborsClassifier*>(&from));
}

void KNearestNeighborsClassifier::MergeFrom(const KNearestNeighborsClassifier& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:CoreML.Specification.KNearestNeighborsClassifier)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  floatsamples_.MergeFrom(from.floatsamples_);
  if (from.dimensionality() != 0) {
    set_dimensionality(from.dimensionality());
  }
  if (from.k() != 0) {
    set_k(from.k());
  }
  switch (from.ClassLabels_case()) {
    case kStringClassLabels: {
      mutable_stringclasslabels()->::CoreML::Specification::StringVector::MergeFrom(from.stringclasslabels());
      break;
    }
    case kInt64ClassLabels: {
      mutable_int64classlabels()->::CoreML::Specification::Int64Vector::MergeFrom(from.int64classlabels());
      break;
    }
    case CLASSLABELS_NOT_SET: {
      break;
    }
  }
}

void KNearestNeighborsClassifier::CopyFrom(const KNearestNeighborsClassifier& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:CoreML.Specification.KNearestNeighborsClassifier)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool KNearestNeighborsClassifier::IsInitialized() const {
  return true;
}

void KNearestNeighborsClassifier::Swap(KNearestNeighborsClassifier* other) {
  if (other == this) return;
  InternalSwap(other);
}
void KNearestNeighborsClassifier::InternalSwap(KNearestNeighborsClassifier* other) {
  floatsamples_.InternalSwap(&other->floatsamples_);
  std::swap(dimensionality_, other->dimensionality_);
  std::swap(k_, other->k_);
  std::swap(ClassLabels_, other->ClassLabels_);
  std::swap(_oneof_case_[0], other->_oneof_case_[0]);
  std::swap(_cached_size_, other->_cached_size_);
}

::std::string KNearestNeighborsClassifier::GetTypeName() const {
  return "CoreML.Specification.KNearestNeighborsClassifier";
}

#if PROTOBUF_INLINE_NOT_IN_HEADERS
// KNearestNeighborsClassifier

// int32 dimensionality = 1;
void KNearestNeighborsClassifier::clear_dimensionality() {
  dimensionality_ = 0;
}
::google::protobuf::int32 KNearestNeighborsClassifier::dimensionality() const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.KNearestNeighborsClassifier.dimensionality)
  return dimensionality_;
}
void KNearestNeighborsClassifier::set_dimensionality(::google::protobuf::int32 value) {
  
  dimensionality_ = value;
  // @@protoc_insertion_point(field_set:CoreML.Specification.KNearestNeighborsClassifier.dimensionality)
}

// repeated .CoreML.Specification.KNearestNeighborsClassifier.FloatVector floatSamples = 4;
int KNearestNeighborsClassifier::floatsamples_size() const {
  return floatsamples_.size();
}
void KNearestNeighborsClassifier::clear_floatsamples() {
  floatsamples_.Clear();
}
const ::CoreML::Specification::KNearestNeighborsClassifier_FloatVector& KNearestNeighborsClassifier::floatsamples(int index) const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.KNearestNeighborsClassifier.floatSamples)
  return floatsamples_.Get(index);
}
::CoreML::Specification::KNearestNeighborsClassifier_FloatVector* KNearestNeighborsClassifier::mutable_floatsamples(int index) {
  // @@protoc_insertion_point(field_mutable:CoreML.Specification.KNearestNeighborsClassifier.floatSamples)
  return floatsamples_.Mutable(index);
}
::CoreML::Specification::KNearestNeighborsClassifier_FloatVector* KNearestNeighborsClassifier::add_floatsamples() {
  // @@protoc_insertion_point(field_add:CoreML.Specification.KNearestNeighborsClassifier.floatSamples)
  return floatsamples_.Add();
}
::google::protobuf::RepeatedPtrField< ::CoreML::Specification::KNearestNeighborsClassifier_FloatVector >*
KNearestNeighborsClassifier::mutable_floatsamples() {
  // @@protoc_insertion_point(field_mutable_list:CoreML.Specification.KNearestNeighborsClassifier.floatSamples)
  return &floatsamples_;
}
const ::google::protobuf::RepeatedPtrField< ::CoreML::Specification::KNearestNeighborsClassifier_FloatVector >&
KNearestNeighborsClassifier::floatsamples() const {
  // @@protoc_insertion_point(field_list:CoreML.Specification.KNearestNeighborsClassifier.floatSamples)
  return floatsamples_;
}

// int32 k = 10;
void KNearestNeighborsClassifier::clear_k() {
  k_ = 0;
}
::google::protobuf::int32 KNearestNeighborsClassifier::k() const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.KNearestNeighborsClassifier.k)
  return k_;
}
void KNearestNeighborsClassifier::set_k(::google::protobuf::int32 value) {
  
  k_ = value;
  // @@protoc_insertion_point(field_set:CoreML.Specification.KNearestNeighborsClassifier.k)
}

// .CoreML.Specification.StringVector stringClassLabels = 100;
bool KNearestNeighborsClassifier::has_stringclasslabels() const {
  return ClassLabels_case() == kStringClassLabels;
}
void KNearestNeighborsClassifier::set_has_stringclasslabels() {
  _oneof_case_[0] = kStringClassLabels;
}
void KNearestNeighborsClassifier::clear_stringclasslabels() {
  if (has_stringclasslabels()) {
    delete ClassLabels_.stringclasslabels_;
    clear_has_ClassLabels();
  }
}
 const ::CoreML::Specification::StringVector& KNearestNeighborsClassifier::stringclasslabels() const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.KNearestNeighborsClassifier.stringClassLabels)
  return has_stringclasslabels()
      ? *ClassLabels_.stringclasslabels_
      : ::CoreML::Specification::StringVector::default_instance();
}
::CoreML::Specification::StringVector* KNearestNeighborsClassifier::mutable_stringclasslabels() {
  if (!has_stringclasslabels()) {
    clear_ClassLabels();
    set_has_stringclasslabels();
    ClassLabels_.stringclasslabels_ = new ::CoreML::Specification::StringVector;
  }
  // @@protoc_insertion_point(field_mutable:CoreML.Specification.KNearestNeighborsClassifier.stringClassLabels)
  return ClassLabels_.stringclasslabels_;
}
::CoreML::Specification::StringVector* KNearestNeighborsClassifier::release_stringclasslabels() {
  // @@protoc_insertion_point(field_release:CoreML.Specification.KNearestNeighborsClassifier.stringClassLabels)
  if (has_stringclasslabels()) {
    clear_has_ClassLabels();
    ::CoreML::Specification::StringVector* temp = ClassLabels_.stringclasslabels_;
    ClassLabels_.stringclasslabels_ = NULL;
    return temp;
  } else {
    return NULL;
  }
}
void KNearestNeighborsClassifier::set_allocated_stringclasslabels(::CoreML::Specification::StringVector* stringclasslabels) {
  clear_ClassLabels();
  if (stringclasslabels) {
    set_has_stringclasslabels();
    ClassLabels_.stringclasslabels_ = stringclasslabels;
  }
  // @@protoc_insertion_point(field_set_allocated:CoreML.Specification.KNearestNeighborsClassifier.stringClassLabels)
}

// .CoreML.Specification.Int64Vector int64ClassLabels = 101;
bool KNearestNeighborsClassifier::has_int64classlabels() const {
  return ClassLabels_case() == kInt64ClassLabels;
}
void KNearestNeighborsClassifier::set_has_int64classlabels() {
  _oneof_case_[0] = kInt64ClassLabels;
}
void KNearestNeighborsClassifier::clear_int64classlabels() {
  if (has_int64classlabels()) {
    delete ClassLabels_.int64classlabels_;
    clear_has_ClassLabels();
  }
}
 const ::CoreML::Specification::Int64Vector& KNearestNeighborsClassifier::int64classlabels() const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.KNearestNeighborsClassifier.int64ClassLabels)
  return has_int64classlabels()
      ? *ClassLabels_.int64classlabels_
      : ::CoreML::Specification::Int64Vector::default_instance();
}
::CoreML::Specification::Int64Vector* KNearestNeighborsClassifier::mutable_int64classlabels() {
  if (!has_int64classlabels()) {
    clear_ClassLabels();
    set_has_int64classlabels();
    ClassLabels_.int64classlabels_ = new ::CoreML::Specification::Int64Vector;
  }
  // @@protoc_insertion_point(field_mutable:CoreML.Specification.KNearestNeighborsClassifier.int64ClassLabels)
  return ClassLabels_.int64classlabels_;
}
::CoreML::Specification::Int64Vector* KNearestNeighborsClassifier::release_int64classlabels() {
  // @@protoc_insertion_point(field_release:CoreML.Specification.KNearestNeighborsClassifier.int64ClassLabels)
  if (has_int64classlabels()) {
    clear_has_ClassLabels();
    ::CoreML::Specification::Int64Vector* temp = ClassLabels_.int64classlabels_;
    ClassLabels_.int64classlabels_ = NULL;
    return temp;
  } else {
    return NULL;
  }
}
void KNearestNeighborsClassifier::set_allocated_int64classlabels(::CoreML::Specification::Int64Vector* int64classlabels) {
  clear_ClassLabels();
  if (int64classlabels) {
    set_has_int64classlabels();
    ClassLabels_.int64classlabels_ = int64classlabels;
  }
  // @@protoc_insertion_point(field_set_allocated:CoreML.Specification.KNearestNeighborsClassifier.int64ClassLabels)
}

bool KNearestNeighborsClassifier::has_ClassLabels() const {
  return ClassLabels_case() != CLASSLABELS_NOT_SET;
}
void KNearestNeighborsClassifier::clear_has_ClassLabels() {
  _oneof_case_[0] = CLASSLABELS_NOT_SET;
}
KNearestNeighborsClassifier::ClassLabelsCase KNearestNeighborsClassifier::ClassLabels_case() const {
  return KNearestNeighborsClassifier::ClassLabelsCase(_oneof_case_[0]);
}
#endif  // PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

}  // namespace Specification
}  // namespace CoreML

// @@protoc_insertion_point(global_scope)
