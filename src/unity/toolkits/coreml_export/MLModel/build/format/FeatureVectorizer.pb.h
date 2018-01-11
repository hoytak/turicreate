/* Copyright © 2017 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: FeatureVectorizer.proto

#ifndef PROTOBUF_FeatureVectorizer_2eproto__INCLUDED
#define PROTOBUF_FeatureVectorizer_2eproto__INCLUDED

#include <string>

#include <protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3001000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3001000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <protobuf/arena.h>
#include <protobuf/arenastring.h>
#include <protobuf/generated_message_util.h>
#include <protobuf/message_lite.h>
#include <protobuf/repeated_field.h>
#include <protobuf/extension_set.h>
// @@protoc_insertion_point(includes)

namespace CoreML {
namespace Specification {

// Internal implementation detail -- do not call these.
void protobuf_AddDesc_FeatureVectorizer_2eproto();
void protobuf_InitDefaults_FeatureVectorizer_2eproto();
void protobuf_AssignDesc_FeatureVectorizer_2eproto();
void protobuf_ShutdownFile_FeatureVectorizer_2eproto();

class FeatureVectorizer;
class FeatureVectorizer_InputColumn;

// ===================================================================

class FeatureVectorizer_InputColumn : public ::google::protobuf::MessageLite /* @@protoc_insertion_point(class_definition:CoreML.Specification.FeatureVectorizer.InputColumn) */ {
 public:
  FeatureVectorizer_InputColumn();
  virtual ~FeatureVectorizer_InputColumn();

  FeatureVectorizer_InputColumn(const FeatureVectorizer_InputColumn& from);

  inline FeatureVectorizer_InputColumn& operator=(const FeatureVectorizer_InputColumn& from) {
    CopyFrom(from);
    return *this;
  }

  static const FeatureVectorizer_InputColumn& default_instance();

  static const FeatureVectorizer_InputColumn* internal_default_instance();

  void Swap(FeatureVectorizer_InputColumn* other);

  // implements Message ----------------------------------------------

  inline FeatureVectorizer_InputColumn* New() const { return New(NULL); }

  FeatureVectorizer_InputColumn* New(::google::protobuf::Arena* arena) const;
  void CheckTypeAndMergeFrom(const ::google::protobuf::MessageLite& from);
  void CopyFrom(const FeatureVectorizer_InputColumn& from);
  void MergeFrom(const FeatureVectorizer_InputColumn& from);
  void Clear();
  bool IsInitialized() const;

  size_t ByteSizeLong() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  void DiscardUnknownFields();
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(FeatureVectorizer_InputColumn* other);
  void UnsafeMergeFrom(const FeatureVectorizer_InputColumn& from);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _arena_ptr_;
  }
  inline ::google::protobuf::Arena* MaybeArenaPtr() const {
    return _arena_ptr_;
  }
  public:

  ::std::string GetTypeName() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional string inputColumn = 1;
  void clear_inputcolumn();
  static const int kInputColumnFieldNumber = 1;
  const ::std::string& inputcolumn() const;
  void set_inputcolumn(const ::std::string& value);
  void set_inputcolumn(const char* value);
  void set_inputcolumn(const char* value, size_t size);
  ::std::string* mutable_inputcolumn();
  ::std::string* release_inputcolumn();
  void set_allocated_inputcolumn(::std::string* inputcolumn);

  // optional uint64 inputDimensions = 2;
  void clear_inputdimensions();
  static const int kInputDimensionsFieldNumber = 2;
  ::google::protobuf::uint64 inputdimensions() const;
  void set_inputdimensions(::google::protobuf::uint64 value);

  // @@protoc_insertion_point(class_scope:CoreML.Specification.FeatureVectorizer.InputColumn)
 private:

  ::google::protobuf::internal::ArenaStringPtr _unknown_fields_;
  ::google::protobuf::Arena* _arena_ptr_;

  ::google::protobuf::internal::ArenaStringPtr inputcolumn_;
  ::google::protobuf::uint64 inputdimensions_;
  mutable int _cached_size_;
  friend void  protobuf_InitDefaults_FeatureVectorizer_2eproto_impl();
  friend void  protobuf_AddDesc_FeatureVectorizer_2eproto_impl();
  friend void protobuf_AssignDesc_FeatureVectorizer_2eproto();
  friend void protobuf_ShutdownFile_FeatureVectorizer_2eproto();

  void InitAsDefaultInstance();
};
extern ::google::protobuf::internal::ExplicitlyConstructed<FeatureVectorizer_InputColumn> FeatureVectorizer_InputColumn_default_instance_;

// -------------------------------------------------------------------

class FeatureVectorizer : public ::google::protobuf::MessageLite /* @@protoc_insertion_point(class_definition:CoreML.Specification.FeatureVectorizer) */ {
 public:
  FeatureVectorizer();
  virtual ~FeatureVectorizer();

  FeatureVectorizer(const FeatureVectorizer& from);

  inline FeatureVectorizer& operator=(const FeatureVectorizer& from) {
    CopyFrom(from);
    return *this;
  }

  static const FeatureVectorizer& default_instance();

  static const FeatureVectorizer* internal_default_instance();

  void Swap(FeatureVectorizer* other);

  // implements Message ----------------------------------------------

  inline FeatureVectorizer* New() const { return New(NULL); }

  FeatureVectorizer* New(::google::protobuf::Arena* arena) const;
  void CheckTypeAndMergeFrom(const ::google::protobuf::MessageLite& from);
  void CopyFrom(const FeatureVectorizer& from);
  void MergeFrom(const FeatureVectorizer& from);
  void Clear();
  bool IsInitialized() const;

  size_t ByteSizeLong() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  void DiscardUnknownFields();
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(FeatureVectorizer* other);
  void UnsafeMergeFrom(const FeatureVectorizer& from);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _arena_ptr_;
  }
  inline ::google::protobuf::Arena* MaybeArenaPtr() const {
    return _arena_ptr_;
  }
  public:

  ::std::string GetTypeName() const;

  // nested types ----------------------------------------------------

  typedef FeatureVectorizer_InputColumn InputColumn;

  // accessors -------------------------------------------------------

  // repeated .CoreML.Specification.FeatureVectorizer.InputColumn inputList = 1;
  int inputlist_size() const;
  void clear_inputlist();
  static const int kInputListFieldNumber = 1;
  const ::CoreML::Specification::FeatureVectorizer_InputColumn& inputlist(int index) const;
  ::CoreML::Specification::FeatureVectorizer_InputColumn* mutable_inputlist(int index);
  ::CoreML::Specification::FeatureVectorizer_InputColumn* add_inputlist();
  ::google::protobuf::RepeatedPtrField< ::CoreML::Specification::FeatureVectorizer_InputColumn >*
      mutable_inputlist();
  const ::google::protobuf::RepeatedPtrField< ::CoreML::Specification::FeatureVectorizer_InputColumn >&
      inputlist() const;

  // @@protoc_insertion_point(class_scope:CoreML.Specification.FeatureVectorizer)
 private:

  ::google::protobuf::internal::ArenaStringPtr _unknown_fields_;
  ::google::protobuf::Arena* _arena_ptr_;

  ::google::protobuf::RepeatedPtrField< ::CoreML::Specification::FeatureVectorizer_InputColumn > inputlist_;
  mutable int _cached_size_;
  friend void  protobuf_InitDefaults_FeatureVectorizer_2eproto_impl();
  friend void  protobuf_AddDesc_FeatureVectorizer_2eproto_impl();
  friend void protobuf_AssignDesc_FeatureVectorizer_2eproto();
  friend void protobuf_ShutdownFile_FeatureVectorizer_2eproto();

  void InitAsDefaultInstance();
};
extern ::google::protobuf::internal::ExplicitlyConstructed<FeatureVectorizer> FeatureVectorizer_default_instance_;

// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
// FeatureVectorizer_InputColumn

// optional string inputColumn = 1;
inline void FeatureVectorizer_InputColumn::clear_inputcolumn() {
  inputcolumn_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& FeatureVectorizer_InputColumn::inputcolumn() const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.FeatureVectorizer.InputColumn.inputColumn)
  return inputcolumn_.GetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void FeatureVectorizer_InputColumn::set_inputcolumn(const ::std::string& value) {

  inputcolumn_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:CoreML.Specification.FeatureVectorizer.InputColumn.inputColumn)
}
inline void FeatureVectorizer_InputColumn::set_inputcolumn(const char* value) {

  inputcolumn_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:CoreML.Specification.FeatureVectorizer.InputColumn.inputColumn)
}
inline void FeatureVectorizer_InputColumn::set_inputcolumn(const char* value, size_t size) {

  inputcolumn_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:CoreML.Specification.FeatureVectorizer.InputColumn.inputColumn)
}
inline ::std::string* FeatureVectorizer_InputColumn::mutable_inputcolumn() {

  // @@protoc_insertion_point(field_mutable:CoreML.Specification.FeatureVectorizer.InputColumn.inputColumn)
  return inputcolumn_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* FeatureVectorizer_InputColumn::release_inputcolumn() {
  // @@protoc_insertion_point(field_release:CoreML.Specification.FeatureVectorizer.InputColumn.inputColumn)

  return inputcolumn_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void FeatureVectorizer_InputColumn::set_allocated_inputcolumn(::std::string* inputcolumn) {
  if (inputcolumn != NULL) {

  } else {

  }
  inputcolumn_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), inputcolumn);
  // @@protoc_insertion_point(field_set_allocated:CoreML.Specification.FeatureVectorizer.InputColumn.inputColumn)
}

// optional uint64 inputDimensions = 2;
inline void FeatureVectorizer_InputColumn::clear_inputdimensions() {
  inputdimensions_ = GOOGLE_ULONGLONG(0);
}
inline ::google::protobuf::uint64 FeatureVectorizer_InputColumn::inputdimensions() const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.FeatureVectorizer.InputColumn.inputDimensions)
  return inputdimensions_;
}
inline void FeatureVectorizer_InputColumn::set_inputdimensions(::google::protobuf::uint64 value) {

  inputdimensions_ = value;
  // @@protoc_insertion_point(field_set:CoreML.Specification.FeatureVectorizer.InputColumn.inputDimensions)
}

inline const FeatureVectorizer_InputColumn* FeatureVectorizer_InputColumn::internal_default_instance() {
  return &FeatureVectorizer_InputColumn_default_instance_.get();
}
// -------------------------------------------------------------------

// FeatureVectorizer

// repeated .CoreML.Specification.FeatureVectorizer.InputColumn inputList = 1;
inline int FeatureVectorizer::inputlist_size() const {
  return inputlist_.size();
}
inline void FeatureVectorizer::clear_inputlist() {
  inputlist_.Clear();
}
inline const ::CoreML::Specification::FeatureVectorizer_InputColumn& FeatureVectorizer::inputlist(int index) const {
  // @@protoc_insertion_point(field_get:CoreML.Specification.FeatureVectorizer.inputList)
  return inputlist_.Get(index);
}
inline ::CoreML::Specification::FeatureVectorizer_InputColumn* FeatureVectorizer::mutable_inputlist(int index) {
  // @@protoc_insertion_point(field_mutable:CoreML.Specification.FeatureVectorizer.inputList)
  return inputlist_.Mutable(index);
}
inline ::CoreML::Specification::FeatureVectorizer_InputColumn* FeatureVectorizer::add_inputlist() {
  // @@protoc_insertion_point(field_add:CoreML.Specification.FeatureVectorizer.inputList)
  return inputlist_.Add();
}
inline ::google::protobuf::RepeatedPtrField< ::CoreML::Specification::FeatureVectorizer_InputColumn >*
FeatureVectorizer::mutable_inputlist() {
  // @@protoc_insertion_point(field_mutable_list:CoreML.Specification.FeatureVectorizer.inputList)
  return &inputlist_;
}
inline const ::google::protobuf::RepeatedPtrField< ::CoreML::Specification::FeatureVectorizer_InputColumn >&
FeatureVectorizer::inputlist() const {
  // @@protoc_insertion_point(field_list:CoreML.Specification.FeatureVectorizer.inputList)
  return inputlist_;
}

inline const FeatureVectorizer* FeatureVectorizer::internal_default_instance() {
  return &FeatureVectorizer_default_instance_.get();
}
#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace Specification
}  // namespace CoreML

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_FeatureVectorizer_2eproto__INCLUDED
