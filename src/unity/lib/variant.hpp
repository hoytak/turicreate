/* Copyright © 2017 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#ifndef TURI_UNITY_VARIANT_HPP
#define TURI_UNITY_VARIANT_HPP
#include <string>
#include <map>
#include <vector>
#include <boost/variant.hpp>
#include <flexible_type/flexible_type.hpp>
#include <sframe/dataframe.hpp>
#include <serialization/serialization_includes.hpp>

namespace turi {
class model_base;
struct function_closure_info;
class unity_sframe_base;
class unity_sgraph_base;
class unity_sarray_base;

/**
 * A variant object that can be communicated between Python and C++ which
 * contains either a
 * \li flexible_type
 * \li std::shared_ptr<unity_sgraph>
 * \li dataframe_t
 * \li model
 * \li std::shared_ptr<unity_sframe>
 * \li std::shared_ptr<unity_sarray>
 * \li std::map<variant>
 * \li std::vector<variant>
 *
 * See the boost variant documentation for details.
 *
 * The variant should not be accessed directly. See \ref to_variant
 * and \ref variant_get_value for powerful ways to extract or store values
 * from a variant.
 */
typedef typename boost::make_recursive_variant<
            flexible_type,
            std::shared_ptr<unity_sgraph_base>,
            dataframe_t,
            std::shared_ptr<model_base>,
            std::shared_ptr<unity_sframe_base>,
            std::shared_ptr<unity_sarray_base>,
            std::map<std::string, boost::recursive_variant_>,
            std::vector<boost::recursive_variant_>,
            boost::recursive_wrapper<function_closure_info> >::type variant_type;

/*
 * A map of string to variant. Also a type the variant type can store.
 */


typedef std::map<std::string, variant_type> variant_map_type;

/*
 * A map of vector to variant. Also a type that the variant_type can store
 */
typedef std::vector<variant_type> variant_vector_type;

// Type codes for the various variant types.  Do not change these or all
// deserialization will fail.
static const int VT_FLEXIBLE_TYPE = 0;
static const int VT_SGRAPH = 1;
static const int VT_DATAFRAME = 2;
static const int VT_MODEL = 3;
static const int VT_SFRAME = 4;
static const int VT_SARRAY = 5;
static const int VT_DICTIONARY = 6;
static const int VT_LIST = 7;
static const int VT_FUNCTION = 8;

/**
 * Given variant.which() gets the name of the type inside it.
 */
inline std::string get_variant_which_name(int variant_type_code) {
  switch(variant_type_code) {
   case VT_FLEXIBLE_TYPE:
     return "flexible_type";
   case VT_SGRAPH:
     return "SGraph";
   case VT_DATAFRAME:
     return "Dataframe";
   case VT_MODEL:
     return "Model";
   case VT_SFRAME:
     return "SFrame";
   case VT_SARRAY:
     return "SArray";
   case VT_DICTIONARY:
     return "Dictionary";
   case VT_LIST:
     return "List";
   case VT_FUNCTION:
     return "Function";
   default:
     return "";
  }
}

} // namespace turi


namespace turi{ namespace archive_detail {

template <>
struct serialize_impl<oarchive, turi::variant_type, false> {
  static void exec(oarchive& arc, const turi::variant_type& tval);
};

template <>
struct deserialize_impl<iarchive, turi::variant_type, false> {
  static void exec(iarchive& arc, turi::variant_type& tval);
};
}
}



namespace turi {

template <typename T>
std::string full_type_name(const T& example = T()) {
  // Is this a flexible type?  If so, then use that for the name instead.
  if (is_flexible_type_convertible<T>::value) {
    flexible_type f = convert_to_flexible_type(example);
    return flex_type_enum_to_name(f.get_type());
  } else {
    return get_variant_which_name(to_variant(example).which());
  }
}

std::string full_type_name(const variant_type& example) {
  if(example.which() == 0) {
    return flex_type_enum_to_name(variant_get_ref<flexible_type>(example).get_type());
  } else {
    return get_variant_which_name(example.which());
  }
}


// A list of accessors to help Cython access the variant
template <typename T>
GL_COLD_NOINLINE_ERROR
void _throw_variant_error(const variant_type& v) {
  std::string errormsg =  //
      std::string("Variant type error: Expecting ") + full_type_name<T>() +
      " but got a " + full_type_name(v);
  std_log_and_throw(errormsg);
}

/**
 * Gets a reference to a content of a variant.
 * Throws if variant contains an inappropriate type.
 */
template <typename T>
static inline T& variant_get_ref(variant_type& v) {
  try {
    return boost::get<T>(v);
  } catch (...) {
    _throw_variant_error<T>(v);
  }
}

/**
 * Gets a const reference to the content of a variant.
 * Throws if variant contains an inappropriate type.
 */
template <typename T>
static inline const T& variant_get_ref(const variant_type& v) {
  try {
  return boost::get<T>(v);
  } catch (...) {
    _throw_variant_error<T>(v);
  }
}





// Convenience functions 
template <typename T>
GL_HOT_INLINE_FLATTEN 
inline bool variant_is(const variant_type&) { 
   return false; 
}

template <>
GL_HOT_INLINE_FLATTEN 
inline bool variant_is<flexible_type>(const variant_type& t) {
   return t.which() == 0;
}

template <>
GL_HOT_INLINE_FLATTEN 
inline bool variant_is<flex_string>(const variant_type& t) {
   return variant_is<flexible_type>(t) && (variant_get_ref<flexible_type>(t).get_type() == flex_type_enum::STRING);
}

template <>
GL_HOT_INLINE_FLATTEN 
inline bool variant_is<flex_vec>(const variant_type& t) {
   return variant_is<flexible_type>(t) && (variant_get_ref<flexible_type>(t).get_type() == flex_type_enum::VECTOR);
}

template <>
GL_HOT_INLINE_FLATTEN 
inline bool variant_is<flex_int>(const variant_type& t) {
   return variant_is<flexible_type>(t) && (variant_get_ref<flexible_type>(t).get_type() == flex_type_enum::INTEGER);
}

template <>
GL_HOT_INLINE_FLATTEN 
inline bool variant_is<flex_float>(const variant_type& t) {
   return variant_is<flexible_type>(t) && (variant_get_ref<flexible_type>(t).get_type() == flex_type_enum::FLOAT);
}

template <>
GL_HOT_INLINE_FLATTEN 
inline bool variant_is<flex_list>(const variant_type& t) {
   return variant_is<flexible_type>(t) && (variant_get_ref<flexible_type>(t).get_type() == flex_type_enum::LIST);
}

template <>
GL_HOT_INLINE_FLATTEN 
inline bool variant_is<flex_dict>(const variant_type& t) {
   return variant_is<flexible_type>(t) && (variant_get_ref<flexible_type>(t).get_type() == flex_type_enum::DICT);
}

template <>
GL_HOT_INLINE_FLATTEN 
inline bool variant_is<flex_image>(const variant_type& t) {
   return variant_is<flexible_type>(t) && (variant_get_ref<flexible_type>(t).get_type() == flex_type_enum::IMAGE);
}

template <>
GL_HOT_INLINE_FLATTEN 
inline bool variant_is<flex_date_time>(const variant_type& t) {
   return variant_is<flexible_type>(t) && (variant_get_ref<flexible_type>(t).get_type() == flex_type_enum::DATETIME);
}

template <>
GL_HOT_INLINE_FLATTEN 
inline bool variant_is<flex_nd_vec>(const variant_type& t) {
   return variant_is<flexible_type>(t) && (variant_get_ref<flexible_type>(t).get_type() == flex_type_enum::ND_VECTOR);
}

template <>
GL_HOT_INLINE_FLATTEN 
inline bool variant_is<flex_undefined>(const variant_type& t) {
   return variant_is<flexible_type>(t) && (variant_get_ref<flexible_type>(t).get_type() == flex_type_enum::UNDEFINED);
}

template <>
GL_HOT_INLINE_FLATTEN 
inline bool variant_is<std::shared_ptr<unity_sgraph_base> >(const variant_type& t) {
   return t.which() == 1;
}

template <>
GL_HOT_INLINE_FLATTEN 
inline bool variant_is<dataframe_t>(const variant_type& t) {
   return t.which() == 2;
}

template <>
GL_HOT_INLINE_FLATTEN 
inline bool variant_is<std::shared_ptr<model_base> >(const variant_type& t) {
   return t.which() == 3;
}

template <>
GL_HOT_INLINE_FLATTEN 
inline bool variant_is<std::shared_ptr<unity_sframe_base> >(const variant_type& t) {
   return t.which() == 4;
}

class gl_sframe; 

template <>
GL_HOT_INLINE_FLATTEN 
inline bool variant_is<gl_sframe>(const variant_type& t) {
   return t.which() == 4;
}

template <>
GL_HOT_INLINE_FLATTEN 
inline bool variant_is<std::shared_ptr<unity_sarray_base> >(const variant_type& t) {
   return t.which() == 5;
}

class gl_sarray; 

template <>
GL_HOT_INLINE_FLATTEN 
inline bool variant_is<gl_sarray>(const variant_type& t) {
   return t.which() == 5;
}

template <>
GL_HOT_INLINE_FLATTEN 
inline bool variant_is<variant_map_type>(const variant_type& t) {
   return t.which() == 6;
}

template <>
GL_HOT_INLINE_FLATTEN 
inline bool variant_is<boost::recursive_wrapper<function_closure_info> >(const variant_type& t) {
   return t.which() == 7;
}

}


#include <unity/lib/variant_converter.hpp>

namespace turi {
/**
 * Stores an arbitrary value in a variant
 */
template <typename T>
inline void variant_set_value(variant_type& v, const T& f) {
  v = variant_converter<typename std::decay<T>::type>().set(f);
}


/**
 * Converts an arbitrary value to a variant.
 * T can be \b alot of possibilities. See the \ref sec_supported_type_list
 * "supported type list" for details.
 */
template <typename T>
inline variant_type to_variant(const T& f) {
  return variant_converter<typename std::decay<T>::type>().set(f);
}

/**
 * Reads an arbitrary type from a variant.
 * T can be \b alot of possibilities. See the \ref sec_supported_type_list
 * "supported type list" for details.
 */
template <typename T>
inline typename std::decay<T>::type variant_get_value(const variant_type& v) {
  return variant_converter<typename std::decay<T>::type>().get(v);
}


// Convenience function to better raise errors when there is a bad value in a
// variant map type.
template <typename T>
static inline typename std::decay<T>::type variant_map_extract(
    const variant_map_type& m, const std::string& name) {

  // Is it in the map?
  auto it = m.find(name);

  if(it == m.end()) {
    auto __raise_error = [&]() GL_COLD_NOINLINE_ERROR {
      std::ostringstream ss;
      ss << "Value for requested key \"" << name << "\" not found in parameter set.";
      std_log_and_throw(std::out_of_range, ss.str());
    };

    __raise_error();
  }

  if(!variant_is<T>(it->second)) {
    auto __raise_error = [&]() GL_COLD_NOINLINE_ERROR {
      std::ostringstream ss;
      ss << "Value for key \"" << name << "\" is of type "
         << full_type_name(it->second) << "; Expected type "
         << full_type_name<T>() << ".";
      std_log_and_throw(std::out_of_range, ss.str());
    };

    __raise_error();
  }

  return variant_get_value<T>(it->second);
}

// Convenience function to better raise errors when there is a bad value in a
// variant map type.  Overload with default value provided.
template <typename T>
static inline typename std::decay<T>::type variant_map_extract(
    const variant_map_type& m, const std::string& name,
    const T& default_value) {

  auto it = m.find(name);

  if(it == m.end()) {
    return default_value;
  }

  try {
    return variant_get_value<T>(it->second);
  } catch(const std::exception& e) {
    std::ostringstream ss;
    ss << "Value for key \"" << name << "\" is of type "
       << full_type_name(it->second) << "; Error converting to required type "
       << full_type_name<T>() << "."
       << " (conversion error: " << e.msg() << ").";

    std_log_and_throw(std::runtime_error, ss.str());
  }

}




} // namespace turi
#include <unity/lib/api/function_closure_info.hpp>
#endif
