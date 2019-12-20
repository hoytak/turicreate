/* Copyright © 2017 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#ifndef TURI_MODEL_SERVER_V2_REGSTRATION_HPP_
#define TURI_MODEL_SERVER_V2_REGSTRATION_HPP_ 


#include <atomic>
#include <mutex>
#include <array>
#include <model_server_v2/model_server.hpp> 

namespace turi { 
  namespace v2 {

// Pointer to the last 
extern const std::type_info* __previous_model_type_info;

// A helper class to use a static initializer to do a lightweight registration 
// of class loading at library load time.  Intended to be used as a component of 
// the registration macro.   
class __model_server_static_class_registration_hook {
  public:
    inline __model_server_static_class_registration_hook(
       const std::type_info* model_type_info, model_server_impl::registration_callback f) {

      // Quick check to cut out duplicate registrations.  This can 
      // happen, e.g. if the class or the function macros appear in a header,  
      // which is fine and something we are designed to handle.  
      // However, this means that multiple registration calls can occur for the same 
      // class, and this quickly filters those registrations out. 
      if(__previous_model_type_info != nullptr
          && (*__previous_model_type_info) != (*model_type_info)) {
        __previous_model_type_info = model_type_info;
        model_server().add_registration_callback(f);
      }
    }
};



#define REGISTER_MODEL(model) \
  static void register_##name(model_server_impl& server) { \
    server.template register_new_model<model>(); \
  } \
  \
  static turi::v2::__model_server_static_class_registration_hook \
    __register_##model##_hook(&typeid(model), register_##model)



extern const void* __previous_registerd_function_ptr;

// A helper class to use a static initializer to do a lightweight registration 
// of class loading at library load time.  Intended to be used as a component of 
// the 
class __model_server_static_function_registration_hook {
  public:
    inline __model_server_static_function_registration_hook(
       const void* function_ptr, model_server_impl::registration_callback f) {

      // Quick check to cut out duplicate registrations.  This can 
      // happen, e.g. if the class or the function macros appear in a header,  
      // which is fine and something we are designed to handle.  
      // However, this means that multiple registration calls can occur for the same 
      // class, and this quickly filters those registrations out. 
      if(__previous_registerd_function_ptr != nullptr
          && __previous_registerd_function_ptr != function_ptr) {

        __previous_registerd_function_ptr = function_ptr;

        model_server().add_registration_callback(f);
      }
    }
};



#define REGISTER_NAMED_FUNCTION(name, function, ...) \
\
  static void register_function_##function(model_server_impl& server) {\
    server.register_new_function(name, function, {__VA_ARGS__});\
  } \
   __model_server_static_function_registration_hook \
__register_function_##function##_hook(register_function_##function)

#define REGISTER_FUNCTION(function, ...) \
  REGISTER_NAMED_FUNCTION(#function, function, __VA_ARGS__)


}
}

#endif
