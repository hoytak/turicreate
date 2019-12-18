/* Copyright © 2017 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#ifndef TURI_MODEL_SERVER_HPP
#define TURI_MODEL_SERVER_HPP

#include <core/export.hpp>

#define TC_MAX_REGISTERED_CALLBACKS 512

namespace turi {

class model_server;


// Returns a singleton instance of the model server when called.  In a given
// process, there can only be one instance of the model server.
EXPORT model_server_impl& model_server();









EXPORT class model_server_impl { 
 public: 


   /** 
    *
    *
    *
    */ 





 public:
 
   
   /** Registration of a function.
    *
    *  Registers a new function that can be called through the call_function
    *  call above.   
    *    
    *  \param name     The name of the function.
    *  \param function A pointer to the function itself.
    *  \param argument_names A container giving the list of argument names in string form.
    */
   template <typename FuncReturn, typename... FuncArgs, typename FuncArgNameList> 
     void register_new_function(
       const std::string& name, (FuncReturn)(*function)(FuncArgs...), const FuncArgNameList& argument_names);


   /** Registration of new models.
    *
    *  A model is registered through a call to register new model, which 
    *  instantiates it and populates the required options and method call 
    *  lookups.  Copies of these options and method call lookups are stored 
    *  internally in a registry here so new models can be instantiated quickly.
    *
    *   
    *  The new model's name() method provides the name of the model being 
    *  registered.
    *  
    *  This method can be called at any point.
    *
    */
   template <typename ModelClass> void register_new_model();


   /** Fast on-load model registration. 
    *
    *  The callbacks below provide a fast method for registering new models 
    *  on library load time.  This works by first registering a callback  
    *  using a simple callback function.  
    *
    */
  typedef void (*_registration_callback)(model_server_impl&);
   
  /** Register a callback function to be processed when a model is served.
   *
   *  Function is reentrant and fast enough to be called from a static initializer.
   */
  inline void add_registration_callback(class_registration_callback callback) TC_HOT_INLINE_FLATTEN;
  
 private:

  std::mutex _model_registration_lock;




  /** An intermediate buffer of registration callbacks.  
   *
   *  These queues are used on library load to register callback functions, which
   *  are then processed when any model is requested to ensure that library loading
   *  is done efficiently.  check_registered_callback_queue() should be called 
   *  before any lookups are done to ensure that all possible lookups have been 
   *  registered.
   */ 
  std::array<class_registration_callback, TC_MAX_REGISTERED_CALLBACKS> registration_callback_list; 
  std::atomic<size_t> _callback_pushback_index;
  std::atomic<size_t> _callback_last_processed_index;
  const std::type_info* previous_model_registration_type = nullptr;


  /** Process the registered callbacks.
   *
   *  First performs a fast inline check to see if it's needed, so 
   *  this function can be called easily.
   */
  inline void check_registered_callback_queue();


  /** Does the work of registering things with the callbacks. 
   */
  void _process_registered_callbacks_internal(); 

  /** Does the work of registering things with the callbacks. 
   */
  void _process_registered_callbacks_internal(); 


};


/** Implementations of inline functions for the model server class
 *
 */ 

inline void model_server_impl::check_registered_callback_queue() { 
  if(_callback_last_processed_index < _callback_pushback_index) { 
    _process_registered_callbacks_internal(); 
  }
}


inline void model_server_impl::add_registration_callback(
    model_server_impl::class_registration_callback callback) {

    size_t insert_index_raw = (_callback_pushback_index++);
    
    do {
      // Check to make sure this can be safely inserted.
      size_t processed_index_raw = _callback_last_processed_index;

      // We aren't so far behind the number of re
      if(processed_index_raw + TC_MAX_REGISTERED_CALLBACKS > insert_index_raw) { 
        break; 
      } else {
        // This will process the next block of insertions.
        _process_registered_callbacks_internal();
      }

    } while(true);

    size_t insert_index = insert_index_raw % TC_MAX_REGISTERED_CALLBACKS;

    ASSERT_TRUE(registration_callback_list[insert_index] == nullptr);
    registration_callback_list[insert_index] = callback;
  }


   /** Registration of a function.
    *
    *  Registers a new function that can be called through the call_function
    *  call above.   
    *    
    *  \param name     The name of the function.
    *  \param function A pointer to the function itself.
    *  \param argument_names A container giving the list of argument names in string form.
    */
   template <typename FuncReturn, typename... FuncArgs, typename FuncArgNameList> 
     void model_server_impl::register_new_function(
       const std::string& name, (FuncReturn)(*function)(FuncArgs...), const FuncArgNameList& argument_names) { 

       // First, get the global function registry and see if that's good.
       






     }


   /** Registration of new models.
    *
    *  A model is registered through a call to register new model, which 
    *  instantiates it and populates the required options and method call 
    *  lookups.  Copies of these options and method call lookups are stored 
    *  internally in a registry here so new models can be instantiated quickly.
    *
    *   
    *  The new model's name() method provides the name of the model being 
    *  registered.
    *  
    *  This method can be called at any point.
    *
    */
   template <typename ModelClass> void register_new_model();












} // End turi namespace
#endif
