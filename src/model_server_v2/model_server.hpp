/* Copyright © 2017 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#ifndef TURI_MODEL_SERVER_HPP
#define TURI_MODEL_SERVER_HPP

#include <core/export.hpp>
#include <core/util/code_optimizations.hpp>

#define TC_MAX_REGISTERED_CALLBACKS 512

namespace turi {
namespace v2 {

class model_server_impl;


/** Returns the singleton version of the model server. 
 *
 */
EXPORT model_server_impl& model_server();


EXPORT class model_server_impl { 
  private:

    // Disable instantiation outside of the global instance.
    model_server_impl(){}
    friend model_server_impl model_server();

    // TODO: Explicitly disable copying, etc.

 public: 
   
    ////////////////////////////////////////////////////////////////////////////
    // Calling models. 

    /** Instantiate a previously registered model by name.
     *
     */
    std::shared_ptr<model_base> create_model(const std::string& model_name) const;
   

    /** Instantiate a previously registered model by type.
     */
    template <typename ModelType> 
      std::shared_ptr<ModelType> create_model() const; 


    /** Call a previously registered function.
     */
    variant_type call_function(
        const std::string& function_name,
        const std::vector<variant_type>& ordered_arguments,
        const std::unordered_map<std::string, variant_type>& named_arguments) const; 

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


  //////////////////////////////////////////////////////////////////////////////
  // Registered model lookups. 
  typedef std::function<std::shared_ptr<model_base>()> model_creation_function; 
  std::unordered_map<std::string, model_creation_function> m_model_by_name;




  ///////////////////////////////////////////////////////////////////////////////
  // TODO: Registered function lookups.
  //

  /** Lock to ensure that model registration is queued correctly.
   */
  std::mutex m_model_registration_lock;

  /** An intermediate buffer of registration callbacks.  
   *
   *  These queues are used on library load to register callback functions, which
   *  are then processed when any model is requested to ensure that library loading
   *  is done efficiently.  check_registered_callback_queue() should be called 
   *  before any lookups are done to ensure that all possible lookups have been 
   *  registered.
   *
   */
  static constexpr size_t max_registered_callbacks() { return 512; } 
  std::array<class_registration_callback, max_registered_callbacks()> m_registration_callback_list; 
  std::atomic<size_t> m_callback_pushback_index;
  std::atomic<size_t> m_callback_last_processed_index;
  const std::type_info m_previous_model_registration_type = nullptr;


  /** Process the registered callbacks.
   *
   *  First performs a fast inline check to see if it's needed, so 
   *  this function can be called easily.
   */
  inline void check_registered_callback_queue();

  /** Does the work of registering things with the callbacks. 
   */
  void _process_registered_callbacks_internal();
};

/////////////////////////////////////////////////////////////////////////////////
//
// Implementations of inline functions for the model server class
//

/** Fast inline check 
 */ 
inline void model_server_impl::check_registered_callback_queue() { 
  if(m_callback_last_processed_index < m_callback_pushback_index) { 
    _process_registered_callbacks_internal(); 
  }
}

/** Add the callback to the registration function. 
 *
 *  This works by putting the callback function into a round-robin queue to avoid
 *  potential allocations or deallocations during library load time and to 
 *  preserve thread safety.
 */
inline void model_server_impl::add_registration_callback(
  model_server_impl::class_registration_callback callback) {

  size_t insert_index_raw = (m_callback_pushback_index++);
  
  do {
    // Check to make sure this can be safely inserted.
    size_t processed_index_raw = m_callback_last_processed_index;

    // Check to make sure we aren't so far behind the number of actually 
    // registered callbacks that we're out of space. 
    if(processed_index_raw + max_registered_callbacks() > insert_index_raw) { 
      break; 
    } else {
      // This will process the next block of insertions.
      _process_registered_callbacks_internal();
    }

  } while(true);

  size_t insert_index = insert_index_raw % max_registered_callbacks();

  ASSERT_TRUE(m_registration_callback_list[insert_index] == nullptr);
  m_registration_callback_list[insert_index] = callback;
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

   // TODO: Implement this. 
   //
   // This would be done on top of the method registration mechanisms, using 
   // void for the class and making the necessary adjustments in the 
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
template <typename ModelClass> void model_server_impl::register_new_model() {

  const std::string& name = ModelClass().name(); // We assume these are cheap to construct. 

  model_creation_function mcf = [=](){ return this->create_model<ModelClass>(); };

  m_model_by_name.insert({name, mcf});
}

    

/** Instantiate a previously registered model by name.
 */
std::shared_ptr<model_base> create_model(const std::string& model_name) const {

  // Make sure there aren't new models waiting on the horizon.
  check_registered_callback_queue();

  auto it = m_model_by_name.find(model_name); 

  if(it == m_model_by_name.end()) {
    // TODO: make this more informative.
    throw std::invalid_argument("Model not recognized.");
  }

  return it->second();
}


/** Instantiate a previously registered model by type.
 */
template <typename ModelType> 
  std::shared_ptr<ModelType> model_server_impl::create_model() const {

  // Make sure there aren't new models waiting on the horizon.
  check_registered_callback_queue();

  // Create the model and configure it.  
  auto ret = std::make_shared<ModelType>(); 
  ret->configure();
  return; 
}



} // End turi namespace
}
#endif
