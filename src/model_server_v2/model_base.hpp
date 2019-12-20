/* Copyright © 2019 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#ifndef TURI_MODEL_BASE_V2_HPP
#define TURI_MODEL_BASE_V2_HPP

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <core/export.hpp>
#include <variant.hpp>

namespace turi { 
  namespace v2 {  


/**
 * The base class from which all new models must inherit.
 *
 * This class defines a generic object interface, listing properties and
 * callable methods, so that instances can be naturally wrapped and exposed to
 * other languages, such as Python.
 *
 * Subclasses that wish to support saving and loading should also override the
 * save_impl, load_version, and get_version functions below.
 */
class EXPORT model_base { 
 public:
 
  model_base();

  virtual ~model_base();

  // These public member functions define the communication between model_base
  // instances and the unity runtime. Subclasses define the behavior of their
  // instances using the protected interface below.

  /**
   * Returns the name of the toolkit class, as exposed to client code. For
   * example, the Python proxy for this instance will have a type with this
   * name.
   *
   */
  virtual const char* name() = 0;
  
  
  /** Initialize base configuration of class.
   *
   *  This is called by the model server to create the method 
   *  registry and  for this class type.  Future instantiations of 
   *  this model use the cached method registry. 
   *
   */
  virtual void configure() = 0;

  /** Sets up the class given the options present.  
   *  
   *  TODO: implement all of this. 
   */
  virtual void setup(const variant_map_type& options) {
  //   option_manager.update(options); 
  } 


  /** Call one of the methods registered using the configure() method above.  
   *
   */
  // TODO: Overload const
  variant_type call_method(const std::string& name, const argument_pack& args);

  /** Call one of the methods here.  
   */
  // TODO: Overload const
  template <typename Args...> 
    variant_type call_method(const std::string& name, const Args&... args); 

  /** Register a method that can be called by name using the registry above.  
   *
   */
  // TODO: Overload const functions
   template <typename Class, typename RetType, typename... FuncParams>
    void register_method(
        const std::string& name, 
        RetType (*Class::method)(FuncArgs...),
        const std::vector<Parameter>& parameter_list);


   // TODO: add back in load and save routines.

 private: 
   std::shared_ptr<method_registry<model_base> > m_registry;

};

////////////////////////////////////////////////////////////////////////////////
//
//  Implementations of above functions.

template <typename Args...> 
  variant_type model_base::call_method(const std::string& name, const Args&... args) {

    return m_registry->call_method(this, name, args...);
}

template <typename Class, typename RetType, typename... FuncParams>
void model_base::register_method(
    const std::string& name, 
    RetType (*Class::method)(FuncArgs...),
    const std::vector<Parameter>& parameter_list) { 

  m_registry->register_method(name, method, parameter_list);
}





}}
