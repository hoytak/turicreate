/* Copyright © 2017 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#ifndef TURI_METHOD_WRAPPER_HPP_
#define TURI_METHOD_WRAPPER_HPP_

#include <core/export.hpp>
#include <variant.hpp>


namespace turi {

// Helper for enable if in templates. 
// TODO: move this to a common location
template <bool B> using enable_if_ = typename std::enable_if<B,int>::type;

namespace v2 { 

// How the arguments are bundled up and packaged.
struct argument_pack {
  std::vector<variant_type> ordered_arguments; 
  variant_map_type named_arguments; 
};

// This can become more later on.
struct Parameter {

  Parameter() {}

  // Allow implicit here.
  Parameter(const std::string& n) : name(n) {}

  // Name 
  const std::string name;
}; 


/** Call the function call 
 */
template <typename BaseClass> class method_wrapper { 

  public:
   
   virtual ~method_wrapper(){}

   /** The type of the Class on which our method rests.
    *  
    *  May be void, in which case it's a standalone function.
    */
   typedef BaseClass class_type;  

   /// TODO: overload for const and no-class.
   virtual variant_type call(BaseClass* C, const argument_pack& args) const = 0;
   
   /// Returns the paramater type 
   virtual std::type_info parameter_type(size_t n) const = 0;

   /// Returns the parameter info for a particular 
   inline const Parameter& parameter_info(size_t n) const { 
     return m_parameter_list.at(n);
   }

   /// Returns the name of the parameter 
   inline const std::string& parameter_name(size_t n) const {
     return parameter_info(n).name; 
   }

   /** Factory method.
    *
    *  Call this method to create the function call wrapper around 
    *  the method.  One of these overloads will 
    *
    */
   template <typename Class, typename RetType, typename... FuncParams> 
   static std::shared_ptr<method_wrapper> 
   create(RetType(*Class::method)(FuncArgs...), 
          const std::vector<Parameter>& param_defs);


  protected:

   // To be called only from the instantiating class
   method_wrapper(const std::vector<Parameter>& _parameter_list) 
     : m_parameter_list(_parameter_list)
   {}
   

   // Information about the function / method 
   std::vector<Parameter> m_parameter_list;
 };




namespace function_call_impl { 

/////////////////////////////////////////////////////////////////////////////////
//
//  Implementation details of the above.
//  

/** Child class for resolving the arguments passed into a function call.  
 *
 *  This class is mainly a container to define the types present 
 *  during the recursive parameter expansion stage.  
 *
 */ 
template <typename Class, typename BaseClass, 
          typename RetType, typename... FuncParams> 
  class method_wrapper_impl :: public method_wrapper<Class> {

 private:

  /// The number of parameters required for the function.
  static constexpr size_t N = sizeof...(FuncParams);
  
  /// The type of the function pointer we're calling.
  typedef RetType (Class::*method_type)(FuncParams...);

  /// Function pointer to the method we're calling.
  method_type m_method;
  
  /** Constructor.  
   */
  method_wrapper_impl(
      const std::string& _name, 
      const std::vector<Parameter>& _parameter_list,
      method_type method)
    : method_wrapper(_name, _parameter_list)
    , m_method(method)
  {
    if(param_defs.size() != N) {
      // TODO: Helpful error message.
       throw std::string("Mismatch in the number of arguments.");
    }

    // TODO: ensure that the parameter names are all unique.
  }
  
  //////////////////////////////////////////////////////////////////////////////
  //
  //  Calling methods

  /** A handy way to refer to the type of the nth argument.
   */
  template <int idx>
  struct nth_param_type { 
    typedef typename std::tuple_element<idx, std::tuple<FuncParams...> >::type raw_type;
    typedef typename std::decay<raw_type>::type type;
  };
  
  /// Container for passing the calling arguments around after unpacking from argument_list
  typedef std::array<const variant_type*, N> arg_v_type;

  /// Recursively case for argument expansion.
  template <int arg_idx, typename... Expanded, enable_if_<arg_idx != N> = 0>
    variant_type _call(Class* inst, const arg_v_type& arg_v, const Expanded&... args) {

      // TODO: Separate out the case where the unpacking can be done by 
      // reference.
      typedef typename nth_param_type<arg_idx>::type arg_type;

      // TODO: Add intelligent error messages here on failure
      arg_type next_arg = from_variant<arg_type>(*(arg_v[arg_idx]));
     
      // Call the next unpacking routine. 
      return _call<arg_idx+1, ExpandedArgs..., arg_type>(inst, arg_set, args..., next_arg);
    }
  }; 
  
  // Stopping case of expansion -- we've unpacked and translated all the 
  // arguments, now it's time to actually call the method. 
  template <int arg_idx, typename... Expanded, enable_if_<arg_idx == N> = 0>
    variant_type _call(Class* inst, const arg_v_type& arg_v, const Expanded&... args) {
      
       return to_variant( (inst->*m_method)(args...) );
    }
  };

  /**  Main calling function.
   */
  variant_type call(BaseClass* _inst, const argument_pack& args) const override { 
    
    Class* inst = std::dynamic_cast<Class*>(_inst);
    
    arg_v_type arg_v; 

    size_t n_ordered = args.ordered_arguments.size();
    for(size_t i = 0; i < n_ordered; ++i) { 
      arg_v[i] = &args.ordered_arguments[i];
    }

    // TODO: check if more ordered arguments given than are 
    // possible here.
    size_t used_counter = n_ordered;
    for(size_t i = n_ordered; i < N; ++i) {
      auto it = args.named_arguments.find(parameter_list[i].name);
      if(it == args.named_arguments.end()) {
        // TODO: intelligent error message.
        throw std::string("Missing argument.");
      } else {
        arg_v[i] = &(it->second);
        ++used_counter; 
      }
    }

    // TODO: check that all the arguments have been used up.  If not,
    // generate a good error message.
   
    
    // Now that the argument list arg_v is filled out, we can call the 
    // recursive calling function and return the value.
    return _call<0>(inst, arg_v);
  }

}; 

   
/** Implementation of the factory methods above.
 */
template <typename BaseClass>
template <typename Class, typename RetType, typename... FuncParams>
std::shared_ptr<method_wrapper<BaseClass> > method_wrapper<BaseClass>::create(
    const std::string& method_name,
    (RetType)(*Class::method)(FuncArgs...), 
    const std::vector<Parameter>& param_defs) {

  return std::make_shared<method_wrapper_impl<Class, BaseClass, RetType, FuncParams...> >
        (method_name, param_defs, method);
 };

}
}
}
