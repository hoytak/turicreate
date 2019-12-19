/* Copyright © 2017 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#ifndef TURI_FUNCTION_CALL_WRAPPER_HPP_
#define TURI_FUNCTION_CALL_WRAPPER_HPP_

#include <core/export.hpp>
#include <variant.hpp>

namespace turi {

namespace function_wrapper {

// How the arguments are bundled up and packaged.
struct argument_pack {
  std::vector<variant_type> ordered_arguments; 
  std::unordered_map<std::string, variant_type> named_arguments; 
};

// This can become more later on.
struct Parameter {
  std::string name; 
}; 


/** Call the function call 
 */
template <typename BaseClass> class function_call_wrapper { 

  public:

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
   static std::shared_ptr<function_call_wrapper> 
   create((RetType)(*Class::method)(FuncArgs...), 
          const std::vector<Parameter>& param_defs);


  protected:


   // To be called only from the instantiating class
   function_call_wrapper(const std::vector<Parameter>& _parameter_list) 
     : m_parameter_list(_parameter_list)
   {}
   
   virtual ~function_call_wrapper(){}

   // Information about the function / method 
   std::vector<Parameter> m_parameter_list;
 };



/** Manager all the methods in a given class / model. 
 *
 *  This class exists mainly to manage 
 *
 *  The main function of this class is to provide wrappers around the 
 *  calling functions.  These wrappers mainly provide helpful error messages 
 *  that give the right context for the calling.
 *
 */
template <typename BaseClass>
class method_registry {
  public:

   method_registry()
     : m_class_name()
    {}

   method_registry(const std::string& _name)
     : m_class_name(_name)
    {}

   // TODO: overload for const.
   template <typename Class, typename RetType, typename... FuncParams>
    void register_method(
        const std::string& name, 
        (RetType)(*Class::method)(FuncArgs...),
        const std::vector<Parameter>& parameter_list) {

      try { 

        auto wrapper = function_call_wrapper<BaseClass>::create(method, parameter_list);
      
        m_method_lookup[name] = wrapper; 
      } catch(std::exception e) {
        // TODO: Expand these exceptions to make them coherant.
        std::rethrow_exception(e); 
      }
    }

   // Lookup a call function information.
   std::shared_ptr<function_call_wrapper<BaseClass> > lookup(const std::string& name) const { 

     // TODO: proper error message here
     return m_method_lookup.at(name);
   }

   variant_type call_method(const BaseClass* inst, const std::string& name, 
       const argument_list& arguments) const { 

      try { 
         return lookup(name)->call(inst, arguments); 

      } catch(std::exception e) {
        // TODO: Expand these exceptions to make them coherant.
        std::rethrow_exception(e); 
      }
   }

  private:

   std::string m_class_name; 

   std::unordered_map<std::string, std::shared_ptr<function_call_wrapper<BaseClass> > >
     m_method_lookup;


};





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
  class function_call_wrapper_impl :: public function_call_wrapper<Class> {

 private:

  /// The number of parameters required for the function.
  static constexpr size_t N = sizeof...(FuncParams);
  
  /// The type of the function pointer we're calling.
  typedef RetType (Class::*method_type)(FuncParams...);

  /// Function pointer to the method we're calling.
  method_type m_method;
  
  /** Constructor.  
   */
  function_call_wrapper_impl(
      const std::string& _name, 
      const std::vector<Parameter>& _parameter_list,
      method_type method)
    : function_call_wrapper(_name, _parameter_list)
    , m_method(method)
  {
    if(param_defs.size() != N) {
      // TODO: Helpful error message.
       throw std::string("Mismatch in the number of arguments.");
    }

    // TODO: ensure that the parameter names are all unique.
  }
  
  //////////////////////////////////////////////////////////////////////////////

  // A simple helper struct to carry the information about a particular call. 
  struct call_info { 
    const function_call_wrapper* res; 
    Class* inst;
    std::array<const variant_type*, N> arg_list; 
  }; 
  
  // A handy way to refer to the type of the nth argument.
  template <int idx>
  struct nth_param_type { 
    typedef typename std::tuple_element<idx, std::tuple<FuncParams...> >::type raw_type;
    typedef typename std::decay<raw_type>::type type;
  };

  // Recursively expand the arguments.
  template <int arg_idx, typename... ExpandedArgs>
    struct __caller {
    typedef typename nth_param_type<arg_idx>::type arg_type;

      // TODO: Separate out the cases where the unpacking can be done by 
      // reference.
      static inline variant_type call(
          const call_info& ci, const ExpandedArgs&... args) {

      // TODO: Add intelligent error messages here on failure
      arg_type next_arg = from_variant<arg_type>(*(ci.arg_list[arg_idx]));
     
      // Call the next unpacking routine. 
      return __caller<arg_idx+1, ExpandedArgs..., arg_type>::call(ci, args..., next_arg);
    }
  }; 

  // Stopping case of expansion -- we've unpacked and translated all the 
  // arguments, now it's time to actually call the method. 
  template <typename... ExpandedArgs> struct __caller<N, ExpandedArgs...> {
    static inline variant_type call(
        const call_info& ci, const ExpandedArgs&... args) {
      
       return to_variant( (ci.inst->*(ci.res->method))(args...) );
    }
  };

  /**  Main calling function.
   */
  variant_type _call(Class* inst, const argument_pack& args) const { 
    call_info ci; 
    ci.res = this; 
    ci.inst = inst; 

    size_t n_ordered = args.ordered_arguments.size();
    for(size_t i = 0; i < n_ordered; ++i) { 
      ci.arg_list[i] = &args.ordered_arguments[i];
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
        ci.arg_list[i] = &(it->second);
        ++used_counter; 
      }
    }

    // TODO: check that all the arguments have been used up.  If not, go
    // back and generate a good error message.
   
    
    // Okay, now that the argument list is filled out, we can call the 
    // function with it and return the value.
    return __caller<0>::call(ci);
  }

  // Because the base class is not present for many of the interfaces in the 
  variant_type call(BaseClass* inst, const argument_pack& args) const override {
    return _call(std::dynamic_cast<Class*>(inst), args);
  }

}; 

   
/** Implementation of the factory methods above.
 */
template <typename BaseClass>
template <typename Class, typename RetType, typename... FuncParams>
std::shared_ptr<function_call_wrapper<BaseClass> > function_call_wrapper<BaseClass>::create(
    const std::string& method_name,
    (RetType)(*Class::method)(FuncArgs...), 
    const std::vector<Parameter>& param_defs) {

  return std::make_shared<function_call_wrapper_impl<Class, BaseClass, RetType, FuncParams...> >
        (method_name, param_defs, method);
 }; 

}

