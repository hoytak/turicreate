/* Copyright © 2017 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#ifndef TURI_FUNCTION_CALL_WRAPPER_HPP_
#define TURI_FUNCTION_CALL_WRAPPER_HPP_

#include <core/export.hpp>
#include <variant.hpp>
#include <unordered_map.hpp>
#include <method_wrapper.hpp>

namespace turi {
namespace v2 { 


/** Manager all the methods in a given class / model. 
 *
 *  This class exists to manage a collection of methods associated with a given 
 *  class.  It provides an interface to call previously registered methods on 
 *  this class by name, along with helpful error messages if the call is wrong. 
 *
 *  If the BaseClass is void, it provides a registry for standalone functions.
 *  TODO: implement this. 
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
        // TODO: Expand these exceptions to make them informative.
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
        // TODO: Expand these exceptions to make them informative.
        std::rethrow_exception(e); 
      }
   }


   private:

     template <size_t i, size_t N, typename Tuple, enable_if_<i != N> = 0>
     inline void _arg_unpack(std::vector<variant_type>& dest, const Tuple& t) { 
        dest[i] = to_variant(std::get<i>(t)); 
        _arg_unpack<i + 1, N>(dest, t);
     }

     template <size_t i, size_t N, typename Tuple, enable_if_<i == N> = 0>
     inline void _arg_unpack(std::vector<variant_type>& dest, const Tuple& t) { 
     }


   public: 
     template <typename... Args>
     variant_type call_method(const BaseClass* inst, const std::string& name, const Args&... args) { 

     argument_list arg_list; 
     arg_list.ordered_arguments.resize(sizeof(args...));

     _arg_unpack<0, sizeof(args...)>(arg_list.ordered_arguments, std::make_tuple<const Args&...>(args));

     return call_method(inst, name, arg_list); 
   }
  
  private:

   std::string m_class_name; 

   std::unordered_map<std::string, std::shared_ptr<function_call_wrapper<BaseClass> > >
     m_method_lookup;
};

}
}
