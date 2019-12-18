#include <string>
#include <unordered_map>
#include <type_traits>
#include <array>
#include <iostream>
#include <vector>
#include <tuple>

typedef int variant_type;

// Stubs here
variant_type to_variant(int x) { 
  return x; 
} 

template <typename T> 
T from_variant(const variant_type& v) { 
  return T(v); 
} 


// This can become more later on.
struct Parameter {
  std::string name; 
}; 

struct argument_pack {
  std::vector<variant_type> ordered_arguments; 
  std::unordered_map<std::string, variant_type> named_arguments; 
};





// Outer class. 
template <typename Class, typename RetType, typename... FuncParams> 
  struct ArgumentResolver {

  // The type of the function pointer we're calling.
  typedef RetType (Class::*method_type)(FuncParams...);

  // A handy way to refer to the type of the nth argument type.
  template <int idx>
  struct nth_param_type { 
    typedef typename std::decay<typename std::tuple_element<idx, std::tuple<FuncParams...> >::type>::type type;
  };
  
  static constexpr size_t N = sizeof...(FuncParams);

  const std::string method_name;
  std::array<Parameter, N> parameter_list;
  method_type method;
  
  /** Constructor.  Store all the specific information 
   */
  ArgumentResolver(const std::string& _name, 
      method_type _method, 
      const std::vector<Parameter>& param_defs)
    : method_name(_name)
    , method(_method)
  {
    if(param_defs.size() != N) {
      // TODO: make robust
       throw std::string("Mismatch in the number of arguments.");
    }

    // TODO: ensure that the parameter names are all unique.


    std::copy(param_defs.begin(), param_defs.end(), parameter_list.begin()); 
  }
  
  ///////////////////////////////////
 
  struct call_info { 
    const ArgumentResolver* res; 
    Class* inst;
    std::array<const variant_type*, N> arg_list; 
  }; 

  template <int arg_idx, typename... ExpandedArgs>
    struct __caller {
    typedef typename nth_param_type<arg_idx>::type arg_type;

      static inline variant_type call(
          const call_info& ci, const ExpandedArgs&... args) {

      // TODO: add intelligent error messages here on failure
      arg_type next_arg = from_variant<arg_type>(*(ci.arg_list[arg_idx]));
      
      return __caller<arg_idx+1, ExpandedArgs..., arg_type>::call(ci, args..., next_arg);
    }
  }; 

  template <typename... ExpandedArgs> struct __caller<N, ExpandedArgs...> {
    static inline variant_type call(
        const call_info& ci, const ExpandedArgs&... args) {
      
       return to_variant( (ci.inst->*(ci.res->method))(args...) );
    }
  };

  /**  Main calling function.
   */
  variant_type call(Class* inst, const argument_pack& args) const { 
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
}; 


/** Create the function wrapper
 */
template <typename Class, typename RetType, typename... FuncParams>
std::function<variant_type(Class*, const argument_pack&)>
  create_function_wrapper(
      const std::string& method_name,
      RetType (Class::*method)(FuncParams...), 
      const std::vector<Parameter>& param_defs) {
      
    auto resolver = std::make_shared<ArgumentResolver<Class, RetType, FuncParams...> >
        (method_name, method, param_defs);

    return [=](Class* inst, const argument_pack& args) {
      return resolver->call(inst, args);
    };
  }



class A { 
 public:
  int f(int x, int y) { 
    return x + y; 
  }
};

int main(int argc, char** argv) {

  std::function<variant_type(A*, const argument_pack&)> caller 
    = create_function_wrapper("f", &A::f, std::vector<Parameter>{Parameter{"x"}, Parameter{"y"}});


  argument_pack args; 
  args.ordered_arguments = {1, 2};

  A a; 

  int result = caller(&a, args);

  std::cout << "result 1 = " << result << std::endl; 

  return 0; 
}

