#include <model_server_v2/model_base.hpp>

namespace turi {
namespace v2 {

model_base::model_base() 
  : m_registry(new method_registry<model_base>())
{}


variant_type model_base::call_method(const std::string& name, const argument_pack& args) { 
  return m_registry->call_method(this, name, args);
}

}
}

