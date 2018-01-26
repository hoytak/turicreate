/* Copyright © 2017 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#include <fileio/general_fstream.hpp>
#include <unity/lib/variant_deep_serialize.hpp>
#include <unity/toolkits/ml_model/sdk_model.hpp>
#include <unity/lib/unity_global.hpp>
#include <unity/lib/unity_global_singleton.hpp>
#include <unity/lib/toolkit_function_macros.hpp>

namespace turi {
namespace sdk_model {

/**
 * Get the default option dictionary.
 */
std::map<std::string, variant_type> _toolkits_sdk_get_default_options(
                             std::string model_name) {

  std::shared_ptr<sdk_model_base> model
    = std::dynamic_pointer_cast<sdk_model_base>(get_unity_global_singleton()
                                  ->get_toolkit_class_registry()
                                  ->get_toolkit_class(model_name));
  if (model == nullptr) {
    log_and_throw("Internal Error:" + model_name + " is not defined.");
  }

  // Empty options
  std::map<std::string, flexible_type> empty_opts =
                                      std::map<std::string, flexible_type>();
  model->init_options(empty_opts);
  std::vector<option_handling::option_info> options = model->get_option_info();

  // Return the options as JSON
  std::map<std::string, variant_type> results;
  for (const auto& opt : options) {
    results[opt.name] = opt.to_dictionary();
  }
  return results;
}

/**
 * List all the keys that are present in the state.
 */
std::vector<std::string> sdk_model_base::list_fields(){
  std::vector<std::string> ret;
  for (const auto& kvp: state){
    ret.push_back(kvp.first);
  }
  return ret;
}


/**
 * Returns the value of an option. Throws an error if the option does not exist.
 */
const flexible_type& sdk_model_base::get_option_value(const std::string& name) const{
  return options.value(name);
}


/**
 * Get current options.
 */
const std::map<std::string, flexible_type>& sdk_model_base::get_current_options()
  const{
  return options.current_option_values();
}

/**
 * Get default options.
 */
std::map<std::string, flexible_type> sdk_model_base::get_default_options()
  const{
  return options.get_default_options();
}

/**
 * Get an option. Return an error if the option does not exist.
 * \note This function will also modify the model "state"
 */
void sdk_model_base::set_options(const std::map<std::string, flexible_type>& _options){
  if (options.current_option_values().size() == 0) {
    log_and_throw("Model options have not been initialized. This is required before calling set_options.");
  }
  for(const auto & kvp: _options) {
    options.set_option(kvp.first, kvp.second);
    add_or_update_state({{kvp.first, kvp.second}});
  }
}

/**
 * Return the "state" map.
 */
std::map<std::string, variant_type> sdk_model_base::get_state() const{
  return state;
}

/**
 * Get value in the given dictionary.
 */
variant_type sdk_model_base::get_value_from_state(std::string key){

  // Field does not exist
  if(state.count(key) == 0){
    std::stringstream ss;
    ss << "Field '" << key << "' does not exist. Use list_fields() for a "
       << "list of fields that can be queried." << std::endl;
    log_and_throw(ss.str());

  // Field exists: Cast the object to the variant type and return!
  } else {
    return state[key];
  }
}


/**
 * Check if trained.
 *
 * \note For now, trained is always returned to True. This will change
 * when we move to async models.
 */
bool sdk_model_base::is_trained() const{
  return true;
}

/**
 * Append the key value store of the model.
 */
void sdk_model_base::add_or_update_state(const std::map<std::string,
    variant_type>& dict){
  for(const auto & kvp: dict){
    state[kvp.first] = kvp.second;
  }
}

// Returns the option information struct for each of the set
// parameters.
const std::vector<option_handling::option_info>&
                                      sdk_model_base::get_option_info() const {
  return options.get_option_info();
}

BEGIN_FUNCTION_REGISTRATION
REGISTER_FUNCTION(_toolkits_sdk_get_default_options, "model_name")
END_FUNCTION_REGISTRATION

} // namespace sdk_model
} // namespace turi
