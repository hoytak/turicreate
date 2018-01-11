/* Copyright © 2017 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#ifndef TURI_UNITY_SDK_MODEL_HPP
#define TURI_UNITY_SDK_MODEL_HPP

#include <export.hpp>

#include <unity/lib/variant.hpp>
#include <unity/lib/unity_base_types.hpp>
#include <unity/lib/toolkit_util.hpp>
#include <unity/lib/toolkit_function_specification.hpp>

#include <unity/lib/toolkit_class_base.hpp>
#include <unity/lib/api/client_base_types.hpp>
#include <unity/toolkits/options/option_manager.hpp>


namespace turi {
namespace sdk_model {

/**
 * Obtains the registration for the toolkit.
 */
std::vector<toolkit_function_specification> get_toolkit_function_registration();

/**
 * Call the default options using a registered model.
 *
 * \param[in] name  Name of the model registered in the class.
 */
std::map<std::string, variant_type> _toolkits_get_default_options(
                       std::string model_name);

/**
 * sdk_model base class interface.
 * ---------------------------------------
 *
 * Base class for handling machine learning models written using the SDK.
 *
 * This is a temporary interface and will be replaced by an updated
 * version of sdk_model_base.
 *
 * TODO: Port over ml_model_base to sdk_model_base
 *
 */
class EXPORT sdk_model_base: public toolkit_class_base {

 public:

  static constexpr size_t SDK_MODEL_BASE_VERSION = 0;

  // virtual destructor
  virtual ~sdk_model_base() { }

  /**
   * Returns the name of the model.
   *
   * \returns Name of the model.
   * \ref model_base for details.
   */
  virtual std::string name() = 0;


  /**
   * Returns the current model version
   */
  virtual size_t get_version() const = 0;

  /**
   * Serializes the model. Must save the model to the file format version
   * matching that of get_version()
   */
  virtual void save_impl(oarchive& oarc) const = 0;

  /**
   * Loads a model previously saved at a particular version number.
   * Should raise an exception on failure.
   */
  virtual void load_version(iarchive& iarc, size_t version) = 0;

  /**
   * Set one of the options in the algorithm. Use the option manager to set
   * these options. If the option does not satisfy the conditions that the
   * option manager has imposed on it. Errors will be thrown.
   *
   * \param[in] options Options to set
   */
  virtual void init_options(const std::map<std::string,flexible_type>& _options) = 0;

  /**
   * function implemented by begin_class_member_registration
   */
  virtual void perform_registration() = 0;
  virtual std::string uid() = 0;

  /**
   * Methods with already meaningful default implementations.
   * -------------------------------------------------------------------------
   */

  /**
   * Lists all the keys accessible in the "model" map.
   *
   * \returns List of keys in the model map.
   * \ref model_base for details.
   *
   * Python side interface
   * ------------------------
   *
   * This is the function that the list_fields should call in python.
   */
  std::vector<std::string> list_fields();

  /**
   * Returns the value of a particular key from the state.
   *
   * \returns Value of a key
   * \ref model_base for details.
   *
   * Python side interface
   * ------------------------
   *
   * From the python side, this is interfaced with the get() function or the
   * [] operator in python.
   *
   */
  variant_type get_value_from_state(std::string key);

  /**
   * Get current options.
   *
   * \returns Dictionary containing current options.
   *
   * Python side interface
   * ------------------------
   *  Interfaces with the get_current_options function in the Python side.
   */
  const std::map<std::string, flexible_type>& get_current_options() const;

  /**
   * Get default options.
   *
   * \returns Dictionary with default options.
   *
   * Python side interface
   * ------------------------
   *  Interfaces with the get_default_options function in the Python side.
  */
  std::map<std::string, flexible_type> get_default_options() const;

  /**
   * Returns the value of an option. Throws an error if the option does not
   * exist.
   *
   * \param[in] name Name of the option to get.
   */
  const flexible_type& get_option_value(const std::string& name) const;

  /**
   * Get model.
   *
   * \returns Model map.
   */
  std::map<std::string, variant_type> get_state() const;

  /**
   * Is this model trained.
   *
   * \returns True if already trained.
   */
  bool is_trained() const;

  /**
   * Set one of the options in the algorithm.
   *
   * The value are checked with the requirements given by the option
   * instance.
   *
   * \param[in] name  Name of the option.
   * \param[in] value Value for the option.
   */
  void set_options(const std::map<std::string, flexible_type>& _options);

  /**
   * Append the key value store of the model.
   *
   * \param[in] dict Options (Key-Valye pairs) to set
   */
  void add_or_update_state(const std::map<std::string, variant_type>& dict);

  /** Returns the option information struct for each of the set
   *  parameters.
   */
  const std::vector<option_handling::option_info>& get_option_info() const;

 protected:

  option_manager options;                                 /* Option manager */
  std::map<std::string, variant_type> state;         /**< All things python */


};

} // sdk_model
} // turicreate
#endif
