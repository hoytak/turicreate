
/* Copyright © 2018 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#ifndef TURI_UNITY_ITERATIVE_ML_MODEL_HPP
#define TURI_UNITY_ITERATIVE_ML_MODEL_HPP

#include <unity/lib/variant.hpp>
#include <unity/lib/unity_base_types.hpp>
#include <unity/lib/toolkit_util.hpp>
#include <unity/lib/toolkit_function_specification.hpp>
#include <unity/lib/toolkit_class_macros.hpp>

#include <unity/lib/extensions/ml_model.hpp>
#include <export.hpp>

namespace turi {

/**
 * ml_model model base class.
 * ---------------------------------------
 *
 *  Base class for handling iterative machine learning models.
 *
 *
 *  Each machine learning iterative model following this pattern contains the
 *  following functions:
 *
 *  *) init_training -- Sets up all the data structures needed for training.
 *
 *  *) next_iteration -- Performs the next iteration in the training set.
 *
 *  *) finalize -- finalizes the iteration patterns for the
 *
 *
 *
*
 *
*
 * *) version: A get version for this model
 *
 *
 */
class EXPORT iterative_ml_model_base: public ml_model_base {

 public:

  static constexpr size_t ML_MODEL_BASE_VERSION = 0;

  // virtual destructor
  inline virtual ~iterative_ml_model_base() { }

  virtual void setup_iterative_training(const std::map<std::string, variant_type>& data) = 0;

  /**  Perform the next iteration of training.
   *
   *   After each iteration, the model must be in a valid state.
   *
   */
  virtual bool next_training_iteration() = 0;

  virtual const std::map<std::string, variant_type>& current_status() = 0;

  virtual void finalize_iterative_training() = 0;

  virtual void checkpointing_available() const { return false;  }


  virtual void save_impl(oarchive& oarc, bool save_full_checkpoint) const = 0;


  // Load the model.  If load_full_checkpoint is true, then
  virtual void load_impl(iarchive& iarc, bool load_full_checkpoint) = 0;






  /////////////////////////////////////

  void train(gl_sframe data, const std::map<std::string, flexible_type>& options);


  void train_with_callback(gl_sframe data, const std::map<std::string, flexible_type>& options,
      const std::function<bool(const variant_map_type&)>& callback);





 protected:

  std::map<std::string, variant_type> current_state;






};


}  // namespace turi

#endif
