/* Copyright © 2017 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
// ML Data
#include <sframe/sframe.hpp>
#include <sframe/algorithm.hpp>
#include <unity/lib/unity_sframe.hpp>

// Toolkits
#include <toolkits/supervised_learning/logistic_regression.hpp>
#include <toolkits/supervised_learning/logistic_regression_opt_interface.hpp>
#include <toolkits/supervised_learning/supervised_learning_utils-inl.hpp>

// Solvers
#include <optimization/utils.hpp>
#include <optimization/newton_method-inl.hpp>
#include <optimization/lbfgs-inl.hpp>
#include <optimization/gradient_descent-inl.hpp>
#include <optimization/accelerated_gradient-inl.hpp>

// Regularizer
#include <optimization/regularizers-inl.hpp>

// Utilities
#include <numerics/armadillo.hpp>
#include <cmath>
#include <serialization/serialization_includes.hpp>


namespace turi {
namespace supervised {


/*
* Logistic Regression Solver Interface
*
*******************************************************************************
*/


/**
* Constructor for logistic regression solver object
*/
logistic_regression_opt_interface::logistic_regression_opt_interface(
    const ml_data& _data, bool enable_scaling) {

  data = _data;

  // Initialize reader and other data
  examples = data.num_rows();
  features = data.num_columns();

  // Initialize the number of variables to 1 (bias term)
  auto metadata = _data.metadata();

  n_classes = metadata->target_index_size();
  n_variables_per_class = get_number_of_coefficients(metadata);

  is_dense = (n_variables_per_class <= 8 * data.max_row_size()) ? true : false;
  n_variables = n_variables_per_class * (n_classes - 1);

  class_weights = arma::ones(n_classes);

}


/**
* Destructor for the logistic regression solver object
*/
logistic_regression_opt_interface::~logistic_regression_opt_interface() {
}

/**
* Set the class weights (as a flex_dict which is already validated)
*/
void logistic_regression_opt_interface::set_class_weights(const DenseVector& _class_weights) {
  class_weights = _class_weights;
}

/**
 * Set feature rescaling.
 */
void logistic_regression_opt_interface::set_feature_scaler(
    const std::shared_ptr<l2_rescaling>& scaler) {
  m_scaler = scaler;
}

/**
 * Transform final solution back to the original scale.
 */
void logistic_regression_opt_interface::rescale_solution(DenseVector& coefs) {

  size_t m = n_variables_per_class;

   DenseVector coefs_per_class;
   coefs_per_class.resize(n_variables_per_class);

   for (size_t i = 0; i < classes - 1; i++) {
     coefs_per_class = coefs.subvec(i * m, (i + 1) * m - 1 /*end inclusive*/);
     scaler->transform(coefs_per_class);
     coefs.subvec(i * m, (i + 1) * m - 1) = coefs_per_class;
  }
}
