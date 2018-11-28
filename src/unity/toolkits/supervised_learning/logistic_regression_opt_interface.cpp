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

/**
 * Compute first order statistics at the given point. (Gradient & Function
 * value)
 *
 * \param[in]  point           Point at which we are computing the stats.
 * \param[out] gradient        Dense gradient
 * \param[out] function_value  Function value
 *
 */
void logistic_regression_opt_interface::compute_first_order_statistics(
    const DenseVector& point, DenseVector& gradient, double& f_value) const {
  if (is_dense) {
    _compute_statistics<DenseVector>(point, f_value, gradient, nullptr);
  } else {
    _compute_statistics<SparseVector>(point, f_value, gradient, nullptr);
  }
}

/**
 * Compute second order statistics at the given point. (Gradient & Function
 * value)
 *
 * \param[in]  point           Point at which we are computing the stats.
 * \param[out] hessian         Hessian (Dense)
 * \param[out] gradient        Dense gradient
 * \param[out] function_value  Function value
 *
 */
void logistic_regression_opt_interface::compute_second_order_statistics(
    const DenseVector& point, DenseMatrix& hessian, DenseVector& gradient,
    double& f_value) const {

  if (is_dense) {
    _compute_statistics<DenseVector>(point, f_value, gradient, &hessian);
  } else {
    _compute_statistics<SparseVector>(point, f_value, gradient, &hessian);
  }
}

/**
 * Compute the second order statistics
*/
template <typename PointVector>
void logistic_regression_opt_interface::_compute_statistics(
    const DenseVector& point, double& _function_value, DenseVector& _gradient,
    DenseMatrix* _hessian = nullptr) const {

  const bool _compute_hessian = (_hessian != nullptr); 

  timer t;
  double start_time = t.current_time();
  logstream(LOG_INFO) << "Starting second order stats computation" << std::endl; 
  
  // Initialize computation
  size_t num_threads = thread::cpu_count();
  thread_compute_buffers.resize(num_threads);

  auto& pointMat = thread_compute_buffers.pointMat;

  // Get the point in matrix form
  pointMat = point;
  pointMat.reshape(n_variables_per_class, n_classes);

  // Dense data
  in_parallel([&](size_t thread_idx, size_t n_threads) {

    // Set up all the buffers;
    PointVector x(n_variables_per_class);
    auto& gradient = thread_compute_buffers[thread_idx].gradient;
    auto& hessian = thread_compute_buffers[thread_idx].hessian;
    auto& A = thread_compute_buffers[thread_idx].A;
    auto& XXt = thread_compute_buffers[thread_idx].XXt 
    auto& r = thread_compute_buffers[thread_idx].r;
    auto& f_value = thread_compute_buffers.f_value;

    gradient.zeros();
    f_value = 0; 

    if(compute_hessian) {
      hessian.resize(n_variables, n_variables);
      hessian.setZeros();
    }

    for (auto it = data.get_iterator(thread_idx, n_threads); !it.done(); ++it) {

      fill_reference_encoding(*it, x);
      x(n_variables_per_class - 1) = 1;

      if (scaler != nullptr) {
        scaler->transform(x);
      }

      size_t class_idx = it->target_index();

      // margin = pointMat.t() * x;
      r = pointMat.t() * x;
      double margin_dot_class = (class_idx > 0) ? r(class_idx - 1) : 0;

      // kernel = arma::exp(margin);
      r = arma::exp(r); 

      // kernel_sum = arma::sum(kernel);
      double kernel_sum = arma::sum(r);

      // row_prob = kernel * (1.0 / (1 + kernel_sum));
      r *= (1.0 / (1 + kernel_sum));

      double row_func = log1p(kernel_sum) - margin_dot_class;
      
      if (class_idx > 0) {
        r(class_idx - 1) -= 1;
      }

      // r is row_prob here.
      gradient += arma::vectorise(class_weights[class_idx] * (x * r.t()));

      f_value += class_weights[class_idx] * row_func;

      if(compute_hessian) {
        // TODO: change this so we only care about the upper triangular part til
        // the very end.
        A = diagmat(row_prob) - r * r.t();
        XXT = x * x.t();

        for (size_t a = 0; a < n_classes - 1; a++) {
          for (size_t b = 0; b < n_classes - 1; b++) {

            size_t m = n_variables_per_class;

            hessian.submat(a * m, b * m, (a + 1) * m - 1, (b + 1) * m - 1) +=
                (class_weights[class_idx] * A(a, b)) * XXt;
          }
        }
      }
    }
  });

  // Reduce
  function_value = thread_compute_buffers[0].f_value;
  gradient = thread_compute_buffers[0].gradient;

  if (_hessian != nullptr) {
    (*_hessian) = thread_compute_buffers[0].hessian;
  }

  for (size_t i = 1; i < num_threads; i++) {
    function_value += thread_compute_buffers[i].f_value;
    gradient += thread_compute_buffers[i].gradient;

    if (_hessian != nullptr) {
      (*_hessian) += thread_compute_buffers[i].hessian;
    }
  }

  logstream(LOG_INFO) << "Computation done at "
                      << (t.current_time() - start_time) << "s" << std::endl;
}


} // supervised
} // turicreate
