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
* Perform a specialized operation of a outer product between a sparse 
* vector and a dense vector and flatten the result.
*
* out = a * b.t()
* out.resize(9,1);
*
*/
void flattened_sparse_vector_outer_prod(const SparseVector& a, 
                                        const DenseVector& b,
                                        SparseVector& out) {
  DASSERT_TRUE(out.size() == a.size() * b.size());
  out.clear();
  out.reserve(a.num_nonzeros() * b.size());
  size_t a_size = a.size();
  for(size_t j = 0; j < b.size(); j++){
    for(auto p : a) {
      out.insert(p.first + a_size * j, b(j) * p.second);
    }
  }
}

/**
* Constructor for logistic regression solver object
*/
logistic_regression_opt_interface::logistic_regression_opt_interface(
    const ml_data& _data) {
  data = _data;

  // Initialize reader and other data
  examples = data.num_rows();
  features = data.num_columns();

  // Initialize the number of variables to 1 (bias term)
  auto metadata = _data.metadata();

  n_classes = metadata->target_index_size();
  n_variables_per_class = get_number_of_coefficients(metadata);

  is_dense = (n_variables_per_class <= 3 * data.max_row_size()) ? true : false;
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
void logistic_regression_opt_interface::init_feature_rescaling() {
  scaler.reset(new l2_rescaling(data.metadata(), true));
  coefs_per_class.resize(n_variables_per_class);
}

/**
 * Transform final solution back to the original scale.
 */
void logistic_regression_opt_interface::rescale_solution(DenseVector& coefs) {
  DASSERT_TRUE(scaler != nullptr);

  size_t m = n_variables_per_class;

  for (size_t i = 0; i < classes - 1; i++) {
    coefs_per_class = coefs.subvec(i * m, (i + 1) * m - 1 /*end inclusive*/);
    scaler->transform(coefs_per_class);
    coefs.subvec(i * m, (i + 1) * m - 1) = coefs_per_class;
  }
}

/**
* Get the number of examples for the model
*/
size_t logistic_regression_opt_interface::num_examples() const{
  return examples;
}


/**
* Get the number of variables for the model
*/
size_t logistic_regression_opt_interface::num_variables() const{
  return variables;
}


/**
* Get the number of classes for the model
*/
size_t logistic_regression_opt_interface::num_classes() const{
  return classes;
}


double logistic_regression_opt_interface::get_validation_accuracy() {
  DASSERT_TRUE(valid_data.num_rows() > 0);

  auto eval_results = smodel.evaluate(valid_data, "train");
  auto results = eval_results.find("accuracy");
  if(results == eval_results.end()) {
    log_and_throw("No Validation Accuracy.");
  }

  variant_type variant_accuracy = results->second;
  double accuracy = variant_get_value<flexible_type>(variant_accuracy).to<double>();
  return accuracy;
}

double logistic_regression_opt_interface::get_training_accuracy() {
  auto eval_results = smodel.evaluate(data, "train");
  auto results = eval_results.find("accuracy");

  if(results == eval_results.end()) {
    log_and_throw("No Validation Accuracy.");
  }
  variant_type variant_accuracy = results->second;
  double accuracy = variant_get_value<flexible_type>(variant_accuracy).to<double>();

  return accuracy;
}

/**
 * Get strings needed to print a row of the progress table.
 */
std::vector<std::string> logistic_regression_opt_interface::get_status(
    const DenseVector& coefs, 
    const std::vector<std::string>& stats) {

  DenseVector coefs_tmp = coefs;
  rescale_solution(coefs_tmp);
  smodel.set_coefs(coefs_tmp); 

  auto ret = make_progress_row_string(smodel, data, valid_data, stats);
  return ret;
}

/**
 * Compute the first order statistics
*/
template <typename PointVector>
void logistic_regression_opt_interface::_compute_first_order_statistics(
    const ml_data& data, const DenseVector& point, DenseVector& gradient) {


  // Initialize computation
  size_t num_threads = thread::cpu_count();
  thread_compute_buffers.resize(num_threads);

  auto& pointMat = thread_compute_buffers.pointMat;

  // Get the point in matrix form
  pointMat = point;
  pointMat.reshape(n_variables_per_class, n_classes);

  timer t;
  double start_time = t.current_time();

  logstream(LOG_INFO) << "Starting first order stats computation" << std::endl;

  in_parallel([&](size_t thread_idx, size_t n_threads) {

    // set up all the buffers; 
    PointVector x(n_variables_per_class);
    auto& gradient = thread_compute_buffers[thread_idx].gradient;
    auto& margin = thread_compute_buffers[thread_idx].margin;
    auto& kernel = thread_compute_buffers[thread_idx].kernel;
    auto& row_prob = thread_compute_buffers[thread_idx].row_prob;
    auto& f_value = thread_compute_buffers.f_value; 

    graident.resize(n_variables); 
    gradient.setZero();

    for (auto it = data.get_iterator(thread_idx, n_threads); !it.done(); ++it) {
      size_t class_idx = it->target_index();

      if (class_idx >= n_classes) {
        continue;
      }

      fill_reference_encoding(*it, x);
      
      x(n_variables_per_class - 1) = 1;

      if (scaler != nullptr) {
        scaler->transform(x);
      }

      margin = pointMat.t() * x;
      double margin_dot_class = (class_idx > 0) ? margin(class_idx - 1) : 0;

      kernel = arma::exp(margin);
      double kernel_sum = arma::sum(kernel);
      double row_func = log1p(kernel_sum) - margin_dot_class;
      row_prob = kernel * (1.0 / (1 + kernel_sum));

      if (class_idx > 0) {
        row_prob(class_idx - 1) -= 1;
      }

      gradient +=
          arma::vectorise(class_weights[class_idx] * (x * row_prob.t()));

      f_value += class_weights[class_idx] * row_func;
    }
  });
}

/**
 * Compute the second order statistics
*/
template <typename PointVector>
void logistic_regression_opt_interface::_compute_statistics(
    const DenseVector& point, double& _function_value, DenseVector& _gradient,
    DenseMatrix* _hessian = nullptr) {


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
    auto& margin = thread_compute_buffers[thread_idx].margin;
    auto& kernel = thread_compute_buffers[thread_idx].kernel;
    auto& row_prob = thread_compute_buffers[thread_idx].row_prob;
    auto& f_value = thread_compute_buffers.f_value;

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

      margin = pointMat.t() * x;
      double margin_dot_class = (class_idx > 0) ? margin(class_idx - 1) : 0;

      kernel = arma::exp(margin);
      double kernel_sum = arma::sum(kernel);
      row_prob = kernel * (1.0 / (1 + kernel_sum));

      double row_func = log1p(kernel_sum) - margin_dot_class;
      
      if (class_idx > 0) {
        row_prob(class_idx - 1) -= 1;
      }
      gradient +=
          arma::vectorise(class_weights[class_idx] * (x * row_prob.t()));

      f_value += class_weights[class_idx] * row_func;

      if(compute_hessian) {
        // TODO: change this so we only care about the upper triangular part til
        // the very end.
        A = diagmat(row_prob) - row_prob * row_prob.t();
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

  for(size_t i=1; i < n_threads; i++) {
    gradient += thread_compute_buffers[i].gradient;
    function_value += thread_compute_buffers[i].f_value;
  }

  // Reduce
  function_value = f[0];
  hessian = H[0];
  gradient = G[0];
  for(size_t i=1; i < n_threads; i++){
    hessian += H[i];
    gradient += G[i];
    function_value += f[i];
  }

  logstream(LOG_INFO) << "Computation done at " 
                      << (t.current_time() - start_time) << "s" << std::endl; 
}

void logistic_regression_opt_interface::compute_first_order_statistics(const
    DenseVector& point, DenseVector& gradient, double& function_value, const
    size_t mbStart, const size_t mbSize) {
  compute_first_order_statistics(
      data, point, gradient, function_value, mbStart, mbSize);
}

void
logistic_regression_opt_interface::compute_validation_first_order_statistics(
    const DenseVector& point, DenseVector& gradient, double& function_value) {
  compute_first_order_statistics(
      valid_data, point, gradient, function_value);
}


} // supervised
} // turicreate
