/* Copyright © 2017 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#ifndef TURI_REGR_LOGISTIC_REGRESSION_OPT_INTERFACE_H_
#define TURI_REGR_LOGISTIC_REGRESSION_OPT_INTERFACE_H_

// ML-Data Utils
#include <ml_data/ml_data.hpp>

// Toolkits
#include <toolkits/supervised_learning/standardization-inl.hpp>
#include <toolkits/supervised_learning/supervised_learning.hpp>
#include <toolkits/supervised_learning/logistic_regression.hpp>

// Optimization Interface
#include <optimization/optimization_interface.hpp>


namespace turi {
namespace supervised {

/*
 * Logistic Regression Solver
 * *****************************************************************************
 *
 */


 /**
 * Solver interface for logistic regression.
 *
 * Let J denote the number of classes, K the number of features, and N the 
 * number of examples.
 *
 * coefs = [coef_1 ... coef_{J-1}] := (K * (J-1)) x 1 column vector
 * where each 
 * coef_j for j = 1 .. J-1 is a K x 1 column vector representing coefficients
 * for the class j.
 *
 */
class logistic_regression_opt_interface: public
  optimization::second_order_opt_interface {

  protected:

  ml_data data;

  // number of examples, features, and total variables
  size_t examples = 0;
  size_t n_classes = 2;
  size_t n_features = 0;
  size_t n_variables = 0;
  size_t n_variables_per_class = 0; 

  DenseVector class_weights; 

  std::shared_ptr<l2_rescaling> scaler;        /** <Scale features */
  bool is_dense = false;                       /** Is the data dense? */


  private:
   // Buffer variables to avoid excessive memory use
   DenseVector coefs_per_class;

  public:
   /**
    * Default constructor
    *
    * \param[in] _data        ML Data containing everything
    *
    * \note Default options are used when the interface is called from the
    * logistic regression class.
    */
   logistic_regression_opt_interface(const ml_data& _data);

   /**
    * Default destructor
    */
   ~logistic_regression_opt_interface();

   /**
    * Set feature scaling
    */
   void init_feature_rescaling();

   /**
    * Transform the final solution back to the original scale.
    *
    * \param[in,out] coefs Solution vector
    */
   void rescale_solution(DenseVector& coefs);

   /**
    * Set the class weights (as a flex_dict which is already validated)
    *
    * \param[in] class_weights Validated flex_dict
    *            Key   : Index of the class in the target_metadata
    *            Value : Weights on the class
    */
   void set_class_weights(const flex_dict& class_weights);

   /**
    * Get the number of examples for the model
    *
    * \returns Number of examples
    */
   size_t num_examples() const;

   /**
    * Get the number of variables in the model
    *
    * \returns Number of variables
    */
   size_t num_variables() const;

   /**
    * Get the number of classes in the model
    *
    * \returns Number of classes
    */
   size_t num_classes() const;

   /**
    * Compute first order statistics at the given point. (Gradient & Function
    * value)
    *
    * \param[in]  point           Point at which we are computing the stats.
    * \param[out] gradient        Dense gradient
    * \param[out] function_value  Function value
    * \param[in]  mbStart         Minibatch start index
    * \param[in]  mbSize          Minibatch size (-1 implies all)
    *
    */
   void compute_first_order_statistics(const DenseVector& point,
                                       DenseVector& gradient,
                                       double& function_value,
                                       const size_t mbStart = 0,
                                       const size_t mbSize = -1);

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
   void compute_second_order_statistics(const DenseVector& point,
                                        DenseMatrix& hessian,
                                        DenseVector& gradient,
                                        double& function_value);

   /**
    * Compute first order statistics at the given point with respect to the
    * a provided dataset. (Gradient & Function value)
    *
    * \param[in]  point           Point at which we are computing the stats.
    * \param[out] gradient        Dense gradient
    * \param[out] function_value  Function value
    *
    */
   void compute_first_order_statistics(const ml_data& data,
                                       const DenseVector& point,
                                       DenseVector& gradient,
                                       double& function_value);

  private: 

   template <typename PointVector>
   void _compute_first_order_statistics(const ml_data& data,
                                       const DenseVector& point,
                                       DenseVector& gradient,
                                       double& function_value);

   template <typename PointVector>
   void _compute_second_order_statistics(const ml_data& data,
                                         const DenseVector& point,
                                         DenseVector& gradient,
                                         DenseMatrix& hessian,
                                         double& function_value);

   DenseMatrix pointMat;

   struct thread_compute_buffer_type {
     DenseVector margin, kernel, row_prob;
     DenseVector gradient; 
     DenseMatrix hessian;
     double f_value; 
   };

   std::vector<thread_compute_buffer_type> thread_compute_buffers;


};


} // supervised
} // turicreate

#endif

