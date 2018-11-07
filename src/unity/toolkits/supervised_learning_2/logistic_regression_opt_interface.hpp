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


  private:

   ml_data data;

   // number of examples, features, and total variables


   std::shared_ptr<l2_rescaling> m_scaler; /** <Scale features? */

   bool is_dense = false;                /** Is the data dense? */

   /**
    * Compute second order statistics at the given point. (Gradient & Function
    * value)
    *
    * \param[in]  point           Point at which we are computing the stats.
    * \param[out] function_value  Function value
    * \param[out] gradient        Dense gradient
    * \param[out] hessian_ptr     Hessian, optional, nullptr, etc.
    *
    */
   template <typename PointVector>
   void _compute_second_order_statistics(
       const ml_data& data, const DenseVector& point, double& function_value,
       DenseVector& gradient, DenseMatrix* hessian_ptr = nullptr) const;
};


} // supervised
} // turicreate

#endif
