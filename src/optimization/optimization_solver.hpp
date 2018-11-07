#ifndef TURI_OPTIMIZATION_SOLVER_H_
#define TURI_OPTIMIZATION_SOLVER_H_

#include <optimization/optimization_interface.hpp>
#include <flexible_type/flexible_type.hpp>
#include <numerics/armadillo.hpp>
#include <sframe/sframe.hpp>

#include <optimization/utils.hpp>
#include <optimization/optimization_interface.hpp>
#include <optimization/regularizer_interface.hpp>
#include <optimization/line_search-inl.hpp>
#include <numerics/armadillo.hpp>

namespace turi {
namespace optimization {

/**
 * Solver status.
 */
struct solver_status {
  size_t iteration = -1;                        /*!< Iterations taken */
  double solver_time = -1;                      /*!< Wall clock time (s) */
  DenseVector solution;                         /*!< Current Solution */
  DenseVector gradient;                         /*!< Current gradient */
  DenseMatrix hessian;                          /*!< Current hessian */
  double residual = NAN;                        /*!< Residual norm */
  double func_value = NAN;                      /*!< Function value */
  size_t func_evals = 0;                        /*!< Function evals */
  size_t gradient_evals = 0;                    /*!< Gradient evals */
  size_t num_passes = 0;                        /*!< Number of passes over the data */
  OPTIMIZATION_STATUS status = OPTIMIZATION_STATUS::OPT_UNSET;  /*!< Status */
};


class iterative_optimization_solver {
 public:

   virtual ~iterative_optimization_solver(){}


  /** Sets up (or resets) the solver with the initial conditions.
   *
   */
  virtual void setup(const DenseVector& init_point,
             const std::map<std::string, flexible_type>& opts,
             const std::shared_ptr<smooth_regularizer_interface>& reg) = 0;

  /** Call this method repeatedly, with each HERE
   *
   *
   */
  virtual bool next_iteration() = 0;

  // Return the current status.
  const solver_status& status() const { return m_status; }


 protected:
  solver_status m_status;

};

}}
