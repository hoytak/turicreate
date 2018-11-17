#include <optimization/lbfgs.hpp>

namespace turi {
namespace optimization {

solver_status lbfgs(std::shared_ptr<first_order_opt_interface> model,
                    const DenseVector& init_point,
                    const std::map<std::string, flexible_type>& opts,
                    const std::shared_ptr<smooth_regularizer_interface>& reg) {
  
  // Get the max number of iterations to perform.
  auto it_n = opts.find("max_iterations");
  size_t num_iterations = (it_n != opts.end())
                              ? it_n->second
                              : default_solver_options.at("max_iterations");

  lbfgs_solver solver(model);

  solver.setup(init_point, opts, reg);

  for (size_t i = 0; i < num_iterations; ++i) {
    bool result = solver.next_iteration();

    if (result) {
      return solver.status();
    }
  }

  solver_status status = solver.status();
  status.status = OPTIMIZATION_STATUS::OPT_ITERATION_LIMIT;
  return status;
}

void lbfgs_solver::setup(
    const DenseVector& init_point,
    const std::map<std::string, flexible_type>& opts,
    const std::shared_ptr<smooth_regularizer_interface>& _reg) {
  
  m_status = solver_status();
  reg = _reg;

  auto get_param = [&](const std::string& n) { 
    auto it = opts.find(n);
    return (it != opts.end()) ? it->second : default_solver_options.at(n);
  };


  lbfgs_memory_level = get_param("lbfgs_memory_level");
  convergence_threshold = get_param("convergence_threshold");
  step_size = 1; // get_param("step_size");


  num_variables = model->num_variables();  // Dimension of point
  DASSERT_EQ(num_variables, init_point.size());

  // Set up the internal parts of the LBFGS information.
  y.resize(num_variables, lbfgs_memory_level);
  y.zeros();

  s.resize(num_variables, lbfgs_memory_level);
  s.zeros();
  
  q.resize(num_variables);
  q.zeros();

  rho.resize(lbfgs_memory_level);
  rho.zeros();

  alpha.resize(lbfgs_memory_level);
  alpha.zeros();

  m_status.solution = init_point;
  
  m_status.gradient.resize(num_variables);
  m_status.gradient.zeros();

  reg_gradient.resize(num_variables);
  previous_gradient.resize(num_variables);

  // Initialize the statistics
  m_status.status = OPTIMIZATION_STATUS::OPT_IN_PROGRESS;
  m_status.iteration = 0;
  m_status.func_value = NAN;

}

/////////////////////////////////////////////////////////////////////////////////

bool lbfgs_solver::next_iteration() {
  double iteration_start_time = compute_timer.current_time();
  compute_timer.start();

  // Set up some convenience notations to make the expressions below more
  // compact.
  const size_t m = lbfgs_memory_level;

  // A function to fill out the status before return.
  auto fill_current_status = [&](OPTIMIZATION_STATUS status) {
    m_status.status = status;
    m_status.solver_time += compute_timer.current_time() - iteration_start_time;
  };

  // Set up references to the containers already held in the stats
  DenseVector& point = m_status.solution;
  DenseVector& gradient = m_status.gradient;
  size_t current_iteration = m_status.iteration;

  // Record the previous gradient and function_value. 
  std::swap(previous_gradient, gradient);  // Fine to swap; gradient is about to be rewritten
  
  double previous_function_value = m_status.func_value;
  double& func_value = m_status.func_value;

  // Computing the gradient and value of the current point.
  model->compute_first_order_statistics(point, gradient, func_value);
  m_status.num_passes++;

  // Check for nan's in the function value.
  if (!std::isfinite(func_value)) {
    fill_current_status(OPTIMIZATION_STATUS::OPT_NUMERIC_ERROR);
    return true;
  }

  // Add regularizer to gradients.
  if (reg != nullptr) {
    reg->compute_gradient(point, reg_gradient);
    gradient += reg_gradient;
  }

  double residual = compute_residual(gradient);

  // Nan Checking!
  if (!std::isfinite(residual)) {
    fill_current_status(OPTIMIZATION_STATUS::OPT_NUMERIC_OVERFLOW);
    return true;
  }

  // Have we converged yet?
  if (residual < convergence_threshold) {
    fill_current_status(OPTIMIZATION_STATUS::OPT_NUMERIC_OVERFLOW);
    return true;
  }

  if (current_iteration == 0) {

    // Do a line search
    double reg_func = 0;

    if (reg != NULL) {
      reg_func = reg->compute_function_value(point);
    }

    // Initialize with a line search
    ls_return lsm_status =
        more_thuente(*model, step_size, func_value + reg_func, point, gradient,
                     -gradient, reg);

    step_size = lsm_status.step_size;

    // Add info from line search
    m_status.func_evals += lsm_status.func_evals;
    m_status.gradient_evals += lsm_status.gradient_evals;
    m_status.num_passes += lsm_status.num_passes;

    // Line search failed
    if (lsm_status.status == false) {
      fill_current_status(OPTIMIZATION_STATUS::OPT_LS_FAILURE);
      return true;
    }

    // Store this delta for use in the next iteration of the algorithm
    point += (delta_point = (-step_size) * gradient);

  } else {
    
    size_t store_point = (current_iteration - 1) % m;

    // Store the gradient differences, step difference and rho for the next
    // iteration.
    s.col(store_point) = delta_point;
    y.col(store_point) = gradient - previous_gradient;
    rho(store_point) = 1.0 / (dot(delta_point, y.col(store_point)));
    
    // Two loop recursion to compute the direction
    // Algorithm 7.4 of Reference [1]

    q = gradient;

    /**
     *  Data is stored in a cyclic format using the following indexiing:
     *
     *   Iteration              Storage location
     *  *****************************************************
     *     iter-1               store_point
     *     iter-2               (store_point + 1) % m
     *      ...                  ...
     *     iter-m               (store_point + m - 1) % m
     *
     **/

    for (size_t j = 0; j < std::min(current_iteration, m); ++j) {
      size_t i = (store_point + m - j) % m;
      alpha(i) = rho(i) * dot(s.col(i), q);
      q -= alpha(i) * y.col(i);
    }
    

    // Scaling factor according to Pg 178 of [1]. This ensures that the
    // problem is better scaled and that a step size of 1 is mostly accepted.
    q *= 1.0 / (squared_norm(y.col(store_point)) * rho(store_point));
    
    for (size_t j = std::min(current_iteration, m); (j--) > 0;) {
      size_t i = (store_point + m - j) % m;
      double beta = rho(i) * dot(y.col(i), q);
      q += s.col(i) * (alpha(i) - beta);
    }



    // Update the new point and gradient
    double reg_func = 0;
    if (reg != NULL) {
      reg_func = reg->compute_function_value(point);
    }

    // Check if we need to retune the step size.
    if(current_iteration == 1 || func_value > previous_function_value) {
    

      // Reset the step size.
      ls_return lsm_status = more_thuente(*model, step_size, func_value + reg_func, point,
                                        gradient, -q, reg);

      // Line search failed
      if (lsm_status.status == false) {
        fill_current_status(OPTIMIZATION_STATUS::OPT_LS_FAILURE);
        return true;
      }

      step_size = lsm_status.step_size;

      // Record statistics from line search
      m_status.func_evals += lsm_status.func_evals;
      m_status.gradient_evals += lsm_status.gradient_evals;
      m_status.num_passes += lsm_status.num_passes;
    }

    point += (delta_point = -step_size * q);
  }

  // Check up on the statisics of how we moved.

  // Numerical error: Insufficient progress.
  if (squared_norm(delta_point) <= OPTIMIZATION_ZERO * OPTIMIZATION_ZERO) {
    fill_current_status(OPTIMIZATION_STATUS::OPT_NUMERIC_ERROR);
    return true;
  }

  // Numerical error: Numerical overflow. (Step size was too large)
  if (!delta_point.is_finite()) {
    fill_current_status(OPTIMIZATION_STATUS::OPT_NUMERIC_OVERFLOW);
    return true;
  }
    
  // Now, report that all is well and return.
  fill_current_status(OPTIMIZATION_STATUS::OPT_IN_PROGRESS);

  m_status.residual = residual;
  ++m_status.iteration;

  return false;
}

}} // End namespaces
