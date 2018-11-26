/* Copyright © 2017 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#ifndef TURI_REGR_LOGISTIC_REGRESSION_H_
#define TURI_REGR_LOGISTIC_REGRESSION_H_

// ML-Data Utils
#include <ml_data/ml_data.hpp>

// Toolkits
#include <toolkits/supervised_learning/supervised_learning.hpp>
#include <unity/toolkits/coreml_export/mlmodel_wrapper.hpp>

// Optimization Interface
#include <optimization/optimization_interface.hpp>

#include <unity/toolkits/supervised_learning/supervised_learning_constants.hpp>

#include <export.hpp>

namespace turi {
namespace supervised {


/*
 * Logistic Regression Model
 * ****************************************************************************
 */

/**
 * Logistic regression model class definition.
 *
 */
class EXPORT logistic_regression: public supervised_learning_model_base {
  private:

   /** Initialize the model specific options.
    */
   void internal_init_options() override;

   void internal_setup_iterative_training(
       const ml_data& data, const ml_data& validation_data,
       const std::map<std::string, variant_type>& alt_data) override;

   void internal_training_finalize() override;

   /**  Internal prediction function.
    *
    *   This function is up to the prediction
    */
   flexible_type internal_predict_row(
       const ml_data_row_reference& row,
       const std::vector<prediction_type_enum>& output_type) override;

   std::shared_ptr<logistic_regression_opt_interface> lr_interface;

   bool m_simple_mode;

   DenseVector coefs;    // Coefficients of the logisitic regression.
   DenseVector std_err;  // The standard error on each coefficient

   size_t m_num_classes = 0;      /**< fast access: num classes */
   size_t m_num_coefficients = 0; /**< fast access: num coefs   */
  public:


   static constexpr size_t LOGISTIC_REGRESSION_MODEL_VERSION = 6;
   /**
    * Gets the model version number
    */
   size_t get_version() const override {
     return LOGISTIC_REGRESSION_MODEL_VERSION;
  }

 private:
  void _setup_class_parameters();  

  size_t n_variables_per_class = 0;
  size_t n_classes = 0;



 public:

  /**
   * Destructor. Make sure bad things don't happen
   */
  ~logistic_regression();

  
  /**
   * Set the default evaluation metric during model evaluation..
   */
  void set_default_evaluation_metric() override {
    set_evaluation_metric({
        "accuracy", 
        "auc", 
        "confusion_matrix",
        "f1_score", 
        "log_loss",
        "precision", 
        "recall",  
        "roc_curve",
        }); 
  }
  
  /**
   * Set the default evaluation metric for progress tracking.
   */
  void set_default_tracking_metric() override {
    set_tracking_metric({
        "accuracy", 
       }); 
  }

  /**
   * Initialize things that are specific to your model.
   *
   * \param[in] data ML-Data object created by the init function.
   *
   */
  void model_specific_init(const ml_data& data, const ml_data& valid_data) override;
  
  bool is_classifier() const override { return true; }

  /**
   * Initialize the options.
   *
   * \param[in] _options Options to set
   */
  void init_options(const std::map<std::string,flexible_type>& _options) override;
  


  /**
   * Train a regression model.
   */
  void train() override;

  /**
   * Setter for model coefficieints.
   */
  void set_coefs(const DenseVector& _coefs) override;
  
  /**
   * Serialize the object.
   */
  void save_impl(turi::oarchive& oarc) const override;

  /**
   * Load the object
   */
  void load_version(turi::iarchive& iarc, size_t version) override;
  
  /**
   * Predict for a single example. 
   *
   * \param[in] x  Single example.
   * \param[in] output_type Type of prediction.
   *
   * \returns Prediction for a single example.
   *
   */
  flexible_type predict_single_example(
    const DenseVector& x,
    const prediction_type_enum& output_type=prediction_type_enum::NA) override;
  
  /**
   * Fast path predictions given a row of flexible_types.
   *
   * \param[in] rows List of rows (each row is a flex_dict)
   * \param[in] output_type Output type. 
   */
  gl_sframe fast_predict_topk(
      const std::vector<flexible_type>& rows,
      const std::string& missing_value_action ="error",
      const std::string& output_type="",
      const size_t topk = 5) override;
  
  /**
   * Predict for a single example. 
   *
   * \param[in] x  Single example.
   * \param[in] output_type Type of prediction.
   *
   * \returns Prediction for a single example.
   *
   */
  flexible_type predict_single_example(
    const SparseVector& x,
    const prediction_type_enum& output_type=prediction_type_enum::NA) override;

  /**
  * Get coefficients for a trained model.
  */
  void get_coefficients(DenseVector& _coefs) const{
    _coefs.resize(coefs.size());
    _coefs = coefs;
  }
  
  std::shared_ptr<coreml::MLModelWrapper> export_to_coreml() override;

  BEGIN_CLASS_MEMBER_REGISTRATION("classifier_logistic_regression");
  IMPORT_BASE_CLASS_REGISTRATION(supervised_learning_model_base);
  END_CLASS_MEMBER_REGISTRATION

};
} // supervised
} // turicreate

#endif

