/* Copyright © 2017 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#ifndef TURI_SUPERVISED_LEARNING_H_
#define TURI_SUPERVISED_LEARNING_H_

#include <unity/lib/gl_sarray.hpp>
#include <unity/lib/gl_sframe.hpp>

// Interfaces
#include <unity/lib/extensions/ml_model.hpp>

// ML-Data Utils
#include <ml_data/ml_data.hpp>
#include <ml_data/ml_data_iterator.hpp>

// Types
#include <unity/lib/variant.hpp>
#include <unity/lib/unity_base_types.hpp>
#include <unity/toolkits/coreml_export/mlmodel_wrapper.hpp>

#include <numerics/armadillo.hpp>

#include <export.hpp>

#include <unity/lib/toolkit_class_macros.hpp>

// TODO: List of todo's for this file
//------------------------------------------------------------------------------
// 1. ml_data_example type for predict.
//

namespace turi {
namespace supervised {

class supervised_learning_model_base;
typedef arma::vec  DenseVector;
typedef sparse_vector<double>  SparseVector;


/**
 * Supervised_learning model base class.
 * ---------------------------------------
 *
 *  Base class for handling supervised learning class. This class is meant to 
 *  be a guide to aid model writing and not a hard and fast rule of how the
 *  code must be structured. 
 *
 *  Each supervised learning C++ toolkit contains the following:
 *
 *  *) model: This is the key-value map that stores the "model" attributes.
 *            The value is of type "variant_type" which is fully interfaced
 *            with python. You can add basic types, vectors, SFrames etc.
 *
 *  *) metadata: A globally consistent object with column wise metadata. This
 *               metadata changes with time (even after training). If you 
 *               want to freeze the metadata after training, you have to do 
 *               so yourself.
 *
 *  *) train_feature_size: Feature sizes (i.e column sizes) during train time.
 *                         Numerical features are of size 1, categorical features
 *                         are of size (# categories), vector features are of size
 *                         length, and dictionary features are of size # keys.
 *
 *  *) options: Option manager which keeps track of default options, current
 *              options, option ranges, type etc. This must be initialized only
 *              once in the set_options() function.
 *
 * 
 * Functions that should always be implemented. Here are some notes about
 * each of these functions that may help guide you in writing your model.
 *
 *
 * *) name: Get the name of this model. You might thinks that this is silly but
 *          the name holds the key to everything. The unity_server can construct
 *          model_base objects and they can be cast to a model of this type.
 *          The name determine how the casting happens. The init_models()
 *          function in unity_server.cpp will give you an idea of how 
 *          this interface happens.
 *
 * *) train: A train function for the model. 
 *
 * *) predict_single_example: A predict function for the model for single 
 *             example. If this is implemented, batch predictions and evaluation
 *             need not be implemented.
 *
 * *) predict: A predict function for the model for batch predictions. 
 *             The result of this function can be an SArray of predictions.
 *             One for each value of the input SFrame.
 *             
 * *) evaluate: An evaluattion function for the model for evaluations. 
 *              The result of this function must be an updated evaluation_stats
 *              map which can be queried with the get_evaluation_stats().
 *
 * *) save: Save the model with the turicreate iarc. Turi is a server-client
 *          module. DO NOT SAVE ANYTHING in the client side. Make sure that 
 *          everything is in the server side. For example: You might be tempted
 *          do keep options that the user provides into the server side but
 *          DO NOT do that because save and load will break things for you!
 *
 * *) load: Load the model with the turicreate oarc.
 *
 * *) init_options: Init the options
 *    
 *
 * This class interfaces with the SupervisedLearning class in Python and works
 * end to end once the following set of fuctions are implemented by the user.
 *
 *
 * Example Class
 * -----------------------------------------------------------------------------
 *
 * See the file supervised_learning_model.cxx for an example of how to use 
 * this class in building your supervised learning method.
 *
 *
 *
 */
class EXPORT supervised_learning_model_base : public ml_model_base {
 protected:
  bool show_extra_warnings = true;                /* If true, be more verbose.*/

  std::shared_ptr<ml_metadata> _metadata;          /* ML Data  metadata. */

   /**
   * Get ml_metadata.
   *
   * \returns Get the ml_metadata.
   */
  std::shared_ptr<ml_metadata> get_ml_metadata() const {
    return this->metadata;
  }

 public: 


  // Set up the init options component 
  virtual void init_options() = 0; 

  // virtual destructor
  virtual ~supervised_learning_model_base() { }

  void set_options(const std::map<std::string, flexible_type>& opts);

  void setup_training(gl_sframe data, const std::string& target,
                      const variant_type& validation_data = FLEX_UNDEFINED);

  flexible_type predict_row(const flex_dict& row,
                            std::string output_type,
                            std::string missing_value_action = "");

  /**
   * Extract features! 
   */
  virtual gl_sarray extract_features(const gl_sframe& X,
                                     const std::string& missing_value_action) {
    log_and_throw("Model does not support feature extraction");
  }

  /**
   * Export to CoreML.
   */
  virtual std::shared_ptr<coreml::MLModelWrapper> export_to_coreml() = 0;

  variant_map_type evaluate(gl_sframe data, std::string metric = "auto",
                            std::string missing_value_action = "");

  gl_sarray predict(gl_sframe data, std::string missing_value_action, std::string output_type); 


protected:
 void internal_train(const ml_data& data, const ml_data& validation_data);

 /**
  *
  * Predict for a single example.
  *
  * \param[in] x  Single example.
  * \param[in] output_type Type of prediction.
  *
  * \returns Prediction for a single example.
  *
  */
 virtual flexible_type predict_single_example(
     const ml_data_row_reference& row,
     const prediction_type_enum& output_type = prediction_type_enum::NA) = 0;

 /**
  *  Train the model
  */
 void train(gl_sframe data, const std::string& target,
            const variant_type& validation_data = FLEX_UNDEFINED,
            const std::map<std::string, flexible_type>& _options = {});

 /**
  * Methods that must be implemented in a new supervised_learning model.
  * -------------------------------------------------------------------------
  */
 /**
  * Get metadata mapping.
  */
 std::vector<std::vector<flexible_type> > get_metadata_mapping();

 /**
  * Methods with default implementations but are in-flux during the
  * Trees and NeuralNetworks integration
  * -------------------------------------------------------------------------
  */

 /**
  * Predict for a single example.
  *
  * \param[in] x  Single example.
  * \param[in] output_type Type of prediction.
  *
  * \returns Prediction for a single example.
  *
  */
 virtual flexible_type predict_single_example(
     const DenseVector& x,
     const prediction_type_enum& output_type = prediction_type_enum::NA) {
   return 0.0;
  }

  /**
   * Predict for a single example. 
   *
   * \param[in] x  Single example.
   * \param[in] output_type Type of prediction.
   *
   * \returns Prediction for a single example.
   *
   */
  virtual flexible_type predict_single_example(
          const SparseVector & x,
          const prediction_type_enum& output_type=prediction_type_enum::NA) {
    return 0.0;
  }
  
  /**
   * Evaluate the model.
   *
   * \param[in] test_data          Test data.
   * \param[in] evaluation_type Evalution type.
   *
   * \note Already assumes that data is of the right shape. Test data
   * must contain target column also.
   *
   */
  virtual std::map<std::string, variant_type> evaluate(const ml_data&
                test_data, const std::string& evaluation_type="");

  /**
   * Same as evaluate(ml_data), but take SFrame as input.
   */
  virtual std::map<std::string, variant_type> evaluate(const sframe& X,
                const sframe &y, const std::string& evaluation_type="") {
    ml_data data = construct_ml_data_using_current_metadata(X, y);
    return this->evaluate(data, evaluation_type);
  }

  /**
   * Make predictions using a trained supervised_learning model.
   *
   * \param[in] test_X      Test data (only independent variables)
   * \param[in] output_type Type of prediction.
   * \returns ret   Shared pointer to an SArray containing predicions.
   *
   * \note Already assumes that data is of the right shape.
   */
  virtual std::shared_ptr<sarray<flexible_type>> predict(
    const ml_data& test_data, const std::string& output_type="");

  /**
   * Same as predict(ml_data), but takes SFrame as input.
   */
  virtual std::shared_ptr<sarray<flexible_type>> predict(
    const sframe& X, const std::string& output_type="") {
    ml_data data = construct_ml_data_using_current_metadata(X);
    return predict(data, output_type);
  }


  /**
   * Make multiclass predictions using a trained supervised_learning model.
   *
   * \param[in] test_X      Test data (only independent variables)
   * \param[in] output_type Type of prediction.
   * \param[in] topk        Number of classes to return.
   * \returns ret   SFrame containing {row_id, class, output_type}.
   *
   * \note Already assumes that data is of the right shape.
   * \note Default throws error, model supporting this method should override 
   * this function.
   */
  virtual sframe predict_topk(const sframe& test_data,
                              const std::string& output_type="",
                              size_t topk=5) {
    log_and_throw("Predicting multiple classes is not supported by this model.");
  }

  /**
   * Make multiclass predictions using a trained supervised_learning model.
   *
   * \param[in] test_X      Test data (only independent variables)
   * \param[in] output_type Type of prediction.
   * \param[in] topk        Number of classes to return.
   * \returns ret   SFrame containing {row_id, class, output_type}.
   *
   * \note Already assumes that data is of the right shape.
   */
  virtual sframe predict_topk(const ml_data& test_data,
                              const std::string& output_type="",
                              size_t topk=5);

  /**
   * Make classification using a trained supervised_learning model.
   *
   * \param[in] X           Test data (only independent variables)
   * \param[in] output_type Type of classifcation (future proof).
   * \returns ret   SFrame with "class" and probability (if applicable)
   *
   * \note Already assumes that data is of the right shape.
   */
  virtual sframe classify(const ml_data& test_data, 
                          const std::string& output_type="");

  /**
   * Same as classify(ml_data), but takes SFrame as input.
   */
  virtual sframe classify(const sframe& X,
                          const std::string& output_type="") {

    ml_data data = construct_ml_data_using_current_metadata(X);
    return classify(data, output_type);
  };
  
  /**
   * Fast path predictions given a row of flexible_types.
   *
   * \param[in] rows List of rows (each row is a flex_dict)
   * \param[in] missing_value_action Missing value action string
   * \param[in] output_type Output type. 
   */
  virtual gl_sarray fast_predict(
      const std::vector<flexible_type>& rows,
      const std::string& missing_value_action = "error",
      const std::string& output_type = "");
  
  /**
   * Fast path predictions given a row of flexible_types.
   *
   * \param[in] rows List of rows (each row is a flex_dict)
   * \param[in] missing_value_action Missing value action string
   * \param[in] output_type Output type. 
   * \param[in] topk Number of classes to return
   */
  virtual gl_sframe fast_predict_topk(
      const std::vector<flexible_type>& rows,
      const std::string& missing_value_action ="error",
      const std::string& output_type="", 
      const size_t topk = 5) {
    log_and_throw("Not implemented yet");
  }
  
  /**
   * Fast path predictions given a row of flexible_types
   *
   * \param[in] rows List of rows (each row is a flex_dict)
   * \param[in] output_type Output type. 
   */
  virtual gl_sframe fast_classify(
      const std::vector<flexible_type>& rows,
      const std::string& missing_value_action ="error");

  /**
   * Methods with already meaningful default implementations.
   * -------------------------------------------------------------------------
   */


  /**
   * Init the model with the data.
   *
   * \param[in] X              Predictors
   * \param[in] y              target
   *
   * Python side interface
   * ------------------------
   *  NA.
   *
   */
  virtual void init(
      const sframe& X, const sframe& y, 
      const sframe& valid_X=sframe(), 
      const sframe& valid_y=sframe(),
      ml_missing_value_action mva = ml_missing_value_action::ERROR);
  
  /**
   * A setter for models that use Armadillo for model coefficients.
   */
  virtual void set_coefs(const DenseVector& coefs) {
    DASSERT_TRUE(false);
  }

  /**
   * Set the evaluation metric. Set to RMSE by default.
   */
  void set_evaluation_metric(std::vector<std::string> _metrics){
    metrics = _metrics;
  }


  /**
   * Set the evaluation metric. Set to RMSE by default.
   */
  void set_tracking_metric(std::vector<std::string> _metrics){
    tracking_metrics = _metrics;
  }
  
  /**
   * Set the Extra Warnings output. These warnings include telling the user
   * about low-variance features, etc...
   */
  void set_more_warnings(bool more_warnings){
    show_extra_warnings = more_warnings;
  }

  /**
   * Set the default evaluation metric during model evaluation.
   */
  virtual void set_default_evaluation_metric(){
    set_evaluation_metric({"max_error", "rmse"});
  }
  
  /**
   * Set the default evaluation metric for progress tracking.
   */
  virtual void set_default_tracking_metric(){
    set_tracking_metric({"max_error", "rmse"}); 
  }

  /**
   * Get training stats.
   *
   * \returns The training stats map.
   *
   * Python side interface
   * ------------------------
   *  The dictionary returned to the user can be transfered as is to the python
   *  side. You MUST use this to return a dictionary to the object.
   */
  std::map<std::string, flexible_type> get_train_stats() const;

  /**
   * Impute missing columns with 'None' values.
   *
   * \param[in] X  Predictors
   *
   * \returns An SFrame with 'None' written to all columns that are missing.
   *
   */
  sframe impute_missing_columns_using_current_metadata(const sframe& X) const;

  /**
   * Construct ml-data from the predictors and target using the current
   * value of the metadata.
   *
   * \param[in] X        Predictors
   * \param[in] y        target
   * \param[in] new_opts Additional options.
   *
   * \returns A constructed ml_data object
   *
   */
  ml_data construct_ml_data_using_current_metadata(
    const sframe& X, const sframe& y, 
    ml_missing_value_action mva = ml_missing_value_action::ERROR) const;
  
  /**
   * Construct ml-data from the predictors using the current
   * value of the metadata.
   *
   * \param[in] X         Predictors
   * \param[in] new_opts  Additional options.
   *
   * \returns A constructed ml_data object
   *
   */
  ml_data construct_ml_data_using_current_metadata(
    const sframe& X, 
    ml_missing_value_action mva = ml_missing_value_action::ERROR) const;

  /**
   * Get the number of feature columns in the model
   *
   * \returns Number of features.
   */
  size_t num_features() const;

  /**
   * Get the number of examples in the model
   *
   * \returns Number of examples.
   */
  size_t num_examples() const;
  
  /**
   * Get the number of features in the model (unpacked)
   *
   * \returns Number of features.
   */
  size_t num_unpacked_features() const;

  /**
   * Get names of predictor variables.
   *
   * \returns Names of features (Vector of string names).
   */
  std::vector<std::string> get_feature_names() const;

  /**
   * Get name of the target column.
   *
   * \returns Names of target.
   */
  std::string get_target_name() const;

  /**
   * Get ml_metadata.
   *
   * \returns Get the ml_metadata.
   */
  std::shared_ptr<ml_metadata> get_ml_metadata() const {
    return this->metadata;
  }

  /**
   * Returns true if the model is a classifier.
   */
  virtual bool is_classifier() const = 0;

  /**
   * Returns true if the model is a classifier.
   */
  bool is_dense() {
    return ((this->metadata)->num_dimensions() <= 3 * num_features()) ? true : false;
  }

  /**
   * Get metrics strings.
   */
  std::vector<std::string> get_metrics()  const;
  
  /**
   * Get tracking metrics strings.
   */
  std::vector<std::string> get_tracking_metrics()  const;

  /**
   * Get metric display name.
   */
  std::string get_metric_display_name(const std::string& metric) const;

  /**
   * Display model training data summary for regression.
   *
   * \param[in] model_display_name   Name to be displayed
   *
   */
  void display_regression_training_summary(std::string model_display_name) const;
  
  /**
   * Display model training data summary for classifier.
   *
   * \param[in] model_display_name   Name to be displayed
   *
   */
  void display_classifier_training_summary(std::string model_display_name, bool simple_mode = false) const;

  /**
   * Methods with no current implementation (or empty implementations)
   * -------------------------------------------------------------------------
   */
  
  /**
   * Initialize things that are specific to your model.
   *
   * \param[in] data ML-Data object created by the init function.
   *
   */
  virtual void model_specific_init(const ml_data& data, 
                                   const ml_data& validation_data) { }

  /**
   * Returns true if the model can handle missing value
   */
  virtual bool support_missing_value() const { return false; }

 /**
   *  API interface through the unity server.
   *
   *  Run prediction. 
   */
  gl_sarray api_predict(gl_sframe data, std::string missing_value_action,
                        std::string output_type);  // TODO: This should be const

  /**
   *  API interface through the unity server.
   *
   *  Run multiclass prediction.
   */
  gl_sframe api_predict_topk(gl_sframe data, std::string missing_value_action,
			     std::string output_type, size_t topk = 5);
  // TODO: This function should be const.

  /**
   *  API interface through the unity server.
   *
   *  Run classification.
   */
  // TODO: This function should be const
  gl_sframe api_classify(gl_sframe data, std::string missing_value_action,
                         std::string output_type); // TODO: This should be const

  /**
   *  API interface through the unity server.
   *
   *  Evaluate the model
   */
  // TODO: This function should be const
  variant_map_type api_evaluate(
      gl_sframe data, std::string missing_value_action, std::string metric);

  /**
   *  API interface through the unity server.
   *
   *  Extract features!
   */
  // TODO: This function should be const
  gl_sarray api_extract_features(
      gl_sframe data, std::string missing_value_action);

  std::shared_ptr<coreml::MLModelWrapper> api_export_to_coreml(const std::string& file);

#define SUPERVISED_LEARNING_METHODS_REGISTRATION(name, class_name)             \
                                                                               \
  BEGIN_CLASS_MEMBER_REGISTRATION(name)                                        \
                                                                               \
  REGISTER_CLASS_MEMBER_FUNCTION(class_name::list_fields)                      \
  REGISTER_NAMED_CLASS_MEMBER_FUNCTION(                                        \
      "get_value", class_name::get_value_from_state, "field");                 \
                                                                               \
  REGISTER_NAMED_CLASS_MEMBER_FUNCTION("train", class_name::api_train, "data", \
                                       "target", "validation_data",            \
                                       "options");                             \
  register_defaults(                                                           \
      "train",                                                                 \
      {{"validation_data", to_variant(gl_sframe())},                           \
       {"options", to_variant(std::map<std::string, flexible_type>())}});      \
                                                                               \
  REGISTER_NAMED_CLASS_MEMBER_FUNCTION("predict", class_name::api_predict,     \
                                       "data", "missing_value_action",         \
                                       "output_type");                         \
                                                                               \
  register_defaults("predict", {{"missing_value_action", std::string("auto")}, \
                     {"output_type", std::string("")}});                       \
                                                                               \
  REGISTER_NAMED_CLASS_MEMBER_FUNCTION("fast_predict",                         \
                                       class_name::fast_predict, "rows",       \
                                       "missing_value_action", "output_type"); \
                                                                               \
  register_defaults("fast_predict",                                            \
                    {{"missing_value_action", std::string("auto")},            \
                     {"output_type", std::string("")}});                       \
                                                                               \
  REGISTER_NAMED_CLASS_MEMBER_FUNCTION(                                        \
      "predict_topk", class_name::api_predict_topk, "data",                    \
      "missing_value_action", "output_type", "topk");                          \
                                                                               \
  register_defaults("predict_topk",                                            \
                    {{"missing_value_action", std::string("error")},           \
                     {"output_type", std::string("")}});                       \
                                                                               \
  REGISTER_NAMED_CLASS_MEMBER_FUNCTION(                                        \
      "fast_predict_topk", class_name::fast_predict_topk, "rows",              \
      "missing_value_action", "output_type", "topk");                          \
                                                                               \
  register_defaults("fast_predict_topk",                                       \
                    {{"missing_value_action", std::string("auto")},            \
                     {"output_type", std::string("")}});                       \
                                                                               \
  REGISTER_NAMED_CLASS_MEMBER_FUNCTION("classify", class_name::api_classify,   \
                                       "data", "missing_value_action");        \
                                                                               \
  register_defaults("classify",                                                \
                    {{"missing_value_action", std::string("auto")}});          \
                                                                               \
  REGISTER_NAMED_CLASS_MEMBER_FUNCTION("fast_classify",                        \
                                       class_name::fast_classify, "rows",      \
                                       "missing_value_action");                \
                                                                               \
  register_defaults("fast_classify",                                           \
                    {{"missing_value_action", std::string("auto")}});          \
                                                                               \
  REGISTER_NAMED_CLASS_MEMBER_FUNCTION("evaluate", class_name::api_evaluate,   \
                                       "data", "missing_value_action",         \
                                       "metric");                              \
                                                                               \
  register_defaults("evaluate",                                                \
                    {{"metric", std::string("_report")},                        \
                     {"missing_value_action", std::string("auto")}});          \
                                                                               \
  REGISTER_NAMED_CLASS_MEMBER_FUNCTION("extract_features",                     \
                                       class_name::api_extract_features,       \
                                       "data", "missing_value_action");        \
                                                                               \
  register_defaults("extract_features",                                        \
                    {{"missing_value_action", std::string("auto")}});          \
                                                                               \
  REGISTER_CLASS_MEMBER_FUNCTION(class_name::is_trained);                      \
  REGISTER_CLASS_MEMBER_FUNCTION(class_name::get_train_stats);                 \
  REGISTER_CLASS_MEMBER_FUNCTION(class_name::get_feature_names);               \
  REGISTER_CLASS_MEMBER_FUNCTION(class_name::get_option_value);                \
                                                                               \
  REGISTER_NAMED_CLASS_MEMBER_FUNCTION(                                        \
      "export_to_coreml", class_name::api_export_to_coreml, "filename");       \
                                                                               \
  register_defaults("export_to_coreml", {{"filename", std::string("")}});      \
                                                                               \
  END_CLASS_MEMBER_REGISTRATION


  protected:
   ml_missing_value_action get_missing_value_enum_from_string(
       const std::string& missing_value_str) const;
};

/**
 * Obtains the function registration for the toolkit.
 */
std::vector<toolkit_function_specification> get_toolkit_function_registration();

/**
 * Fast prediction path for in-memory predictions on a list of rows.
 * \param[in] model Supervised learning model.
 * \param[in] rows  List of rows to make a prediction with. 
 */
gl_sarray _fast_predict(
    std::shared_ptr<supervised_learning_model_base> model, 
    const std::vector<flexible_type>& rows, 
    const std::string& missing_value_action = "error",
    const std::string& output_type = "probability");

/**
 * Fast path for in-memory predictions on a list of rows.
 *
 * \param[in] model Supervised learning model.
 * \param[in] rows  List of rows to make a prediction with. 
 */
gl_sframe _fast_predict_topk(
    std::shared_ptr<supervised_learning_model_base> model, 
    const std::vector<flexible_type>& rows,
    const std::string& missing_value_action = "error",
    const std::string& output_type = "probability",
    const size_t topk = 5);

/**
 * Fast path for in-memory predictions on a list of rows.
 *
 * \param[in] model Supervised learning model.
 * \param[in] rows  List of rows to make a prediction with. 
 */
gl_sframe _fast_classify(
    std::shared_ptr<supervised_learning_model_base> model, 
    const std::vector<flexible_type>& rows,
    const std::string& missing_value_action = "error");

/**
 * Get the metadata mapping.
 *
 * \param[in] model Supervised learning model. 
 */
std::vector<std::vector<flexible_type>> _get_metadata_mapping(
    std::shared_ptr<supervised_learning_model_base> model);

} // supervised
} // turicreate

#endif

