/* Copyright © 2017 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#ifndef TURI_SUPERVISED_LEARNING_UTILS_H_
#define TURI_SUPERVISED_LEARNING_UTILS_H_

#include <numerics/armadillo.hpp>
// SFrame
#include <sframe/sarray.hpp>
#include <sframe/sframe.hpp>

// ML-Data Utils
#include <ml_data/ml_data.hpp>
#include <ml_data/metadata.hpp>
#include <util/testing_utils.hpp>
// Supervised learning includes. 
#include <toolkits/supervised_learning/supervised_learning.hpp>

// Types
#include <unity/lib/variant.hpp>
#include <unity/lib/unity_base_types.hpp>
#include <unity/lib/variant_deep_serialize.hpp>
#include <unity/lib/flex_dict_view.hpp>

/// SKD
#include <unity/lib/toolkit_function_macros.hpp>
#include <serialization/serialization_includes.hpp>
 

namespace turi {
namespace supervised {


/**
 * Get standard errors from hessian. 
 */
inline arma::vec get_stderr_from_hessian(
    const arma::mat& hessian) {
  DASSERT_EQ(hessian.n_rows, hessian.n_cols);
  arma::mat I;
  I.eye(hessian.n_rows, hessian.n_cols);
  return arma::sqrt(arma::diagvec(arma::inv_sympd(hessian + 1E-8 * I)));
}



/**
 * Setup the ml_data for prediction.
 */
inline ml_data setup_ml_data_for_prediction(
                const sframe& X,
                const std::shared_ptr<supervised_learning_model_base>& model,
                ml_missing_value_action missing_value_action) {

  ml_data data;
  data = model->construct_ml_data_using_current_metadata(X, missing_value_action);
  return data;
}

/**
 * Setup the ml_data for evaluation.
 */
inline ml_data setup_ml_data_for_evaluation(const sframe& X, const sframe& y,
                const std::shared_ptr<supervised_learning_model_base>& model,
                ml_missing_value_action missing_value_action) {
  ml_data data;
  data = model->construct_ml_data_using_current_metadata(X, y, missing_value_action);
  return data;
}


/**
 * Check that the target types are right.
 * 
 * Regression vs classifier:
 *
 * One user in our forum complained that he got an error message for logistic
 * regression suggesting that his columns was not of numeric type. He
 * should have gotten a message that said. Column not if integer type.
 *
 * Let us separate our (for the purposes of error messages) logging
 * for classifier vs regression tasks.
 *
 */
inline void check_target_column_type(std::string model_name, sframe y){
  DASSERT_TRUE(y.num_columns() == 1);

  std::stringstream ss;
  std::string model_name_for_display = "";

  if (model_name == "classifier_svm"){
    model_name_for_display = "SVM";
  } else if (model_name == "classifier_logistic_regression"){
    model_name_for_display = "Logistic Regression";
  }

  // classifier tasks.
  if(model_name == "classifier_svm" || 
     model_name == "classifier_logistic_regression" || 
     model_name == "random_forest_classifier" || 
     model_name == "decision_tree_classifier" || 
     model_name == "boosted_trees_classifier"){

    flex_type_enum ctype = y.column_type(0);
    if (ctype != flex_type_enum::INTEGER && ctype != flex_type_enum::STRING){
      ss.str("");
      ss << "Column type of target '" << y.column_name(0) 
         << "' must be int or str." 
         << std::endl;
      log_and_throw(ss.str());
    }

  } else {
    
    flex_type_enum ctype = y.column_type(0);
    if ((ctype != flex_type_enum::INTEGER) && (ctype !=
          flex_type_enum::FLOAT)){
      ss.str("");
      ss << "Column type of target '" << y.column_name(0) 
         << "' must be int or float." 
         << std::endl;
      log_and_throw(ss.str());
    }
  } 
}

/**
 * Setup an SFrame as test input to predict, predict_topk, or classify function. 
 */
inline sframe setup_test_data_sframe(const sframe& sf,
                                     std::shared_ptr<supervised_learning_model_base> model,
                                     ml_missing_value_action missing_value_action) {
  sframe ret;
  check_empty_data(sf);

  auto expected_columns = model->get_feature_names();
  switch (missing_value_action) {
    case ml_missing_value_action::IMPUTE:
      ret = model->impute_missing_columns_using_current_metadata(sf);
      break;
    case ml_missing_value_action::USE_NAN:
      if (model->support_missing_value()) {
        ret = model->impute_missing_columns_using_current_metadata(sf);
      } else {
        log_and_throw("Model doesn't support missing value, please set missing_value_action to \"impute\"");
      }
      break;
    case ml_missing_value_action::ERROR:
      ret = sf;
      break;
    default:
      log_and_throw("Invalid missing value action");
  }
  ret = ret.select_columns(expected_columns);
  return ret;
}




/**
 * For each of the provided keys, get a string of the corresponding value.
 */
inline std::vector<std::string> make_evaluation_progress(
    const std::map<std::string, float>& eval_map,
    const std::vector<std::string> keys) {
  std::vector<std::string> ret;
  if (!eval_map.empty()) {
    for (auto& k : keys)
      // TODO: Check that k exists in eval_map.
      ret.push_back(std::to_string(eval_map.at(k)));
  }
  return ret;
}

inline std::vector<std::string> make_progress_string(
    size_t iter, size_t examples, double time,
    const std::vector<std::string>& train_eval,
    const std::vector<std::string>& valid_eval,
    float speed, bool padding_valid_eval) {

  std::vector<std::string> ret; 
  ret.push_back(std::to_string(iter));
  ret.push_back(std::to_string(examples));
  ret.push_back(std::to_string(time));
  for (size_t i = 0 ; i < train_eval.size(); ++i) {
    ret.push_back(train_eval[i]);
    if (!valid_eval.empty()) {
      ret.push_back(valid_eval[i]);
    } else if(padding_valid_eval) {
      ret.push_back("");
    }
  }
  ret.push_back(std::to_string(speed));
  return ret;
}

/**
 * For the provided model, print all of its desired metrics using
 * the provided headers.
 */
inline std::vector<std::pair<std::string, size_t>> make_progress_header(
    supervised_learning_model_base& smodel, 
    const std::vector<std::string>& stat_headers, 
    bool has_validation_data) {

  auto header = std::vector<std::pair<std::string, size_t>>();
  for (const auto& s : stat_headers) {
    header.push_back({s, 8});
  }

  auto metrics = std::vector<std::string>();
  for (const auto& metric: smodel.get_tracking_metrics()) {
    metrics.push_back(metric);
  }

  for (const auto& m: metrics) {
    std::string dm = smodel.get_metric_display_name(m);
    header.push_back({std::string("Training ") + dm, 6});
    if (has_validation_data) 
      header.push_back({std::string("Validation ") + dm, 6});
  }

  return header;
}

inline std::vector<std::string> make_progress_row_string(
    supervised_learning_model_base& smodel,
    const ml_data& data,
    const ml_data& valid_data,
    const std::vector<std::string>& stats) {

  auto train_eval = std::vector<std::string>();
  for (auto& kv : smodel.evaluate(data, "train")) {
    train_eval.push_back(std::to_string(variant_get_value<double>(kv.second)));
  }

  auto valid_eval = std::vector<std::string>();
  bool has_validation_data = valid_data.num_rows() > 0;
  if (has_validation_data) {
    for (auto& kv : smodel.evaluate(valid_data, "train")) {
      valid_eval.push_back(std::to_string(variant_get_value<double>(kv.second)));
    }
  }

  auto ret = std::vector<std::string>();
  for (const auto& s : stats)
    ret.push_back(s);

  for (size_t i = 0 ; i < train_eval.size(); ++i) {
    ret.push_back(train_eval[i]);
    if (!valid_eval.empty()) {
      ret.push_back(valid_eval[i]);
    } else if(has_validation_data) {
      ret.push_back("");
    }
  }

  return ret;
}

/**
 * Get number of examples per class
 * 
 * \param[in] metadata
 * \returns Break down of examples per class.
 *
 * \warning For now, this only does it for binary classificaiton problems.
 *
 */
inline std::map<flexible_type, size_t> get_num_examples_per_class( 
                std::shared_ptr<ml_metadata> metadata){

  std::map<flexible_type, size_t> examples_per_class;
  for(size_t k = 0; k < metadata->target_index_size(); k++){
    examples_per_class[metadata->target_indexer()->map_index_to_value(k)] = 
                      metadata->target_statistics()->count(k);
  }
  return examples_per_class;
} 




/**
 * Get feature names from the metadata.
 * \param[in] metadata
 * \returns Names of features
 */
template <class T>
inline std::vector<std::string> get_feature_names_from_metadata(
    std::shared_ptr<T> metadata){

  std::vector<std::string> feature_names;
  for (size_t i = 0; i < metadata->num_columns(); ++i) {
    std::string name = metadata->column_name(i);

    // Vector
    if (metadata->column_type(i) == flex_type_enum::VECTOR){
      for (size_t j = 0; j < metadata->column_size(i); ++j) {
        std::string level = std::to_string(j);
        feature_names.push_back(name + std::string("[") + level + std::string("]"));
      }

    // Dict
    } else if (metadata->column_type(i) == flex_type_enum::DICT){
      for (size_t j = 0; j < metadata->column_size(i); ++j) {
        std::string level = (std::string)(
                metadata->indexer(i)->map_index_to_value(j));
        feature_names.push_back(name + std::string("[") + level + std::string("]"));
      }

    // Numeric
    } else {
      feature_names.push_back(name);
    }
  }

  return feature_names;
}

/**
 * Get feature names from the metadata.
 * \param[in] metadata
 * \returns Names of feature columns.
 */

inline std::vector<std::string> get_feature_column_names_from_metadata(
       std::shared_ptr<ml_metadata> metadata){
  return metadata->column_names();
}


/**
 * Get the number of coefficients from meta_data.
 * \param[in] metadata
 * \returns Number of coefficients.
 */

inline size_t get_number_of_coefficients(std::shared_ptr<ml_metadata> metadata){

  size_t num_coefficients = 1;
  for(size_t i = 0; i < metadata->num_columns(); i++) {
    if (metadata->is_categorical(i)) {
      num_coefficients += metadata->index_size(i) - 1;
    } else {
      num_coefficients += metadata->index_size(i);
    }
  }
  return num_coefficients;
}


/**
* Add a column of None values to the SFrame of coefficients.
*
* \returns coefs (as SFrame)
*/
inline sframe add_na_std_err_to_coef(const sframe& sf_coef) {
  auto sa = std::make_shared<sarray<flexible_type>>(
                   sarray<flexible_type>(FLEX_UNDEFINED, sf_coef.size(), 1,
                   flex_type_enum::FLOAT));
  return sf_coef.add_column(sa, std::string("stderr"));
}

/**
* Get one-hot-coefficients
*
* \params[in] coefs     Coefficients as EigenVector
* \params[in] metadata  Metadata
*
* \returns coefs (as SFrame)
*/
inline void get_one_hot_encoded_coefs(const arma::vec&
    coefs, std::shared_ptr<ml_metadata> metadata,
    std::vector<double>& one_hot_coefs) {

  size_t idx = 0;
  size_t num_classes = metadata->target_index_size();
  bool is_classifier = metadata->target_is_categorical();
  if (is_classifier) {
    num_classes -= 1; // reference class
  }

  for (size_t c = 0; c < num_classes; c++) {
    for (size_t i = 0; i < metadata->num_columns(); ++i) {
      // Categorical
      if (metadata->is_categorical(i)) {
        one_hot_coefs.push_back(0.0);
        // 0 is the reference
        for (size_t j = 1; j < metadata->index_size(i); ++j) {
          one_hot_coefs.push_back(coefs[idx++]);
        }
      // Vector
      } else if (metadata->column_type(i) == flex_type_enum::VECTOR){
        for (size_t j = 0; j < metadata->index_size(i); ++j) {
          one_hot_coefs.push_back(coefs[idx++]);
        }

      // Dict
      } else if (metadata->column_type(i) == flex_type_enum::DICT){
        for (size_t j = 0; j < metadata->index_size(i); ++j) {
          one_hot_coefs.push_back(coefs[idx++]);
        }
      } else {
        one_hot_coefs.push_back(coefs[idx++]);
      }
    }

    // Intercept
    one_hot_coefs.push_back(coefs[idx++]);
  }
}

/**
* Save coefficients to an SFrame, retrievable in Python
*
* \params[in] coefs     Coefficients as EigenVector
* \params[in] metadata  Metadata
*
* \returns coefs (as SFrame)
*/
inline sframe get_coefficients_as_sframe(
         const arma::vec& coefs,
         std::shared_ptr<ml_metadata> metadata, 
         const arma::vec& std_err) {

  DASSERT_TRUE(coefs.size() > 0);
  DASSERT_TRUE(metadata);

  // Classifiers need to provide target_metada to print out the class in 
  // the coefficients.
  bool is_classifier = metadata->target_is_categorical();
  bool has_stderr = std_err.size() > 0;
  DASSERT_EQ(std_err.size(), has_stderr * coefs.size());

  sframe sf_coef;
  std::vector<std::string> coef_names;
  coef_names.push_back("name");
  coef_names.push_back("index");
  if (is_classifier) coef_names.push_back("class");
  coef_names.push_back("value");
  if (has_stderr) coef_names.push_back("stderr");

  std::vector<flex_type_enum> coef_types;
  coef_types.push_back(flex_type_enum::STRING);
  coef_types.push_back(flex_type_enum::STRING);
  if (is_classifier) coef_types.push_back(metadata->target_column_type());
  coef_types.push_back(flex_type_enum::FLOAT);
  if (has_stderr) coef_types.push_back(flex_type_enum::FLOAT);

  sf_coef.open_for_write(coef_names, coef_types, "", 1);
  auto it_sf_coef = sf_coef.get_output_iterator(0);

  // Get feature names
  std::vector<flexible_type> feature_names;
  std::vector<flexible_type> feature_index;

  for (size_t i = 0; i < metadata->num_columns(); ++i) {
    std::string name = metadata->column_name(i);

    // Categorical
    if (metadata->is_categorical(i)) {
      // 0 is the reference
      for (size_t j = 1; j < metadata->index_size(i); ++j) {
        std::string level =
          std::string(metadata->indexer(i)->map_index_to_value(j));
        feature_names.push_back(name);
        feature_index.push_back(level);
      }
    // Vector
    } else if (metadata->column_type(i) == flex_type_enum::VECTOR){
      for (size_t j = 0; j < metadata->index_size(i); ++j) {
        std::string level = std::to_string(j);
        feature_names.push_back(name);
        feature_index.push_back(level);
      }

    // Dict
    } else if (metadata->column_type(i) == flex_type_enum::DICT){
      for (size_t j = 0; j < metadata->index_size(i); ++j) {
        std::string level = (std::string)metadata->indexer(i)->map_index_to_value(j);
        feature_names.push_back(name);
        feature_index.push_back(level);
      }
    } else {
      feature_names.push_back(name);
      feature_index.push_back(FLEX_UNDEFINED);
    }
  }

  // Classification
  if (is_classifier) {

    // GLC 1.0.1- did not save things as categorical variables.
    size_t num_classes = metadata->target_index_size();
    size_t variables_per_class = coefs.size() / (num_classes - 1);
    for(size_t k = 1; k < num_classes; k++){

      // Intercept
      std::vector<flexible_type> x(4 + has_stderr);
      x[0] = "(intercept)";
      x[1] = FLEX_UNDEFINED;
      x[2] = (metadata->target_indexer())->map_index_to_value(k);
      x[3] = coefs(variables_per_class * k - 1);
      if (has_stderr) x[4] = std_err(variables_per_class * k - 1);
      *it_sf_coef = x;
      ++it_sf_coef;

      // Write feature coefficients
      for (size_t i = 0; i < feature_names.size(); ++i) {
        x[0] = feature_names[i];
        x[1] = feature_index[i];
        x[2] = (metadata->target_indexer())->map_index_to_value(k);
        x[3] = coefs(variables_per_class * (k-1) + i);
        if (has_stderr) x[4] = std_err(variables_per_class * (k-1) + i);
        *it_sf_coef = x;
        ++it_sf_coef;
      }

    }

  // Regression
  } else {

    // Intercept
    std::vector<flexible_type> x(3 + has_stderr);
    x[0] = "(intercept)";
    x[1] = FLEX_UNDEFINED;
    x[2] = coefs(coefs.size() - 1);
    if (has_stderr) x[3] = std_err(std_err.size() - 1);
    *it_sf_coef = x;
    ++it_sf_coef;

    // Write feature coefficients
    for (size_t i = 0; i < feature_names.size(); ++i) {
      x[0] = feature_names[i];
      x[1] = feature_index[i];
      x[2] = coefs(i);
      if (has_stderr) x[3] = std_err(i);
      *it_sf_coef = x;
      ++it_sf_coef;
    }
  }
  sf_coef.close();
  return sf_coef;
}
inline sframe get_coefficients_as_sframe(
         const arma::vec& coefs,
         std::shared_ptr<ml_metadata> metadata) {
  arma::vec EMPTY;
  return get_coefficients_as_sframe(coefs, metadata, EMPTY); 
} 

/**
 * Get number of examples per class
 *
 * \param[in] target sarray
 * \returns Break down of examples per class.
 */
inline std::map<flexible_type, size_t>get_num_examples_per_class_from_sarray(
                                   std::shared_ptr<sarray<flexible_type>> sa){
  auto reader = sa->get_reader();
  std::map<flexible_type, size_t> unique_values;
  for(size_t seg_id = 0; seg_id < sa->num_segments(); seg_id++){
    auto iter = reader->begin(seg_id);
    auto enditer = reader->end(seg_id);
    while(iter != enditer) {
      if(unique_values.find(*iter) == unique_values.end()){
        unique_values.insert({*iter,0});
      } else {
        ++unique_values[*iter];
      }
      ++iter;
    }
  }
  return unique_values;
}

} // supervised
} // turicreate
#endif
