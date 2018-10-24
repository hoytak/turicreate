/* Copyright © 2017 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#ifndef TURI_RECSYS_MODEL_ITEMCF_H_
#define TURI_RECSYS_MODEL_ITEMCF_H_

#include <vector>
#include <string>
#include <unity/lib/extensions/option_manager.hpp>
#include <unity/toolkits/coreml_export/mlmodel_wrapper.hpp>
#include <unity/toolkits/recsys/recsys_model_base.hpp>
#include <generics/symmetric_2d_array.hpp>

#include <numerics/armadillo.hpp>
#include <limits>

namespace turi {

namespace v2 {
class ml_data;
}

class sframe;
class sgraph;
class flexible_type;
class sparse_similarity_lookup;

namespace recsys {

/**
 * This provides an implementation of a collaborative filtering algorithm.
 * The premise is to compute similarities (or distances) between all pairs
 * of items. Several choices of similarity will be available, and these
 * are functions of the list of users that were observed with the pair of
 * items. Some choices of similarity can also leverage a score that the
 * item was given by the user, e.g. a rating.
 *
 * In the following, let u(a) be the set of users who rated item a, let
 * E be the set of all (user, item) pairs, and let \f$ r_{u,i} \f$ be the
 * rating that user \f$ u \f$ gave to item \f$ i \f$.
 *
 * The three functions currently implemented are:
 * Jaccard similarity:
 *  Let \f$u(a) = {k: (k,a) in E}\f$. Then Jaccard similarity is defined by:
 * \f[J(a,b) = \frac{| u(a) \cap u(b) |}{ | u(a) \cup u(b) |} \f]
 *
 * Cosine similarity:
 * This compares the ratings given by all users who rated both items (where all
 * unobserved ratings \f$ r_{ua}\f$ are considered to be 0.
 * \f[ d(a,b) = \frac{\sum_{k} r_{ka} * r_{kb}}
 * {\sqrt{\sum_{k} r_{ka}^2} \sqrt{\sum_{k} r_{kb}^2}} \f]
 *
 * Pearson Correlation similarity:
 * A problem with Cosine similarity measure is that it does not consider the 
 * differences in the mean and variance of the ratings of items a and b.
 * Pearson Correlation is a popular measure where the effects of mean and variance
 * have been removed. 
 * Let \f$ u(a,b) = \{k: (k,a) \in E and (k,b) \in E\} \f$ denote the set of users who 
 * rated both items a and b.
 * \f[ d(a,b) = \frac{\sum_{k \in u(a,b)} (r_{ka} - \bar{r}_a) * (r_{kb} - \bar{r}_b)}
 * {\sqrt{\sum_{k \in u(a,b)} (r_{ka} - \bar{r}_a)^2} \sqrt{\sum_{k \in u(a,b)} (r_{kb} - \bar{r}_b)^2}} \f]
 * 
 * =============================================================
 * Implementation details:
 *
 * - Jaccard is implemented using two sufficient statistics:
 *    -# C(i): The number of times item i was rated.
 *    -# C(i,j): The number of times item i and j were rated by the same user.
 *    .
 *    The final similarity is computed by: \f$ C(i,j)/(C(i) + C(j) - C(i,j)) \f$
 *
 * - Cosine similarity is implemented using two statistics.
 *   -# C(i): The sum of squared ratings given to item i.
 *   -# C(i,j): The sum of products of the ratings for all users who rated both i and j.
 *   .
 *    The distance is computed as: \f$ d(i,j) = \frac{C(i,j)}{\sqrt{C(i)C(j)}} \f$
 *
 * - Pearson Correlation similarity is implemented using three statistics.
 *   -# C(i) : the variance of ratings given to item i. 
 *   -# C(i,j) : the sum of correlation score by all users who rated both items i and j
 *   . 
 *    The final similarity is computed by: \f$ d(i,j) = \frac{C(i,j)}{\sqrt{C(i)C(j)}} \f$
 * 
 * Details of computing item similarities:
 *
 * 1. Get the individual statistics of items C(i)
 * 2. For each pair of items i and j that both rated by user u, update the overlapping 
 *    statistics C(i,j).
 * 3. Get the final score matrix by normalizing C(i,j) with individual statistics
 * 4. Sort each row of the score matrix to get top-k similar items
 *
 */

class recsys_itemcf : public recsys_model_base {
 public:
  bool use_target_column(bool target_is_present) const override {
    return target_is_present;
  }

  static constexpr size_t ITEMCF_VERSION = 2;

  void init_options(const std::map<std::string, flexible_type>& options) override;

 private:
  /** Handling extra data given by the
   *
   */
  void set_extra_data(const std::map<std::string, variant_type>& extra_data) override;
  void load_user_provided_data();

  struct user_provided_data_struct {
    sframe nearest_items;
  };

  std::shared_ptr<user_provided_data_struct> user_provided_data;

 public:
  /**
   *  When the number of items is less than 20k, it uses in memory computations train_in_memory().
   *  Otherwise, it uses the implementation based on SGraph train_using_sgraph().
   */
  std::map<std::string, flexible_type> train(const v2::ml_data& data) override;

  /**
   * During the predict phase, we perform the "vector matrix product"
   * where we compute a score for a particular (user, item) pair.
   * This score is a sum of similarities between an item and all the items
   * observed for the given user. For similarity functions that incorporate some
   * target value for each (user, item) pair, this prediction also
   * multiples each similarity by that value, e.g. a rating they gave the
   * item in question.
   */
  sframe predict(const v2::ml_data& test_data) const override;
  
  std::vector<double> predict_all_items(
      const std::vector<flexible_type>& base_observation) const;

private:
  mutable atomic<int> user_item_buffers_setup;
  mutable std::vector<std::vector<std::pair<size_t, double> > > user_item_buffers_by_thread;
  mutable mutex init_user_item_buffers_lock;

  // The internal function for calling the score. 
  void _score_items(std::vector<std::pair<size_t, double> >& item_scores, 
                    const std::vector<std::pair<size_t, double> >& user_scores) const;
  
public:
  /** For a given base observation, predict the score for all the
   * items with all non-item columns replaced by the values in the
   * base observation.
   *
   * The base_observation vector is used to generate all the
   * observations predicted.  New observations are generated by
   * repeatedly copying template_observation, then replacing the
   * values in item_column_index by each possible item value.
   */
 void score_all_items(
      std::vector<std::pair<size_t, double> >& scores,
      const std::vector<v2::ml_data_entry>& query_row,
      size_t top_k,
      const std::vector<std::pair<size_t, double> >& user_item_list,
      const std::vector<std::pair<size_t, double> >& new_user_item_data,
      const std::vector<v2::ml_data_row_reference>& new_observation_data,
      const std::shared_ptr<v2::ml_data_side_features>& known_side_features) const override;
  
 /**
   * Utilities
   */
  std::string response_column_name() const;

  inline size_t internal_get_version() const override {
    return ITEMCF_VERSION;
  }

  void internal_save(turi::oarchive& oarc) const override;
  void internal_load(turi::iarchive& iarc, size_t version) override;

 private:

  /**  The primary tool for the item similarity modeling part.
   */
  std::shared_ptr<sparse_similarity_lookup> item_sim;

  // For completely new users, keep track of some of the popular items
  // and use these to seed predictions for them.  in addition, the
  // mean score is also dealt with. 
  std::vector<std::pair<size_t, double> > new_user_seed_items;
  std::vector<double> item_mean_score;
  double item_mean_min=0, item_mean_max=0; 
  
 public:

  /**
   *  Get the nearest neighbors of a set of items.
   *  
   *  \param[in] indexed_items A SArray of items in flexible_type
   *  \param[in] topk Number of neighbors returned for each item
   *  \returns A SFrame with columns {"item", "similar", "score", "rank"}
   */
  sframe get_similar_items(
    std::shared_ptr<sarray<flexible_type> > items, size_t topk=0) const override;

  /**
   *  Get the nearest neighbors of a set of users.
   *
   *  \param[in] indexed_users A SArray of users in flexible_type
   *  \param[in] topk Number of neighbors returned for each item
   *  \returns A SFrame with columns {"user", "similar", "score", "rank"}
   */
  sframe get_similar_users(
    std::shared_ptr<sarray<flexible_type> > items, size_t topk=0) const override {
    log_and_throw("get_similar_users currently not supported for item similarity models. "
                  "To get the neighborhood of users, train a model with the items and users reversed, "
                  "then call get_similar_items.");
    
    return sframe();
  }

  virtual void export_to_coreml(std::shared_ptr<recsys_model_base> recsys_model,
                                const std::string& filename) override;
  
 private:
  
  void make_user_item_graph(const v2::ml_data& data,
                            const std::shared_ptr<sarray<flex_dict> >& user_item_lists,
                            sgraph& g);

  std::shared_ptr<sparse_similarity_lookup> create_similarity_lookup() const;
 
 public: 
  // TODO: convert interface above to use the extensions methods here
  BEGIN_CLASS_MEMBER_REGISTRATION("item_similarity")
  REGISTER_CLASS_MEMBER_FUNCTION(recsys_itemcf::list_fields)


  REGISTER_NAMED_CLASS_MEMBER_FUNCTION(
      "get_value", recsys_itemcf::get_value_from_state, "field"); //"get_value"
  REGISTER_NAMED_CLASS_MEMBER_FUNCTION(
      "get_similar_items", recsys_itemcf::api_get_similar_items, "items", "k", "verbose", "get_all_items");
  REGISTER_NAMED_CLASS_MEMBER_FUNCTION(
      "get_similar_users", recsys_itemcf::api_get_similar_users, "users", "k", "get_all_users");
  REGISTER_NAMED_CLASS_MEMBER_FUNCTION(
      "predict", recsys_itemcf::api_predict, "data_to_predict", "new_user_data", "new_item_data");
  REGISTER_NAMED_CLASS_MEMBER_FUNCTION(
      "get_current_options", recsys_itemcf::api_get_current_options);
  REGISTER_NAMED_CLASS_MEMBER_FUNCTION(
    "set_current_options", recsys_itemcf::api_set_current_options, "options");
  REGISTER_NAMED_CLASS_MEMBER_FUNCTION(
    "get_train_stats", recsys_itemcf::api_get_train_stats);
  REGISTER_NAMED_CLASS_MEMBER_FUNCTION(
    "train_test_split", recsys_itemcf::api_train_test_split, "dataset", "user_column", "item_column",
    "max_num_users", "item_test_proportion", "random_seed");
  REGISTER_NAMED_CLASS_MEMBER_FUNCTION(
    "train", recsys_itemcf::api_train, "dataset", "user_data", "item_data", "nearest_items", "opts");
  REGISTER_NAMED_CLASS_MEMBER_FUNCTION(
    "recommend", recsys_itemcf::api_recommend, "query", "exclude", "restrictions", "new_data", "new_user_data",
    "new_item_data", "exclude_training_interactions", "top_k", "diversity", "random_seed");
  // //REGISTER_NAMED_CLASS_MEMBER_FUNCTION(
  //  "get_popularity_baseline", recsys_itemcf::get_popularity_baseline);
  REGISTER_NAMED_CLASS_MEMBER_FUNCTION(
    "get_item_intersection_info", recsys_itemcf::api_get_item_intersection_info, "item_pairs");
  REGISTER_NAMED_CLASS_MEMBER_FUNCTION(
    "export_to_coreml", recsys_itemcf::export_to_coreml, "model", "filename"); ///
  REGISTER_NAMED_CLASS_MEMBER_FUNCTION(
    "precision_recall_stats", recsys_itemcf::api_precision_recall_stats, "indexed_validation_data", "recommend_output", "cutoffs");

  REGISTER_NAMED_CLASS_MEMBER_FUNCTION(
    "get_data_schema", recsys_itemcf::api_get_data_schema);


  REGISTER_NAMED_CLASS_MEMBER_FUNCTION(
      "get_value", recsys_itemcf::get_value_from_state, "field");
  REGISTER_CLASS_MEMBER_FUNCTION(recsys_itemcf::recommend_extension_wrapper, 
    "reference_data", "new_observation_data", "top_k")
  END_CLASS_MEMBER_REGISTRATION
}; 

}}

#endif


