


/**
 * An enumeration over the possible types of prediction that are supported.
 * \see prediction_type_enum_from_name 
 */
enum class prediction_type_enum: char {
  NA = 0,                 /**< NA: Default/Not-applicable.*/       
  CLASS = 1, 
  CLASS_INDEX = 2,        /**< Index of the class (performance reasons) .*/       
  PROBABILITY = 3, 
  MAX_PROBABILITY = 4,    /**< Max probability for classify .*/       
  MARGIN = 5, 
  RANK = 6, 
  PROBABILITY_VECTOR = 7, /** < A vector of probabilities .*/ 
  FEATURE_VECTOR = 8      /** < A bunch of feature vectors. */
};

/**
 * Given the printable name of a prediction_type_enum type, it returns the name.
 * 
 * \param[in] name Name of the prediction_type_enum type.
 * \returns prediction_type_enum 
 */
inline prediction_type_enum prediction_type_enum_from_name(const std::string& name) {
  static std::map<std::string, prediction_type_enum> type_map{
    {"na", prediction_type_enum::NA},
    {"", prediction_type_enum::NA},
    {"class", prediction_type_enum::CLASS},
    {"class_index", prediction_type_enum::CLASS_INDEX},
    {"probability", prediction_type_enum::PROBABILITY},
    {"max_probability", prediction_type_enum::MAX_PROBABILITY},
    {"margin", prediction_type_enum::MARGIN},
    {"rank", prediction_type_enum::RANK},
    {"probability_vector", prediction_type_enum::PROBABILITY_VECTOR},
  };
  if (type_map.count(name) == 0) {
    log_and_throw(std::string("Invalid prediction type name " + name));
  }
  return type_map.at(name);
}


