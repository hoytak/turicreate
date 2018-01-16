#ifndef TURI_CAPI_H
#define TURI_CAPI_H

#ifdef __cplusplus
extern "C" { 
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/******************************************************************************/
/*                                                                            */
/*    CLASS DECLARATIONS                                                      */
/*                                                                            */
/******************************************************************************/

// Error message struct
struct tc_error_struct; 
typedef struct tc_error_struct tc_error; 

// Flexible type -- holds numeric, string, array, list, datetime, and image
// types for use in SFrame or SArray.
struct tc_flexible_type_struct; 
typedef struct tc_flexible_type_struct tc_flexible_type;

// flex_list -- list of flexible types
struct tc_flex_list_struct;
typedef struct tc_flex_list_struct tc_flex_list;

// flex_dict -- list of key/value pairs of flexible types
struct tc_flex_dict_struct;
typedef struct tc_flex_dict_struct tc_flex_dict;

// datetime
struct tc_datetime_struct;
typedef struct tc_datetime_struct tc_datetime;

// Image
struct tc_flex_image_struct;
typedef struct tc_flex_image_struct tc_flex_image;

// SArray
struct tc_sarray_struct; 
typedef struct tc_sarray_struct tc_sarray; 

// SFrame
struct tc_sframe_struct; 
typedef struct tc_sframe_struct tc_sframe; 

// Variant type -- extends flexible type; holds sarrays, sframes, and models as well
struct tc_variant_struct;
typedef struct tc_variant_struct tc_variant;

// Parameters -- map of string to variant type
struct tc_parameters_struct;
typedef struct tc_parameters_struct tc_parameters;

// Model offering predictions
struct tc_model_struct; 
typedef struct tc_model_struct tc_model;

/******************************************************************************/
/*                                                                            */
/*    INITIALIZATION                                                          */
/*                                                                            */
/******************************************************************************/

/** Initialize the framework. Call before calling any previous function.
 *
 */
void tc_initialize(const char* log_file, tc_error**); 

/******************************************************************************/
/*                                                                            */
/*    ERROR HANDLING                                                          */
/*                                                                            */
/******************************************************************************/


  /**********************

   // Example Error checking code 
    
   tc_error *error = NULL; 

   tc_create_flexible_type_str("hello", &error); 

   if(error) {   
      const char* msg = tc_error_message(error); 

      // ...

      tc_error_destroy(&error); 
   }

  *************************/

/** Retrieves the error message on an active error.
 *
 *  Return object is a null-terminated c-style message string. 
 *
 *  The char buffer returned is invalidated by calling tc_error_destroy. 
 */
const char* tc_error_message(const tc_error* error); 


/** Destroys an error structure, deallocating error content data. 
 *
 *  Only needs to be called if an error occured.
 *
 *  Sets the pointer to the error struct to NULL. 
 */
void tc_error_destroy(tc_error** error_ptr);




/******************************************************************************/
/*                                                                            */
/*    FLEXIBLE TYPE                                                           */
/*                                                                            */
/******************************************************************************/


/*****************************************************/
/* Creating flexible type                            */
/*****************************************************/

tc_flexible_type* tc_ft_create_empty(tc_error** error);
tc_flexible_type* tc_ft_create_copy(const tc_flexible_type*, tc_error** error);
tc_flexible_type* tc_ft_create_from_int64(int64_t, tc_error** error);
tc_flexible_type* tc_ft_create_from_double(double, tc_error** error);
tc_flexible_type* tc_ft_create_from_cstring(const char* str, tc_error** error);
tc_flexible_type* tc_ft_create_from_string(const char* str, uint64_t n, tc_error** error);
tc_flexible_type* tc_ft_create_from_double_array(const double* data, uint64_t n, tc_error** error);
tc_flexible_type* tc_ft_create_from_flex_list(const tc_flex_list*, tc_error** error);
tc_flexible_type* tc_ft_create_from_flex_dict(const tc_flex_dict*, tc_error** error);
tc_flexible_type* tc_ft_create_from_datetime(const tc_datetime* dt, tc_error**);
tc_flexible_type* tc_ft_create_from_image(const tc_flex_image*, tc_error** error);


/*****************************************************/
/* Testing types in flexible type                    */
/*****************************************************/

/** Type enum. */
typedef enum {
  FT_TYPE_INTEGER = 0,  
  FT_TYPE_DOUBLE  = 1,   
  FT_TYPE_STRING  = 2,   
  FT_TYPE_ARRAY   = 3, 
  FT_TYPE_LIST    = 4, 
  FT_TYPE_DICT    = 5, 
  FT_TYPE_DATETIME = 6, 
  FT_TYPE_UNDEFINED = 7,
  FT_TYPE_IMAGE   = 8
} tc_ft_type_enum; 

tc_ft_type_enum tc_ft_type(const tc_flexible_type*);

int tc_ft_is_double(const tc_flexible_type*);
int tc_ft_is_int64(const tc_flexible_type*); 
int tc_ft_is_string(const tc_flexible_type*); 
int tc_ft_is_array(const tc_flexible_type*); 
int tc_ft_is_list(const tc_flexible_type*); 
int tc_ft_is_dict(const tc_flexible_type*); 
int tc_ft_is_datetime(const tc_flexible_type*); 
int tc_ft_is_undefined(const tc_flexible_type*); 
int tc_ft_is_image(const tc_flexible_type*); 
int tc_ft_is_datetime(const tc_flexible_type*);

/*****************************************************/
/* Extracting values from flexible type              */
/*****************************************************/

int64_t tc_ft_int64(const tc_flexible_type* ft, tc_error** error); 

double tc_ft_double(const tc_flexible_type* ft, tc_error** error); 

uint64_t tc_ft_string_length(const tc_flexible_type* ft, tc_error** error); 
const char* tc_ft_string_data(const tc_flexible_type* ft, tc_error** error);

uint64_t tc_ft_array_length(const tc_flexible_type* ft, tc_error** error); 
const double* tc_ft_array_data(const tc_flexible_type* ft, tc_error** error);

tc_flex_list* tc_ft_flex_list(const tc_flexible_type*, tc_error**);
tc_flex_dict* tc_ft_flex_dict(const tc_flexible_type*, tc_error**);
tc_datetime* tc_ft_datetime(const tc_flexible_type* dt, tc_error**);
tc_flex_image* tc_ft_flex_image(const tc_flexible_type*, tc_error**); 

/*****************************************************/
/*    Casting flexible types                         */
/*****************************************************/

// Cast any type to a string.  Sets the error and returns NULL if it's not possible.
// Casting to string can be used to print the value. 
tc_flexible_type* tc_ft_as_string(const tc_flexible_type*, tc_error** error);

/*****************************************************/
/*    Destructor                                     */
/*****************************************************/

void tc_ft_destroy(tc_flexible_type*); 

/******************************************************************************/
/*                                                                            */
/*    flex_list                                                               */
/*                                                                            */
/******************************************************************************/


tc_flex_list* tc_flex_list_create(tc_error**);
tc_flex_list* tc_flex_list_create_with_capacity(uint64_t capacity, tc_error**);

uint64_t tc_flex_list_add_element(tc_flex_list*, const tc_flexible_type*, tc_error**);

tc_flexible_type* tc_flex_list_extract_element(
    const tc_flex_list*, uint64_t index, tc_error**);

uint64_t tc_flex_list_size(const tc_flex_list*);

void tc_flex_list_destroy(tc_flex_list*);


/******************************************************************************/
/*                                                                            */
/*    flex_dict                                                               */
/*                                                                            */
/******************************************************************************/

// NOTE: flex_dicts are simply key-value lists; lookup-by-key is not efficient
// and thus not implemented.

// Creates an empty flex_dict object.
tc_flex_dict* tc_flex_dict_create(tc_error**);

// Adds a key to the dictionary, returning the entry index.. 
uint64_t tc_flex_dict_add_element(tc_flex_dict* ft, const tc_flexible_type* first, const tc_flexible_type* second, tc_error**);

// Extract the (key, value) pair corresponding to the entry at entry_index. 
void tc_flex_dict_extract_entry(const tc_flex_dict* ft, uint64_t entry_index, tc_flexible_type* key_dest, tc_flexible_type* value_dest, tc_error**); 

// Destroy the dictionary. 
void tc_flex_dict_destroy(tc_flex_dict*);

/******************************************************************************/
/*                                                                            */
/*    flex_datetime                                                           */
/*                                                                            */
/******************************************************************************/

tc_datetime* tc_datetime_create_empty(tc_error**);

// Create and set a datetime object from a posix timestamp value -- 
// the number of seconds since January 1, 1970, UTC. 
tc_datetime* tc_datetime_create_from_posix_timestamp(int64_t posix_timestamp, tc_error**);

// Create and set a datetime object from a high res posix timestamp value -- 
// the number of seconds since January 1, 1970, UTC, in double precision. 
tc_datetime* tc_datetime_create_from_posix_highres_timestamp(double posix_timestamp, tc_error**);

// Set the datetime value from a string timestamp of the date and/or time.
tc_datetime* tc_datetime_create_from_string(const char* datetime_str, tc_error**);

// Set and get the time zone.  The time zone has 15 min resolution.  
void tc_datetime_set_time_zone_offset(tc_datetime* dt, int64_t n_tz_hour_offset, int64_t n_tz_15min_offsets, tc_error**);
int64_t tc_datetime_get_time_zone_offset_minutes(const tc_datetime* dt, tc_error**);  

// Set and get the microsecond part of the time zone.
void tc_datetime_set_microsecond(tc_datetime* dt, uint64_t microseconds, tc_error**);
uint64_t tc_datetime_get_microsecond(const tc_datetime* dt, tc_error**);

// Set and get the posix style timestamp -- number of seconds since January 1, 1970, UTC. 
void tc_datetime_set_timestamp(tc_datetime* dt, int64_t d, tc_error**);
int64_t tc_datetime_get_timestamp(tc_datetime* dt, tc_error**);

// Set and get the posix style timestamp with high res counter -- number of seconds since January 1, 1970, UTC.  
void tc_datetime_set_highres_timestamp(tc_datetime* dt, double d, tc_error**);
double tc_datetime_get_highres_timestamp(tc_datetime* dt, tc_error**);

// Returns true if the time dt1 is before the time dt2 
bool tc_datetime_less_than(const tc_datetime* dt1, const tc_datetime* dt2, tc_error**); 

// Returns true if the time dt1 is equal to the time dt2
bool tc_datetime_equal(const tc_datetime* dt1, const tc_datetime* dt2, tc_error**); 

// Destructor
void tc_datetime_destroy(tc_datetime*); 


/******************************************************************************/
/*                                                                            */
/*    flex_image                                                              */
/*                                                                            */
/******************************************************************************/

// Load an image into a flexible type from a path
tc_flex_image* tc_flex_image_create_from_path(
    const char* path, const char* format, tc_error** error);

// Load an image into a flexible type from raw data
tc_flex_image* tc_flex_image_create_from_data(
    const char* data, uint64_t height, uint64_t width, uint64_t channels, 
    uint64_t total_data_size, const char* format, tc_error** error); 

// Methods to query the image size and width
uint64_t tc_flex_image_width(const tc_flex_image*, tc_error**); 
uint64_t tc_flex_image_height(const tc_flex_image*, tc_error**); 
uint64_t tc_flex_image_num_channels(const tc_flex_image*, tc_error**); 
uint64_t tc_flex_image_data_size(const tc_flex_image*, tc_error**); 
const char* tc_flex_image_data(const tc_flex_image*, tc_error**); 
const char* tc_flex_image_format(const tc_flex_image*, tc_error**); 

// Destructor
void tc_flex_image_destroy(tc_flex_image*); 


/******************************************************************************/
/*                                                                            */
/*    SARRAY                                                                  */
/*                                                                            */
/******************************************************************************/


tc_sarray* tc_sarray_create_empty(tc_error**);

tc_sarray* tc_sarray_create_from_data(const tc_flex_list* data, tc_error**);

tc_sarray* tc_sarray_create_from_sequence(
    uint64_t start, uint64_t end, tc_error** error);

tc_sarray* tc_sarray_create_from_const(
    const tc_flexible_type* value, uint64_t n, tc_error** error);

tc_sarray* tc_sarray_create_from_list(
    const tc_flex_list* values, tc_error** error);

tc_sarray* tc_sarray_create_copy(const tc_sarray* src, tc_error** error);

// Save in native format
void tc_sarray_save(const tc_sarray* src, const char* directory, const char* format, tc_error**);

// Gets a particular element.
tc_flexible_type* tc_sarray_extract_element(const tc_sarray*, uint64_t index, tc_error**);

// Gets the sarry size. 
uint64_t tc_sarray_size(const tc_sarray*, tc_error**); 

// Gets the type of the sarray. 
tc_ft_type_enum tc_sarray_type(const tc_sarray*); 

// Gets the type of the sarray. 
tc_sarray* tc_op_sarray_plus_sarray(const tc_sarray*, const tc_sarray*, tc_error**); 
tc_sarray* tc_op_sarray_minus_sarray(const tc_sarray*, const tc_sarray*, tc_error**); 
tc_sarray* tc_op_sarray_div_sarray(const tc_sarray*, const tc_sarray*, tc_error**); 
tc_sarray* tc_op_sarray_mult_sarray(const tc_sarray*, const tc_sarray*, tc_error**); 
tc_sarray* tc_op_sarray_plus_ft(const tc_sarray*, const tc_flexible_type*, tc_error**); 
tc_sarray* tc_op_sarray_minus_ft(const tc_sarray*, const tc_flexible_type*, tc_error**); 
tc_sarray* tc_op_sarray_div_ft(const tc_sarray*, const tc_flexible_type*, tc_error**); 
tc_sarray* tc_op_sarray_mult_ft(const tc_sarray*, const tc_flexible_type*, tc_error**); 

// Returns 1 if all elements are equal and 0 otherwise.
int tc_sarray_equals(const tc_sarray*, const tc_sarray*, tc_error**);

// Wrap the printing.  Returns a string flexible type.
tc_flexible_type* tc_sarray_text_summary(const tc_sarray* sf, tc_error**);

// To implement.
tc_sarray* tc_op_sarray_lt_sarray(const tc_sarray*, const tc_sarray*, tc_error**); 
tc_sarray* tc_op_sarray_gt_sarray(const tc_sarray*, const tc_sarray*, tc_error**); 
tc_sarray* tc_op_sarray_le_sarray(const tc_sarray*, const tc_sarray*, tc_error**); 
tc_sarray* tc_op_sarray_ge_sarray(const tc_sarray*, const tc_sarray*, tc_error**); 
tc_sarray* tc_op_sarray_eq_sarray(const tc_sarray*, const tc_sarray*, tc_error**); 

tc_sarray* tc_op_sarray_lt_ft(const tc_sarray*, const tc_flexible_type*, tc_error**); 
tc_sarray* tc_op_sarray_gt_ft(const tc_sarray*, const tc_flexible_type*, tc_error**); 
tc_sarray* tc_op_sarray_ge_ft(const tc_sarray*, const tc_flexible_type*, tc_error**); 
tc_sarray* tc_op_sarray_le_ft(const tc_sarray*, const tc_flexible_type*, tc_error**); 
tc_sarray* tc_op_sarray_eq_ft(const tc_sarray*, const tc_flexible_type*, tc_error**); 

tc_sarray* tc_op_sarray_lt_ft(const tc_sarray*, const tc_flexible_type*, tc_error**); 
tc_sarray* tc_op_sarray_gt_ft(const tc_sarray*, const tc_flexible_type*, tc_error**); 
tc_sarray* tc_op_sarray_ge_ft(const tc_sarray*, const tc_flexible_type*, tc_error**); 
tc_sarray* tc_op_sarray_le_ft(const tc_sarray*, const tc_flexible_type*, tc_error**); 
tc_sarray* tc_op_sarray_eq_ft(const tc_sarray*, const tc_flexible_type*, tc_error**); 

tc_sarray* tc_op_sarray_logical_and_sarray(const tc_sarray*, const tc_sarray*, tc_error**); 
tc_sarray* tc_op_sarray_logical_and_ft(const tc_sarray*, const tc_flexible_type*, tc_error**); 
tc_sarray* tc_op_sarray_bitwise_and_sarray(const tc_sarray*, const tc_sarray*, tc_error**); 
tc_sarray* tc_op_sarray_bitwise_and_ft(const tc_sarray*, const tc_flexible_type*, tc_error**); 

tc_sarray* tc_op_sarray_logical_or_sarray(const tc_sarray*, const tc_sarray*, tc_error**); 
tc_sarray* tc_op_sarray_logical_or_ft(const tc_sarray*, const tc_flexible_type*, tc_error**); 
tc_sarray* tc_op_sarray_bitwise_or_sarray(const tc_sarray*, const tc_sarray*, tc_error**); 
tc_sarray* tc_op_sarray_bitwise_or_ft(const tc_sarray*, const tc_flexible_type*, tc_error**);

// gl_sarray operator[](const gl_sarray& slice);
tc_sarray* tc_sarray_apply_mask(const tc_sarray* src, const tc_sarray* mask, tc_error**); 
  
int tc_sarray_all_nonzero(const tc_sarray*, tc_error**);

int tc_sarray_any_nonzero(const tc_sarray*, tc_error**);

// Force materialization
void tc_sarray_materialize(const tc_sarray* src, tc_error**);

tc_sarray* tc_sarray_head(const tc_sarray* src, uint64_t n, tc_error**);
tc_sarray* tc_sarray_tail(const tc_sarray* src, uint64_t n, tc_error**);

//gl_sarray count_words(bool to_lower=true, turi::flex_list delimiters={"\r", "\v", "\n", "\f", "\t", " "});
tc_sarray* tc_sarray_count_words(const tc_sarray*, int to_lower, tc_error**); 
tc_sarray* tc_sarray_count_words_with_delimiters(const tc_sarray*, int to_lower, tc_flex_list* delimiters, tc_error**); 

// wraps  gl_sarray count_ngrams(size_t n=2, std::string method="word", 
//                         bool to_lower=true, bool ignore_space=true);
tc_sarray* tc_sarray count_word_ngrams(const tc_sarray* src, uint64_t n, bool to_lower, tc_error**);

tc_sarray* tc_sarray count_character_ngrams(const tc_sarray* src, size_t n, bool to_lower, bool ignore_space, tc_error**);

tc_sarray* tc_sarray_dict_trim_by_keys(const tc_sarray* src, const tc_flex_list* keys, int exclude_keys, tc_error**); 

tc_sarray* tc_sarray_dict_trim_by_value_range(const tc_sarray* src, const tc_flexible_type* lower, const tc_flexible_type* upper, tc_error**);

tc_flexible_type* tc_sarray_max(const tc_sarray*, tc_error**);

tc_flexible_type* tc_sarray_min(const tc_sarray*, tc_error**);

tc_flexible_type* tc_sarray_sum(const tc_sarray*, tc_error**);

tc_flexible_type* tc_sarray_mean(const tc_sarray*, tc_error**);

tc_flexible_type* tc_sarray_std(const tc_sarray*, tc_error**);

uint64_t tc_sarray_nnz(const tc_sarray*, tc_error**);

size_t tc_sarray_num_missing(const tc_sarray*, tc_error**);

tc_sarray* tc_sarray_dict_keys(const tc_sarray* src, tc_error**);

tc_sarray* tc_sarray_dict_has_any_keys(const tc_sarray* src, const tc_flex_list* keys, tc_error**);

tc_sarray* tc_sarray_dict_has_all_keys(const tc_sarray* src, const tc_flex_list* keys, tc_error**);

tc_sarray* tc_sarray_sample(const tc_sarray* src, double fraction, uint64_t seed, tc_error**);

tc_sarray* tc_sarray_datetime_to_str_with_format(const tc_sarray* src, const char* format, tc_error**);

tc_sarray* tc_sarray_datetime_to_str(const tc_sarray* src, tc_error**);

tc_sarray* tc_sarray_str_to_datetime(const tc_sarray* src, const char* format, tc_error**);

tc_sarray* tc_sarray_astype(const tc_sarray* src, flex_type_enum dtype, int undefined_on_failure, tc_error**);

tc_sarray* tc_sarray_clip(const tc_sarray* src, tc_flexible_type* lower, tc_flexible_type* upper, tc_error**);

tc_sarray* tc_sarray_drop_nan(const tc_sarray* src, tc_error**);

tc_sarray* tc_sarray_replace_nan(const tc_sarray* src, tc_flexible_type* value, tc_error**);

tc_sarray* tc_sarray_topk_index(const tc_sarray* src, size_t topk, int reverse, tc_error**);
  
tc_sarray* tc_sarray_append(const tc_sarray* other, tc_error**);

tc_sarray* tc_sarray_unique(const tc_sarray* src, tc_error**);


void materialize_to_callback(
      std::function<bool(size_t, const std::shared_ptr<sframe_rows>&)> callback,
      size_t nthreads = (size_t)(-1));

  
int tc_sarray_is_materialized(const tc_sarray* src, tc_error**);

void tc_sarray_materialize(const tc_sarray* other, tc_error**);

////////////////////////////////////////////////
// General purpose application functions.

// SArray Apply. The flexible_type* returned is not automatically freed
tc_sarray* tc_sarray_apply(const tc_sarray*,
                           tc_flexible_type*(*callback)(tc_flexible_type*, void* userdata),
                           void (*userdata_release_callback)(void* userdata),
                           void* userdata,
                           tc_ft_type_enum type,
                           bool skip_undefined,
                           tc_error**);

// Destructor
void tc_sarray_destroy(tc_sarray* sa);



/******************************************************************************/
/*                                                                            */
/*   SFRAME                                                                   */
/*                                                                            */
/******************************************************************************/

tc_sframe* tc_sframe_create_empty(tc_error**);

tc_sframe* tc_sframe_create_copy(tc_sframe*, tc_error**);

// Adds the column to the sframe.
void tc_sframe_add_column(tc_sframe* sf, const char* column_name, 
    const tc_sarray* sarray, tc_error**);

// Remove a certain column.
void tc_sframe_remove_column(tc_sframe* sf, const char* column_name, tc_error**);

tc_sarray* tc_sframe_extract_column_by_name(
    tc_sframe* sf, const char* column_name, tc_error**);

// Wrap the printing.  Returns a string flexible type. 
tc_flexible_type* tc_sframe_text_summary(const tc_sframe* sf, tc_error**);

// Number of rows 
uint64_t tc_sframe_num_rows(const tc_sframe* sf, tc_error**); 

// Number of columns.
uint64_t tc_sframe_num_columns(const tc_sframe* sf, tc_error**); 

// Return the name of a particular column.
const char* tc_sframe_column_name(const tc_sframe* sf, size_t column_index, tc_error**);

// Return the type of a particular column.
tc_ft_type_enum tc_sframe_column_type(const tc_sframe* sf, const char* column_name, tc_error**); 

// Return all column types as a list.
tc_flex_list* tc_sframe_column_names(const tc_sframe* sf, tc_error**);

// Read csv
tc_sframe* tc_sframe_read_csv(const char *url, tc_error**);

// Write csv
void tc_sframe_write_csv(const tc_sframe* sf, const char *url, tc_error **error);

// Write other format
void tc_sframe_write(const tc_sframe* sf, const char *url, const char *format, tc_error **error);

// Head
tc_sframe* tc_sframe_head(const tc_sframe* sf, size_t n, tc_error **error);

// Tail
tc_sframe* tc_sframe_tail(const tc_sframe* sf, size_t n, tc_error **error);

// Random split.
void tc_sframe_random_split(const tc_sframe* sf, double proportion, 
                            size_t seed, tc_sframe** left, tc_sframe** right, tc_error **error);
  

void save(const char* path, const char* format="");


// Whizbangery
//
// Join two sframes.  
//
// column is the column name to join on. 
// how is "inner", "outer", "left", or "right" 
tc_sframe* tc_sframe_join_on_single_column(
    tc_sframe* left, tc_sframe* right, 
    const char* column, 
    const char* how, tc_error**);

// Join two sframes.  
//
// column is the column name to join on. 
// how is "inner", "outer", "left", or "right" 
tc_sframe* tc_sframe_join_on_multiple_columns(
    tc_sframe* left, tc_sframe* right, 
    const tc_flex_list* column, 
    const char* how, tc_error**);

// Append one sframe onto another.
tc_sframe* tc_sframe_append(const tc_sframe* top, tc_sframe* bottom, tc_error **error);

// wrap versions oftc_sframe* operator[](const std::initializer_list<int64_t>& slice);
tc_sframe* tc_sframe_slice(const tc_sframe* src, uint64_t row_index_start, uint64_t row_index_end, tc_error**); 
   
tc_sframe* tc_sframe_slice_with_stride(const tc_sframe* src, uint64_t row_index_start, uint64_t row_index_end, uint64_t stride, tc_error**); 
 
int tc_sframe_is_materialized(const tc_sframe* src, tc_error**);

void tc_sframe_materialize(tc_sframe* src, tc_error**);

int tc_sframe_size_is_known(const tc_sframe* src, tc_error**);

void tc_sframe_save_reference(const tc_sframe*, const char* path, tc_error**);

bool tc_sframe_contains_column(const tc_sframe*, const char* col_name, tc_error**);

tc_sframe* tc_sframe_sample(const tc_sframe*, double fraction, uint64_t seed, tc_error**);

tc_sframe* tc_sframe_topk(const tc_sframe* src, const char* column_name, uint64_t k, bool reverse, tc_error**);

void tc_sframe_replace_add_column(const tc_sframe* sf, const char* name, const tc_sarray* new_column, tc_error**);

void tc_sframe_add_constant_column(tc_sframe* sf, const char* column_name, const tc_flexible_type* value, tc_error**);

void tc_sframe_add_columns(tc_sframe* sf, const tc_sframe* other, tc_error**);

void tc_sframe_swap_columns(tc_sframe* sf, const char* column_1, const char* column_2, tc_error**);

void tc_sframe_rename_column(tc_sframe* sf, const char* old_name, const char* new_name, tc_error**);

void tc_sframe_rename_columns(tc_sframe* sf, const tc_flex_dict* name_mapping, tc_error**);

tc_sframe* filter_by(const tc_sframe* src, const tc_sframe* values, const char* column_name, bool exclude, tc_error**);

// Apologies -- these arguments are not completely translated...
tc_sframe* tc_sframe_pack_columns(const tc_sframe* data, const tc_flex_list* columns,
                        const char* new_column_name,
                        flex_type_enum dtype,
                        const tc_flexible_type* fill_na, tc_error**);

tc_sframe* tc_sframe_split_datetime(const tc_sframe* data, const char* expand_column,
                                    const char* column_name_prefix, 
                                    const tc_flex_list* components_to_expose,
                                    bool use_timezone, tc_error**);

tc_sframe* tc_sframe_unpack(const tc_sframe* data,const char* unpack_column,
                            const char* column_name_prefix = "X", 
                            const tc_flexible_type* na_value, tc_error**);
                   
tc_sframe* tc_sframe_unpack_detailed(const tc_sframe* data,const char* unpack_column,
                            const char* column_name_prefix = "X", 
                            const tc_flexible_type* na_value, 
                            const tc_flex_enum* column_types, size_t n_column_types_given, 
                            const tc_flex_list* include_set, 
                            tc_error**);
                   
tc_sframe* tc_sframe_stack(const tc_sframe* data,
                           const char* column_name,
                           const char* new_column_names,
                           bool drop_na);

tc_sframe* tc_sframe_stack_and_rename(const tc_sframe* data,
                  const char* column_name,
                  const tc_flex_list* new_column_names,
                  bool drop_na);
 
tc_sframe* tc_sframe_unstack(const tc_sframe* data,const char* columns,
                    const char* new_column_name = "");

tc_sframe* tc_sframe_unstack(const tc_sframe* data,const std::vector<std::string>& columns,
                    const char* new_column_name = "");
 
tc_sframe* tc_sframe_unique(const tc_sframe* data);

tc_sframe* sort(const tc_sframe* data,const char* column, bool ascending = true);

tc_sframe* sort(const tc_sframe* data,const std::vector<std::string>& columns, bool ascending = true);

tc_sframe* sort(const tc_sframe* data,const std::vector<std::pair<std::string, bool>>& column_and_ascending);

tc_sframe* dropna(const tc_sframe* data,const std::vector<std::string>& columns = std::vector<std::string>(),
                   std::string how = "any");

   std::pair<gl_sframe, gl_sframe> dropna_split(
      const tc_sframe* data,const std::vector<std::string>& columns=std::vector<std::string>(),
      std::string how = "any");

tc_sframe* fillna(const tc_sframe* data,const char* column, flexible_type value);

  
  void materialize_to_callback(
      std::function<bool(size_t, const std::shared_ptr<sframe_rows>&)> callback,
      size_t nthreads = (size_t)(-1));

  gl_sarray apply(std::function<flexible_type(const sframe_rows::row&)> fn,
                  flex_type_enum dtype);
 
tc_sframe* groupby(const std::vector<std::string>& groupkeys, 
                    const std::map<std::string, aggregate::groupby_descriptor_type>& operators 
                    = std::map<std::string, aggregate::groupby_descriptor_type>());


// Destructor
void tc_sframe_destroy(tc_sframe* sa);




/******************************************************************************/
/*                                                                            */
/*   Variant Container Type                                                   */
/*                                                                            */
/******************************************************************************/

// A variant type can hold almost any object type, but cannot go inside of a
// SFrame or SArray.   

/*****************************************************/
/* Creating variant types                            */
/*****************************************************/

tc_variant* tc_variant_create_from_int64(int64_t, tc_error** error);
tc_variant* tc_variant_create_from_double(double, tc_error** error);
tc_variant* tc_variant_create_from_cstring(const char* str, tc_error** error);
tc_variant* tc_variant_create_from_string(const char* str, uint64_t n, tc_error** error);
tc_variant* tc_variant_create_from_double_array(const double* data, uint64_t n, tc_error** error);
tc_variant* tc_variant_create_from_flex_list(const tc_flex_list*, tc_error** error);
tc_variant* tc_variant_create_from_flex_dict(const tc_flex_dict*, tc_error** error);
tc_variant* tc_variant_create_from_datetime(const tc_datetime* dt, tc_error**);
tc_variant* tc_variant_create_from_image(const tc_flex_image*, tc_error** error);
tc_variant* tc_variant_create_from_flexible_type(const tc_flexible_type*, tc_error** error);
tc_variant* tc_variant_create_from_sarray(const tc_sarray*, tc_error** error);
tc_variant* tc_variant_create_from_sframe(const tc_sframe*, tc_error** error);
tc_variant* tc_variant_create_from_parameters(const tc_parameters*, tc_error** error);
tc_variant* tc_variant_create_from_model(const tc_model*, tc_error** error);
tc_variant* tc_variant_create_copy(const tc_variant*, tc_error** error);

int tc_variant_is_int64(const tc_variant*);
int tc_variant_is_double(const tc_variant*);
int tc_variant_is_cstring(const tc_variant*);
int tc_variant_is_string(const tc_variant*);
int tc_variant_is_double_array(const tc_variant*);
int tc_variant_is_flex_list(const tc_variant*);
int tc_variant_is_flex_dict(const tc_variant*);
int tc_variant_is_datetime(const tc_variant*);
int tc_variant_is_image(const tc_variant*);
int tc_variant_is_flexible_type(const tc_variant*);
int tc_variant_is_sarray(const tc_variant*);
int tc_variant_is_sframe(const tc_variant*);
int tc_variant_is_parameters(const tc_variant*);
int tc_variant_is_model(const tc_variant*);


int64_t tc_variant_int64(const tc_variant* ft, tc_error** error); 

double tc_variant_double(const tc_variant* ft, tc_error** error); 

uint64_t tc_variant_string_length(const tc_variant* ft, tc_error** error); 
const char* tc_variant_string_data(const tc_variant* ft, tc_error** error);

uint64_t tc_variant_array_length(const tc_variant* ft, tc_error** error); 
const double* tc_variant_array_data(const tc_variant* ft, tc_error** error);

tc_flex_list* tc_variant_flex_list(const tc_variant*, tc_error**);
tc_flex_dict* tc_variant_flex_dict(const tc_variant*, tc_error**);
tc_datetime* tc_variant_datetime(const tc_variant* dt, tc_error**);
tc_flex_image* tc_variant_flex_image(const tc_variant*, tc_error**); 
tc_flexible_type* tc_variant_flexible_type(const tc_variant*, tc_error**); 
tc_sarray* tc_variant_sarray(const tc_variant*, tc_error**); 
tc_sframe* tc_variant_sframe(const tc_variant*, tc_error**); 
tc_parameters* tc_variant_parameters(const tc_variant*, tc_error** error);
tc_model* tc_variant_model(const tc_variant*, tc_error**); 


/******************************************************************************/
/*                                                                            */
/*   Parameter Specification                                                  */
/*                                                                            */
/******************************************************************************/

// A parameter specification is simply a map of names to variant types holding the 
// possible parameters. 

// Primary methods.
tc_parameters* tc_parameters_create_empty(tc_error**);
void tc_parameters_add(tc_parameters*, const char* name, const tc_variant*, tc_error** error);
int tc_parameters_entry_exists(tc_parameters*, const char* name);
tc_variant* tc_parameters_retrieve(tc_parameters*, const char* name, tc_error** error);

// Convenience methods -- these can be expressed as combinations of the above methods, 
// but are provided here for convenience and to avoid the additional overhead of multiple function calls. 
void tc_parameters_add_int64(tc_parameters*, const char* name, int64_t value, tc_error** error);
void tc_parameters_add_double(tc_parameters*, const char* name, double value, tc_error** error);
void tc_parameters_add_cstring(tc_parameters*, const char* name, const char* str, tc_error** error);
void tc_parameters_add_string(tc_parameters*, const char* name, const char* str, uint64_t n, tc_error** error);
void tc_parameters_add_double_array(tc_parameters*, const char* name, const double* data, uint64_t n, tc_error** error);
void tc_parameters_add_flex_list(tc_parameters*, const char* name, const tc_flex_list* value, tc_error** error);
void tc_parameters_add_flex_dict(tc_parameters*, const char* name, const tc_flex_dict* value, tc_error** error);
void tc_parameters_add_datetime(tc_parameters*, const char* name, const tc_datetime* dt, tc_error**);
void tc_parameters_add_image(tc_parameters*, const char* name, const tc_flex_image*, tc_error** error);
void tc_parameters_add_flexible_type(tc_parameters*, const char* name, const tc_flexible_type*, tc_error** error);
void tc_parameters_add_sarray(tc_parameters*, const char* name, const tc_sarray*, tc_error** error);
void tc_parameters_add_sframe(tc_parameters*, const char* name, const tc_sframe*, tc_error** error);
void tc_parameters_add_parameters(tc_parameters*, const char* name, const tc_parameters*, tc_error** error);
void tc_parameters_add_model(tc_parameters*, const char* name, const tc_model*, tc_error** error);

int tc_parameter_is_int64(const tc_parameters*, const char* name, tc_error** error);
int tc_parameter_is_double(const tc_parameters*, const char* name, tc_error** error);
int tc_parameter_is_cstring(const tc_parameters*, const char* name, tc_error** error);
int tc_parameter_is_string(const tc_parameters*, const char* name, tc_error** error);
int tc_parameter_is_double_array(const tc_parameters*, const char* name, tc_error** error);
int tc_parameter_is_flex_list(const tc_parameters*, const char* name, tc_error** error);
int tc_parameter_is_flex_dict(const tc_parameters*, const char* name, tc_error** error);
int tc_parameter_is_datetime(const tc_parameters*, const char* name, tc_error** error);
int tc_parameter_is_image(const tc_parameters*, const char* name, tc_error** error);
int tc_parameter_is_flexible_type(const tc_parameters*, const char* name, tc_error** error);
int tc_parameter_is_sarray(const tc_parameters*, const char* name, tc_error** error);
int tc_parameter_is_sframe(const tc_parameters*, const char* name, tc_error** error);
int tc_parameter_is_parameters(const tc_parameters*, const char* name, tc_error** error);
int tc_parameter_is_model(const tc_parameters*, const char* name, tc_error** error);

int64_t tc_parameter_retrieve_int64(const tc_parameters* ft, tc_error** error); 
double tc_parameters_retrieve_double(const tc_parameters* ft, tc_error** error); 
tc_flexible_type* tc_parameters_retrieve_string(const tc_parameters* ft, const char* name, tc_error** error);
tc_flexible_type* tc_parameters_retrieve_array(const tc_parameters* ft, const char* name, tc_error** error); 
tc_flex_list* tc_parameters_retrieve_flex_list(const tc_parameters*, const char* name, tc_error**);
tc_flex_dict* tc_parameters_retrieve_flex_dict(const tc_parameters*, const char* name, tc_error**);
tc_datetime* tc_parameters_retrieve_datetime(const tc_parameters* dt, const char* name, tc_error**);
tc_flex_image* tc_parameters_retrieve_image(const tc_parameters*, const char* name, tc_error**); 
tc_flexible_type* tc_parameters_retrieve_flexible_type(const tc_parameters*, const char* name, tc_error**);
tc_sarray* tc_parameters_retrieve_sarray(const tc_parameters*, const char* name, tc_error**); 
tc_sframe* tc_parameters_retrieve_sframe(const tc_parameters*, const char* name, tc_error**); 
tc_parameters* tc_parameters_retrieve_parameters(const tc_parameters*, const char* name, tc_error**); 
tc_model* tc_parameters_retrieve_model(const tc_parameters*, const char* name, tc_error**); 

// delete the parameter container. 
void tc_parameters_destroy(tc_parameters*); 


/******************************************************************************/
/*                                                                            */
/*   Interaction with registered models                                       */
/*                                                                            */
/******************************************************************************/

tc_model* tc_model_create(const char* model_name, tc_error**);

const char* tc_model_name(const tc_model*, tc_error**);

tc_variant* tc_model_call_method(const tc_model* model, const char* method, 
                                 const tc_parameters* arguments, tc_error**);

void tc_model_destroy(tc_model*);

/******************************************************************************/
/*                                                                            */
/*   Interaction with registered functions                                    */
/*                                                                            */
/******************************************************************************/

tc_parameters* tc_model_call_function(const char* function_name, const tc_parameters* arguments, tc_error**);






/******************************************************************************/
/*                                                                            */
/*    SKETCH                                                                  */
/*                                                                            */
/******************************************************************************/

struct tc_sketch_struct;
typedef struct tc_sketch_struct tc_sketch;

tc_sketch* tc_sketch_create(const tc_sarray*, bool background, const tc_flex_list* keys, tc_error **);

bool tc_sketch_ready(tc_sketch*);
size_t tc_sketch_num_elements_processed(tc_sketch*);
double tc_sketch_get_quantile(tc_sketch*, double quantile, tc_error**);
double tc_sketch_frequency_count(tc_sketch*, const tc_flexible_type* value, tc_error**);
tc_flex_dict* tc_sketch_frequent_items(tc_sketch*);
double tc_sketch_num_unique(tc_sketch*);
tc_sketch* tc_sketch_element_sub_sketch(const tc_sketch*, const tc_flexible_type* key, tc_error**);
tc_sketch* tc_sketch_element_length_summary(const tc_sketch*, tc_error**);
tc_sketch* tc_sketch_element_summary(const tc_sketch*, tc_error**);
tc_sketch* tc_sketch_dict_key_summary(const tc_sketch*, tc_error **error);
tc_sketch* tc_sketch_dict_value_summary(const tc_sketch*, tc_error **error);
double tc_sketch_mean(const tc_sketch*, tc_error**);
double tc_sketch_max(const tc_sketch*, tc_error**);
double tc_sketch_min(const tc_sketch*, tc_error**);
double tc_sketch_sum(const tc_sketch*, tc_error**);
double tc_sketch_variance(const tc_sketch*, tc_error**);
size_t tc_sketch_size(const tc_sketch*);
size_t tc_sketch_num_undefined(const tc_sketch*);
void tc_sketch_cancel(tc_sketch*);

void tc_sketch_destroy(tc_sketch *);








#ifdef __cplusplus
}
#endif



#endif


