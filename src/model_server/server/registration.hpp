/* Copyright © 2017 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#ifndef TURI_UNITY_REGSTRATION_HPP_
#define TURI_UNITY_REGSTRATION_HPP_

#define TC_MAX_REGISTERED_CALLBACKS 512

#include <atomic>
#include <mutex>
#include <array>

namespace turi {


  void process_registered_callbacks_internal();

  typedef void (*class_registration_callback)(void); 

  extern std::array<class_registration_callback, TC_MAX_REGISTERED_CALLBACKS> registration_callback_list; 
  extern std::atomic<size_t> _callback_pushback_index;
  extern std::atomic<size_t> _callback_last_processed_index;

  // Register the callbacks if need be.
  static inline void process_registered_callbacks() { 
    if(_callback_last_processed_index < _callback_pushback_index) { 
      process_registered_callbacks_internal(); 
    }
  }


  static inline void _add_registration_callback(class_registration_callback callback) {

    size_t insert_index_raw = (_callback_pushback_index++);
    
    do {
      // Check to make sure this can be safely inserted.
      size_t processed_index_raw = _callback_last_processed_index;

      if(processed_index_raw + TC_MAX_REGISTERED_CALLBACKS > insert_index_raw) { 
        break; 
      } else {
        // This will process the next block of insertions.
        process_registered_callbacks();
      }

    } while(true);

    size_t insert_index = insert_index_raw % TC_MAX_REGISTERED_CALLBACKS;

    ASSERT_TRUE(registration_callback_list[insert_index] == nullptr);
    registration_callback_list[insert_index] = callback;
  } 


  // A helper class to use a static initializer to do a lightweight registration 
  // of class loading at library load time.
  class _static_registration_hook {
    public: 
      inline _static_registration_hook(class_registration_callback* f) { 
        f(); 
      }
  };



}

#endif
