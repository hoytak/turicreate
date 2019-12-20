/* Copyright © 2017 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#include <model_server_v2/model_server.hpp>


namespace turi {
namespace v2 { 

EXPORT model_server_impl& model_server() { 
  static model_server_impl global_model_server;
  return global_model_server;
}

/** Does the work of registering things with the callbacks. 
 */
void model_server_impl::_process_registered_callbacks_internal() {

  std::lock_guard<std::mutex> _lg(m_model_registration_lock);

  size_t cur_idx; 

  while( (cur_idx = m_callback_last_processed_index) != m_callback_pushback_index) { 

    
    // Call the callback function to perform the registration, simultaneously 
    // zeroing out the pointer. 
    class_registration_callback reg_f = nullptr;
    std::swap(reg_f, m_registration_callback_list[cur_idx % max_registered_callbacks()]);
    reg_f(*this);

    // We're done here; advance.
    ++m_callback_last_processed_index;  
  }
}
















}
}
