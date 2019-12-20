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
void model_server_impl::_process_registered_callbacks_internal();


/** The internal registration of the new model.  
 */ 
void model_server_impl::_register_new_model_internal(
    std::shared_ptr<model_base> new_model) {

  // Get the name of the model.  NOTE: these are 
  // supposed to be 
  const std::string& name = new_model->name();


  // TODO: Add alias names for model load back-compat. 
  //
  m_




}
}
