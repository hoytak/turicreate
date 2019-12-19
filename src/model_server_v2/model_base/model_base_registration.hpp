/* Copyright © 2019 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#ifndef TURI_MODEL_BASE_REGISTRATION_HPP
#define TURI_MODEL_BASE_REGISTRATION_HPP





/** Register a method that can be called by name using the registry above.  
*
*/
// TODO: Overload const functions
template <typename Class, typename RetType, typename... FuncParams>
void model_base::register_method(
    const std::string& name, 
    (RetType)(*Class::method)(FuncArgs...),
    const std::vector<Parameter>& parameter_list) {

  m_registry->register_method(name, method, parameter_list);
}













#endif


