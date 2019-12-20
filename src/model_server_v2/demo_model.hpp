/* Copyright © 2019 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#ifndef TURI_V2_DEMO_MODEL_HPP
#define TURI_V2_DEMO_MODEL_HPP

#include <string>
#include <model_server_v2/model_base.hpp>

namespace turi { 
  namespace v2 {


class demo_model : public model_base { 

  const char* name() { return "demo"; } 


  /**  Sets up the registration.
   */ 
  void configure() {  
    register_method(demo_model::add, "x", "y"); 
    register_method(demo_model::append_strings, "s1", "s2");
  } 


  /** Add two numbers.
   */
  size_t add(size_t x, size_t y) { return x + y; }


  /** Append two strings with a +
   */
  std::string append_strings(const std::string& s1, const std::string& s2) 
  {
    return s1 + "+" + s2;
  }

};

REGISTER_MODEL(demo_model);

}}

#endif


