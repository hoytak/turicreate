#include <string>
#include "demo_model.hpp"
#include "model_server.hpp"

using namespace turi;
using namespace turi::v2;





int main(int argc, char** argv) { 

 
  std::shared_ptr<model_base> dm = model_server().create_model("demo_model"); 


  size_t result = dm->call("add", 5, 9); 

  std::cout << "Demo: 5 + 9 = " << result << std::endl; 

  std::string s_res = dm->call("concat_strings", "A", "B");

  std::cout << "Demo: Concat A, +, B: " << s_res << std::endl; 

  return 0; 

} 
