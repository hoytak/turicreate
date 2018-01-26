#ifndef TURI_CAPI_MODELS_H_
#define TURI_CAPI_MODELS_H_

#include <capi/TuriCore.h>
#include <unity/lib/toolkit_class_base.hpp>


extern "C" {

struct tc_model_struct {
  std::shared_ptr<turi::toolkit_class_base> value;
};

}

static inline tc_model* new_tc_model() {
  return new tc_model;
}

static inline tc_model* new_tc_model(std::shared_ptr<turi::toolkit_class_base> m) {
  tc_model* model = new_tc_model();
  model->value = m;
  return model;
}

#endif:
