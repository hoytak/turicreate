/* Copyright © 2019 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at
 * https://opensource.org/licenses/BSD-3-Clause
 */

#include <toolkits/object_detection/od_serialization.hpp>

#include <stdlib.h>
#include <cstdio>

namespace turi {
namespace object_detection {

using turi::neural_net::model_spec;
using padding_type = model_spec::padding_type;
using turi::neural_net::float_array_map;
using turi::neural_net::zero_weight_initializer;

void _save_impl(oarchive& oarc, const neural_net::model_spec& nn_spec,
                const std::map<std::string, variant_type>& state) {
  // Save model attributes.
  variant_deep_save(state, oarc);

  // Save neural net weights.
  oarc << nn_spec.export_params_view();
}

void _load_version(iarchive& iarc, size_t version,
                   neural_net::model_spec& nn_spec,
                   std::map<std::string, variant_type>& state,
                   const std::vector<std::pair<float, float>>& anchor_boxes) {
  // Load model attributes.
  variant_deep_load(state, iarc);

  // Load neural net weights.
  float_array_map nn_params;
  iarc >> nn_params;
  init_darknet_yolo(nn_spec, variant_get_value<size_t>(state.at("num_classes")),
                    anchor_boxes);
  nn_spec.update_params(nn_params);
}

void init_darknet_yolo(
    neural_net::model_spec& nn_spec, const size_t num_classes,
    const std::vector<std::pair<float, float>>& anchor_boxes) {
  int num_features = 3;

  // Initialize Layer 0 to Layer 7
  const std::map<int, int> layer_num_to_channels = {
      {0, 16},  {1, 32},  {2, 64},   {3, 128},
      {4, 256}, {5, 512}, {6, 1024}, {7, 1024}};
  for (const auto& x : layer_num_to_channels) {
    if (x.first == 7) {
      // The conv7 weights have shape [1024, 1024, 3, 3], so fan in and fan out
      // are both 1024*3*3.
      nn_spec.add_convolution(/* name */ "conv7_fwd",
                              /* input */ "leakyrelu6_fwd",
                              /* num_output_channels */ 1024,
                              /* num_kernel_channels */ 1024,
                              /* kernel_height */ 3,
                              /* kernel_width */ 3,
                              /* stride_height */ 1,
                              /* stride_width */ 1,
                              /* padding */ padding_type::SAME,
                              /* weight_init_fn */ zero_weight_initializer());
    } else {
      nn_spec.add_convolution(
          /* name */ "conv" + std::to_string(x.first) + "_fwd",
          /* input */ "leakyrelu" + std::to_string(x.first - 1) + "_fwd",
          /* num_output_channels */ x.second,
          /* num_kernel_channels */ num_features,
          /* kernel_height */ 3,
          /* kernel_width */ 3,
          /* stride_height */ 1,
          /* stride_width */ 1,
          /* padding */ padding_type::SAME,
          /* weight_init_fn */ zero_weight_initializer());
    }

    // Append batchnorm.
    nn_spec.add_batchnorm(
        /* name */ "batchnorm" + std::to_string(x.first) + "_fwd",
        /* input */ "conv" + std::to_string(x.first) + "_fwd",
        /* num_channels */ x.second,
        /* epsilon */ 0.00001f);

    // Append leakyrelu.
    nn_spec.add_leakyrelu(
        /* name */ "leakyrelu" + std::to_string(x.first) + "_fwd",
        /* input */ "batchnorm" + std::to_string(x.first) + "_fwd",
        /* alpha */ 0.1f);

    num_features = x.second;
  }

  // Append conv8.
  const size_t num_predictions = 5 + num_classes;  // Per anchor box
  const size_t conv8_c_out = anchor_boxes.size() * num_predictions;
  nn_spec.add_convolution(/* name */ "conv8_fwd",
                          /* input */ "leakyrelu7_fwd",
                          /* num_output_channels */ conv8_c_out,
                          /* num_kernel_channels */ 1024,
                          /* kernel_height */ 1,
                          /* kernel_width */ 1,
                          /* stride_height */ 1,
                          /* stride_width */ 1,
                          /* padding */ padding_type::SAME,
                          /* weight_init_fn */ zero_weight_initializer(),
                          /* bias_init_fn */ zero_weight_initializer());
}

}  // namespace object_detection
}  // namespace turi
