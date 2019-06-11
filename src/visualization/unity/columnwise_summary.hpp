/* Copyright © 2017 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#ifndef __TC_COLUMN_SUMMARY
#define __TC_COLUMN_SUMMARY

#include <visualization/unity/process_wrapper.hpp>
#include <visualization/unity/histogram.hpp>
#include <visualization/unity/item_frequency.hpp>
#include <visualization/unity/transformation.hpp>
#include <visualization/unity/thread.hpp>
#include <visualization/unity/summary_view.hpp>
#include <visualization/unity/vega_data.hpp>
#include <visualization/unity/vega_spec.hpp>

namespace turi {
  class unity_sframe_base;

  namespace visualization {
    std::shared_ptr<Plot> plot_columnwise_summary(std::shared_ptr<unity_sframe_base> sf);
  }
}

#endif // __TC_ITEM_FREQUENCY
