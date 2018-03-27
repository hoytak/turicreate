/* Copyright © 2017 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#include <string>
#include <boost/algorithm/string/predicate.hpp>
#include <fileio/sanitize_url.hpp>
#include <export.hpp>
#ifdef TC_HAS_REMOTEFS
#include <fileio/s3_api.hpp>
#endif

namespace turi {

EXPORT std::string sanitize_url(std::string url) {
#ifdef TC_HAS_REMOTEFS
  if (boost::algorithm::starts_with(url, "s3://")) {
    return sanitize_s3_url(url);
  } else {
    return url;
  }
#else
    return url;
#endif
}

}
