/* Copyright © 2017 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#ifndef COREML_MLMODEL_HPP
#define COREML_MLMODEL_HPP

// Include this first.  We need to undefine some defines here.
#include <unity/toolkits/coreml_export/MLModel/src/transforms/TreeEnsemble.hpp>
#include <unity/toolkits/coreml_export/MLModel/src/transforms/Pipeline.hpp>

#include <unity/toolkits/coreml_export/MLModel/src/transforms/OneHotEncoder.hpp>
#include <unity/toolkits/coreml_export/MLModel/src/transforms/FeatureVectorizer.hpp>
#include <unity/toolkits/coreml_export/MLModel/src/transforms/DictVectorizer.hpp>
#include <unity/toolkits/coreml_export/MLModel/src/transforms/LinearModel.hpp>


#ifdef MAX
#undef MAX
#endif

#ifdef MIN
#undef MIN
#endif



#endif
