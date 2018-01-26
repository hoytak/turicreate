/* Copyright © 2017 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#include "MLModelTests.hpp"

// TODO -- Fix these headers.
#include "../src/Model.hpp"
#include "../src/transforms/OneHotEncoder.hpp"
#include "../src/transforms/LinearModel.hpp"
#include "../src/transforms/TreeEnsemble.hpp"


#include "framework/TestUtils.hpp"
#include <cassert>
#include <cstdio>

using namespace CoreML;

int testBasicSaveLoad () {
    Result r;

    const std::string path("/tmp/a.modelasset");
    OneHotEncoder ohe;
    ML_ASSERT_GOOD(ohe.addInput("foo", FeatureType::String()));
    ohe.getProto().mutable_onehotencoder()->mutable_stringcategories()->add_vector()->assign("foo");
    ML_ASSERT_GOOD(ohe.addOutput("bar", FeatureType::Array({})));
    ML_ASSERT_GOOD(ohe.save(path));
    Model a2;
    ML_ASSERT_GOOD(Model::load(path, a2));
    ML_ASSERT_EQ(ohe, a2);
    ML_ASSERT_EQ(std::remove(path.c_str()), 0);

    return 0;
}
