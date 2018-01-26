/* Copyright © 2017 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#ifndef ValidatorUtils_h
#define ValidatorUtils_h

#include "Comparison.hpp"
#include "Format.hpp"
#include "../build/format/FeatureTypes_enums.h"
#include <sstream>

namespace CoreML {


    /*
     * Utility that make sures the feature types are valid.
     *
     * @param  allowedFeatureTypes Allowed feature types.
     * @param featureType type of this operation.
     * @retun
     */
    inline Result validateSchemaTypes(const std::vector<Specification::FeatureType::TypeCase>& allowedFeatureTypes,
                 const Specification::FeatureDescription& featureDesc) {

        // Check the types
        auto type = featureDesc.type().Type_case();
        for (const auto& t : allowedFeatureTypes) {
            if (type == t) {
                // no invariant broken -- type matches one of the allowed types
                return Result();
            }
        }

        // Invalid type
        std::stringstream out;
        out << "Unsupported type \"" << MLFeatureTypeType_Name(static_cast<MLFeatureTypeType>(featureDesc.type().Type_case()))
        << "\" for feature \"" << featureDesc.name() + "\". Should be one of: ";
        bool isFirst = true;
        for (const auto& t: allowedFeatureTypes) {
            if (!isFirst) {
                out << ", ";
            }
            out << MLFeatureTypeType_Name(static_cast<MLFeatureTypeType>(t));
            isFirst = false;
        }
        out << "." << std::endl;
        return Result(ResultType::UNSUPPORTED_FEATURE_TYPE_FOR_MODEL_TYPE, out.str());
    }

    /*
     * Utility that checks all feature types are vectorizable
     */
    template <typename Descriptions>
    inline Result validateDescriptionsAreAllVectorizableTypes(const Descriptions &features) {
        Result result;
        for (int i = 0; i < features.size(); i++) {
            result = validateSchemaTypes({
                Specification::FeatureType::kDoubleType,
                Specification::FeatureType::kInt64Type,
                Specification::FeatureType::kMultiArrayType,
            }, features[i]);
            if (!result.good()) {
                return result;
            }
        }

        return result;
    }

    /*
     * Utility that checks a set of descriptions to validate
     * there is a feature with a specific name and type in an allowed set
     */
    template <typename Descriptions>
    inline Result validateDescriptionsContainFeatureWithTypes(const Descriptions &features,
                                                              int maxFeatureCount,
                                                              const std::vector<Specification::FeatureType::TypeCase>& allowedFeatureTypes) {
        Result result;

        // 0 means no maximum fixed feature count.
        if (maxFeatureCount != 0 && features.size() > maxFeatureCount) {
            return Result(ResultType::TOO_MANY_FEATURES_FOR_MODEL_TYPE, "Feature descriptions exceeded " + std::to_string(maxFeatureCount));
        }

        for (int i = 0; i < features.size(); i++) {
            result = validateSchemaTypes(allowedFeatureTypes, features[i]);
            if (!result.good()) {
                return result;
            }
        }

        return result;
    }

    /*
     * Utility that checks a set of descriptions to validate
     * there is a feature with a specific name and type in an allowed set
     */
    template <typename Descriptions>
    inline Result validateDescriptionsContainFeatureWithNameAndType(const Descriptions &features,
                                                                    const std::string &name,
                                                                    const std::vector<Specification::FeatureType::TypeCase>& allowedFeatureTypes) {
        Result result;
        for (int i = 0; i < features.size(); i++) {
            if (name.compare(features[i].name()) != 0) {
                continue;
            }
            return validateSchemaTypes(allowedFeatureTypes, features[i]);
        }

        return Result(ResultType::INTERFACE_FEATURE_NAME_MISMATCH, "Expected feature '" + name + "' to the model is not present in the model description.");
    }


}
#endif /* ValidatorUtils_h */
