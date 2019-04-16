//
//  KNearestNeighborsClassifierValidator.cpp
//  CoreML_framework
//
//  Created by Bill March on 10/4/18.
//  Copyright © 2018 Apple Inc. All rights reserved.
//

#include "../build/format/KNearestNeighborsClassifier_enums.h"
#include "Validators.hpp"
#include "ValidatorUtils-inl.hpp"

#include <algorithm>
#include <sstream>

namespace CoreML {

    template <>
    Result validate<MLModelType_kNearestNeighborsClassifier>(const Specification::Model& format) {
#pragma unused(format)
        const Specification::KNearestNeighborsClassifier& kNN = format.knearestneighborsclassifier();

        if (kNN.k() <= 0) {
            std::stringstream out;
            out << "KNearestNeighborsModel requires k to be a positive integer." << std::endl;
            return Result(ResultType::INVALID_MODEL_PARAMETERS, out.str());
        }

        if (kNN.floatsamples_size() == 0) {
            std::stringstream out;
            out << "KNearestNeighborsModel has no data points." << std::endl;
            return Result(ResultType::INVALID_MODEL_PARAMETERS, out.str());
        }

        // Only need to check that the length of the individual vectors are equivalent to the dimensionality (and thus eachother)
        for (int i = 0; i < kNN.floatsamples_size(); i++) {
            if (kNN.floatsamples(i).vector_size() != kNN.dimensionality()) {
                std::stringstream out;
                out << "Unsupported length \"" << kNN.floatsamples_size() << "\" given the provided dimensionality \"" << kNN.dimensionality() << "." << std::endl;
                return Result(ResultType::INVALID_MODEL_PARAMETERS, out.str());
            }
        }
        
        // And that the number of labels is equal to the number of examples
        bool hasLabels = kNN.has_int64classlabels() || kNN.has_stringclasslabels();
        if (!hasLabels) {
            std::stringstream out;
            out << "KNearestNeighborsModel has no labels." << std::endl;
            return Result(ResultType::INVALID_MODEL_PARAMETERS, out.str());
        }

        int intLabelCount = (hasLabels && kNN.has_int64classlabels()) ? kNN.int64classlabels().vector_size() : 0;
        int stringLabelCount = (hasLabels && kNN.has_stringclasslabels()) ? kNN.stringclasslabels().vector_size() : 0;
        
        if (hasLabels && MAX(intLabelCount, stringLabelCount) != kNN.floatsamples_size()) {
            std::stringstream out;
            out << "Unsupported number of labels \"" << MAX(intLabelCount, stringLabelCount) << "\" for the given number of examples: \"" << kNN.floatsamples_size() << "." << std::endl;
            return Result(ResultType::INVALID_MODEL_PARAMETERS, out.str());
        }
        
        return Result();
    }

}
