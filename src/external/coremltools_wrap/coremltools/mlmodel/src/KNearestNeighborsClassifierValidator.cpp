//
//  KNearestNeighborsClassifierValidator.cpp
//  CoreML_framework
//
//  Created by Bill March on 10/4/18.
//  Copyright © 2018 Apple Inc. All rights reserved.
//

#include "../build/format/NearestNeighbors_enums.h"
#include "Validators.hpp"
#include "ValidatorUtils-inl.hpp"

#include <algorithm>
#include <sstream>

namespace CoreML {

    static Result validateNearestNeighborsIndex(const Specification::Model& format, int expectedSampleCount) {

        const Specification::NearestNeighborsIndex& nnIndex = format.knearestneighborsclassifier().nearestneighborsindex();

        // A valid index should have some data points
        if (nnIndex.floatsamples_size() == 0) {
            std::stringstream out;
            out << "KNearestNeighborsClassifier has no data points." << std::endl;
            return Result(ResultType::INVALID_MODEL_PARAMETERS, out.str());
        }

        if (nnIndex.floatsamples_size() != expectedSampleCount) {
            std::stringstream out;
            out << "Unexpected number of labels \"" << expectedSampleCount << "\" for the given number of examples: \"" << nnIndex.floatsamples_size() << "." << std::endl;
            return Result(ResultType::INVALID_MODEL_PARAMETERS, out.str());
        }

        // Only need to check that the length of the individual vectors are equivalent to the dimensionality (and thus eachother)
        for (int i = 0; i < nnIndex.floatsamples_size(); i++) {
            if (nnIndex.floatsamples(i).vector_size() != nnIndex.numberofdimensions()) {
                std::stringstream out;
                out << "Unexpected length \"" << nnIndex.floatsamples_size() << "\" given the provided number of dimensions \"" << nnIndex.numberofdimensions() << "." << std::endl;
                return Result(ResultType::INVALID_MODEL_PARAMETERS, out.str());
            }
        }

        // Should we require the user to always specify an index type?
        bool hasLinearBackend = nnIndex.has_linearindex();
        bool hasKdTreeBackend = nnIndex.has_singlekdtreeindex();
        if (!hasLinearBackend && !hasKdTreeBackend) {
            std::stringstream out;
            out << "KNearestNeighborsClassifier has no index type specified." << std::endl;
            return Result(ResultType::INVALID_MODEL_PARAMETERS, out.str());
        }

        if (hasKdTreeBackend) {
            if (nnIndex.singlekdtreeindex().leafsize() <= 0) {
                std::stringstream out;
                out << "KNearestNeighborsClassifier requires leaf size to be a positive integer." << std::endl;
                return Result(ResultType::INVALID_MODEL_PARAMETERS, out.str());
            }
        }

        switch (nnIndex.DistanceFunction_case()) {
            case Specification::NearestNeighborsIndex::kSquaredEuclideanDistance:
                // Valid distance function
                break;

            case Specification::NearestNeighborsIndex::DISTANCEFUNCTION_NOT_SET:
                std::stringstream out;
                out << "KNearestNeighborsClassifier requires a distance function to be set." << std::endl;
                return Result(ResultType::INVALID_MODEL_PARAMETERS, out.str());
        }

        return Result();

    }

    template <>
    Result validate<MLModelType_kNearestNeighborsClassifier>(const Specification::Model& format) {

        const Specification::KNearestNeighborsClassifier& knnClassifier = format.knearestneighborsclassifier();

        if (knnClassifier.k() <= 0) {
            std::stringstream out;
            out << "KNearestNeighborsClassifier requires k to be a positive integer." << std::endl;
            return Result(ResultType::INVALID_MODEL_PARAMETERS, out.str());
        }

        switch (knnClassifier.WeightingScheme_case()) {
            case Specification::KNearestNeighborsClassifier::kUniformWeighting:
                // Valid weighting scheme
                break;

            case Specification::KNearestNeighborsClassifier::WEIGHTINGSCHEME_NOT_SET:
                std::stringstream out;
                out << "KNearestNeighborsClassifier requires a weighting scheme to be set." << std::endl;
                return Result(ResultType::INVALID_MODEL_PARAMETERS, out.str());
        }

        // And that the number of labels is equal to the number of examples
        bool hasLabels = knnClassifier.has_int64classlabels() || knnClassifier.has_stringclasslabels();
        if (!hasLabels) {
            std::stringstream out;
            out << "KNearestNeighborsClassifier has no labels." << std::endl;
            return Result(ResultType::INVALID_MODEL_PARAMETERS, out.str());
        }

        int intLabelCount = knnClassifier.has_int64classlabels() ? knnClassifier.int64classlabels().vector_size() : 0;
        int stringLabelCount = knnClassifier.has_stringclasslabels() ? knnClassifier.stringclasslabels().vector_size() : 0;

        int expectedSampleCount = MAX(intLabelCount, stringLabelCount);

        return validateNearestNeighborsIndex(format, expectedSampleCount);

    }

}
