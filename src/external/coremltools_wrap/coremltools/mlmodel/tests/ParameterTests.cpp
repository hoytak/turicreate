//
//  UpdatableModelValidatorTests.cpp
//  CoreML_framework
//
//  Created by aseem wadhwa on 2/12/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "MLModelTests.hpp"
#include "../src/Format.hpp"
#include "../src/Model.hpp"
#include "ParameterTests.hpp"
#include "ModelCreationUtils.hpp"

#include "framework/TestUtils.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"

using namespace CoreML;

void addLearningRate(Specification::NeuralNetwork *nn, Specification::Optimizer::OptimizerTypeCase optimizerType, double defaultValue, double minValue,  double maxValue) {
    
    auto updateParameters = nn->mutable_updateparams();
    auto optimizer = updateParameters->mutable_optimizer();

    Specification::DoubleParameter *learningRate = NULL;
    
    switch (optimizerType) {
        case Specification::Optimizer::kSgdOptimizer:
            learningRate = optimizer->mutable_sgdoptimizer()->mutable_learningrate();
            break;
        case Specification::Optimizer::kAdamOptimizer:
            learningRate = optimizer->mutable_adamoptimizer()->mutable_learningrate();
            break;
        default:
            break;
    }
    
    learningRate->set_defaultvalue(defaultValue);
    
    auto doubleRange = learningRate->mutable_range();
    doubleRange->set_minvalue(minValue);
    doubleRange->set_maxvalue(maxValue);
}

void addMiniBatchSize(Specification::NeuralNetwork *nn, Specification::Optimizer::OptimizerTypeCase optimizerType, int64_t defaultValue, int64_t minValue,  int64_t maxValue, std::set<int64_t> allowedValues) {
    
    auto updateParameters = nn->mutable_updateparams();
    auto optimizer = updateParameters->mutable_optimizer();
    
    Specification::Int64Parameter *miniBatchSize = NULL;
    
    switch (optimizerType) {
        case Specification::Optimizer::kSgdOptimizer:
            miniBatchSize = optimizer->mutable_sgdoptimizer()->mutable_minibatchsize();
            break;
        case Specification::Optimizer::kAdamOptimizer:
            miniBatchSize = optimizer->mutable_adamoptimizer()->mutable_minibatchsize();
            break;
        default:
            break;
    }
    
    miniBatchSize->set_defaultvalue(defaultValue);
    
    if (allowedValues.size() == 0) {
        auto int64Range = miniBatchSize->mutable_range();
        int64Range->set_minvalue(minValue);
        int64Range->set_maxvalue(maxValue);
    }
    else {
        auto int64Set = miniBatchSize->mutable_set();
        for (auto& x : allowedValues) {
            int64Set->add_values(x);
        }
    }
}

void addMomentum(Specification::NeuralNetwork *nn, Specification::Optimizer::OptimizerTypeCase optimizerType, double defaultValue, double minValue, double maxValue) {
    
    auto updateParameters = nn->mutable_updateparams();
    auto optimizer = updateParameters->mutable_optimizer();
    auto realOptimizer = (::CoreML::Specification::SGDOptimizer*)NULL;
    
    switch (optimizerType) {
        case Specification::Optimizer::kSgdOptimizer:
            realOptimizer = optimizer->mutable_sgdoptimizer();
            break;
        default:
            break;
    }
    
    auto momentum = realOptimizer->mutable_momentum();
    
    momentum->set_defaultvalue(defaultValue);
    
    auto doubleRange = momentum->mutable_range();
    doubleRange->set_minvalue(minValue);
    doubleRange->set_maxvalue(maxValue);
}

void addBeta1(Specification::NeuralNetwork *nn, Specification::Optimizer::OptimizerTypeCase optimizerType, double defaultValue, double minValue, double maxValue) {
    
    auto updateParameters = nn->mutable_updateparams();
    auto optimizer = updateParameters->mutable_optimizer();
    auto realOptimizer = (::CoreML::Specification::AdamOptimizer*)NULL;
    
    switch (optimizerType) {
        case Specification::Optimizer::kAdamOptimizer:
            realOptimizer = optimizer->mutable_adamoptimizer();
            break;
        default:
            break;
    }
    
    auto beta1 = realOptimizer->mutable_beta1();
    
    beta1->set_defaultvalue(defaultValue);
    
    auto doubleRange = beta1->mutable_range();
    doubleRange->set_minvalue(minValue);
    doubleRange->set_maxvalue(maxValue);
}

void addBeta2(Specification::NeuralNetwork *nn, Specification::Optimizer::OptimizerTypeCase optimizerType, double defaultValue, double minValue, double maxValue) {
    
    auto updateParameters = nn->mutable_updateparams();
    auto optimizer = updateParameters->mutable_optimizer();
    auto realOptimizer = (::CoreML::Specification::AdamOptimizer*)NULL;
    
    switch (optimizerType) {
        case Specification::Optimizer::kAdamOptimizer:
            realOptimizer = optimizer->mutable_adamoptimizer();
            break;
        default:
            break;
    }
    
    auto beta2 = realOptimizer->mutable_beta2();
    
    beta2->set_defaultvalue(defaultValue);
    
    auto doubleRange = beta2->mutable_range();
    doubleRange->set_minvalue(minValue);
    doubleRange->set_maxvalue(maxValue);
}

void addEps(Specification::NeuralNetwork *nn, Specification::Optimizer::OptimizerTypeCase optimizerType, double defaultValue, double minValue, double maxValue) {
    
    auto updateParameters = nn->mutable_updateparams();
    auto optimizer = updateParameters->mutable_optimizer();
    auto realOptimizer = (::CoreML::Specification::AdamOptimizer*)NULL;
    
    switch (optimizerType) {
        case Specification::Optimizer::kAdamOptimizer:
            realOptimizer = optimizer->mutable_adamoptimizer();
            break;
        default:
            break;
    }
    
    auto eps = realOptimizer->mutable_eps();
    
    eps->set_defaultvalue(defaultValue);
    
    auto doubleRange = eps->mutable_range();
    doubleRange->set_minvalue(minValue);
    doubleRange->set_maxvalue(maxValue);
}

void addEpochs(Specification::NeuralNetwork *nn, int64_t defaultValue, int64_t minValue,  int64_t maxValue, std::set<int64_t> allowedValues) {
    
    auto updateParameters = nn->mutable_updateparams();
    auto epochs = updateParameters->mutable_epochs();
    
    epochs->set_defaultvalue(defaultValue);
    
    if (allowedValues.size() == 0) {
        auto int64Range = epochs->mutable_range();
        int64Range->set_minvalue(minValue);
        int64Range->set_maxvalue(maxValue);
    }
    else {
        auto int64Set = epochs->mutable_set();
        for (auto& x : allowedValues) {
            int64Set->add_values(x);
        }
    }
}

int testMiniBatchSizeOutOfAllowedRange() {
    
    Specification::Model m;
    
    // basic neural network model without any updatable model parameters.
    auto nn = buildBasicUpdatableNeuralNetworkModel(m);
    
    addLearningRate(nn, Specification::Optimizer::kSgdOptimizer, 0.7f, 0.0f, 1.0f);
    addMiniBatchSize(nn, Specification::Optimizer::kSgdOptimizer, 5, 10, 100);
    
    // expect validation to fail due to conflict in updatable model parameter names.
    Result res = Model::validate(m);
    ML_ASSERT_BAD(res);
    return 0;
}

int testMiniBatchSizeOutOfAllowedSet() {
    
    Specification::Model m;
    
    // basic neural network model without any updatable model parameters.
    auto nn = buildBasicUpdatableNeuralNetworkModel(m);
    
    std::set<int64_t> allowedValues = std::set<int64_t>();
    allowedValues.insert(10);
    allowedValues.insert(20);
    allowedValues.insert(30);
    allowedValues.insert(40);
    
    addLearningRate(nn, Specification::Optimizer::kSgdOptimizer, 0.7f, 0.0f, 1.0f);
    addMiniBatchSize(nn, Specification::Optimizer::kSgdOptimizer, 5, 10, 100, allowedValues);
    
    // expect validation to fail due to conflict in updatable model parameter names.
    Result res = Model::validate(m);
    ML_ASSERT_BAD(res);
    return 0;
}

int testLearningRateOutOfAllowedRange() {
    
    Specification::Model m;
    
    // basic neural network model without any updatable model parameters.
    auto nn = buildBasicUpdatableNeuralNetworkModel(m);
    
    addLearningRate(nn, Specification::Optimizer::kSgdOptimizer, 5.0f, 0.1f, 1.0f);
    
    Result res = Model::validate(m);
    ML_ASSERT_BAD(res);
    return 0;
}

int testMomentumOutOfAllowedRange() {
    
    Specification::Model m;
    
    // basic neural network model without any updatable model parameters.
    auto nn = buildBasicUpdatableNeuralNetworkModel(m);
    
    addMomentum(nn, Specification::Optimizer::kSgdOptimizer, 5.0f, 0.1f, 1.0f);
    
    Result res = Model::validate(m);
    ML_ASSERT_BAD(res);
    return 0;
}

int testBeta1OutOfAllowedRange() {
    
    Specification::Model m;
    
    // basic neural network model without any updatable model parameters.
    auto nn = buildBasicUpdatableNeuralNetworkModel(m);
    
    addBeta1(nn, Specification::Optimizer::kAdamOptimizer, 5.0f, 0.1f, 1.0f);
    
    Result res = Model::validate(m);
    ML_ASSERT_BAD(res);
    return 0;
}

int testBeta2OutOfAllowedRange() {
    
    Specification::Model m;
    
    // basic neural network model without any updatable model parameters.
    auto nn = buildBasicUpdatableNeuralNetworkModel(m);
    
    addBeta2(nn, Specification::Optimizer::kAdamOptimizer, 5.0f, 0.1f, 1.0f);
    
    Result res = Model::validate(m);
    ML_ASSERT_BAD(res);
    return 0;
}

int testEpsOutOfAllowedRange() {
    
    Specification::Model m;
    
    // basic neural network model without any updatable model parameters.
    auto nn = buildBasicUpdatableNeuralNetworkModel(m);
    
    addEps(nn, Specification::Optimizer::kAdamOptimizer, 5.0f, 0.1f, 1.0f);
    
    Result res = Model::validate(m);
    ML_ASSERT_BAD(res);
    return 0;
}

int testEpochsOutOfAllowedRange() {
    
    Specification::Model m;
    
    // basic neural network model without any updatable model parameters.
    auto nn = buildBasicUpdatableNeuralNetworkModel(m);
    
    addLearningRate(nn, Specification::Optimizer::kSgdOptimizer, 0.7f, 0.0f, 1.0f);
    addMiniBatchSize(nn, Specification::Optimizer::kSgdOptimizer, 20, 10, 100);
    addEpochs(nn, 100, 0, 50, std::set<int64_t>());
    
    // expect validation to fail due to conflict in updatable model parameter names.
    Result res = Model::validate(m);
    ML_ASSERT_BAD(res);
    return 0;
}

int testEpochsOutOfAllowedSet() {
    
    Specification::Model m;
    
    // basic neural network model without any updatable model parameters.
    auto nn = buildBasicUpdatableNeuralNetworkModel(m);
    
    std::set<int64_t> allowedValues = std::set<int64_t>();
    allowedValues.insert(10);
    allowedValues.insert(20);
    allowedValues.insert(30);
    allowedValues.insert(40);
    
    addLearningRate(nn, Specification::Optimizer::kSgdOptimizer, 0.7f, 0.0f, 1.0f);
    addMiniBatchSize(nn, Specification::Optimizer::kSgdOptimizer, 20, 10, 100);
    addEpochs(nn, 100, 0, 0, allowedValues);
    
    // expect validation to fail due to conflict in updatable model parameter names.
    Result res = Model::validate(m);
    ML_ASSERT_BAD(res);
    return 0;
}
