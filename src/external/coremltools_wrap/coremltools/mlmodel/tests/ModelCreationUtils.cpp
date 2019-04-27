//
//  ModelCreationUtils.mm
//  CoreML_tests
//
//  Created by Anil Katti on 4/8/19.
//  Copyright © 2019 Apple Inc. All rights reserved.
//

#include "ModelCreationUtils.hpp"
#include "ParameterTests.hpp"

using namespace CoreML;

Specification::NeuralNetwork* buildBasicUpdatableNeuralNetworkModel(Specification::Model& m) {
    TensorAttributes inTensorAttr = { "A", 3 };
    TensorAttributes outTensorAttr = { "B", 1 };
    
    return buildBasicNeuralNetworkModel(m, true, &inTensorAttr, &outTensorAttr);
}

Specification::NeuralNetwork* buildBasicNeuralNetworkModel(Specification::Model& m, bool isUpdatable, const TensorAttributes *inTensorAttr, const TensorAttributes *outTensorAttr) {
    auto inTensor = m.mutable_description()->add_input();
    inTensor->set_name(inTensorAttr->name);
    auto inTensorShape = inTensor->mutable_type()->mutable_multiarraytype();
    inTensorShape->set_datatype(Specification::ArrayFeatureType_ArrayDataType_FLOAT32);
    for (int i = 0; i < inTensorAttr->dimension; i++) {
        inTensorShape->add_shape(1);
    }
    
    auto outTensor = m.mutable_description()->add_output();
    outTensor->set_name(outTensorAttr->name);
    auto outTensorShape = outTensor->mutable_type()->mutable_multiarraytype();
    outTensorShape->set_datatype(Specification::ArrayFeatureType_ArrayDataType_FLOAT32);
    for (int i = 0; i < outTensorAttr->dimension; i++) {
        outTensorShape->add_shape(1);
    }
    
    m.set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS13);
    
    auto neuralNet = m.mutable_neuralnetwork();
    auto layer = neuralNet->add_layers();
    layer->set_name("inner_layer");
    layer->add_input(inTensorAttr->name);
    layer->add_output(outTensorAttr->name);
    
    Specification::InnerProductLayerParams *innerProductParams = layer->mutable_innerproduct();
    innerProductParams->set_inputchannels(1);
    innerProductParams->set_outputchannels(1);
    innerProductParams->mutable_weights()->add_floatvalue(1.0);
    innerProductParams->set_hasbias(true);
    innerProductParams->mutable_bias()->add_floatvalue(1.0);
    
    if (isUpdatable) {
        m.set_isupdatable(true);
        layer->set_isupdatable(true);
        innerProductParams->mutable_weights()->set_isupdatable(true);
        innerProductParams->mutable_bias()->set_isupdatable(true);
    }
    
    return neuralNet;
}

Specification::KNearestNeighborsClassifier* buildBasicNearestNeighborClassifier(Specification::Model& m, bool isUpdatable, const TensorAttributes *inTensorAttr, const char *outTensorName) {
    auto inTensor = m.mutable_description()->add_input();
    inTensor->set_name(inTensorAttr->name);
    auto inTensorShape = inTensor->mutable_type()->mutable_multiarraytype();
    inTensorShape->set_datatype(Specification::ArrayFeatureType_ArrayDataType_FLOAT32);
    for (int i = 0; i < inTensorAttr->dimension; i++) {
        inTensorShape->add_shape(1);
    }
    
    auto outTensor = m.mutable_description()->add_output();
    outTensor->set_name(outTensorName);
    outTensor->mutable_type()->mutable_stringtype();
    
    m.set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS13);
    
    auto nearestNeighborClassifier = m.mutable_knearestneighborsclassifier();
    nearestNeighborClassifier->set_k(3);
    
    auto nearestNeighborIndex = nearestNeighborClassifier->mutable_nearestneighborsindex();
    
    nearestNeighborIndex->mutable_singlekdtreeindex()->set_leafsize(30);
    nearestNeighborIndex->mutable_squaredeuclideandistance();
    
    nearestNeighborIndex->set_numberofdimensions(inTensorAttr->dimension);
    
    auto floatVector = nearestNeighborIndex->add_floatsamples();
    auto pointVector = floatVector->mutable_vector();
    for (int i = 0; i < inTensorAttr->dimension; i++) {
        pointVector->Add((float)i);
    }
    
    nearestNeighborClassifier->mutable_uniformweighting();
    nearestNeighborClassifier->mutable_stringclasslabels()->add_vector(std::string("zero"));
    
    if (isUpdatable) {
        m.set_isupdatable(true);
    }
    
    return nearestNeighborClassifier;
}

Specification::Pipeline* buildEmptyPipelineModel(Specification::Model& m, bool isUpdatable, const TensorAttributes *inTensorAttr, const TensorAttributes *outTensorAttr) {
    auto inTensor = m.mutable_description()->add_input();
    inTensor->set_name(inTensorAttr->name);
    auto inTensorShape = inTensor->mutable_type()->mutable_multiarraytype();
    inTensorShape->set_datatype(Specification::ArrayFeatureType_ArrayDataType_FLOAT32);
    for (int i = 0; i < inTensorAttr->dimension; i++) {
        inTensorShape->add_shape(1);
    }
    
    auto outTensor = m.mutable_description()->add_output();
    outTensor->set_name(outTensorAttr->name);
    auto outTensorShape = outTensor->mutable_type()->mutable_multiarraytype();
    outTensorShape->set_datatype(Specification::ArrayFeatureType_ArrayDataType_FLOAT32);
    for (int i = 0; i < outTensorAttr->dimension; i++) {
        outTensorShape->add_shape(1);
    }
    
    m.set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS13);
    
    auto pipeline = m.mutable_pipeline();
    
    if (isUpdatable) {
        m.set_isupdatable(true);
    }
    
    return pipeline;
}

Specification::Pipeline* buildEmptyPipelineModelWithStringOutput(Specification::Model& m, bool isUpdatable, const TensorAttributes *inTensorAttr, const char *outTensorName) {
    auto inTensor = m.mutable_description()->add_input();
    inTensor->set_name(inTensorAttr->name);
    auto inTensorShape = inTensor->mutable_type()->mutable_multiarraytype();
    inTensorShape->set_datatype(Specification::ArrayFeatureType_ArrayDataType_FLOAT32);
    for (int i = 0; i < inTensorAttr->dimension; i++) {
        inTensorShape->add_shape(1);
    }
    
    auto outTensor = m.mutable_description()->add_output();
    outTensor->set_name(outTensorName);
    outTensor->mutable_type()->mutable_stringtype();
    
    m.set_specificationversion(MLMODEL_SPECIFICATION_VERSION_IOS13);
    
    auto pipeline = m.mutable_pipeline();
    
    if (isUpdatable) {
        m.set_isupdatable(true);
    }
    
    return pipeline;
}

void addCrossEntropyLossAndSGDOptimizer(Specification::Model& m, const char *lossInputName) {
    
    auto neuralNets = m.mutable_neuralnetwork();
    
    Specification::NetworkUpdateParameters *updateParams = neuralNets->mutable_updateparams();
    Specification::LossLayer *lossLayer = updateParams->add_losslayers();
    lossLayer->set_name("cross_entropy_loss_layer");
    
    Specification::CrossEntropyLossLayer *ceLossLayer = lossLayer->mutable_crossentropylosslayer();
    ceLossLayer->set_input(lossInputName);
    ceLossLayer->set_target("target");
    
    addLearningRate(neuralNets, Specification::Optimizer::kSgdOptimizer, 0.7f, 0.0f, 1.0f);
    addMiniBatchSize(m.mutable_neuralnetwork(), Specification::Optimizer::kSgdOptimizer, 10, 5, 100, std::set<int64_t>());
    addEpochs(m.mutable_neuralnetwork(), 100, 0, 100, std::set<int64_t>());
}


