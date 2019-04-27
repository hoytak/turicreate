# Copyright (c) 2017, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
import os,shutil
import numpy as _np
import coremltools
import coremltools.models.datatypes as datatypes
import unittest
import tempfile
from coremltools.proto import Model_pb2
from coremltools.models.utils import rename_feature, save_spec, macos_version
from coremltools.models import MLModel
from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models.pipeline import PipelineRegressor, PipelineClassifier
import pytest

class MLModelUpdatableTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.model_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(self):
        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)

    def create_base_builder(self):
        self.input_features = [('input', datatypes.Array(3))]
        self.output_features = [('output', None)]
        self.output_names = ["output"]

        builder = NeuralNetworkBuilder(self.input_features, self.output_features, disable_rank5_shape_mapping=True)

        W1 = _np.random.uniform(-0.5, 0.5, (3, 3))
        W2 = _np.random.uniform(-0.5, 0.5, (3, 3))
        builder.add_inner_product(name='ip1',
                                  W=W1,
                                  b=None,
                                  input_channels=3,
                                  output_channels=3,
                                  has_bias=False,
                                  input_name='input',
                                  output_name='hidden')
        builder.add_inner_product(name='ip2',
                                  W=W2,
                                  b=None,
                                  input_channels=3,
                                  output_channels=3,
                                  has_bias=False,
                                  input_name='hidden',
                                  output_name='output')

        builder.make_updatable(['ip1', 'ip2'])  # or a dict for weightParams
        return builder

    def test_updatable_model_creation_ce_sgd(self):

        builder = self.create_base_builder()

        builder.make_updatable(['ip1', 'ip2']) # or a dict for weightParams

        builder.set_cross_entropy_loss(name='cross_entropy', input='output', target='target')

        default_learning_rate = builder.create_learning_rate(default_value=1e-2, allowed_range=[0,1])
        default_mini_batch_size = builder.create_mini_batch_size(default_value=10, allowed_range=[10,100])
        default_momentum = builder.create_momentum(default_value=0.0, allowed_range=[0,1])
        default_epochs = builder.create_epochs(default_value=20, allowed_set=[10, 20, 30, 40])

        builder.set_sgd_optimizer(learning_rate=default_learning_rate, mini_batch_size=default_mini_batch_size, momentum=default_momentum)
        builder.set_epochs(epochs=default_epochs)

        model_path = os.path.join(self.model_dir, 'updatable_creation.mlmodel')
        print(model_path)
        save_spec(builder.spec, model_path)

        mlmodel = MLModel(model_path)
        self.assertTrue(mlmodel is not None)
        spec = mlmodel.get_spec()
        self.assertTrue(spec.isUpdatable)
        self.assertTrue(spec.neuralNetwork.layers[0].isUpdatable)
        self.assertTrue(spec.neuralNetwork.layers[0].innerProduct.weights.isUpdatable)
        self.assertTrue(spec.neuralNetwork.layers[1].isUpdatable)
        self.assertTrue(spec.neuralNetwork.layers[1].innerProduct.weights.isUpdatable)

        self.assertTrue(spec.neuralNetwork.updateParams.lossLayers[0].crossEntropyLossLayer is not None)
        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.sgdOptimizer is not None)

        self.assertTrue(_np.isclose(spec.neuralNetwork.updateParams.optimizer.sgdOptimizer.learningRate.defaultValue, 1e-2, atol=1e-4))
        self.assertTrue(_np.isclose(spec.neuralNetwork.updateParams.optimizer.sgdOptimizer.miniBatchSize.defaultValue, 10, atol=1e-4))
        self.assertTrue(_np.isclose(spec.neuralNetwork.updateParams.optimizer.sgdOptimizer.momentum.defaultValue, 0, atol=1e-8))

        self.assertTrue(_np.isclose(spec.neuralNetwork.updateParams.epochs.defaultValue, 20, atol=1e-4))

        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.sgdOptimizer.learningRate.range.minValue == 0)
        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.sgdOptimizer.learningRate.range.maxValue == 1)

        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.sgdOptimizer.miniBatchSize.range.minValue == 10)
        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.sgdOptimizer.miniBatchSize.range.maxValue == 100)

        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.sgdOptimizer.momentum.range.minValue == 0)
        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.sgdOptimizer.momentum.range.maxValue == 1)

        self.assertTrue(spec.neuralNetwork.updateParams.epochs.set.values is not None)
        self.assertTrue(spec.neuralNetwork.updateParams.epochs.set.values[0] == 10)
        self.assertTrue(spec.neuralNetwork.updateParams.epochs.set.values[1] == 20)
        self.assertTrue(spec.neuralNetwork.updateParams.epochs.set.values[2] == 30)
        self.assertTrue(spec.neuralNetwork.updateParams.epochs.set.values[3] == 40)


    def test_updatable_model_creation_ce_adam(self):

        builder = self.create_base_builder()

        builder.make_updatable(['ip1', 'ip2']) # or a dict for weightParams

        builder.set_cross_entropy_loss(name='cross_entropy', input='output', target='target')

        default_learning_rate = builder.create_learning_rate(default_value=1e-2, allowed_range=[0,1])
        default_mini_batch_size = builder.create_mini_batch_size(default_value=10, allowed_range=[10,100])
        default_beta1 = builder.create_beta1(default_value=0.9, allowed_range=[0,1])
        default_beta2 = builder.create_beta2(default_value=0.999, allowed_range=[0,1])
        default_eps = builder.create_eps(default_value=1e-8, allowed_range=[0,1])
        default_epochs = builder.create_epochs(default_value=20, allowed_set=[10, 20, 30, 40])

        builder.set_adam_optimizer(learning_rate=default_learning_rate, mini_batch_size=default_mini_batch_size,
                                   beta1=default_beta1, beta2=default_beta2, eps=default_eps)
        builder.set_epochs(epochs=default_epochs)

        model_path = os.path.join(self.model_dir, 'updatable_creation.mlmodel')
        print(model_path)
        save_spec(builder.spec, model_path)

        mlmodel = MLModel(model_path)
        self.assertTrue(mlmodel is not None)
        spec = mlmodel.get_spec()
        self.assertTrue(spec.isUpdatable)
        self.assertTrue(spec.neuralNetwork.layers[0].isUpdatable)
        self.assertTrue(spec.neuralNetwork.layers[0].innerProduct.weights.isUpdatable)
        self.assertTrue(spec.neuralNetwork.layers[1].isUpdatable)
        self.assertTrue(spec.neuralNetwork.layers[1].innerProduct.weights.isUpdatable)

        self.assertTrue(spec.neuralNetwork.updateParams.lossLayers[0].crossEntropyLossLayer is not None)
        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.adamOptimizer is not None)

        self.assertTrue(_np.isclose(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.learningRate.defaultValue, 1e-2, atol=1e-4))
        self.assertTrue(_np.isclose(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.miniBatchSize.defaultValue, 10, atol=1e-4))
        self.assertTrue(_np.isclose(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.beta1.defaultValue, 0.9, atol=1e-4))
        self.assertTrue(_np.isclose(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.beta2.defaultValue, 0.999, atol=1e-4))
        self.assertTrue(_np.isclose(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.eps.defaultValue, 1e-8, atol=1e-8))

        self.assertTrue(_np.isclose(spec.neuralNetwork.updateParams.epochs.defaultValue, 20, atol=1e-4))

        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.learningRate.range.minValue == 0)
        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.learningRate.range.maxValue == 1)

        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.miniBatchSize.range.minValue == 10)
        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.miniBatchSize.range.maxValue == 100)

        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.beta1.range.minValue == 0)
        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.beta1.range.maxValue == 1)

        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.beta2.range.minValue == 0)
        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.beta2.range.maxValue == 1)

        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.eps.range.minValue == 0)
        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.eps.range.maxValue == 1)

        self.assertTrue(spec.neuralNetwork.updateParams.epochs.set.values is not None)
        self.assertTrue(spec.neuralNetwork.updateParams.epochs.set.values[0] == 10)
        self.assertTrue(spec.neuralNetwork.updateParams.epochs.set.values[1] == 20)
        self.assertTrue(spec.neuralNetwork.updateParams.epochs.set.values[2] == 30)
        self.assertTrue(spec.neuralNetwork.updateParams.epochs.set.values[3] == 40)

    def test_updatable_model_creation_mse_sgd(self):

        builder = self.create_base_builder()

        builder.make_updatable(['ip1', 'ip2'])  # or a dict for weightParams

        builder.set_mse_loss(name='mse', input='output', target='target')

        default_learning_rate = builder.create_learning_rate(default_value=1e-2, allowed_range=[0, 1])
        default_mini_batch_size = builder.create_mini_batch_size(default_value=10, allowed_range=[10, 100])
        default_momentum = builder.create_momentum(default_value=0.0, allowed_range=[0, 1])
        default_epochs = builder.create_epochs(default_value=20, allowed_set=[10, 20, 30, 40])

        builder.set_sgd_optimizer(learning_rate=default_learning_rate, mini_batch_size=default_mini_batch_size,
                                  momentum=default_momentum)
        builder.set_epochs(epochs=default_epochs)

        model_path = os.path.join(self.model_dir, 'updatable_creation.mlmodel')
        print(model_path)
        save_spec(builder.spec, model_path)

        mlmodel = MLModel(model_path)
        self.assertTrue(mlmodel is not None)
        spec = mlmodel.get_spec()
        self.assertTrue(spec.isUpdatable)
        self.assertTrue(spec.neuralNetwork.layers[0].isUpdatable)
        self.assertTrue(spec.neuralNetwork.layers[0].innerProduct.weights.isUpdatable)
        self.assertTrue(spec.neuralNetwork.layers[1].isUpdatable)
        self.assertTrue(spec.neuralNetwork.layers[1].innerProduct.weights.isUpdatable)

        self.assertTrue(spec.neuralNetwork.updateParams.lossLayers[0].crossEntropyLossLayer is not None)
        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.sgdOptimizer is not None)

        self.assertTrue(
            _np.isclose(spec.neuralNetwork.updateParams.optimizer.sgdOptimizer.learningRate.defaultValue, 1e-2,
                        atol=1e-4))
        self.assertTrue(
            _np.isclose(spec.neuralNetwork.updateParams.optimizer.sgdOptimizer.miniBatchSize.defaultValue, 10,
                        atol=1e-4))
        self.assertTrue(_np.isclose(spec.neuralNetwork.updateParams.optimizer.sgdOptimizer.momentum.defaultValue, 0, atol=1e-8))
        self.assertTrue(_np.isclose(spec.neuralNetwork.updateParams.epochs.defaultValue, 20, atol=1e-4))

        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.sgdOptimizer.learningRate.range.minValue == 0)
        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.sgdOptimizer.learningRate.range.maxValue == 1)

        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.sgdOptimizer.miniBatchSize.range.minValue == 10)
        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.sgdOptimizer.miniBatchSize.range.maxValue == 100)

        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.sgdOptimizer.momentum.range.minValue == 0)
        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.sgdOptimizer.momentum.range.maxValue == 1)

        self.assertTrue(spec.neuralNetwork.updateParams.epochs.set.values is not None)
        self.assertTrue(spec.neuralNetwork.updateParams.epochs.set.values[0] == 10)
        self.assertTrue(spec.neuralNetwork.updateParams.epochs.set.values[1] == 20)
        self.assertTrue(spec.neuralNetwork.updateParams.epochs.set.values[2] == 30)
        self.assertTrue(spec.neuralNetwork.updateParams.epochs.set.values[3] == 40)


    def test_updatable_model_creation_mse_adam(self):

        builder = self.create_base_builder()

        builder.make_updatable(['ip1', 'ip2']) # or a dict for weightParams

        builder.set_mse_loss(name='cross_entropy', input='output', target='target')

        default_learning_rate = builder.create_learning_rate(default_value=1e-2, allowed_range=[0,1])
        default_mini_batch_size = builder.create_mini_batch_size(default_value=10, allowed_range=[10,100])
        default_beta1 = builder.create_beta1(default_value=0.9, allowed_range=[0,1])
        default_beta2 = builder.create_beta2(default_value=0.999, allowed_range=[0,1])
        default_eps = builder.create_eps(default_value=1e-8, allowed_range=[0,1])
        default_epochs = builder.create_epochs(default_value=20, allowed_set=[10, 20, 30, 40])

        builder.set_adam_optimizer(learning_rate=default_learning_rate, mini_batch_size=default_mini_batch_size,
                                   beta1=default_beta1, beta2=default_beta2, eps=default_eps)
        builder.set_epochs(epochs=default_epochs)

        model_path = os.path.join(self.model_dir, 'updatable_creation.mlmodel')
        print(model_path)
        save_spec(builder.spec, model_path)

        mlmodel = MLModel(model_path)
        self.assertTrue(mlmodel is not None)
        spec = mlmodel.get_spec()
        self.assertTrue(spec.isUpdatable)
        self.assertTrue(spec.neuralNetwork.layers[0].isUpdatable)
        self.assertTrue(spec.neuralNetwork.layers[0].innerProduct.weights.isUpdatable)
        self.assertTrue(spec.neuralNetwork.layers[1].isUpdatable)
        self.assertTrue(spec.neuralNetwork.layers[1].innerProduct.weights.isUpdatable)

        self.assertTrue(spec.neuralNetwork.updateParams.lossLayers[0].crossEntropyLossLayer is not None)
        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.adamOptimizer is not None)

        self.assertTrue(_np.isclose(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.learningRate.defaultValue, 1e-2, atol=1e-4))
        self.assertTrue(_np.isclose(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.miniBatchSize.defaultValue, 10, atol=1e-4))
        self.assertTrue(_np.isclose(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.beta1.defaultValue, 0.9, atol=1e-4))
        self.assertTrue(_np.isclose(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.beta2.defaultValue, 0.999, atol=1e-4))
        self.assertTrue(_np.isclose(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.eps.defaultValue, 1e-8, atol=1e-8))
        self.assertTrue(_np.isclose(spec.neuralNetwork.updateParams.epochs.defaultValue, 20, atol=1e-4))

        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.learningRate.range.minValue == 0)
        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.learningRate.range.maxValue == 1)

        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.miniBatchSize.range.minValue == 10)
        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.miniBatchSize.range.maxValue == 100)

        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.beta1.range.minValue == 0)
        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.beta1.range.maxValue == 1)

        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.beta2.range.minValue == 0)
        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.beta2.range.maxValue == 1)

        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.eps.range.minValue == 0)
        self.assertTrue(spec.neuralNetwork.updateParams.optimizer.adamOptimizer.eps.range.maxValue == 1)

        self.assertTrue(spec.neuralNetwork.updateParams.epochs.set.values is not None)
        self.assertTrue(spec.neuralNetwork.updateParams.epochs.set.values[0] == 10)
        self.assertTrue(spec.neuralNetwork.updateParams.epochs.set.values[1] == 20)
        self.assertTrue(spec.neuralNetwork.updateParams.epochs.set.values[2] == 30)
        self.assertTrue(spec.neuralNetwork.updateParams.epochs.set.values[3] == 40)

    def test_pipeline_regressor_make_updatable(self):
        builder = self.create_base_builder()
        builder.spec.isUpdatable = False

        # fails due to missing sub-models
        p_regressor = PipelineRegressor(self.input_features, self.output_names)
        with self.assertRaises(ValueError):
            p_regressor.make_updatable()
        self.assertEqual(p_regressor.spec.isUpdatable, False)

        # fails due to sub-model being not updatable
        p_regressor.add_model(builder.spec)
        with self.assertRaises(ValueError):
            p_regressor.make_updatable()
        self.assertEqual(p_regressor.spec.isUpdatable, False)

        builder.spec.isUpdatable = True
        p_regressor.add_model(builder.spec)

        self.assertEqual(p_regressor.spec.isUpdatable, False)
        p_regressor.make_updatable();
        self.assertEqual(p_regressor.spec.isUpdatable, True)

        # fails since once updatable does not allow adding new models
        with self.assertRaises(ValueError):
            p_regressor.add_model(builder.spec)
        self.assertEqual(p_regressor.spec.isUpdatable, True)

    def test_pipeline_classifier_make_updatable(self):
        builder = self.create_base_builder()
        builder.spec.isUpdatable = False

        # fails due to missing sub-models
        p_classifier = PipelineClassifier(self.input_features, self.output_names)
        with self.assertRaises(ValueError):
            p_classifier.make_updatable()
        self.assertEqual(p_classifier.spec.isUpdatable, False)

        # fails due to sub-model being not updatable
        p_classifier.add_model(builder.spec)
        with self.assertRaises(ValueError):
            p_classifier.make_updatable()
        self.assertEqual(p_classifier.spec.isUpdatable, False)

        builder.spec.isUpdatable = True
        p_classifier.add_model(builder.spec)

        self.assertEqual(p_classifier.spec.isUpdatable, False)
        p_classifier.make_updatable();
        self.assertEqual(p_classifier.spec.isUpdatable, True)

        # fails since once updatable does not allow adding new models
        with self.assertRaises(ValueError):
            p_classifier.add_model(builder.spec)
        self.assertEqual(p_classifier.spec.isUpdatable, True)