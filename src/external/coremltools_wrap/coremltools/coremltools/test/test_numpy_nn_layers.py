from __future__ import print_function

import itertools
import math
import os
import random
import shutil
import tempfile
import unittest
import uuid

import numpy as np

import coremltools
import coremltools.models.datatypes as datatypes
from coremltools.models import _MLMODEL_FULL_PRECISION, _MLMODEL_HALF_PRECISION
from coremltools.models import neural_network as neural_network
from coremltools.models.utils import macos_version

np.random.seed(10)

MIN_MACOS_VERSION_REQUIRED = (10, 13)
LAYERS_10_15_MACOS_VERSION = (10, 15)


def _get_unary_model_spec(x, mode, alpha=1.0):
    input_dim = x.shape
    input_features = [('data', datatypes.Array(*input_dim))]
    output_features = [('output', datatypes.Array(*input_dim))]

    builder = neural_network.NeuralNetworkBuilder(input_features,
                                                  output_features)

    builder.add_unary(name='unary', input_name='data',
                      output_name='output', mode=mode, alpha=alpha)
    return builder.spec


class CorrectnessTest(unittest.TestCase):
    def runTest(self):
        pass

    def _compare_shapes(self, np_preds, coreml_preds):
        if np.squeeze(np_preds).shape != np.squeeze(coreml_preds).shape:
            return False
        return True

    def _compare_predictions(self, np_preds, coreml_preds, delta=.01):
        np_preds = np_preds.flatten()
        coreml_preds = coreml_preds.flatten()
        for i in range(len(np_preds)):
            max_den = max(1.0, np_preds[i], coreml_preds[i])
            if np.abs(
                    np_preds[i] / max_den - coreml_preds[i] / max_den) > delta:
                return False
        return True

    def _test_model(self,
                    model,
                    input,
                    expected,
                    model_precision=_MLMODEL_FULL_PRECISION,
                    useCPUOnly=False,
                    validate_shapes_only=False):
        model_dir = None
        # if we're given a path to a model
        if isinstance(model, str):
            model = coremltools.models.MLModel(model)

        # If we're passed in a specification, save out the model
        # and then load it back up
        elif isinstance(model, coremltools.proto.Model_pb2.Model):
            model_dir = tempfile.mkdtemp()
            model_name = str(uuid.uuid4()) + '.mlmodel'
            model_path = os.path.join(model_dir, model_name)
            coremltools.utils.save_spec(model, model_path)
            model = coremltools.models.MLModel(model, useCPUOnly=useCPUOnly)

        # If we want to test the half precision case
        if model_precision == _MLMODEL_HALF_PRECISION:
            model = coremltools.utils.convert_neural_network_weights_to_fp16(
                model)

        prediction = model.predict(input, useCPUOnly=useCPUOnly)

        for output_name in expected:
            assert (self._compare_shapes(expected[output_name],
                                         prediction[output_name]))
            if not validate_shapes_only:
                assert (self._compare_predictions(expected[output_name],
                                                  prediction[output_name]))

        # Remove the temporary directory if we created one
        if model_dir and os.path.exists(model_dir):
            shutil.rmtree(model_dir)


@unittest.skipIf(macos_version() < MIN_MACOS_VERSION_REQUIRED,
                 'macOS 10.13+ is required. Skipping tests.')
class SimpleTest(CorrectnessTest):

    def test_tiny_upsample_linear_mode(self):
        input_dim = (1, 1, 3)  # (C,H,W)
        input_features = [('data', datatypes.Array(*input_dim))]
        output_features = [('output', None)]

        builder = neural_network.NeuralNetworkBuilder(input_features,
                                                      output_features)
        builder.add_upsample(name='upsample',
                             scaling_factor_h=2, scaling_factor_w=3,
                             input_name='data', output_name='output',
                             mode='BILINEAR')

        input = {
            'data': np.reshape(np.array([1.0, 2.0, 3.0]), (1, 1, 3))
        }
        expected = {
            'output': np.array(
                [[1, 1.333, 1.666, 2, 2.333, 2.666, 3, 3, 3],
                 [1, 1.333, 1.6666, 2, 2.33333, 2.6666, 3, 3, 3]
                 ])
        }

        self._test_model(builder.spec, input, expected)

    def test_LRN(self):
        input_dim = (1, 3, 3)
        input_features = [('data', datatypes.Array(*input_dim))]
        output_features = [('output', datatypes.Array(*input_dim))]

        builder = neural_network.NeuralNetworkBuilder(input_features,
                                                      output_features)
        builder.add_lrn(name='lrn', input_name='data', output_name='output',
                        alpha=2, beta=3, local_size=1, k=8)

        input = {
            'data': np.ones((1, 3, 3))
        }
        expected = {
            'output': 1e-3 * np.ones((1, 3, 3))
        }

        self._test_model(builder.spec, input, expected)

    def test_MVN(self):
        input_dim = (2, 2, 2)
        input_features = [('data', datatypes.Array(*input_dim))]
        output_features = [('output', datatypes.Array(*input_dim))]

        builder = neural_network.NeuralNetworkBuilder(input_features,
                                                      output_features)
        builder.add_mvn(name='mvn', input_name='data', output_name='output',
                        across_channels=False, normalize_variance=False)

        input = {
            'data': np.reshape(np.arange(8, dtype=np.float32), (2, 2, 2))
        }
        expected = {
            'output': np.reshape(np.arange(8) - np.array(
                [1.5, 1.5, 1.5, 1.5, 5.5, 5.5, 5.5, 5.5]), (2, 2, 2))
        }

        self._test_model(builder.spec, input, expected)

    def test_L2_normalize(self):
        input_dim = (1, 2, 2)
        input_features = [('data', datatypes.Array(*input_dim))]
        output_features = [('output', datatypes.Array(*input_dim))]

        builder = neural_network.NeuralNetworkBuilder(input_features,
                                                      output_features)
        builder.add_l2_normalize(name='mvn', input_name='data',
                                 output_name='output')

        input = {
            'data': np.reshape(np.arange(4, dtype=np.float32), (1, 2, 2))
        }
        expected = {
            'output': np.reshape(np.arange(4, dtype=np.float32),
                                 (1, 2, 2)) / np.sqrt(14)
        }

        self._test_model(builder.spec, input, expected)

    def test_unary_sqrt(self):
        x = np.reshape(np.arange(1, 5, dtype=np.float32), (1, 2, 2))
        input = {'data': x}
        expected = {'output': np.sqrt(x)}
        spec = _get_unary_model_spec(x, 'sqrt')
        self._test_model(spec, input, expected)

    def test_unary_rsqrt(self):
        x = np.reshape(np.arange(1, 5, dtype=np.float32), (1, 2, 2))
        input = {'data': x}
        expected = {'output': 1 / np.sqrt(x)}
        spec = _get_unary_model_spec(x, 'rsqrt')
        self._test_model(spec, input, expected)

    def test_unary_inverse(self):
        x = np.reshape(np.arange(1, 5, dtype=np.float32), (1, 2, 2))
        input = {'data': x}
        expected = {'output': 1 / x}
        spec = _get_unary_model_spec(x, 'inverse')
        self._test_model(spec, input, expected)

    def test_unary_power(self):
        x = np.reshape(np.arange(1, 5, dtype=np.float32), (1, 2, 2))
        input = {'data': x}
        expected = {'output': x ** 3}
        spec = _get_unary_model_spec(x, 'power', 3)
        self._test_model(spec, input, expected)

    def test_unary_exp(self):
        x = np.reshape(np.arange(1, 5, dtype=np.float32), (1, 2, 2))
        input = {'data': x}
        expected = {'output': np.exp(x)}
        spec = _get_unary_model_spec(x, 'exp')
        self._test_model(spec, input, expected)

    def test_unary_log(self):
        x = np.reshape(np.arange(1, 5, dtype=np.float32), (1, 2, 2))
        input = {'data': x}
        expected = {'output': np.log(x)}
        spec = _get_unary_model_spec(x, 'log')
        self._test_model(spec, input, expected)

    def test_unary_abs(self):
        x = np.reshape(np.arange(1, 5, dtype=np.float32), (1, 2, 2))
        input = {'data': x}
        expected = {'output': np.abs(x)}
        spec = _get_unary_model_spec(x, 'abs')
        self._test_model(spec, input, expected)

    def test_unary_threshold(self):
        x = np.reshape(np.arange(1, 5, dtype=np.float32), (1, 2, 2))
        input = {'data': x}
        expected = {'output': np.maximum(x, 2)}
        spec = _get_unary_model_spec(x, 'threshold', 2)
        self._test_model(spec, input, expected)

    def test_split(self):
        input_dim = (9, 2, 2)
        x = np.random.rand(*input_dim)

        input_features = [('data', datatypes.Array(*input_dim))]
        output_names = []
        output_features = []
        for i in range(3):
            out = 'out_' + str(i)
            output_names.append(out)
            output_features.append((out, None))

        builder = neural_network.NeuralNetworkBuilder(input_features,
                                                      output_features)
        builder.add_split(name='split', input_name='data',
                          output_names=output_names)

        input = {'data': x}
        expected = {
            'out_0': x[0: 3, :, :],
            'out_1': x[3: 6, :, :],
            'out_2': x[6: 9, :, :]
        }

        self._test_model(builder.spec, input, expected)

    def test_scale_constant(self):
        input_dim = (1, 2, 2)
        input_features = [('data', datatypes.Array(*input_dim))]
        output_features = [('output', None)]

        builder = neural_network.NeuralNetworkBuilder(input_features,
                                                      output_features)
        builder.add_scale(name='scale', W=5, b=45, has_bias=True,
                          input_name='data', output_name='output')

        x = np.reshape(np.arange(4, dtype=np.float32), (1, 2, 2))
        input = {'data': x}
        expected = {'output': 5 * x + 45}

        self._test_model(builder.spec, input, expected)

    def test_scale_matrix(self):
        input_dim = (1, 2, 2)
        input_features = [('data', datatypes.Array(*input_dim))]
        output_features = [('output', None)]

        builder = neural_network.NeuralNetworkBuilder(input_features,
                                                      output_features)
        W = np.reshape(np.arange(5, 9), (1, 2, 2))

        builder.add_scale(name='scale', W=W, b=None, has_bias=False,
                          input_name='data', output_name='output',
                          shape_scale=[1, 2, 2])

        x = np.reshape(np.arange(4, dtype=np.float32), (1, 2, 2))
        input = {'data': x}
        expected = {'output': W * x}

        self._test_model(builder.spec, input, expected)

    def test_bias_constant(self):
        input_dim = (1, 2, 2)
        input_features = [('data', datatypes.Array(*input_dim))]
        output_features = [('output', None)]

        builder = neural_network.NeuralNetworkBuilder(input_features,
                                                      output_features)
        builder.add_bias(name='bias', b=45, input_name='data',
                         output_name='output')

        x = np.reshape(np.arange(4, dtype=np.float32), (1, 2, 2))
        input = {'data': x}
        expected = {'output': x + 45}

        self._test_model(builder.spec, input, expected)

    def test_bias_matrix(self):
        input_dim = (1, 2, 2)
        input_features = [('data', datatypes.Array(*input_dim))]
        output_features = [('output', None)]

        builder = neural_network.NeuralNetworkBuilder(input_features,
                                                      output_features)
        b = np.reshape(np.arange(5, 9), (1, 2, 2))

        builder.add_bias(name='bias', b=b, input_name='data',
                         output_name='output',
                         shape_bias=[1, 2, 2])

        x = np.reshape(np.arange(4, dtype=np.float32), (1, 2, 2))
        input = {'data': x}
        expected = {'output': x + b}

        self._test_model(builder.spec, input, expected)

    def test_load_constant(self, model_precision=_MLMODEL_FULL_PRECISION):
        input_dim = (1, 2, 2)
        input_features = [('data', datatypes.Array(*input_dim))]
        output_features = [('output', None)]

        builder = neural_network.NeuralNetworkBuilder(input_features,
                                                      output_features)
        b = np.reshape(np.arange(5, 9), (1, 2, 2))

        builder.add_load_constant(name='load_constant', output_name='bias',
                                  constant_value=b, shape=[1, 2, 2])
        builder.add_elementwise(name='add', input_names=['data', 'bias'],
                                output_name='output', mode='ADD')

        # save the model
        model_dir = tempfile.mkdtemp()
        model_path = os.path.join(model_dir, 'test_layer.mlmodel')
        coremltools.utils.save_spec(builder.spec, model_path)

        x = np.reshape(np.arange(4, dtype=np.float32), (1, 2, 2))
        input = {'data': x}
        expected = {'output': x + b}

        self._test_model(builder.spec, input, expected, model_precision)

    def test_load_constant_half_precision(self):
        self.test_load_constant(model_precision=_MLMODEL_HALF_PRECISION)

    def test_min(self):
        input_dim = (1, 2, 2)
        input_features = [('data_0', datatypes.Array(*input_dim)),
                          ('data_1', datatypes.Array(*input_dim))]
        output_features = [('output', None)]

        builder = neural_network.NeuralNetworkBuilder(input_features,
                                                      output_features)

        builder.add_elementwise(name='min', input_names=['data_0', 'data_1'],
                                output_name='output', mode='MIN')
        x1 = np.reshape(np.arange(4, dtype=np.float32), (1, 2, 2))
        x2 = np.reshape(np.arange(2, 6, dtype=np.float32), (1, 2, 2))

        input = {'data_0': x1, 'data_1': x2}
        expected = {'output': np.minimum(x1, x2)}

        self._test_model(builder.spec, input, expected)

    def test_conv_same_padding(self):
        input_dim = (10, 15, 15)
        input_features = [('data', datatypes.Array(*input_dim))]
        output_features = [('output', None)]

        builder = neural_network.NeuralNetworkBuilder(input_features,
                                                      output_features)
        W = np.random.rand(3, 3, 10, 20)

        builder.add_convolution(name='conv', kernel_channels=10,
                                output_channels=20,
                                height=3, width=3, stride_height=2,
                                stride_width=2,
                                border_mode='same', groups=1,
                                W=W, b=None, has_bias=False,
                                input_name='data', output_name='output',
                                same_padding_asymmetry_mode='TOP_LEFT_HEAVY')

        x = np.random.rand(*input_dim)
        input = {'data': x}
        expected = {'output': np.random.rand(20, 8, 8)}

        self._test_model(
            builder.spec, input, expected, validate_shapes_only=True)

    def test_deconv_valid_padding(self):
        input_dim = (10, 15, 15)
        input_features = [('data', datatypes.Array(*input_dim))]
        output_features = [('output', None)]

        builder = neural_network.NeuralNetworkBuilder(input_features,
                                                      output_features)
        W = np.random.rand(3, 3, 10, 20)

        builder.add_convolution(name='deconv', kernel_channels=10,
                                output_channels=20,
                                height=3, width=3, stride_height=2,
                                stride_width=2,
                                border_mode='valid', groups=1,
                                W=W, b=None, has_bias=False,
                                is_deconv=True,
                                input_name='data', output_name='output',
                                padding_top=2, padding_bottom=3,
                                padding_left=2, padding_right=3)

        x = np.random.rand(*input_dim)
        input = {'data': x}
        expected = {'output': np.random.rand(20, 26, 26)}

        self._test_model(
            builder.spec, input, expected, validate_shapes_only=True)

    def test_deconv_non_unit_groups(self):
        input_dim = (16, 15, 15)
        input_features = [('data', datatypes.Array(*input_dim))]
        output_features = [('output', None)]

        builder = neural_network.NeuralNetworkBuilder(
            input_features, output_features)

        W = np.random.rand(3, 3, 16, 5)
        builder.add_convolution(name='deconv', kernel_channels=16,
                                output_channels=20,
                                height=3, width=3, stride_height=2,
                                stride_width=2,
                                border_mode='valid', groups=4,
                                W=W, b=None, has_bias=False,
                                is_deconv=True,
                                input_name='data', output_name='output',
                                padding_top=2, padding_bottom=3,
                                padding_left=2, padding_right=3)

        x = np.random.rand(*input_dim)
        input = {'data': x}
        expected = {'output': np.random.rand(20, 26, 26)}

        self._test_model(
            builder.spec, input, expected, validate_shapes_only=True)

    def test_linear_activation(self):
        input_dim = (10, 15, 15)
        input_features = [('data', datatypes.Array(*input_dim))]
        output_features = [('output', None)]

        builder = neural_network.NeuralNetworkBuilder(input_features,
                                                      output_features)
        builder.add_activation(name='activation',
                               non_linearity='LINEAR',
                               input_name='data',
                               output_name='output', params=[34.0, 67.0])

        x = np.random.rand(*input_dim)
        input = {'data': x}
        expected = {'output': 34.0 * x + 67.0}

        self._test_model(builder.spec, input, expected)

    def test_padding_constant(self):
        input_dim = (1, 2, 3)
        input_features = [('data', datatypes.Array(*input_dim))]
        output_features = [('output', None)]

        builder = neural_network.NeuralNetworkBuilder(
            input_features, output_features)
        builder.add_padding(name='pad',
                            left=1, right=0, top=2, bottom=0,
                            value=-1,
                            input_name='data',
                            output_name='output')

        x = np.reshape(np.array([[1, 2, 3], [4, 5, 6]]), (1, 2, 3)).astype(
            np.float32)
        input = {'data': x}
        y = np.reshape(
            np.array([[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, 1, 2, 3],
                      [-1, 4, 5, 6]]), (1, 4, 4)).astype(np.float32)
        expected = {'output': y}

        self._test_model(builder.spec, input, expected)

    def test_padding_replication(self):
        input_dim = (1, 2, 3)
        input_features = [('data', datatypes.Array(*input_dim))]
        output_features = [('output', None)]

        builder = neural_network.NeuralNetworkBuilder(input_features,
                                                      output_features)
        builder.add_padding(name='pad',
                            left=1, top=2,
                            input_name='data',
                            output_name='output', padding_type='replication')

        x = np.reshape(np.array([[1, 2, 3], [4, 5, 6]]), (1, 2, 3)).astype(
            np.float32)
        input = {'data': x}
        y = np.reshape(np.array([[1, 1, 2, 3], [1, 1, 2, 3], [1, 1, 2, 3],
                                 [4, 4, 5, 6]]), (1, 4, 4)).astype(np.float32)
        expected = {'output': y}

        self._test_model(builder.spec, input, expected)

    def test_reshape_target_shape_3(self):
        input_dim = (1, 2, 5)  # (C,H,W)
        input_features = [('data', datatypes.Array(*input_dim))]
        output_features = [('output', None)]

        builder = neural_network.NeuralNetworkBuilder(input_features,
                                                      output_features)
        builder.add_reshape(name='reshape', input_name='data',
                            output_name='output', target_shape=(10, 1, 1),
                            mode=0)

        x = np.random.rand(*input_dim)
        input = {'data': x}
        expected = {'output': np.reshape(x, (10, 1, 1))}

        self._test_model(builder.spec, input, expected)

    def test_reshape_target_shape_4(self):
        input_dim = (1, 2, 5)  # (C,H,W)
        input_features = [('data', datatypes.Array(*input_dim))]
        output_features = [('output', None)]

        builder = neural_network.NeuralNetworkBuilder(input_features,
                                                      output_features)
        builder.add_reshape(name='reshape', input_name='data',
                            output_name='output', target_shape=(1, 10, 1, 1),
                            mode=0)

        x = np.random.rand(*input_dim)
        input = {'data': x}
        expected = {'output': np.reshape(x, (1, 10, 1, 1))}

        self._test_model(builder.spec, input, expected)

    def test_bias_matrix_CPU(self):
        input_dim = (1, 2, 2)
        input_features = [('data', datatypes.Array(*input_dim))]
        output_features = [('output', None)]

        builder = neural_network.NeuralNetworkBuilder(input_features,
                                                      output_features)
        b = np.reshape(np.arange(5, 9), (1, 2, 2))

        builder.add_bias(name='bias', b=b, input_name='data',
                         output_name='output',
                         shape_bias=[1, 2, 2])

        x = np.reshape(np.arange(4, dtype=np.float32), (1, 2, 2))
        input = {'data': x}
        expected = {'output': x + b}

        self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_linear_activation_CPU(self):
        input_dim = (10, 15, 15)
        input_features = [('data', datatypes.Array(*input_dim))]
        output_features = [('output', None)]

        builder = neural_network.NeuralNetworkBuilder(input_features,
                                                      output_features)
        builder.add_activation(name='activation',
                               non_linearity='LINEAR',
                               input_name='data',
                               output_name='output', params=[34.0, 67.0])

        x = np.random.rand(*input_dim)
        input = {'data': x}
        expected = {'output': 34.0 * x + 67.0}

        self._test_model(builder.spec, input, expected, useCPUOnly=True)


@unittest.skipIf(macos_version() < LAYERS_10_15_MACOS_VERSION,
                 'macOS 10.15+ required. Skipping tests.')
class NewLayersSimpleTest(CorrectnessTest):
    def test_transposeND_CPU(self):
        for rank in range(1, 6):
            axes = np.random.permutation(rank)
            input_dim = np.random.randint(low=2, high=6, size=rank)
            output_dim = tuple(input_dim[axes])
            input_dim = tuple(input_dim)
            input_features = [('data', datatypes.Array(*input_dim))]
            output_features = [('output', datatypes.Array(*output_dim))]

            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True)

            if rank <= 5:
                builder.add_transposeND(name='TransposeND',
                                        axes=axes,
                                        input_name='data',
                                        output_name='output',
                                        rank=rank)
            else:
                builder.add_transposeND(name='TransposeND',
                                        axes=axes,
                                        input_name='data',
                                        output_name='output',
                                        rank=rank,
                                        input_shape=input_dim,
                                        output_shape=output_dim)

            x = np.random.rand(*input_dim)
            input = {'data': x}
            expected = {'output': np.transpose(x, axes)}

            self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_batchedMatMul_CPU(self):
        aShapes = [(10,), (4, 10), (10,), (10,), (2, 3), (1, 3, 4),
                   (1, 3, 1, 2, 3),
                   (1, 1, 2, 3, 1, 3, 4)]
        bShapes = [(10,), (10,), (10, 3), (2, 10, 3), (3, 4), (3, 2, 4, 5),
                   (1, 4, 3, 2),
                   (2, 1, 1, 2, 4, 5)]
        outShapes = [(1, 1), (4, 1), (1, 3), (2, 1, 3), (2, 4), (3, 2, 3, 5),
                     (1, 3, 4, 2, 2), (1, 2, 2, 3, 2, 3, 5)]
        transposeA = False
        transposeB = False
        for aShape, bShape, outShape in zip(aShapes, bShapes, outShapes):
            input_shapes = [aShape, bShape]
            output_shape = outShape
            input_ranks = [len(i) for i in input_shapes]
            output_rank = len(output_shape)
            input_names = ['A', 'B']
            input_features = [('A', datatypes.Array(*input_shapes[0]))]
            input_features.append(('B', datatypes.Array(*input_shapes[1])))
            output_features = [('output', datatypes.Array(*output_shape))]

            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True)

            if output_rank <= 5:
                builder.add_batchedMatMul(name='BatchedMatMul',
                                          input_names=input_names,
                                          output_name='output',
                                          transposeA=transposeA,
                                          transposeB=transposeB,
                                          input_ranks=input_ranks,
                                          output_rank=output_rank)
            else:
                builder.add_batchedMatMul(name='BatchedMatMul',
                                          input_names=input_names,
                                          output_name='output',
                                          transposeA=transposeA,
                                          transposeB=transposeB,
                                          input_ranks=input_ranks,
                                          output_rank=output_rank,
                                          input_shapes=input_shapes,
                                          output_shape=output_shape)

            a = np.random.rand(*input_shapes[0])
            b = np.random.rand(*input_shapes[1])
            input = {'A': a, 'B': b}
            expected = {'output': np.array(np.matmul(a, b))}

            self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_batchedMatMul_withTransposes_CPU(self):
        for transposeA, transposeB in itertools.product([True, False],
                                                        [True, False]):
            aShape = (3, 4)
            bShape = (4, 5)
            outShape = (3, 5)
            aShape = aShape[::-1] if transposeA else aShape
            bShape = bShape[::-1] if transposeB else bShape
            input_shapes = [aShape, bShape]
            output_shape = outShape
            input_ranks = [len(i) for i in input_shapes]
            output_rank = len(output_shape)
            input_names = ['A', 'B']
            input_features = [('A', datatypes.Array(*input_shapes[0]))]
            input_features.append(('B', datatypes.Array(*input_shapes[1])))
            output_features = [('output', datatypes.Array(*output_shape))]

            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True
            )
            builder.add_batchedMatMul(
                name='BatchedMatMul', input_names=input_names,
                output_name='output', transposeA=transposeA,
                transposeB=transposeB, input_ranks=input_ranks,
                output_rank=output_rank
            )
            a = np.random.rand(*input_shapes[0])
            b = np.random.rand(*input_shapes[1])
            input = {'A': a, 'B': b}
            a = a.T if transposeA else a
            b = b.T if transposeB else b
            expected = {'output': np.matmul(a, b)}

            self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_batchedMatMul_single_input_CPU(
            self, model_precision=_MLMODEL_FULL_PRECISION):
        X1 = 11
        X2 = 23
        W = np.random.rand(X1, X2)
        bias = np.random.rand(X2)
        input_shapes = [(X1,), (5, X1), (2, 3, X1), (4, 1, X1), (12, 5, 8, X1),
                        (2, 3, 1, 5, X1)]
        for input_shape in input_shapes:
            x = np.random.rand(*input_shape)
            np_out = np.matmul(x, W) + bias
            expected = {'output': np_out}

            input_features = [('data', datatypes.Array(*input_shape))]
            output_features = [('output', None)]
            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True)

            builder.add_batchedMatMul(name='batched_matmul',
                                      input_names=['data'],
                                      output_name='output',
                                      weight_matrix_rows=X1,
                                      weight_matrix_columns=X2,
                                      W=W, bias=bias)
            input = {'data': x}

            self._test_model(
                builder.spec, input, expected,
                model_precision=model_precision, useCPUOnly=True)

    def test_batchedMatMul_single_input_half_precision_CPU(self):
        self.test_batchedMatMul_single_input_CPU(
            model_precision=_MLMODEL_HALF_PRECISION)

    def test_embeddingND_CPU(
            self, model_precision=_MLMODEL_FULL_PRECISION, useCPUOnly=True):
        vocab_size = 10
        embedding_size = 19
        W = np.random.rand(embedding_size, vocab_size)
        input_shapes = [(5, 1), (2, 3, 1), (4, 1, 1), (12, 5, 8, 1),
                        (2, 3, 1, 5, 1)]
        for input_shape in input_shapes:
            x = np.random.randint(vocab_size, size=input_shape)

            np_out = np.take(np.transpose(W), np.squeeze(x, axis=-1), axis=0)
            expected = {'output': np_out}

            input_features = [('data', datatypes.Array(*input_shape))]
            output_features = [('output', None)]
            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True)

            builder.add_embeddingND(name='embeddingND',
                                    input_name='data', output_name='output',
                                    vocab_size=vocab_size,
                                    embedding_size=embedding_size,
                                    W=W)

            input = {'data': x.astype(np.float32)}

            self._test_model(
                builder.spec, input, expected,
                model_precision=model_precision, useCPUOnly=useCPUOnly)

    def test_embeddingND_half_precision_CPU(self):
        self.test_embeddingND_CPU(
            model_precision=_MLMODEL_HALF_PRECISION, useCPUOnly=True)

    def test_embeddingND_GPU(self):
        self.test_embeddingND_CPU(
            model_precision=_MLMODEL_FULL_PRECISION, useCPUOnly=False)

    def test_embeddingND_half_precision_GPU(self):
        self.test_embeddingND_CPU(
            model_precision=_MLMODEL_HALF_PRECISION, useCPUOnly=False)

    def test_softmaxND_CPU(self):
        input_shapes = [(10, 15), (10, 15, 2, 3), (10, 2, 4, 15, 1, 4), (6,)]
        axes = [1, -3, -6, 0]
        for axis, input_dim in zip(axes, input_shapes):
            input_features = [('data', datatypes.Array(*input_dim))]
            output_features = [('output', datatypes.Array(*input_dim))]

            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True)

            if len(input_dim) > 5:
                builder.add_softmaxND(name='softmaxND', input_name='data',
                                      output_name='output', axis=axis,
                                      rank=len(input_dim), shape=input_dim)
            else:
                builder.add_softmaxND(name='softmaxND', input_name='data',
                                      output_name='output', axis=axis,
                                      rank=len(input_dim))

            x = np.random.rand(*input_dim)
            input = {'data': x}
            y = np.exp(x - np.max(x, axis=axis, keepdims=True))
            y = y / np.sum(y, axis=axis, keepdims=True)
            expected = {'output': y}

            self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_concatND_CPU(self):
        for rank in range(1, 6):
            for axis in range(-rank, rank):
                nInputs = np.random.choice(range(2, 5))
                output_shape = np.random.randint(low=2, high=5, size=rank)
                output_shape[axis] = 0
                input_shapes = []
                input_features = []
                input_names = []
                for _ in range(nInputs):
                    input_shapes.append(np.copy(output_shape))
                    input_shapes[-1][axis] = np.random.choice(range(2,8))
                    output_shape[axis] += input_shapes[-1][axis]
                for i,input_dim in enumerate(input_shapes):
                    input_name = 'input_%s' % str(i)
                    input_names.append(input_name)
                    input_features.append((input_name, datatypes.Array(*input_dim)))

                output_features = [('output', datatypes.Array(*output_shape))]

                builder = neural_network.NeuralNetworkBuilder(input_features, output_features, disable_rank5_shape_mapping=True)

                builder.add_concatND(name='concatND', input_names = input_names, output_name='output', axis = axis)

                input_tensors = []
                for input_dim in input_shapes:
                    input_tensors.append(np.random.rand(*input_dim))
                input = dict(zip(input_names, input_tensors))
                expected = {'output': np.concatenate(input_tensors, axis)}

                self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_fill_CPU(self):
        for rank in range(1, 6):
            shape = np.random.randint(low=2, high=8, size=rank)
            input_features = []
            input_names = ["tensor", "fill_value"]
            input_shapes = [shape, [1]]
            for i, input_dim in enumerate(input_shapes):
                input_features.append(
                    (input_names[i], datatypes.Array(*input_dim)))

            output_features = [('output', datatypes.Array(*shape))]

            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True)
            builder.add_fill(name='fill', input_names=input_names,
                             output_name='output', rank=rank)

            input_tensors = []
            for input_dim in input_shapes:
                input_tensors.append(np.random.rand(*input_dim))

            input_tensors[0].fill(input_tensors[1][0])
            input = dict(zip(input_names, input_tensors))
            expected = {'output': input_tensors[0]}

            self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_broadcastTo_CPU(self):

        input_shapes = [(2, 1, 3, 1), (1, 2, 1, 3), (3, 4, 5), (1, 4, 1, 3),
                        (4, 5), (2, 1), (1, 3, 1), (1, 2, 1, 3, 1),
                        (1, 2, 1, 3, 1, 2, 1)]
        target_shapes = [(2, 2, 3, 3), (2, 2, 3, 3), (3, 4, 5), (3, 4, 2, 3),
                         (2, 3, 4, 5), (3, 2, 1, 2, 2), (2, 3, 3, 4),
                         (3, 2, 2, 3, 1), (2, 2, 2, 3, 3, 2, 2, 3)]
        for input_shape, target_shape in zip(input_shapes, target_shapes):
            input_features = [('data', datatypes.Array(*input_shape))]
            output_features = [('output', datatypes.Array(*target_shape))]

            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True)

            if len(target_shape) <= 5:
                builder.add_broadcastTo(name='broadcastTo', input_name='data',
                                        output_name='output',
                                        target_shape=target_shape,
                                        input_rank=len(input_shape))
            else:
                builder.add_broadcastTo(name='broadcastTo', input_name='data',
                                        output_name='output',
                                        target_shape=target_shape,
                                        input_rank=len(input_shape),
                                        input_shape=input_shape)

            x = np.random.rand(*input_shape)
            input = {'data': x}
            expected = {'output': np.broadcast_to(x, target_shape)}

            self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_sine_CPU(self):
        for rank in range(1, 6):
            shape = np.random.randint(low=2, high=8, size=rank)
            input_features = [('data', datatypes.Array(*shape))]
            output_features = [('output', datatypes.Array(*shape))]

            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True)
            builder.add_sine(name='sine', input_name='data',
                             output_name='output', rank=rank)

            x = np.random.rand(*shape)
            input = {'data': x}
            expected = {'output': np.sin(x)}

            self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_cosine_CPU(self):
        for rank in range(1, 6):
            shape = np.random.randint(low=2, high=8, size=rank)
            input_features = [('data', datatypes.Array(*shape))]
            output_features = [('output', datatypes.Array(*shape))]

            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True)

            builder.add_cosine(name='cosine', input_name='data',
                               output_name='output', rank=rank)

            x = np.random.rand(*shape)
            input = {'data': x}
            expected = {'output': np.cos(x)}

            self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_exp_CPU(self):
        for rank in range(1, 6):
            shape = np.random.randint(low=2, high=8, size=rank)
            input_features = [('data', datatypes.Array(*shape))]
            output_features = [('output', datatypes.Array(*shape))]
            withBase2 = random.choice([True, False])
            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True)
            builder.add_exp(name='exp', input_name='data', output_name='output',
                            withBase2=withBase2, rank=rank)

            x = np.random.rand(*shape)
            input = {'data': x}
            expected = {'output': np.exp2(x) if withBase2 else np.exp(x)}

            self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_addBroadcastable_CPU(self):
        aShapes = [(1, 2, 3, 1), (3, 1, 2, 1, 2), (1, 2, 1, 3), (2, 3),
                   (2, 1, 1), (2, 3, 4), (2, 4), (1,), (1,)]
        bShapes = [(2, 3, 1, 2, 1, 1, 2), (2, 2, 3, 1), (2, 1, 3, 1), (2, 3),
                   (2, 3, 4), (3, 1), (1,), (2, 3), (1,)]
        outShapes = [(2, 3, 1, 2, 2, 3, 2), (3, 2, 2, 3, 2), (2, 2, 3, 3),
                     (2, 3), (2, 3, 4), (2, 3, 4), (2, 4), (2, 3), (1,)]
        for aShape, bShape, outShape in zip(aShapes, bShapes, outShapes):
            input_shapes = [aShape, bShape]
            output_shape = outShape
            input_ranks = [len(i) for i in input_shapes]
            output_rank = len(output_shape)
            input_names = ['A', 'B']
            input_features = [('A', datatypes.Array(*input_shapes[0])),
                              ('B', datatypes.Array(*input_shapes[1]))]
            output_features = [('output', datatypes.Array(*output_shape))]

            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True)

            if output_rank <= 5:
                builder.add_addBroadcastable(name='addBroadcastable',
                                             input_names=input_names,
                                             output_name='output',
                                             input_ranks=input_ranks,
                                             output_rank=output_rank)
            else:
                builder.add_addBroadcastable(name='addBroadcastable',
                                             input_names=input_names,
                                             output_name='output',
                                             input_ranks=input_ranks,
                                             output_rank=output_rank,
                                             input_shapes=input_shapes,
                                             output_shape=output_shape)

            a = np.random.rand(*input_shapes[0])
            b = np.random.rand(*input_shapes[1])
            input = {'A': a, 'B': b}
            expected = {'output': np.add(a, b)}

            self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_multiplyBroadcastable_CPU(self):
        aShapes = [(1, 2, 3, 1), (3, 1, 2, 1, 2), (1, 2, 1, 3), (2, 3),
                   (2, 1, 1), (2, 3, 4), (2, 4), (1,), (1,)]
        bShapes = [(2, 3, 1, 2, 1, 1, 2), (2, 2, 3, 1), (2, 1, 3, 1), (2, 3),
                   (2, 3, 4), (3, 1), (1,), (2, 3), (1,)]
        outShapes = [(2, 3, 1, 2, 2, 3, 2), (3, 2, 2, 3, 2), (2, 2, 3, 3),
                     (2, 3), (2, 3, 4), (2, 3, 4), (2, 4), (2, 3), (1,)]
        for aShape, bShape, outShape in zip(aShapes, bShapes, outShapes):
            input_shapes = [aShape, bShape]
            output_shape = outShape
            input_ranks = [len(i) for i in input_shapes]
            output_rank = len(output_shape)
            input_names = ['A', 'B']
            input_features = [('A', datatypes.Array(*input_shapes[0])),
                              ('B', datatypes.Array(*input_shapes[1]))]
            output_features = [('output', datatypes.Array(*output_shape))]

            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True)

            if output_rank <= 5:
                builder.add_multiplyBroadcastable(name='multiplyBroadcastable',
                                                  input_names=input_names,
                                                  output_name='output',
                                                  input_ranks=input_ranks,
                                                  output_rank=output_rank)
            else:
                builder.add_multiplyBroadcastable(name='multiplyBroadcastable',
                                                  input_names=input_names,
                                                  output_name='output',
                                                  input_ranks=input_ranks,
                                                  output_rank=output_rank,
                                                  input_shapes=input_shapes,
                                                  output_shape=output_shape)

            a = np.random.rand(*input_shapes[0])
            b = np.random.rand(*input_shapes[1])
            input = {'A': a, 'B': b}
            expected = {'output': np.multiply(a, b)}

            self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_divideBroadcastable_CPU(self):
        aShapes = [(1, 2, 3, 1), (3, 1, 2, 1, 2), (1, 2, 1, 3), (2, 3),
                   (2, 1, 1), (2, 3, 4), (2, 4), (1,), (1,)]
        bShapes = [(2, 3, 1, 2, 1, 1, 2), (2, 2, 3, 1), (2, 1, 3, 1), (2, 3),
                   (2, 3, 4), (3, 1), (1,), (2, 3), (1,)]
        outShapes = [(2, 3, 1, 2, 2, 3, 2), (3, 2, 2, 3, 2), (2, 2, 3, 3),
                     (2, 3), (2, 3, 4), (2, 3, 4), (2, 4), (2, 3), (1,)]
        for aShape, bShape, outShape in zip(aShapes, bShapes, outShapes):
            input_shapes = [aShape, bShape]
            output_shape = outShape
            input_ranks = [len(i) for i in input_shapes]
            output_rank = len(output_shape)
            input_names = ['A', 'B']
            input_features = [('A', datatypes.Array(*input_shapes[0])),
                              ('B', datatypes.Array(*input_shapes[1]))]
            output_features = [('output', datatypes.Array(*output_shape))]

            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True)

            if output_rank <= 5:
                builder.add_divideBroadcastable(name='divideBroadcastable',
                                                input_names=input_names,
                                                output_name='output',
                                                input_ranks=input_ranks,
                                                output_rank=output_rank)
            else:
                builder.add_divideBroadcastable(name='divideBroadcastable',
                                                input_names=input_names,
                                                output_name='output',
                                                input_ranks=input_ranks,
                                                output_rank=output_rank,
                                                input_shapes=input_shapes,
                                                output_shape=output_shape)

            a = np.random.rand(*input_shapes[0])
            b = np.random.rand(*input_shapes[1])
            input = {'A': a, 'B': b}
            expected = {'output': np.divide(a, b)}

            self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_subtractBroadcastable_CPU(self):
        aShapes = [(1, 2, 3, 1), (3, 1, 2, 1, 2), (1, 2, 1, 3), (2, 3),
                   (2, 1, 1), (2, 3, 4), (2, 4), (1,), (1,)]
        bShapes = [(2, 3, 1, 2, 1, 1, 2), (2, 2, 3, 1), (2, 1, 3, 1), (2, 3),
                   (2, 3, 4), (3, 1), (1,), (2, 3), (1,)]
        outShapes = [(2, 3, 1, 2, 2, 3, 2), (3, 2, 2, 3, 2), (2, 2, 3, 3),
                     (2, 3), (2, 3, 4), (2, 3, 4), (2, 4), (2, 3), (1,)]
        for aShape, bShape, outShape in zip(aShapes, bShapes, outShapes):
            input_shapes = [aShape, bShape]
            output_shape = outShape
            input_ranks = [len(i) for i in input_shapes]
            output_rank = len(output_shape)
            input_names = ['A', 'B']
            input_features = [('A', datatypes.Array(*input_shapes[0])),
                              ('B', datatypes.Array(*input_shapes[1]))]
            output_features = [('output', datatypes.Array(*output_shape))]

            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True)

            if output_rank <= 5:
                builder.add_subtractBroadcastable(name='subtractBroadcastable',
                                                  input_names=input_names,
                                                  output_name='output',
                                                  input_ranks=input_ranks,
                                                  output_rank=output_rank)
            else:
                builder.add_subtractBroadcastable(name='subtractBroadcastable',
                                                  input_names=input_names,
                                                  output_name='output',
                                                  input_ranks=input_ranks,
                                                  output_rank=output_rank,
                                                  input_shapes=input_shapes,
                                                  output_shape=output_shape)

            a = np.random.rand(*input_shapes[0])
            b = np.random.rand(*input_shapes[1])
            input = {'A': a, 'B': b}
            expected = {'output': np.subtract(a, b)}

            self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_elementwise_boolean_binary_CPU(self):
        input_names = ['A', 'B']
        aShapes = [(1, 2, 3, 1), (3, 1, 2, 1, 2), (1, 2, 1, 3), (2, 3),
                   (2, 1, 1), (2, 3, 4), (2, 4), (1,), (1,)]
        bShapes = [(3, 2, 1, 1, 2), (2, 2, 3, 1), (2, 1, 3, 1), (2, 3),
                   (2, 3, 4), (3, 1), (1,), (2, 3), (1,)]
        test_cases = ['greater', 'less', 'equal', 'not_equal', 'greater_equal',
                      'less_equal', 'logical_and', 'logical_or', 'logical_xor']
        for aCase in test_cases:
            for aShape, bShape in zip(aShapes, bShapes):
                input_shapes = [aShape, bShape]
                input_features = [('A', datatypes.Array(*input_shapes[0])),
                                  ('B', datatypes.Array(*input_shapes[1]))]

                builder = neural_network.NeuralNetworkBuilder(input_features, [
                    ('output', None)], disable_rank5_shape_mapping=True)
                func = getattr(np, aCase)
                if aCase == 'greater':
                    builder.add_greater_than(aCase, input_names=input_names,
                                             output_name='output')
                elif aCase == 'less':
                    builder.add_less_than(aCase, input_names=input_names,
                                          output_name='output')
                elif aCase == 'equal':
                    builder.add_equal(aCase, input_names=input_names,
                                      output_name='output')
                elif aCase == 'not_equal':
                    builder.add_not_equal(aCase, input_names=input_names,
                                          output_name='output')
                elif aCase == 'greater_equal':
                    builder.add_greater_than(aCase, input_names=input_names,
                                             output_name='output',
                                             use_greater_than_equal=True)
                elif aCase == 'less_equal':
                    builder.add_less_than(aCase, input_names=input_names,
                                          output_name='output',
                                          use_less_than_equal=True)
                elif aCase == 'logical_and':
                    builder.add_logical(aCase, input_names=input_names,
                                        output_name='output', mode='AND')
                elif aCase == 'logical_or':
                    builder.add_logical(aCase, input_names=input_names,
                                        output_name='output', mode='OR')
                elif aCase == 'logical_xor':
                    builder.add_logical(aCase, input_names=input_names,
                                        output_name='output', mode='XOR')

                a = np.random.rand(*input_shapes[0])
                b = np.random.rand(*input_shapes[1])
                input = {'A': a, 'B': b}
                expected = {'output': func(a, b, dtype=np.float32)}

                self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_elementwise_boolean_unary_CPU(self):
        input_names = ['input']
        aShapes = [(1, 2, 3, 1), (3, 1, 2, 1, 2), (1, 2, 1, 3), (2, 3),
                   (2, 1, 1), (2, 3, 4), (2, 4), (1,), (1,)]
        test_cases = ['greater', 'less', 'equal', 'not_equal', 'greater_equal',
                      'less_equal']
        for aCase in test_cases:
            for aShape in aShapes:
                input_features = [('input', datatypes.Array(*aShape))]
                b = np.random.rand()
                builder = neural_network.NeuralNetworkBuilder(
                    input_features, [('output', None)],
                    disable_rank5_shape_mapping=True)

                func = getattr(np, aCase)
                if aCase == 'greater':
                    builder.add_greater_than(aCase, input_names=input_names,
                                             output_name='output', alpha=b)
                elif aCase == 'less':
                    builder.add_less_than(aCase, input_names=input_names,
                                          output_name='output', alpha=b)
                elif aCase == 'equal':
                    builder.add_equal(aCase, input_names=input_names,
                                      output_name='output', alpha=b)
                elif aCase == 'not_equal':
                    builder.add_not_equal(aCase, input_names=input_names,
                                          output_name='output', alpha=b)
                elif aCase == 'greater_equal':
                    builder.add_greater_than(aCase, input_names=input_names,
                                             output_name='output',
                                             use_greater_than_equal=True,
                                             alpha=b)
                elif aCase == 'less_equal':
                    builder.add_less_than(aCase, input_names=input_names,
                                          output_name='output',
                                          use_less_than_equal=True, alpha=b)

                a = np.random.rand(*aShape)
                input = {'input': a}
                expected = {'output': func(a, b, dtype=np.float32)}

                self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_logical_not_CPU(self):
        input_names = ['input']
        aShapes = [(1, 2, 3, 1), (3, 1, 2, 1, 2), (1, 2, 1, 3), (2, 3),
                   (2, 1, 1), (2, 3, 4), (2, 4), (1,), (1,)]
        for aShape in aShapes:
            input_features = [('input', datatypes.Array(*aShape))]
            builder = neural_network.NeuralNetworkBuilder(
                input_features, [('output', None)],
                disable_rank5_shape_mapping=True)
            builder.add_logical('logical_not', input_names=input_names,
                                output_name='output', mode='NOT')

            a = np.random.rand(*aShape)
            input = {'input': a}
            expected = {'output': np.logical_not(a)}

            self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_gather_CPU(self):
        for rankParams, rankIndices in [(i, j) for i in range(1, 5) for j in
                                        range(1, 5)]:
            for axis in range(-rankParams, rankParams):
                shapeParams = np.random.randint(low=2, high=5, size=rankParams)
                shapeIndices = np.random.randint(low=2, high=5,
                                                 size=rankIndices)
                input_shapes = [shapeParams, shapeIndices]
                posAxis = axis if axis >= 0 else axis + rankParams
                output_shape = list(shapeParams[:posAxis]) + list(
                    shapeIndices) + list(shapeParams[posAxis + 1:])
                input_ranks = [len(i) for i in input_shapes]
                output_rank = len(output_shape)
                input_names = ['params', 'indices']
                input_features = [
                    ('params', datatypes.Array(*input_shapes[0])),
                    ('indices', datatypes.Array(*input_shapes[1]))
                ]
                output_features = [
                    ('output', datatypes.Array(*output_shape))
                ]

                builder = neural_network.NeuralNetworkBuilder(
                    input_features, output_features,
                    disable_rank5_shape_mapping=True)

                if output_rank <= 5:
                    builder.add_gather(name='gather', input_names=input_names,
                                       output_name='output',
                                       axis=axis, input_ranks=input_ranks,
                                       output_rank=output_rank)
                else:
                    builder.add_gather(name='gather', input_names=input_names,
                                       output_name='output',
                                       axis=axis, input_ranks=input_ranks,
                                       output_rank=output_rank,
                                       input_shapes=input_shapes,
                                       output_shape=output_shape)

                a = np.random.rand(*input_shapes[0])
                b = np.random.randint(-shapeParams[axis], shapeParams[axis],
                                      size=shapeIndices)
                input = {'params': a, 'indices': b.astype(np.float)}
                expected = {'output': np.take(a, b, axis=axis)}

                self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_stackND_CPU(self):
        for input_rank in range(1, 6):
            for axis in range(-input_rank, input_rank):
                nInputs = np.random.choice(range(2, 5))
                input_shape = np.random.randint(low=2, high=5, size=input_rank)
                posAxis = axis if axis >= 0 else axis + input_rank + 1
                output_shape = list(input_shape[:posAxis]) + [nInputs] + list(
                    input_shape[posAxis:])
                output_rank = input_rank + 1
                input_features = []
                input_names = []
                for i in range(nInputs):
                    input_name = 'input_%s' % str(i)
                    input_names.append(input_name)
                    input_features.append(
                        (input_name, datatypes.Array(*input_shape)))

                output_features = [('output', datatypes.Array(*output_shape))]

                builder = neural_network.NeuralNetworkBuilder(
                    input_features, output_features,
                    disable_rank5_shape_mapping=True)

                if output_rank <= 5:
                    builder.add_stackND(name='stackND', input_names=input_names,
                                        output_name='output', axis=axis,
                                        input_rank=len(input_shape))
                else:
                    builder.add_stackND(name='stackND', input_names=input_names,
                                        output_name='output', axis=axis,
                                        input_rank=len(input_shape),
                                        input_shape=input_shape,
                                        output_shape=output_shape)

                input_tensors = []
                for _ in range(nInputs):
                    input_tensors.append(np.random.rand(*input_shape))
                input = dict(zip(input_names, input_tensors))
                expected = {'output': np.stack(input_tensors, axis)}

                self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_ceil_CPU(self):
        for rank in range(1, 6):
            shape = np.random.randint(low=2, high=8, size=rank)
            input_features = [('data', datatypes.Array(*shape))]
            output_features = [('output', datatypes.Array(*shape))]

            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True)

            builder.add_ceil(name='ceil', input_name='data', output_name='output')

            x = np.random.rand(*shape)
            input = {'data': x}
            expected = {'output': np.ceil(x)}

            self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_floor_CPU(self):
        for rank in range(1, 6):
            shape = np.random.randint(low=2, high=8, size=rank)
            input_features = [('data', datatypes.Array(*shape))]
            output_features = [('output', datatypes.Array(*shape))]

            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True)

            builder.add_floor(name='floor', input_name='data', output_name='output')

            x = np.random.rand(*shape)
            input = {'data': x}
            expected = {'output': np.floor(x)}

            self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_clip_CPU(self):
        for rank in range(1, 6):
            shape = np.random.randint(low=2, high=6, size=rank)
            input_features = [('data', datatypes.Array(*shape)),
                              ('min_value', datatypes.Array(1, )),
                              ('max_value', datatypes.Array(1, ))]
            output_features = [('output', datatypes.Array(*shape))]

            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True)
            builder.add_clip(name='clip',
                             input_names=['data', 'min_value', 'max_value'],
                             output_name='output', rank=rank)

            x = np.random.rand(*shape)
            min = np.percentile(x, 25)
            max = np.percentile(x, 75)
            input = {'data': x, 'min_value': np.reshape(min, (1,)),
                     'max_value': np.reshape(max, (1,))}
            expected = {'output': np.clip(x, min, max)}

            self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_splitND_CPU(self):
        for rank in range(1, 6):
            for axis in range(-rank, rank):
                nOutputs = np.random.choice(range(2, 4))
                input_shape = np.random.randint(low=2, high=5, size=rank)
                input_shape[axis] = 0
                output_shapes = []
                output_features = []
                output_names = []
                almostEqual = random.choice([True, False])
                remainder = np.random.choice(
                    range(1, nOutputs)) if almostEqual else 0
                value = np.random.choice(range(2, 5))
                for k in range(nOutputs):
                    output_shapes.append(np.copy(input_shape))
                    output_shapes[-1][
                        axis] = value + 1 if k < remainder else value
                    input_shape[axis] += output_shapes[-1][axis]

                for i, output_dim in enumerate(output_shapes):
                    output_name = 'output_%s' % str(i)
                    output_names.append(output_name)
                    output_features.append(
                        (output_name, datatypes.Array(*output_dim)))

                input_features = [('data', datatypes.Array(*input_shape))]

                builder = neural_network.NeuralNetworkBuilder(
                    input_features, output_features,
                    disable_rank5_shape_mapping=True)

                if rank <= 5:
                    builder.add_splitND(name='splitND', input_name='data',
                                        output_names=output_names, axis=axis,
                                        num_splits=nOutputs, rank=rank)
                else:
                    builder.add_splitND(name='splitND', input_name='data',
                                        output_names=output_names, axis=axis,
                                        num_splits=nOutputs, rank=rank,
                                        input_shape=input_shape,
                                        output_shapes=output_shapes)

                x = np.random.rand(*input_shape)
                input = {'data': x}
                expected = dict(
                    zip(
                        output_names, np.array_split(x, nOutputs, axis=axis)
                        if almostEqual else
                        np.split(x, nOutputs, axis=axis)
                    )
                )  # Explicitly trying to compare against both versions of numpy split

                self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_splitND_with_split_sizes_CPU(self):
        for rank in range(1, 6):
            for axis in range(-rank, rank):
                nOutputs = np.random.choice(range(2, 4))
                input_shape = np.random.randint(low=2, high=5, size=rank)
                input_shape[axis] = 0
                output_shapes = []
                output_features = []
                output_names = []
                sections = []
                split_sizes = []
                for _ in range(nOutputs):
                    output_shapes.append(np.copy(input_shape))
                    output_shapes[-1][axis] = np.random.choice(range(2, 5))
                    input_shape[axis] += output_shapes[-1][axis]
                    sections.append(input_shape[axis])
                    split_sizes.append(output_shapes[-1][axis])

                sections.pop()
                for i, output_dim in enumerate(output_shapes):
                    output_name = 'output_%s' % str(i)
                    output_names.append(output_name)
                    output_features.append(
                        (output_name, datatypes.Array(*output_dim)))

                input_features = [('data', datatypes.Array(*input_shape))]

                builder = neural_network.NeuralNetworkBuilder(
                    input_features, output_features,
                    disable_rank5_shape_mapping=True)

                if rank <= 5:
                    builder.add_splitND(name='splitND', input_name='data',
                                        output_names=output_names, axis=axis,
                                        num_splits=len(split_sizes),
                                        split_sizes=split_sizes, rank=rank)
                else:
                    builder.add_splitND(name='splitND', input_name='data',
                                        output_names=output_names, axis=axis,
                                        num_splits=len(split_sizes),
                                        split_sizes=split_sizes, rank=rank,
                                        input_shape=input_shape,
                                        output_shapes=output_shapes)

                x = np.random.rand(*input_shape)
                input = {'data': x}
                expected = dict(
                    zip(output_names, np.split(x, sections, axis=axis)))

                self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_sliceND_CPU(self):
        for rank in range(1, 6):
            input_shape = np.array([5 for _ in range(rank)])
            objs, strides, begin_masks, end_ids, end_masks, begin_ids, output_shape = [], [], [], [], [], [], []
            for dim in range(rank):
                stride = random.choice([-3, -1, 1, 2])
                begin_mask = random.choice([True, False])
                end_mask = random.choice([True, False])
                length = 0
                while length <= 0:
                    begin_id = np.random.randint(low=-input_shape[dim],
                                                 high=input_shape[dim])
                    end_id = np.random.randint(low=-input_shape[dim],
                                               high=input_shape[dim])
                    obj = slice(None if begin_mask else begin_id,
                                None if end_mask else end_id, stride)
                    length = np.arange(input_shape[dim])[(obj,)].shape[0]

                objs.append(obj), strides.append(stride), begin_masks.append(
                    begin_mask)
                end_masks.append(end_mask), begin_ids.append(
                    begin_id), end_ids.append(end_id)
                output_shape.append(length)

            input_features = [('data', datatypes.Array(*input_shape))]
            output_features = [('output', datatypes.Array(*output_shape))]

            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True)
            if rank <= 5:
                builder.add_sliceND('SliceND', 'data', 'output', begin_ids,
                                    end_ids, begin_masks, end_masks, strides,
                                    rank)
            else:
                builder.add_sliceND('SliceND', 'data', 'output', begin_ids,
                                    end_ids, begin_masks, end_masks, strides,
                                    rank, input_shape, output_shape)

            # save the model
            model_dir = tempfile.mkdtemp()
            model_path = os.path.join(model_dir, 'test_layer.mlmodel')
            coremltools.utils.save_spec(builder.spec, model_path)

            x = np.random.rand(*input_shape)
            input = {'data': x}
            expected = {'output': x[tuple(objs)]}

            self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_powBroadcastable_CPU(self):
        aShapes = [(1, 2, 3, 1), (3, 1, 2, 1, 2), (1, 2, 1, 3), (2, 3),
                   (2, 1, 1), (2, 3, 4), (2, 4), (1,), (1,)]
        bShapes = [(2, 3, 1, 2, 1, 1, 2), (2, 2, 3, 1), (2, 1, 3, 1), (2, 3),
                   (2, 3, 4), (3, 1), (1,), (2, 3), (1,)]
        outShapes = [(2, 3, 1, 2, 2, 3, 2), (3, 2, 2, 3, 2), (2, 2, 3, 3),
                     (2, 3), (2, 3, 4), (2, 3, 4), (2, 4), (2, 3), (1,)]
        for aShape, bShape, outShape in zip(aShapes, bShapes, outShapes):
            input_shapes = [aShape, bShape]
            output_shape = outShape
            input_ranks = [len(i) for i in input_shapes]
            output_rank = len(output_shape)
            input_names = ['A', 'B']
            input_features = [('A', datatypes.Array(*input_shapes[0])),
                              ('B', datatypes.Array(*input_shapes[1]))]
            output_features = [('output', datatypes.Array(*output_shape))]

            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True)

            if output_rank <= 5:
                builder.add_powBroadcastable(name='powBroadcastable',
                                             input_names=input_names,
                                             output_name='output',
                                             input_ranks=input_ranks,
                                             output_rank=output_rank)
            else:
                builder.add_powBroadcastable(name='powBroadcastable',
                                             input_names=input_names,
                                             output_name='output',
                                             input_ranks=input_ranks,
                                             output_rank=output_rank,
                                             input_shapes=input_shapes,
                                             output_shape=output_shape)

            a = np.random.rand(*input_shapes[0])
            b = np.random.rand(*input_shapes[1])
            input = {'A': a, 'B': b}
            expected = {'output': np.power(a, b)}

            self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_tile_CPU(self):
        for rank in range(1, 6):
            input_shape = np.random.randint(low=2, high=5, size=rank)
            reps = np.random.randint(low=1, high=4, size=rank)
            output_shape = [i * j for i, j in zip(reps, input_shape)]

            input_features = [('data', datatypes.Array(*input_shape))]
            output_features = [('output', datatypes.Array(*output_shape))]

            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True
            )
            if rank <= 5:
                builder.add_tile('Tile', 'data', 'output', reps, rank)
            else:
                builder.add_tile('Tile', 'data', 'output', reps, rank,
                                 input_shape, output_shape)

            # save the model
            model_dir = tempfile.mkdtemp()
            model_path = os.path.join(model_dir, 'test_layer.mlmodel')
            coremltools.utils.save_spec(builder.spec, model_path)

            x = np.random.rand(*input_shape)
            input = {'data': x}
            expected = {'output': np.tile(x, reps)}

            self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_sliding_windows_CPU(self):

        def numpy_sliding(a, axis, size, step):
            N = (a.shape[axis] - size) // step + 1
            shape = list(a.shape)
            shape[axis] = N
            if axis < 0:
                axis += len(shape)
            shape.insert(axis + 1, size)
            strides = list(a.strides)
            effstride = strides[axis] * step
            strides.insert(axis, effstride)
            return np.lib.stride_tricks.as_strided(a, shape, strides)

        for rank in range(1, 6):
            for axis in range(-rank, rank):
                input_shape = np.random.randint(low=2, high=5, size=rank)
                output_shape = list(input_shape)
                window_size = np.random.randint(low=1, high=input_shape[axis])

                length = 0
                while length <= 0:
                    step = np.random.randint(low=1, high=input_shape[axis])
                    length = (input_shape[axis] - window_size) // step + 1

                output_shape[axis] = length

                posAxis = axis if axis >= 0 else axis + rank
                output_shape.insert(posAxis + 1, window_size)
                output_shape = np.array(output_shape)
                output_rank = rank + 1

                input_features = [('data', datatypes.Array(*input_shape))]

                if output_rank <= 5:
                    output_features = [('output', None)]
                    builder = neural_network.NeuralNetworkBuilder(
                        input_features, output_features,
                        disable_rank5_shape_mapping=True)
                    builder.add_sliding_windows('sw', 'data', 'output', axis,
                                                window_size, step, rank)
                else:
                    output_features = [
                        ('output', datatypes.Array(*output_shape))]
                    builder = neural_network.NeuralNetworkBuilder(
                        input_features, output_features,
                        disable_rank5_shape_mapping=True)
                    builder.add_sliding_windows('sw', 'data', 'output', axis,
                                                window_size, step, rank,
                                                input_shape, output_shape)

                # save the model
                model_dir = tempfile.mkdtemp()
                model_path = os.path.join(model_dir, 'test_layer.mlmodel')
                coremltools.utils.save_spec(builder.spec, model_path)

                x = np.random.rand(*input_shape)
                input = {'data': x}
                expected = {'output': numpy_sliding(x, axis, window_size, step)}

                self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_range_CPU(self):
        params = [(-10.4, 23, 12.2), (0, 1000, 1), (50.5, 90.5, 1.5), (5, 8, 2),
                  (5, 8, 98), (5, 8, 1.5), (10, 5, -0.6), (24, -65, -2)]

        # mode 0: all parameters as input: in that case the op is dynamic
        # mode 1: static case when start, end, step are all parameters of the layer
        for mode in [0, 1]:
            for param in params:
                start, end, step = param
                if mode == 0:
                    input_features = [('start', datatypes.Array(1)),
                                      ('end', datatypes.Array(1)),
                                      ('step', datatypes.Array(1))]
                else:
                    input_features = [
                        ('multiplicative_input', datatypes.Array(1))]

                output_features = [('output', None)]
                builder = neural_network.NeuralNetworkBuilder(
                    input_features, output_features,
                    disable_rank5_shape_mapping=True)

                if mode == 0:
                    builder.add_range('Range', 'output',
                                      input_names=['start', 'end', 'step'])
                else:
                    builder.add_range('Range', 'output_range', input_names=None,
                                      params=[start, end, step])
                    builder.add_multiplyBroadcastable(
                        name='multiplyBroadcastable',
                        input_names=['multiplicative_input', 'output_range'],
                        output_name='output',
                        input_ranks=[1, 1], output_rank=1)

                # save the model
                model_dir = tempfile.mkdtemp()
                model_path = os.path.join(model_dir, 'test_layer.mlmodel')
                coremltools.utils.save_spec(builder.spec, model_path)

                input = dict()
                if mode == 0:
                    input['start'] = start * np.ones((1,), dtype=np.float64)
                    input['end'] = end * np.ones((1,), dtype=np.float64)
                    input['step'] = step * np.ones((1,), dtype=np.float64)
                else:
                    input['multiplicative_input'] = np.ones((1,),
                                                            dtype=np.float64)
                expected = {'output': np.arange(start, end, step)}

                self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_linear_activation_different_ranks_CPU(self):
        for input_dim in [(10, 15), (10, 15, 2, 3),
                          (10, 2, 4, 15, 1, 4), (6,)]:
            input_features = [('data', datatypes.Array(*input_dim))]
            output_features = [('output', datatypes.Array(*input_dim))]

            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True)

            builder.add_activation(name='activation',
                                   non_linearity='LINEAR',
                                   input_name='data',
                                   output_name='output', params=[34.0, 67.0])

            x = np.random.rand(*input_dim)
            input = {'data': x}
            expected = {'output': 34.0 * x + 67.0}

            self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_rank_preserving_reshape(self):
        input_shapes = [(20, 10), (20, 10, 5), (10, 3, 5)]
        target_shapes = [(5, -1), (0, 2, 25), (25, 0, -1)]
        output_shapes = [(5, 40), (20, 2, 25), (25, 3, 2)]

        for i in range(len(input_shapes)):
            input_features = [('data', datatypes.Array(*input_shapes[i]))]
            output_features = [('output', None)]
            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True)

            builder.add_rankPreservingReshape(
                name='reshape_layer', input_name='data', output_name='output',
                target_shape=target_shapes[i])

            x = np.random.rand(*input_shapes[i])
            input = {'data': x}
            expected = {'output': np.reshape(x, output_shapes[i])}

            self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_expand_dims(self):
        input_shapes = [(10, 5), (10, 5), (10, 5), (10, 5)]
        axes = [(0, 1), (0, 2), (2, 0), (-2, -1)]
        output_shapes = [(1, 1, 10, 5), (1, 10, 1, 5), (1, 10, 1, 5),
                         (10, 5, 1, 1)]

        for i in range(len(input_shapes)):
            input_features = [('data', datatypes.Array(*input_shapes[i]))]
            output_features = [('output', None)]
            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True)

            builder.add_expandDims(
                name='ed_layer', input_name='data', output_name='output',
                input_rank=len(input_shapes[i]), axes=axes[i]
            )

            x = np.random.rand(*input_shapes[i])
            input = {'data': x}
            expected = {'output': np.reshape(x, output_shapes[i])}

            self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_squeeze(self):
        input_shapes = [(1, 1, 10, 5), (1, 10, 1, 5), (10, 5, 1, 1),
                        (10, 5, 1, 1)]
        axes = [(0, 1), (0, 2), (-2, -1), (-1, -2)]
        output_shapes = [(10, 5), (10, 5), (10, 5), (10, 5)]

        for i in range(len(input_shapes)):
            input_features = [('data', datatypes.Array(*input_shapes[i]))]
            output_features = [('output', None)]

            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True
            )
            builder.add_squeeze(name='squeeze_layer', input_name='data',
                                output_name='output',
                                input_rank=len(input_shapes[i]), axes=axes[i])

            x = np.random.rand(*input_shapes[i])
            input = {'data': x}
            expected = {'output': np.reshape(x, output_shapes[i])}

            self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_get_shape(self):
        dims = [1, 2, 3, 4, 5]
        for rank in range(1, len(dims) + 1):
            input_shape = dims[:rank]
            input_features = [('data', datatypes.Array(*input_shape))]
            output_features = [('output', None)]
            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True
            )
            builder.add_get_shape(name='get_shape_layer', input_name='data',
                                  output_name='output')

            feed = {'data': np.random.rand(*input_shape)}
            expected = {'output': np.array(input_shape)}

            self._test_model(builder.spec, feed, expected, useCPUOnly=True)

    def test_load_constant_nd(self):
        dims = [2, 3, 4, 5, 6]
        for rank in range(1, len(dims) + 1):
            input_shape = dims[:rank]
            input_features = [('data', datatypes.Array(*input_shape))]
            output_features = [('output', None)]
            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features,
                disable_rank5_shape_mapping=True
            )
            builder.add_loadConstantND('load_const_nd_layer', 'tmp',
                                       constant_value=np.ones(input_shape),
                                       shape=input_shape)
            builder.add_elementwise('add_layer', ['data', 'tmp'], 'output',
                                    mode='ADD')
            feed = {'data': np.random.rand(*input_shape)}
            expected = {'output': feed['data'] + 1}

            self._test_model(builder.spec, feed, expected, useCPUOnly=True)

    def test_simple_array_alloc_scatter(self):
        alloc_shape = [2, 3, 4]
        value_shape = [1, 3, 4]
        input_features = [('alloc_shape', datatypes.Array(len(alloc_shape))),
                          ('value', datatypes.Array(*value_shape)),
                          ('index', datatypes.Array(1))]
        output_features = [('output', None)]

        builder = neural_network.NeuralNetworkBuilder(
            input_features, output_features, disable_rank5_shape_mapping=True)
        builder.add_alloc(name='alloc_layer', input_name='alloc_shape',
                          output_name='array')
        # CoreML input order: container (array), indices, slices (value)
        builder.add_scatter(name='scatter_layer',
                            input_names=['array', 'index', 'value'],
                            output_name='output')

        value = np.random.rand(*value_shape).astype('float')
        feed = {'alloc_shape': np.array(alloc_shape, dtype='float'),
                'value': value,
                'index': np.array([1], dtype='float')}

        ref = np.zeros(alloc_shape)
        ref[1, :, :] = value
        expected = {'output': ref}

        self._test_model(builder.spec, feed, expected, useCPUOnly=True)

    def test_erf_activation(self):
        input_features = [('data', datatypes.Array(10, 45))]
        output_features = [('output', datatypes.Array(10, 45))]

        builder = neural_network.NeuralNetworkBuilder(
            input_features, output_features, disable_rank5_shape_mapping=True)
        builder.add_erfActivation(name='erf', input_name='data',
                                  output_name='output')
        x = np.random.rand(10, 45)
        input = {'data': x}
        expected = {
            'output': np.asarray([math.erf(i) for i in
                                  x.flatten().tolist()]).reshape(10, 45)
        }

        self._test_model(builder.spec, input, expected, useCPUOnly=True)

    def test_gelu_activation(self):
        input_features = [('data', datatypes.Array(10, 45))]
        output_features = [('output', datatypes.Array(10, 45))]

        builder = neural_network.NeuralNetworkBuilder(
            input_features, output_features, disable_rank5_shape_mapping=True)
        builder.add_geluActivation(name='erf', input_name='data',
                                   output_name='output')

        x = np.random.rand(10, 45)
        input = {'data': x}
        y = np.asarray([0.5 * i * (1.0 + math.erf(i / math.sqrt(2)))
                        for i in x.flatten().tolist()]).reshape(10, 45)
        expected = {'output': y}

        self._test_model(builder.spec, input, expected, useCPUOnly=True)


def get_size_after_stride(X, params):
    start = params["start"]
    end = params["end"]
    stride = params["stride"]
    if params["axis"] == 'width': axis = 2
    if params["axis"] == 'height': axis = 1
    if params["axis"] == 'channel': axis = 0
    N = X.shape[axis]
    if end < 0:
        end = end + N
    end = min(end, N)
    if start > N - 1:
        L = 0
    else:
        L = np.floor((end - 1 - start) / stride) + 1
        if L < 0:
            L = 0
    return L


def get_numpy_predictions_slice(X, params):
    start = params["start"]
    end = params["end"]
    stride = params["stride"]
    if params["axis"] == 'width':
        return X[:, :, start:end:stride]
    if params["axis"] == 'height':
        return X[:, start:end:stride, :]
    if params["axis"] == 'channel':
        return X[start:end:stride, :, :]


def get_coreml_predictions_slice(X, params):
    coreml_preds = []
    eval = True
    try:
        input_dim = X.shape
        output_dim = (1, 1,
                      1)  # some random dimensions here: we are going to remove this information later
        input_features = [('data', datatypes.Array(*input_dim))]
        output_features = [('output', datatypes.Array(*output_dim))]
        builder = neural_network.NeuralNetworkBuilder(input_features,
                                                      output_features)
        builder.add_slice('slice', 'data', 'output',
                          start_index=params["start"],
                          end_index=params["end"], stride=params["stride"],
                          axis=params["axis"])
        # Remove output shape by deleting and adding an output
        del builder.spec.description.output[-1]
        output = builder.spec.description.output.add()
        output.name = 'output'
        output.type.multiArrayType.dataType = coremltools.proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.Value(
            'DOUBLE')
        # save the model
        model_dir = tempfile.mkdtemp()
        model_path = os.path.join(model_dir, 'test_layer.mlmodel')
        coremltools.utils.save_spec(builder.spec, model_path)
        # preprare input and get predictions
        coreml_model = coremltools.models.MLModel(model_path)
        coreml_input = {'data': X}
        if macos_version() >= (10, 13):
            coreml_preds = coreml_model.predict(coreml_input)['output']
        else:
            coreml_preds = None
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
    except RuntimeError as e:
        print(e)
        eval = False

    return coreml_preds, eval


def get_numpy_predictions_reduce(X, params):
    if params["axis"] == 'CHW': axis = (0, 1, 2)
    if params["axis"] == 'HW': axis = (1, 2)
    if params["axis"] == 'C': axis = 0
    if params["axis"] == 'H': axis = 1
    if params["axis"] == 'W': axis = 2

    if params["mode"] == 'sum': return np.sum(X, axis)
    if params["mode"] == 'avg': return np.mean(X, axis)
    if params["mode"] == 'prod': return np.prod(X, axis)
    if params["mode"] == 'logsum': return np.sum(np.log(X + 1e-6), axis)
    if params["mode"] == 'sumsquare': return np.sum(X ** 2, axis)
    if params["mode"] == 'L2': return np.sqrt(np.sum(X ** 2, axis))
    if params["mode"] == 'L1': return np.sum(np.abs(X), axis)
    if params["mode"] == 'max': return np.amax(X, axis)
    if params["mode"] == 'min': return np.amin(X, axis)
    if params["mode"] == 'argmax': return np.argmax(X, axis)


def get_coreml_predictions_reduce(X, params):
    coreml_preds = []
    eval = True
    try:
        input_dim = X.shape
        output_dim = (1, 1,
                      1)  # some random dimensions here: we are going to remove this information later
        input_features = [('data', datatypes.Array(*input_dim))]
        output_features = [('output', datatypes.Array(*output_dim))]
        builder = neural_network.NeuralNetworkBuilder(input_features,
                                                      output_features)
        builder.add_reduce('reduce', 'data', 'output', axis=params["axis"],
                           mode=params["mode"])
        # Remove output shape by deleting and adding an output
        del builder.spec.description.output[-1]
        output = builder.spec.description.output.add()
        output.name = 'output'
        output.type.multiArrayType.dataType = coremltools.proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.Value(
            'DOUBLE')
        # save the model
        model_dir = tempfile.mkdtemp()
        model_path = os.path.join(model_dir, 'test_layer.mlmodel')
        coremltools.utils.save_spec(builder.spec, model_path)
        # preprare input and get predictions
        coreml_model = coremltools.models.MLModel(model_path)
        coreml_input = {'data': X}
        if macos_version() >= (10, 13):
            coreml_preds = coreml_model.predict(coreml_input)['output']
        else:
            coreml_preds = None
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
    except RuntimeError as e:
        print(e)
        eval = False

    return coreml_preds, eval


class StressTest(CorrectnessTest):

    def test_slice_layer(self):
        params_dict = dict(
            input_shape=[[30, 100, 8], [80, 50, 5], [4, 12, 5], [56, 8, 14]],
            axis=['channel', 'height', 'width'],
            start=[0, 1, 2, 5],
            end=[5, 100, 56, -1, -2, -4],
            stride=[1, 2, 3]
        )
        params = list(itertools.product(*params_dict.values()))
        all_candidates = [dict(zip(params_dict.keys(), x)) for x in params]
        valid_params = []
        for pr in all_candidates:
            X = np.random.rand(*pr["input_shape"])
            if get_size_after_stride(X, pr):
                valid_params.append(pr)
        print("Total params to be tested: ", len(valid_params),
              "out of canditates: ", len(all_candidates))
        '''
        Test
        '''
        failed_tests_compile = []
        failed_tests_shape = []
        failed_tests_numerical = []
        for i in range(len(valid_params)):
            params = valid_params[i]
            X = np.random.rand(*params["input_shape"])
            np_preds = get_numpy_predictions_slice(X, params)
            coreml_preds, eval = get_coreml_predictions_slice(X, params)
            if eval is False:
                failed_tests_compile.append(params)
            elif coreml_preds is not None:
                if not self._compare_shapes(np_preds, coreml_preds):
                    failed_tests_shape.append(params)
                elif not self._compare_predictions(np_preds, coreml_preds):
                    failed_tests_numerical.append(params)

        self.assertEqual(failed_tests_compile, [])
        self.assertEqual(failed_tests_shape, [])
        self.assertEqual(failed_tests_numerical, [])

    def test_reduce_layer(self):
        params_dict = dict(
            input_shape=[[3, 10, 8], [8, 5, 5], [4, 12, 10], [7, 1, 14]],
            mode=['sum', 'avg', 'prod', 'sumsquare', 'L1', 'L2', 'max',
                  'min', 'argmax'],
            axis=['CHW', 'HW', 'C', 'H', 'W'],
        )
        params = list(itertools.product(*params_dict.values()))
        all_candidates = [dict(zip(params_dict.keys(), x)) for x in params]
        valid_params = []
        for pr in all_candidates:
            if pr["mode"] == 'argmax':
                if pr["axis"] == 'CHW' or pr["axis"] == 'HW':
                    continue
            valid_params.append(pr)
        print("Total params to be tested: ", len(valid_params),
              "out of canditates: ", len(all_candidates))
        '''
        Test
        '''
        failed_tests_compile = []
        failed_tests_shape = []
        failed_tests_numerical = []
        for i in range(len(valid_params)):
            params = valid_params[i]
            X = np.random.rand(*params["input_shape"])
            np_preds = get_numpy_predictions_reduce(X, params)
            coreml_preds, eval = get_coreml_predictions_reduce(X, params)
            if eval is False:
                failed_tests_compile.append(params)
            elif coreml_preds is not None:
                if not self._compare_shapes(np_preds, coreml_preds):
                    failed_tests_shape.append(params)
                elif not self._compare_predictions(np_preds, coreml_preds):
                    failed_tests_numerical.append(params)

        self.assertEqual(failed_tests_compile, [])
        self.assertEqual(failed_tests_shape, [])
        self.assertEqual(failed_tests_numerical, [])


if __name__ == '__main__':
    unittest.main()
    #suite = unittest.TestSuite()
    #suite.addTest(NewLayersSimpleTest("test_load_constant_nd"))
    #unittest.TextTestRunner().run(suite)
