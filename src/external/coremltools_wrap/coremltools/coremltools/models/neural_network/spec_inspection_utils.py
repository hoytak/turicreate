from ...proto import NeuralNetwork_pb2 as _NeuralNetwork_pb2

def _get_weight_param_summary(wp):
    """Get a summary of _NeuralNetwork_pb2.WeightParams
    Args:
    wp : _NeuralNetwork_pb2.WeightParams - the _NeuralNetwork_pb2.WeightParams message to display
    Returns:
    a str summary for wp
    """
    summary_str = ''
    if wp.HasField('quantization'):
        nbits = wp.quantization.numberOfBits
        quant_type = 'linearly' if wp.quantization.HasField('linearQuantization') else 'lookup-table'
        summary_str += '{}-bit {} quantized'.format(nbits, quant_type)

    if len(wp.floatValue) > 0:
        summary_str += '({} floatValues)'.format(len(wp.floatValue))
    if len(wp.float16Value) > 0:
        summary_str += '({} bytes float16Values)'.format(len(wp.float16Value))
    if len(wp.rawValue) > 0:
        summary_str += '({} bytes rawValues)'.format(len(wp.rawValue))

    return summary_str


def _get_lstm_weight_param_summary(lstm_wp):
    weight_name_list = [
        'W_i','W_f','W_z','W_o','H_i','H_f','H_z','H_o','b_i','b_f','b_z','b_o','p_i','p_f','p_o'
    ]
    wp_summary_list = [
        _get_weight_param_summary(lstm_wp.inputGateWeightMatrix),
        _get_weight_param_summary(lstm_wp.forgetGateWeightMatrix),
        _get_weight_param_summary(lstm_wp.blockInputWeightMatrix),
        _get_weight_param_summary(lstm_wp.outputGateWeightMatrix),
        _get_weight_param_summary(lstm_wp.inputGateRecursionMatrix),
        _get_weight_param_summary(lstm_wp.forgetGateRecursionMatrix),
        _get_weight_param_summary(lstm_wp.blockInputRecursionMatrix),
        _get_weight_param_summary(lstm_wp.outputGateRecursionMatrix),
        _get_weight_param_summary(lstm_wp.inputGateBiasVector),
        _get_weight_param_summary(lstm_wp.forgetGateBiasVector),
        _get_weight_param_summary(lstm_wp.blockInputBiasVector),
        _get_weight_param_summary(lstm_wp.outputGateBiasVector),
        _get_weight_param_summary(lstm_wp.inputGatePeepholeVector),
        _get_weight_param_summary(lstm_wp.forgetGatePeepholeVector),
        _get_weight_param_summary(lstm_wp.outputGatePeepholeVector),
    ]
    lstm_wp_summary_list = []
    for idx, summary in enumerate(wp_summary_list):
        if len(summary) > 0:
            lstm_wp_summary_list.append(weight_name_list[idx] + ', ' + summary)

    return ('\n' + ' '*8).join(lstm_wp_summary_list)


def _get_feature_description_summary(feature):
    if feature.type.HasField('multiArrayType'):
        return str(list(feature.type.multiArrayType.shape))
    else:
        return ('({})'.format(str(feature.type))).replace('\n', '')


def summarize_network_layer_info(layer):
    """
    Args:
    layer - an MLModel NeuralNetwork Layer protobuf message
    Returns:
    layer_type : str - type of layer
    layer_name : str - name of the layer
    layer_inputs : list[str] - a list of strings representing input blobs of the layer
    layer_outputs : list[str] - a list of strings representing output blobs of the layer
    layer_field_content : list[(str, str)] - a list of two-tuple of (parameter_name, content)
    """
    layer_type_str = layer.WhichOneof('layer')

    layer_name = layer.name
    layer_inputs = list(layer.input)
    layer_outputs = list(layer.output)

    typed_layer = getattr(layer, layer_type_str)
    layer_field_names = [l.name for l in typed_layer.DESCRIPTOR.fields]
    layer_field_content = []

    for name in layer_field_names:
        field = getattr(typed_layer,name)
        summary_str = ''
        if type(field) == _NeuralNetwork_pb2.LSTMWeightParams:
            summary_str = _get_lstm_weight_param_summary(field)
        elif type(field) == _NeuralNetwork_pb2.WeightParams:
            summary_str = _get_weight_param_summary(field)
        else:
            field_str = str(field)
            if len(field_str) > 0:
                summary_str = field_str.replace('\n', ' ')
        if len(summary_str) > 0:
            layer_field_content.append([name, summary_str])

    return layer_type_str, layer_name, layer_inputs, layer_outputs, layer_field_content


def summarize_neural_network_spec(mlmodel_spec):
    """ Summarize network into the following structure.
    Args:
    mlmodel_spec : mlmodel spec
    Returns:
    inputs : list[(str, str)] - a list of two tuple (name, descriptor) for each input blob.
    outputs : list[(str, str)] - a list of two tuple (name, descriptor) for each output blob
    layers : list[(str, list[str], list[str], list[(str, str)])] - a list of layers represented by
        layer name, input blobs, output blobs, a list of (parameter name, content)
    """
    inputs = [(blob.name, _get_feature_description_summary(blob)) for blob in mlmodel_spec.description.input]
    outputs = [(blob.name, _get_feature_description_summary(blob)) for blob in mlmodel_spec.description.output]
    nn = None

    if mlmodel_spec.HasField('neuralNetwork'):
        nn = mlmodel_spec.neuralNetwork
    elif mlmodel_spec.HasField('neuralNetworkClassifier'):
        nn = mlmodel_spec.neuralNetworkClassifier
    elif mlmodel_spec.HasField('neuralNetworkRegressor'):
        nn = mlmodel_spec.neuralNetworkRegressor

    layers = [summarize_network_layer_info(layer) for layer in nn.layers] if nn != None else None
    return (inputs, outputs, layers)
