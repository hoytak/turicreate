# Copyright (c) 2018, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from .spec_inspection_utils import *

def print_network_spec(mlmodel_spec, interface_only=False):
    """ Print the network information summary.
    Args:
    mlmodel_spec : the mlmodel spec
    interface_only : Shows only the input and output of the network
    """
    inputs, outputs, layers_info = summarize_neural_network_spec(mlmodel_spec)

    print('Inputs:')
    for i in inputs:
        name, description = i
        print('  {} {}'.format(name, description))
    
    print('Outputs:')
    for o in outputs:
        name, description = o
        print('  {} {}'.format(name, description))

    if layers_info is None:
        print('\n(This MLModel is not a neural network model or does not contain any layers)')

    if layers_info and not interface_only:
        print('\nLayers:')
        for idx, l in enumerate(layers_info):
            layer_type, name, in_blobs, out_blobs, params_info = l
            print('[{}] ({}) {}'.format(idx, layer_type, name))
            print('  Input blobs: {}'.format(in_blobs))
            print('  Output blobs: {}'.format(out_blobs))
            if len(params_info) > 0:
                print('  Parameters: ')
            for param in params_info:
                print('    {} = {}'.format(param[0], param[1]))

    print('\n')

