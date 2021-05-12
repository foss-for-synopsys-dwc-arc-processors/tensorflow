from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import re
from pathlib import Path

try:
    from tensorflow.lite.python.util import convert_bytes_to_c_source, _convert_model_from_object_to_bytearray, \
        _convert_model_from_bytearray_to_object
except:
    print('Install TensorFlow package first to use MLI adaptation tool.')
    sys.exit(1)


# Model conversion functions
def convert_c_source_to_bytes(input_cc_file):
    pattern = re.compile(r'\W*(0x[0-9a-fA-F,x ]+).*')
    model_bytearray = bytearray()

    with open(input_cc_file) as file_handle:
        for line in file_handle:
            values_match = pattern.match(line)

            if values_match is None:
                continue

            list_text = values_match.group(1)
            values_text = filter(None, list_text.split(','))

            values = [int(x, base=16) for x in values_text]
            model_bytearray.extend(values)

    return bytes(model_bytearray)

def convert_c_source_to_object(input_cc_file):
    model_bytes = convert_c_source_to_bytes(input_cc_file)
    return _convert_model_from_bytearray_to_object(model_bytes)

def read_model(input_tflite_file):
    model_bytearray = bytearray(input_tflite_file.read())
    return _convert_model_from_bytearray_to_object(model_bytearray)

def write_model(model_object, output_tflite_file):
    model_bytearray = _convert_model_from_object_to_bytearray(model_object)
    if output_tflite_file.endswith('.cc'):
        mode = 'w'
        converted_model = convert_bytes_to_c_source(data=model_bytearray,
                                                    array_name='g_' + str(Path(output_tflite_file).stem),
                                                    include_path=str(Path(output_tflite_file).with_suffix('.h')),
                                                    use_tensorflow_license=True)[0]
    elif output_tflite_file.endswith('.tflite'):
        mode = 'wb'
        converted_model = model_bytearray
    else:
        raise ValueError('File format not supported')

    with open(output_tflite_file, mode) as output_file:
        output_file.write(converted_model)

# Helper functions
def transpose_weights(tensor, buffer, transpose_shape):
    buffer.data \
        .reshape(tensor.shape) \
        .transpose(transpose_shape) \
        .flatten()

    tensor.shape = tensor.shape[transpose_shape]

    tensor.quantization.quantizedDimension = \
        transpose_shape.index(tensor.quantization.quantizedDimension)

# Layer-specific adaptation functions
def adapt_conv(operator, tensors, buffers):
    transpose_weights(tensors[operator.inputs[1]], buffers[tensors[operator.inputs[1]].buffer], [1, 2, 3, 0])

def adapt_dw(operator, tensors, buffers):
    return

def adapt_fc(operator, tensors, buffers):
    transpose_weights(tensors[operator.inputs[1]], buffers[tensors[operator.inputs[1]].buffer], [1, 0])

# Op_codes that require additional adaptation for MLI
adapt_op_codes = {
    3: adapt_conv,  # CONV_2D
    4: adapt_dw,    # DEPTHWISE_CONV_2D
    9: adapt_fc     # FULLY_CONNECTED
    }

def adapt_model_to_mli(model):
    op_codes = [op_code.builtinCode if op_code.builtinCode != 0 else op_code.deprecatedBuiltinCode
                for op_code in model.operatorCodes]
    for subgraph in model.subgraphs:
        for operator in subgraph.operators:
            try:
                adapt_op_codes[op_codes[operator.opcodeIndex]](operator, subgraph.tensors, model.buffers)
            except KeyError:
                continue

def main(argv):
    try:
        if len(sys.argv) == 3:
            tflite_input = argv[1]
            tflite_output = argv[2]
        elif len(sys.argv) == 2:
            tflite_input = argv[1]
            tflite_output = argv[1]
    except IndexError:
        print("Usage: %s <input cc/tflite> <output cc/tflite>" % (argv[0]))
    else:
        if tflite_input.endswith('.cc'):
            model = convert_c_source_to_object(tflite_input)
        elif tflite_input.endswith('.tflite'):
            model = read_model(tflite_input)
        else:
            raise ValueError('File format not supported')

        adapt_model_to_mli(model)
        write_model(model, tflite_output)

        print('Model was adapted to be used with MLI.')

if __name__ == "__main__":
    main(sys.argv)