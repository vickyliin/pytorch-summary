import operator as op
import itertools as it

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np


def _sizeof(tensor):
    storage = tensor.storage()
    return storage.size() * storage.element_size()

def _total_numel(tensors):
    return sum(map(torch.Tensor.numel, tensors))

def _shape_and_sizeof(tensors):
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    shape = [tuple(o.size()) for o in tensors]
    size = sum(map(_sizeof, tensors))
    return shape, size

def _get_hook(summary, name):

    def summary_hook(module, input, output):
        layer = {}
        layer["name"] = name
        layer["type"] = type(module).__name__
        layer["input_shape"], layer["total_input_size"] = _shape_and_sizeof(input)
        layer["output_shape"], layer["total_output_size"] = _shape_and_sizeof(output)
        layer["total_param_size"] = sum(map(_sizeof, module.parameters()))
        layer["nb_params"] = _total_numel(module.parameters())

        trainable_params = filter(op.attrgetter('requires_grad'), module.parameters())
        layer["nb_trainable_params"] = _total_numel(trainable_params)

        summary.append(layer)

    return summary_hook

def summary(model, *args, **kwargs):
    # create properties
    summary = []

    # register hook
    for name, module in model.named_children():
        hook = _get_hook(summary, name)
        module.register_forward_hook(hook)

    # make a forward pass
    with torch.no_grad():
        model(*args, **kwargs)

    layer_name_width = max(max(map(len, map(op.itemgetter('name'), summary))), 20)
    layer_type_width = max(max(map(len, map(op.itemgetter('type'), summary))), 20)
    shape_width, param_width = 40, 15
    widths = layer_name_width, layer_type_width, shape_width, shape_width, param_width
    fmt = "{:<%d}  {:<%d}  {:<%d} {:<%d} {:>%d}" % widths

    line_width = sum(widths) + len(widths) + 1
    line = "-" * line_width
    dline = "=" * line_width
    print(line)
    line_new = fmt.format("Layer Name", "Layer Type", "Input Shape", "Output Shape", "Param #")
    print(line_new)
    print(dline)
    layer_names = set()
    total_params = 0
    total_param_size = 0
    total_output_size = 0
    trainable_params = 0

    def format_shapes(shapes):
        return ' '.join(map(str, shapes))

    for layer in summary:
        name = layer["name"]

        nb_params = ""
        if name not in layer_names:
            total_params += layer["nb_params"]
            total_param_size += layer["total_param_size"]
            trainable_params += layer["nb_trainable_params"]
            layer_names.add(name)
            nb_params = "{0:,}".format(layer["nb_params"])

        total_output_size += layer["total_output_size"]

        output_shape = format_shapes(layer["output_shape"])
        input_shape = format_shapes(layer["input_shape"])
        print(fmt.format(name, layer["type"], input_shape, output_shape, nb_params))

    # assume 4 bytes/number (float on cuda).
    input_tensors = filter(lambda i: isinstance(i, torch.Tensor),
                           it.chain(args, kwargs.values()))
    total_input_size = sum(map(_sizeof, input_tensors)) / (1024 ** 2.)
    total_output_size = 2. * total_output_size / (1024 ** 2.)  # x2 for gradients
    total_param_size /= 1024 ** 2.
    total_size = total_param_size + total_output_size + total_input_size

    print(dline)
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print(line)
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_param_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print(line)
