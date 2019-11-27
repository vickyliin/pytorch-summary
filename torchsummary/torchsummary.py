import operator as op
import itertools as it

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np


def _sizeof(tensor):
    storage = tensor.storage()
    return storage.size() * storage.element_size()

def _get_hook(summary, name):

    def summary_hook(module, input, output):
        class_name = module.__class__.__name__

        layer = {}
        layer["name"] = "%s (%s)" % (name, class_name)
        if isinstance(output, (list, tuple)):
            layer["output_shape"] = [tuple(o.size()) for o in output]
            layer["total_output_size"] = sum(map(_sizeof, output))
        else:
            layer["output_shape"] = [tuple(output.size())]
            layer["total_output_size"] = _sizeof(output)

        layer["total_param_size"] = sum(map(_sizeof, module.parameters()))
        layer["nb_params"] = sum(map(torch.Tensor.numel, module.parameters()))
        trainable_params = filter(op.attrgetter('requires_grad'), module.parameters())
        layer["nb_trainable_params"] = sum(map(torch.Tensor.numel, trainable_params))

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
    model(*args, **kwargs)

    layer_width = max(max(map(len, map(op.itemgetter('name'), summary))), 20)
    output_shape_width = 40
    line_width = layer_width + output_shape_width + 15 + 5
    line = "-" * line_width
    dline = "=" * line_width
    fmt = "{:<%d}  {:^%d} {:>15}" % (layer_width, output_shape_width)
    print(line)
    line_new = fmt.format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print(dline)
    layer_names = set()
    total_params = 0
    total_param_size = 0
    total_output_size = 0
    trainable_params = 0
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

        output_shape = ' '.join(map(str, layer["output_shape"]))
        print(fmt.format(name, output_shape, nb_params))

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
