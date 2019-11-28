import itertools as it
import operator as op
import functools
from collections import namedtuple

import torch
from torch import nn


class Summary(dict):
    nb_params = namedtuple("nb_params", ("trainable", "non_trainable"))
    nb_bytes = namedtuple("nb_bytes", ("params", "forward_backward_pass", "input"))
    toM = (1. / 1024 ** 2).__mul__

    @classmethod
    def create(cls, layers, nb_params, nb_bytes):
        return cls({
            "layers": layers,
            "nb_params": cls.nb_params(*nb_params),
            "nb_bytes": cls.nb_bytes(*nb_bytes)
        })

    def describe(self):
        trainable_params, non_trainable_params = self["nb_params"]
        param_size, output_size, input_size = map(self.toM, self["nb_bytes"])

        params = trainable_params + non_trainable_params
        size = param_size + output_size + input_size
        print("Total params: {0:,}".format(params))
        print("Trainable params: {0:,}".format(trainable_params))
        print("Non-trainable params: {0:,}".format(non_trainable_params))
        print()
        print("Input size (MB): %0.2f" % input_size)
        print("Forward/backward pass size (MB): %0.2f" % output_size)
        print("Params size (MB): %0.2f" % param_size)
        print("Total Size (MB): %0.2f" % size)


def _sizeof(tensor):
    storage = tensor.storage()
    return storage.size() * storage.element_size()

def _total_numel(tensors):
    return sum(map(torch.Tensor.numel, tensors))

def _shape_and_sizeof(tensors):
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    shape = [tuple(o.size()) for o in tensors]
    yield shape
    yield sum(map(_sizeof, tensors))

def _get_hook(layers, name):

    def summary_hook(module, input, output):
        layer = {}
        layer["name"] = name
        layer["type"] = type(module).__name__
        layer["input_shape"] = next(_shape_and_sizeof(input))
        layer["output_shape"], layer["total_output_size"] = _shape_and_sizeof(output)
        layer["total_param_size"] = sum(map(_sizeof, module.parameters()))
        layer["nb_params"] = _total_numel(module.parameters())

        trainable_params = filter(op.attrgetter("requires_grad"), module.parameters())
        layer["nb_trainable_params"] = _total_numel(trainable_params)

        layers.append(layer)

    return summary_hook

def summary(model, *args, **kwargs):
    # create properties
    layers = []
    hooks = []

    # register hook
    for name, module in model.named_children():
        hook = _get_hook(layers, name)
        hooks.append(module.register_forward_hook(hook))

    # make a forward pass
    with torch.no_grad():
        model(*args, **kwargs)

    # clean up hooks
    for hook in hooks:
        hook.remove()

    layer_names = set()
    total_params = 0
    total_param_size = 0
    total_output_size = 0
    trainable_params = 0

    def format_shapes(shapes):
        return " ".join(map(str, shapes))

    for layer in layers:
        name = layer["name"]

        nb_params = ""
        total_output_size += layer.pop("total_output_size")
        param_size = layer.pop("total_param_size")

        if name not in layer_names:
            total_param_size += param_size
            total_params += layer["nb_params"]
            trainable_params += layer["nb_trainable_params"]
            layer_names.add(name)
            nb_params = "{0:,}".format(layer["nb_params"])

        output_shape = format_shapes(layer["output_shape"])
        input_shape = format_shapes(layer["input_shape"])

    input_tensors = filter(lambda i: isinstance(i, torch.Tensor),
                           it.chain(args, kwargs.values()))
    total_input_size = sum(map(_sizeof, input_tensors))
    total_output_size = 2. * total_output_size  # x2 for gradients

    nb_params = trainable_params, total_params - trainable_params
    nb_bytes = total_param_size, total_output_size, total_input_size

    return Summary.create(layers, nb_params, nb_bytes)
