import logging
from copy import copy, deepcopy
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from typing_extensions import Protocol

from qsparse.util import nn_module


class BNFuser(Protocol):
    """Type signature of the handers used in fuse_bn."""

    def __call__(self, layer: nn.Module, bn: nn.Module) -> nn.Module:
        """Fuse batch norm into the previous layer.

        Args:
            layer (nn.Module): layers like Conv2d, Linear, etc.
            bn (nn.Module): batch norm layer, could be BatchNorm2d, BatchNorm1d, etc.

        Returns:
            nn.Module: fused layer
        """


def conv2d_bn_fuser(conv: nn.Module, bn: nn.Module) -> nn.Module:
    """BNFuser for Conv2d"""
    w = conv._parameters["weight"].detach()
    b = conv._parameters["bias"].detach() if conv.bias is not None else 0
    mean = bn.running_mean.detach()
    var_sqrt = torch.sqrt(bn.running_var.detach().add(1e-5))
    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    new_weight = w * (gamma / var_sqrt)[:, None, None, None]
    new_bias = (b - mean) * gamma / var_sqrt + beta
    conv._parameters["weight"].data = new_weight
    conv._parameters["bias"] = nn.Parameter(new_bias)
    return conv


def linear_bn_fuser(linear: nn.Module, bn: nn.Module) -> nn.Module:
    """BNFuser for Linear"""
    w = linear._parameters["weight"].detach()
    b = linear._parameters["bias"].detach() if linear.bias is not None else 0
    mean = bn.running_mean.detach()
    var_sqrt = torch.sqrt(bn.running_var.detach().add(1e-5))
    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    new_weight = w * (gamma / var_sqrt)[:, None]
    new_bias = (b - mean) * gamma / var_sqrt + beta
    linear._parameters["weight"].data = new_weight
    linear._parameters["bias"] = nn.Parameter(new_bias)
    return linear


def deconv2d_bn_fuser(deconv: nn.Module, bn: nn.Module) -> nn.Module:
    """BNFuser for ConvTranspose2d"""
    w = deconv._parameters["weight"].detach()
    b = deconv._parameters["bias"].detach() if deconv.bias is not None else 0
    mean = bn.running_mean.detach()
    var_sqrt = torch.sqrt(bn.running_var.detach().add(1e-5))
    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    new_weight = w * (gamma / var_sqrt)[None, :, None, None]
    new_bias = (b - mean) * gamma / var_sqrt + beta
    deconv._parameters["weight"].data = new_weight
    deconv._parameters["bias"] = nn.Parameter(new_bias)
    return deconv


default_handlers = dict(
    Conv2d=conv2d_bn_fuser, Linear=linear_bn_fuser, ConvTranspose2d=deconv2d_bn_fuser
)  # type: Dict[str, BNFuser]


def fuse_bn(  # noqa: C901
    model: nn.Module,
    layers: Iterable[str] = ["Conv2d", "Linear", "ConvTranspose2d"],
    handlers: Optional[Mapping[str, BNFuser]] = None,
    log: bool = True,
    inplace: bool = True,
) -> nn.Module:
    """Fuse the batch norm layers back to the previous conv/deconv/linear layers in a newtwork.

    Args:
        model (nn.Module): network
        layers (Iterable[str], optional): [description]. Defaults to ["Conv2d", "Linear", "ConvTranspose2d"].
        handlers (Optional[Mapping[str, BNFuser]], optional): Mapping from layer type to [BNFuser][qsparse.fuse.BNFuser]. Defaults to None, will use { Linear: [fuse\_bn\_linear][qsparse.fuse.fuse_bn_linear], Conv2d: [fuse\_bn\_conv2d][qsparse.fuse.fuse_bn_conv2d], ConvTranspose2d: [fuse\_bn\_deconv2d][qsparse.fuse.fuse_bn_deconv2d] }.
        log (bool, optional): whether print the fuse log. Defaults to True.
        inplace (bool, optional): whether mutates the original module. Defaults to False.

    Returns:
        nn.Module: network with bn fused
    """
    handlers = {**copy(default_handlers), **(handlers or {})}
    layers = set(layers)
    for name in layers:
        assert name in handlers, f"layer {name} is not in handlers"

    if not inplace:
        model = deepcopy(model)

    def is_bn(layer: nn.Module) -> bool:
        return layer.__class__.__name__.lower().startswith("batchnorm")

    def get_layer_type(layer: Optional[nn.Module]) -> str:
        if layer is None:
            return ""
        else:
            return layer.__class__.__name__

    def fuse_bn_sequential(
        seq: nn.Sequential, input: Optional[nn.Module] = None
    ) -> Tuple[nn.Module, Optional[nn.Module]]:
        sequence = []

        def get_prev_layer():
            return sequence[-1] if len(sequence) > 0 else input

        for layer in seq.children():
            if is_bn(layer):
                bn = layer
                operation = get_prev_layer()
                layer_type = get_layer_type(operation)
                if layer_type in layers:
                    if log:
                        logging.info(f"Fuse {bn} into {operation}")
                    operation = handlers[layer_type](operation, bn)
                    if len(sequence) > 0:
                        sequence[-1] = operation
                    else:
                        input = operation
                else:
                    sequence.append(bn)
            elif isinstance(layer, nn.Sequential):
                layer, prev_layer = fuse_bn_sequential(layer, get_prev_layer())
                if prev_layer is not None:
                    if len(sequence) > 0:
                        sequence[-1] = prev_layer
                    else:
                        input = prev_layer
                if layer is not None:
                    sequence.append(layer)
            else:
                sequence.append(layer)
        if len(sequence) == 0:
            return None, input
        elif len(sequence) == 1:
            return sequence[0], input
        else:
            return nn.Sequential(*sequence), input

    if isinstance(nn_module(model), nn.Sequential):
        _model = fuse_bn_sequential(nn_module(model))[0]
        if model == nn_module(model):
            model = _model
        else:
            model.module = _model
    else:
        for name, m in nn_module(model).named_children():
            if isinstance(m, nn.Sequential):
                nn_module(model)._modules[name] = fuse_bn_sequential(m)[0]
    return model
