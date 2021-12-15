from copy import copy
from typing import Dict, Iterable, Mapping, Optional

import torch
import torch.nn as nn
from typing_extensions import Protocol


class BNFuser(Protocol):
    """Type signature of the handers used in fuse_bn."""

    def __call__(self, layer: nn.Module, bn: nn.Module) -> nn.Module:
        """fuse the batch normalization module into the corresponding layer.

        Args:
            layer (nn.Module): layers like Conv2d, Linear, etc.
            bn (nn.Module): batch norm layer, could be BatchNorm2d, BatchNorm1d, etc.

        Returns:
            nn.Module: fused layer
        """


def fuse_bn_conv2d(conv: nn.Module, bn: nn.Module) -> nn.Module:
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


def fuse_bn_linear(linear: nn.Module, bn: nn.Module) -> nn.Module:
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


def fuse_bn_deconv2d(deconv: nn.Module, bn: nn.Module) -> nn.Module:
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
    Conv2d=fuse_bn_conv2d, Linear=fuse_bn_linear, ConvTranspose2d=fuse_bn_deconv2d
)  # type: Dict[str, BNFuser]


def fuse_bn(
    model: nn.Module,
    layers: Iterable[str] = ["Conv2d", "Linear", "ConvTranspose2d"],
    handlers: Optional[Mapping[str, BNFuser]] = None,
) -> nn.Module:
    """Fuse the batch norm layers back to the previous conv/deconv/linear layers in a newtwork.

    Args:
        model (nn.Module): network
        layers (Iterable[str], optional): [description]. Defaults to ["Conv2d", "Linear", "ConvTranspose2d"].
        handlers (Optional[Mapping[str, BNFuser]], optional): Mapping from layer type to [BNFuser][qsparse.fuse.BNFuser]. Defaults to None, will use { Linear: [fuse\_bn\_linear][qsparse.fuse.fuse_bn_linear], Conv2d: [fuse\_bn\_conv2d][qsparse.fuse.fuse_bn_conv2d], ConvTranspose2d: [fuse\_bn\_deconv2d][qsparse.fuse.fuse_bn_deconv2d] }.

    Returns:
        nn.Module: network with bn fused
    """
    handlers = copy(default_handlers).update(handlers or {})
    layers = set(layers)
    for name in layers:
        assert name in handlers, f"layer {name} is not in handlers"

    def is_bn(layer):
        return layer.__class__.__name__.lower().startswith("batchnorm")

    def fuse_bn_sequential(seq):
        sequence = []
        for layer in seq.children():
            if is_bn(layer):
                bn = layer
                operation = sequence[-1]
                layer_type = operation.__class__.__name__
                if layer_type in layers:
                    sequence[-1] = handlers[layer_type](operation, bn)
                else:
                    sequence.append(bn)
            elif isinstance(layer, nn.Sequential):
                sequence.append(fuse_bn_sequential(layer))
            else:
                sequence.append(layer)
        return nn.Sequential(*sequence) if len(sequence) > 1 else sequence[0]

    if isinstance(model, nn.Sequential):
        return fuse_bn_sequential(model)
    else:
        for name, m in model.named_children():
            if isinstance(m, nn.Sequential):
                model._modules[name] = fuse_bn_sequential(m)
        return model
