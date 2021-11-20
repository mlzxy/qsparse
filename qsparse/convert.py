import copy
from typing import Optional, Sequence, Type, Union

import torch
import torch.nn as nn

from qsparse.quantize import QuantizeLayer, quantize
from qsparse.sparse import PruneLayer, prune
from qsparse.util import auto_name_prune_quantize_layers


def convert(
    model: nn.Module,
    operator: Union[PruneLayer, QuantizeLayer],
    inplace: bool = False,
    weight_layers: Sequence[Type[nn.Module]] = [nn.Conv2d],
    activation_layers: Sequence[Type[nn.Module]] = [nn.BatchNorm2d],
    input: bool = False,
    output: bool = False,
    log: bool = False,
    excluded_weight_layer_indexes: Sequence[int] = [],
    excluded_activation_layer_indexes: Sequence[int] = [],
) -> nn.Module:
    """Automatically convert a model to a new model with its weights and
    activations transformed by the operator, e.g. [prune][qsparse.sparse.prune] or [quantize][qsparse.quantize.quantize].

    Args:
        model (nn.Module): input network module
        operator (Union[PruneLayer, QuantizeLayer]): operator used to transform the weights and activations.
        inplace (bool, optional): whether mutates the original module. Defaults to False.
        weight_layers (Sequence[Type[nn.Module]], optional): which layers to apply operator to transform weights. Defaults to [nn.Conv2d].
        activation_layers (Sequence[Type[nn.Module]], optional): which layers to apply operator to transform output activations. Defaults to [nn.BatchNorm2d].
        input (bool, optional): whether apply operator to input. Defaults to False.
        output (bool, optional): whether apply operator to output. Defaults to False.
        log (bool, optional): whether print the conversion log. Defaults to False.
        excluded_weight_layer_indexes (Sequence[int], optional): indexes of layers excluded in weight transformations from conversion. Defaults to [].
        excluded_activation_layer_indexes (Sequence[int], optional): indexes of layers excluded in activation transformations from conversion. Defaults to [].

    Returns:
        nn.Module: converted module
    """
    assert isinstance(
        operator, (PruneLayer, QuantizeLayer)
    ), "`operator` does not belong to (PruneLayer, QuantizeLayer)"

    def _print(msg):
        if log:
            print(msg)

    def apply_operator(layer: Optional[nn.Module] = None) -> nn.Module:
        if layer is not None:
            if isinstance(operator, QuantizeLayer):
                return quantize(layer, **operator._kwargs)
            else:
                return prune(layer, **operator._kwargs)
        else:
            return copy.deepcopy(operator)

    if not inplace:
        model = copy.deepcopy(model)

    weight_counter = {str(c): 0 for c in weight_layers}
    activation_counter = {str(c): 0 for c in activation_layers}

    def _convert(mod: nn.Module) -> nn.Module:
        reassign = {}
        for name, m in mod.named_children():
            origin_m = m
            if not isinstance(m, nn.Sequential):
                if str(m.__class__) in weight_counter:
                    m = apply_operator(m)

                if str(m.__class__) in activation_counter:
                    m = [m, apply_operator()]

                if origin_m is not m:
                    if isinstance(m, list):
                        reassign[name] = nn.Sequential(*m)
                    else:
                        reassign[name] = m
            else:
                _convert(m)
        for key, value in reassign.items():
            mod._modules[key] = value
        return mod

    model = _convert(model)
    model = auto_name_prune_quantize_layers(model)
    return model
