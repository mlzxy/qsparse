from typing import Sequence, Type, Union

import torch
import torch.nn as nn

from qsparse.quantize import QuantizeLayer
from qsparse.sparse import PruneLayer


def convert(
    mod: nn.Module,
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
        mod (nn.Module): input network module
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
    raise NotImplementedError
