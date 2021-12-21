import copy
import logging
import warnings
from collections import defaultdict
from typing import Mapping, Optional, Sequence, Tuple, Type, Union

import torch.nn as nn
from torch.nn.modules.container import Sequential

from qsparse.quantize import QuantizeLayer, quantize
from qsparse.sparse import PruneLayer, prune
from qsparse.util import auto_name_prune_quantize_layers, nn_module


def convert(  # noqa: C901
    model: nn.Module,
    operator: Union[PruneLayer, QuantizeLayer],
    inplace: bool = False,
    weight_layers: Sequence[Type[nn.Module]] = [],
    activation_layers: Sequence[Type[nn.Module]] = [],
    input: bool = False,
    log: bool = True,
    excluded_weight_layer_indexes: Sequence[Tuple[Type[nn.Module], Sequence[int]]] = [],
    excluded_activation_layer_indexes: Sequence[
        Tuple[Type[nn.Module], Sequence[int]]
    ] = [],
) -> nn.Module:
    """Automatically convert a model to a new model with its weights and
    activations transformed by the operator, e.g. [prune][qsparse.sparse.prune] or [quantize][qsparse.quantize.quantize].

    Args:
        model (nn.Module): input network module
        operator (Union[PruneLayer, QuantizeLayer]): operator used to transform the weights and activations.
        inplace (bool, optional): whether mutates the original module. Defaults to False.
        weight_layers (Sequence[Type[nn.Module]], optional): which layers to apply operator to transform weights. Defaults to [].
        activation_layers (Sequence[Type[nn.Module]], optional): which layers to apply operator to transform output activations. Defaults to [].
        input (bool, optional): whether apply operator to input. Defaults to False.
        log (bool, optional): whether print the conversion log. Defaults to True.
        excluded_weight_layer_indexes (Sequence[Tuple[Type[nn.Module], Sequential[int]]], optional): indexes of layers excluded in weight transformations from conversion. Defaults to [].
        excluded_activation_layer_indexes (Sequence[Tuple[Type[nn.Module], Sequential[int]]], optional): indexes of layers excluded in activation transformations from conversion. Defaults to [].

    Returns:
        nn.Module: converted module
    """
    assert isinstance(
        operator, (PruneLayer, QuantizeLayer)
    ), "`operator` does not belong to (PruneLayer, QuantizeLayer)"

    if (len(weight_layers) + len(activation_layers)) == 0:
        warnings.warn(
            "No weight or activation layers specified, nothing will be converted."
        )

    def _print(msg):
        if log:
            logging.info(msg)

    def mstr(m) -> str:
        if isinstance(m, nn.Sequential):
            return mstr(m[0])
        elif isinstance(m, nn.Module):
            return m.__class__.__name__
        else:
            return m.__name__

    def is_container(m: nn.Module) -> bool:
        if len(m._modules) == 0:
            return False
        else:
            for k in [
                "quantize",
                "prune",
                "quantize_bias",
            ]:  # a leaf module injected with qsparse layers
                if hasattr(m, k):
                    return False
            return True

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

    def count_occurrence(
        mod: nn.Module, layer_types: Sequence[Type[nn.Module]]
    ) -> Mapping[str, int]:
        def _count_occurrence(m: nn.Module, layer_type):
            total = 0
            for _, layer in m.named_children():
                if not is_container(layer):
                    if mstr(layer) == mstr(layer_type):
                        total += 1
                else:
                    total += _count_occurrence(layer, layer_type)
            return total

        return {mstr(l): _count_occurrence(mod, l) for l in layer_types}

    def excluded_layer_indexes_to_dict(
        layer_indexes, layers_count
    ) -> Mapping[str, Sequence[int]]:
        return defaultdict(
            list,
            {
                mstr(cls): [
                    l if l >= 0 else (l + layers_count[mstr(cls)]) for l in indexes
                ]
                for cls, indexes in layer_indexes
            },
        )

    weight_counter = {mstr(c): 0 for c in weight_layers}
    weight_total_layers_counts = count_occurrence(model, weight_layers)
    excluded_weight_layer_indexes = excluded_layer_indexes_to_dict(
        excluded_weight_layer_indexes, weight_total_layers_counts
    )

    activation_counter = {mstr(c): 0 for c in activation_layers}
    activation_total_layers_counts = count_occurrence(model, activation_layers)
    excluded_activation_layer_indexes = excluded_layer_indexes_to_dict(
        excluded_activation_layer_indexes, activation_total_layers_counts
    )

    operator_name = str(operator).lower()
    for c in ["(", ")", "layer"]:
        operator_name = operator_name.replace(c, "")
    operator_name = f"`{operator_name}`"

    def _convert_weight(mod: nn.Module, scope: str = "") -> nn.Module:
        reassign = {}
        for name, m in mod.named_children():
            modified = False
            if not is_container(m):
                if mstr(m) in weight_counter:
                    if (
                        weight_counter[mstr(m)]
                        not in excluded_weight_layer_indexes[mstr(m)]
                    ):
                        _print(f"Apply {operator_name} on the {scope}.{name} weight")
                        m = apply_operator(m)
                        modified = True
                    else:
                        _print(f"Exclude {scope}.{name} weight")
                    weight_counter[mstr(m)] += 1
                if modified:
                    reassign[name] = m
            else:
                _convert_weight(m, f"{scope}.{name}.")

        for key, value in reassign.items():
            mod._modules[key] = value
        return mod

    def _convert_activation(mod: nn.Module, scope: str = "") -> nn.Module:
        reassign = {}
        for name, m in mod.named_children():
            origin_m = m
            if (not is_container(m)) or hasattr(m, "_qsparse_conversion"):
                if mstr(m) in activation_counter:
                    if (
                        activation_counter[mstr(m)]
                        not in excluded_activation_layer_indexes[mstr(m)]
                    ):
                        _print(
                            f"Apply {operator_name} on the {scope}.{name} activation"
                        )
                        m = nn.Sequential(m, apply_operator())
                        setattr(m, "_qsparse_conversion", True)
                    else:
                        _print(f"Exclude {scope}.{name} activation")
                    activation_counter[mstr(m)] += 1
                if origin_m is not m:
                    reassign[name] = m
            else:
                _convert_activation(m, f"{scope}.{name}.")

        for key, value in reassign.items():
            mod._modules[key] = value
        return mod

    def apply_to_input(mod):
        return nn.Sequential(apply_operator(), mod)

    model = _convert_weight(nn_module(model))
    model = _convert_activation(nn_module(model))
    if input:
        _model = apply_to_input(nn_module(model))
        if model == nn_module(model):
            model = _model
        else:
            model.module = _model

    model = auto_name_prune_quantize_layers(model)
    return model
