from typing import List

import torch
import torch.nn as nn


def auto_name_prune_quantize_layers(net: nn.Module) -> nn.Module:
    """Set name attribute of Prune/Quantize layers based on their torch module
    paths. This utility can be applied for better logging.

    Args:
        net (nn.Module): network module with [PruneLayer][qsparse.sparse.PruneLayer] and [QuantizeLayer][qsparse.quantize.QuantizeLayer].

    Returns:
        nn.Module: modified module
    """

    from qsparse.quantize import QuantizeLayer
    from qsparse.sparse import PruneLayer

    for name, mod in net.named_modules():
        if isinstance(mod, (PruneLayer, QuantizeLayer)):
            mod.name = name
    return net


def nd_slice(d: int, dim: int = 0, start: int = 0, end: int = 1) -> List[slice]:
    """Create a multi-dimensional slice.

    Args:
        d (int): number of dimensions
        dim (int, optional): target dimension that will be sliced. Defaults to 0.
        start (int, optional): start index in the target dimension. Defaults to 0.
        end (int, optional): end index in the target dimension. Defaults to 1.

    Returns:
        List[slice]: multi-dimensional slice
    """
    indexes = [slice(None)] * d
    indexes[dim] = slice(start, end)
    return indexes
