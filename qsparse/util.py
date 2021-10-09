import torch
import torch.nn as nn


def auto_name_prune_quantize_layers(net: nn.Module):
    from qsparse.quantize import QuantizeLayer
    from qsparse.sparse import PruneLayer

    for name, mod in net.named_modules():
        if isinstance(mod, (PruneLayer, QuantizeLayer)):
            mod.name = name


def nd_slice(d, dim=0, start=0, end=1):
    indexes = [slice(None)] * d
    indexes[dim] = slice(start, end)
    return indexes
