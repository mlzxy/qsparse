import math

import torch
import torch.nn as nn

from qsparse.quantize import QuantizeLayer
from qsparse.sparse import PruneLayer


def auto_name_prune_quantize_layers(net: nn.Module):
    for name, mod in net.named_modules():
        if isinstance(mod, (PruneLayer, QuantizeLayer)):
            mod.name = name


def approx_quantile(t: torch.Tensor, fraction: float) -> float:
    # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Sorting.cpp#L221
    size = t.numel()
    bound = 2 ** 24
    if size <= bound:
        return torch.quantile(t, fraction)
    else:
        t = t.view(-1)
        qs = []
        num_chunks = math.ceil(size / bound)
        for i in range(num_chunks):
            if i == (num_chunks - 1):
                chunk = t[
                    size - bound :
                ]  # ensure won't be biased if the last chunk is very small
            else:
                chunk = t[i * bound : (i + 1) * bound]
            qs.append(torch.quantile(chunk, fraction))
        return sum(qs) / len(qs)
