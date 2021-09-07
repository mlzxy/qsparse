from qsparse.quantize import QuantizeLayer
from qsparse.sparse import PruneLayer
import torch.nn as nn


def auto_name_prune_quantize_layers(net: nn.Module):
    for name, mod in net.named_modules():
        if isinstance(mod, (PruneLayer, QuantizeLayer)):
            mod.name = name
