import logging as logging_module
from typing import List, Optional, TypeVar

import torch
import torch.nn as nn

T = TypeVar("T")

_options_ = {"log_on_created": True, "log_during_train": True}


def set_options(
    log_on_created: Optional[bool] = None, log_during_train: Optional[bool] = None
):
    """set QSPARSE options. Only the options given will be updated. The exported alias of this function is `set_qsparse_options`.

    Args:
        log_on_created (Optional[bool], optional): If set to True, QSPARSE will log into console when every prune/quantize layer is created, the built-in value is True. Defaults to None.
        log_during_train (Optional[bool], optional): If set to True, QSPARSE will log into console when pruning and quantization happen, the built-in value is True. Defaults to None.
    """
    if log_on_created is not None:
        _options_["log_on_created"] = log_on_created

    if log_during_train is not None:
        _options_["log_during_train"] = log_during_train


def get_option(key: str):
    """return the requested option. The exported alias of this function is `get_qsparse_option`.

    Args:
        key (str): option name

    Returns:
        option value
    """
    assert key in ("log_on_created", "log_during_train")

    return _options_[key]


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



def nn_module(mod: nn.Module) -> nn.Module:
    """Return actual module of a `nn.Module` or `nn.DataParallel`.

    Args:
        mod (nn.Module): input pytorch module

    Returns:
        nn.Module: actual module
    """
    if hasattr(mod, "module"):
        return mod.module
    else:
        return mod


def squeeze_tensor_to_shape(x: torch.Tensor, shape: List[int]) -> torch.Tensor:
    """squeeze a tensor to a given shape through averaging.

    Args:
        x (torch.Tensor): input tensor
        shape (List[int]): target shape

    Raises:
        ValueError: when the input tensor has different number of dimensions than the target shape, or the target shape provides a non-1 dimension to reduce

    Returns:
        torch.Tensor: aligned tensor
    """
    assert len(x.shape) == len(shape), "mismatch between the input tensor and mask"
    for i, (sx, sm) in enumerate(zip(x.shape, shape)):
        if sx != sm:
            if sm == 1:
                x = x.mean(i, keepdim=True)
            else:
                raise ValueError("mismatch between the input tensor and mask")
    return x



def calculate_mask_given_importance(importance: torch.Tensor, sparsity: float) -> torch.Tensor:
    """return a binary torch tensor with sparsity equals to the given sparsity. 

    Args:
        importance (torch.Tensor): Floating-point tensor represents importance.
        sparsity (float): sparsity level in `[0, 1]`

    Returns:
        torch.Tensor: binary mask
    """
    values = importance.flatten().sort()[0]
    n = len(values)
    idx = max(int(sparsity * n - 1), 0)
    threshold = values[idx + 1]
    return importance >= threshold




def log_functor(name: str):
    def log(msg: str):
        root = logging_module.root
        if root.hasHandlers():
            getattr(root, name)(msg)
        else:
            print(msg)

    return log


class logging:
    """wrapper of logging module. use `print` if logging module is not configured."""

    info = log_functor("info")
    warn = log_functor("warn")
    warning = log_functor("warning")
    error = log_functor("error")
    exception = log_functor("exception")
    debug = log_functor("debug")
