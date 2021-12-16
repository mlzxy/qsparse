from typing import List, Union

import torch
import torch.nn as nn
from typing_extensions import Protocol

TensorOrInt = Union[int, torch.Tensor]
"""Alias for `Union[int, torch.Tensor]`"""


def ensure_tensor(v) -> torch.Tensor:
    """convert the input to `torch.Tensor` if necessary."""
    if isinstance(v, torch.Tensor):
        return v
    else:
        return torch.tensor(v)


class QuantizeCallback(Protocol):
    """Type signature of the callback used in QuantizeLayer."""

    def __call__(
        self,
        inp: torch.Tensor,
        bits: int = 8,
        decimal: TensorOrInt = 5,
        channel_index: int = 1,
    ) -> torch.Tensor:
        """quantize an input tensor based on provided bitwidth and decimal
        bits.

        Args:
            inp (torch.Tensor): input tensor
            bits (int, optional): bitwidth. Defaults to 8.
            decimal (TensorOrInt, optional): decimal bits. Defaults to 5.
            channel_index (int, optional): the index of the channel dimension, used to implement vector quantization. Defaults to 1.

        Returns:
            torch.Tensor: Quantized tensor. It still has the same data type as the input tensor, but can be
                          identically represented using integer format with provided bitwidth.
        """


class PruneCallback(Protocol):
    """Type signature of the callback used in PruneLayer."""

    def __call__(
        self,
        inp: List[torch.Tensor],
        sparsity: float,
        current_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """calculate the binary mask used for pruning with the input tensor and
        target sparsity.

        Args:
            inp (List[torch.Tensor]): input tensor list. Tensors in this list shall have the same shape and no batch dimension. For example, for CIFAR10 images, the shapes of the input tensors are `[(3, 32, 32), (3, 32, 32), ...]`
            sparsity (float): target sparsity (ratio of zeros)
            current_mask (torch.Tensor, optional): current mask of the pruning procedure. Defaults to None.

        Returns:
            torch.Tensor: binary mask
        """
