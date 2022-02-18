from typing import List, Union

import torch
import torch.nn as nn
from typing_extensions import Protocol

TensorOrInt = Union[int, torch.Tensor]
"""Alias for `Union[int, torch.Tensor]`"""

TensorOrFloat = Union[float, torch.Tensor]
"""Alias for `Union[float, torch.Tensor]`"""


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


class QuantizeOptimizer(Protocol):
    """Type signature of the optimizer used in QuantizeLayer."""

    def __call__(
        self,
        inp: torch.Tensor,
        bits: int,
        init: Union[float, int],
        quantize_callback: QuantizeCallback,
    ) -> Union[float, int]:
        """calculate the best weight for quantizing the given tensor.

        Args:
            tensor (torch.Tensor): input tensor
            bits (int): bitwidth
            init (Union[float, int]): initial quantization weight
            quantize_callback (QuantizeCallback, optional): callback for actual operation of quantizing tensor.

        Returns:
            Union[float, int]: optimal quantization weight (e.g. scaler, fractional bits)
        """


class PruneCallback(Protocol):
    """Type signature of the callback used in PruneLayer."""

    def __call__(
        self,
        inp: Union[List[torch.Tensor], torch.Tensor],
        sparsity: float,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """calculate the binary mask used for pruning with the input tensor and
        target sparsity.

        Args:
            inp (Union[List[torch.Tensor], torch.Tensor]): input tensor or input tensor list. If tensors are provided in a list, they shall have the same shape and no batch dimension. For example, for CIFAR10 images, the shapes of the input tensors are `[(3, 32, 32), (3, 32, 32), ...]`
            sparsity (float): target sparsity (ratio of zeros)
            mask (torch.Tensor, optional): init mask of the pruning procedure. Defaults to None.

        Returns:
            torch.Tensor: binary mask or pruned version of input
        """
