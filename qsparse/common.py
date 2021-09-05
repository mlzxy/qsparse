from typing import List, Optional, Union

import torch
import torch.nn as nn
from typing_extensions import Protocol

OptionalTensorOrModule = Optional[Union[torch.Tensor, nn.Module]]

TensorOrInt = Union[int, torch.Tensor]


def ensure_tensor(v):
    if isinstance(v, torch.Tensor):
        return v
    else:
        return torch.tensor(v)


class QuantizeCallback(Protocol):
    def __call__(
        self,
        inp: torch.Tensor,
        bits: int = 8,
        decimal: TensorOrInt = 5,
        channel_index: int = 1,
    ) -> torch.Tensor:
        pass


class PruneCallback(Protocol):
    def __call__(self, inp: List[torch.Tensor], sparsity: float) -> torch.Tensor:
        pass
