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