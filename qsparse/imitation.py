from typing import Type

import torch
import torch.nn as nn


def imitate(human: nn.Module, name: str, thing: Type):
    InputClass = human.__class__

    def get_prev_weight(self: nn.Module):
        if hasattr(InputClass, "weight"):
            return getattr(InputClass, "weight").__get__(self)
        else:
            return self._parameters["weight"]

    setattr(human, name, thing)

    class Imitation(InputClass):
        @property
        def weight(self):
            return getattr(self, name)((get_prev_weight(self)))

    human.__class__ = Imitation
    return human
