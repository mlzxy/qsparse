from typing import Callable

import torch
import torch.nn as nn


def imitate(
    human: nn.Module, name: str, thing: Callable, bias_thing: Callable = None
) -> nn.Module:
    """Patch the class of the input module to in-place transform its weight /
    bias attributes.

    Suppose we have an input module `mod` with a weight attribute, and `mod.weight` returns `tensor([1, 2])`.

    ```python
    mod2 = imitate(mod, "add1", lambda x: x + 1)
    ```

    Then `mod2` will behave identically to `mod`, except that `mod2.weight` will return `tensor([2, 3])`.
    This mechanism enables us to design a much simpler set of APIs than existing libraries.

    The only downside here is the returned module is not pickleable because python pickle doesn't support closure.
    We believe it is a minor issue because

    1. In most scenarios, only the state dict is pickled.
    2. We could use [cloudpickle](https://github.com/cloudpipe/cloudpickle) like `torch.save(module, path, pickle_module=cloudpickle)`, which supports closures.

    >_This function is inspired from the John Carpenter's movie `The Thing`._

    Args:
        human (nn.Module): input module
        name (str): name of the transform
        thing (Callable): function that transforms the weight attribute
        bias_thing (Callable, optional): function that transforms the bias attribute. Defaults to None.

    Returns:
        nn.Module: output module whose weight / bias attributes are transformed
    """

    InputClass = human.__class__

    def get_prev_weight(self: nn.Module):
        if hasattr(InputClass, "weight"):
            return getattr(InputClass, "weight").__get__(self)
        else:
            return self._parameters["weight"]

    def get_prev_bias(self: nn.Module):
        if hasattr(InputClass, "bias"):
            return getattr(InputClass, "bias").__get__(self)
        else:
            return self._parameters["bias"]

    setattr(human, name, thing)

    if bias_thing is not None:
        setattr(human, name + "_bias", bias_thing)
    else:
        setattr(human, name + "_bias", lambda x: x)

    class Imitation(InputClass):
        @property
        def weight(self):
            return getattr(self, name)((get_prev_weight(self)))

        @property
        def bias(self):
            return getattr(self, name + "_bias")((get_prev_bias(self)))

    Imitation.__name__ = InputClass.__name__
    human.__class__ = Imitation
    return human
