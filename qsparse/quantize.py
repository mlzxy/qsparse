from collections import deque
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from typing_extensions import Protocol

__all__ = [
    "quantize",
]


def ensure_tensor(v):
    if isinstance(v, torch.Tensor):
        return v
    else:
        return torch.tensor(v)


class LinearQuantization(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        bits: int = 8,
        decimal: Union[int, torch.Tensor] = 5,
        channel_index: int = 1,
    ):
        limit = 2.0 ** (bits - 1)
        tof = 2.0 ** -decimal
        toi = 2.0 ** decimal
        shape = [1 for _ in input.shape]
        if isinstance(decimal, torch.Tensor) and sum(decimal.shape) > 1:
            assert (
                len(decimal) == input.shape[channel_index]
            ), "channel of input and decimal must be equal in vector quantization"
            shape[channel_index] = -1
            tof, toi = tof.view(*shape), toi.view(*shape)
        ctx.save_for_backward(ensure_tensor(limit), ensure_tensor(tof))
        q = (input * toi).int()
        q.clamp_(-limit, limit - 1)
        return q.float() * tof

    @staticmethod
    def backward(ctx, grad_output):
        """
        Straight Through Gradient Estimator
        """
        limit, tof = ctx.saved_tensors
        v = grad_output.clamp_(-limit * tof, (limit - 1) * tof)
        return (v, None, None, None, None)


class QuantizationCallback(Protocol):
    def __call__(
        self,
        inp: torch.Tensor,
        bits: int = 8,
        decimal: Union[int, torch.Tensor] = 5,
        channel_index: int = 1,
    ) -> torch.Tensor:
        pass


def linear_quantize_callback(
    inp: torch.Tensor,
    bits: int = 8,
    decimal: Union[int, torch.Tensor] = 5,
    channel_index: int = 1,
) -> torch.Tensor:
    return LinearQuantization.apply(inp, bits, decimal, channel_index)


def arg_decimal_min_mse(
    tensor: torch.Tensor, bits: int, decimal_range: Tuple[int, int] = (1, 20)
):
    """
    calculate the best decimal point for quantizing tensor
    """
    err = float("inf")
    best_n = None
    assert len(decimal_range) == 2
    tensor = tensor.reshape(-1)  # flatten
    for n in range(*decimal_range):
        tensor_q = quantize(tensor, bits, decimal=n)
        err_ = torch.sum((tensor - tensor_q) ** 2).item()
        if err_ < err:
            best_n = n
            err = err_
    return best_n


def quantize(
    arg: Optional[Union[torch.Tensor, nn.Module]] = None,
    bits: int = 8,
    channelwise: int = 1,
    decimal_range: Tuple[int, int] = (1, 20),
    # for tensor computation
    decimal: Optional[Union[int, torch.Tensor]] = None,
    # for step-wise training
    timeout: int = 1000,
    buffer_size: int = 1,
    # for customization
    callback: QuantizationCallback = linear_quantize_callback,
    # for debug purpose
    name: str = "",
) -> Optional[Union[torch.Tensor, nn.Module]]:
    class QuantizeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            print(
                f"[Quantize @ {name}] bits={bits} channelwise={channelwise} buffer_size={buffer_size} timeout={timeout}"
            )
            self.buffer = deque(maxlen=buffer_size)
            self._inited = False

        def forward(self, x):
            if not self._inited:
                self.decimal = nn.Parameter(
                    torch.ones(1 if channelwise < 0 else x.shape[channelwise]).to(
                        x.device
                    ),
                    requires_grad=False,
                )
                self._n_updates = nn.Parameter(
                    torch.zeros(1, dtype="int"),
                    requires_grad=False,
                )
                self._inited = True

            if self._n_updates <= timeout:
                self.buffer.append(x.detach().to("cpu"))

            if self._n_updates == timeout:
                print(f"Quantizing {name}")
                if channelwise >= 0:
                    for i in range(x.shape[channelwise]):
                        n = arg_decimal_min_mse(
                            x[
                                tuple(
                                    [
                                        slice(0, cs) if ci != channelwise else i
                                        for ci, cs in enumerate(x.shape)
                                    ]
                                )
                            ],
                            bits,
                            decimal_range,
                        )
                        print(f"{name} decimal for channel {i} = {n}")
                        self.decimal.data[i] = n
                else:
                    n = arg_decimal_min_mse(
                        torch.cat([a for a in self.buffer], dim=0), bits, decimal_range
                    )
                    print(f"{name} decimal = {n}")
                    self.decimal.data[:] = n

            if self._n_updates < timeout:
                out = x
            else:
                out = callback(x, bits, self.decimal, channelwise)

            if self.training:
                self._n_updates += 1
            return out

    if arg is None:
        return QuantizeLayer()
    elif isinstance(arg, torch.Tensor):
        assert (
            decimal is not None
        ), "decimal points for the input tensor must be provided"
        return linear_quantize_callback(arg, bits, decimal, channelwise)
    elif isinstance(arg, nn.Module):
        InputClass = arg.__class__

        def get_prev_weight(self: nn.Module):
            if "Imitation" in str(InputClass):
                return InputClass.weight.__get__(
                    self
                )  # prune is already called, so the weight is a property
            else:
                return self._parameters["weight"]

        arg.quantize = QuantizeLayer()

        class ImitationQuantize(nn.Module):
            @property
            def weight(self):
                return self.quantize(get_prev_weight(self))

        arg.__class__ = ImitationQuantize
        return arg


if __name__ == "__main__":
    print(quantize())
    print(quantize(torch.nn.Conv2d(10, 30, 3)))
