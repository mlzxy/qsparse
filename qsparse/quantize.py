from collections import deque
from typing import Optional, Tuple

import torch
import torch.nn as nn

from qsparse.common import (
    OptionalTensorOrModule,
    QuantizeCallback,
    TensorOrInt,
    ensure_tensor,
)
from qsparse.imitation import imitate

__all__ = [
    "quantize",
]


class LinearQuantization(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        bits: int = 8,
        decimal: TensorOrInt = 5,
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


def linear_quantize_callback(
    inp: torch.Tensor,
    bits: int = 8,
    decimal: TensorOrInt = 5,
    channel_index: int = 1,
) -> torch.Tensor:
    return LinearQuantization.apply(inp, bits, decimal, channel_index)


def arg_decimal_min_mse(
    tensor: torch.Tensor, bits: int, decimal_range: Tuple[int, int] = (0, 20)
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


class QuantizeLayer(nn.Module):
    def __init__(
        self,
        bits: int = 8,
        channelwise: int = 1,
        decimal_range: Tuple[int, int] = (0, 20),
        # for step-wise training
        timeout: int = 1000,
        interval: int = -1,
        buffer_size: int = 1,
        # for customization
        callback: QuantizeCallback = linear_quantize_callback,
        # for debug purpose
        name: str = "",
    ):
        super().__init__()
        print(
            f"[Quantize @ {name}] bits={bits} channelwise={channelwise} buffer_size={buffer_size} timeout={timeout}"
        )
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.name = name
        self.channelwise = channelwise
        self.timeout = timeout
        self._bits = bits
        self.callback = callback
        self.interval = interval
        self.decimal_range = decimal_range
        self._init = False
        self._quantized = False

        for k in [
            "decimal",
            "_n_updates",
            "bits",
        ]:
            self.register_parameter(k, None)

    def forward(self, x):
        if not self._init:
            self.decimal = nn.Parameter(
                torch.ones(1 if self.channelwise < 0 else x.shape[self.channelwise]).to(
                    x.device
                ),
                requires_grad=False,
            )
            self._n_updates = nn.Parameter(
                torch.zeros(1, dtype=torch.int).to(x.device),
                requires_grad=False,
            )
            self.bits = nn.Parameter(
                torch.tensor(self._bits, dtype=torch.int).to(x.device),
                requires_grad=False,
            )
            self._init = True

        if (
            (self.timeout - self.buffer_size) < self._n_updates <= self.timeout
        ):  # only collecting buffer when needed to speedup
            self.buffer.append(x.detach().to("cpu"))
        else:
            if self.interval <= 0:
                self.buffer.clear()

        # add buffer size check to avoid quantize a layer which always set to eval
        if (self.training or (not self._quantized)) and (len(self.buffer) > 0):
            if (self._n_updates == self.timeout and not self._quantized) or (
                self._n_updates > self.timeout
                and ((self._n_updates - self.timeout) % self.interval) == 0
                and self.interval > 0
            ):
                if self.channelwise >= 0:
                    print(f"Quantizing {self.name} (channelwise)")
                    for i in range(x.shape[self.channelwise]):
                        n = arg_decimal_min_mse(
                            x[
                                tuple(
                                    [
                                        slice(0, cs) if ci != self.channelwise else i
                                        for ci, cs in enumerate(x.shape)
                                    ]
                                )
                            ],
                            self.bits,
                            self.decimal_range,
                        )
                        # print(f"{self.name} decimal for channel {i} = {n}")
                        self.decimal.data[i] = n
                else:
                    n = arg_decimal_min_mse(
                        torch.cat([a.view(-1) for a in self.buffer], dim=0),
                        self.bits,
                        self.decimal_range,
                    )
                    print(f"{self.name} decimal = {n}")
                    self.decimal.data[:] = n

                self._quantized = True

                # proactively free up memory
                if self.interval <= 0:
                    self.buffer.clear()

        if self._n_updates >= self.timeout and self._quantized:
            out = self.callback(x, self.bits, self.decimal, self.channelwise)
        else:
            out = x

        if self.training:
            self._n_updates += 1
        return out


def quantize(
    arg: OptionalTensorOrModule = None,
    bits: int = 8,
    channelwise: int = 1,
    decimal_range: Tuple[int, int] = (0, 20),
    # for tensor computation
    decimal: Optional[TensorOrInt] = None,
    # for step-wise training
    timeout: int = 1000,
    interval: int = -1,
    buffer_size: int = 1,
    # for customization
    callback: QuantizeCallback = linear_quantize_callback,
    # for debug purpose
    name: str = "",
) -> OptionalTensorOrModule:
    def get_quantize_layer():
        return QuantizeLayer(
            bits=bits,
            channelwise=channelwise,
            decimal_range=decimal_range,
            timeout=int(timeout),
            interval=int(interval),
            buffer_size=buffer_size,
            callback=callback,
            name=name,
        )

    if arg is None:
        return get_quantize_layer()
    elif isinstance(arg, torch.Tensor):
        assert (
            decimal is not None
        ), "decimal points for the input tensor must be provided"
        return linear_quantize_callback(arg, bits, decimal, channelwise)
    elif isinstance(arg, nn.Module):
        return imitate(arg, "quantize", get_quantize_layer())
    else:
        raise ValueError(f"{arg} is not a valid argument for quantize")


if __name__ == "__main__":
    layer = quantize(timeout=0)
    print(layer)
    print(quantize(torch.nn.Conv2d(10, 30, 3)))

    data = torch.rand(10, 10)
    print(data)
    print(layer(data))
