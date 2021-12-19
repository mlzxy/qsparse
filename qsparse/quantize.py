import logging
import math
import warnings
from collections import deque
from typing import Tuple

import torch
import torch.nn as nn

from qsparse.common import QuantizeCallback, TensorOrInt, ensure_tensor
from qsparse.imitation import imitate
from qsparse.util import get_option, nd_slice

__all__ = [
    "quantize",
]


def approx_quantile(t: torch.Tensor, fraction: float, bound: int = 2 ** 24) -> float:
    """calculate approximate quantiles of input tensor.

    The reason we use this instead of `torch.quantile` is that `torch.quantile` has
    size limit as indicated in https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Sorting.cpp#L221

    Args:
        t (torch.Tensor): input tensor
        fraction (float): quantile percentage, ranges [0, 1]
        bound (int, optional): size threshold of input tensor to trigger approximate computation

    Returns:
        float: quantile value
    """
    size = t.numel()
    if size <= bound:
        return torch.quantile(t, fraction)
    else:
        t = t.view(-1)
        qs = []
        num_chunks = math.ceil(size / bound)
        for i in range(num_chunks):
            if i == (num_chunks - 1):
                chunk = t[
                    size - bound :
                ]  # ensure won't be biased if the last chunk is very small
            else:
                chunk = t[i * bound : (i + 1) * bound]
            qs.append(torch.quantile(chunk, fraction))
        return sum(qs) / len(qs)


class LinearQuantization(torch.autograd.Function):
    """Straight-Through Gradient Estimator.

    Please look for detailed description on arguments in [linear\_quantize\_callback][qsparse.quantize.linear_quantize_callback].
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        bits: int = 8,
        decimal: TensorOrInt = 5,
        channel_index: int = 1,
    ):
        """quantize the input tensor and prepare for backward computation."""
        limit = 2.0 ** (bits - 1)
        tof = 2.0 ** -decimal
        toi = 2.0 ** decimal
        shape = [1 for _ in input.shape]
        if isinstance(decimal, torch.Tensor) and sum(decimal.shape) > 1:
            assert (
                len(decimal) == input.shape[channel_index]
            ), "channel of input and decimal must be equal in channel-wise quantization"
            shape[channel_index] = -1
            tof, toi = tof.view(*shape), toi.view(*shape)
        ctx.save_for_backward(ensure_tensor(limit), ensure_tensor(tof))
        q = (input * toi).int()
        q.clamp_(-limit, limit - 1)
        return q.float() * tof

    @staticmethod
    def backward(ctx, grad_output):
        """gradient computation for quantization operation."""
        limit, tof = ctx.saved_tensors
        v = grad_output.clamp_(-limit * tof, (limit - 1) * tof)
        return (v, None, None, None, None)


def linear_quantize_callback(
    inp: torch.Tensor,
    bits: int = 8,
    decimal: TensorOrInt = 5,
    channel_index: int = 1,
) -> torch.Tensor:
    """quantization function with type signature of [QuantizeCallback][qsparse.common.QuantizeCallback].

    Args:
        inp (torch.Tensor): input tensor
        bits (int, optional): bitwidth. Defaults to 8.
        decimal (TensorOrInt, optional): decimal bits, will be tensor of decimal bits for channel-wise quantization. Defaults to 5.
        channel_index (int, optional): dimension index for channel. Defaults to 1.

    Returns:
        torch.Tensor: quantized tensor
    """
    return LinearQuantization.apply(inp, bits, decimal, channel_index)


def arg_decimal_min_mse(
    tensor: torch.Tensor,
    bits: int,
    decimal_range: Tuple[int, int] = (0, 20),
    saturate_range: Tuple[float, float] = (0, 1),
    callback: QuantizeCallback = linear_quantize_callback,
) -> int:
    """calculate the best decimal bits for given tensor.

    Args:
        tensor (torch.Tensor): input tensor
        bits (int): bitwidth
        decimal_range (Tuple[int, int], optional): search range of decimal bits. Defaults to (0, 20).
        saturate_range (Tuple[float, float], optional): quantiles used to clamp the input tensor before searching decimal bits. Defaults to (0, 1).
        callback (QuantizeCallback, optional): callback for actual operation of quantizing tensor. Defaults to [linear\_quantize\_callback][qsparse.quantize.linear_quantize_callback].

    Returns:
        int: decimal bits
    """
    err = float("inf")
    best_n = None
    assert len(decimal_range) == 2
    tensor = tensor.reshape(-1)  # flatten
    if saturate_range != (0, 1):
        assert (
            0 <= saturate_range[0] <= saturate_range[1] <= 1
        ), f"illegal saturate_range {saturate_range}"
        tensor = torch.clamp(
            tensor,
            approx_quantile(tensor, saturate_range[0]),
            approx_quantile(tensor, saturate_range[1]),
        )
    for n in range(*decimal_range):
        tensor_q = callback(tensor, bits, decimal=n)
        err_ = torch.sum((tensor - tensor_q) ** 2).item()
        if err_ < err:
            best_n = n
            err = err_
    return best_n


class QuantizeLayer(nn.Module):
    """Applies quantization over input tensor.

    Please look for detailed description in [quantize][qsparse.quantize.quantize]
    """

    def __init__(
        self,
        bits: int = 8,
        channelwise: int = 1,
        decimal_range: Tuple[int, int] = (0, 20),
        saturate_range: Tuple[float, float] = (0, 1),
        # for step-wise training
        timeout: int = 1000,
        interval: int = -1,
        window_size: int = 1,
        # for customization
        callback: QuantizeCallback = linear_quantize_callback,
        # for debug purpose
        collapse: int = 0,
        name: str = "",
    ):
        super().__init__()
        if get_option("log_on_created"):
            logging.info(
                f"[Quantize{name if name == '' else f' @ {name}'}] bits={bits} channelwise={channelwise} window_size={window_size} timeout={timeout}"
            )
        self.window = deque(maxlen=window_size)
        self.window_size = window_size
        self.name = name
        self.channelwise = channelwise
        self.timeout = timeout
        self._bits = bits
        self.callback = callback
        self.interval = interval
        self._collapse = collapse
        self.decimal_range = decimal_range
        self.saturate_range = saturate_range

        for k in ["decimal", "_n_updates", "bits", "_quantized"]:
            self.register_parameter(
                k,
                nn.Parameter(
                    torch.tensor(-1, dtype=torch.int), requires_grad=False
                ),  # placeholder
            )

    @property
    def initted(self) -> bool:
        """whether the parameters of the quantize layer are initialized."""
        return self._n_updates.item() != -1

    def forward(self, x):
        """Quantize input tensor according to given configuration.

        Args:
            x (torch.Tensor): tensor to be quantized

        Returns:
            torch.Tensor: quantized tensor
        """
        if not self.initted:
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
            self._quantized = nn.Parameter(
                torch.tensor([False], dtype=torch.bool).to(x.device),
                requires_grad=False,
            )

        if (
            (self.timeout - self.window_size) < self._n_updates <= self.timeout
        ):  # only collecting when needed to speedup
            if self._collapse >= 0:
                for t in (
                    x[nd_slice(len(x.shape), self._collapse, end=self.window_size)]
                    .detach()
                    .split(1)
                ):  # type: torch.Tensor
                    self.window.append(t.to("cpu"))
            else:
                self.window.append(x.detach().to("cpu"))
        else:
            if self.interval <= 0:
                self.window.clear()

        # add window size check to avoid quantize a layer which always set to eval
        if (self.training or (not self._quantized)) and (len(self.window) > 0):
            if (self._n_updates == self.timeout and not self._quantized) or (
                self._n_updates > self.timeout
                and ((self._n_updates - self.timeout) % self.interval) == 0
                and self.interval > 0
            ):
                if len(self.window) < self.window_size:
                    warnings.warn(
                        f"window is not full when quantization, this will cause performance degradation! (window has {len(self.window)} elements while window_size parameter is {self.window_size})"
                    )
                if self.channelwise >= 0:
                    for i in range(x.shape[self.channelwise]):
                        n = arg_decimal_min_mse(
                            torch.cat(
                                [
                                    a[
                                        tuple(
                                            [
                                                slice(0, cs)
                                                if ci != self.channelwise
                                                else i
                                                for ci, cs in enumerate(x.shape)
                                            ]
                                        )
                                    ].reshape(-1)
                                    for a in self.window
                                ],
                                dim=0,
                            ),
                            self.bits,
                            self.decimal_range,
                            self.saturate_range,
                            self.callback,
                        )
                        self.decimal.data[i] = n
                    if get_option("log_during_train"):
                        logging.info(
                            f"[Quantize{self.name if self.name == '' else f' @ {self.name}'}] (channelwise) avg decimal = {self.decimal.float().mean().item()}"
                        )
                else:
                    n = arg_decimal_min_mse(
                        torch.cat([a.reshape(-1) for a in self.window], dim=0),
                        self.bits,
                        self.decimal_range,
                        self.saturate_range,
                        self.callback,
                    )
                    if get_option("log_during_train"):
                        logging.info(
                            f"[Quantize{self.name if self.name == '' else f' @ {self.name}'}] decimal = {n}"
                        )
                    self.decimal.data[:] = n

                self._quantized[0] = True

                # proactively free up memory
                if self.interval <= 0:
                    self.window.clear()

        if self._quantized:
            out = self.callback(x, self.bits, self.decimal, self.channelwise)
        else:
            out = x

        if self.training:
            self._n_updates += 1
        return out


def quantize(
    inp: nn.Module = None,
    bits: int = 8,
    channelwise: int = 1,
    decimal_range: Tuple[int, int] = (0, 20),
    saturate_range: Tuple[float, float] = (0, 1),
    # for step-wise training
    timeout: int = 1000,
    interval: int = -1,
    window_size: int = 1,
    # for customization
    callback: QuantizeCallback = linear_quantize_callback,
    # for bias quantization, default to -1 is to not quantize bias
    bias_bits: int = -1,
    # for debug purpose
    name: str = "",
) -> nn.Module:
    """Creates a [QuantizeLayer][qsparse.quantize.QuantizeLayer] which is
    usually used for feature quantization if no input module is provided, or
    creates a weight-quantized version of the input module.

    Args:
        inp (nn.Module, optional): input module whose weight is to be quantized. Defaults to None.
        bits (int, optional): bitwidth for weight. Defaults to 8.
        channelwise (int, optional): dimension index for channel. Defaults to 1. When channelwise >= 0, channel-wise quantization is enabled. When set to -1, channel-wise quantization is disabled.
        decimal_range (Tuple[int, int], optional): search range of decimal bits. Defaults to (0, 20).
        saturate_range (Tuple[float, float], optional): quantiles used to clamp tensors before searching decimal bits. Defaults to (0, 1).
        timeout (int, optional): the steps to compute the best decimal bits. Defaults to 1000.
        interval (int, optional): interval of steps before each time to compute the best decimal bits. Defaults to -1, means only calculating the decimal bits once.
        window_size (int, optional): number of tensors used for computing the decimal bits. Defaults to 1.
        callback (QuantizeCallback, optional):  callback for actual operation of quantizing tensor, used for customization. Defaults to [linear\_quantize\_callback][qsparse.quantize.linear_quantize_callback].
        bias_bits (int, optional): bitwidth for bias. Defaults to -1, means not quantizing bias.
        name (str, optional): name of the quantize layer created, used for better logging. Defaults to "".

    Returns:
        nn.Module: input module with its weight quantized or a instance of [QuantizeLayer][qsparse.quantize.QuantizeLayer] for feature quantization
    """

    kwargs = dict(
        bits=bits,
        channelwise=channelwise,
        decimal_range=decimal_range,
        saturate_range=saturate_range,
        timeout=timeout,
        interval=interval,
        window_size=window_size,
        callback=callback,
        bias_bits=bias_bits,
        name=name,
    )

    def get_quantize_layer(feature_collapse=0, is_bias=False):
        if bias_bits == -1 and is_bias:
            return lambda a: a
        else:
            return QuantizeLayer(
                bits=bias_bits if is_bias else bits,
                channelwise=0 if is_bias else channelwise,
                decimal_range=decimal_range,
                saturate_range=saturate_range,
                timeout=int(timeout),
                interval=int(interval),
                window_size=window_size,
                callback=callback,
                name=name,
                collapse=feature_collapse,
            )

    if inp is None:
        layer = get_quantize_layer()
        setattr(layer, "_kwargs", kwargs)
        return layer
    elif isinstance(inp, nn.Module):
        return imitate(
            inp,
            "quantize",
            get_quantize_layer(-1),
            get_quantize_layer(-1, is_bias=True),
        )
    else:
        raise ValueError(f"{inp} is not a valid argument for quantize")


if __name__ == "__main__":
    layer = quantize(timeout=0)
    print(layer)
    print(quantize(torch.nn.Conv2d(10, 30, 3)))

    data = torch.rand(10, 10)
    print(data)
    print(layer(data))
