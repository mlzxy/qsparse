# fmt: off
import gc
import math
import warnings
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import optimize

from qsparse.common import (QuantizeCallback, QuantizeOptimizer, TensorOrFloat,
                            TensorOrInt, ensure_tensor)
from qsparse.imitation import imitate
from qsparse.util import get_option, logging, nd_slice

# fmt: on


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
        use_uint: bool = False,
        backward_passthrough: bool = False,
        flip_axis: bool = False,
    ):
        """quantize the input tensor and prepare for backward computation."""
        ctx.backward_passthrough = backward_passthrough
        ctx.notch = 1 if flip_axis else 0
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
        if use_uint:
            q.clamp_(0, 2 * limit - 1)
        else:
            q.clamp_(
                -limit + ctx.notch,
                limit - 1 + ctx.notch,
            )
        return q.float() * tof

    @staticmethod
    def backward(ctx, grad_output):
        """gradient computation for quantization operation."""
        limit, tof = ctx.saved_tensors
        if ctx.backward_passthrough:
            v = grad_output
        else:
            v = grad_output.clamp_(
                (-limit + ctx.notch) * tof,
                (limit - 1 + ctx.notch) * tof,
            )
        return (v,) + (None,) * 6


def linear_quantize_callback(
    inp: torch.Tensor,
    bits: int = 8,
    decimal: TensorOrInt = 5,
    channel_index: int = 1,
    use_uint: bool = False,
    backward_passthrough: bool = False,
    flip_axis: bool = False,
) -> torch.Tensor:
    """shift-based quantization function with type signature of [QuantizeCallback][qsparse.common.QuantizeCallback].

    Args:
        inp (torch.Tensor): input tensor
        bits (int, optional): bitwidth. Defaults to 8.
        decimal (TensorOrInt, optional): decimal bits, will be tensor of decimal bits for channel-wise quantization. Defaults to 5.
        channel_index (int, optional): dimension index for channel. Defaults to 1.
        use_uint (bool, optional): whether to use uint to quantize (useful for ReLu activations). Defaults to False.
        backward_passthrough (bool, optional): whether to just use `identity` function for backward pass. Defaults to False.
        flip_axis (bool, optional): whether to quantize positive values with negative axis and vice versa. Examples: normal int8 quantization will cast input to `[-128, 127]`, but with axis flipped, the input will be mapped to `[-127, 128]`.  Defaults to False.

    Returns:
        torch.Tensor: quantized tensor
    """
    return LinearQuantization.apply(
        inp, bits, decimal, channel_index, use_uint, backward_passthrough, flip_axis
    )


class DecimalOptimizer:
    """calculate the best fractional bits for given tensor."""

    def __init__(
        self,
        decimal_range: Tuple[int, int] = (0, 20),
        saturate_range: Tuple[float, float] = (0, 1),
    ):
        """
        Args:
            decimal_range (Tuple[int, int], optional): search range of fractional bits. Defaults to (0, 20).
            saturate_range (Tuple[float, float], optional): quantiles used to clamp the input tensor before searching decimal bits. Defaults to (0, 1).
        """
        assert len(decimal_range) == 2
        assert (
            0 <= saturate_range[0] <= saturate_range[1] <= 1
        ), f"illegal saturate_range {saturate_range}"
        self.decimal_range = decimal_range
        self.saturate_range = saturate_range

    def __call__(
        self,
        tensor: torch.Tensor,
        bits: int,
        init: int,
        quantize_callback: QuantizeCallback,
    ) -> int:
        """calculate the best fractional bits for given tensor.

        Args:
            tensor (torch.Tensor): input tensor
            bits (int): bitwidth
            init (Union[float, int]): initial quantization weight
            quantize_callback (QuantizeCallback, optional): callback for actual operation of quantizing tensor. Defaults to [linear\_quantize\_callback][qsparse.quantize.linear_quantize_callback].

        Returns:
            int: fractional bits (decimal point)
        """
        with torch.no_grad():
            err = float("inf")
            best_n = None
            tensor = tensor.reshape(-1)  # flatten
            if self.saturate_range != (0, 1):
                tensor = torch.clamp(
                    tensor,
                    approx_quantile(tensor, self.saturate_range[0]),
                    approx_quantile(tensor, self.saturate_range[1]),
                )
            for n in range(*self.decimal_range):
                tensor_q = quantize_callback(tensor, bits, decimal=n)
                err_ = torch.sum((tensor - tensor_q) ** 2).item()
                if err_ < err:
                    best_n = n
                    err = err_
            return best_n


class ScalerQuantization(torch.autograd.Function):
    """Straight-Through Gradient Estimator (with scaler).

    Please look for detailed description on arguments in [scaler\_quantize\_callback][qsparse.quantize.scaler_quantize_callback].
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        bits: int = 8,
        scaler: TensorOrFloat = 0.1,
        channel_index: int = 1,
        use_uint: bool = False,
        backward_passthrough: bool = False,
        flip_axis: bool = False,
    ):
        """quantize the input tensor and prepare for backward computation."""
        ctx.backward_passthrough = backward_passthrough
        ctx.notch = 1 if flip_axis else 0
        limit = 2.0 ** (bits - 1)
        shape = [1 for _ in input.shape]
        if isinstance(scaler, torch.Tensor) and sum(scaler.shape) > 1:
            assert (
                len(scaler) == input.shape[channel_index]
            ), "channel of input and decimal must be equal in channel-wise quantization"
            shape[channel_index] = -1
            scaler = scaler.view(*shape)
        ctx.save_for_backward(ensure_tensor(limit), ensure_tensor(scaler))
        q = (input / scaler).round().int()
        if use_uint:
            q.clamp_(0, 2 * limit - 1)
        else:
            q.clamp_(
                -limit + ctx.notch,
                limit - 1 + ctx.notch,
            )
        return q.float() * scaler

    @staticmethod
    def backward(ctx, grad_output):
        """gradient computation for quantization operation."""
        limit, scaler = ctx.saved_tensors
        if ctx.backward_passthrough:
            v = grad_output
        else:
            v = grad_output.clamp_(
                (-limit + ctx.notch) * scaler,
                (limit - 1 + ctx.notch) * scaler,
            )
        return (v,) + (None,) * 6


def scaler_quantize_callback(
    inp: torch.Tensor,
    bits: int = 8,
    scaler: TensorOrFloat = 0.1,
    channel_index: int = 1,
    use_uint: bool = False,
    backward_passthrough: bool = False,
    flip_axis: bool = False,
) -> torch.Tensor:
    """scaler-based quantization function with type signature of [QuantizeCallback][qsparse.common.QuantizeCallback].

    Args:
        inp (torch.Tensor): input tensor
        bits (int, optional): bitwidth. Defaults to 8.
        scaler (TensorOrFloat, optional): quantization scaler, will be tensor of scalers for channel-wise quantization. Defaults to 0.1.
        channel_index (int, optional): dimension index for channel. Defaults to 1.
        use_uint (bool, optional): whether to use uint to quantize (useful for ReLu activations). Defaults to False.
        backward_passthrough (bool, optional): whether to just use `identity` function for backward pass. Defaults to False.
        flip_axis (bool, optional): whether to quantize positive values with negative axis and vice versa. Examples: normal int8 quantization will cast input to `[-128, 127]`, but with axis flipped, the input will be mapped to `[-127, 128]`.  Defaults to False.

    Returns:
        torch.Tensor: quantized tensor
    """
    return ScalerQuantization.apply(
        inp, bits, scaler, channel_index, use_uint, backward_passthrough, flip_axis
    )


class ScalerOptimizer:
    """calculate the best quantization scaler for given tensor."""

    def __init__(
        self,
        saturate_range: Tuple[float, float] = (0, 1),
    ):
        """
        Args:
            saturate_range (Tuple[float, float], optional): quantiles used to clamp the input tensor before searching decimal bits. Defaults to (0, 1).
        """
        assert (
            0 <= saturate_range[0] <= saturate_range[1] <= 1
        ), f"illegal saturate_range {saturate_range}"
        self.saturate_range = saturate_range

    def __call__(
        self,
        tensor: torch.Tensor,
        bits: int,
        init: float,
        quantize_callback: QuantizeCallback,
    ) -> float:
        """calculate the best quantization scaler for given tensor.

        Args:
            tensor (torch.Tensor): input tensor
            bits (int): bitwidth
            init (Union[float, int]): initial quantization scaler
            quantize_callback (QuantizeCallback, optional): callback for actual operation of quantizing tensor. Defaults to [scaler\_quantize\_callback][qsparse.quantize.scaler_quantize_callback].

        Returns:
            float: quantization scaler
        """
        with torch.no_grad():
            tensor = tensor.reshape(-1)  # flatten
            if self.saturate_range != (0, 1):
                tensor = torch.clamp(
                    tensor,
                    approx_quantile(tensor, self.saturate_range[0]),
                    approx_quantile(tensor, self.saturate_range[1]),
                )

            tensor = tensor.detach()
            if init == 0:
                init = tensor.abs().mean().item()

            x0 = np.array(init)

            def func(x):
                tensor_q = quantize_callback(tensor, bits, float(x))
                return torch.mean((tensor - tensor_q) ** 2).item()

            result = optimize.minimize(func, x0, method="Nelder-Mead")
            best = abs(float(result.x))
            if best == 0:
                logging.warning("Encounter zero scaler, using 1e-4 instead")
                best = 1e-4
            return best


class QuantizeLayer(nn.Module):
    """Applies quantization over input tensor.

    Please look for detailed description in [quantize][qsparse.quantize.quantize]
    """

    def __init__(
        self,
        bits: int = 8,
        channelwise: int = 1,
        # for step-wise training
        timeout: int = 1000,
        interval: int = -1,
        window_size: int = 1,
        on_device_window: bool = False,
        # for customization
        optimizer: QuantizeOptimizer = DecimalOptimizer(),
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
        self.on_device_window = on_device_window
        self.name = name
        self.channelwise = channelwise
        self.timeout = timeout
        self._bits = bits
        self.callback = callback
        self.optimizer = optimizer
        self.interval = interval
        self._collapse = collapse

        for k in ["weight", "_n_updates", "bits", "_quantized"]:
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

    def _to_win_dev(self, tensor: torch.Tensor):
        if self.on_device_window:
            return tensor
        else:
            return tensor.to("cpu")

    def forward(self, x):
        """Quantize input tensor according to given configuration.

        Args:
            x (torch.Tensor): tensor to be quantized

        Returns:
            torch.Tensor: quantized tensor
        """
        if not self.initted:
            self.weight = nn.Parameter(
                torch.zeros(
                    1 if self.channelwise < 0 else x.shape[self.channelwise]
                ).to(x.device),
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

        def time_to_next_quantization(offset=0):
            t = self._n_updates.item() + offset
            post_q_time_delta = t - self.timeout
            if t <= self.timeout:
                return self.timeout - t
            else:
                if self.interval > 0:
                    remaining_fraction = (
                        math.ceil(post_q_time_delta / self.interval)
                        - post_q_time_delta / self.interval
                    )
                    return int(self.interval * remaining_fraction)
                else:
                    return float("inf")

        batch_size = len(x)
        if time_to_next_quantization() <= math.ceil(
            self.window_size / batch_size
        ):  # to speedup by only collecting when needed
            if self._collapse >= 0:
                for t in (
                    x[nd_slice(len(x.shape), self._collapse, end=self.window_size)]
                    .detach()
                    .split(1)
                ):  # type: torch.Tensor
                    self.window.append(self._to_win_dev(t))
            else:
                self.window.append(self._to_win_dev(x.detach()))
        else:
            self.window.clear()

        # add window size check to avoid quantize a layer which always set to eval
        if (self.training or (not self._quantized)) and (len(self.window) > 0):
            if time_to_next_quantization() == 0:
                if len(self.window) < self.window_size:
                    warnings.warning(
                        f"window is not full when quantization, this will cause performance degradation! (window has {len(self.window)} elements while window_size parameter is {self.window_size})"
                    )
                if self.channelwise >= 0:
                    for i in range(x.shape[self.channelwise]):
                        sl = [
                            slice(0, cs) if ci != self.channelwise else i
                            for ci, cs in enumerate(x.shape)
                        ]
                        n = self.optimizer(
                            torch.cat(
                                [a[sl].reshape(-1) for a in self.window],
                                dim=0,
                            ),
                            self.bits,
                            self.weight[i].item(),
                            self.callback,
                        )
                        self.weight.data[i] = n
                    if get_option("log_during_train"):
                        logging.info(
                            f"[Quantize{self.name if self.name == '' else f' @ {self.name}'}] (channelwise) avg quantization weight = {self.weight.float().mean().item()}"
                        )
                else:
                    n = self.optimizer(
                        torch.cat([a.reshape(-1) for a in self.window], dim=0),
                        self.bits,
                        self.weight.item(),
                        self.callback,
                    )
                    if get_option("log_during_train"):
                        logging.info(
                            f"[Quantize{self.name if self.name == '' else f' @ {self.name}'}] quantization weight = {n}"
                        )
                    self.weight.data[:] = n

                self._quantized[0] = True
                self.window.clear()
                gc.collect()
                torch.cuda.empty_cache()

        if self._quantized:
            out = self.callback(x, self.bits, self.weight, self.channelwise)
        else:
            out = x

        if self.training:
            self._n_updates += 1
        return out


def quantize(
    inp: nn.Module = None,
    bits: int = 8,
    channelwise: int = 1,
    # for step-wise training
    timeout: int = 1000,
    interval: int = -1,
    window_size: int = 1,
    on_device_window: bool = False,
    # for customization
    optimizer: QuantizeOptimizer = None,
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
        timeout (int, optional): the steps to compute the best decimal bits. Defaults to 1000.
        interval (int, optional): interval of steps before each time to compute the best decimal bits. Defaults to -1, means only calculating the decimal bits once.
        window_size (int, optional): number of tensors used for computing the decimal bits. Defaults to 1.
        on_device_window (bool, optional): whether keep the tensor window on gpu device being used, or move to cpu. Default to False, means moving to cpu.
        optimizer (QuantizeOptimizer, optional): optimizer used to compute the best quantization weight. Defaults to `DecimalOptimizer()`.
        callback (QuantizeCallback, optional):  callback for actual operation of quantizing tensor, used for customization. Defaults to [linear\_quantize\_callback][qsparse.quantize.linear_quantize_callback].
        bias_bits (int, optional): bitwidth for bias. Defaults to -1, means not quantizing bias.
        name (str, optional): name of the quantize layer created, used for better logging. Defaults to "".

    Returns:
        nn.Module: input module with its weight quantized or a instance of [QuantizeLayer][qsparse.quantize.QuantizeLayer] for feature quantization
    """

    kwargs = dict(
        bits=bits,
        channelwise=channelwise,
        optimizer=optimizer,
        timeout=timeout,
        interval=interval,
        window_size=window_size,
        callback=callback,
        bias_bits=bias_bits,
        name=name,
        on_device_window=on_device_window,
    )

    optimizer = optimizer or DecimalOptimizer()

    def get_quantize_layer(feature_collapse=0, is_bias=False):
        if bias_bits == -1 and is_bias:
            return lambda a: a
        else:
            return QuantizeLayer(
                bits=bias_bits if is_bias else bits,
                channelwise=(0 if channelwise >= 0 else -1) if is_bias else channelwise,
                optimizer=optimizer,
                timeout=int(timeout),
                interval=int(interval),
                window_size=window_size,
                on_device_window=on_device_window,
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
