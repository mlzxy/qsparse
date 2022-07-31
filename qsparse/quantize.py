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
        tof = 2.0**-decimal
        toi = 2.0**decimal
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


class LineQuantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, bits: int = 8, lines=(-0.1, 0.9)):
        N = 2**bits
        x = torch.clamp(x, lines[0], lines[1])
        step = (lines[1] - lines[0]) / (2**N)
        qa = ((x - lines[0]) / step).round() * step + lines[0]
        return qa

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output,) + (None,) * 2


class BaseQuantizer(nn.Module):
    weight_size = 1


class DecimalQuantizer(BaseQuantizer):
    weight_size = 1

    def __init__(
        self,
        use_uint: bool = False,
        backward_passthrough: bool = False,
        flip_axis: bool = False,
    ):
        super().__init__()
        self.use_uint = use_uint
        self.backward_passthrough = backward_passthrough
        self.flip_axis = flip_axis
        self.function = LinearQuantization.apply

    def quantize(self, tensor, bits, decimal):
        return self.function(
            tensor,
            bits,
            decimal,
            -1,
            self.use_uint,
            self.backward_passthrough,
            self.flip_axis,
        )

    def forward(self, tensor, bits, decimal, batch_dim=-1):
        if decimal == 0:
            with torch.no_grad():
                err = float("inf")
                best_n = None
                for n in range(0, 20):
                    tensor_q = self.quantize(tensor, bits, decimal)
                    err_ = torch.sum((tensor - tensor_q) ** 2).item()
                    if err_ < err:
                        best_n = n
                        err = err_
            if isinstance(decimal, torch.Tensor):
                decimal.data[:] = best_n
        return self.quantize(tensor, bits, decimal)


class ScalerQuantizer(DecimalQuantizer):
    weight_size = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.function = ScalerQuantization.apply

    def forward(self, tensor, bits, scaler, batch_dim=-1):
        if scaler == 0:
            with torch.no_grad():
                init = tensor.abs().mean().item()
                x0 = np.array(init)

                def func(x):
                    tensor_q = self.quantize(tensor, bits, float(x))
                    return torch.mean((tensor - tensor_q) ** 2).item()

                result = optimize.minimize(func, x0, method="Nelder-Mead")
                best = abs(float(result.x))
                scaler.data[:] = best
        return self.quantize(tensor, bits, scaler)


# today just test this
class AdaptiveLineQuantizer(nn.Module):
    weight_size = 2

    def __init__(self, alpha=0.1, outlier_ratio=0.0001):
        super().__init__()
        self.alpha = alpha
        self.outlier_ratio = outlier_ratio

    def estimate_bound(self, bound, lower_bound):
        dist = (bound - lower_bound).abs()
        flag = ((dist / bound) < 1.1) | (dist < 0.2)
        v = torch.cat([bound.view(-1, 1), lower_bound.view(-1, 1)], dim=1)
        return v[torch.cat([~flag.view(-1, 1), flag.view(-1, 1)], dim=1)]

    def forward(self, tensor, bits, scaler, batch_dim=-1):
        origin_shape = tuple(tensor.shape)
        tensor = tensor.view(1 if batch_dim == -1 else origin_shape[0], -1)
        if self.training:
            lb = self.estimate_bound(
                tensor.quantile(self.outlier_ratio, dim=1), tensor.min(dim=1)
            )
            ub = self.estimate_bound(
                tensor.quantile(1 - self.outlier_ratio, dim=1), tensor.max(dim=1)
            )
            scaler.data[0] = scaler.data[0] * (1 - self.alpha) + lb.mean() * self.alpha
            scaler.data[1] = scaler.data[1] * (1 - self.alpha) + ub.mean() * self.alpha
            lines = torch.cat([lb.view(1, -1), ub.view(1, -1)], dim=0)
        else:
            lines = scaler
        result = LineQuantization.apply(tensor, bits, lines)
        return result.view(origin_shape)


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
        # for customization
        callback: BaseQuantizer = None,
        # for debug purpose
        collapse: int = 0,
        name: str = "",
    ):
        super().__init__()
        if get_option("log_on_created"):
            logging.info(
                f"[Quantize{name if name == '' else f' @ {name}'}] bits={bits} channelwise={channelwise}  timeout={timeout}"
            )
        self.name = name
        self.channelwise = channelwise
        self.timeout = timeout
        self._bits = bits
        self.callback = callback  # type: BaseQuantizer
        self._batch_dim = collapse

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
            self.weight = nn.Parameter(
                torch.zeros(
                    1 if self.channelwise < 0 else len(x.shape[self.channelwise]),
                    self.callback.weight_dimension,
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
        if self._n_updates.item() >= self.timeout:
            if self.channelwise >= 0:
                sl = [None] * len(x.shape)
                slice_shape = list(x.shape)
                slice_shape[self.channelwise] = 1
                out = []
                for i in range(len(x.shape)):
                    sl[self.channelwise] = i
                    out.append(
                        self.callback(
                            x[sl], self.weight[i], self.bits, batch_dim=self._batch_dim
                        ).view(slice_shape)
                    )
                out = torch.cat(out, dim=self.channelwise)
            else:
                out = self.callback(
                    x, self.bits, self.weight[0], batch_dim=self._batch_dim
                )
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
    # for customization
    callback: BaseQuantizer = None,
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
    callback = callback or DecimalQuantizer()

    kwargs = dict(
        bits=bits,
        channelwise=channelwise,
        timeout=timeout,
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
                channelwise=(0 if channelwise >= 0 else -1) if is_bias else channelwise,
                timeout=int(timeout),
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
