# fmt: off
import gc
import os.path as osp
import math
import warnings
from collections import deque
from typing import Tuple, Union, List
import warnings
import numpy as np
import torch.nn.functional as F
import torch
from sklearn.cluster import AgglomerativeClustering
import torch.nn as nn
from scipy import optimize

from qsparse.common import (TensorOrFloat, TensorOrInt, ensure_tensor)
from qsparse.imitation import imitate
from qsparse.util import get_option, logging

# fmt: on



class DecimalQuantization(torch.autograd.Function):
    """Straight-Through Gradient Estimator (with shift).

    Please look for detailed description on arguments in [quantize\_with\_decimal][qsparse.quantize.quantize_with_decimal].
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
        if isinstance(decimal, torch.Tensor) and decimal.numel() > 1:
            assert (
                len(decimal) == input.shape[channel_index]
            ), "channel of input and decimal must be equal in channel-wise quantization"
            shape[channel_index] = -1
            tof, toi = tof.view(*shape), toi.view(*shape)
        ctx.save_for_backward(ensure_tensor(limit), ensure_tensor(tof))
        q = (input * toi).int()
        if use_uint:
            q.float().clamp_(0, 2 * limit - 1)
        else:
            q.float().clamp_(
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
            v[v != grad_output] = 0  # reset the clampped values to 0
        return (v,) + (None,) * 6


class ScalerQuantization(torch.autograd.Function):
    """Straight-Through Gradient Estimator (with scaler).

    Please look for detailed description on arguments in [quantize\_with\_scaler][qsparse.quantize.quantize_with_scaler].
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
        if isinstance(scaler, torch.Tensor) and math.prod(scaler.shape) > 1:
            assert (
                len(scaler) == input.shape[channel_index]
            ), "channel of input and decimal must be equal in channel-wise quantization"
            shape[channel_index] = -1
            scaler = scaler.view(*shape)
        ctx.save_for_backward(ensure_tensor(limit), ensure_tensor(scaler))
        q = (input / scaler).round().int()
        if use_uint:
            q.float().clamp_(0, 2 * limit - 1)
        else:
            q.float().clamp_(
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
            v[v != grad_output] = 0  # reset the clampped values to 0
        return (v,) + (None,) * 6


class LineQuantization(torch.autograd.Function):
    """Straight-Through Gradient Estimator (asymmetric).

    Please look for detailed description on arguments in [quantize\_with\_line][qsparse.quantize.quantize_with_line].
    """
    
    @staticmethod
    def forward(ctx, 
                x: torch.Tensor, 
                bits: int = 8, 
                lines=(-0.1, 0.9), 
                channel_index=-1, 
                inplace=False, 
                float_zero_point=True):
        with torch.no_grad():
            N = 2**bits
            shape = [1] * len(x.shape)
            if not isinstance(lines, torch.Tensor):
                lines = torch.tensor(lines).view(-1, 2).to(x.device)
            if channel_index >= 0:
                shape[channel_index] = -1
                assert x.shape[channel_index] == lines.shape[0]
            assert lines.shape[1] == 2
            start, end = lines[:, 0].view(shape), lines[:, 1].view(shape)
            x = torch.clamp(x, start, end)
            step = (end - start) / N
            step[step == 0] = 0.0001
            if not float_zero_point:
                qa = (x / step).round()
                qstart = (start / step).round()
                qa = (qa - qstart).clamp(0, N-1)
                qa = (qa + qstart) * step
                return qa
            else:
                if inplace:
                    x = x - start
                    x /= step
                    x = x.round_().clamp_(0, N - 1)
                    x = x * step
                    x += start
                    return x
                else:
                    qa = x - start
                    qa /= step
                    qa = qa.round_().clamp_(0, N - 1)
                    qa = qa * step
                    qa += start
                    return qa

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output,) + (None,) * 5


def quantize_with_decimal(
        input: torch.Tensor,
        bits: int = 8,
        decimal: TensorOrInt = 5,
        channel_index: int = -1,
        use_uint: bool = False,
        backward_passthrough: bool = False,
        flip_axis: bool = False) -> torch.Tensor:
    """Applying power-of-2 uniform quantization over input tensor

    Args:
        input (torch.Tensor): tensor to be quantized
        bits (int, optional): Bitwidth. Defaults to 8.
        decimal (TensorOrInt, optional): Number of bits used to represent fractional number (shift). Defaults to 5.
        channel_index (int, optional): Channel axis, for channelwise quantization. Defaults to -1, which means tensorwise.
        use_uint (bool, optional): Whether use uint to quantize input. If so, it will ignores the negative number, which could be used for ReLU output. Defaults to False.
        backward_passthrough (bool, optional): Whether to skip the saturation operation of STE on gradients during the backward pass. Defaults to False.
        flip_axis (bool, optional): Whether use flip the axis to represent numbers (the largest positive number increases from `2^{d}-1` to `2^{d}`, while the smallest negative number reduces its absolute value). Defaults to False.

    Returns:
        torch.Tensor: quantized tensor
    """
    return DecimalQuantization.apply(input, bits, decimal, channel_index, use_uint, backward_passthrough, flip_axis)

def quantize_with_scaler(
        input: torch.Tensor,
        bits: int = 8,
        scaler: TensorOrFloat = 0.1,
        channel_index: int = -1,
        use_uint: bool = False,
        backward_passthrough: bool = False,
        flip_axis: bool = False) -> torch.Tensor:
    """Applying scaling-factor based uniform quantization over input tensor

    Args:
        input (torch.Tensor): tensor to be quantized
        bits (int, optional): Bitwidth. Defaults to 8.
        scaler (TensorOrFloat, optional): Scaling factor. Defaults to 0.1.
        channel_index (int, optional): Channel axis, for channelwise quantization. Defaults to -1, which means tensorwise.
        use_uint (bool, optional): Whether use uint to quantize input. If so, it will ignores the negative number, which could be used for ReLU output. Defaults to False.
        backward_passthrough (bool, optional): Whether to skip the saturation operation of STE on gradients during the backward pass. Defaults to False.
        flip_axis (bool, optional): Whether use flip the axis to represent numbers (the largest positive number increases from `2^{d}-1` to `2^{d}`, while the smallest negative number reduces its absolute value). Defaults to False.

    Returns:
        torch.Tensor: quantized tensor
    """
    return ScalerQuantization.apply(input, bits, scaler, channel_index, use_uint, backward_passthrough, flip_axis)

def quantize_with_line(x: torch.Tensor, 
                bits: int = 8, 
                lines: Union[Tuple[float, float], List[Tuple[float, float]]]=(-0.1, 0.9), 
                channel_index: int=-1, 
                inplace: bool=False, 
                float_zero_point: bool=True) -> torch.Tensor:
    """Applying asymmetric uniform quantization over input tensor

    Args:
        x (torch.Tensor): tensor to be quantized
        bits (int, optional): Bitwidth. Defaults to 8.
        lines (Union[Tuple[float, float], List[Tuple[float, float]]], optional): The estimated lower and upper bound of input data. Defaults to (-0.1, 0.9).
        channel_index (int, optional): Channel axis, for channelwise quantization. Defaults to -1, which means tensorwise.
        inplace (bool, optional): Whether the operation is inplace. Defaults to False.
        float_zero_point (bool, optional): Whether use floating-point value to store zero-point. Defaults to True, recommend to turn on for training and off for evaluation. 

    Returns:
        torch.Tensor: quantized tensor
    """
    return LineQuantization.apply(x, bits, lines, channel_index, inplace, float_zero_point)


class BaseQuantizer(nn.Module):
    """Base class for quantizer, interface for the callback function of [quantize][qsparse.quantize.quantize].
    """
    weight_size = 1

    def optimize(self, tensor, bits, weight=None, batched=False, channel_index=-1) -> torch.Tensor:
        """return the updated weight for each step"""
        raise NotImplementedError

    def forward(self, tensor, bits, weight=None, batched=False, channel_index=-1) -> torch.Tensor:
        """return quantized tensor"""
        raise NotImplementedError

    def get_weight_shape(self, x, channelwise):
        return (1 if channelwise < 0 else x.shape[channelwise], self.weight_size)


class DecimalQuantizer(BaseQuantizer):
    """
    The quantizer that implements the algorithm 3 of the MDPI paper. The `forward` function covers the quantization logic and the `optimize` function covers the parameter update.

    It always restricts the scaling factor to be power of 2.
    """
    

    weight_size = 1

    def __init__(
        self,
        use_uint: bool = False,
        backward_passthrough: bool = False,
        flip_axis: bool = False,
        group_num=-1,
        group_timeout=512
    ):
        """
        Args:
            use_uint (bool, optional): See [quantize\_with\_decimal][qsparse.quantize.quantize_with_decimal]. Defaults to False.
            backward_passthrough (bool, optional): See [quantize\_with\_decimal][qsparse.quantize.quantize_with_decimal]. Defaults to False.
            flip_axis (bool, optional): See [quantize\_with\_decimal][qsparse.quantize.quantize_with_decimal]. Defaults to False.
            group_num (int, optional): Number of groups used for groupwise quantization. Defaults to -1, which disables groupwise quantization.
            group_timeout (int, optional): Number of steps when the clustering starts after the activation of the quantization operator. Defaults to 512.
        """
        super().__init__()
        self.use_uint = use_uint
        self.backward_passthrough = backward_passthrough
        self.flip_axis = flip_axis
        self.use_float_scaler = False
        self.function = DecimalQuantization.apply
        self.t = 0
        self.group_timeout = group_timeout
        self.groups = None
        self.group_num = group_num

    def quantize(self, tensor, bits, scaler, channel_index=-1, **kwargs):
        if self.use_float_scaler:
            weight = scaler
        else:
            weight = (1 / scaler).nan_to_num(posinf=1, neginf=1).log2().round()
        return self.function(
            tensor,
            bits,
            weight,
            channel_index,
            self.use_uint,
            self.backward_passthrough,
            self.flip_axis,
        )

    def optimize(self, x, bits, weight=None, batched=False, channel_index=-1, **kwargs):
        with torch.no_grad():
            x = x.abs()
            batch_size = x.shape[0] if batched else -1
            wshape = self.get_weight_shape(x, channel_index)
            if channel_index == -1:
                x = x.view(1, -1)
            elif channel_index != 0:
                num_channel = x.shape[channel_index]
                x = x.transpose(0, channel_index)
                x = x.contiguous().view(num_channel, -1)
            else:
                x = x.view(x.shape[0], -1)
            new_weight = x.max(dim=1).values / (2 ** (bits - 1))
            if batched:
                new_weight = new_weight.view(-1, batch_size).mean(dim=1)
            new_weight = new_weight.view(wshape)
        if self.t == 0:
            weight = new_weight
        else:
            weight.data[:] = (self.t * weight + new_weight) / (self.t + 1)
        self.t += 1
        return weight

    def forward(self, tensor, bits, scaler, channel_index=-1, **kwargs):
        if self.t >= self.group_timeout and self.group_num > 0 and scaler.numel() > self.group_num:
            if self.groups is None:
                logging.danger( f"clustering {len(scaler)} channels into {self.group_num} groups")
                clustering = AgglomerativeClustering(
                    n_clusters=self.group_num)
                clustering.fit(scaler.detach().cpu().numpy())
                self.groups = nn.Parameter(torch.from_numpy(
                    clustering.labels_).to(scaler.device), requires_grad=False)

            group_scaler = torch.clone(scaler)
            for ci in range(self.group_num):
                ind = self.groups == ci
                avg = group_scaler[ind].mean(dim=0)
                group_scaler[ind] = avg
            scaler = group_scaler
        return self.quantize(tensor, bits, scaler, channel_index, **kwargs)


class ScalerQuantizer(DecimalQuantizer):
    """The quantizer that implements the algorithm 3 of the MDPI paper, without the power of 2 restriction.
    """
    weight_size = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_float_scaler = True
        self.function = ScalerQuantization.apply


class AdaptiveQuantizer(DecimalQuantizer):
    """The quantizer that implements the algorithm 2 of the MDPI paper.
    """
    weight_size = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.function = LineQuantization.apply

    def quantize(self, tensor, bits, lines, channel_index=-1, **kwargs):
        return self.function(tensor, bits, lines, channel_index, kwargs.get("inplace", False), self.training)

    def optimize(self, x, bits, weight=None, channel_index=-1, batched=False, **kwargs):
        batch_size = x.shape[0]
        with torch.no_grad():
            if channel_index >= 0:
                if batched:
                    if channel_index != 1:
                        x = x.transpose(1, channel_index)
                    shape = tuple(x.shape)
                    x = x.view(-1, math.prod(shape[2:]))
                else:
                    if channel_index != 0:
                        x = x.transpose(0, channel_index)
                    shape = tuple(x.shape)
                    x = x.contiguous().view(-1, math.prod(shape[1:]))
            else:
                x = x.view(len(x) if batched else 1, -1)

            lb = x.min(dim=1).values
            ub = x.max(dim=1).values
            _buf = torch.cat([lb.view(-1, 1), ub.view(-1, 1)], dim=1)
            _lines = _buf[:len(lb), :]

            if batched:
                _lines = _lines.view(batch_size, -1, 2)
                sample_avg_lines = torch.cat([_lines[:, :, 0].min(
                    dim=0).values.view(-1, 1), _lines[:, :, 1].max(dim=0).values.view(-1, 1)], dim=1).view(-1, 2)
            else:
                sample_avg_lines = _lines

            if weight is None:
                self.t = nn.Parameter(torch.zeros(1).to(
                    x.device), requires_grad=False)
                self.t += 1
                return sample_avg_lines
            else:
                assert sample_avg_lines.shape == weight.shape
                self.t += 1
                return (weight * (self.t - 1) + sample_avg_lines) / self.t



class QuantizeLayer(nn.Module):
    """Applies quantization over input tensor.

    Please look for detailed description in [quantize][qsparse.quantize.quantize]
    """

    def __str__(self):
        return f"QuantizeLayer(bits={self.bits}, timeout={self.timeout}, callback={self.callback.__class__.__name__}, channelwise={self.channelwise})"

    def __repr__(self):
        return str(self)

    def __init__(
        self,
        bits: int = 8,
        channelwise: int = 1,
        timeout: int = 1000,
        callback: BaseQuantizer = None,
        batch_dimension: int = 0,
        name: str = "",
    ):
        super().__init__()
        if get_option("log_on_created"):
            logging.info(
                f"[Quantize{name if name == '' else f' @ {name}'}] bits={bits} channelwise={channelwise} timeout={timeout}"
            )
        self.name = name
        self.channelwise = channelwise
        self.timeout = timeout
        self.bits = bits
        self.callback = callback  # type: BaseQuantizer
        self.batch_dimension = batch_dimension # `batch_dimension == 0` means activation
        self._quantized = False

    @property
    def initted(self) -> bool:
        """whether the parameters of the quantize layer are initialized."""
        return hasattr(self, '_n_updates')

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
                    1 if self.channelwise < 0 else x.shape[self.channelwise],
                    self.callback.weight_size,
                ).to(x.device),
                requires_grad=False,
            )
            self._n_updates = nn.Parameter(
                torch.zeros(1, dtype=torch.int).to(x.device),
                requires_grad=False,
            )

        t = self._n_updates.item()
        if self.timeout > 0:
            if t >= self.timeout:
                if self.training:
                    if t == self.timeout:
                        logging.warn(f"quantizing {self.name} with {self.bits} bits")
                    new_weight = self.callback.optimize(x, self.bits, self.weight, 
                                                        batched=self.batch_dimension == 0, channel_index=self.channelwise)
                    if new_weight is not None:
                        self.weight.data[:] = new_weight
                    self._quantized = True
                if self._quantized:
                    out = self.callback(
                        x, self.bits, self.weight, channel_index=self.channelwise, inplace=self.batch_dimension == 0)
                else:
                    out = x
            else:
                out = x

            if self.training:
                self._n_updates += 1
        else:
            out = x
        return out


def quantize(
    inp: nn.Module = None,
    bits: int = 8,
    channelwise: int = 1,
    timeout: int = 1000,
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
        callback (BaseQuantizer, optional):  callback module for actual operation of quantizing tensor and finding quantization parameters. Defaults to [ScalerQuantizer][qsparse.quantize.ScalerQuantizer].
        bias_bits (int, optional): bitwidth for bias. Defaults to -1, means not quantizing bias.
        name (str, optional): name of the quantize layer created, used for better logging. Defaults to "".

    Returns:
        nn.Module: input module with its weight quantized or a instance of [QuantizeLayer][qsparse.quantize.QuantizeLayer] for feature quantization
    """
    callback = callback or ScalerQuantizer()

    kwargs = dict(
        bits=bits,
        channelwise=channelwise,
        timeout=timeout,
        callback=callback,
        bias_bits=bias_bits,
        name=name
    )

    def get_quantize_layer(batch_dimension=0, is_bias=False):
        if bias_bits == -1 and is_bias:
            return lambda a: a
        else:
            return QuantizeLayer(
                bits=bias_bits if is_bias else bits,
                channelwise=(0 if channelwise >= 0 else
                             - 1) if is_bias else channelwise,
                timeout=int(timeout),
                callback=callback,
                name=name,
                batch_dimension=batch_dimension
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
