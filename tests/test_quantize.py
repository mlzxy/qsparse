# fmt: off
import numpy as np
import pytest
import torch
import torch.autograd as autograd
import torch.nn.functional as F

from qsparse import quantize
from qsparse.quantize import quantize_with_decimal, quantize_with_scaler, quantize_with_line
from qsparse.quantize import DecimalQuantizer, ScalerQuantizer, AdaptiveQuantizer

# fmt: on


def test_feature():
    data = (torch.rand((1, 10, 32, 32)) - 0.5) * 4
    timeout = 5
    quantize_layer = quantize(bits=8, timeout=timeout, channelwise=-1, callback=DecimalQuantizer())
    for _ in range(timeout + 1):  # ensure the quantization has been triggered
        output = quantize_layer(data).numpy()

    output_ref = quantize_with_decimal(
        data, bits=8, 
        decimal=(1 / quantize_layer.weight).nan_to_num(posinf=1, neginf=1).log2().round(), 
        channel_index=-1
    ).numpy()
    assert np.all(output == output_ref)


def test_weight():
    timeout = 5
    data = torch.rand((1, 10, 32, 32))

    qconv = quantize(
        torch.nn.Conv2d(10, 30, 3),
        bits=8,
        bias_bits=8,
        timeout=timeout,
        callback=ScalerQuantizer(),
        channelwise=0
    )
    qconv.train()
    for _ in range(timeout + 1):
        qconv(data)

    assert (
        qconv.weight - quantize_with_scaler(qconv._parameters["weight"], 8, qconv.quantize.weight, channel_index=0)
    ).sum().item() == 0, "weight shall be fully quantized"
    assert (
        qconv.bias - quantize_with_scaler(
            qconv._parameters["bias"], 8, qconv.quantize_bias.weight, channel_index=0
        )
    ).sum().item() == 0, "bias shall be fully quantized"

    assert (
       qconv._parameters["weight"] - qconv.weight
    ).sum().item() != 0, (
        "parameter['weight'] shall store the untouched weight with full precision"
    )

    qconv = quantize(torch.nn.Conv2d(10, 30, 3), bits=8, timeout=timeout, channelwise=0, callback=ScalerQuantizer())
    qconv.eval()
    for _ in range(timeout * 2):
        qconv(data)
    assert (
        qconv.weight - quantize_with_scaler(qconv._parameters["weight"], 8, qconv.quantize.weight, channel_index=0)
    ).sum().item() != 0, "quantization schedule shall only be triggered during training"

    with pytest.raises(ValueError):  # shall only accept module input or no input
        quantize(torch.rand((10,)))


def test_integer_arithmetic():
    ni = 7
    no = 6
    input = torch.randint(-128, 127, size=(3, 10, 32, 32))
    input_float = input.float() / 2 ** ni

    timeout = 5
    # quantized output in 32-bit float
    qconv = quantize(
        torch.nn.Conv2d(10, 30, 3, bias=False), bits=8, timeout=timeout, channelwise=0, callback=DecimalQuantizer()
    )  # vector quantization on output channel
    qconv.train()
    for _ in range(timeout + 1):  # ensure the quantization has been triggered
        qconv(input_float)
    output_float = quantize_with_decimal(qconv(input_float), 8, no)

    # quantized output in 8-bit integer
    decimal = (1 / qconv.quantize.weight).nan_to_num(posinf=1, neginf=1).log2().round().int()
    weight = qconv.weight * (2.0 ** decimal).view(-1, 1, 1, 1)
    output_int = F.conv2d(input.int(), weight.int())
    for i in range(output_int.shape[1]):
        output_int[:, i] = (
            output_int[:, i].float() / 2 ** (ni + decimal[i] - no)
        ).int()

    diff = (
        output_float.detach().numpy() - (output_int.float() / 2 ** no).detach().numpy()
    )
    assert np.all(diff == 0), "shall be able to fully match with integer arithmetic"


def test_adaptive_quantization():
    data = (torch.rand((1, 10, 32, 32)) - 0.5) * 4
    timeout = 5
    quantize_layer = quantize(bits=8, timeout=timeout, channelwise=1, callback=AdaptiveQuantizer())
    for _ in range(timeout + 1):  # ensure the quantization has been triggered
        output = quantize_layer(data).numpy()

    output_ref = quantize_with_line(
        data, bits=8, 
        lines=quantize_layer.weight,
        channel_index=1
    ).numpy()
    assert np.all(output == output_ref)

    quantize_layer.eval()
    output = quantize_layer(data).numpy()
    output_ref = quantize_with_line(
        data, bits=8, 
        lines=quantize_layer.weight,
        channel_index=1,
        float_zero_point=False
    ).numpy()
    assert np.all(output == output_ref)



def test_groupwise_quantization():
    data = (torch.rand((64, 10, 32, 32)) - 0.5) * 4
    timeout = 5
    quantize_layer = quantize(bits=8, timeout=timeout, channelwise=1, callback=AdaptiveQuantizer(group_num=4, group_timeout=20))
    for _ in range(timeout + 30):  # ensure the quantization has been triggered
        output = quantize_layer(data).numpy()
        
    group_weight = torch.clone(quantize_layer.weight)
    for ci in range(4):
        ind = quantize_layer.callback.groups == ci
        avg = group_weight[ind].mean(dim=0)
        group_weight[ind] = avg

    output_ref = quantize_with_line(
        data, bits=8, 
        lines=group_weight,
        channel_index=1
    ).numpy()
    assert np.all(output == output_ref)
