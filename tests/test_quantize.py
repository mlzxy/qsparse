import numpy as np
import torch
import torch.nn.functional as F

from qsparse import linear_quantize_callback, quantize


def test_feature():
    # add saturation
    pass


def test_weight():
    timeout = 5
    data = torch.rand((1, 10, 32, 32))

    qconv = quantize(torch.nn.Conv2d(10, 30, 3), bits=8, bias_bits=8, timeout=timeout)
    qconv.train()
    for _ in range(timeout + 1):
        qconv(data)

    assert (
        qconv.weight.detach().numpy()
        - linear_quantize_callback(qconv.weight, 8, qconv.quantize.decimal)
        .detach()
        .numpy()
    ).sum() == 0, "weight shall be fully quantized"
    assert (
        qconv.bias.detach().numpy()
        - linear_quantize_callback(
            qconv.bias, 8, qconv.quantize_bias.decimal, channel_index=0
        )
        .detach()
        .numpy()
    ).sum() == 0, "bias shall be fully quantized"

    assert (
        dict(qconv.named_parameters())["weight"].detach().numpy()
        - qconv.weight.detach().numpy()
    ).sum() != 0, (
        "parameter['weight'] shall store the untouched weight with full precision"
    )

    qconv = quantize(torch.nn.Conv2d(10, 30, 3), bits=8, timeout=timeout)
    qconv.eval()
    for _ in range(timeout * 2):
        qconv(data)
    assert (
        qconv.weight.detach().numpy()
        - linear_quantize_callback(qconv.weight, 8, qconv.quantize.decimal)
        .detach()
        .numpy()
    ).sum() != 0, "quantization schedule shall only be triggered during training"


def test_callback():
    # scalar quantization
    data = torch.rand(10, 3, 32, 32)
    qdata = linear_quantize_callback(data, bits=8, decimal=7)
    assert np.allclose(data.numpy(), qdata.numpy(), atol=1 / 2 ** 7)

    # vector quantization
    for i in range(1, 4):
        decimals = torch.randint(1, 7, (data.shape[i],))
        qdata = linear_quantize_callback(
            data, bits=8, decimal=decimals, channel_index=i
        )
        for j in range(data.shape[i]):
            indices = [
                slice(None),
            ] * 4
            indices[i] = j
            assert np.allclose(
                data[indices].numpy(),
                qdata[indices].numpy(),
                atol=1 / 2 ** decimals[j].item(),
            )


def test_interger_arithmetic():
    ni = 7
    no = 6
    input = torch.randint(-128, 127, size=(3, 10, 32, 32))
    input_float = input.float() / 2 ** ni

    timeout = 5
    # quantized output in 32-bit float
    qconv = quantize(
        torch.nn.Conv2d(10, 30, 3, bias=False), bits=8, timeout=timeout, channelwise=0
    )  # vector quantization on output channel
    qconv.train()
    for _ in range(timeout + 1):  # ensure the weight is quantized
        qconv(input_float)
    output_float = linear_quantize_callback(qconv(input_float), 8, no)

    # quantized output in 8-bit interger
    weight = qconv.weight * (2.0 ** qconv.quantize.decimal).view(-1, 1, 1, 1)
    output_int = F.conv2d(input.int(), weight.int())
    for i in range(output_int.shape[1]):
        output_int[:, i] = (
            output_int[:, i].float() / 2 ** (ni + qconv.quantize.decimal[i] - no)
        ).int()

    diff = (
        output_float.detach().numpy() - (output_int.float() / 2 ** no).detach().numpy()
    )
    assert np.all(diff == 0), "shall be able to fully match with interger arithmetic"
