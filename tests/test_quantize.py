import numpy as np
import torch

from qsparse import linear_quantize_callback


def test_feature():
    pass


def test_weight():
    pass


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
