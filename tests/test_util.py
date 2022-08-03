# fmt: off
import logging
import uuid

import numpy as np
import torch
import torch.nn as nn

from qsparse import (auto_name_prune_quantize_layers, get_qsparse_option, 
                     prune, quantize, set_qsparse_options)
from qsparse.util import squeeze_tensor_to_shape, calculate_mask_given_importance
# fmt: on


def test_auto_name_prune_quantize_layers():
    class TwoLayerNet(torch.nn.Module):
        def __init__(self, D_in, H, D_out):
            super(TwoLayerNet, self).__init__()
            self.linear1 = quantize(prune(torch.nn.Linear(D_in, H)))
            self.linear2 = quantize(prune(torch.nn.Linear(H, D_out)))

        def forward(self, x):
            h_relu = self.linear1(x).clamp(min=0)
            y_pred = self.linear2(h_relu)
            return y_pred

    net = TwoLayerNet(10, 30, 1)
    net = auto_name_prune_quantize_layers(net)
    assert net.linear1.prune.name == "linear1.prune"
    assert net.linear1.quantize.name == "linear1.quantize"
    assert net.linear2.prune.name == "linear2.prune"
    assert net.linear2.quantize.name == "linear2.quantize"







def test_option(capsys):
    fpath = f"/tmp/log.{uuid.uuid4()}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(fpath)],
    )

    set_qsparse_options()
    set_qsparse_options(log_on_created=False)
    assert get_qsparse_option("log_on_created") is False
    prune(sparsity=0.5)
    quantize(bits=8)
    logging.root.handlers[0].flush()
    captured = open(fpath).read()
    assert "[Prune]" not in captured
    assert "[Quantize]" not in captured
    with open(fpath, "w") as f:
        f.write("")

    set_qsparse_options(log_during_train=False)
    assert get_qsparse_option("log_during_train") is False
    layer = nn.Sequential(
        quantize(bits=8, timeout=1), prune(sparsity=0.5, start=1, interval=1)
    )
    for _ in range(20):
        layer(torch.rand(1, 10))
    captured = open(fpath).read()
    assert "[Prune" not in captured
    assert "[Quantize" not in captured



def test_squeeze_tensor_to_shape():
    tensor = torch.rand(10,30,7,8)
    target_shape = (1, 30, 7, 1)
    assert tuple(squeeze_tensor_to_shape(tensor, target_shape).shape) == target_shape



def test_calculate_mask_given_importance():
    tensor = torch.rand(10,30,7,8)
    target_sparsity = 0.47
    mask = calculate_mask_given_importance(tensor, target_sparsity)
    assert (1 - mask.sum().item() / mask.numel()) == target_sparsity
    

def test_preload_state_dict():
    from qsparse.util import preload_qsparse_state_dict

    def make_conv():
        return quantize(prune(nn.Conv2d(16, 32, 3), 
                            sparsity=0.5, start=200, 
                            interval=10, repetition=4), 
                    bits=8, timeout=100)

    conv = make_conv()

    for _ in range(241):
        conv(torch.rand(10, 16, 7, 7))

    try:
        conv2 = make_conv()
        conv2.load_state_dict(conv.state_dict())
    except RuntimeError as e:
        print(f'\nCatch error as expected: {e}\n' )

    conv3 = make_conv()
    preload_qsparse_state_dict(conv3, conv.state_dict())
    conv3.load_state_dict(conv.state_dict())

    tensor = torch.rand(10, 16, 7, 7)
    assert np.allclose(conv(tensor).detach().numpy(), conv3(tensor).detach().numpy(), atol=1e-5)