import logging

import numpy as np
import torch
import torch.nn as nn

from qsparse import (
    auto_name_prune_quantize_layers,
    get_qsparse_option,
    prune,
    quantize,
    set_qsparse_options,
)
from qsparse.quantize import approx_quantile


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


def test_approx_quantile():
    data = torch.rand((1000,))
    accurate = torch.quantile(data, 0.5)
    approximate = approx_quantile(
        data, 0.5, bound=500
    )  # use a small bound to trigger approximate quantile computation
    assert np.isclose(accurate, approximate, rtol=0.05)


def test_option(capsys):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    set_qsparse_options()
    set_qsparse_options(log_on_created=False)
    assert get_qsparse_option("log_on_created") is False
    prune(sparsity=0.5)
    quantize(bits=8)
    captured = capsys.readouterr()
    assert "[Prune]" not in captured.out
    assert "[Quantize]" not in captured.out

    set_qsparse_options(log_during_train=False)
    assert get_qsparse_option("log_during_train") is False
    layer = nn.Sequential(
        quantize(bits=8, timeout=1), prune(sparsity=0.5, start=1, interval=1)
    )
    for _ in range(20):
        layer(torch.rand(1, 10))
    captured = capsys.readouterr()
    assert "[Prune" not in captured.out
    assert "[Quantize" not in captured.out
