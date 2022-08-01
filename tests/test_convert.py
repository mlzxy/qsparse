from collections import OrderedDict

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from qsparse import convert, prune, quantize
from qsparse.sparse import MagnitudePruningCallback 


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_lenet_convert():
    lenet_float = LeNet()
    # ignore the first conv2d and last linear layer during pruning
    pruning_excluded_layers = [
        (
            nn.Conv2d,
            [
                0,
            ],
        ),
        (
            nn.Linear,
            [
                -1,
            ],
        ),
    ]

    lenet_pruned = convert(
        lenet_float,
        prune(sparsity=0.5, callback=MagnitudePruningCallback()),
        weight_layers=[nn.Conv2d, nn.Linear],
        activation_layers=[nn.Conv2d, nn.Linear],
        excluded_weight_layer_indexes=pruning_excluded_layers,
        excluded_activation_layer_indexes=pruning_excluded_layers,
    )

    lenet_pruned_quantized = convert(
        lenet_pruned,
        quantize(bits=8),
        weight_layers=[nn.Conv2d, nn.Linear],
        activation_layers=[nn.Conv2d, nn.Linear],
        input=True,  # input layer is quantized
    )
    dct = dict(lenet_pruned_quantized.named_modules())
    assert dct['1.fc1.0.0.prune'].__class__.__name__ == "PruneLayer"
    assert dct['1.fc1.0.0.quantize'].__class__.__name__ == "QuantizeLayer"
    assert dct['1.fc1.0.1'].__class__.__name__ == "PruneLayer"
    assert dct['1.fc1.1'].__class__.__name__ == "QuantizeLayer"


def test_nop():
    lenet = LeNet()
    lenet_converted = convert(
        lenet, prune(sparsity=0.5)
    )  # no weight/activation layers supplied
    assert str(lenet) == str(lenet_converted)


def test_data_parallel():
    lenet = LeNet()
    data_parallel = nn.DataParallel(lenet)
    data_parallel_quantized = convert(
        data_parallel,
        quantize(bits=8),
        weight_layers=[nn.Conv2d, nn.Linear],
        activation_layers=[nn.Conv2d, nn.Linear],
        input=False,
    )
    assert "quantize" in str(data_parallel_quantized).lower()


def test_filter():
    net = nn.Sequential(
        OrderedDict(
            [
                ("conv1", nn.Conv2d(3, 6, kernel_size=5)),
                ("special", nn.Sequential(nn.Conv2d(6, 16, kernel_size=5))),
            ]
        )
    )
    converted = convert(
        net, quantize(bits=8), weight_layers=[nn.Conv2d], include=["special"]
    )

    result = str(converted)
    assert result.count("quantize") == 1 and result.index("special") < result.index(
        "quantize"
    )


def test_order():
    net = nn.Sequential(
        OrderedDict(
            [
                ("conv1", nn.Conv2d(3, 6, kernel_size=5)),
                ("fc1", nn.Linear(84, 10)),
            ]
        )
    )
    converted = convert(
        net, quantize(bits=8), activation_layers=[nn.Conv2d, nn.Linear], order="pre"
    )
    result = str(converted).lower()
    assert result.count("quantizelayer") == 2
    assert result.index("quantize") < result.index("conv2d")
    L = result.index("conv2d")
    assert result[L:].index("linear") > result[L:].index("quantize")

    nested_converted = convert(
        net, prune(sparsity=0.5), activation_layers=[nn.Conv2d, nn.Linear]
    )
    result = str(nested_converted).lower()
    assert result.index("quantize") < result.index("prune")



if __name__ == "__main__":
    test_lenet_convert()
    test_nop()
    test_data_parallel()
    test_filter()
    test_order()