import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from qsparse import convert, prune, quantize


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
        prune(sparsity=0.5),
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

    gt = """Sequential(
  (0): QuantizeLayer()
  (1): LeNet(
    (conv1): Sequential(
      (0): Conv2d(
        3, 6, kernel_size=(5, 5), stride=(1, 1)
        (quantize): QuantizeLayer()
      )
      (1): QuantizeLayer()
    )
    (conv2): Sequential(
      (0): Sequential(
        (0): Conv2d(
          6, 16, kernel_size=(5, 5), stride=(1, 1)
          (prune): PruneLayer()
          (quantize): QuantizeLayer()
        )
        (1): PruneLayer()
      )
      (1): QuantizeLayer()
    )
    (fc1): Sequential(
      (0): Sequential(
        (0): Linear(
          in_features=400, out_features=120, bias=True
          (prune): PruneLayer()
          (quantize): QuantizeLayer()
        )
        (1): PruneLayer()
      )
      (1): QuantizeLayer()
    )
    (fc2): Sequential(
      (0): Sequential(
        (0): Linear(
          in_features=120, out_features=84, bias=True
          (prune): PruneLayer()
          (quantize): QuantizeLayer()
        )
        (1): PruneLayer()
      )
      (1): QuantizeLayer()
    )
    (fc3): Sequential(
      (0): Linear(
        in_features=84, out_features=10, bias=True
        (quantize): QuantizeLayer()
      )
      (1): QuantizeLayer()
    )
  )
)"""
    assert str(lenet_pruned_quantized) == gt


def test_nop():
    lenet = LeNet()
    lenet_converted = convert(
        lenet, prune(sparsity=0.5)
    )  # no weight/activation layers supplied
    assert str(lenet) == str(lenet_converted)
