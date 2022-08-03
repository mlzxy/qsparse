# fmt: off
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from qsparse import prune, convert
from qsparse.sparse import (MagnitudePruningCallback, UniformPruningCallback, 
                            devise_layerwise_pruning_schedule, PruneLayer)

# fmt: on

def get_sparsity(tensor: torch.Tensor):
    nz = tensor.nonzero()
    if isinstance(nz, tuple):
        nz = nz[0]
    return 1 - len(nz) / np.prod(tensor.shape)


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


def test_feature():
    start, interval, repetition = 5, 2, 3
    data = torch.rand((1, 10, 32, 32))
    data2x = torch.rand((1, 10, 64, 64))
    prune_layer = prune(
        sparsity=0.5, start=start, interval=interval, repetition=repetition
    )
    for _ in range(
        start + interval * (repetition + 1)
    ):  # ensure the sparsification has been triggered
        output = prune_layer(data).numpy()
    assert np.isclose(get_sparsity(output), 0.5, atol=1 / np.prod(output.shape))
    assert np.all((output == 0) == (prune_layer.mask.numpy() == 0))

    prune_layer = prune(
        sparsity=0.5,
        start=start,
        interval=interval,
        repetition=repetition,
        dimensions={0,1,2,3},
        callback=UniformPruningCallback(),
    )
    prune_layer.train()
    for _ in range(
        start + interval * (repetition + 2)
    ):  # ensure the sparsification has been triggered
        if _ == start + interval * (repetition + 2) - 1:
            prune_layer.eval()
        output = prune_layer(data).numpy()
    assert np.isclose(get_sparsity(output), 0.5, atol=1 / np.prod(output.shape))
    assert np.all((output == 0) == (prune_layer.mask.numpy() == 0))

    with pytest.raises(RuntimeError):
        prune_layer.eval()
        prune_layer(data2x)  # input shape shall be stationary

    # changing input shape during evaluation
    prune_layer = prune(
        sparsity=0.5,
        start=start,
        interval=interval,
        repetition=repetition,
        dimensions={1}
    )
    for _ in range(
        start + interval * (repetition + 1)
    ):  # ensure the sparsification has been triggered
        output = prune_layer(data).numpy()
    prune_layer.eval()
    output = prune_layer(data2x)
    assert output.shape == data2x.shape
    assert np.isclose(
        get_sparsity(output), 0.5, atol=len(output.shape) / np.prod(output.shape)
    )


def test_weight():
    start, interval, repetition = 5, 2, 3

    data = torch.rand((1, 10, 32, 32))
    pconv = prune(
        torch.nn.Conv2d(10, 30, 3),
        sparsity=0.5,
        start=start,
        interval=interval,
        repetition=repetition,
        callback=MagnitudePruningCallback(running_average=False),
    )
    pconv.train()
    for _ in range(
        start + interval * (repetition + 1)
    ):  # ensure the sparsification has been triggered
        pconv(data)
    assert np.isclose(
        get_sparsity(pconv.weight), 0.5, atol=1 / np.prod(pconv.weight.shape)
    )
    assert np.all(
        (pconv.weight.detach().numpy() == 0) == (pconv.prune.mask.detach().numpy() == 0)
    )
    assert not np.isclose(
        get_sparsity(dict(pconv.named_parameters())["weight"]), 0.5, atol=0.1
    ), "parameter['weight'] shall store the untouched weight without pruning"

    pconv = prune(
        torch.nn.Conv2d(10, 30, 3),
        sparsity=0.5,
        start=start,
        interval=interval,
        repetition=repetition,
        callback=UniformPruningCallback(),
    )
    pconv.train()
    for _ in range(
        start + interval * (repetition + 1)
    ):  # ensure the sparsification has been triggered
        pconv(data)
        print(pconv.prune._n_updates.item())
    assert np.isclose(
        get_sparsity(pconv.weight), 0.5, atol=1 / np.prod(pconv.weight.shape)
    )
    pconv.eval()
    assert np.all(
        (pconv.weight.detach().numpy() == 0) == (pconv.prune.mask.detach().numpy() == 0)
    )

    pconv = prune(
        torch.nn.Conv2d(10, 30, 3),
        sparsity=0.5,
        start=start,
        interval=interval,
        repetition=repetition,
    )
    pconv.eval()
    for _ in range(
        start + interval * (repetition + 1)
    ):  # ensure the sparsification has been triggered
        pconv(data)
    assert not np.isclose(
        get_sparsity(pconv.weight), 0.5, atol=0.1
    ), "sparsification schedule shall only be triggered during training"

    with pytest.raises(ValueError):  # shall only accept module input or no input
        prune(torch.rand((10,)))



def test_more_pruning_options():
    # gradient magnitude
    shape = (3, 24, 24)
    mean = torch.rand(*shape)
    mask = torch.ones((1,) + shape)
    layer = MagnitudePruningCallback(use_gradient=True)
    for _ in range(1500):
        inp = torch.normal(mean, 1)
        inp = inp.view(1, *inp.shape)
        inp.requires_grad = True
        out = layer(inp, 0.5, mask)
        out.backward(torch.rand((1,) + shape) / 10)
    layer.eval()
    assert np.isclose(get_sparsity(mask), 0.5, atol=2 / np.prod(shape))


    mask = torch.ones(shape)
    layer = MagnitudePruningCallback(l0=True)
    for _ in range(1500):
        inp = (torch.rand(*shape) > 0.5).float()
        out = layer(inp, 0.5, mask)
    assert np.isclose(get_sparsity(mask), 0.5, atol=2 / np.prod(shape))   


def test_layerwise():
    net = LeNet()
    pnet = convert(
        net,
        prune(sparsity=0.5, callback=MagnitudePruningCallback()),
        activation_layers=[nn.Conv2d, nn.Linear]
    )
    pnet = devise_layerwise_pruning_schedule(pnet, start=10, interval=100, mask_refresh_interval=10)
    sparsification_start_iters = [mod.start for mod in net.modules() if isinstance(mod, PruneLayer)]
    assert sparsification_start_iters == sorted(sparsification_start_iters)
