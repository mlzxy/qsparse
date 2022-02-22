# fmt: off
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from qsparse import prune
from qsparse.sparse import (BanditPruningCallback, MagnitudePruningCallback,
                            UniformPruningCallback)

# fmt: on


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
        callback=BanditPruningCallback(),
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
        strict=False,
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


def get_sparsity(tensor: torch.Tensor):
    nz = tensor.nonzero()
    if isinstance(nz, tuple):
        nz = nz[0]
    return 1 - len(nz) / np.prod(tensor.shape)


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
        callback=BanditPruningCallback(mask_refresh_interval=interval),
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


def test_callback():
    sparsity = 0.5

    # unstructured
    for shape in [(3, 32, 32), (3, 32), (32,)]:
        mask = torch.ones(*shape).bool()
        MagnitudePruningCallback()(torch.rand(shape), sparsity, mask)
        result_sparsity = (~mask).sum() / np.prod(mask.shape)
        assert np.isclose(sparsity, result_sparsity, atol=1 / np.prod(mask.shape))

    for shape in [(3, 32, 32), (3, 32), (32,)]:
        mask = torch.ones(*shape).bool()
        UniformPruningCallback()(torch.rand(shape), sparsity, mask)
        result_sparsity = (~mask).sum() / np.prod(mask.shape)
        assert np.isclose(sparsity, result_sparsity, atol=1 / np.prod(mask.shape))

    mask = torch.ones(*shape).bool()
    UniformPruningCallback()(torch.rand(shape), 0.7, mask)
    result_sparsity = (~mask).sum() / np.prod(mask.shape)
    assert np.isclose(0.7, result_sparsity, atol=1 / np.prod(mask.shape))

    # structured
    shape = (32, 32, 32)
    data = torch.rand(shape)
    for channels in [[0], [0, 1], [0, 1, 2]]:
        mask = torch.ones(
            *[shape[i] if i not in channels else 1 for i in range(3)]
        ).bool()
        MagnitudePruningCallback()(data, sparsity, mask)
        for i in range(3):
            if i not in channels:
                assert mask.shape[i] == 32

        result_sparsity = (~mask).sum() / np.prod(mask.shape)
        assert np.isclose(sparsity, result_sparsity, atol=1 / np.prod(mask.shape))


def test_bandit_pruning():
    shape = (3, 24, 24)
    mean = torch.rand(*shape)
    MSE = nn.MSELoss(reduce="sum")
    mask = torch.ones((1,) + shape)
    layer = BanditPruningCallback()

    for _ in range(1500):
        inp = torch.normal(mean, 1)
        inp.requires_grad = True
        out = layer(inp, 0.5, mask)
        loss = MSE(inp, out)
        loss.backward()

    layer.eval()
    result = layer(mean, 0.5, mask)[0]
    assert np.isclose(get_sparsity(mask), 0.5, atol=1 / np.prod(shape))
    keep, drop = mean[result > 0].mean().item(), mean[result <= 0].mean().item()
    assert keep > 0.7 and drop < 0.3


def test_grad_pruning():
    shape = (3, 24, 24)
    mean = torch.rand(*shape)
    MSE = nn.MSELoss(reduce="sum")
    mask = torch.ones((1,) + shape)
    layer = MagnitudePruningCallback(use_gradient=True)

    for _ in range(1500):
        inp = torch.normal(mean, 1)
        inp = inp.view(1, *inp.shape)
        inp.requires_grad = True
        out = layer(inp, 0.5, mask)
        out.backward(torch.rand((1,) + shape) / 10)

    layer.eval()
    assert np.isclose(get_sparsity(mask), 0.5, atol=1 / np.prod(shape))
