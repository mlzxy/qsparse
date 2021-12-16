import numpy as np
import pytest
import torch

from qsparse import (
    prune,
    structured_prune_callback,
    unstructured_prune_callback,
    unstructured_uniform_prune_callback,
)


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
    for shape in [(3, 32, 32), (3, 32), (32)]:
        mask = unstructured_prune_callback(
            [torch.rand(shape) for _ in range(10)], sparsity
        )
        result_sparsity = (~mask).sum() / np.prod(mask.shape)
        assert np.isclose(sparsity, result_sparsity, atol=1 / np.prod(mask.shape))

    for shape in [(3, 32, 32), (3, 32), (32)]:
        mask = unstructured_uniform_prune_callback(
            [torch.rand(shape) for _ in range(10)], sparsity
        )
        result_sparsity = (~mask).sum() / np.prod(mask.shape)
        assert np.isclose(sparsity, result_sparsity, atol=1 / np.prod(mask.shape))
    mask = unstructured_uniform_prune_callback(
        [torch.rand(shape) for _ in range(10)], 0.7, current_mask=mask
    )
    result_sparsity = (~mask).sum() / np.prod(mask.shape)
    assert np.isclose(0.7, result_sparsity, atol=1 / np.prod(mask.shape))

    # structured
    data = [torch.rand((32, 32, 32)) for _ in range(10)]
    for channels in [{0}, {0, 1}, {0, 1, 2}]:
        mask = structured_prune_callback(data, sparsity, prunable=channels)
        for i in range(3):
            if i not in channels:
                assert mask.shape[i] == 1

        result_sparsity = (~mask).sum() / np.prod(mask.shape)
        assert np.isclose(sparsity, result_sparsity, atol=1 / np.prod(mask.shape))
