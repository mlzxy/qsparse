import numpy as np
import torch

from qsparse import structured_prune_callback, unstructured_prune_callback


def test_feature():
    pass


def test_weight():
    pass


def test_callback():
    sparsity = 0.5

    # unstructured
    for shape in [(3, 32, 32), (3, 32), (32)]:
        mask = unstructured_prune_callback(
            [torch.rand(shape) for _ in range(10)], sparsity
        )
        result_sparsity = mask.sum() / np.prod(mask.shape)
        assert np.isclose(sparsity, result_sparsity, atol=1 / np.prod(mask.shape))

    # structured
    data = [torch.rand((32, 32, 32)) for _ in range(10)]
    for channels in [{0}, {0, 1}, {0, 1, 2}]:
        mask = structured_prune_callback(data, sparsity, prunable=channels)
        for i in range(3):
            if i not in channels:
                assert mask.shape[i] == 1

        result_sparsity = mask.sum() / np.prod(mask.shape)
        assert np.isclose(sparsity, result_sparsity, atol=1 / np.prod(mask.shape))
