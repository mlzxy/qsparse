import torch
import torch.autograd as autograd
import torch.nn as nn
from math import pow
from functools import reduce  # Valid in Python 2.6+, required in Python 3
import operator
from collections import deque


def prod(lst):
    return reduce(operator.mul, lst, 1)


class _InternalSparseFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, args):
        (
            mask,
            cur_sparsity,
            n_updates,
            final_sparsity,
            prune_freq,
            start_steps,
            n_prunes,
            train_mode,
            name,
        ) = args
        if (n_updates > start_steps) and (cur_sparsity < final_sparsity) and train_mode:
            if ((n_updates - start_steps) % prune_freq) == 0:
                ratio = pow(
                    (1.0 - (n_updates - start_steps) / (prune_freq * n_prunes)), 3
                )
                cur_sparsity = final_sparsity * (1 - ratio)
                # update mask
                saliency = x.abs().mean(dim=0)
                values = saliency.flatten().sort()[0]
                n = prod(saliency.shape)
                idx = max(int(cur_sparsity * n - 1), 0)
                threshold = values[idx]
                new_mask = saliency >= threshold
                mask[:] = new_mask
                valid_ratio = mask.sum().item() / n
                print(
                    f"[SparseLayer @ {n_updates} # {name}] valid {valid_ratio:.02f}, sparse {1 - valid_ratio:.02f}"
                )
        ctx.save_for_backward(mask)
        return (
            x * mask.expand((x.shape[0],) + tuple(mask.shape)),
            mask,
            torch.Tensor(
                [
                    cur_sparsity,
                ]
            ).to(x.device),
        )

    @staticmethod
    def backward(ctx, *grad_out):
        grad_input = grad_out[0].clone()
        mask = ctx.saved_tensors[0]
        grad_input = grad_input * mask.expand(
            (grad_input.shape[0],) + tuple(mask.shape)
        )
        return (grad_input, None, None)


class SparseLayer(nn.Module):
    """ Custom layer that gradually applies a binary sparse mask on input"""

    def __init__(
        self,
        sparsity=0.5,
        prune_freq=1000,
        start_steps=1000,
        n_prunes=4,
        name="",
        gamma=0,
    ):
        """

        bt + bt-1 * gamma + bt-2 * gamma^2 + ...

        Args:
            sparsity (float, optional): target sparse ratio, the larger, the more zeros we have. Defaults to 0.5.
            prune_freq (int, optional): how many steps between each pruning stage. Defaults to 1000.
            start_steps (int, optional): when to start pruning. Defaults to 1000.
            n_prunes (int, optional): how many pruning stages. Defaults to 4.
            gamma (int, optional): how much weight to assign for history batch at every step. Default to 0 (only use current batch).
        """

        super().__init__()
        self.func = _InternalSparseFunc.apply

        self.final_sparsity = sparsity
        self.prune_freq = prune_freq
        self.start_steps = start_steps
        self.n_prunes = n_prunes
        self.gamma = gamma
        self.name = name

        self._cur_sparsity = 0
        self._n_updates = 0
        self._init = False

    def forward(self, x: torch.Tensor):
        if not self._init:
            assert len(x.shape) > 1
            self.mask = nn.Parameter(
                torch.ones(
                    *x.shape[1:], dtype=torch.bool, requires_grad=False, device=x.device
                ),
                requires_grad=False,
            )
            self._init = True

        spx, new_mask, new_sparsity = self.func(
            x,
            [
                self.mask,
                self._cur_sparsity,
                self._n_updates,
                self.final_sparsity,
                self.prune_freq,
                self.start_steps,
                self.n_prunes,
                self.training,
                self.name,
            ],
        )
        self.mask = new_mask
        self._cur_sparsity = new_sparsity.item()
        if self.training:
            self._n_updates += 1
        return spx


if __name__ == "__main__":
    layer = SparseLayer(name="Pruning", n_prunes=8)
    data = torch.rand(10, 100)

    for _ in range(5000):
        layer(data)

    print(list(layer.parameters()))
