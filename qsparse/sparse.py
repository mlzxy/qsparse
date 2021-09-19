import warnings
from collections import deque
from typing import Iterable, List, Union

import torch
import torch.nn as nn

from qsparse.common import OptionalTensorOrModule, PruneCallback
from qsparse.imitation import imitate

__all__ = ["prune", "unstructured_prune_callback", "structured_prune_callback"]


def unstructured_prune_callback(
    inp: List[torch.Tensor], sparsity: float
) -> torch.Tensor:
    inp = torch.cat([v.view(1, *v.shape) for v in inp], dim=0)
    saliency = inp.abs().mean(dim=0)
    values = saliency.flatten().sort()[0]
    n = len(values)
    idx = max(int(sparsity * n - 1), 0)
    threshold = values[idx]
    mask = saliency >= threshold
    return mask


def structured_prune_callback(
    inp: List[torch.Tensor], sparsity: float, prunable: Union[Iterable[int], int] = {1}
) -> torch.Tensor:
    prunables = {prunable} if isinstance(prunable, int) else prunable
    saliency_lst = []
    for saliency in inp:
        for _i in range(len(saliency.shape)):
            if _i not in prunables:
                saliency = saliency.abs().mean(dim=_i, keepdim=True)
        saliency_lst.append(saliency)
    saliency = torch.cat(saliency_lst, dim=0).abs().mean(dim=0, keepdim=True)
    values = saliency.flatten().sort()[0]
    n = len(values)
    idx = max(int(sparsity * n - 1), 0)
    threshold = values[idx]
    mask = saliency >= threshold
    return mask


class PruneLayer(nn.Module):
    def __init__(
        self,
        sparsity: float = 0.5,
        # for step-wise training
        start: int = 1000,
        interval: int = 1000,
        repetition: int = 4,
        buffer_size: int = 1,
        strict: bool = True,
        # for customization
        callback: PruneCallback = unstructured_prune_callback,
        collapse: int = 0,
        # for debug purpose
        name="",
    ):
        super().__init__()
        print(
            f"[Prune @ {name} Args] start = {start} interval = {interval} repetition = {repetition} sparsity = {sparsity} buffer_size = {buffer_size} collapse = {collapse} "
        )
        self.schedules = [start + interval * (1 + i) for i in range(repetition)]
        self.buffer = deque(maxlen=buffer_size)
        self.start = start
        self.interval = interval
        self.repetition = repetition
        self.sparsity = sparsity
        self.name = name
        self.callback = callback
        self.buffer_size = buffer_size
        self._collapse = collapse
        self.strict = strict
        self._init = False

        for k in [
            "mask",
            "_n_updates",
            "_cur_sparsity",
        ]:
            self.register_parameter(
                k,
                nn.Parameter(
                    torch.tensor(-1, dtype=torch.int), requires_grad=False
                ),  # placeholder
            )

    @property
    def initted(self) -> bool:
        return self._n_updates.item() != -1

    def collapse(self, x: torch.Tensor):
        if self._collapse >= 0:
            return x.detach().abs().mean(self._collapse, keepdim=True)
        else:
            return x.detach()

    def forward(self, x: torch.Tensor):
        if not self.initted:
            assert len(x.shape) > 1
            with torch.no_grad():
                m_example = self.callback([self.collapse(x)], 0)
            self.mask = nn.Parameter(
                torch.ones(*m_example.shape, dtype=torch.bool).to(x.device),
                requires_grad=False,
            )
            self._n_updates = nn.Parameter(
                torch.zeros(1, dtype=torch.int).to(x.device),
                requires_grad=False,
            )
            self._cur_sparsity = nn.Parameter(
                torch.zeros(1).to(x.device), requires_grad=False
            )

        def should_prune(n):
            cond = any([(0 <= (s - n) <= self.buffer_size) for s in self.schedules])
            if self.strict:
                return cond
            else:
                return cond and self.training

        if should_prune(self._n_updates):
            self.buffer.append(self.collapse(x).to("cpu"))

        # add buffer size check to avoid prune a layer which always set to eval
        if (
            (self._n_updates > self.start)
            and (self._cur_sparsity < self.sparsity)
            and self.training
            and len(self.buffer) > 0
        ):
            if ((self._n_updates - self.start) % self.interval) == 0:
                ratio = (
                    1.0
                    - (self._n_updates.item() - self.start)
                    / (self.interval * self.repetition)
                ) ** 3
                self._cur_sparsity[0] = self.sparsity * (1 - ratio)

                if self._cur_sparsity > 0:
                    self.mask.data = self.callback(
                        self.buffer,
                        self._cur_sparsity.item(),
                    ).to(x.device)

                    active_ratio = self.mask.sum().item() / self.mask.size().numel()
                    print(
                        f"[Prune @ {self.name} Step {self._n_updates.item()}] active {active_ratio:.02f}, pruned {1 - active_ratio:.02f}, buffer_size = {len(self.buffer)}"
                    )
                    if len(self.buffer) < self.buffer_size:
                        warnings.warn(
                            f"buffer is not full when pruning, this will cause performance degradation! (buffer has {len(self.buffer)} elements while buffer_size parameter is {self.buffer_size})"
                        )
                    if not should_prune(self._n_updates + 1):
                        # proactively free up memory
                        self.buffer.clear()
        if self.training:
            self._n_updates += 1

        if self.strict or self.training:
            return x * self.mask.expand(x.shape)
        else:
            mask = self.mask
            if len(self.mask.shape) != len(x.shape):
                if len(self.mask.shape) == (len(x.shape) - 1):
                    mask = mask.view(1, *mask.shape)
                else:
                    raise RuntimeError(
                        f"mask shape not matched: mask {mask.shape} vs input {x.shape}"
                    )
            target_shape = x.shape[1:]
            final_mask = torch.ones(
                (1,) + target_shape, device=x.device, dtype=mask.dtype
            )
            repeats = [x.shape[i] // mask.shape[i] for i in range(1, len(x.shape))]
            mask = mask.repeat(1, *repeats)
            slices = [0] + [
                slice(
                    (x.shape[i] - mask.shape[i]) // 2,
                    (x.shape[i] - mask.shape[i]) // 2 + mask.shape[i],
                )
                for i in range(1, len(x.shape))
            ]
            final_mask[slices] = mask[0, :]
            return x * final_mask


def prune(
    arg: OptionalTensorOrModule = None,
    sparsity: float = 0.5,
    # for step-wise training
    start: int = 1000,
    interval: int = 1000,
    repetition: int = 4,
    buffer_size: int = 1,
    strict: bool = True,
    # for customization
    callback: PruneCallback = unstructured_prune_callback,
    # for debug purpose
    name="",
) -> OptionalTensorOrModule:
    def get_prune_layer(feature_collapse=0):
        return PruneLayer(
            start=int(start),
            sparsity=sparsity,
            interval=int(interval),
            repetition=repetition,
            buffer_size=buffer_size,
            name=name,
            strict=strict,
            callback=callback,
            collapse=feature_collapse,
        )

    if arg is None:
        return get_prune_layer()
    elif isinstance(arg, torch.Tensor):
        return callback(arg, sparsity)
    elif isinstance(arg, nn.Module):
        return imitate(arg, "prune", get_prune_layer(-1))
    else:
        raise ValueError(f"{arg} is not a valid argument for prune")


if __name__ == "__main__":
    print(prune())
    print(prune(torch.nn.Conv2d(10, 30, 3)))
