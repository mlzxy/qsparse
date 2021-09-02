from collections import deque

import torch
import torch.nn as nn

from qsparse.common import OptionalTensorOrModule, PruneCallback
from qsparse.imitation import imitate

__all__ = [
    "prune",
]


def unstructured_prune_callback(inp: torch.Tensor, sparsity: float) -> torch.Tensor:
    saliency = inp.abs().mean(dim=0)
    values = saliency.flatten().sort()[0]
    n = len(values)
    idx = max(int(sparsity * n - 1), 0)
    threshold = values[idx]
    mask = saliency >= threshold
    return mask


def prune(
    arg: OptionalTensorOrModule = None,
    sparsity: float = 0.5,
    # for step-wise training
    start: int = 1000,
    interval: int = 1000,
    repetition: int = 4,
    buffer_size: int = 1,
    # for customization
    callback: PruneCallback = unstructured_prune_callback,
    # for debug purpose
    name="",
) -> OptionalTensorOrModule:

    schedules = [start + interval * (1 + i) for i in range(repetition)]

    class PruneLayer(nn.Module):
        def __init__(
            self,
        ):
            super().__init__()
            print(
                f"[Prune @ {name} Args] start = {start} interval = {interval} repetition = {repetition} sparsity = {sparsity} buffer_size = {buffer_size}"
            )
            self.buffer = deque(maxlen=buffer_size)
            self._init = False

        def forward(self, x: torch.Tensor):
            if not self._init:
                assert len(x.shape) > 1
                self.mask = nn.Parameter(
                    torch.ones(
                        *x.shape[1:],
                        dtype=torch.bool,
                        device=x.device,
                    ),
                    requires_grad=False,
                )
                self._n_updates = nn.Parameter(
                    torch.zeros(1, dtype=torch.int),
                    requires_grad=False,
                )
                self._cur_sparsity = nn.Parameter(torch.zeros(1), requires_grad=False)
                self._init = True

            if any([(0 <= (s - self._n_updates) <= buffer_size) for s in schedules]):
                self.buffer.append(x.detach().abs().mean(dim=0).to("cpu"))

            if (
                (self._n_updates > start)
                and (self._cur_sparsity < sparsity)
                and self.training
            ):
                if ((self._n_updates - start) % interval) == 0:
                    ratio = (
                        1.0 - (self._n_updates - start) / (interval * repetition)
                    ) ** 3
                    self._cur_sparsity[0] = sparsity * (1 - ratio)

                    if self._cur_sparsity > 0:
                        self.mask.data = callback(
                            torch.cat(
                                [v.view(1, *v.shape) for v in self.buffer], dim=0
                            ),
                            self._cur_sparsity.item(),
                        ).to(x.device)

                        active_ratio = self.mask.sum().item() / self.mask.size().numel()
                        print(
                            f"[Prune @ {name} Step {self._n_updates.item()}] active {active_ratio:.02f}, pruned {1 - active_ratio:.02f}, buffer_size = {len(self.buffer)}"
                        )
            if self.training:
                self._n_updates += 1
            return x * self.mask.expand((x.shape[0],) + tuple(self.mask.shape))

    if arg is None:
        return PruneLayer()
    elif isinstance(arg, torch.Tensor):
        return callback(arg, sparsity)
    elif isinstance(arg, nn.Module):
        return imitate(arg, "prune", PruneLayer())
    else:
        raise ValueError(f"{arg} is not a valid argument for prune")


if __name__ == "__main__":
    print(prune())
    print(prune(torch.nn.Conv2d(10, 30, 3)))
