import logging
import warnings
from collections import deque
from typing import Iterable, List, Union

import numpy as np
import torch
import torch.nn as nn

from qsparse.common import PruneCallback
from qsparse.imitation import imitate
from qsparse.util import get_option, nd_slice

__all__ = [
    "prune",
    "unstructured_prune_callback",
    "structured_prune_callback",
    "unstructured_uniform_prune_callback",
]


def unstructured_uniform_prune_callback(
    inp: List[torch.Tensor], sparsity: float, current_mask: torch.Tensor = None
) -> torch.Tensor:
    """unstructured uniform pruning function with type signature of [PruneCallback][qsparse.common.PruneCallback].
    This function will prune uniformly without considering magnitude of the input tensors. If a current mask is provided,
    this function will not reactivate those already pruned locations in current mask.

    Args:
        inp (List[torch.Tensor]): input tensor list (see more in [PruneCallback][qsparse.common.PruneCallback])
        sparsity (float): target sparsity
        current_mask (torch.Tensor, optional): current mask of the pruning procedure. Defaults to None.

    Returns:
        torch.Tensor: binary mask
    """
    assert len(inp) >= 1, "no input tensor is provided"
    shape = inp[0].shape
    if current_mask is not None:
        cur_sparsity = (~current_mask).sum().item() / current_mask.numel()
        mask = current_mask.to("cpu")
    else:
        mask = torch.ones(*shape, dtype=torch.bool)
        cur_sparsity = 0
    assert sparsity >= cur_sparsity, "only support pruning to a larger sparsity"
    budget = int((sparsity - cur_sparsity) * np.prod(shape))
    slots = mask.nonzero(as_tuple=True)
    selected_indexes = np.random.choice(
        range(len(slots[0])), size=budget, replace=False
    )
    mask[[slot[selected_indexes] for slot in slots]] = False
    return mask


def unstructured_prune_callback(
    inp: List[torch.Tensor], sparsity: float, current_mask: torch.Tensor = None
) -> torch.Tensor:
    """unstructured pruning function with type signature of [PruneCallback][qsparse.common.PruneCallback].

    Args:
        inp (List[torch.Tensor]): input tensor list (see more in [PruneCallback][qsparse.common.PruneCallback])
        sparsity (float): target sparsity
        current_mask (torch.Tensor, optional): current mask of the pruning procedure. Defaults to None.

    Returns:
        torch.Tensor: binary mask
    """
    inp = torch.cat([v.view(1, *v.shape) for v in inp], dim=0)
    saliency = inp.abs().mean(dim=0)
    values = saliency.flatten().sort()[0]
    n = len(values)
    idx = max(int(sparsity * n - 1), 0)
    threshold = values[idx]
    mask = saliency >= threshold
    return mask


def structured_prune_callback(
    inp: List[torch.Tensor], sparsity: float, prunable: Union[Iterable[int], int] = {0}
) -> torch.Tensor:
    """structured pruning function with type signature of [PruneCallback][qsparse.common.PruneCallback].

    Args:
        inp (List[torch.Tensor]): input tensor list (see more in [PruneCallback][qsparse.common.PruneCallback])
        sparsity (float): target sparsity
        prunable (Union[Iterable[int], int], optional): dimension indexes that are prunable. Defaults to {0}, which corresponds to channel dimension when batch dimension is not present.

    Returns:
        torch.Tensor: binary mask
    """

    prunables = {prunable} if isinstance(prunable, int) else prunable
    saliency_lst = []
    for saliency in inp:
        for _i in range(len(saliency.shape)):
            if _i not in prunables:
                saliency = saliency.abs().mean(dim=_i, keepdim=True)
        saliency_lst.append(saliency.view(1, *saliency.shape))
    saliency = torch.cat(saliency_lst, dim=0).abs().mean(dim=0)
    values = saliency.flatten().sort()[0]
    n = len(values)
    idx = max(int(sparsity * n - 1), 0)
    threshold = values[idx]
    mask = saliency >= threshold
    return mask


class PruneLayer(nn.Module):
    """Applies pruning over input tensor.

    Please look for detailed description in [prune][qsparse.sparse.prune]
    """

    def __init__(
        self,
        sparsity: float = 0.5,
        # for step-wise training
        start: int = 1000,
        interval: int = 1000,
        repetition: int = 4,
        window_size: int = 1,
        strict: bool = True,
        # for customization
        callback: PruneCallback = unstructured_prune_callback,
        collapse: int = 0,
        # for debug purpose
        name="",
    ):
        super().__init__()
        if get_option("log_on_created"):
            logging.info(
                f"[Prune{name if name == '' else f' @ {name}'}] start = {start} interval = {interval} repetition = {repetition} sparsity = {sparsity} window_size = {window_size} collapse = {collapse} "
            )
        self.schedules = [start + interval * (1 + i) for i in range(repetition)]
        self.window = deque(maxlen=window_size)
        self.start = start
        self.interval = interval
        self.repetition = repetition
        self.sparsity = sparsity
        self.name = name
        self.callback = callback
        self.window_size = window_size
        self._collapse = collapse
        self.strict = strict

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
        """whether the parameters of the prune layer are initialized."""
        return self._n_updates.item() != -1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Prunes input tensor according to given sparsification schedule.

        Args:
            x (torch.Tensor): tensor to be pruned

        Raises:
            RuntimeError: when the shape of input tensors mismatch with the shape of binary mask

        Returns:
            torch.Tensor: pruned tensor
        """
        if not self.initted:
            assert len(x.shape) > 1
            with torch.no_grad():
                m_example = self.callback(
                    [x.detach() if self._collapse < 0 else x.mean(self._collapse)],
                    0,
                )
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
            cond = any([(0 <= (s - n) <= self.window_size) for s in self.schedules])
            if self.strict:
                return cond
            else:
                return cond and self.training

        if should_prune(self._n_updates):
            if self._collapse >= 0:
                for t in (
                    x[nd_slice(len(x.shape), self._collapse, end=self.window_size)]
                    .abs()
                    .detach()
                    .split(1)
                ):  # type: torch.Tensor
                    self.window.append(t.squeeze(0).to("cpu"))
            else:
                self.window.append(x.abs().detach().to("cpu"))

        # add window size check to avoid prune a layer which always set to eval
        if (
            (self._n_updates > self.start)
            and (self._cur_sparsity < self.sparsity)
            and self.training
            and len(self.window) > 0
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
                        self.window,
                        self._cur_sparsity.item(),
                        current_mask=self.mask.data,
                    ).to(x.device)

                    active_ratio = self.mask.sum().item() / self.mask.size().numel()
                    if get_option("log_during_train"):
                        logging.info(
                            f"[Prune{self.name if self.name == '' else f' @ {self.name}'}] [Step {self._n_updates.item()}] active {active_ratio:.02f}, pruned {1 - active_ratio:.02f}, window_size = {len(self.window)}"
                        )
                    if len(self.window) < self.window_size:
                        warnings.warn(
                            f"window is not full when pruning, this will cause performance degradation! (window has {len(self.window)} elements while window_size parameter is {self.window_size})"
                        )
                    if not should_prune(self._n_updates + 1):
                        # proactively free up memory
                        self.window.clear()
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
    inp: nn.Module = None,
    sparsity: float = 0.5,
    # for step-wise training
    start: int = 1000,
    interval: int = 1000,
    repetition: int = 4,
    window_size: int = 1,
    strict: bool = True,
    collapse: Union[str, int] = "auto",
    # for customization
    callback: PruneCallback = unstructured_prune_callback,
    # for debug purpose
    name="",
) -> nn.Module:
    """Creates a [PruneLayer][qsparse.sparse.PruneLayer] which is usually used
    for feature pruning if no input module is provided, or creates a weight-
    pruned version of the input module.

    Args:
        inp (nn.Module, optional): input module whose weight is to be pruned. Defaults to None.
        sparsity (float, optional): target sparsity. Defaults to 0.5.
        start (int, optional): starting step to apply pruning. Defaults to 1000.
        interval (int, optional): interval of steps between each pruning operation. Defaults to 1000.
        repetition (int, optional): number of pruning operations. Defaults to 4.
        window_size (int, optional): number of input tensors used for computing the binary mask. Defaults to 1, means using only current input.
        strict (bool, optional): whether enforcing the shape of the binary mask to be equal to the input tensor. Defaults to True.
                                 When strict=False, it will try to expand the binary mask to matched the input tensor shape during evaluation, useful for tasks whose test images are larger, like super resolution.
        collapse (Union[str, int]): which dimension to ignore when creating binary mask. It is usually set to 0 for the batch dimension during pruning activations, and -1 when pruning weights. Default to "auto", means setting `collapse` automatically based on `inp` parameter.
        callback (PruneCallback, optional): callback for actual operation of pruning tensor, used for customization. Defaults to [unstructured\_prune\_callback][qsparse.sparse.unstructured_prune_callback].
        name (str, optional): name of the prune layer created, used for better logging. Defaults to "".

    Returns:
        nn.Module: input module with its weight pruned or a instance of [PruneLayer][qsparse.sparse.PruneLayer] for feature pruning
    """

    kwargs = dict(
        sparsity=sparsity,
        start=start,
        interval=interval,
        repetition=repetition,
        window_size=window_size,
        strict=strict,
        collapse=collapse,
        callback=callback,
        name=name,
    )

    def get_prune_layer(feature_collapse=0):
        if collapse != "auto":
            feature_collapse = collapse
        return PruneLayer(
            start=int(start),
            sparsity=sparsity,
            interval=int(interval),
            repetition=repetition,
            window_size=window_size,
            name=name,
            strict=strict,
            callback=callback,
            collapse=feature_collapse,
        )

    if inp is None:
        layer = get_prune_layer()
        setattr(layer, "_kwargs", kwargs)
        return layer
    elif isinstance(inp, nn.Module):
        return imitate(inp, "prune", get_prune_layer(-1))
    else:
        raise ValueError(f"{inp} is not a valid argument for prune")


if __name__ == "__main__":
    print(prune())
    print(prune(torch.nn.Conv2d(10, 30, 3)))
