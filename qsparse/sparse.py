import math
import copy
import warnings
from argparse import ArgumentError
import os.path as osp
from collections import deque
from typing import Callable, Iterable, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from qsparse.imitation import imitate
from qsparse.util import squeeze_tensor_to_shape, calculate_mask_given_importance, get_option, logging


class MagnitudePruningCallback(nn.Module):
    def __init__(
        self,
        mask_refresh_interval: int = -1,
        stop_mask_refresh: int = float("inf"),
        use_gradient: bool = False,
        running_average: bool = True,
        l0: bool = False,
        forward_hook: Callable[[torch.Tensor, str], None] = None
    ):
        """
        Magnitude-based pruning function  as the callback of [prune][qsparse.sparse.prune].

        Args:
            mask_refresh_interval (int, optional): number of steps to refresh mask. Defaults to 1.
            stop_mask_refresh (int, optional): when to stop refreshing mask. Defaults to float('inf').
            use_gradient (bool, optional): whether use the magnitude of gradients
            running_average (bool, optional): whether use the running average of magnitude. Defaults to True.
            l0 (bool, optional): whether to use l0 magnitude instead of l0
            forward_hook (Callable, optional): callback function that gets executed at each forward. Defaults to None.
        """
        super().__init__()
        self.mask_refresh_interval = mask_refresh_interval
        self.stop_mask_refresh = stop_mask_refresh
        self.use_gradient = use_gradient
        self.t = nn.Parameter(torch.full((1,), -1), requires_grad=False)
        if use_gradient and not running_average:
            raise ArgumentError(
                "the combination of `use_gradient=True` and `running_average=False` is not supported"
            )
        self.running_average = running_average
        self.prev_grad_hook = None
        self.l0 = l0
        self.forward_hook = forward_hook


    @property
    def initted(self) -> bool:
        return self.t.item() != -1

    def prune_and_update_mask(
        self, x: torch.Tensor, sparsity: float, mask: torch.Tensor
    ) -> torch.Tensor:
        if self.running_average:
            importance = self.magnitude
        else:
            importance = squeeze_tensor_to_shape(x.abs(), mask.shape)
        mask.data[:] = calculate_mask_given_importance(importance, sparsity)
        return x * mask


    def receive_input(self, x: torch.Tensor):
        if self.use_gradient:
            if self.prev_grad_hook is not None:
                self.prev_grad_hook.remove()
            if x.requires_grad:
                self.prev_grad_hook = x.register_hook(
                    lambda grad: self.update_magnitude(grad))
            else:
                logging.error("meeting no-grad tensor")
                self.prev_grad_hook = None
        else:
            self.update_magnitude(x)

    def update_magnitude(self, x):
        if self.running_average:
            with torch.no_grad():
                if self.l0 and x.min().item() == 0:
                    x = (x != 0).float()
                x = squeeze_tensor_to_shape(x.abs(), self.magnitude.shape)
                t = self.t.item()
                self.magnitude.data[:] = (t * self.magnitude + x) / (t + 1)

    def initialize(self, mask: torch.Tensor):
        if self.running_average:
            self.magnitude = nn.Parameter(
                torch.zeros(*mask.shape, device=mask.device,
                            dtype=torch.float),
                requires_grad=False,
            )

    def forward(self, x: torch.Tensor, sparsity: float, mask: torch.Tensor, name=""):
        if self.training:
            if not self.initted:
                self.initialize(mask)
                self.t.data[:] = 0
                if self.mask_refresh_interval <= 0:
                    self.mask_refresh_interval = 1

            t_item = self.t.item()
            if t_item < self.stop_mask_refresh:
                self.receive_input(x)
            if (
                sparsity >= 0
                and (t_item % self.mask_refresh_interval == 0 and t_item <= self.stop_mask_refresh ) and (t_item > 0 or not self.running_average) 
            ):
                out = self.prune_and_update_mask(x, sparsity, mask)
            else:
                out = x * mask
            self.t += 1
            if self.forward_hook is not None:
                self.forward_hook(mask, name)
            return out
        else:
            return x * mask


class UniformPruningCallback(MagnitudePruningCallback):
    """unstructured uniform pruning function.

    This function will prune uniformly without considering magnitude of the input tensors. If a init mask is provided,
    it will not reactivate those already pruned locations in init mask.
    """

    def initialize(self, mask: torch.Tensor):
        pass

    def receive_input(self, x: torch.Tensor):
        pass

    def prune_and_update_mask(
        self, x: torch.Tensor, sparsity: float, mask: torch.Tensor
    ) -> torch.Tensor:
        if sparsity == 0.5:
            print(None)
        cur_sparsity = (~mask).sum().item() / mask.numel()
        if cur_sparsity > sparsity:
            logging.warning("sparsity is decreasing, which shall not happen")
        budget = int(round((sparsity - cur_sparsity) * np.prod(mask.shape)))
        slots = mask.nonzero(as_tuple=True)
        selected_indexes = np.random.choice(
            range(len(slots[0])), size=budget, replace=False
        )
        mask.data[[slot[selected_indexes] for slot in slots]] = False
        return x * mask




class PruneLayer(nn.Module):
    """Applies pruning over input tensor.
    Please look for detailed description in [prune][qsparse.sparse.prune]
    """

    def __str__(self):
        return f"PruneLayer(sparsity={self.sparsity}, start={self.start}, interval={self.interval}, repetition={self.repetition}, dimensions={self.dimensions})"

    def __repr__(self):
        return str(self)

    def __init__(
        self,
        sparsity: float = 0.5,
        dimensions: Iterable[int] = {1}, 
        callback: MagnitudePruningCallback = MagnitudePruningCallback(),
        # for step-wise training
        start: int = 1000,
        interval: int = 1000,
        repetition: int = 4,
        rampup: bool = False,
        name="",
    ):
        super().__init__()
        if get_option("log_on_created"):
            logging.warning(
                f"[Prune{name if name == '' else f' @ {name}'}] start = {start} interval = {interval} repetition = {repetition} sparsity = {sparsity} dimensions = {dimensions}"
            )

        self.schedules = [
            start + interval * ((1 if rampup else 0) + i) for i in range(repetition)
        ]
        self.start = start
        self.interval = interval
        self.repetition = repetition
        self.sparsity = sparsity
        self.name = name
        self.callback = callback
        self.rampup_interval = 0 if rampup else interval
        self.dimensions = set(dimensions)

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
                mask_shape = [1 if i not in self.dimensions else s
                              for i, s in enumerate(list(x.shape))]
                self.mask = nn.Parameter(
                    torch.ones(
                        *mask_shape,
                        dtype=torch.bool,
                    ).to(x.device),
                    requires_grad=False,
                )
                if self.mask.numel() == 1:
                    logging.warn(f"the mask shape of {self.name} is {tuple(self.mask.shape)}, which is not prunable")
                    
            self._n_updates = nn.Parameter(
                torch.zeros(1, dtype=torch.int).to(x.device),
                requires_grad=False,
            )
            self._cur_sparsity = nn.Parameter(
                torch.zeros(1).to(x.device), requires_grad=False
            )

        if (self._n_updates.item() in self.schedules) and self.training:
            ratio = (
                1.0
                - (self._n_updates.item() - self.start + self.rampup_interval)
                / (self.interval * self.repetition)
            ) ** 3
            self._cur_sparsity[0] = self.sparsity * (1 - ratio)
            logging.warning(
                f"[Prune{self.name if self.name == '' else f' @ {self.name}'}] [Step {self._n_updates.item()}] pruned {self._cur_sparsity.item():.02f}"
            )
        
        if not self.training or self.mask.numel() == 1:
            out = x * self.mask
        else:
            n_updates = self._n_updates.item()
            if n_updates >= self.start:
                if n_updates == self.start:
                    logging.warning(f"Start pruning at {self.name} @ {n_updates}")
                out = self.callback(x, self._cur_sparsity.item(), mask=self.mask, name=self.name)
            else:
                out = x
            self._n_updates += 1
        return out


def prune(
    inp: nn.Module = None,
    sparsity: float = 0.5,
    dimensions: Iterable[int] = {1},
    callback: MagnitudePruningCallback = None,
    # step-wise pruning parameters
    start: int = 1000,
    interval: int = 1000,
    repetition: int = 4,
    rampup: bool = False,
    name=""
) -> nn.Module:
    """Creates a [PruneLayer][qsparse.sparse.PruneLayer] which is usually used
    for feature pruning if no input module is provided, or creates a weight-
    pruned version of the input module.

    Args:
        inp (nn.Module, optional): input module whose weight is to be pruned. Defaults to None.
        sparsity (float, optional): target sparsity. Defaults to 0.5.
        dimensions (Iterable[int]): which dimensions to prune. Defaults to {1}, pruning the channel dimension of conv feature map.
        callback (MagnitudePruningCallback, optional): callback for actual operation of calculating binary mask and prune inputs. Defaults to [MagnitudePruningCallback][qsparse.sparse.MagnitudePruningCallback].
        start (int, optional): starting step to apply pruning. Defaults to 1000.
        interval (int, optional): interval of iterations between each sparsity increasing steps. Defaults to 1000.
        repetition (int, optional): number of sparsity increasing steps. Defaults to 4.
        rampup (bool, optional): whether to wait another interval before starting to prune. Defaults to False.
        name (str, optional): name of the prune layer created, used for better logging. Defaults to "".

    Returns:
        nn.Module: input module with its weight pruned or a instance of [PruneLayer][qsparse.sparse.PruneLayer] for feature pruning
    """
    callback = callback or MagnitudePruningCallback()

    kwargs = dict(
           start=int(start),
            sparsity=sparsity,
            interval=int(interval),
            repetition=repetition,
            rampup=rampup,
            name=name,
            callback=callback,
            dimensions=dimensions
    )

    def get_prune_layer(
    ):
        return PruneLayer(
            start=int(start),
            sparsity=sparsity,
            interval=int(interval),
            repetition=repetition,
            rampup=rampup,
            name=name,
            callback=callback,
            dimensions=dimensions
        )

    if inp is None:
        layer = get_prune_layer()
        setattr(layer, "_kwargs", kwargs)
        return layer
    elif isinstance(inp, nn.Module):
        return imitate(inp, "prune", get_prune_layer())
    else:
        raise ValueError(f"{inp} is not a valid argument for prune")

        

def devise_layerwise_pruning_schedule(net: nn.Module, start:int = 1, interval:int=10, mask_refresh_interval:int = 1, inplace=False):
    if not inplace:
        net = copy.deepcopy(net)
    players = [mod for mod in net.modules() if isinstance(mod, PruneLayer)]
    is_weight_prune = all([p.name.endswith('.prune') for p in players])
    for l in players:
        l.start = start
        l.interval = interval
        l.repetition = 1
        l.schedules = [start, ]
        l.callback.mask_refresh_interval = mask_refresh_interval 
        l.callback.stop_mask_refresh = interval
        if is_weight_prune:
            l.callback.running_average = False
        start += (interval + 1)
    logging.danger(f"Pruning stops at iteration - {start}")
    return net


if __name__ == "__main__":
    print(prune())
    print(prune(torch.nn.Conv2d(10, 30, 3)))
