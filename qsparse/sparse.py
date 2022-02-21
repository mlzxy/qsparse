import math
import warnings
from collections import deque
from typing import Iterable, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from matplotlib import use

from qsparse.common import PruneCallback
from qsparse.imitation import imitate
from qsparse.util import align_tensor_to_shape, get_option, logging, nd_slice


class MagnitudePruningCallback(nn.Module):
    def __init__(self, mask_refresh_interval: int = -1, use_gradient: bool = False):
        """
        Magnitude-based pruning function with type signature of [PruneCallback][qsparse.common.PruneCallback].

        Args:
            mask_refresh_interval (int, optional): number of steps to refresh mask. Defaults to 1.
            use_gradient (bool, optional): whether use the magnitude of gradients
        """
        super().__init__()
        self.mask_refresh_interval = mask_refresh_interval
        self.use_gradient = use_gradient
        self.t = nn.Parameter(torch.full((1,), -1), requires_grad=False)
        self.prev_hook = None

    @property
    def initted(self) -> bool:
        return self.t.item() != -1

    def prune_and_update_mask(
        self, x: torch.Tensor, sparsity: float, mask: torch.Tensor
    ) -> torch.Tensor:
        importance = self.magnitude
        values = importance.flatten().sort()[0]
        n = len(values)
        idx = max(int(sparsity * n - 1), 0)
        threshold = values[idx]
        mask.data[:] = importance >= threshold
        return self.broadcast_mul(x, mask)

    def broadcast_mul(self, x: torch.Tensor, mask: torch.Tensor):
        return x * mask

    def receive_input(self, x: torch.Tensor):
        if self.use_gradient:
            if self.prev_hook is not None:
                self.prev_hook.remove()
            self.prev_hook = x.register_hook(lambda grad: self.update_magnitude(grad))
        else:
            self.update_magnitude(x)

    def update_magnitude(self, x):
        with torch.no_grad():
            x = align_tensor_to_shape(x.abs(), self.magnitude.shape)
            self.magnitude.data[:] = (self.t * self.magnitude + x) / (self.t + 1)

    def initialize(self, mask: torch.Tensor):
        self.magnitude = nn.Parameter(
            torch.zeros(*mask.shape, device=mask.device, dtype=torch.float),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor, sparsity: float, mask: torch.Tensor):
        if self.training:
            if not self.initted:
                self.initialize(mask)
                self.t.data[:] = 0
                if self.mask_refresh_interval <= 0:
                    self.mask_refresh_interval = 1
            self.receive_input(x)
            if sparsity >= 0 and self.t.item() % self.mask_refresh_interval == 0:
                out = self.prune_and_update_mask(x, sparsity, mask)
            else:
                out = self.broadcast_mul(x, mask)
            self.t += 1
            return out
        else:
            return self.broadcast_mul(x, mask)


class UniformPruningCallback(MagnitudePruningCallback):
    """unstructured uniform pruning function with type signature of [PruneCallback][qsparse.common.PruneCallback].
    This function will prune uniformly without considering magnitude of the input tensors. If a init mask is provided,
    this function will not reactivate those already pruned locations in init mask.
    """

    def initialize(self, mask: torch.Tensor):
        pass

    def receive_input(self, x: torch.Tensor):
        pass

    def prune_and_update_mask(
        self, x: torch.Tensor, sparsity: float, mask: torch.Tensor
    ) -> torch.Tensor:
        cur_sparsity = (~mask).sum().item() / mask.numel()
        if cur_sparsity > sparsity:
            logging.warning("sparsity is decreasing, which shall not happen")
        budget = int((sparsity - cur_sparsity) * np.prod(mask.shape))
        slots = mask.nonzero(as_tuple=True)
        selected_indexes = np.random.choice(
            range(len(slots[0])), size=budget, replace=False
        )
        mask.data[[slot[selected_indexes] for slot in slots]] = False
        return self.broadcast_mul(x, mask)


class BanditPruningFunction(torch.autograd.Function):
    """Pruning method based on multi-arm bandits"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        sparsity: float,
        mask_shape: Tuple[int],
        cumsum: torch.Tensor,  # initialized as all zeros
        cumsum_square: torch.Tensor,  # initialized as all zeros
        count: torch.Tensor,  # initialize as all zeros
        t: torch.Tensor,  # total number of experiments, t/0.8 => real T
        normalizer: torch.Tensor,  # use to normalize gradient distribution
        deterministic: bool,
        mask_out: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for bandit-based pruning algorithm

        Args:
            ctx: pytorch context
            inp (torch.Tensor): input tensor to be pruned
            sparsity (float): target sparsity ratio
            mask_shape (Tuple[int]): shape of the output mask
            cumsum (torch.Tensor): cumulative sum of the cost for each arm / neuron
            deterministic (bool): whether run in a deterministic mode, True will disable bandit parameters updates
            mask_out (torch.Tensor): output binary mask

        Returns:
            torch.Tensor: pruned input
        """

        ctx.sparsity = sparsity
        ctx.deterministic = deterministic

        dim = cumsum.numel()
        m = int(sparsity * dim)

        # UCBVTune Iteration Equation
        safe_count = count + 0.0001
        mean = cumsum / safe_count
        variance = (cumsum_square / safe_count) - mean ** 2
        if t.item() == 0:
            T = t + 1
        else:
            T = t + 0.0001
        variance += torch.sqrt(2.0 * torch.log(T) / safe_count)
        lower_conf_costs = mean - torch.sqrt(torch.log(T) * variance / safe_count)
        lower_conf_costs[count < 1] = -float("inf")

        # select the topk
        indexes = torch.topk(lower_conf_costs, m, largest=False).indices
        mask = torch.ones(dim, device=inp.device, dtype=torch.bool)
        mask[indexes] = 0  # top m -> pruned
        mask = mask.view(mask_shape)
        if deterministic:
            ctx.save_for_backward(mask)
        else:
            ctx.save_for_backward(
                mask, indexes, cumsum, cumsum_square, count, t, normalizer
            )
        mask_out.data[:] = mask
        return inp * mask

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.deterministic:
            mask = ctx.saved_tensors[0]
        else:
            (
                mask,
                indexes,
                cumsum,
                cumsum_square,
                count,
                t,
                normalizer,
            ) = ctx.saved_tensors

            grad = align_tensor_to_shape(grad_output.abs(), mask.shape)
            costs = grad.view(-1)[indexes]

            if normalizer.item() <= 0:  # set one time
                normalizer.data[:] = costs.quantile(0.95)

            costs /= normalizer

            # update bandit parameters
            count[indexes] += 1
            t.data += 1
            cumsum[indexes] += costs
            cumsum_square[indexes] += costs ** 2
        result = grad_output * mask.expand(grad_output.shape)
        return (result,) + (None,) * 10


class BanditPruningCallback(MagnitudePruningCallback):
    def __init__(self, exploration_steps: int = float("inf"), **kwargs):
        """Callback to prune the network based on multi-arm bandits algorithms (UCBVTuned is used here)

        Args:
            exploration_steps (int): How many steps used for bandit learning
            collapse_batch_dim (bool, optional): whether treat the first dimension as batch dimension. Defaults to True.
        """
        if "mask_refresh_interval" not in kwargs:
            kwargs["mask_refresh_interval"] = 1
        super().__init__(**kwargs)
        self.exploration_steps = exploration_steps

    def initialize(self, mask: torch.Tensor):
        device = mask.device
        self.normalizer = nn.Parameter(torch.zeros(1).to(device), requires_grad=False)
        self.ucbT = nn.Parameter(torch.zeros(1).to(device), requires_grad=False)
        self.count = nn.Parameter(
            torch.zeros(*mask.shape).to(device), requires_grad=False
        )
        self.cumsum = nn.Parameter(
            torch.zeros(*mask.shape).to(device), requires_grad=False
        )
        self.cumsum_square = nn.Parameter(
            torch.zeros(*mask.shape).to(device), requires_grad=False
        )

    def receive_input(self, x: torch.Tensor):
        pass

    def prune_and_update_mask(
        self, x: torch.Tensor, sparsity: float, mask: torch.Tensor
    ) -> torch.Tensor:
        deterministic = self.t.item() >= self.exploration_steps
        out = BanditPruningFunction.apply(
            x,
            sparsity,
            tuple(self.cumsum.shape),
            self.cumsum.view(-1),
            self.cumsum_square.view(-1),
            self.count.view(-1),
            self.ucbT,
            self.normalizer,
            deterministic,
            mask,
        )
        return out


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
        strict: bool = True,
        # for customization
        callback: PruneCallback = MagnitudePruningCallback(),
        collapse: Union[int, List[int]] = 0,
        # for debug purpose
        name="",
    ):
        super().__init__()
        if get_option("log_on_created"):
            logging.info(
                f"[Prune{name if name == '' else f' @ {name}'}] start = {start} interval = {interval} repetition = {repetition} sparsity = {sparsity} collapse dimension = {collapse}"
            )
        self.schedules = [start + interval * (1 + i) for i in range(repetition)]
        self.start = start
        self.interval = interval
        self.repetition = repetition
        self.sparsity = sparsity
        self.name = name
        self.callback = callback
        self._collapse = (
            collapse
            if isinstance(collapse, list)
            else [
                collapse,
            ]
        )
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
                self.mask = nn.Parameter(
                    torch.ones(
                        *[
                            1 if i in self._collapse else s
                            for i, s in enumerate(list(x.shape))
                        ],
                        dtype=torch.bool,
                    ).to(x.device),
                    requires_grad=False,
                )
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
                - (self._n_updates.item() - self.start)
                / (self.interval * self.repetition)
            ) ** 3
            self._cur_sparsity[0] = self.sparsity * (1 - ratio)
            logging.info(
                f"[Prune{self.name if self.name == '' else f' @ {self.name}'}] [Step {self._n_updates.item()}] pruned {self._cur_sparsity.item():.02f}"
            )

        if not self.training:
            if self.strict:
                out = x * self.mask.expand(x.shape)
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
                out = x * final_mask
        else:
            if self._n_updates >= self.start:
                out = self.callback(x, self._cur_sparsity.item(), mask=self.mask)
            else:
                out = x
            self._n_updates += 1
        return out


def prune(
    inp: nn.Module = None,
    sparsity: float = 0.5,
    # for step-wise training
    start: int = 1000,
    interval: int = 1000,
    repetition: int = 4,
    strict: bool = True,
    collapse: Union[str, int, List[int]] = "auto",
    # for customization
    callback: PruneCallback = None,
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
        interval (int, optional): interval of iterations between each sparsity increasing steps. Defaults to 1000.
        repetition (int, optional): number of sparsity increasing steps. Defaults to 4.
        strict (bool, optional): whether enforcing the shape of the binary mask to be equal to the input tensor. Defaults to True.
                                 When strict=False, it will try to expand the binary mask to matched the input tensor shape during evaluation, useful for tasks whose test images are larger, like super resolution.
        collapse (Union[str, int, List[int]]): which dimension to ignore when creating binary mask. It is usually set to 0 for the batch dimension during pruning activations, and -1 when pruning weights. Default to "auto", means setting `collapse` automatically based on `inp` parameter.
        callback (PruneCallback, optional): callback for actual operation of calculating pruning mask (mask refreshing), used for customization. Defaults to [unstructured\_prune\_callback][qsparse.sparse.unstructured_prune_callback].
        name (str, optional): name of the prune layer created, used for better logging. Defaults to "".

    Returns:
        nn.Module: input module with its weight pruned or a instance of [PruneLayer][qsparse.sparse.PruneLayer] for feature pruning
    """

    if hasattr(callback, "mask_refresh_interval"):
        if callback.mask_refresh_interval <= 0:
            callback.mask_refresh_interval = interval

    callback = callback or MagnitudePruningCallback()

    kwargs = dict(
        sparsity=sparsity,
        start=start,
        interval=interval,
        repetition=repetition,
        strict=strict,
        collapse=collapse,
        callback=callback,
        name=name,
    )

    def get_prune_layer(
        feature_collapse=[
            0,
        ]
    ):
        if collapse != "auto":
            feature_collapse = collapse
        return PruneLayer(
            start=int(start),
            sparsity=sparsity,
            interval=int(interval),
            repetition=repetition,
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
        return imitate(inp, "prune", get_prune_layer([]))
    else:
        raise ValueError(f"{inp} is not a valid argument for prune")


if __name__ == "__main__":
    print(prune())
    print(prune(torch.nn.Conv2d(10, 30, 3)))
