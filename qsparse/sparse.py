import math
import warnings
from collections import deque
from typing import Iterable, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from qsparse.common import PruneCallback
from qsparse.imitation import imitate
from qsparse.util import get_option, logging, nd_slice, wrap_tensor_with_list


def unstructured_uniform_prune_callback(
    inp: List[torch.Tensor], sparsity: float, mask: torch.Tensor = None
) -> torch.Tensor:
    """unstructured uniform pruning function with type signature of [PruneCallback][qsparse.common.PruneCallback].
    This function will prune uniformly without considering magnitude of the input tensors. If a init mask is provided,
    this function will not reactivate those already pruned locations in init mask.

    Args:
        inp (List[torch.Tensor]): input tensor list (see more in [PruneCallback][qsparse.common.PruneCallback])
        sparsity (float): target sparsity
        mask (torch.Tensor, optional): init mask of the pruning procedure. Defaults to None.

    Returns:
        torch.Tensor: binary mask
    """
    inp = wrap_tensor_with_list(inp)
    assert len(inp) >= 1, "no input tensor is provided"
    shape = inp[0].shape
    if mask is not None:
        cur_sparsity = (~mask).sum().item() / mask.numel()
        mask = mask.to("cpu")
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
    inp: List[torch.Tensor], sparsity: float, mask: torch.Tensor = None
) -> torch.Tensor:
    """unstructured pruning function with type signature of [PruneCallback][qsparse.common.PruneCallback].

    Args:
        inp (List[torch.Tensor]): input tensor list (see more in [PruneCallback][qsparse.common.PruneCallback])
        sparsity (float): target sparsity
        mask (torch.Tensor, optional): init mask of the pruning procedure. Defaults to None.

    Returns:
        torch.Tensor: binary mask
    """
    inp = wrap_tensor_with_list(inp)
    inp = torch.cat([v.view(1, *v.shape) for v in inp], dim=0)
    saliency = inp.abs().mean(dim=0)
    values = saliency.flatten().sort()[0]
    n = len(values)
    idx = max(int(sparsity * n - 1), 0)
    threshold = values[idx]
    mask = saliency >= threshold
    return mask


def structured_prune_callback(
    inp: List[torch.Tensor],
    sparsity: float,
    mask: torch.Tensor = None,
    prunable: Union[Iterable[int], int] = {0},
) -> torch.Tensor:
    """structured pruning function with type signature of [PruneCallback][qsparse.common.PruneCallback].

    Args:
        inp (List[torch.Tensor]): input tensor list (see more in [PruneCallback][qsparse.common.PruneCallback])
        sparsity (float): target sparsity
        prunable (Union[Iterable[int], int], optional): dimension indexes that are prunable. Defaults to {0}, which corresponds to channel dimension when batch dimension is not present.
        mask (torch.Tensor, optional): init mask of the pruning procedure. Defaults to None.

    Returns:
        torch.Tensor: binary mask
    """
    inp = wrap_tensor_with_list(inp)
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


class BanditPruning(torch.autograd.Function):
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
        collapse_index: int,
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
        ctx.collapse_index = collapse_index

        dim = cumsum.numel()
        m = int(sparsity * dim)

        # UCBVTune Iteration Equation
        safe_count = count + 0.0001
        mean = cumsum / safe_count
        variance = (cumsum_square / safe_count) - mean ** 2
        if t.item() == 0:
            T = t + 1
        else:
            T = (t + 0.0001) / sparsity
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
            mask_out.data[:] = mask
        else:
            ctx.save_for_backward(
                mask, indexes, cumsum, cumsum_square, count, t, normalizer
            )
        return inp * mask.expand(inp.shape)

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

            grad = grad_output.abs()
            if ctx.collapse_index >= 0:
                grad = grad.mean(ctx.collapse_index)
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


class BanditPruningCallback(nn.Module):
    def __init__(self, exploration_steps: int, collapse_batch_dim: bool = True):
        """Callback to prune the network based on multi-arm bandits algorithms (UCBVTuned is used here)

        Args:
            exploration_steps (int): How many steps used for bandit learning
            collapse_batch_dim (bool, optional): whether treat the first dimension as batch dimension. Defaults to True.
        """
        self.exploration_steps = exploration_steps
        self._collapse = 0 if collapse_batch_dim else -1
        self.mask_shape = None

        for k in [
            "cumsum",
            "cumsum_square",
            "t",
            "count",
            "normalizer",
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
        return self.t.item() != -1

    def forward(self, x: torch.Tensor, sparsity: float, mask: torch.Tensor = None):
        if not self.initted:
            if self._collapse < 0:
                self.mask_shape = tuple(x.shape)
            else:
                self.mask_shape = tuple(x.shape)[1:]

            dim = math.prod(self.mask_shape)
            self.normalizer = nn.Parameter(
                torch.zeros(1).to(x.device), requires_grad=False
            )
            self.t = nn.Parameter(torch.zeros(1).to(x.device), requires_grad=False)
            self.count = nn.Parameter(
                torch.zeros(dim).to(x.device), requires_grad=False
            )
            self.cumsum = nn.Parameter(
                torch.zeros(dim).to(x.device), requires_grad=False
            )
            self.cumsum_square = nn.Parameter(
                torch.zeros(dim).to(x.device), requires_grad=False
            )

        if mask is None:
            return torch.zeros(self.mask_shape)

        deterministic = (not self.training) or (self.t.item() >= self.exploration_steps)
        out = BanditPruning.apply(
            x,
            sparsity,
            self.mask_shape,
            self.cumsum,
            self.cumsum_square,
            self.count,
            self.t,
            self.normalizer,
            self._collapse,  # parameters
            deterministic,
            self.mask,
        )
        if self.training:
            self.t.data += 1
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
        window_size: int = 1,
        strict: bool = True,
        # for customization
        mask_refresh_interval: float = -1,
        callback: PruneCallback = unstructured_prune_callback,
        callback_prune_input: bool = False,
        collapse: int = 0,
        # for debug purpose
        name="",
    ):
        super().__init__()
        if get_option("log_on_created"):
            logging.info(
                f"[Prune{name if name == '' else f' @ {name}'}] start = {start} interval = {interval} repetition = {repetition} sparsity = {sparsity} window_size = {window_size} collapse = {collapse} "
            )
        self.schedules = [start + interval * i for i in range(repetition)]
        self.window = deque(maxlen=window_size)
        self.start = start
        self.interval = interval
        self.mask_refresh_interval = (
            mask_refresh_interval if mask_refresh_interval > 0 else interval
        )
        self.repetition = repetition
        self.sparsity = sparsity
        self.name = name
        self.callback = callback
        self.window_size = window_size
        self._collapse = collapse
        self.strict = strict
        self.callback_prune_input = callback_prune_input

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
                if self.callback_prune_input:
                    m_example = self.callback(x, 0)
                else:
                    m_example = self.callback(
                        x if self._collapse < 0 else x.mean(self._collapse),
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

        def should_update_sparsity():
            t = self._n_updates.item()
            return (t in self.schedules) and self.training

        def time_to_update_mask(offset=0):
            t = self._n_updates.item() - self.start + offset
            remaining_fraction = (
                math.ceil(t / self.mask_refresh_interval)
                - t / self.mask_refresh_interval
            )
            return int(self.mask_refresh_interval * remaining_fraction)

        if should_update_sparsity():
            ratio = (
                1.0
                - (self._n_updates.item() - self.start + self.interval)
                / (self.interval * self.repetition)
            ) ** 3
            self._cur_sparsity[0] = self.sparsity * (1 - ratio)

        if not self.callback_prune_input:
            if self.training:
                if (
                    time_to_update_mask() < self.window_size
                ):  # set `window_size` to 0 will disable windows
                    if self._collapse >= 0:
                        sl = nd_slice(
                            len(x.shape), self._collapse, end=self.window_size
                        )
                        for t in x[sl].abs().detach().split(1):  # type: torch.Tensor
                            self.window.append(t.squeeze(0).to("cpu"))
                    else:
                        self.window.append(x.abs().detach().to("cpu"))
                else:
                    self.window.clear()

                if should_update_sparsity() or time_to_update_mask() == 0:
                    self.mask.data = self.callback(
                        self.window,
                        self._cur_sparsity.item(),
                        mask=self.mask,
                    ).to(x.device)

                    active_ratio = self.mask.sum().item() / self.mask.size().numel()
                    if get_option("log_during_train"):
                        logging.info(
                            f"[Prune{self.name if self.name == '' else f' @ {self.name}'}] [Step {self._n_updates.item()}] active {active_ratio:.02f}, pruned {1 - active_ratio:.02f}, window_size = {len(self.window)}"
                        )

                    if time_to_update_mask(offset=1) > 0:  # proactively free up memory
                        self.window.clear()

            if self.strict or self.training:
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
        if self.training:
            self._n_updates += 1
        return out


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
    mask_refresh_interval: int = -1,
    # for customization
    callback: PruneCallback = unstructured_prune_callback,
    callback_prune_input: bool = False,
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
        window_size (int, optional): number of input tensors used for computing the binary mask. Defaults to 1, means using only current input.
        strict (bool, optional): whether enforcing the shape of the binary mask to be equal to the input tensor. Defaults to True.
                                 When strict=False, it will try to expand the binary mask to matched the input tensor shape during evaluation, useful for tasks whose test images are larger, like super resolution.
        collapse (Union[str, int]): which dimension to ignore when creating binary mask. It is usually set to 0 for the batch dimension during pruning activations, and -1 when pruning weights. Default to "auto", means setting `collapse` automatically based on `inp` parameter.
        mask_refresh_interval (int, optional): interval of iterations between each mask refreshing. Default to -1, will be set to equal with `interval`.
        callback (PruneCallback, optional): callback for actual operation of calculating pruning mask (mask refreshing), used for customization. Defaults to [unstructured\_prune\_callback][qsparse.sparse.unstructured_prune_callback].
        callback_prune_input (bool, optional): whether the callback directly prunes input instead of creating the mask. Defaults to False.
        name (str, optional): name of the prune layer created, used for better logging. Defaults to "".

    Returns:
        nn.Module: input module with its weight pruned or a instance of [PruneLayer][qsparse.sparse.PruneLayer] for feature pruning
    """
    if mask_refresh_interval <= 0:
        mask_refresh_interval = interval

    if callback_prune_input:
        window_size = 0

    kwargs = dict(
        sparsity=sparsity,
        start=start,
        interval=interval,
        repetition=repetition,
        window_size=window_size,
        mask_refresh_interval=mask_refresh_interval,
        strict=strict,
        collapse=collapse,
        callback=callback,
        callback_prune_input=callback_prune_input,
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
            mask_refresh_interval=mask_refresh_interval,
            callback_prune_input=callback_prune_input,
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
