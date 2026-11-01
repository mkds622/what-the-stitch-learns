"""Where the coder's training data comes from.

The coder trains on activations which can be produced two ways:
1. run the frozen backbone every step. or 
2. run it once, write the activations to disk and stream them back. 

Which is faster depends on the machine, so both sit
behind one interface and the training loop never knows which it got.

Both yield the same thing: a dict of layer key to a tensor of shape
``(token_batch, width)``, as an endless stream. Stopping is the training loop's
job, since the budget is counted in tokens and a backbone produces a different
number of tokens per image depending on what it is.

Shuffling matters more than it looks. Tokens from one image are highly
correlated, and a batch drawn from a handful of images is not a sample of the
distribution the coder is meant to model. Tokens are therefore accumulated into
a buffer and permuted before being handed out. Every layer is permuted by the
same order, so a token in one layer still lines up with the same token in
another, which is what makes a second layer usable as a target at all.

The buffer is the memory cost of this module: ``shuffle_buffer_tokens`` times
the width times four bytes, per layer. It is held once: the blocks are released
as soon as they are concatenated, the scaling is applied in place, and batches
are gathered through the permutation rather than the whole buffer being
permuted into a second copy.

Scaling happens here too. Activations are larger at some depths than others and
differ between models, and the reconstruction error is measured in whatever
units they arrive in. Left alone, one sparsity coefficient would mean a
different trade at every layer and in every case. Each layer is therefore
divided by a scalar fixed once at the start of the run, chosen so a typical
token has the norm an isotropic unit-variance vector of that width would have.

The centre of each layer is measured in the same pass, because the coder needs
somewhere to put the constant part of an activation and cannot work it out for
itself. It is the geometric median rather than the mean. A residual stream
carries a small number of tokens with enormous norm, and the mean is pulled by
them while the median is not: at ``add_12`` on the main pair the two differ by
23 percent of the mean's own length, almost all of it in direction rather than
magnitude. The median is also what the coder's reference uses, for the same
reason.

Centres are returned scaled, like the batches, so they arrive in the units the
coder is trained in. The geometric median commutes with a positive scalar, so
measuring before scaling and scaling afterwards is the same thing.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader

from stitch.models import layer_key, make_extractor, parse_layer_key

log = logging.getLogger(__name__)

TokenBatch = dict[str, torch.Tensor]


def geometric_median(x: torch.Tensor, iterations: int = 50,
                     eps: float = 1e-8) -> torch.Tensor:
    """The point minimising total distance to every row, by Weiszfeld's method.

    Started from the mean, which is the usual choice and converges in a few
    tens of steps on activation-shaped data. ``eps`` keeps a token that sits
    exactly on the current estimate from dividing by zero.

    Args:
        x: ``(tokens, width)``.
        iterations: fixed rather than tolerance-based, so the cost of building
            a coder does not depend on the data.
        eps: floor on the distance used as a weight.

    Returns:
        ``(width,)``, in the units ``x`` arrived in.
    """
    m = x.mean(dim=0)
    for _ in range(iterations):
        distance = (x - m).norm(dim=1).clamp_min(eps)
        weight = 1.0 / distance
        m = (x * weight[:, None]).sum(dim=0) / weight.sum()
    return m


class ActivationSource(ABC):
    """An endless stream of token batches, one entry per configured layer."""

    @abstractmethod
    def batches(self) -> Iterator[TokenBatch]:
        """Yield ``{layer key: (token_batch, width)}`` forever."""

    @abstractmethod
    def widths(self) -> dict[str, int]:
        """Activation width per layer key, which sets the coder's dimensions."""

    @abstractmethod
    def scales(self) -> dict[str, float]:
        """The scalar each layer is multiplied by.

        Recorded with the run, because evaluation has to apply the same numbers
        and anything reported in the original units has to divide them out.
        """

    @abstractmethod
    def set_scales(self, scales: dict[str, float]) -> None:
        """Adopt scales measured elsewhere instead of measuring them.

        An evaluation stream has to use the numbers the training stream
        measured. Measuring its own would put the two in different units, and
        every comparison between them would be off by a factor nobody recorded.
        """

    @abstractmethod
    def centres(self) -> dict[str, torch.Tensor]:
        """The centre of each layer, scaled, for the coder's centring terms.

        One vector per layer key, at that layer's width, in the units
        :meth:`batches` yields. Measured rather than configured, for the same
        reason the widths are: it is a property of the activations at this
        depth in this model.
        """


class OnTheFlyActivations(ActivationSource):
    """Runs the frozen backbones every step.

    No storage and no preparation, at the cost of paying for a forward pass
    through the backbone on every batch. One pass per model captures every layer
    requested from it, so asking for six layers costs what asking for one costs.
    """

    def __init__(self, cfg, models: dict[str, nn.Module], loader: DataLoader):
        """
        Args:
            cfg: the run configuration.
            models: traced backbones by label, matching the labels in
                ``activations.layers``.
            loader: images to run through them.

        Raises:
            KeyError: if a layer names a model that was not supplied.
            ValueError: on an unknown scaling mode.
        """
        self.cfg = cfg
        self.loader = loader
        self.device = torch.device(cfg.train.device)
        self.token_batch = cfg.train.token_batch

        # At least one step's worth, or the buffer cannot fill a batch.
        self.buffer_tokens = max(cfg.activations.shuffle_buffer_tokens, self.token_batch)

        self.scaling = cfg.activations.scaling
        if self.scaling not in ("unit_norm", "none"):
            raise ValueError(
                f"unknown activations.scaling {self.scaling!r}, "
                f"expected 'unit_norm' or 'none'")
        self.scale_tokens = cfg.activations.scale_tokens

        parsed = [parse_layer_key(key) for key in cfg.activations.layers]
        self.extractors: dict[str, nn.Module] = {}
        for label in sorted({label for label, _ in parsed}):
            if label not in models:
                raise KeyError(
                    f"activations.layers names model {label!r}, which was not supplied")
            nodes = [node for other, node in parsed if other == label]
            self.extractors[label] = make_extractor(models[label], nodes).to(self.device).eval()

        self.generator = torch.Generator().manual_seed(cfg.train.seed)
        self._widths: dict[str, int] = {}
        self._scales: dict[str, float] | None = None
        self._centres: dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------

    def _blocks(self) -> Iterator[TokenBatch]:
        """One entry per image batch, flattened to tokens, unscaled."""
        while True:
            for images, _ in self.loader:
                images = images.to(self.device, non_blocking=True)
                block = {}
                with torch.no_grad():
                    for label, extractor in self.extractors.items():
                        for node, value in extractor(images).items():
                            block[layer_key(label, node)] = value.reshape(-1, value.shape[-1])

                if not self._widths:
                    self._widths = {key: value.shape[-1] for key, value in block.items()}
                yield block

    def _measure_scales(self) -> None:
        """Fix one scalar per layer from the opening tokens of the run.

        Measured once rather than per batch, so the units do not drift as
        training proceeds and a checkpoint resumed later means the same thing.
        """
        if self._scales is not None:
            return

        if self.scaling == "none":
            self._scales = {key: 1.0 for key in self.widths()}
            return

        totals: dict[str, float] = {}
        seen = 0
        for block in self._blocks():
            for key, value in block.items():
                totals[key] = totals.get(key, 0.0) + float(value.norm(dim=-1).sum())
            seen += next(iter(block.values())).shape[0]
            if seen >= self.scale_tokens:
                break

        self._scales = {}
        for key, total in totals.items():
            mean_norm = total / seen
            # A layer that is identically zero has no scale worth setting.
            self._scales[key] = math.sqrt(self._widths[key]) / mean_norm if mean_norm > 0 else 1.0

        log.info("activation scales: %s",
                 {key: round(value, 4) for key, value in self._scales.items()})

    def _measure_centres(self) -> None:
        """Fix one centre per layer, over the same tokens the scales used.

        A separate pass from the scales rather than the same one. The median
        needs the tokens all at once rather than as a running sum, so folding
        it into :meth:`_measure_scales` would make every stream pay for the
        memory, including an evaluation stream that has adopted its scales
        from elsewhere and will never be asked for a centre.

        The tokens are held on the host while it runs: ``scale_tokens`` by the
        width by four bytes, per layer, which at the configured hundred
        thousand is a few hundred megabytes, released as soon as it is done.
        """
        if self._centres is not None:
            return

        self._measure_scales()

        held: dict[str, list[torch.Tensor]] = {}
        seen = 0
        for block in self._blocks():
            for key, value in block.items():
                held.setdefault(key, []).append(value.detach().to("cpu", torch.float32))
            seen += next(iter(block.values())).shape[0]
            if seen >= self.scale_tokens:
                break

        # Scaled after the fact rather than before, which gives the same
        # vector, because the geometric median commutes with a positive scalar.
        self._centres = {}
        for key, chunks in held.items():
            tokens = torch.cat(chunks)[:self.scale_tokens]
            chunks.clear()
            self._centres[key] = geometric_median(tokens) * self._scales[key]

        log.info("activation centres, norm: %s",
                 {key: round(float(value.norm()), 4)
                  for key, value in self._centres.items()})

    def batches(self) -> Iterator[TokenBatch]:
        """Shuffled, scaled token batches, forever."""
        self._measure_scales()

        held: list[TokenBatch] = []
        count = 0

        for block in self._blocks():
            held.append(block)
            count += next(iter(block.values())).shape[0]
            if count < self.buffer_tokens:
                continue

            merged = {key: torch.cat([entry[key] for entry in held]) for key in held[0]}

            # Released as soon as they are copied. Holding the blocks while the
            # concatenation exists doubles the buffer for no reason.
            held.clear()

            # Scaled once per buffer rather than once per batch, and in place,
            # since the concatenation is not shared with anything.
            for key, value in merged.items():
                value.mul_(self._scales[key])

            # One permutation for every layer, so tokens stay aligned across
            # them. Batches are gathered through it rather than the whole buffer
            # being permuted first, which would allocate a second copy of it.
            order = torch.randperm(count, generator=self.generator)
            if merged:
                order = order.to(next(iter(merged.values())).device)

            # The tail shorter than a step is dropped rather than carried over.
            # Every step then sees the same number of tokens, and the loss is a
            # few thousand tokens out of the buffer.
            for start in range(0, count - self.token_batch + 1, self.token_batch):
                rows = order[start:start + self.token_batch]
                yield {key: value[rows] for key, value in merged.items()}

            count = 0

    def widths(self) -> dict[str, int]:
        """Widths, taken from what the model actually produces.

        Costs one forward pass if nothing has been read yet, and nothing
        afterwards. Reading the configuration instead would let a coder be built
        at a width the backbone does not produce.
        """
        if not self._widths:
            next(iter(self._blocks()))
        return dict(self._widths)

    def scales(self) -> dict[str, float]:
        """The scalar each layer is multiplied by, measuring it if necessary."""
        self._measure_scales()
        return dict(self._scales)

    def set_scales(self, scales: dict[str, float]) -> None:
        """Adopt scales measured elsewhere instead of measuring them.

        The centres are left alone. An evaluation stream has no use for them,
        since the coder is built once from the training stream, and silently
        rescaling a centre measured against different scalars would be worse
        than having none.
        """
        self._scales = dict(scales)

    def centres(self) -> dict[str, torch.Tensor]:
        """The centre of each layer, scaled, measuring it if necessary."""
        self._measure_centres()
        return dict(self._centres)


def build_source(cfg, models: dict[str, nn.Module], loader: DataLoader) -> ActivationSource:
    """Pick an implementation from ``activations.mode``."""
    mode = cfg.activations.mode

    if mode == "on_the_fly":
        return OnTheFlyActivations(cfg, models, loader)

    if mode == "cached":
        # Writes one pass to activations.cache_dir and streams it back, behind the
        # same interface, so nothing above this function changes.
        raise NotImplementedError("cached activations are not implemented")

    raise ValueError(f"unknown activations.mode {mode!r}, expected 'on_the_fly' or 'cached'")