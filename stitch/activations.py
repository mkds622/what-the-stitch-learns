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
the width times four bytes, per layer.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader

from stitch.models import layer_key, make_extractor, parse_layer_key

log = logging.getLogger(__name__)

TokenBatch = dict[str, torch.Tensor]


class ActivationSource(ABC):
    """An endless stream of token batches, one entry per configured layer."""

    @abstractmethod
    def batches(self) -> Iterator[TokenBatch]:
        """Yield ``{layer key: (token_batch, width)}`` forever."""

    @abstractmethod
    def widths(self) -> dict[str, int]:
        """Activation width per layer key, which sets the coder's dimensions."""


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
        """
        self.cfg = cfg
        self.loader = loader
        self.device = torch.device(cfg.train.device)
        self.token_batch = cfg.train.token_batch

        # At least one step's worth, or the buffer cannot fill a batch.
        self.buffer_tokens = max(cfg.activations.shuffle_buffer_tokens, self.token_batch)

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

    # ------------------------------------------------------------------

    def _blocks(self) -> Iterator[TokenBatch]:
        """One entry per image batch, flattened to tokens, forever."""
        while True:
            for images, _ in self.loader:
                images = images.to(self.device, non_blocking=True)
                block = {}
                with torch.no_grad():
                    for label, extractor in self.extractors.items():
                        for node, value in extractor(images).items():
                            block[layer_key(label, node)] = value.reshape(-1, value.shape[-1])
                yield block

    def batches(self) -> Iterator[TokenBatch]:
        """Shuffled token batches, forever."""
        held: list[TokenBatch] = []
        count = 0

        for block in self._blocks():
            if not self._widths:
                self._widths = {key: value.shape[-1] for key, value in block.items()}

            held.append(block)
            count += next(iter(block.values())).shape[0]
            if count < self.buffer_tokens:
                continue

            merged = {key: torch.cat([entry[key] for entry in held]) for key in held[0]}

            # One permutation for every layer, so tokens stay aligned across them.
            order = torch.randperm(count, generator=self.generator)
            merged = {key: value[order] for key, value in merged.items()}

            # The tail shorter than a step is dropped rather than carried over.
            # Every step then sees the same number of tokens, and the loss is a
            # few thousand tokens out of the buffer.
            for start in range(0, count - self.token_batch + 1, self.token_batch):
                stop = start + self.token_batch
                yield {key: value[start:stop] for key, value in merged.items()}

            held, count = [], 0

    def widths(self) -> dict[str, int]:
        """Widths, taken from what the model actually produces.

        Costs one forward pass if nothing has been read yet, and nothing
        afterwards. Reading the configuration instead would let a coder be built
        at a width the backbone does not produce.
        """
        if not self._widths:
            block = next(iter(self._blocks()))
            self._widths = {key: value.shape[-1] for key, value in block.items()}
        return dict(self._widths)


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