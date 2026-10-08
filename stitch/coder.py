"""The sparse coder.

A wide layer that takes an activation, represents it as a short list of active
features drawn from an overcomplete dictionary, and reconstructs a target from
those features.

    activation  ->  encoder  ->  many features, few active  ->  decoder  ->  target

The dictionary is larger than the activation is wide, by the expansion factor,
which is what lets individual features be meaningful rather than each neuron
carrying several overlapping things at once. Sparsity is what forces the
representation to commit: without it the encoder can spread an activation
thinly across every feature and reconstruct perfectly while describing nothing.

Input and output widths are separate. When they are equal this is an ordinary
sparse autoencoder reconstructing its own input. When they differ it
reconstructs something else, which is the only structural difference between
the objectives being compared.
"""

from __future__ import annotations

import logging
import math

import torch
from torch import Tensor, nn

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# the coder
# ----------------------------------------------------------------------

class SparseCoder(nn.Module):
    """An overcomplete dictionary with a sparsity constraint.

    Attributes:
        d_in: width of the activation read.
        d_out: width of the activation produced.
        d_latent: dictionary size, ``expansion`` times ``d_in``.
    """

    def __init__(self, d_in: int, d_out: int, expansion: int,
                 sparsity: str = "l1", k: int | None = None,
                 normalize_decoder: bool = True):
        """
        Args:
            d_in: width of the activation read.
            d_out: width of the activation to reconstruct. Equal to ``d_in``
                when the coder reproduces its own input.
            expansion: dictionary size as a multiple of ``d_in``.
            sparsity: ``l1``, which penalises activity, or ``topk``, which keeps
                the k largest and zeroes the rest.
            k: active features per token. Required for ``topk``, ignored for
                ``l1``, where sparsity is a penalty rather than a constraint.
            normalize_decoder: hold every dictionary vector at unit length.

        Raises:
            ValueError: on an unknown sparsity mode, a missing or out of range
                ``k``, or a non-positive width or expansion.
        """
        super().__init__()

        if min(d_in, d_out, expansion) < 1:
            raise ValueError(
                f"widths and expansion must be positive, got "
                f"d_in={d_in}, d_out={d_out}, expansion={expansion}")

        if sparsity not in ("l1", "topk"):
            raise ValueError(f"unknown sparsity mode {sparsity!r}, expected 'l1' or 'topk'")

        self.d_in = d_in
        self.d_out = d_out
        self.d_latent = d_in * expansion
        self.sparsity = sparsity
        self.normalize_decoder = normalize_decoder

        if sparsity == "topk":
            if k is None:
                raise ValueError("topk sparsity needs k, the number of active features")
            if not 1 <= k <= self.d_latent:
                raise ValueError(f"k must be between 1 and {self.d_latent}, got {k}")
        self.k = k

        # Subtracted before encoding. A standard autoencoder reuses the decoder
        # bias here, which only works when the input and the target are the same
        # thing. They are not in general, so the centring term is its own
        # parameter at the input width.
        self.b_pre = nn.Parameter(torch.zeros(d_in))

        self.encoder = nn.Linear(d_in, self.d_latent, bias=True)
        self.decoder = nn.Linear(self.d_latent, d_out, bias=True)

        self.reset_parameters()

    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        """Initialise the dictionary and the encoder.

        Dictionary vectors start at unit length pointing in random directions,
        which is the state the normalisation maintains, so training does not
        begin by undoing the initialisation. The encoder starts at the
        dictionary's transpose scaled down, so that early features respond to
        the directions they will later reconstruct rather than to noise.
        """
        with torch.no_grad():
            weight = torch.randn(self.d_out, self.d_latent)
            weight /= weight.norm(dim=0, keepdim=True)
            self.decoder.weight.copy_(weight)
            self.decoder.bias.zero_()

            scale = 1.0 / math.sqrt(self.d_in)
            if self.d_in == self.d_out:
                self.encoder.weight.copy_(weight.T * scale)
            else:
                nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
            self.encoder.bias.zero_()
            self.b_pre.zero_()

    # ------------------------------------------------------------------

    def encode(self, x: Tensor) -> Tensor:
        """Features for one batch of activations, most of them zero."""
        latents = torch.relu(self.encoder(x - self.b_pre))

        if self.sparsity == "topk":
            # Keep the k largest per token and zero the rest, so sparsity is
            # exact rather than something a penalty has to be tuned towards.
            values, indices = latents.topk(self.k, dim=-1)
            latents = torch.zeros_like(latents).scatter_(-1, indices, values)

        return latents

    def decode(self, latents: Tensor) -> Tensor:
        """Reconstruct from features."""
        return self.decoder(latents)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Returns:
            ``(reconstruction, latents)``. The latents are returned because
            every sparsity metric is computed from them and recomputing them
            would double the encoder's cost.
        """
        latents = self.encode(x)
        return self.decode(latents), latents

    # ------------------------------------------------------------------
    # dictionary normalisation
    #
    # Two halves of one mechanism, both called by the training loop around the
    # optimiser step. Without them an L1 penalty is trivially satisfiable: the
    # encoder shrinks the features and the decoder grows its vectors to
    # compensate, leaving the reconstruction unchanged and the penalty smaller.
    # Nothing has become sparser.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def project_decoder_gradient(self) -> None:
        """Remove the part of the gradient that would change vector lengths.

        Called after ``backward`` and before the optimiser step. Without it the
        step moves the vectors off unit length and the renormalisation
        afterwards silently discards part of the update, which is not the same
        as having optimised on the sphere.
        """
        if not self.normalize_decoder or self.decoder.weight.grad is None:
            return

        weight = self.decoder.weight
        grad = weight.grad
        parallel = (grad * weight).sum(dim=0, keepdim=True) * weight
        grad.sub_(parallel)

    @torch.no_grad()
    def normalize_decoder_columns(self) -> None:
        """Return every dictionary vector to unit length.

        Called after the optimiser step.
        """
        if not self.normalize_decoder:
            return
        self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True) + 1e-8)

    @torch.no_grad()
    def decoder_norms(self) -> Tensor:
        """Length of each dictionary vector."""
        return self.decoder.weight.norm(dim=0)


# ----------------------------------------------------------------------
# dead features
# ----------------------------------------------------------------------

class DeadFeatureTracker(nn.Module):
    """Counts how long each feature has been silent.

    A feature that never fires is dictionary capacity that is not being used,
    and a coder reporting a good reconstruction at a good sparsity while half
    its features are dead is a smaller coder than it claims to be. The fraction
    is therefore a reported metric rather than a diagnostic.

    State is held in a buffer so that it is saved and restored with a
    checkpoint, and a resumed run does not start by calling every feature
    alive.

    Nothing is done about them yet. Dead features are commonly revived by
    periodic resampling or by an auxiliary term that makes them fire; which, if
    either, is needed depends on how large the fraction actually gets, so we will
    measure it first. TODO: add a mechanism to revive dead features if required, 
    and a test that it works.
    """

    def __init__(self, d_latent: int, window: int):
        """
        Args:
            d_latent: dictionary size.
            window: steps of silence after which a feature counts as dead.
        """
        super().__init__()
        if window < 1:
            raise ValueError(f"window must be positive, got {window}")

        self.window = window
        self.register_buffer("steps_since_fired", torch.zeros(d_latent, dtype=torch.long))

    @torch.no_grad()
    def update(self, latents: Tensor) -> None:
        """Record one step's activity.

        Args:
            latents: ``(tokens, d_latent)``. A feature counts as having fired if
                it was non-zero for any token in the batch.
        """
        fired = (latents > 0).any(dim=0)
        self.steps_since_fired += 1
        self.steps_since_fired[fired] = 0

    @property
    def dead_mask(self) -> Tensor:
        """True for each feature silent for longer than the window."""
        return self.steps_since_fired > self.window

    @property
    def dead_fraction(self) -> float:
        """Share of the dictionary currently dead."""
        return float(self.dead_mask.float().mean())


# ----------------------------------------------------------------------
# loss and metrics
# ----------------------------------------------------------------------

def compute_loss(coder: SparseCoder, x: Tensor, target: Tensor,
                 coeff: float) -> tuple[Tensor, dict[str, float]]:
    """One step's loss, with everything worth logging alongside it.

    Args:
        coder: the coder.
        x: ``(tokens, d_in)``, what it reads.
        target: ``(tokens, d_out)``, what it should produce. The same tensor as
            ``x`` when the coder reproduces its own input.
        coeff: weight on the sparsity penalty. Unused under topk, where
            sparsity is enforced rather than encouraged.

    Returns:
        ``(loss, metrics)``. The loss carries gradients; the metrics are plain
        floats, detached, ready to log.
    """
    reconstruction, latents = coder(x)

    mse = torch.nn.functional.mse_loss(reconstruction, target)

    if coder.sparsity == "l1":
        # Weighted by dictionary vector length, so the penalty measures the size
        # of the contribution a feature makes rather than a number the decoder
        # is free to rescale. With normalisation on the weights are one and this
        # is the plain sum. Not decoder_norms(), which is detached for
        # reporting; the penalty has to be able to push back on a vector growing.
        penalty = (latents * coder.decoder.weight.norm(dim=0)).sum(dim=-1).mean()
        loss = mse + coeff * penalty
    else:
        penalty = torch.zeros((), device=x.device)
        loss = mse

    with torch.no_grad():
        residual = target - reconstruction
        variance = target.var(dim=0).sum()
        explained = 1.0 - (residual.var(dim=0).sum() / variance) if variance > 0 else 0.0

        metrics = {
            "loss": float(loss),
            "mse": float(mse),
            "sparsity_penalty": float(penalty),
            "l0": float((latents > 0).float().sum(dim=-1).mean()),
            "explained_variance": float(explained),
        }

    return loss, metrics


# ----------------------------------------------------------------------
# construction from configuration
# ----------------------------------------------------------------------

def build_coder(cfg, d_in: int, d_out: int) -> tuple[SparseCoder, DeadFeatureTracker]:
    """A coder and its tracker, sized from the widths the backbones produce.

    Widths are arguments rather than settings. A coder built at a width the
    model does not produce would fail at the first batch, and a coder built at
    the wrong one of two widths would not fail at all.
    """
    sparsity = cfg.coder.sparsity
    k = sparsity.k
    if sparsity.mode == "topk" and k is None:
        k = max(1, round(sparsity.k_frac * d_in))
    coder = SparseCoder(
        d_in=d_in,
        d_out=d_out,
        expansion=cfg.coder.expansion,
        sparsity=cfg.coder.sparsity.mode,
        k=k,
        normalize_decoder=cfg.coder.sparsity.normalize_decoder,
    )
   
    tracker = DeadFeatureTracker(coder.d_latent, cfg.coder.dead_features.track_window)

    log.info("coder %d -> %d latents -> %d, %s sparsity",
             d_in, coder.d_latent, d_out, coder.sparsity)
    return coder, tracker