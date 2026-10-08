"""Ranking layer pairs before any coder is trained.

A sparse coder is expensive. Training one at a depth where no map of any kind
connects the two models wastes the whole run, and the dictionary it learns
describes nothing, because there was nothing there to describe.

A linear map is cheap and has a closed form. If the best possible linear map
from one layer to another explains almost none of the target's variance, a
sparse one will not do better by enough to matter. So every candidate pair is
fitted linearly first and ranked, and the training budget goes to the depths
that survive.

The fit never holds the tokens. It accumulates the sums a least squares
solution needs, which are fixed in size, so the number of tokens it is fitted
over costs time and no memory:

    sum of x, sum of x x^T, sum of y, sum of y^2, and sum of x y^T per pair

Memory is the pairs, not the data. Each cross term is one width by the other,
which at 768 by 768 in double precision is 4.7 MB. Matched depths only is
twelve of those; every pair against every other is 144.

Scaling is not applied here. The score is a ratio of variances, so it is
unchanged by scaling either side.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# what to compare
# ----------------------------------------------------------------------

def candidate_pairs(nodes_a: list[str], nodes_b: list[str],
                    matched_depth_only: bool) -> list[tuple[str, str]]:
    """Which layer of one model to compare against which layer of the other.

    Args:
        nodes_a: cut points in the model being read, in depth order.
        nodes_b: cut points in the model being reconstructed, in depth order.
        matched_depth_only: compare block i against block i only. Otherwise
            every layer against every layer, which is the square of the count
            and answers whether the models align at a shift.

    Raises:
        ValueError: if matched depths are asked for and the models have
            different numbers of blocks, which leaves "matched" undefined.
    """
    if not matched_depth_only:
        return [(a, b) for a in nodes_a for b in nodes_b]

    if len(nodes_a) != len(nodes_b):
        raise ValueError(
            f"matched depths need the same number of blocks, got "
            f"{len(nodes_a)} and {len(nodes_b)}")
    return list(zip(nodes_a, nodes_b))


# ----------------------------------------------------------------------
# the sums a least squares fit needs
# ----------------------------------------------------------------------

@dataclass
class CrossMoments:
    """Running sums over tokens, fixed in size however many tokens there are.

    Held in double precision. The sums run over hundreds of thousands of
    tokens, and in single precision the later terms stop changing the total
    once it is large enough, which quietly biases the fit.
    """

    pairs: list[tuple[str, str]]
    widths: dict[str, int]
    device: torch.device = torch.device("cpu")

    sum_x: dict[str, Tensor] = field(default_factory=dict)
    sum_xx: dict[str, Tensor] = field(default_factory=dict)
    sum_y: dict[str, Tensor] = field(default_factory=dict)
    sum_y2: dict[str, Tensor] = field(default_factory=dict)
    sum_xy: dict[tuple[str, str], Tensor] = field(default_factory=dict)
    tokens: int = 0

    def __post_init__(self):
        zeros = lambda *shape: torch.zeros(*shape, dtype=torch.float64, device=self.device)
        for source, target in self.pairs:
            self.sum_x.setdefault(source, zeros(self.widths[source]))
            self.sum_xx.setdefault(source, zeros(self.widths[source], self.widths[source]))
            self.sum_y.setdefault(target, zeros(self.widths[target]))
            self.sum_y2.setdefault(target, zeros(self.widths[target]))
            self.sum_xy.setdefault(
                (source, target), zeros(self.widths[source], self.widths[target]))

    @torch.no_grad()
    def update(self, activations: dict[str, Tensor]) -> None:
        """Fold one batch of tokens in.

        Args:
            activations: ``{node name: (tokens, width)}``, every node named by
                any pair.
        """
        values = {name: activations[name].to(self.device, torch.float64)
                  for name in set(self.sum_x) | set(self.sum_y)}

        for name, tensor in values.items():
            if name in self.sum_x:
                self.sum_x[name] += tensor.sum(dim=0)
                self.sum_xx[name] += tensor.T @ tensor
            if name in self.sum_y:
                self.sum_y[name] += tensor.sum(dim=0)
                self.sum_y2[name] += (tensor ** 2).sum(dim=0)

        for source, target in self.pairs:
            self.sum_xy[(source, target)] += values[source].T @ values[target]

        self.tokens += next(iter(values.values())).shape[0]


def fit_linear(moments: CrossMoments, source: str, target: str,
               ridge: float) -> tuple[Tensor, Tensor, float]:
    """The best linear map from one layer to the other, in closed form.

    Centres both sides, solves the normal equations, and reports how much of
    the target's variance the result accounts for.

    Args:
        moments: the accumulated sums.
        source: the layer read.
        target: the layer reconstructed.
        ridge: added to the diagonal, as a fraction of its mean. An activation
            covariance is close to singular, since a few directions carry most
            of the variance and the rest carry almost none, and solving without
            it produces a map fitted to noise in those directions.

    Returns:
        ``(weight, bias, explained_variance)``. The map is
        ``x @ weight + bias``.
    """
    if moments.tokens == 0:
        raise ValueError("no tokens were accumulated")

    n = moments.tokens
    mean_x = moments.sum_x[source] / n
    mean_y = moments.sum_y[target] / n

    # Centred second moments.
    sxx = moments.sum_xx[source] - n * torch.outer(mean_x, mean_x)
    sxy = moments.sum_xy[(source, target)] - n * torch.outer(mean_x, mean_y)
    syy = float((moments.sum_y2[target] - n * mean_y ** 2).sum())

    damping = ridge * float(torch.diagonal(sxx).mean())
    sxx = sxx + damping * torch.eye(sxx.shape[0], dtype=sxx.dtype, device=sxx.device)

    weight = torch.linalg.solve(sxx, sxy)
    bias = mean_y - mean_x @ weight

    # Residual without forming predictions: the explained part of the total.
    explained = float((sxy * weight).sum())
    score = explained / syy if syy > 0 else 0.0

    return weight.to(torch.float32), bias.to(torch.float32), score


class LinearStitch(nn.Module):
    """A fitted linear map, shaped like a coder so it can be evaluated like one.

    Lets the same substitution measurement run on the linear reference and on a
    trained coder, which is the only way the two numbers are comparable.
    """

    def __init__(self, weight: Tensor, bias: Tensor):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        output = x @ self.weight + self.bias
        return output, torch.zeros(x.shape[0], 0, device=x.device)


# ----------------------------------------------------------------------
# the sweep
# ----------------------------------------------------------------------

def accumulate(extractor, loader, pairs, widths, fit_tokens: int,
               device: torch.device | str = "cpu") -> CrossMoments:
    """Run images through the models until enough tokens have been seen.

    Args:
        extractor: a callable returning ``{node name: activation}`` for every
            node any pair names, in one pass.
        loader: images.
        pairs: the pairs being fitted.
        widths: width per node name.
        fit_tokens: stop once this many tokens have been folded in.
        device: where to accumulate.

    Returns:
        The moments, ready to fit from.
    """
    moments = CrossMoments(pairs=pairs, widths=widths, device=torch.device(device))

    with torch.no_grad():
        for images, _ in loader:
            if moments.tokens >= fit_tokens:
                break
            captured = extractor(images.to(device))
            moments.update({name: value.reshape(-1, value.shape[-1])
                            for name, value in captured.items()})

    if moments.tokens == 0:
        raise ValueError("the loader produced no images")

    log.info("fitted over %d tokens", moments.tokens)
    return moments


def rank(moments: CrossMoments, ridge: float) -> list[dict]:
    """Every pair, scored and sorted best first."""
    records = []
    for source, target in moments.pairs:
        _, _, score = fit_linear(moments, source, target, ridge)
        records.append({"source": source, "target": target,
                        "explained_variance": score})

    records.sort(key=lambda record: record["explained_variance"], reverse=True)
    for position, record in enumerate(records):
        record["rank"] = position
    return records


def write_ranking(records: list[dict], path: Path, context: dict | None = None) -> None:
    """Write the ranking where the training grid will read it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"context": context or {}, "ranking": records}, indent=2))
    log.info("wrote %d pairs to %s", len(records), path)


def read_ranking(path: Path) -> list[dict]:
    """The ranking, best first."""
    return json.loads(Path(path).read_text())["ranking"]


def top_pairs(path: Path, count: int) -> list[tuple[str, str]]:
    """The best few pairs, as layer names.

    What the training grid consumes. The full ranking stays on disk, so a
    decision to take something other than the top few needs no refit.
    """
    records = read_ranking(path)[:count]
    return [(record["source"], record["target"]) for record in records]


# ----------------------------------------------------------------------
# driving it
# ----------------------------------------------------------------------

def run_sweep(cfg) -> list[dict]:
    """Fit and rank every candidate pair, and write the ranking.

    Returns:
        The ranking, best first, also written to ``sweep.out_file``.
    """
    from stitch.data import build_dataset, build_loader, build_transform
    from stitch.models import block_outputs, layer_key, load_backbone, make_extractor, trace

    if cfg.models.b is None:
        raise ValueError("a sweep compares two models, so models.b must be set")

    specs = {"A": cfg.models.a, "B": cfg.models.b}
    backbones, models = {}, {}
    for label, spec in specs.items():
        backbones[label] = load_backbone(spec.name, pretrained=spec.pretrained,
                                         checkpoint=spec.checkpoint, source=spec.source)
        models[label] = trace(backbones[label])
        log.info("loaded %s as %s", spec.name, label)

    transform = build_transform(backbones["A"])
    loader = build_loader(cfg, build_dataset(cfg, cfg.dataset.split, transform),
                          cfg.activations.extraction_batch_size, shuffle=True)

    probe = next(iter(loader))[0][:2]
    nodes, widths = {}, {}
    for label, model in models.items():
        found = block_outputs(model, probe)
        if cfg.sweep.depths is not None:
            found = [found[index] for index in cfg.sweep.depths]
        nodes[label] = [layer_key(label, info.name) for info in found]
        widths.update({layer_key(label, info.name): info.shape[-1] for info in found})

    pairs = candidate_pairs(nodes["A"], nodes["B"], cfg.layer_pairs.matched_depth_only)
    log.info("sweeping %d pairs over %d tokens", len(pairs), cfg.sweep.fit_tokens)

    extractors = {label: make_extractor(model, [key.split("@", 1)[1] for key in nodes[label]])
                  for label, model in models.items()}

    def extract(images):
        """Both models, one pass each, keyed so the two cannot collide."""
        out = {}
        for label, extractor in extractors.items():
            for node, value in extractor(images).items():
                out[layer_key(label, node)] = value
        return out

    moments = accumulate(extract, loader, pairs, widths,
                         cfg.sweep.fit_tokens, cfg.train.device)
    records = rank(moments, cfg.sweep.ridge)

    write_ranking(records, Path(cfg.sweep.out_file), context={
        "models": {label: spec.name for label, spec in specs.items()},
        "dataset": cfg.dataset.name,
        "tokens": moments.tokens,
        "ridge": cfg.sweep.ridge,
        "matched_depth_only": cfg.layer_pairs.matched_depth_only,
    })
    return records