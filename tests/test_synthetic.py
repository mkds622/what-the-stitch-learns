"""Tests for the covariance-matched control.

The control is only a control if what it generates really does carry the same
second-order structure as the activations it stands in for, and nothing else.
These check that against statistics known by construction: tokens drawn from a
known covariance, folded into moments, and drawn again from those moments must
come back with the covariance they started from.

CPU, small widths, seconds.

    python -m pytest tests/test_synthetic.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from stitch.config import Config                                       # noqa: E402
from stitch.sweep import CrossMoments                                  # noqa: E402
from stitch.synthetic import (                                         # noqa: E402
    GaussianActivations, cholesky_factor, joint_statistics,
)

WIDTH = 5
TOKENS = 200_000
A = "A@add_14"
B = "B@add_14"


def correlated(tokens: int, width: int = WIDTH, seed: int = 0):
    """Tokens with a definite, non-trivial covariance and a non-zero mean."""
    generator = torch.Generator().manual_seed(seed)
    mixing = torch.randn(width, width, generator=generator)
    offset = torch.arange(1.0, width + 1.0)
    return torch.randn(tokens, width, generator=generator) @ mixing + offset


def cross_moments(x: torch.Tensor, y: torch.Tensor) -> CrossMoments:
    moments = CrossMoments(pairs=[(A, B)],
                           widths={A: x.shape[-1], B: y.shape[-1]})
    moments.update({A: x, B: y})
    return moments


def self_moments(x: torch.Tensor) -> CrossMoments:
    moments = CrossMoments(pairs=[(A, A)], widths={A: x.shape[-1]})
    moments.update({A: x})
    return moments


@pytest.fixture
def cfg():
    c = Config()
    c.train.device = "cpu"
    c.train.token_batch = 4096
    c.train.seed = 0
    c.activations.scale_tokens = 8192
    return c


def covariance_of(tokens: torch.Tensor) -> torch.Tensor:
    centred = (tokens - tokens.mean(dim=0)).to(torch.float64)
    return centred.T @ centred / tokens.shape[0]


# ----------------------------------------------------------------------
# the statistics being matched
# ----------------------------------------------------------------------

def test_one_layer_gives_its_own_mean_and_covariance():
    x = correlated(TOKENS)
    mean, covariance = joint_statistics(self_moments(x), [A])

    assert torch.allclose(mean, x.mean(dim=0).to(torch.float64), atol=1e-3)
    assert torch.allclose(covariance, covariance_of(x), atol=1e-3)


def test_two_layers_keep_the_relationship_between_them():
    """The off-diagonal blocks are the thing the coder is asked to exploit, so
    a control that dropped them would control for the wrong thing."""
    x = correlated(TOKENS, seed=1)
    y = x @ torch.randn(WIDTH, WIDTH, generator=torch.Generator().manual_seed(2))

    _, covariance = joint_statistics(cross_moments(x, y), [A, B])
    joint = covariance_of(torch.cat([x, y], dim=1))

    assert covariance.shape == (2 * WIDTH, 2 * WIDTH)
    assert torch.allclose(covariance, joint, atol=1e-2)
    assert covariance[:WIDTH, WIDTH:].abs().max() > 0.1


def test_the_covariance_is_symmetric():
    x = correlated(TOKENS)
    _, covariance = joint_statistics(self_moments(x), [A])
    assert torch.allclose(covariance, covariance.T, atol=1e-12)


def test_a_layer_the_moments_do_not_cover_raises():
    x = correlated(1000)
    with pytest.raises(ValueError, match="do not cover"):
        joint_statistics(self_moments(x), ["A@nowhere"])


def test_a_pair_never_accumulated_together_raises():
    """Both layers are covered, but only on their own. Generating a matched
    pair from two separate marginals would make them independent, which is a
    control for the wrong thing rather than a missing field."""
    moments = CrossMoments(pairs=[(A, A), (B, B)], widths={A: WIDTH, B: WIDTH})
    moments.update({A: correlated(1000), B: correlated(1000, seed=9)})

    joint_statistics(moments, [A])              # each on its own is fine
    with pytest.raises(ValueError, match="no cross term"):
        joint_statistics(moments, [A, B])


def test_no_tokens_raises():
    moments = CrossMoments(pairs=[(A, B)], widths={A: WIDTH, B: WIDTH})
    with pytest.raises(ValueError, match="no tokens"):
        joint_statistics(moments, [A, B])


def test_too_many_layers_raises():
    x = correlated(1000)
    with pytest.raises(ValueError, match="one or two"):
        joint_statistics(self_moments(x), [A, A, A])


# ----------------------------------------------------------------------
# factorising a covariance that is nearly singular
# ----------------------------------------------------------------------

def test_a_factor_reproduces_the_covariance():
    x = correlated(TOKENS)
    _, covariance = joint_statistics(self_moments(x), [A])
    factor = cholesky_factor(covariance)
    assert torch.allclose(factor @ factor.T, covariance, atol=1e-3)


def test_a_singular_covariance_still_factorises():
    """Activations have directions carrying essentially nothing, which an
    undamped decomposition fails on rather than approximating."""
    basis = torch.randn(WIDTH, 2, generator=torch.Generator().manual_seed(3))
    singular = (basis @ basis.T).to(torch.float64)

    factor = cholesky_factor(singular)
    assert torch.allclose(factor @ factor.T, singular, atol=1e-4)


def test_something_that_is_not_a_covariance_raises():
    with pytest.raises(ValueError, match="could not be factorised"):
        cholesky_factor(torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64))


# ----------------------------------------------------------------------
# the stream
# ----------------------------------------------------------------------

def drain(source, batches: int) -> dict[str, torch.Tensor]:
    """Several batches, concatenated per layer."""
    held: dict[str, list[torch.Tensor]] = {}
    stream = source.batches()
    for _ in range(batches):
        for key, value in next(stream).items():
            held.setdefault(key, []).append(value)
    return {key: torch.cat(values) for key, values in held.items()}


def test_generated_tokens_carry_the_covariance_they_were_matched_to(cfg):
    """The whole point: same second-order structure, nothing else in it."""
    cfg.activations.scaling = "none"
    x = correlated(TOKENS)
    source = GaussianActivations(cfg, self_moments(x), [A])

    drawn = drain(source, batches=20)[A]
    assert torch.allclose(covariance_of(drawn), covariance_of(x), rtol=0.1, atol=0.1)
    assert torch.allclose(drawn.mean(dim=0), x.mean(dim=0), atol=0.1)


def test_the_relationship_between_two_layers_survives(cfg):
    cfg.activations.scaling = "none"
    x = correlated(TOKENS, seed=4)
    y = x @ torch.randn(WIDTH, WIDTH, generator=torch.Generator().manual_seed(5))
    source = GaussianActivations(cfg, cross_moments(x, y), [A, B])

    drawn = drain(source, batches=20)
    real = covariance_of(torch.cat([x, y], dim=1))[:WIDTH, WIDTH:]
    made = covariance_of(torch.cat([drawn[A], drawn[B]], dim=1))[:WIDTH, WIDTH:]

    assert torch.allclose(made, real, rtol=0.15, atol=0.15)


def test_the_stream_yields_the_layers_it_was_asked_for(cfg):
    x = correlated(10_000)
    y = correlated(10_000, seed=6)
    source = GaussianActivations(cfg, cross_moments(x, y), [A, B])

    batch = next(source.batches())
    assert set(batch) == {A, B}
    assert batch[A].shape == (cfg.train.token_batch, WIDTH)


def test_widths_come_from_the_moments(cfg):
    x = correlated(10_000)
    source = GaussianActivations(cfg, self_moments(x), [A])
    assert source.widths() == {A: WIDTH}


def test_the_stream_does_not_end(cfg):
    x = correlated(10_000)
    stream = GaussianActivations(cfg, self_moments(x), [A]).batches()
    for _ in range(50):
        assert next(stream)[A].shape[0] == cfg.train.token_batch


def test_the_same_seed_draws_the_same_tokens(cfg):
    x = correlated(10_000)
    first = next(GaussianActivations(cfg, self_moments(x), [A]).batches())[A]
    second = next(GaussianActivations(cfg, self_moments(x), [A]).batches())[A]
    assert torch.equal(first, second)


def test_a_different_seed_draws_different_tokens(cfg):
    x = correlated(10_000)
    first = next(GaussianActivations(cfg, self_moments(x), [A]).batches())[A]
    cfg.train.seed = 1
    second = next(GaussianActivations(cfg, self_moments(x), [A]).batches())[A]
    assert not torch.equal(first, second)


# ----------------------------------------------------------------------
# units
#
# A control is compared against the run it controls, so one coefficient has to
# mean the same amount of pressure in both. That holds only if the two streams
# are scaled by the same rule.
# ----------------------------------------------------------------------

def test_scaling_puts_tokens_at_the_width_s_root(cfg):
    x = correlated(50_000)
    source = GaussianActivations(cfg, self_moments(x), [A])

    drawn = drain(source, batches=10)[A]
    assert float(drawn.norm(dim=-1).mean()) == pytest.approx(WIDTH ** 0.5, rel=0.1)


def test_scaling_can_be_turned_off(cfg):
    cfg.activations.scaling = "none"
    x = correlated(10_000)
    assert GaussianActivations(cfg, self_moments(x), [A]).scales() == {A: 1.0}


def test_an_unknown_scaling_raises(cfg):
    cfg.activations.scaling = "whiten"
    x = correlated(1000)
    with pytest.raises(ValueError, match="unknown activations.scaling"):
        GaussianActivations(cfg, self_moments(x), [A])


def test_scales_can_be_adopted_instead_of_measured(cfg):
    """An evaluation stream has to use the numbers the training stream found."""
    x = correlated(10_000)
    source = GaussianActivations(cfg, self_moments(x), [A])
    source.set_scales({A: 2.5})
    assert source.scales() == {A: 2.5}


def test_a_control_lands_in_the_same_units_as_the_run_it_controls(cfg):
    """Both streams measure their scale by the same rule, and the two streams
    match in distribution, so the numbers agree."""
    x = correlated(100_000)
    real_scale = WIDTH ** 0.5 / float(x.norm(dim=-1).mean())

    made = GaussianActivations(cfg, self_moments(x), [A]).scales()[A]
    assert made == pytest.approx(real_scale, rel=0.05)