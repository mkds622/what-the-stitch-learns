#!/usr/bin/env python3
"""Plot a grid's results.

    # where the curve flattens, with the tolerance taken from a seed grid
    python scripts/plot.py --results grids/budget.json \
        --x train.tokens --metric eval_l0 \
        --noise grids/noise_floor.json --out figures/budget_l0.png

    # reconstruction against sparsity, one point per run
    python scripts/plot.py --results grids/coder.json --tradeoff \
        --label-by coder.expansion --out figures/tradeoff.png
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stitch.grid import read_results                                  # noqa: E402
from stitch.plots import (                                            # noqa: E402
    flattening_point, format_x, noise_from, plot_curve, plot_tradeoff, series,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--x", default="train.tokens")
    parser.add_argument("--metric", default="eval_l0")
    parser.add_argument("--noise", type=Path, default=None,
                        help="a seed grid, whose spread becomes the tolerance")
    parser.add_argument("--tolerance", type=float, default=None,
                        help="the tolerance directly, instead of from a seed grid")
    parser.add_argument("--tradeoff", action="store_true",
                        help="reconstruction against sparsity instead of a curve")
    parser.add_argument("--label-by", default=None)
    parser.add_argument("--linear-x", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = read_results(args.results)

    if args.tradeoff:
        plot_tradeoff(results, args.out, label_by=args.label_by)
        return 0

    tolerance = args.tolerance
    if tolerance is None and args.noise is not None:
        tolerance = noise_from(read_results(args.noise), args.metric)
        if tolerance is None:
            print("the noise grid has no repeated configurations, so it sets no tolerance")
        else:
            print(f"tolerance from {args.noise}: {tolerance:.5g}")

    points = series(results, args.x, args.metric)
    for point in points:
        spread = "" if point.std is None else f"  +/- {point.std:.5g}"
        print(f"  {args.x}={format_x(point.x):<14} "
              f"{args.metric}={point.mean:.5g}{spread}  n={point.n}")

    if tolerance is not None:
        flat = flattening_point(points, tolerance)
        print(f"\nflattens at {format_x(flat)}" if flat is not None
              else "\nstill moving at the largest value measured")

    plot_curve(points, args.metric, args.x, args.out,
               tolerance=tolerance, log_x=not args.linear_x)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())