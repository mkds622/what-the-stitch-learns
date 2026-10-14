#!/usr/bin/env python3
"""Compare what a set of trained dictionaries learned.

Every pair of the runs given is compared, so a set of seed repeats of one
objective gives the ceiling, a set spanning two objectives gives the
comparison, and a set containing both gives all three numbers at once.

    # the ceiling, from the seed repeats of one grid
    python scripts/compare.py --from-grid grids/noise_control_200M_c5e-3_x16.json \
        --out comparisons/control_seeds.json

    # the comparison, one depth, both objectives, every seed
    python scripts/compare.py --from-grid grids/main_case1_add_12.json \
                                          grids/main_case2_add_12.json \
        --out comparisons/main_add_12.json --figure figures/comparison_add_12.png

A run directory name carries the objective, the tag, the expansion, the
sparsity mode, the seed and a hash, but not the coefficient, the budget or the
depth. So a pattern over names cannot separate one grid's runs from another's,
and --from-grid is the reliable form. --runs remains for the cases where a
pattern genuinely is unambiguous.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stitch.compare import (                                          # noqa: E402
    compare_runs, from_grids, load_dictionary, random_baseline, summarise,
    write_comparison,
)

log = logging.getLogger("compare")


def expand(patterns: list[str]) -> list[Path]:
    """Every run directory the patterns name, deduplicated and sorted."""
    found: list[Path] = []
    for pattern in patterns:
        found.extend(Path(path) for path in glob.glob(pattern))

    directories = sorted({path for path in found if path.is_dir()})
    if not directories:
        raise SystemExit(f"no run directories matched {patterns}")
    return directories


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=None,
                        help="run directories, or patterns matching them")
    parser.add_argument("--from-grid", nargs="+", type=Path, default=None,
                        dest="from_grid", metavar="GRID",
                        help="take the runs from these grid result files, which "
                             "is the only reliable way to pick out one grid's runs")
    parser.add_argument("--run-root", type=Path, default=Path("runs"),
                        help="where run directories live, with --from-grid")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--close", type=float, default=0.9,
                        help="cosine above which two features count as the same")
    parser.add_argument("--keep-dead", action="store_true",
                        help="include features that never fired")
    parser.add_argument("--no-baseline", action="store_true",
                        help="skip the random reference, which costs a minute")
    parser.add_argument("--figure", type=Path, default=None,
                        help="also draw the comparison here")
    parser.add_argument("--title", default=None,
                        help="heading for the figure, usually which depth this is")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.runs and not args.from_grid:
        raise SystemExit("give either --runs or --from-grid")

    directories = (from_grids(args.from_grid, args.run_root) if args.from_grid
                   else expand(args.runs))
    log.info("comparing %d runs, %d pairs",
             len(directories), len(directories) * (len(directories) - 1) // 2)

    drawing = args.figure is not None

    records = []
    for a, b in itertools.combinations(directories, 2):
        record = compare_runs(a, b, args.close, drop_dead=not args.keep_dead,
                              keep_distribution=drawing)
        records.append(record)
        log.info("%s vs %s: %.4f", record["a"], record["b"], record["mean"])

    baseline = None
    if not args.no_baseline:
        directions, info = load_dictionary(directories[0], drop_dead=False)
        baseline = random_baseline(directions.shape[0], info["features"],
                                   keep_distribution=drawing)
        log.info("random dictionaries of the same shape: %.4f", baseline["mean"])

    summary = summarise(records, baseline)
    write_comparison(records, summary, args.out)

    print()
    for name in ("random", "within", "across"):
        value = summary.get(name)
        if isinstance(value, dict) and value.get("n"):
            print(f"{name:>8}  {value['mean']:.4f}  "
                  f"+/- {value['std']:.4f}  over {value['n']} pairs")
        elif isinstance(value, float):
            print(f"{name:>8}  {value:.4f}")

    if "position" in summary:
        print(f"\nacross sits {summary['position']:.0%} of the way "
              f"from random to a seed repeat")

    if drawing:
        from stitch.plots import plot_comparison
        plot_comparison(summary, args.figure, title=args.title)
        print(f"figure written to {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())