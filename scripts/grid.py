#!/usr/bin/env python3
"""Run many configurations and report the spread between them.

    # five seeds, to measure the noise floor
    python scripts/grid.py --config configs/case1_control.yaml \
        --axis train.seed=0,1,2,3,4

    # tune two settings at once
    python scripts/grid.py --config configs/case1_control.yaml \
        --axis coder.expansion=8,16,32 --axis coder.sparsity.coeff=1e-3,5e-4

    # read an earlier grid without running anything
    python scripts/grid.py --results grids/seeds.json --metric eval_explained_variance
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stitch.grid import (  # noqa: E402
    group_spread, parse_axis, read_results, run_grid, table,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="applied to every point")
    parser.add_argument("--axis", dest="axes", action="append", default=[],
                        metavar="KEY=V1,V2", help="a setting and the values it takes")
    parser.add_argument("--out", type=Path, default=Path("grids/grid.json"))
    parser.add_argument("--results", type=Path, default=None,
                        help="read an existing grid instead of running one")
    parser.add_argument("--metric", default="eval_explained_variance")
    parser.add_argument("--across", default="train.seed",
                        help="the axis to collapse when reporting the spread")
    parser.add_argument("--rerun", action="store_true",
                        help="run points that have already been run")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.results is not None:
        results = read_results(args.results)
    else:
        axes = dict(parse_axis(text) for text in args.axes)
        results = run_grid(args.config, args.overrides, axes, args.out,
                           skip_existing=not args.rerun)

    print(f"\nspread of {args.metric} across {args.across}\n")
    print(table(group_spread(results, args.metric, across=args.across)))

    failed = [record["point"] for record in results if "error" in record]
    if failed:
        print(f"\n{len(failed)} points failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())