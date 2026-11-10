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

    # the same, with the whole set of figures rather than the one
    python scripts/compare.py --from-grid grids/main_case1_add_12.json \
                                          grids/main_case2_add_12.json \
        --out comparisons/main_add_12.json --reading encoder \
        --figures figures/add_12 --title "block 12"

Three of the figures draw individual features, which means reading them back
out of the checkpoints of one pair of runs of each kind. Nothing about them is
written to the comparison file, so those three can only be drawn while the
checkpoints are still there. Everything else is drawn from the comparison file
and keeps working after the runs are archived.

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
    by_kind, compare_runs, detail, from_grids, load_dictionary,
    random_baseline, representatives, shares, summarise, write_comparison,
)

log = logging.getLogger("compare")

#: What each figure is called inside the directory ``--figures`` names.
FIGURES = {
    "dictionary": "dictionary.png",
    "pairing": "pairing.png",
    "matrix": "matrix.png",
    "distributions": "distributions.png",
    "shares": "shares.png",
    "position": "position.png",
    "summary": "summary.png",
    "bands": "frequency_bands.png",
}


def expand(patterns: list[str]) -> list[Path]:
    """Every run directory the patterns name, deduplicated and sorted."""
    found: list[Path] = []
    for pattern in patterns:
        found.extend(Path(path) for path in glob.glob(pattern))

    directories = sorted({path for path in found if path.is_dir()})
    if not directories:
        raise SystemExit(f"no run directories matched {patterns}")
    return directories


def _assigned(baseline: dict | None) -> dict | None:
    """The random floor under its one-to-one names, read as ``summarise`` wants it.

    ``summarise`` looks for ``mean`` and ``histogram`` whatever reading it is
    summarising, so the one-to-one floor is handed over under those names
    rather than its own.
    """
    if baseline is None or "assigned_mean" not in baseline:
        return None
    out = {"mean": baseline["assigned_mean"]}
    if "assigned_histogram" in baseline:
        out["histogram"] = baseline["assigned_histogram"]
    return out


def _chain(summary: dict, reading: str, assigning: bool) -> dict:
    """The floor, ceiling and measurement the figures are drawn from.

    The per-feature figures are drawn from the one-to-one pairing, since that
    is the only pairing in which a feature has a single counterpart to draw a
    line to. So the figures that carry numbers are drawn from the same pairing
    when there is one, or the three would not be on one scale.
    """
    if assigning:
        return summary["assigned" if reading == "decoder" else "encoder_assigned"]
    return summary if reading == "decoder" else summary["encoder"]


def _figures(directory: Path, summary: dict, records: list[dict],
             pairs: list[tuple[Path, Path]], reading: str, assigning: bool,
             close: float, title: str | None) -> None:
    """Draw the set.

    Three of the eight need the features themselves rather than their
    summaries, and nothing about those is written to disk, so they are drawn
    from the checkpoints of one pair of runs of each kind. They are the figures
    that stop being drawable once the checkpoints are deleted.
    """
    from stitch.plots import (
        plot_comparison, plot_distributions, plot_feature_squares,
        plot_frequency_bands, plot_match_matrix, plot_match_shares,
        plot_pairing, plot_position,
    )

    directory = Path(directory)
    chain = _chain(summary, reading, assigning)

    plot_comparison(chain, directory / FIGURES["summary"], title=title)
    plot_distributions(chain, directory / FIGURES["distributions"], title=title)
    plot_position(chain, directory / FIGURES["position"], title=title)

    floor = chain.get("random")
    if floor is not None:
        drawn = {kind: shares(chain[f"{kind}_shape"], floor, close)
                 for kind in ("random", "within", "across")
                 if chain.get(f"{kind}_shape")}
        if drawn:
            plot_match_shares(drawn, directory / FIGURES["shares"], title=title)

    if assigning:
        key = "assigned_by_frequency" if reading == "decoder" \
            else "encoder_assigned_by_frequency"
        bands = by_kind(records, key)
        if bands:
            plot_frequency_bands(bands, directory / FIGURES["bands"], title=title)

    chosen = representatives(pairs, records)
    if not chosen:
        return

    detailed = {}
    for kind, (path_a, path_b) in chosen.items():
        log.info("reading features for the %s panels from %s and %s",
                 kind, Path(path_a).name, Path(path_b).name)
        detailed[kind] = detail(path_a, path_b, side=reading)

    plot_feature_squares({kind: value["squares"] for kind, value in detailed.items()},
                         directory / FIGURES["dictionary"], title=title)
    plot_pairing({kind: value["pairs"] for kind, value in detailed.items()},
                 directory / FIGURES["pairing"], title=title)
    plot_match_matrix({kind: value["block"] for kind, value in detailed.items()},
                      directory / FIGURES["matrix"], title=title)


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
    parser.add_argument("--no-assignment", action="store_true",
                        help="skip the one-to-one pairing and its frequency "
                             "bands, leaving only the best match. The pairing "
                             "is the slow half, so this is for looking at a "
                             "large set of comparisons roughly")
    parser.add_argument("--bands", type=int, default=4,
                        help="how many firing-frequency bands to split the "
                             "one-to-one scores into, rarest first")
    parser.add_argument("--figure", type=Path, default=None,
                        help="also draw the combined comparison figure here")
    parser.add_argument("--figures", type=Path, default=None, metavar="DIR",
                        help="also draw the whole set of figures into this "
                             "directory. Three of them read the features "
                             "themselves back out of the checkpoints of one "
                             "pair of runs of each kind, so they can only be "
                             "drawn while those checkpoints are still on disk")
    parser.add_argument("--title", default=None,
                        help="heading for the figure, usually which depth this is")
    parser.add_argument("--reading", choices=("decoder", "encoder"), default="decoder",
                        help="which reading the figure draws. Both are always "
                             "measured and written out; this picks the one the "
                             "picture is of. Use encoder for any comparison "
                             "between the two objectives, where the decoders "
                             "live in different models' spaces and their "
                             "similarity means nothing")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.runs and not args.from_grid:
        raise SystemExit("give either --runs or --from-grid")

    directories = (from_grids(args.from_grid, args.run_root) if args.from_grid
                   else expand(args.runs))
    log.info("comparing %d runs, %d pairs",
             len(directories), len(directories) * (len(directories) - 1) // 2)

    drawing = args.figure is not None or args.figures is not None

    assigning = not args.no_assignment

    # Kept alongside the records, in the same order, because the figures that
    # draw features need the run directories back and a record carries only
    # the names.
    pairs = list(itertools.combinations(directories, 2))

    records = []
    for a, b in pairs:
        record = compare_runs(a, b, args.close, drop_dead=not args.keep_dead,
                              keep_distribution=drawing, assign=assigning,
                              bands=args.bands)
        records.append(record)
        if assigning:
            log.info("%s vs %s: decoder %.4f best / %.4f paired, "
                     "encoder %.4f best / %.4f paired",
                     record["a"], record["b"], record["mean"],
                     record["assigned_mean"], record["encoder_mean"],
                     record["encoder_assigned_mean"])
        else:
            log.info("%s vs %s: decoder %.4f, encoder %.4f", record["a"], record["b"],
                     record["mean"], record["encoder_mean"])

    baseline = encoder_baseline = None
    if not args.no_baseline:
        # Measured at the size the comparisons actually used, which is the live
        # count rather than the configured one. The floor rises with dictionary
        # size, so taking it over the full dictionary would compare the runs
        # against a floor belonging to a larger problem. Averaged over the runs
        # being compared, since their live counts differ.
        directions, info = load_dictionary(directories[0], drop_dead=not args.keep_dead)
        kept = sorted(record[side] for record in records for side in ("a_kept", "b_kept"))
        features = kept[len(kept) // 2] if kept else info["kept"]

        baseline = random_baseline(directions.shape[0], features,
                                   keep_distribution=drawing, assign=assigning)
        log.info("random dictionaries of %d live features: %.4f",
                 features, baseline["mean"])

        # The floor depends on the space and the number of directions in it,
        # not on which side of the coder they came from, so it is the same
        # number wherever the two widths agree. Measured separately anyway,
        # because the two widths are only guaranteed to agree when a coder
        # reproduces its own input.
        encoder_width = records[0]["encoder_width"] if records else directions.shape[0]
        encoder_baseline = (baseline if encoder_width == directions.shape[0]
                            else random_baseline(encoder_width, features,
                                                 keep_distribution=drawing,
                                                 assign=assigning))

    summary = summarise(records, baseline)
    summary["encoder"] = summarise(records, encoder_baseline, key="encoder_mean",
                                   shape_key="encoder_histogram")
    if assigning:
        # The same three-number chain, read off the one-to-one pairing. Kept
        # as its own entry rather than folded in, because a floor, a ceiling
        # and an across number only sit on one scale when all three come from
        # the same pairing.
        summary["assigned"] = summarise(
            records, _assigned(baseline), key="assigned_mean",
            shape_key="assigned_histogram")
        summary["encoder_assigned"] = summarise(
            records, _assigned(encoder_baseline), key="encoder_assigned_mean",
            shape_key="encoder_assigned_histogram")
    write_comparison(records, summary, args.out)

    def report(heading: str, chain: dict) -> None:
        print(f"\n{heading}")
        for name in ("random", "within", "across"):
            value = chain.get(name)
            if isinstance(value, dict) and value.get("n"):
                print(f"{name:>8}  {value['mean']:.4f}  "
                      f"+/- {value['std']:.4f}  over {value['n']} pairs")
            elif isinstance(value, float):
                print(f"{name:>8}  {value:.4f}")
        if "position" in chain:
            print(f"across sits {chain['position']:.0%} of the way "
                  f"from random to a seed repeat")

    report("decoder, what a feature writes out", summary)
    report("encoder, what a feature looks for", summary["encoder"])

    if assigning:
        report("decoder, one-to-one pairing", summary["assigned"])
        report("encoder, one-to-one pairing", summary["encoder_assigned"])

        bands = by_kind(records, "encoder_assigned_by_frequency")
        if bands:
            print("\nencoder, one-to-one, by firing frequency (rarest band first)")
            for name in ("within", "across"):
                if name in bands:
                    print(f"{name:>8}  "
                          + "  ".join(f"{value:.4f}" for value in bands[name]))
            print("A gap present in every band is a difference in what was "
                  "learned.\nA gap only in the rare bands is a difference in "
                  "how heavily the\ndictionary was used.")

    print("\nOnly the encoder reading is comparable across the two objectives:\n"
          "the decoders write into different models' spaces.")

    if args.figure is not None:
        from stitch.plots import plot_comparison
        drawn = summary if args.reading == "decoder" else summary["encoder"]
        plot_comparison(drawn, args.figure, title=args.title)
        print(f"figure written to {args.figure}, drawn from the {args.reading}")

    if args.figures is not None:
        _figures(args.figures, summary, records, pairs, args.reading, assigning,
                 args.close, args.title)
        print(f"figures written to {args.figures}, drawn from the {args.reading}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())