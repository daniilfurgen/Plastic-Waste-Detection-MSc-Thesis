"""
Export consolidated SSOD results as raw-number CSV tables for the Typst thesis.

Reads the per-run `results.csv` files written by the Mean Teacher trainer (see ABLATION.md) and the
naive self-training results, and writes tidy CSVs into `paper/exp-results/`. Values are unrounded
apart from 4 decimals; formatting and labels are left to Typst. Missing runs are skipped with a
warning, so the export can be re-run as batches finish.

Usage:
  uv run python code/ssod/export_results.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import statistics

import polars as pl
import typer
import yaml

try:
    from ssod.utilities import get_logger
except ImportError:
    from utilities import get_logger  # type: ignore[import-not-found,no-redef]

MT = Path("runs/mean_teacher")
NAIVE_SELF_TRAINING = Path("paper/exp-results/noisy_student.yaml")
FULL_POOL = 27302  # unlabeled video frames; used as the cap value for "whole pool" runs
LATE_EPOCHS = 8  # "late mean" = mean of epochs 8..last

METRICS = {  # output column -> results.csv column
    "map50_95": "coco/mAP_50_95",
    "map50": "coco/mAP_50",
    "map75": "coco/mAP_75",
    "map_s": "coco/mAP_s",
    "map_m": "coco/mAP_m",
    "map_l": "coco/mAP_l",
    "ar100": "coco/AR_100",
    "ar_s": "coco/AR_s",
    "ar_m": "coco/AR_m",
    "ar_l": "coco/AR_l",
    "f1": "coco/peak_f1",
    "precision": "coco/peak_precision",
    "recall": "coco/peak_recall",
}
CURVE_METRICS = {
    "map50_95": "coco/mAP_50_95",
    "map50": "coco/mAP_50",
    "map75": "coco/mAP_75",
    "map_s": "coco/mAP_s",
    "f1": "coco/peak_f1",
    "recall": "metrics/recall(B)",
    "precision": "metrics/precision(B)",
}

app = typer.Typer(
    name="export-results",
    help="Export consolidated SSOD results as CSV tables for the thesis.",
    add_completion=False,
)
logger = get_logger()


@dataclass
class Run:
    """One SSL training run and the settings it varies."""

    run_id: str
    path: Path
    warmup: str = "adamw50"  # adamw50: untuned 50-epoch warm-up; tuned: tuned MuSGD YOLO26m
    pl_cap: int = 4800  # 0 = no pseudo-labels (GT-only control); FULL_POOL = whole pool
    soft_weights: bool = True
    ema: float = 0.9996
    variant: str = ""  # any other change, e.g. "lr0=5e-5"


# 15-epoch YOLO26m runs, in the order they were explored (see ABLATION.md).
DEVELOPMENT_RUNS = [
    Run("naive-full-pool", MT / "ablation/control", pl_cap=FULL_POOL, soft_weights=False, ema=0.999),
    Run("lambda-0.2", MT / "ablation/lambda-0.2", pl_cap=FULL_POOL, soft_weights=False, ema=0.999, variant="lambda_u=0.2"),
    Run("cap-4800", MT / "ablation/cap", soft_weights=False, ema=0.999),
    Run("cap-4800-soft", MT / "ablation/cap-soft", ema=0.999),
    Run("cap-4800-soft-strong-aug", MT / "ablation/cap-soft-strong", ema=0.999, variant="strong_aug"),
    Run("cap-4800-soft-lr-5e-5", MT / "ablation-drift/lr-5e-5", ema=0.999, variant="lr0=5e-5"),
    Run("cap-4800-soft-cls-only", MT / "ablation-drift/cls-only", ema=0.999, variant="cls_only"),
    Run("cap-4800-soft-ema-0.9996", MT / "ablation-drift/ema-0.9996"),
    Run("cap-4800-soft-ema-0.9998", MT / "ablation-settle/ema-0.9998", ema=0.9998),
    Run("cap-4800-soft-ema-0.9996-mosaic-on", MT / "ablation-settle/ema-0.9996-mosaic-on", variant="mosaic_on"),
    Run("gt-only-4800-matched", MT / "ablation-settle/gt-only-control", pl_cap=0, soft_weights=False),
    Run("cap-2400-soft-ema-0.9996", MT / "ablation-settle/ema-0.9996-cap-2400", pl_cap=2400),
    Run("cap-1200-soft-ema-0.9996", MT / "ablation-architectures/yolo26-cap1200", pl_cap=1200),
    Run("gt-only-2400-matched", MT / "ablation-architectures-control/yolo26-gt-only", pl_cap=0, soft_weights=False),
]

COMPONENT_RUNS = [
    Run("final", MT / "ablation-components/final", warmup="tuned", pl_cap=2400),
    Run("no-cap", MT / "ablation-components/no-cap", warmup="tuned", pl_cap=FULL_POOL),
    Run("no-soft-weights", MT / "ablation-components/no-soft-weights", warmup="tuned", pl_cap=2400, soft_weights=False),
    Run("fast-ema", MT / "ablation-components/fast-ema", warmup="tuned", pl_cap=2400, ema=0.999),
    Run("gt-only", MT / "ablation-components/gt-only", warmup="tuned", pl_cap=0, soft_weights=False),
]

# Leave-one-out of the pseudo-labelling and sampling mechanisms, from the untuned warm-up.
MECHANISM_RUNS = [
    Run("final", MT / "ablation-settle/ema-0.9996-cap-2400", pl_cap=2400),
    Run("final-rerun", MT / "ablation-mechanisms/final-rerun", pl_cap=2400, variant="rerun"),
    Run("no-size-bins", MT / "ablation-mechanisms/no-size-bins", pl_cap=2400, variant="no_size_bins"),
    Run("single-view", MT / "ablation-mechanisms/single-view", pl_cap=2400, variant="single_view"),
    Run("no-gt-tail", MT / "ablation-mechanisms/no-gt-tail", pl_cap=2400, variant="no_gt_tail"),
    Run("no-soft-weights", MT / "ablation-mechanisms/no-soft-weights", pl_cap=2400, soft_weights=False),
    Run("gt-only", MT / "ablation-architectures-control/yolo26-gt-only", pl_cap=0, soft_weights=False),
]

# Add-one-in: a stripped configuration sharing only the 2,400 cap, plus each component alone.
_MINIMAL = dict(pl_cap=2400, soft_weights=False, ema=0.999)
ADD_ONE_IN_RUNS = [
    Run("minimal", MT / "ablation-add-one-in/minimal", **_MINIMAL),
    Run("minimal-rerun", MT / "ablation-add-one-in/minimal-rerun", **_MINIMAL, variant="rerun"),
    Run("plus-slow-ema", MT / "ablation-add-one-in/plus-slow-ema", **{**_MINIMAL, "ema": 0.9996}),
    Run("plus-soft-weights", MT / "ablation-add-one-in/plus-soft-weights", **{**_MINIMAL, "soft_weights": True}),
    Run("plus-size-bins", MT / "ablation-add-one-in/plus-size-bins", **_MINIMAL, variant="size_bins"),
    Run("plus-multi-view", MT / "ablation-add-one-in/plus-multi-view", **_MINIMAL, variant="multi_view"),
    Run("plus-gt-tail", MT / "ablation-add-one-in/plus-gt-tail", **_MINIMAL, variant="gt_tail"),
    Run("final", MT / "ablation-settle/ema-0.9996-cap-2400", pl_cap=2400),
    Run("gt-only", MT / "ablation-architectures-control/yolo26-gt-only", pl_cap=0, soft_weights=False),
]

# Long runs before the ablation (50-100 epochs) that show the original degradation.
LONG_RUNS = [
    Run("v2-pre-fix", MT / "ssod-v2-100", pl_cap=FULL_POOL, soft_weights=False, ema=0.999,
        variant="threshold_bug+e2e_gain_bug"),
    Run("v3-full-pool", MT / "ssod-v3-50-50", pl_cap=FULL_POOL, soft_weights=False, ema=0.999),
]


def load(run: Run) -> pl.DataFrame | None:
    results = run.path / "results.csv"
    if not results.is_file():
        logger.warning(f"Missing {results}; skipping {run.run_id}.")
        return None
    return pl.read_csv(results)


def settings(run: Run) -> dict[str, object]:
    return {
        "run": run.run_id,
        "warmup": run.warmup,
        "pl_cap": run.pl_cap,
        "soft_weights": int(run.soft_weights),
        "ema": run.ema,
        "variant": run.variant,
    }


def metric_row(row: dict[str, object], columns: dict[str, str]) -> dict[str, float | None]:
    return {
        out: (round(float(row[src]), 4) if row.get(src) not in (None, "") else None)
        for out, src in columns.items()
    }


def run_stats(df: pl.DataFrame) -> dict[str, float | int]:
    """Supervised (epoch 0), final, late-mean and best-epoch mAP50-95 of one run."""
    ssl = df.filter(pl.col("epoch") > 0)
    m = "coco/mAP_50_95"
    best = ssl.sort(m, descending=True).row(0, named=True)
    late = ssl.filter(pl.col("epoch") >= LATE_EPOCHS)[m].to_list()
    return {
        "epochs": ssl.height,
        "supervised": round(float(df.filter(pl.col("epoch") == 0)[m].item()), 4),
        "final": round(float(ssl[m][-1]), 4),
        "late_mean": round(statistics.mean(late), 4) if late else None,
        "best": round(float(best[m]), 4),
        "best_epoch": int(best["epoch"]),
        "epoch_min": round(float(ssl["epoch_time"].mean()) / 60, 2),
    }


def write(df: pl.DataFrame, out_dir: Path, name: str) -> None:
    path = out_dir / name
    if df.is_empty():
        logger.warning(f"No data for {name}; not written.")
        return
    df.write_csv(path)
    logger.info(f"Wrote {path} ({df.height} rows).")


def curves(runs: list[Run]) -> pl.DataFrame:
    """Long-format per-epoch metrics (epoch 0 = supervised starting point)."""
    rows = []
    for run in runs:
        df = load(run)
        if df is None:
            continue
        for r in df.iter_rows(named=True):
            rows.append(
                {
                    **settings(run),
                    "epoch": int(r["epoch"]),
                    **metric_row(r, CURVE_METRICS),
                    "pl_images": int(r.get("pl_images") or 0),
                    "pl_boxes": int(r.get("pl_boxes") or 0),
                }
            )
    return pl.DataFrame(rows)


def summary(runs: list[Run]) -> pl.DataFrame:
    rows = []
    for run in runs:
        df = load(run)
        if df is not None:
            rows.append({**settings(run), **run_stats(df)})
    return pl.DataFrame(rows)


def best_epoch_metrics(df: pl.DataFrame) -> dict[str, object]:
    ssl = df.filter(pl.col("epoch") > 0).sort("coco/mAP_50_95", descending=True)
    best = ssl.row(0, named=True)
    return {"epoch": int(best["epoch"]), **metric_row(best, METRICS)}


def main_comparison() -> pl.DataFrame:
    """
    Supervised YOLO26m vs. GT-only fine-tuning vs. naive self-training vs. Mean Teacher, best
    results (COCO metrics). The fine-tuning row is the step-matched control of the Mean Teacher run.
    """
    rows: list[dict[str, object]] = []
    final = load(COMPONENT_RUNS[0])
    control = load(next(run for run in COMPONENT_RUNS if run.run_id == "gt-only"))
    if final is not None:
        base = final.filter(pl.col("epoch") == 0).row(0, named=True)
        rows.append({"method": "supervised", "selection": "trained model", "epoch": 0, **metric_row(base, METRICS)})
    if control is not None:
        rows.append({"method": "gt_fine_tuning", "selection": "best epoch", **best_epoch_metrics(control)})

    if NAIVE_SELF_TRAINING.is_file():
        rounds = yaml.safe_load(NAIVE_SELF_TRAINING.read_text(encoding="utf-8"))
        best = max(rounds[1:], key=lambda r: r["coco"]["mAP_50_95"])  # round 0 = supervised teacher
        rows.append(
            {
                "method": "self_training",
                "selection": f"best round ({best['round']})",
                "epoch": None,
                **{out: round(float(best["coco"][src.split("/")[1]]), 4) for out, src in METRICS.items()},
            }
        )
    else:
        logger.warning(f"Missing {NAIVE_SELF_TRAINING}; skipping the self-training row.")

    if final is not None:
        rows.append({"method": "mean_teacher", "selection": "best epoch", **best_epoch_metrics(final)})
    return pl.DataFrame(rows)


def self_training_rounds() -> pl.DataFrame:
    rounds = yaml.safe_load(NAIVE_SELF_TRAINING.read_text(encoding="utf-8"))
    return pl.DataFrame(
        [
            {"round": r["round"], **{out: round(float(r["coco"][src.split("/")[1]]), 4) for out, src in METRICS.items()}}
            for r in rounds
        ]
    )


def architectures() -> pl.DataFrame:
    rows = []
    for arch in ("yolo26", "yolo11", "yolo12"):
        if arch == "yolo26":
            runs = {
                ("gt_only", 0): MT / "ablation-architectures-control/yolo26-gt-only",
                ("ssl", 4800): MT / "ablation-drift/ema-0.9996",
                ("ssl", 2400): MT / "ablation-settle/ema-0.9996-cap-2400",
                ("ssl", 1200): MT / "ablation-architectures/yolo26-cap1200",
            }
        else:
            runs = {("gt_only", 0): MT / f"ablation-architectures-control/{arch}-gt-only"}
            runs |= {("ssl", cap): MT / f"ablation-architectures/{arch}-cap{cap}" for cap in (4800, 2400, 1200)}
        supervised_done = False
        for (method, cap), path in runs.items():
            df = load(Run(f"{arch}-{method}-{cap}", path))
            if df is None:
                continue
            stats = run_stats(df)
            final_row = df.filter(pl.col("epoch") == df["epoch"].max()).row(0, named=True)
            if not supervised_done:
                base = df.filter(pl.col("epoch") == 0).row(0, named=True)
                rows.append({"model": f"{arch}m", "method": "supervised", "pl_cap": None, "final": stats["supervised"],
                             "late_mean": None, "best": None, "best_epoch": None,
                             **metric_row(base, {"map50": "coco/mAP_50", "map_s": "coco/mAP_s", "recall": "metrics/recall(B)"}),
                             "epoch_min": None})
                supervised_done = True
            rows.append({"model": f"{arch}m", "method": method, "pl_cap": cap if method == "ssl" else None,
                         **{k: stats[k] for k in ("final", "late_mean", "best", "best_epoch")},
                         **metric_row(final_row, {"map50": "coco/mAP_50", "map_s": "coco/mAP_s", "recall": "metrics/recall(B)"}),
                         "epoch_min": stats["epoch_min"]})
    return pl.DataFrame(rows)


def label_fractions() -> pl.DataFrame:
    rows = []
    sources = {
        4800: {
            100: (MT / "ablation-drift/ema-0.9996", MT / "ablation-settle/gt-only-control"),
            **{p: (MT / f"ablation-fractions/{p}pct-ssl", MT / f"ablation-fractions/{p}pct-gt-only") for p in (10, 25, 50)},
        },
        2400: {
            100: (MT / "ablation-settle/ema-0.9996-cap-2400", MT / "ablation-architectures-control/yolo26-gt-only"),
            **{p: (MT / f"ablation-fractions-2400/{p}pct-ssl", MT / f"ablation-fractions-2400/{p}pct-gt-only") for p in (10, 25, 50)},
        },
    }
    for cap, by_fraction in sources.items():
        for pct in sorted(by_fraction):
            ssl_path, control_path = by_fraction[pct]
            for method, path in (("ssl", ssl_path), ("gt_only", control_path)):
                df = load(Run(f"{pct}pct-{method}-{cap}", path))
                if df is None:
                    continue
                stats = run_stats(df)
                final_row = df.filter(pl.col("epoch") == df["epoch"].max()).row(0, named=True)
                rows.append({"label_pct": pct, "pl_cap": cap, "method": method,
                             "supervised": stats["supervised"],
                             **{k: stats[k] for k in ("final", "late_mean", "best", "best_epoch")},
                             **metric_row(final_row, {"map50": "coco/mAP_50", "map_s": "coco/mAP_s", "recall": "metrics/recall(B)"})})
    return pl.DataFrame(rows)


def with_deltas(runs: list[Run], reference: str = "final") -> pl.DataFrame:
    """Summary of an ablation batch with each run's change from the `reference` run."""
    df = summary(runs)
    if df.is_empty() or reference not in df["run"].to_list():
        return df
    ref = df.filter(pl.col("run") == reference).row(0, named=True)
    return df.with_columns(
        delta_final=(pl.col("final") - ref["final"]).round(4),
        delta_late_mean=(pl.col("late_mean") - ref["late_mean"]).round(4),
    )


@app.command()
def main(
    out_dir: Path = typer.Option(Path("paper/exp-results"), "--out", help="Output directory for the CSVs."),
) -> None:
    """Write all thesis result tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write(main_comparison(), out_dir, "ssod_main_comparison.csv")
    write(self_training_rounds(), out_dir, "ssod_self_training_rounds.csv")
    write(summary(DEVELOPMENT_RUNS), out_dir, "ssod_mt_settings_summary.csv")
    write(curves(DEVELOPMENT_RUNS), out_dir, "ssod_mt_settings_curves.csv")
    write(curves(LONG_RUNS), out_dir, "ssod_mt_long_runs_curves.csv")
    write(with_deltas(COMPONENT_RUNS), out_dir, "ssod_components.csv")
    write(curves(COMPONENT_RUNS), out_dir, "ssod_components_curves.csv")
    write(with_deltas(MECHANISM_RUNS), out_dir, "ssod_mechanisms.csv")
    write(curves(MECHANISM_RUNS), out_dir, "ssod_mechanisms_curves.csv")
    write(with_deltas(ADD_ONE_IN_RUNS, reference="minimal"), out_dir, "ssod_add_one_in.csv")
    write(curves(ADD_ONE_IN_RUNS), out_dir, "ssod_add_one_in_curves.csv")
    write(architectures(), out_dir, "ssod_architectures.csv")
    write(label_fractions(), out_dir, "ssod_label_fractions.csv")


if __name__ == "__main__":
    app()
