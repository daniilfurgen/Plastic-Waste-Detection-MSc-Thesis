"""
Short overnight ablations of the Mean Teacher settings, organised in named batches.

Every run in a batch starts from the same warm-up weights and uses the same epoch count, so the
LR schedule, mosaic closing and epoch-0 baseline are identical across runs. Runs execute
sequentially; finished runs are skipped and interrupted ones are resumed when the script is
restarted. A summary table is written to <project>/summary.csv after every run; a batch can list
reference runs from earlier batches, which are summarised alongside but never re-run.

The `fractions` batch trains on nested random subsets of the FloW-img train split. Each fraction
gets its own supervised warm-up (run first, then shared by that fraction's runs).

Batches:
  pool      PL pool size, soft weights, strong aug, flat lambda_u (runs/mean_teacher/ablation)
  drift     causes of the late drift, on top of cap-soft (runs/mean_teacher/ablation-drift)
  settle    supervised fine-tuning control; slower Teacher, Mosaic closing, smaller pool
            on top of ema-0.9996 (runs/mean_teacher/ablation-settle)
  fractions 10/25/50% of GT labels: SSL vs. GT-only control (runs/mean_teacher/ablation-fractions)
  architectures          YOLO26/YOLO11/YOLO12 at PL caps 4800/2400/1200 (runs/mean_teacher/ablation-architectures)
  architectures-control  GT-only control per architecture (runs/mean_teacher/ablation-architectures-control)
  fractions-2400         fractions on the final 2,400 cap, reusing the fractions warm-ups (runs/mean_teacher/ablation-fractions-2400)
  components             leave-one-out of the final configuration from the tuned YOLO26m (runs/mean_teacher/ablation-components)
  mechanisms, add-one-in, seeds  see ABLATION.md
  loo-seeds              leave-one-out of every component x seeds 42-44, pipeline warm-up (runs/mean_teacher/ablation-loo-seeds)
  tuned-seeds            final and GT-only control x seeds 42-44 from the tuned YOLO26m (runs/mean_teacher/ablation-tuned-seeds)
  fractions-loo          leave-one-out + cap 4800 at 10% of the labels x seeds 42-44 (runs/mean_teacher/ablation-fractions-loo)
  ucwd                   UCWD at imgsz 640: supervised warm-up on all labels; SSL vs. control on 25% (runs/mean_teacher/ablation-ucwd)
  online-caps            online SSGT, full pipeline at PL caps 2400/4800/9600 x seeds 42-44 (runs/mean_teacher/ablation-online-caps)
  online-fractions       online SSGT at the 4800 cap: 10/25/50% labels vs. GT-only controls, plus the full-label control x seeds 42-44 (runs/mean_teacher/ablation-online-fractions)

All batches now run the online pipeline (per-step pseudo-labels with EMA score histograms). Their
results so far were produced by the offline per-epoch pipeline at git tag `offline-ssgt`; batches up
to fractions-2400 then binned pseudo-labels in original pixels, which is no longer available.

Usage:
  uv run python code/ssod/ablation.py                  # batch "settle"
  uv run python code/ssod/ablation.py --batch fractions
  uv run python code/ssod/ablation.py --batch pool --dry-run
  uv run python code/ssod/ablation.py --only gt-only-control --epochs 10
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys
import time
from typing import Optional

import numpy as np
import typer
import yaml

try:
    from ssod.utilities import get_logger
except ImportError:
    from utilities import get_logger  # type: ignore[import-not-found,no-redef]

MEAN_TEACHER_SCRIPT = Path(__file__).resolve().parent / "mean_teacher.py"
YOLO_CLI = Path(sys.executable).with_name("yolo")

GT_DATA_YAML = Path("data/flow_img.yaml")
DEFAULT_WARMUP_WEIGHTS = Path("runs/mean_teacher/ssod-v3-50-50/weights/warmup_best.pt")
# Strongest supervised YOLO26m: hyperparameter-tuned, MuSGD, 50 epochs (SAHI experiments).
TUNED_WARMUP_WEIGHTS = Path("code/sahi_experiments/yolo26m-FloW-plain.pt")
GT_TRAIN_IMAGES = 1200  # FloW-img train split
WARMUP_MODEL = "yolo26m.pt"
WARMUP_STEPS = 3750  # the full-label warm-up: 50 epochs x 75 steps (batch 16)
SUBSET_SEED = 0
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Warmup:
    """
    Supervised warm-up on the GT train split (or a nested random subset of it). Stored under
    `root` when given (shared across batches), otherwise in the batch's project directory.
    """

    fraction: float
    epochs: int
    model: str = WARMUP_MODEL
    root: Path | None = None
    data: Path = GT_DATA_YAML  # GT dataset YAML the subset is drawn from


@dataclass
class Batch:
    """A set of runs sharing an output directory, plus earlier runs to compare against."""

    project: Path
    runs: dict[str, list[str]]
    references: dict[str, Path] = field(default_factory=dict)
    warmups: dict[str, Warmup] = field(default_factory=dict)
    run_warmup: dict[str, str] = field(default_factory=dict)  # run -> warm-up name
    warmup_weights: Path | None = None  # batch-wide starting weights (overrides the default)
    # Runs on a GT subset use the rest of that dataset's train split as their unlabeled pool
    # (instead of the default FloW video frames)
    unlabeled_rest: bool = False
    baseline_warmups: list[str] = field(default_factory=list)  # warm-ups trained even without runs


def matched_gt_repeat(
    n_gt: int, ssl_gt_repeat: int = 1, pl_images: int = 4500, tail_ratio: float = 0.25
) -> int:
    """
    GT head repetition for a GT-only control so its epochs have as many samples (optimizer steps
    and Teacher EMA updates) as the SSL run it controls. `pl_images` is the mean number of
    pseudo-labelled images per epoch observed with the SSL run's cap (default: 4,800-frame cap).
    """
    tail = round(n_gt * tail_ratio)
    head = n_gt - tail
    ssl_epoch = head * ssl_gt_repeat + tail + pl_images
    return max(1, round((ssl_epoch - tail) / head))


def warmup_epochs(
    fraction: float, train_images: int = GT_TRAIN_IMAGES, step_budget: int = WARMUP_STEPS
) -> int:
    """Epochs giving every warm-up the same optimizer-step budget as the full-label one."""
    steps_per_epoch = max(1, -(-round(train_images * fraction) // 16))
    return round(step_budget / steps_per_epoch)


# Historic batches pin every setting whose trainer default has since changed (EMA 0.999 -> 0.9996,
# full pool -> 2,400 frames, hard labels -> soft weights), so they stay reproducible.
OLD_EMA = ["--ema-decay", "0.999"]
FULL_POOL = ["--pl-images-per-epoch", "-1", "--no-pl-soft-weights", *OLD_EMA]
CAP_SOFT = ["--pl-images-per-epoch", "4800", "--pl-soft-weights"]
# Best configuration so far (drift batch); update after `settle` if one of its runs is better.
BEST_SSL = [*CAP_SOFT, "--ema-decay", "0.9996"]
BEST_SSL_100 = Path("runs/mean_teacher/ablation-drift/ema-0.9996")


# Full-label warm-ups for other architectures: same data and Ultralytics arguments as the YOLO26
# warm-up (50 epochs), shared by the architecture SSL and control batches.
WARMUP_ROOT = Path("runs/mean_teacher/warmups")
ARCH_WARMUPS = {
    arch: Warmup(fraction=1.0, epochs=50, model=f"{arch}m.pt", root=WARMUP_ROOT)
    for arch in ("yolo11", "yolo12")
}
ARCH_CAPS = (4800, 2400, 1200)
CAP_2400_PL_IMAGES = 2200  # mean pseudo-labelled images per epoch observed with the 2,400 cap


def architecture_batches() -> tuple[Batch, Batch]:
    """
    SSL with the adopted configuration (soft weights, EMA 0.9996) at three PL caps for YOLO26,
    YOLO11 and YOLO12, plus one GT-only control per architecture, step-matched to the 2,400 cap.
    YOLO26 at 4,800 and 2,400 already exist with identical settings and are shown as references.
    """
    ssl_runs: dict[str, list[str]] = {}
    control_runs: dict[str, list[str]] = {}
    run_warmup: dict[str, str] = {}
    control_warmup: dict[str, str] = {}
    for arch in ("yolo26", "yolo11", "yolo12"):
        model = ["--model", f"{arch}m.pt"]
        for cap in ARCH_CAPS:
            if arch == "yolo26" and cap != 1200:
                continue  # references below
            name = f"{arch}-cap{cap}"
            ssl_runs[name] = [
                *model, "--pl-images-per-epoch", str(cap), "--pl-soft-weights", "--ema-decay", "0.9996",
            ]
            if arch in ARCH_WARMUPS:
                run_warmup[name] = f"{arch}m-warmup"
        name = f"{arch}-gt-only"
        control_runs[name] = [
            *model,
            "--pl-images-per-epoch", "0",
            "--ema-decay", "0.9996",
            "--gt-repeat", str(matched_gt_repeat(GT_TRAIN_IMAGES, pl_images=CAP_2400_PL_IMAGES)),
        ]
        if arch in ARCH_WARMUPS:
            control_warmup[name] = f"{arch}m-warmup"

    warmups = {f"{arch}m-warmup": spec for arch, spec in ARCH_WARMUPS.items()}
    ssl = Batch(
        project=Path("runs/mean_teacher/ablation-architectures"),
        runs=ssl_runs,
        references={
            "yolo26-cap4800 (ref)": Path("runs/mean_teacher/ablation-drift/ema-0.9996"),
            "yolo26-cap2400 (ref)": Path("runs/mean_teacher/ablation-settle/ema-0.9996-cap-2400"),
        },
        warmups=warmups,
        run_warmup=run_warmup,
    )
    control = Batch(
        project=Path("runs/mean_teacher/ablation-architectures-control"),
        runs=control_runs,
        references={
            "yolo26-gt-only, 4800-matched (ref)": Path("runs/mean_teacher/ablation-settle/gt-only-control"),
        },
        warmups=warmups,
        run_warmup=control_warmup,
    )
    return ssl, control


FRACTIONS_PROJECT = Path("runs/mean_teacher/ablation-fractions")


def fraction_batch(
    ssl_args: list[str] = BEST_SSL,
    pl_images: int = 4500,
    project: Path = FRACTIONS_PROJECT,
    warmup_root: Path | None = None,
    references: dict[str, Path] | None = None,
    percents: tuple[int, ...] = (10, 25, 50),
) -> Batch:
    """
    SSL vs. GT-only control at several label fractions. The SSL run repeats GT 1/fraction times
    so each epoch keeps the same GT:PL sample ratio as the full-label run; the control repeats GT
    to match the SSL run's samples per epoch (`pl_images`: mean PL images per epoch at its cap).
    Warm-ups live in `warmup_root` when given (to reuse another batch's), else in `project`.
    """
    runs: dict[str, list[str]] = {}
    warmups: dict[str, Warmup] = {}
    run_warmup: dict[str, str] = {}
    for pct in percents:
        fraction = pct / 100
        n_gt = round(GT_TRAIN_IMAGES * fraction)
        ssl_repeat = round(1 / fraction)
        warmup = f"{pct}pct-warmup"
        warmups[warmup] = Warmup(fraction=fraction, epochs=warmup_epochs(fraction), root=warmup_root)
        runs[f"{pct}pct-ssl"] = [*ssl_args, "--gt-repeat", str(ssl_repeat)]
        runs[f"{pct}pct-gt-only"] = [
            "--pl-images-per-epoch", "0",
            "--ema-decay", "0.9996",
            "--gt-repeat", str(matched_gt_repeat(n_gt, ssl_repeat, pl_images=pl_images)),
        ]
        run_warmup[f"{pct}pct-ssl"] = warmup
        run_warmup[f"{pct}pct-gt-only"] = warmup
    return Batch(
        project=project,
        runs=runs,
        references=references
        if references is not None
        else {
            "100pct-ssl (ref)": BEST_SSL_100,
            "100pct-gt-only (ref)": Path("runs/mean_teacher/ablation-settle/gt-only-control"),
        },
        warmups=warmups,
        run_warmup=run_warmup,
    )


BATCHES: dict[str, Batch] = {
    # Capped runs first: they are ~4x faster, so their results are ready even if the night runs short.
    "pool": Batch(
        project=Path("runs/mean_teacher/ablation"),
        runs={
            "cap": ["--pl-images-per-epoch", "4800", "--no-pl-soft-weights", *OLD_EMA],
            "cap-soft": [*CAP_SOFT, *OLD_EMA],
            "cap-soft-strong": [*CAP_SOFT, "--pl-strong-aug", *OLD_EMA],
            "lambda-0.2": [*FULL_POOL, "--unsupervised-weight", "0.2"],
            "control": FULL_POOL,  # full pool, lambda_u = 1, hard labels
        },
    ),
    # Each run changes one suspected cause of the post-peak drift of cap-soft.
    "drift": Batch(
        project=Path("runs/mean_teacher/ablation-drift"),
        runs={
            "lr-5e-5": [*CAP_SOFT, "--lr0", "5e-5", *OLD_EMA],
            "ema-0.9996": [*CAP_SOFT, "--ema-decay", "0.9996"],
            "cls-only": [*CAP_SOFT, "--pl-cls-only", *OLD_EMA],
        },
        references={"cap-soft (ref)": Path("runs/mean_teacher/ablation/cap-soft")},
    ),
    # Settle the remaining questions on top of the best drift setting (slow Teacher).
    "settle": Batch(
        project=Path("runs/mean_teacher/ablation-settle"),
        runs={
            # Same loop, schedule and Teacher EMA as ema-0.9996 but no pseudo-labels; GT is repeated
            # so each epoch has as many optimizer steps. Separates the SSL gain from extra fine-tuning.
            "gt-only-control": [
                "--pl-images-per-epoch", "0",
                "--ema-decay", "0.9996",
                "--gt-repeat", str(matched_gt_repeat(GT_TRAIN_IMAGES)),
            ],
            "ema-0.9998": [*CAP_SOFT, "--ema-decay", "0.9998"],
            "ema-0.9996-mosaic-on": [*CAP_SOFT, "--ema-decay", "0.9996", "--close-mosaic-ratio", "0"],
            "ema-0.9996-cap-2400": [
                "--pl-images-per-epoch", "2400", "--pl-soft-weights", "--ema-decay", "0.9996",
            ],
        },
        references={
            "ema-0.9996 (ref)": Path("runs/mean_teacher/ablation-drift/ema-0.9996"),
            "cap-soft (ref)": Path("runs/mean_teacher/ablation/cap-soft"),
        },
    ),
    "fractions": fraction_batch(),
}
BATCHES["architectures"], BATCHES["architectures-control"] = architecture_batches()
# Leave-one-out contribution of each component of the final configuration, started from the
# tuned supervised YOLO26m so the comparison is against the strongest supervised baseline.
FINAL_SSL = ["--pl-images-per-epoch", "2400", "--pl-soft-weights", "--ema-decay", "0.9996"]
BATCHES["components"] = Batch(
    project=Path("runs/mean_teacher/ablation-components"),
    runs={
        "final": FINAL_SSL,
        "gt-only": [
            "--pl-images-per-epoch", "0",
            "--ema-decay", "0.9996",
            "--gt-repeat", str(matched_gt_repeat(GT_TRAIN_IMAGES, pl_images=CAP_2400_PL_IMAGES)),
        ],
        "no-soft-weights": ["--pl-images-per-epoch", "2400", "--no-pl-soft-weights", "--ema-decay", "0.9996"],
        "fast-ema": ["--pl-images-per-epoch", "2400", "--pl-soft-weights", "--ema-decay", "0.999"],
        "no-cap": ["--pl-images-per-epoch", "-1", "--pl-soft-weights", "--ema-decay", "0.9996"],  # slowest, last
    },
    warmup_weights=TUNED_WARMUP_WEIGHTS,
)

# Leave-one-out contribution of the pseudo-labelling and sampling mechanisms, from the shared
# untuned warm-up where the final configuration clearly beats its GT-only control.
BATCHES["mechanisms"] = Batch(
    project=Path("runs/mean_teacher/ablation-mechanisms"),
    runs={
        "no-size-bins": [*FINAL_SSL, "--no-pl-size-bins"],
        "single-view": [*FINAL_SSL, "--no-pl-multi-view"],
        "no-gt-tail": [*FINAL_SSL, "--tail-ratio", "0"],
        "no-soft-weights": ["--pl-images-per-epoch", "2400", "--no-pl-soft-weights", "--ema-decay", "0.9996"],
        # Identical to the final reference: run-to-run noise of a single seed
        "final-rerun": FINAL_SSL,
    },
    references={
        "final (ref)": Path("runs/mean_teacher/ablation-settle/ema-0.9996-cap-2400"),
        "gt-only (ref)": Path("runs/mean_teacher/ablation-architectures-control/yolo26-gt-only"),
    },
)

# Add-one-in: a stripped Mean Teacher sharing only the 2,400 cap with the final configuration
# (one shared threshold, normal view only, no GT tail, hard labels, the original EMA 0.999), plus
# each component on its own. Separates what a component does alone from what the others cover.
MINIMAL_SSL = {
    "--pl-images-per-epoch": "2400",
    "--pl-size-bins": "--no-pl-size-bins",
    "--pl-multi-view": "--no-pl-multi-view",
    "--tail-ratio": "0",
    "--pl-soft-weights": "--no-pl-soft-weights",
    "--ema-decay": "0.999",
}


def minimal_ssl(**added: str) -> list[str]:
    """CLI flags of the stripped configuration with the given components switched back on."""
    settings = {**MINIMAL_SSL, **{f"--{key.replace('_', '-')}": value for key, value in added.items()}}
    args: list[str] = []
    for option, value in settings.items():
        args += [value] if value.startswith("--") else [option, value]
    return args


BATCHES["add-one-in"] = Batch(
    project=Path("runs/mean_teacher/ablation-add-one-in"),
    runs={
        "minimal": minimal_ssl(),
        "plus-slow-ema": minimal_ssl(ema_decay="0.9996"),
        "plus-soft-weights": minimal_ssl(pl_soft_weights="--pl-soft-weights"),
        "plus-size-bins": minimal_ssl(pl_size_bins="--pl-size-bins"),
        "plus-multi-view": minimal_ssl(pl_multi_view="--pl-multi-view"),
        "plus-gt-tail": minimal_ssl(tail_ratio="0.25"),
        # Identical to minimal: run-to-run noise of a single seed
        "minimal-rerun": minimal_ssl(),
    },
    references={
        "final (ref)": Path("runs/mean_teacher/ablation-settle/ema-0.9996-cap-2400"),
        "final-rerun (ref)": Path("runs/mean_teacher/ablation-mechanisms/final-rerun"),
        "gt-only (ref)": Path("runs/mean_teacher/ablation-architectures-control/yolo26-gt-only"),
    },
)

# Seed repeats of the configurations the add-one-in conclusions rest on. `simple` keeps only the
# cap and the slow Teacher: if it matches `final` across seeds, soft weights, size bins,
# multi-view and the GT tail add complexity without a measurable gain. Seed 42 of each exists.
SEED_CONFIGS = {
    "final": FINAL_SSL,
    "simple": minimal_ssl(ema_decay="0.9996"),
    "minimal": minimal_ssl(),
    "gt-only": [
        "--pl-images-per-epoch", "0",
        "--ema-decay", "0.9996",
        "--gt-repeat", str(matched_gt_repeat(GT_TRAIN_IMAGES, pl_images=CAP_2400_PL_IMAGES)),
    ],
}
BATCHES["seeds"] = Batch(
    project=Path("runs/mean_teacher/ablation-seeds"),
    runs={
        f"{name}-seed{seed}": [*args, "--seed", str(seed)]
        for seed in (43, 44)
        for name, args in SEED_CONFIGS.items()
    },
    references={
        "final-seed42 (ref)": Path("runs/mean_teacher/ablation-settle/ema-0.9996-cap-2400"),
        "final-seed42-rerun (ref)": Path("runs/mean_teacher/ablation-mechanisms/final-rerun"),
        "simple-seed42 (ref)": Path("runs/mean_teacher/ablation-add-one-in/plus-slow-ema"),
        "minimal-seed42 (ref)": Path("runs/mean_teacher/ablation-add-one-in/minimal"),
        "minimal-seed42-rerun (ref)": Path("runs/mean_teacher/ablation-add-one-in/minimal-rerun"),
        "gt-only-seed42 (ref)": Path("runs/mean_teacher/ablation-architectures-control/yolo26-gt-only"),
    },
)

# Label fractions on the final configuration (2,400-frame cap), reusing the `fractions` warm-ups.
BATCHES["fractions-2400"] = fraction_batch(
    ssl_args=["--pl-images-per-epoch", "2400", "--pl-soft-weights", "--ema-decay", "0.9996"],
    pl_images=CAP_2400_PL_IMAGES,
    project=Path("runs/mean_teacher/ablation-fractions-2400"),
    warmup_root=FRACTIONS_PROJECT,
    references={
        "100pct-ssl (ref)": Path("runs/mean_teacher/ablation-settle/ema-0.9996-cap-2400"),
        "100pct-gt-only (ref)": Path("runs/mean_teacher/ablation-architectures-control/yolo26-gt-only"),
    },
)


SEEDS = (42, 43, 44)
GT_ONLY_2400 = [
    "--pl-images-per-epoch", "0",
    "--ema-decay", "0.9996",
    "--gt-repeat", str(matched_gt_repeat(GT_TRAIN_IMAGES, pl_images=CAP_2400_PL_IMAGES)),
]


def seeded(configs: dict[str, list[str]], seeds: tuple[int, ...] = SEEDS) -> dict[str, list[str]]:
    """Every configuration once per seed, seed-major so a complete seed finishes first."""
    return {
        f"{name}-seed{seed}": [*args, "--seed", str(seed)]
        for seed in seeds
        for name, args in configs.items()
    }


# Leave-one-out of every component of the final configuration over three seeds, from the
# pipeline's own warm-up. The GT-only control (same start, steps and Teacher EMA, no
# pseudo-labels) separates the SSL gain from extra fine-tuning.
LOO_CONFIGS = {
    "final": FINAL_SSL,
    "no-multi-view": [*FINAL_SSL, "--no-pl-multi-view"],
    "no-size-bins": [*FINAL_SSL, "--no-pl-size-bins"],
    "no-soft-weights": ["--pl-images-per-epoch", "2400", "--no-pl-soft-weights", "--ema-decay", "0.9996"],
    "no-gt-tail": [*FINAL_SSL, "--tail-ratio", "0"],
    "fast-ema": ["--pl-images-per-epoch", "2400", "--pl-soft-weights", "--ema-decay", "0.999"],
    "gt-only": GT_ONLY_2400,
}
BATCHES["loo-seeds"] = Batch(
    project=Path("runs/mean_teacher/ablation-loo-seeds"),
    runs=seeded(LOO_CONFIGS),
)

# The final configuration and its control from the tuned supervised YOLO26m: does SSL add
# anything on top of the strongest supervised model?
BATCHES["tuned-seeds"] = Batch(
    project=Path("runs/mean_teacher/ablation-tuned-seeds"),
    runs=seeded({"final": FINAL_SSL, "gt-only": GT_ONLY_2400}),
    warmup_weights=TUNED_WARMUP_WEIGHTS,
)

# Leave-one-out at 10% of the labels, where SSL helps most, from the existing 10% warm-up. GT is
# repeated 10x so GT and pseudo-labels keep their full-label share of an epoch; the control is
# step-matched. `cap-4800` doubles the pseudo-labelled frames, which looked slightly better at
# 10-25% before.
FRACTION_LOO_PCT = 10
FRACTION_LOO_GT = round(GT_TRAIN_IMAGES * FRACTION_LOO_PCT / 100)
FRACTION_LOO_GT_REPEAT = round(100 / FRACTION_LOO_PCT)
FRACTION_LOO_CONFIGS = {
    **{
        name: [*args, "--gt-repeat", str(FRACTION_LOO_GT_REPEAT)]
        for name, args in LOO_CONFIGS.items()
        if name != "gt-only"
    },
    "cap-4800": [
        "--pl-images-per-epoch", "4800", "--pl-soft-weights", "--ema-decay", "0.9996",
        "--gt-repeat", str(FRACTION_LOO_GT_REPEAT),
    ],
    "gt-only": [
        "--pl-images-per-epoch", "0",
        "--ema-decay", "0.9996",
        "--gt-repeat", str(matched_gt_repeat(
            FRACTION_LOO_GT, FRACTION_LOO_GT_REPEAT, pl_images=CAP_2400_PL_IMAGES
        )),
    ],
}
_fraction_loo_runs = seeded(FRACTION_LOO_CONFIGS)
BATCHES["fractions-loo"] = Batch(
    project=Path("runs/mean_teacher/ablation-fractions-loo"),
    runs=_fraction_loo_runs,
    warmups={
        "10pct-warmup": Warmup(
            fraction=FRACTION_LOO_PCT / 100,
            epochs=warmup_epochs(FRACTION_LOO_PCT / 100),
            root=FRACTIONS_PROJECT,
        )
    },
    run_warmup={run: "10pct-warmup" for run in _fraction_loo_runs},
)

# UCWD (coastal UAV litter, 7 classes) baseline at imgsz 640: the supervised warm-up on all labels,
# and Mean Teacher vs. its GT-only control on 25% of the labels, with the other 75% of the train
# images as the unlabeled pool (all of it pseudo-labelled each epoch). GT is repeated twice so
# pseudo-labelled images make up about the same share of an epoch as in the FloW final runs.
UCWD_DATA_YAML = Path("data/ucwd.yaml")
UCWD_TRAIN_IMAGES = 3071
UCWD_WARMUP_STEPS = 50 * -(-UCWD_TRAIN_IMAGES // 16)  # 50 epochs on all labels (batch 16)
UCWD_PROJECT = Path("runs/mean_teacher/ablation-ucwd")
UCWD_FRACTION = 0.25
UCWD_GT = round(UCWD_TRAIN_IMAGES * UCWD_FRACTION)
UCWD_POOL = UCWD_TRAIN_IMAGES - UCWD_GT
UCWD_SSL_GT_REPEAT = 2
BATCHES["ucwd"] = Batch(
    project=UCWD_PROJECT,
    runs={
        "25pct-ssl": [
            "--pl-images-per-epoch", "-1", "--pl-soft-weights", "--ema-decay", "0.9996",
            "--gt-repeat", str(UCWD_SSL_GT_REPEAT),
        ],
        "25pct-gt-only": [
            "--pl-images-per-epoch", "0",
            "--ema-decay", "0.9996",
            "--gt-repeat", str(matched_gt_repeat(UCWD_GT, UCWD_SSL_GT_REPEAT, pl_images=UCWD_POOL)),
        ],
    },
    warmups={
        "100pct-warmup": Warmup(fraction=1.0, epochs=50, data=UCWD_DATA_YAML),
        "25pct-warmup": Warmup(
            fraction=UCWD_FRACTION,
            epochs=warmup_epochs(UCWD_FRACTION, UCWD_TRAIN_IMAGES, UCWD_WARMUP_STEPS),
            data=UCWD_DATA_YAML,
        ),
    },
    run_warmup={"25pct-ssl": "25pct-warmup", "25pct-gt-only": "25pct-warmup"},
    unlabeled_rest=True,
    baseline_warmups=["100pct-warmup"],
    references={"100pct-supervised (warm-up)": UCWD_PROJECT / "100pct-warmup"},
)

# Every batch above ran with the former 25% GT tail default; pin it so they stay reproducible.
# The trainer now mixes all GT with the pseudo-labels (`--tail-ratio 0`), since the tail showed no
# measurable effect in `mechanisms`, `add-one-in`, `loo-seeds` or `fractions-loo`.
LEGACY_GT_TAIL = ["--tail-ratio", "0.25"]
for _batch in BATCHES.values():
    for _run, _args in _batch.runs.items():
        if "--tail-ratio" not in _args:
            _batch.runs[_run] = [*_args, *LEGACY_GT_TAIL]

# ---- Batches with online SSGT and the current defaults (no GT tail) -------------------------

# The full pipeline with online pseudo-labelling (per-step labels, EMA score histograms) at three
# PL caps over three seeds, from the pipeline's own warm-up. 2,400 was the optimum offline; with
# labels refreshed every step, larger caps may pay off differently.
ONLINE_CAPS = (2400, 4800, 9600)
BATCHES["online-caps"] = Batch(
    project=Path("runs/mean_teacher/ablation-online-caps"),
    runs=seeded({
        f"cap-{cap}": ["--pl-images-per-epoch", str(cap), "--pl-soft-weights", "--ema-decay", "0.9996"]
        for cap in ONLINE_CAPS
    }),
)

# The 4,800 cap with step-matched GT-only controls, at full labels and at 10/25/50% of the labels
# (reusing the `fractions` warm-ups), over seeds 42-44. In `online-caps` the best epoch followed the
# number of pseudo-labelled images seen (~36-48k), which 15 epochs at 4,800 exceed (72k), so runs
# that keep improving beyond that show up here. As in `fractions`, SSL runs repeat GT 1/fraction
# times to keep the full-label GT:PL ratio; every unlabeled image of the cap enters an epoch, so
# controls repeat GT to match 1 x GT + 4,800 samples per epoch (no GT tail).
ONLINE_FRACTION_CAP = 4800
ONLINE_FRACTION_SSL = [
    "--pl-images-per-epoch", str(ONLINE_FRACTION_CAP), "--pl-soft-weights", "--ema-decay", "0.9996",
]


def online_gt_only(n_gt: int, ssl_gt_repeat: int = 1) -> list[str]:
    """GT-only control step-matched to an online SSL run at the 4,800 cap."""
    gt_repeat = matched_gt_repeat(n_gt, ssl_gt_repeat, pl_images=ONLINE_FRACTION_CAP, tail_ratio=0.0)
    return ["--pl-images-per-epoch", "0", "--ema-decay", "0.9996", "--gt-repeat", str(gt_repeat)]


_online_fraction_configs: dict[str, list[str]] = {"100pct-gt-only": online_gt_only(GT_TRAIN_IMAGES)}
_online_fraction_warmup: dict[str, str] = {}
for _pct in (10, 25, 50):
    _ssl_repeat = round(100 / _pct)
    _online_fraction_configs[f"{_pct}pct-ssl"] = [*ONLINE_FRACTION_SSL, "--gt-repeat", str(_ssl_repeat)]
    _online_fraction_configs[f"{_pct}pct-gt-only"] = online_gt_only(
        round(GT_TRAIN_IMAGES * _pct / 100), _ssl_repeat
    )
_online_fraction_runs = seeded(_online_fraction_configs)
BATCHES["online-fractions"] = Batch(
    project=Path("runs/mean_teacher/ablation-online-fractions"),
    runs=_online_fraction_runs,
    references={
        f"100pct-ssl-seed{seed} (ref)": Path(f"runs/mean_teacher/ablation-online-caps/cap-4800-seed{seed}")
        for seed in SEEDS
    },
    warmups={
        f"{pct}pct-warmup": Warmup(fraction=pct / 100, epochs=warmup_epochs(pct / 100), root=FRACTIONS_PROJECT)
        for pct in (10, 25, 50)
    },
    run_warmup={
        run: f"{run.split('pct-')[0]}pct-warmup"
        for run in _online_fraction_runs
        if not run.startswith("100pct")
    },
)

SUMMARY_FIELDS = [
    "run",
    "epochs_done",
    "baseline_mAP50-95",
    "last_mAP50-95",
    "delta_last",
    "best_mAP50-95",
    "best_epoch",
    "last_recall",
    "last_mAP_s",
    "last_pl_boxes",
    "mean_epoch_min",
]

app = typer.Typer(
    name="ssod-ablation",
    help="Run a short Mean Teacher ablation from shared warm-up weights.",
    add_completion=False,
)


def read_results(run_dir: Path) -> list[dict[str, str]]:
    results_csv = run_dir / "results.csv"
    if not results_csv.is_file():
        return []
    with open(results_csv, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def completed_epochs(run_dir: Path) -> int:
    return max((int(row["epoch"]) for row in read_results(run_dir)), default=0)


def summarize(run: str, run_dir: Path) -> dict[str, str] | None:
    rows = read_results(run_dir)
    if not rows:
        return None
    by_epoch = {int(row["epoch"]): row for row in rows}
    ssl = [by_epoch[e] for e in sorted(by_epoch) if e > 0]
    if not ssl:
        return None

    def metric(row: dict[str, str], key: str) -> float:
        return float(row.get(key) or 0.0)

    baseline = metric(by_epoch[0], "metrics/mAP50-95(B)") if 0 in by_epoch else float("nan")
    last = ssl[-1]
    best = max(ssl, key=lambda row: metric(row, "metrics/mAP50-95(B)"))
    return {
        "run": run,
        "epochs_done": str(len(ssl)),
        "baseline_mAP50-95": f"{baseline:.4f}",
        "last_mAP50-95": f"{metric(last, 'metrics/mAP50-95(B)'):.4f}",
        "delta_last": f"{metric(last, 'metrics/mAP50-95(B)') - baseline:+.4f}",
        "best_mAP50-95": f"{metric(best, 'metrics/mAP50-95(B)'):.4f}",
        "best_epoch": best["epoch"],
        "last_recall": f"{metric(last, 'metrics/recall(B)'):.4f}",
        "last_mAP_s": f"{metric(last, 'coco/mAP_s'):.4f}",
        "last_pl_boxes": f"{metric(last, 'pl_boxes'):.0f}",
        "mean_epoch_min": f"{sum(metric(r, 'epoch_time') for r in ssl) / len(ssl) / 60:.1f}",
    }


def write_summary(
    project: Path,
    runs: list[str],
    references: dict[str, Path] | None = None,
    write: bool = True,
) -> list[dict[str, str]]:
    run_dirs = [(run, project / run) for run in runs]
    run_dirs += [(name, path.resolve()) for name, path in (references or {}).items()]
    table = [s for name, run_dir in run_dirs if (s := summarize(name, run_dir)) is not None]
    if not write:
        return table
    with open(project / "summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(table)
    return table


def format_table(table: list[dict[str, str]]) -> str:
    if not table:
        return "(no results yet)"
    widths = {k: max(len(k), *(len(row[k]) for row in table)) for k in SUMMARY_FIELDS}
    lines = ["  ".join(k.ljust(widths[k]) for k in SUMMARY_FIELDS)]
    lines += ["  ".join(row[k].ljust(widths[k]) for k in SUMMARY_FIELDS) for row in table]
    return "\n".join(lines)


def dataset_config(data_yaml: Path) -> dict:
    with open(data_yaml, encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_images(data_yaml: Path) -> list[Path]:
    """Sorted image files of a dataset's train split."""
    config = dataset_config(data_yaml)
    train_dir = Path(config["path"]) / config["train"]
    return sorted(p for p in train_dir.iterdir() if p.suffix.lower() in VALID_IMAGE_EXTS)


def subset_paths(fraction: float, data_yaml: Path = GT_DATA_YAML) -> tuple[Path | None, Path]:
    """
    Image-list and data-YAML paths for a GT train subset (the base YAML for the full split):
    <dataset root>/splits/train_<pct>pct.txt and <base YAML stem>_<pct>pct.yaml next to the base.
    """
    data_yaml = data_yaml.resolve()
    if fraction >= 1.0:
        return None, data_yaml
    pct = round(fraction * 100)
    root = Path(dataset_config(data_yaml)["path"])
    return root / "splits" / f"train_{pct}pct.txt", data_yaml.parent / f"{data_yaml.stem}_{pct}pct.yaml"


def subset_images(fraction: float, data_yaml: Path = GT_DATA_YAML) -> list[Path]:
    """
    Nested random subset of the train images: a prefix of one fixed permutation, so 10% is
    contained in 25%, which is contained in 50%.
    """
    images = train_images(data_yaml)
    order = np.random.default_rng(SUBSET_SEED).permutation(len(images))
    return sorted(images[i] for i in order[: round(len(images) * fraction)])


def ensure_subset(fraction: float, logger, data_yaml: Path = GT_DATA_YAML) -> Path:
    """
    Write a GT train subset (see `subset_images`) and a data YAML pointing at it. The val split
    is unchanged. Existing files are kept, so subsets stay fixed across re-runs.
    """
    list_path, yaml_path = subset_paths(fraction, data_yaml)
    if list_path is None or (list_path.is_file() and yaml_path.is_file()):
        return yaml_path

    base = dataset_config(data_yaml)
    root = Path(base["path"])
    chosen = subset_images(fraction, data_yaml)

    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("".join(f"{p.resolve()}\n" for p in chosen), encoding="utf-8")
    subset = {**base, "train": str(list_path.resolve())}
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(subset, f, sort_keys=False)

    labels = [root / base["train"].replace("images", "labels") / f"{p.stem}.txt" for p in chosen]
    n_boxes = sum(len(l.read_text().split()) // 5 for l in labels if l.is_file())
    logger.info(
        f"Wrote GT subset {fraction:.0%}: {len(chosen)} images, {n_boxes} boxes -> {yaml_path}"
    )
    return yaml_path


def rest_pool_path(fraction: float, data_yaml: Path) -> Path:
    """Images directory of the train images left out of a GT subset."""
    rest = Path(dataset_config(data_yaml.resolve())["path"]) / f"unlabeled_{100 - round(fraction * 100)}pct"
    return rest / "images"


def ensure_rest_pool(fraction: float, data_yaml: Path, logger) -> Path:
    """
    Unlabeled pool for SSL on a GT subset: symlinks to the train images outside the subset (their
    GT labels are never read).
    """
    images_dir = rest_pool_path(fraction, data_yaml)
    subset = {p.name for p in subset_images(fraction, data_yaml)}
    rest = [p for p in train_images(data_yaml) if p.name not in subset]
    if images_dir.is_dir() and len(list(images_dir.iterdir())) == len(rest):
        return images_dir

    images_dir.mkdir(parents=True, exist_ok=True)
    for image in rest:
        link = images_dir / image.name
        if not link.exists():
            link.symlink_to(image.resolve())
    logger.info(f"Unlabeled pool: {len(rest)} train images outside the {fraction:.0%} subset -> {images_dir}")
    return images_dir


def run_subprocess(cmd: list[str], log_path: Path, name: str, logger) -> int:
    """Run a child process with output appended to log_path; on Ctrl+C let it pause, then exit."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
        try:
            return proc.wait()
        except KeyboardInterrupt:
            # The child receives the same Ctrl+C and saves its pause checkpoint; wait for it.
            logger.warning(f"[{name}] interrupted; waiting for the child process to save its state...")
            proc.wait()
            logger.warning("Ablation stopped. Re-run the same command to resume.")
            raise typer.Exit(code=130)


def ensure_warmup(name: str, spec: Warmup, project: Path, dry_run: bool, logger) -> Path:
    """
    Supervised warm-up on a GT subset with the same Ultralytics arguments as
    MeanTeacherTrainer.run_warmup. Returns the best.pt path; skips if already complete.
    """
    warmup_dir = (spec.root.resolve() if spec.root is not None else project) / name
    best = warmup_dir / "weights" / "best.pt"
    done_marker = warmup_dir / "warmup.done"
    if done_marker.is_file() and best.is_file():
        logger.info(f"[{name}] warm-up already complete, skipping.")
        return best

    yaml_path = (
        subset_paths(spec.fraction, spec.data)[1]
        if dry_run
        else ensure_subset(spec.fraction, logger, spec.data)
    )
    last = warmup_dir / "weights" / "last.pt"
    if last.is_file():
        cmd = [str(YOLO_CLI), "detect", "train", "resume", f"model={last}"]
        logger.info(f"[{name}] resuming warm-up.")
    else:
        cmd = [
            str(YOLO_CLI), "detect", "train",
            f"data={yaml_path}",
            f"model={spec.model}",
            f"epochs={spec.epochs}",
            "batch=16",
            "imgsz=640",
            "workers=4",
            f"project={warmup_dir.parent}",
            f"name={name}",
            "exist_ok=True",
        ]
    logger.info(f"[{name}] {' '.join(cmd)}")
    if dry_run:
        return best

    started = time.time()
    returncode = run_subprocess(cmd, warmup_dir / "stdout.log", name, logger)
    if returncode != 0 or not best.is_file():
        raise RuntimeError(
            f"[{name}] warm-up failed (exit code {returncode}); see {warmup_dir / 'stdout.log'}"
        )
    done_marker.touch()
    logger.info(f"[{name}] warm-up finished in {(time.time() - started) / 3600:.1f} h.")
    return best


def data_args(spec: Batch, warmup: Warmup, dry_run: bool, logger) -> list[str]:
    """Trainer flags for the GT data (and unlabeled pool) of a run trained from `warmup`."""
    if warmup.fraction >= 1.0 and warmup.data == GT_DATA_YAML:
        return []
    args = ["--gt-yaml", str(subset_paths(warmup.fraction, warmup.data)[1])]
    if spec.unlabeled_rest and warmup.fraction < 1.0:
        pool = (
            rest_pool_path(warmup.fraction, warmup.data)
            if dry_run
            else ensure_rest_pool(warmup.fraction, warmup.data, logger)
        )
        args += ["--pl-images", str(pool)]
    return args


@app.command()
def main(
    warmup_weights: Optional[Path] = typer.Option(
        None,
        "--warmup-weights",
        help=(
            "Warm-up checkpoint shared by every run (runs with their own warm-up ignore it). "
            f"Defaults to the batch's own, else {DEFAULT_WARMUP_WEIGHTS}."
        ),
        exists=True,
        dir_okay=False,
    ),
    batch: str = typer.Option(
        "settle", "--batch", help=f"Ablation batch to run. Choices: {', '.join(BATCHES)}."
    ),
    epochs: int = typer.Option(15, "--epochs", help="SSL epochs per run.", min=1),
    project: Optional[Path] = typer.Option(
        None,
        "--project",
        help="Override the batch's directory (one sub-directory per run plus summary.csv).",
    ),
    only: Optional[list[str]] = typer.Option(
        None, "--only", help="Run only these runs of the batch (repeatable)."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the commands only."),
) -> None:
    """Run each setting of an ablation batch sequentially from the same warm-up weights."""
    if batch not in BATCHES:
        raise typer.BadParameter(f"Unknown batch {batch!r}. Choices: {list(BATCHES)}")
    spec = BATCHES[batch]
    runs = list(only) if only else list(spec.runs)
    unknown = [r for r in runs if r not in spec.runs]
    if unknown:
        raise typer.BadParameter(
            f"Unknown run(s) for batch {batch!r}: {unknown}. Choices: {list(spec.runs)}"
        )

    project = (project or spec.project).resolve()
    warmup_weights = (warmup_weights or spec.warmup_weights or DEFAULT_WARMUP_WEIGHTS).resolve()
    project.mkdir(parents=True, exist_ok=True)
    logger = get_logger(log_file=project / "ablation.log")
    logger.info(
        f"Ablation batch {batch!r}: runs={runs}, epochs={epochs}, warmup={warmup_weights}, project={project}"
    )

    for name in spec.baseline_warmups:
        try:
            ensure_warmup(name, spec.warmups[name], project, dry_run, logger)
        except RuntimeError as e:
            logger.error(str(e))

    for run in runs:
        run_dir = project / run
        done = completed_epochs(run_dir)
        if done >= epochs:
            logger.info(f"[{run}] already finished ({done}/{epochs} epochs), skipping.")
            continue

        run_warmup_weights = warmup_weights
        run_data_args: list[str] = []
        warmup_name = spec.run_warmup.get(run)
        if warmup_name is not None:
            warmup_spec = spec.warmups[warmup_name]
            try:
                run_warmup_weights = ensure_warmup(
                    warmup_name, warmup_spec, project, dry_run, logger
                ).resolve()
            except RuntimeError as e:
                logger.error(f"{e}. Skipping {run}.")
                continue
            run_data_args = data_args(spec, warmup_spec, dry_run, logger)

        cmd = [sys.executable, str(MEAN_TEACHER_SCRIPT)]
        if (run_dir / "weights" / "paused.pt").is_file() or (
            run_dir / "weights" / "last_checkpoint.pt"
        ).is_file():
            # Settings and epoch count are restored from the run's args.yaml
            cmd += ["--resume", str(run_dir)]
            logger.info(f"[{run}] resuming after {done}/{epochs} epochs.")
        else:
            cmd += [
                "--warmup-weights", str(run_warmup_weights),
                "--ssl-epochs", str(epochs),
                "--project", str(project),
                "--name", run,
                *run_data_args,
                *spec.runs[run],
            ]

        logger.info(f"[{run}] {' '.join(cmd)}")
        if dry_run:
            continue

        started = time.time()
        returncode = run_subprocess(cmd, run_dir / "stdout.log", run, logger)
        elapsed = (time.time() - started) / 3600
        done = completed_epochs(run_dir)
        if returncode != 0 or done < epochs:
            logger.error(
                f"[{run}] stopped after {done}/{epochs} epochs (exit code {returncode}, {elapsed:.1f} h). "
                f"See {run_dir / 'stdout.log'}. Continuing with the next run."
            )
        else:
            logger.info(f"[{run}] finished in {elapsed:.1f} h.")

        write_summary(project, runs, spec.references)

    table = write_summary(project, runs, spec.references, write=not dry_run)
    logger.info(f"Summary ({project / 'summary.csv'}):\n{format_table(table)}")


if __name__ == "__main__":
    app()
