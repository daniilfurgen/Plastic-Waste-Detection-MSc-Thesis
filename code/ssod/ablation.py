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


@dataclass
class Batch:
    """A set of runs sharing an output directory, plus earlier runs to compare against."""

    project: Path
    runs: dict[str, list[str]]
    references: dict[str, Path] = field(default_factory=dict)
    warmups: dict[str, Warmup] = field(default_factory=dict)
    run_warmup: dict[str, str] = field(default_factory=dict)  # run -> warm-up name
    warmup_weights: Path | None = None  # batch-wide starting weights (overrides the default)


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


def warmup_epochs(fraction: float) -> int:
    """Epochs giving every warm-up the same optimizer-step budget as the full-label one."""
    steps_per_epoch = max(1, -(-round(GT_TRAIN_IMAGES * fraction) // 16))
    return round(WARMUP_STEPS / steps_per_epoch)


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


def subset_paths(fraction: float) -> tuple[Path | None, Path]:
    """Image-list and data-YAML paths for a GT train subset (the base YAML for the full split)."""
    if fraction >= 1.0:
        return None, GT_DATA_YAML.resolve()
    pct = round(fraction * 100)
    base_root = GT_DATA_YAML.resolve().parent
    return base_root / "flow_img" / "splits" / f"train_{pct}pct.txt", base_root / f"flow_img_{pct}pct.yaml"


def ensure_subset(fraction: float, logger) -> Path:
    """
    Write a nested random subset of the GT train images (a prefix of one fixed permutation, so
    10% is contained in 25%, which is contained in 50%) and a data YAML pointing at it. The val
    split is unchanged. Existing files are kept, so subsets stay fixed across re-runs.
    """
    list_path, yaml_path = subset_paths(fraction)
    if list_path is None or (list_path.is_file() and yaml_path.is_file()):
        return yaml_path

    with open(GT_DATA_YAML, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    root = Path(base["path"])
    images = sorted(
        p for p in (root / base["train"]).iterdir() if p.suffix.lower() in VALID_IMAGE_EXTS
    )
    order = np.random.default_rng(SUBSET_SEED).permutation(len(images))
    chosen = sorted(images[i] for i in order[: round(len(images) * fraction)])

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

    yaml_path = subset_paths(spec.fraction)[1] if dry_run else ensure_subset(spec.fraction, logger)
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
    save_candidates: bool = typer.Option(
        True,
        "--save-candidates/--no-save-candidates",
        help="Keep per-epoch candidate Parquet files for offline analysis.",
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
            if warmup_spec.fraction < 1.0:
                run_data_args = ["--gt-yaml", str(subset_paths(warmup_spec.fraction)[1])]

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
                *(["--save-candidates"] if save_candidates else []),
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
