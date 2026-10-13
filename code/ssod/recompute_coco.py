"""
Recompute the stored COCO metrics from saved predictions, without re-running any model.

Used when the COCO evaluation in `metrics.py` changes (here: size bins at the network input scale
instead of original pixels). Rewrites the `coco/*` columns of every Mean Teacher `results.csv`
from its per-epoch `val/val_epoch_<N>/predictions.json`, the UCWD full-label warm-up's
`eval_best.csv`, and the `coco` blocks of the naive self-training YAMLs, then regenerates each
ablation batch's `summary.csv`. The first run keeps the originals next to each CSV as
`*.before_size_norm.csv` and later runs start from those, so the script is idempotent. The YAMLs
are tracked in git and are not backed up.

Usage:
  uv run python code/ssod/recompute_coco.py --dry-run
  uv run python code/ssod/recompute_coco.py
"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import polars as pl
import typer
import yaml

try:
    from ssod.ablation import BATCHES, write_summary
    from ssod.metrics import evaluate_coco
    from ssod.utilities import get_logger
except ImportError:
    from ablation import BATCHES, write_summary  # type: ignore[import-not-found,no-redef]
    from metrics import evaluate_coco  # type: ignore[import-not-found,no-redef]
    from utilities import get_logger  # type: ignore[import-not-found,no-redef]

MT = Path("runs/mean_teacher")
FLOW_GT = Path("data/flow_img/val/annotations.json")
UCWD_FULL_WARMUP = (
    MT / "ablation-ucwd" / "100pct-warmup" / "eval_best.csv",
    Path("runs/detect/runs/mean_teacher/ablation-ucwd/100pct-warmup/val_best/predictions.json"),
    Path("data/ucwd/val/annotations.json"),
)
SELF_TRAINING = {  # rounds YAML -> directory with round_<N>/val/predictions.json
    Path("paper/exp-results/noisy_student.yaml"): Path("runs/pseudo-noisy/yolo26m-ssl"),
    Path("paper/exp-results/noisy_student_p2.yaml"): Path("runs/pseudo-noisy/yolo26m-p2-ssl"),
}
BACKUP_SUFFIX = ".before_size_norm.csv"
# COCO metrics over all box sizes; the size-bin change must leave them as they were
UNSCALED = ("mAP_50_95", "mAP_50", "mAP_75", "AR_1", "AR_10", "AR_100", "peak_f1")

app = typer.Typer(
    name="recompute-coco",
    help="Recompute stored COCO metrics from saved predictions.",
    add_completion=False,
)
logger = get_logger()


def backup_of(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.stem + BACKUP_SUFFIX)


def original(csv_path: Path) -> pl.DataFrame:
    """The CSV as it was before the first recompute."""
    backup = backup_of(csv_path)
    return pl.read_csv(backup if backup.is_file() else csv_path)


def save(df: pl.DataFrame, csv_path: Path) -> None:
    backup = backup_of(csv_path)
    if not backup.is_file():
        shutil.copy(csv_path, backup)
    df.write_csv(csv_path)


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def check_unscaled(name: str, old: dict[str, Any], new: dict[str, float]) -> None:
    """Warn when a metric that doesn't depend on the size bins has moved."""
    moved = {k: (old[f"coco/{k}"], new[k]) for k in UNSCALED
             if old.get(f"coco/{k}") is not None and abs(float(old[f"coco/{k}"]) - new[k]) > 1e-6}
    if moved:
        logger.warning(f"{name}: size-independent metrics changed: {moved}")


def run_settings(run_dir: Path) -> tuple[Path, int]:
    """COCO GT and imgsz the run was evaluated with (from its args.yaml)."""
    args_path = run_dir / "args.yaml"
    args = yaml.safe_load(args_path.read_text(encoding="utf-8")) if args_path.is_file() else {}
    return Path(args.get("coco_gt_path") or FLOW_GT), int(args.get("imgsz") or 640)


def recompute_run(run_dir: Path, imgsz: int | None, dry_run: bool) -> bool:
    """Recompute the coco/* columns of one Mean Teacher run. False if it has none."""
    csv_path = run_dir / "results.csv"
    df = original(csv_path)
    if "coco/mAP_s" not in df.columns:
        return False
    gt_path, run_imgsz = run_settings(run_dir)
    rows = []
    for row in df.iter_rows(named=True):
        preds = run_dir / "val" / f"val_epoch_{row['epoch']}" / "predictions.json"
        if not preds.is_file():
            logger.warning(f"{run_dir} epoch {row['epoch']}: no predictions.json, keeping old values")
            rows.append(row)
            continue
        coco = evaluate_coco(gt_path, preds, imgsz=imgsz if imgsz else run_imgsz, logger=logger)
        check_unscaled(f"{run_dir} epoch {row['epoch']}", row, coco)
        rows.append({**row, **{f"coco/{k}": v for k, v in coco.items() if f"coco/{k}" in row}})
    new = pl.DataFrame(rows, schema=df.schema)
    last_old, last_new = df.row(-1, named=True), new.row(-1, named=True)
    changes = ", ".join(
        f"{k} {fmt(last_old[f'coco/{k}'])} -> {fmt(last_new[f'coco/{k}'])}" for k in ("mAP_s", "mAP_m", "mAP_l")
    )
    logger.info(f"{run_dir}: last epoch {changes}")
    if not dry_run:
        save(new, csv_path)
    return True


def recompute_ucwd_full_warmup(imgsz: int, dry_run: bool) -> None:
    csv_path, preds, gt_path = UCWD_FULL_WARMUP
    if not (csv_path.is_file() and preds.is_file()):
        logger.warning(f"Missing {csv_path} or {preds}; skipping the UCWD full-label warm-up.")
        return
    df = original(csv_path)
    row = df.row(0, named=True)
    coco = evaluate_coco(gt_path, preds, imgsz=imgsz, logger=logger)
    check_unscaled(str(csv_path), row, coco)
    logger.info(f"{csv_path}: mAP_s {row['coco/mAP_s']:.4f} -> {coco['mAP_s']:.4f}")
    if not dry_run:
        new = {**row, **{f"coco/{k}": v for k, v in coco.items()}}
        save(pl.DataFrame([new], schema=df.schema), csv_path)


def recompute_self_training(imgsz: int, dry_run: bool) -> None:
    for yaml_path, run_root in SELF_TRAINING.items():
        if not yaml_path.is_file():
            logger.warning(f"Missing {yaml_path}; skipping.")
            continue
        rounds = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        for r in rounds:
            preds = run_root / f"round_{r['round']}" / "val" / "predictions.json"
            if not preds.is_file():
                logger.warning(f"Missing {preds}; keeping round {r['round']} of {yaml_path}.")
                continue
            coco = evaluate_coco(FLOW_GT, preds, imgsz=imgsz, logger=logger)
            check_unscaled(f"{yaml_path} round {r['round']}", {f"coco/{k}": v for k, v in r["coco"].items()}, coco)
            logger.info(f"{yaml_path} round {r['round']}: mAP_s {r['coco']['mAP_s']:.4f} -> {coco['mAP_s']:.4f}")
            r["coco"] = {k: coco[k] for k in r["coco"]}
        if not dry_run:
            yaml_path.write_text(yaml.safe_dump(rounds, sort_keys=False), encoding="utf-8")


@app.command()
def main(
    imgsz: int = typer.Option(
        0, "--imgsz", help="Input scale for the size bins; 0 = each run's own imgsz (640 elsewhere)."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Log old -> new values without writing."),
) -> None:
    """Recompute every stored COCO metric with the current metrics.py."""
    run_dirs = sorted(p.parent for p in MT.rglob("results.csv") if (p.parent / "val").is_dir())
    done = sum(recompute_run(d, imgsz or None, dry_run) for d in run_dirs)
    logger.info(f"Recomputed {done} Mean Teacher runs.")

    recompute_ucwd_full_warmup(imgsz or 640, dry_run)
    recompute_self_training(imgsz or 640, dry_run)

    if dry_run:
        return
    for name, spec in BATCHES.items():
        if spec.project.is_dir():
            write_summary(spec.project, list(spec.runs), spec.references)
            logger.info(f"Rewrote {spec.project / 'summary.csv'} ({name}).")


if __name__ == "__main__":
    app()
