"""
Hyperparameter tuning module for YOLO models using genetic search.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .dataset import build_tuning_subset_yaml
from .utils import load_yaml, log_environment_info

if TYPE_CHECKING:
    from .config import PipelineConfig


def run_hyperparameter_tuning(
    config: PipelineConfig, logger: logging.Logger | None = None
) -> dict[str, Any]:
    """
    Run hyperparameter evolution using Ultralytics' genetic tuning on a training subset.
    Returns the dictionary of best discovered hyperparameters.
    """
    from ultralytics import YOLO

    log = logger or logging.getLogger("pseudo_pipeline")

    subset_yaml = build_tuning_subset_yaml(
        orig_data_yaml=config.orig_data_yaml,
        tune_dir=config.tune_run_dir,
        fraction=config.tune_fraction,
        seed=config.tune_seed,
        logger=log,
    )

    base_cfg = config.get_base_configuration()
    pretrained_weights = config.get_pretrained_weights()

    log.info("Starting hyperparameter tuning (genetic algorithm, fully automatic)...")
    log.info(
        f"Tuning config: batch={config.batch_size}, workers={config.workers}, "
        f"imgsz={config.imgsz}, model={base_cfg}, weights={pretrained_weights}"
    )
    log_environment_info(log)

    model = YOLO(base_cfg)
    model.load(pretrained_weights)

    model.tune(
        data=str(subset_yaml),
        epochs=config.tune_epochs,
        iterations=config.tune_iterations,
        batch=config.batch_size,
        workers=config.workers,
        imgsz=config.imgsz,
        patience=10,
        project=str(config.tune_run_dir),
        name="evolve",
        exist_ok=True,
        plots=False,
    )

    # Locate best_hyperparameters.yaml in tune output directory
    best_hyp_candidates = list(config.tune_run_dir.rglob("best_hyperparameters.yaml"))
    if best_hyp_candidates:
        # Sort by modification time, newest first
        best_hyp_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        best_hyp_path = best_hyp_candidates[0]
        best_hyp = load_yaml(best_hyp_path)
        log.info(
            f"Tuning complete. Best hyperparameters loaded from {best_hyp_path}: {best_hyp}"
        )
        return best_hyp

    log.warning(
        "No best_hyperparameters.yaml found after tuning; returning empty dictionary."
    )
    return {}
