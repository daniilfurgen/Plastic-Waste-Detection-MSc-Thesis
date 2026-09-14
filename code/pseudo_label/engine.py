"""
Semi-Supervised Learning (SSL) pipeline execution engine for YOLO.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from .config import PipelineConfig
from .dataset import (
    build_round_yaml,
    make_pseudo_labels,
    sync_pseudo_labels,
)
from .eval import evaluate_coco_predictions
from .utils import (
    RunStateTracker,
    configure_system,
    log_environment_info,
    setup_logging,
)


class PseudoLabelPipeline:
    """
    Orchestrator for multi-round semi-supervised pseudo-labeling training.
    Supports pausing, resuming from specific rounds, or continuing interrupted rounds.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.logger = logger or setup_logging(
            log_file=self.config.log_file,
            logger_name="pseudo_pipeline",
        )
        self.state_tracker = RunStateTracker(self.config.pseudo_run_dir)

        # Enforce deterministic/reliable cuDNN settings
        configure_system(cudnn_benchmark=False)

    def get_resume_start_round(self) -> int:
        """
        Determine the next round to execute based on saved run artifacts.
        If a round finished training but pseudo-labeling was pending, it completes
        the pseudo-labeling and advances to the next round.
        """
        last_completed = self.state_tracker.get_last_completed_round()
        if last_completed is None:
            self.logger.info("No previous completed rounds found; starting at Round 0.")
            return 0

        self.logger.info(
            f"Found completed checkpoint and summary up to Round {last_completed}."
        )

        # Check if pseudo-labels for round last_completed were generated
        round_predict_dir = (
            self.config.pseudo_run_dir / f"round_{last_completed}" / "predict"
        )
        labels_dir = round_predict_dir / "labels"

        # If pseudo labels for the completed round don't exist yet and more rounds remain
        if (not labels_dir.exists() or not any(labels_dir.glob("*.txt"))) and (
            last_completed < self.config.n_rounds - 1
        ):
            ckpt = self.state_tracker.get_checkpoint_for_round(last_completed)
            if ckpt is not None and ckpt.exists():
                self.logger.info(
                    f"Round {last_completed} pseudo-labels missing. Generating now using {ckpt}..."
                )
                conf = self.config.get_conf_for_round(last_completed)
                model = YOLO(str(ckpt))
                make_pseudo_labels(
                    model=model,
                    predict_dir=round_predict_dir,
                    video_images_dir=self.config.video_images_dir,
                    video_labels_dir=self.config.video_labels_dir,
                    conf=conf,
                    logger=self.logger,
                )

        next_round = last_completed + 1
        self.logger.info(f"Resuming training pipeline at Round {next_round}.")
        return next_round

    def prepare_round_data(self, round_idx: int) -> Path:
        """
        Prepare dataset YAML for the given round.
        For round 0, uses orig_data_yaml.
        For round > 0, verifies/generates pseudo-labels from round_idx - 1 and builds merged dataset.
        """
        if round_idx == 0:
            return self.config.orig_data_yaml

        prev_round = round_idx - 1
        prev_predict_dir = (
            self.config.pseudo_run_dir / f"round_{prev_round}" / "predict"
        )
        prev_labels_dir = prev_predict_dir / "labels"

        # If previous round pseudo-labels are missing, generate them on the fly
        if not prev_labels_dir.exists() or not any(prev_labels_dir.glob("*.txt")):
            prev_ckpt = self.state_tracker.get_checkpoint_for_round(prev_round)
            if prev_ckpt is None or not prev_ckpt.exists():
                raise FileNotFoundError(
                    f"Cannot prepare data for Round {round_idx}: Checkpoint from Round {prev_round} "
                    f"not found at {self.config.pseudo_run_dir / f'round_{prev_round}_train' / 'weights' / 'best.pt'}"
                )
            self.logger.warning(
                f"Pseudo-labels for Round {prev_round} not found. Generating using {prev_ckpt}..."
            )
            conf = self.config.get_conf_for_round(prev_round)
            model = YOLO(str(prev_ckpt))
            make_pseudo_labels(
                model=model,
                predict_dir=prev_predict_dir,
                video_images_dir=self.config.video_images_dir,
                video_labels_dir=self.config.video_labels_dir,
                conf=conf,
                logger=self.logger,
            )
        else:
            # Sync existing pseudo-labels into video_labels_dir
            sync_pseudo_labels(
                source_labels_dir=prev_labels_dir,
                target_labels_dir=self.config.video_labels_dir,
                clean_target=True,
                logger=self.logger,
            )

        round_dir = self.config.pseudo_run_dir / f"round_{round_idx}"
        return build_round_yaml(
            round_idx=round_idx,
            orig_data_yaml=self.config.orig_data_yaml,
            video_images_dir=self.config.video_images_dir,
            video_labels_dir=self.config.video_labels_dir,
            round_dir=round_dir,
            max_pseudo_ratio=self.config.max_pseudo_ratio,
            seed=self.config.seed,
            logger=self.logger,
        )

    def prepare_round_model(
        self, round_idx: int, resume_mid_round: bool = False
    ) -> tuple[YOLO, bool]:
        """
        Instantiate the YOLO model for the given round.
        Returns a tuple of (model_instance, is_resuming_existing_run).
        """
        # Check if mid-round resumption from last.pt was requested
        if resume_mid_round:
            last_ckpt = self.state_tracker.get_last_checkpoint_for_round(round_idx)
            if last_ckpt is not None and last_ckpt.exists():
                self.logger.info(
                    f"Found existing last.pt for Round {round_idx} at {last_ckpt}. Resuming training..."
                )
                return YOLO(str(last_ckpt)), True

        if round_idx == 0:
            base_cfg = self.config.get_base_configuration()
            pretrained_weights = self.config.get_pretrained_weights()
            self.logger.info(
                f"Initializing Round 0 with architecture={base_cfg}, weights={pretrained_weights}"
            )
            model = YOLO(base_cfg)
            model.load(pretrained_weights)
            return model, False

        prev_round = round_idx - 1
        prev_ckpt = self.state_tracker.get_checkpoint_for_round(prev_round)
        if prev_ckpt is None or not prev_ckpt.exists():
            raise FileNotFoundError(
                f"Cannot initialize Round {round_idx}: Checkpoint from Round {prev_round} "
                f"not found at {self.config.pseudo_run_dir / f'round_{prev_round}_train' / 'weights' / 'best.pt'}"
            )

        self.logger.info(
            f"Initializing Round {round_idx} with checkpoint from Round {prev_round}: {prev_ckpt}"
        )
        return YOLO(str(prev_ckpt)), False

    def execute_round(
        self,
        round_idx: int,
        hyp_overrides: dict[str, Any] | None = None,
        resume_mid_round: bool = False,
    ) -> dict[str, Any]:
        """
        Execute a single pseudo-labeling round:
        1. Prepare dataset.
        2. Train model.
        3. Evaluate on validation set.
        4. Record metrics to summary.yaml.
        5. Generate pseudo-labels for the next round (if applicable).
        """
        hyp_overrides = hyp_overrides or {}
        self.logger.info(f"=== Round {round_idx}: Training Start ===")
        log_environment_info(self.logger)

        data_yaml = self.prepare_round_data(round_idx)
        model, is_resuming = self.prepare_round_model(
            round_idx, resume_mid_round=resume_mid_round
        )

        train_kwargs: dict[str, Any] = {
            "data": str(data_yaml),
            "epochs": self.config.epochs_per_round,
            "batch": self.config.batch_size,
            "workers": self.config.workers,
            "patience": self.config.patience,
            "imgsz": self.config.imgsz,
            "project": str(self.config.pseudo_run_dir),
            "name": f"round_{round_idx}_train",
            "exist_ok": True,
            **hyp_overrides,
        }
        if self.config.device is not None:
            train_kwargs["device"] = self.config.device

        if is_resuming:
            self.logger.info(f"Resuming training for Round {round_idx}...")
            train_results = model.train(resume=True)
        else:
            self.logger.info(
                f"Starting training for Round {round_idx} (batch={self.config.batch_size}, "
                f"workers={self.config.workers}, imgsz={self.config.imgsz}, epochs={self.config.epochs_per_round})"
            )
            train_results = model.train(**train_kwargs)

        best_checkpoint = Path(train_results.save_dir) / "weights" / "best.pt"
        self.logger.info(
            f"Round {round_idx} training complete. Best checkpoint: {best_checkpoint}"
        )

        # Fixed validation on the human-labeled ground truth set
        self.logger.info(
            f"Evaluating Round {round_idx} on validation split: {self.config.orig_data_yaml}"
        )
        val_model = YOLO(str(best_checkpoint))
        val_dir = self.config.pseudo_run_dir / f"round_{round_idx}" / "val"
        val_kwargs: dict[str, Any] = {
            "data": str(self.config.orig_data_yaml),
            "split": "val",
            "save_json": self.config.enable_coco_eval,
            "project": str(val_dir.parent),
            "name": val_dir.name,
            "exist_ok": True,
        }
        if self.config.device is not None:
            val_kwargs["device"] = self.config.device

        metrics = val_model.val(**val_kwargs)

        map50 = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)
        self.logger.info(
            f"Round {round_idx} Validation Results (Ultralytics): mAP50={map50:.4f}, mAP50-95={map50_95:.4f}"
        )

        coco_metrics: dict[str, float] = {}
        if self.config.enable_coco_eval and self.config.coco_gt_path:
            predictions_json = val_dir / "predictions.json"
            if predictions_json.exists():
                try:
                    coco_metrics = evaluate_coco_predictions(
                        coco_gt_path=self.config.coco_gt_path,
                        predictions_path=predictions_json,
                        fix_ids=True,
                        default_category_id=0,
                        logger=self.logger,
                    )
                    self.logger.info(
                        f"Round {round_idx} COCO Metrics: "
                        f"mAP_50_95={coco_metrics['mAP_50_95']:.4f}, "
                        f"mAP_50={coco_metrics['mAP_50']:.4f}, "
                        f"mAP_s={coco_metrics['mAP_s']:.4f}, "
                        f"mAP_m={coco_metrics['mAP_m']:.4f}, "
                        f"mAP_l={coco_metrics['mAP_l']:.4f}, "
                        f"peak_f1={coco_metrics['peak_f1']:.4f}"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"COCO evaluation failed for Round {round_idx}: {e}"
                    )
            else:
                self.logger.warning(
                    f"predictions.json not found at {predictions_json}; skipping COCO evaluation."
                )

        # Record metrics in state tracker
        extra_metrics = {"coco": coco_metrics} if coco_metrics else None
        self.state_tracker.record_round(
            round_idx=round_idx,
            weights=str(best_checkpoint),
            mAP50=map50,
            mAP50_95=map50_95,
            extra=extra_metrics,
        )

        # Generate pseudo-labels for subsequent round if within budget
        if round_idx < self.config.n_rounds - 1:
            next_conf = self.config.get_conf_for_round(round_idx)
            self.logger.info(
                f"=== Round {round_idx}: Generating pseudo-labels for Round {round_idx + 1} (conf>={next_conf:.2f}) ==="
            )
            predict_dir = self.config.pseudo_run_dir / f"round_{round_idx}" / "predict"
            make_pseudo_labels(
                model=val_model,
                predict_dir=predict_dir,
                video_images_dir=self.config.video_images_dir,
                video_labels_dir=self.config.video_labels_dir,
                conf=next_conf,
                logger=self.logger,
            )

        # Free GPU memory cache between rounds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result_row: dict[str, Any] = {
            "round": round_idx,
            "weights": str(best_checkpoint),
            "mAP50": map50,
            "mAP50_95": map50_95,
        }
        if coco_metrics:
            result_row["coco"] = coco_metrics
        return result_row

    def run(
        self,
        start_round: int | None = None,
        end_round: int | None = None,
        resume: bool = False,
        resume_mid_round: bool = False,
        hyp_overrides: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run the iterative pseudo-labeling pipeline from start_round to end_round (inclusive).

        Args:
            start_round: Round to begin training at. If None and resume=True, auto-detected.
            end_round: Round to finish at (inclusive). Defaults to (n_rounds - 1).
            resume: If True, automatically resumes from the last completed round.
            resume_mid_round: If True and an interrupted round has last.pt, resumes that round.
            hyp_overrides: Optional hyperparameter overrides dictionary.
        """
        # Determine start round
        if start_round is None:
            start_round = self.get_resume_start_round() if resume else 0

        # Determine end round
        if end_round is None:
            end_round = self.config.n_rounds - 1

        if start_round > end_round:
            self.logger.info(
                f"start_round ({start_round}) is greater than end_round ({end_round}). Nothing to run."
            )
            return self.state_tracker.load_summary()

        # Combine hyperparameter overrides
        merged_hyp = dict(self.config.hyp_overrides)
        if hyp_overrides:
            merged_hyp.update(hyp_overrides)

        # Persist configuration snapshot
        self.config.pseudo_run_dir.mkdir(parents=True, exist_ok=True)
        config_snapshot_path = self.config.pseudo_run_dir / "pipeline_config.yaml"
        self.config.save_yaml(config_snapshot_path)

        self.logger.info(
            f"== Starting Pseudo-Labeling Pipeline: {self.config.experiment_name} =="
        )
        self.logger.info(
            f"Planned execution: Round {start_round} -> Round {end_round} (Total budget: {self.config.n_rounds} rounds)"
        )
        self.logger.info(
            f"Pipeline config: batch={self.config.batch_size}, workers={self.config.workers}, "
            f"imgsz={self.config.imgsz}, epochs_per_round={self.config.epochs_per_round}, "
            f"patience={self.config.patience}, model_size={self.config.model_size}, p2_head={self.config.p2_head}"
        )
        log_environment_info(self.logger)

        try:
            for round_idx in range(start_round, end_round + 1):
                # Pass resume_mid_round only to the first round being executed if relevant
                allow_mid_resume = (
                    resume_mid_round if round_idx == start_round else False
                )
                self.execute_round(
                    round_idx=round_idx,
                    hyp_overrides=merged_hyp,
                    resume_mid_round=allow_mid_resume,
                )

            summary = self.state_tracker.load_summary()
            self.logger.info(
                f"Pipeline session completed successfully. Summary: {summary}"
            )
            return summary

        except KeyboardInterrupt:
            self.logger.warning("Pipeline paused/interrupted by user.")
            self.logger.info(
                "To resume later, run:\n"
                "    uv run python code/pseudo_label_pipeline.py --resume"
            )
            return self.state_tracker.load_summary()

        except Exception as e:
            self.logger.exception(f"Pipeline failed during execution: {e}")
            raise
