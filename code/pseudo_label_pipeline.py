"""
Iterative semi-supervised (pseudo-labeling) training CLI pipeline for YOLO26.

Allows unattended multi-round SSL training, hyperparameter evolution, and flexible
pausing/resuming across training sessions.

Usage examples:
    # 1. Start standard 3-round training session (rounds 0, 1, 2):
    uv run python code/pseudo_label_pipeline.py

    # 2. Resume automatically from where you left off:
    uv run python code/pseudo_label_pipeline.py --resume

    # 3. Run only rounds 4 through 6 (e.g. overnight batch):
    uv run python code/pseudo_label_pipeline.py --start-round 4 --end-round 6

    # 4. Resume an interrupted round from its last.pt checkpoint:
    uv run python code/pseudo_label_pipeline.py --resume --resume-mid-round

    # 5. Quick 1-round, 2-epoch dry run to test paths and GPU setup:
    uv run python code/pseudo_label_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pseudo_label.config import PipelineConfig
from pseudo_label.engine import PseudoLabelPipeline
from pseudo_label.tuner import run_hyperparameter_tuning
from pseudo_label.utils import load_yaml, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterative Semi-Supervised Learning (SSL) Pipeline for YOLO26",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Resumption & round control
    round_group = parser.add_argument_group("Round & Resumption Options")
    round_group.add_argument(
        "--resume",
        action="store_true",
        help="Automatically resume pipeline from the last completed round.",
    )
    round_group.add_argument(
        "--start-round",
        type=int,
        default=None,
        help="Explicit 0-indexed round number to start or resume from.",
    )
    round_group.add_argument(
        "--end-round",
        type=int,
        default=None,
        help="Explicit 0-indexed round number to stop after (inclusive).",
    )
    round_group.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Total number of rounds in the pipeline (overrides config.n_rounds).",
    )
    round_group.add_argument(
        "--resume-mid-round",
        action="store_true",
        help="Resume an interrupted round from its last.pt checkpoint if available.",
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--size",
        type=str,
        choices=["n", "s", "m", "l", "x"],
        default="m",
        help="YOLO26 model size variant.",
    )
    model_group.add_argument(
        "--p2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use P2 small-object detection head (e.g. yolo26m-p2.yaml).",
    )
    model_group.add_argument(
        "--base-cfg",
        type=str,
        default=None,
        help="Override base architecture YAML filename.",
    )
    model_group.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Override initial pretrained weights checkpoint.",
    )

    # Training parameters
    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument(
        "--batch",
        type=int,
        default=10,
        help="Training batch size (10 benchmarked optimal for RTX 4070 12GB).",
    )
    train_group.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader worker subprocesses.",
    )
    train_group.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Maximum training epochs per pseudo-labeling round.",
    )
    train_group.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early stopping patience in epochs.",
    )
    train_group.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution in pixels.",
    )
    train_group.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device identifier (e.g. 0, 'cuda:0', 'cpu').",
    )

    # Pseudo-labeling parameters
    pseudo_group = parser.add_argument_group("Pseudo-Labeling Options")
    pseudo_group.add_argument(
        "--conf",
        type=float,
        nargs="+",
        default=[0.85, 0.80, 0.75],
        help="Confidence threshold schedule across rounds.",
    )
    pseudo_group.add_argument(
        "--max-pseudo-ratio",
        type=float,
        default=1.0,
        help="Maximum ratio of pseudo-labeled frames to ground-truth train images.",
    )

    # Evaluation options
    eval_group = parser.add_argument_group("Evaluation Options")
    eval_group.add_argument(
        "--coco-gt",
        type=str,
        default=None,
        help="Path to reference COCO ground truth annotations JSON file.",
    )
    eval_group.add_argument(
        "--coco-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute standard COCO metrics (mAP_s, mAP_m, mAP_l, peak F1) each round.",
    )

    # Hyperparameter tuning options
    tune_group = parser.add_argument_group("Hyperparameter Tuning Options")
    tune_group.add_argument(
        "--tune",
        action="store_true",
        help="Run genetic hyperparameter evolution before starting training rounds.",
    )
    tune_group.add_argument(
        "--hyp",
        type=str,
        default=None,
        help="Path to YAML file containing hyperparameter overrides.",
    )

    # Configuration file and sanity test options
    config_group = parser.add_argument_group("Configuration & Debug Options")
    config_group.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pipeline YAML configuration file.",
    )
    config_group.add_argument(
        "--save-config",
        type=str,
        default=None,
        help="Export current configuration to specified YAML path and exit.",
    )
    config_group.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Name of the experiment run.",
    )
    config_group.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Custom output run directory for pseudo-labeling rounds.",
    )
    config_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a fast 1-round, 2-epoch sanity check.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Initialize configuration from file or defaults
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig()

    # 2. Apply CLI overrides
    if args.exp_name:
        config.experiment_name = args.exp_name
    if args.run_dir:
        config.pseudo_run_dir = Path(args.run_dir).resolve()
    if args.size:
        config.model_size = args.size
    config.p2_head = args.p2
    if args.base_cfg:
        config.base_configuration = args.base_cfg
    if args.weights:
        config.pretrained_weights = args.weights
    if args.batch:
        config.batch_size = args.batch
    if args.workers:
        config.workers = args.workers
    if args.epochs:
        config.epochs_per_round = args.epochs
    if args.patience:
        config.patience = args.patience
    if args.imgsz:
        config.imgsz = args.imgsz
    if args.device:
        config.device = args.device
    if args.rounds:
        config.n_rounds = args.rounds
    if args.conf:
        config.conf_schedule = list(args.conf)
    if args.max_pseudo_ratio is not None:
        config.max_pseudo_ratio = args.max_pseudo_ratio
    if args.coco_gt:
        config.coco_gt_path = Path(args.coco_gt).resolve()
    config.enable_coco_eval = args.coco_eval

    # Apply dry-run configuration if requested
    if args.dry_run:
        config.n_rounds = 1
        config.epochs_per_round = 2
        config.patience = 2
        config.experiment_name = f"[DRY-RUN] {config.experiment_name}"
        config.pseudo_run_dir = config.runs_dir / "pseudo" / "dry_run"

    # Export config and exit if requested
    if args.save_config:
        config.save_yaml(args.save_config)
        print(f"Configuration exported to: {args.save_config}")
        return

    # 3. Setup logging
    log = setup_logging(log_file=config.log_file, logger_name="pseudo_pipeline")

    # 4. Load or tune hyperparameters
    hyp_overrides: dict = {}
    if args.tune:
        hyp_overrides = run_hyperparameter_tuning(config, logger=log)
    elif args.hyp:
        hyp_path = Path(args.hyp).resolve()
        if hyp_path.exists():
            hyp_overrides = load_yaml(hyp_path)
            log.info(f"Loaded hyperparameters from {hyp_path}")
        else:
            log.warning(f"Hyperparameters file {hyp_path} not found.")
    else:
        # Default candidate search in known project locations
        known_hyp_paths = [
            config.tune_run_dir / "evolve-6" / "best_hyperparameters.yaml",
            config.root_dir / "code" / "best_hyperparameters.yaml",
        ]
        for p in known_hyp_paths:
            if p.exists():
                hyp_overrides = load_yaml(p)
                log.info(f"Using existing tuned hyperparameters from {p}")
                break

    # 5. Instantiate and run pipeline
    pipeline = PseudoLabelPipeline(config=config, logger=log)

    pipeline.run(
        start_round=args.start_round,
        end_round=args.end_round,
        resume=args.resume,
        resume_mid_round=args.resume_mid_round,
        hyp_overrides=hyp_overrides,
    )


if __name__ == "__main__":
    main()
