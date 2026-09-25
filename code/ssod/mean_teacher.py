"""
Mean Teacher Semi-Supervised Object Detection (SSOD) training loop for YOLO.

Features:
1. Supervised Warm-up Phase:
   - Trains Teacher on Ground-Truth data only for n-epochs (or loads from --warmup-weights).
   - Initializes Student model from trained Teacher weights (theta_S <- theta_T).
2. Mean Teacher Training Loop:
   - Starts each epoch by generating pseudo-labels on unlabeled pool using PseudoLabelGenerator.
   - Synchronizes SSODDatasetHandler with 25% GT tail / 75% GT+PL head sequencing and 10% mosaic closing.
   - Trains Student with YOLO default detection loss (with optional unsupervised weight lambda_u).
   - Updates Teacher EMA at every training step.
   - Evaluates Teacher on GT validation split every epoch:
     extracts all Ultralytics default metrics and pycocotools COCO metrics.
   - Selects and saves best.pt checkpoint based on Ultralytics fitness.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm  # type: ignore[import-untyped]
import typer
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace

try:
    from ssod.dataset import SSODAugmentConfig, SSODDatasetHandler
    from ssod.metrics import evaluate_coco
    from ssod.pseudo_labeling import PseudoLabelGenerator
    from ssod.utilities import get_logger
except ImportError:
    from dataset import SSODAugmentConfig, SSODDatasetHandler  # type: ignore[import-not-found,no-redef]
    from metrics import evaluate_coco  # type: ignore[import-not-found,no-redef]
    from pseudo_labeling import PseudoLabelGenerator  # type: ignore[import-not-found,no-redef]
    from utilities import get_logger  # type: ignore[import-not-found,no-redef]

app = typer.Typer(
    name="mean-teacher",
    help="Mean Teacher Semi-Supervised Training for YOLO.",
    add_completion=False,
)


@dataclass
class MeanTeacherConfig:
    """Configuration for Mean Teacher training."""

    # Models & weights
    model_name: str = "yolo26n.pt"
    warmup_weights: Path | None = None

    # Datasets
    gt_data_yaml: Path = field(default_factory=lambda: Path("code/data_sliced.yaml"))
    pl_images_dir: Path = field(default_factory=lambda: Path("data/video_frames/images/all"))
    pl_labels_dir: Path = field(default_factory=lambda: Path("data/video_frames/labels/all"))
    coco_gt_path: Path | None = field(
        default_factory=lambda: Path("data/flow_img/val/annotations.json")
    )

    # Schedules & dimensions
    warmup_epochs: int = 10
    ssl_epochs: int = 50
    batch_size: int = 10
    imgsz: int = 640
    workers: int = 4
    device: str | int | None = None

    # Optimizer & Scheduler
    lr0: float = 0.002
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    amp: bool = True

    # EMA & SSL policies
    ema_decay: float = 0.999
    tail_ratio: float = 0.25
    close_mosaic_ratio: float = 0.10
    unsupervised_weight: float = 1.0

    # Pseudo-labeling parameters
    pseudo_conf: float = 0.001
    pseudo_iou: float = 0.5
    pseudo_half: bool = True

    # Logging and output
    runs_dir: Path = field(default_factory=lambda: Path("runs/mean_teacher"))
    experiment_name: str = "exp"
    seed: int = 42

    def __post_init__(self) -> None:
        self.gt_data_yaml = Path(self.gt_data_yaml).resolve()
        self.pl_images_dir = Path(self.pl_images_dir).resolve()
        self.pl_labels_dir = Path(self.pl_labels_dir).resolve()
        if self.coco_gt_path is not None:
            self.coco_gt_path = Path(self.coco_gt_path).resolve()
        if self.warmup_weights is not None:
            self.warmup_weights = Path(self.warmup_weights).resolve()
        self.runs_dir = Path(self.runs_dir).resolve()


def update_teacher_ema(
    teacher_model: nn.Module,
    student_model: nn.Module,
    decay: float = 0.999,
) -> None:
    """
    Update teacher model parameters via Exponential Moving Average at every step:
    theta_T = decay * theta_T + (1 - decay) * theta_S.
    Non-floating-point buffers (e.g. num_batches_tracked) are copied directly.
    """
    with torch.no_grad():
        t_sd = teacher_model.state_dict()
        s_sd = student_model.state_dict()
        for k, v in t_sd.items():
            if k in s_sd:
                if v.dtype.is_floating_point:
                    v.mul_(decay).add_(s_sd[k].detach().to(v.device), alpha=1.0 - decay)
                else:
                    v.copy_(s_sd[k].detach())


def split_batch(batch: dict[str, Any], mask: torch.Tensor) -> dict[str, Any]:
    """Slice a collated batch dictionary to only include samples matching boolean mask."""
    indices = torch.where(mask)[0]
    new_batch: dict[str, Any] = {"img": batch["img"][mask]}

    if "bboxes" in batch and len(batch["bboxes"]) > 0:
        b_mask = torch.isin(batch["batch_idx"], indices.float())
        new_batch["bboxes"] = batch["bboxes"][b_mask]
        new_batch["cls"] = batch["cls"][b_mask]
        idx_map = {idx.item(): i for i, idx in enumerate(indices)}
        remapped = [idx_map[int(x.item())] for x in batch["batch_idx"][b_mask]]
        new_batch["batch_idx"] = torch.tensor(
            remapped, dtype=torch.float32, device=batch["batch_idx"].device
        )
    else:
        new_batch["bboxes"] = torch.empty((0, 4), device=batch["img"].device)
        new_batch["cls"] = torch.empty((0, 1), device=batch["img"].device)
        new_batch["batch_idx"] = torch.empty((0,), device=batch["img"].device)

    return new_batch


def compute_batch_loss(
    model: nn.Module,
    batch: dict[str, Any],
    unsupervised_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute YOLO default detection loss on batch.
    If unsupervised_weight != 1.0 and both GT and Pseudo items are present,
    applies unsupervised_weight to pseudo-labeled loss.
    Returns (loss_tensor, loss_items_detach).
    """
    bs = batch["img"].shape[0]
    is_pseudo = batch.get("is_pseudo")

    if unsupervised_weight == 1.0 or is_pseudo is None or not torch.is_tensor(is_pseudo):
        loss_tuple = model.loss(batch)
        loss_sum = loss_tuple[0].sum() / max(1, bs)
        loss_items = loss_tuple[1]
        return loss_sum, loss_items

    has_gt = (~is_pseudo).any()
    has_pl = is_pseudo.any()

    if has_gt and has_pl:
        b_gt = split_batch(batch, ~is_pseudo)
        b_pl = split_batch(batch, is_pseudo)
        n_gt = b_gt["img"].shape[0]
        n_pl = b_pl["img"].shape[0]

        l_gt_tuple = model.loss(b_gt)
        l_pl_tuple = model.loss(b_pl)

        l_gt = l_gt_tuple[0].sum()
        l_pl = l_pl_tuple[0].sum()

        eff_n = n_gt + (unsupervised_weight * n_pl)
        loss_sum = (l_gt + (unsupervised_weight * l_pl)) / max(1e-6, eff_n)
        loss_items = l_gt_tuple[1]
        return loss_sum, loss_items
    elif has_pl and not has_gt:
        loss_tuple = model.loss(batch)
        loss_sum = (loss_tuple[0].sum() / max(1, bs)) * unsupervised_weight
        return loss_sum, loss_tuple[1]
    else:
        loss_tuple = model.loss(batch)
        loss_sum = loss_tuple[0].sum() / max(1, bs)
        return loss_sum, loss_tuple[1]


def evaluate_model_coco(
    model: YOLO,
    gt_data_yaml: Path | str,
    coco_gt_path: Path | str | None,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str | int | None = None,
    save_dir: Path | str | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    """
    Evaluate YOLO model on validation split, extracting Ultralytics metrics,
    computing fitness, and evaluating pycocotools COCO metrics.
    """
    log = logger or get_logger()
    val_kwargs: dict[str, Any] = {
        "data": str(gt_data_yaml),
        "split": "val",
        "save_json": True,
        "batch": batch_size,
        "imgsz": imgsz,
        "verbose": False,
        "plots": False,
    }
    if device is not None:
        val_kwargs["device"] = device
    if save_dir is not None:
        val_kwargs["project"] = str(Path(save_dir).parent)
        val_kwargs["name"] = str(Path(save_dir).name)
        val_kwargs["exist_ok"] = True

    val_results = model.val(**val_kwargs)
    res_dict = val_results.results_dict

    # Extract Ultralytics metrics & fitness
    metrics: dict[str, float] = {
        "metrics/precision(B)": float(res_dict.get("metrics/precision(B)", 0.0)),
        "metrics/recall(B)": float(res_dict.get("metrics/recall(B)", 0.0)),
        "metrics/mAP50(B)": float(res_dict.get("metrics/mAP50(B)", 0.0)),
        "metrics/mAP50-95(B)": float(res_dict.get("metrics/mAP50-95(B)", 0.0)),
        "val/box_loss": float(res_dict.get("val/box_loss", 0.0)),
        "val/cls_loss": float(res_dict.get("val/cls_loss", 0.0)),
        "val/dfl_loss": float(res_dict.get("val/dfl_loss", 0.0)),
        "fitness": float(
            res_dict.get(
                "fitness",
                0.1 * float(res_dict.get("metrics/mAP50(B)", 0.0))
                + 0.9 * float(res_dict.get("metrics/mAP50-95(B)", 0.0)),
            )
        ),
    }

    # Evaluate COCO metrics if COCO ground truth JSON is available
    if coco_gt_path is not None and Path(coco_gt_path).exists():
        preds_json = Path(val_results.save_dir) / "predictions.json"
        coco_res = evaluate_coco(
            coco_gt_path=coco_gt_path,
            predictions_path=preds_json,
            logger=log,
        )
        for k, v in coco_res.items():
            metrics[f"coco/{k}"] = float(v)

    return metrics


class MeanTeacherTrainer:
    """
    Orchestrates Supervised Warm-up and Mean Teacher SSL training loops.
    """

    def __init__(
        self,
        config: MeanTeacherConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or MeanTeacherConfig()
        self.exp_dir = self.config.runs_dir / self.config.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir = self.exp_dir / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logger or get_logger(
            log_file=self.exp_dir / "train.log",
            level=logging.INFO,
        )

        # Set device
        if self.config.device is not None:
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.logger.info(
            f"Initialized MeanTeacherTrainer on device={self.device} (runs_dir={self.exp_dir})"
        )

    def run_warmup(self) -> Path:
        """
        Execute Phase 1: Supervised warm-up on Ground-Truth dataset only.
        If warmup_weights was supplied and exists, skips training and returns path.
        """
        if self.config.warmup_weights is not None and self.config.warmup_weights.exists():
            self.logger.info(
                f"Using existing warm-up checkpoint: {self.config.warmup_weights}. Skipping warm-up training."
            )
            return self.config.warmup_weights

        self.logger.info(
            f"=== Phase 1: Supervised Warm-up ({self.config.warmup_epochs} epochs) ==="
        )
        warmup_dir = self.exp_dir / "warmup"

        teacher_yolo = YOLO(self.config.model_name)
        train_kwargs: dict[str, Any] = {
            "data": str(self.config.gt_data_yaml),
            "epochs": self.config.warmup_epochs,
            "batch": self.config.batch_size,
            "imgsz": self.config.imgsz,
            "workers": self.config.workers,
            "project": str(warmup_dir.parent),
            "name": str(warmup_dir.name),
            "exist_ok": True,
            "verbose": True,
        }
        if self.config.device is not None:
            train_kwargs["device"] = self.config.device

        train_results = teacher_yolo.train(**train_kwargs)
        warmup_best = Path(train_results.save_dir) / "weights" / "best.pt"

        # Copy to main weights dir
        dest = self.weights_dir / "warmup_best.pt"
        if warmup_best.exists():
            torch.save(torch.load(warmup_best, weights_only=False), dest)
            self.logger.info(f"Warm-up complete. Saved warm-up weights to {dest}")
            return dest

        last_pt = Path(train_results.save_dir) / "weights" / "last.pt"
        torch.save(torch.load(last_pt, weights_only=False), dest)
        return dest

    def train_mean_teacher(
        self,
        warmup_weights: Path | str | None = None,
    ) -> dict[str, Any]:
        """
        Execute Phase 2: Mean Teacher Semi-Supervised training loop.
        """
        # 1. Obtain warm-up weights
        ckpt_path = warmup_weights or self.run_warmup()

        self.logger.info(f"=== Phase 2: Mean Teacher SSL Training ({self.config.ssl_epochs} epochs) ===")
        self.logger.info(f"Loading Teacher and Student models from {ckpt_path}...")

        # Initialize Teacher and Student
        teacher = YOLO(str(ckpt_path))
        student = YOLO(str(ckpt_path))

        # Ensure model.args has attribute access for default YOLO loss
        cfg_dict = deepcopy(DEFAULT_CFG_DICT)
        if hasattr(student, "overrides") and isinstance(student.overrides, dict):
            cfg_dict.update(student.overrides)
        student.model.args = IterableSimpleNamespace(**cfg_dict)
        teacher.model.args = IterableSimpleNamespace(**cfg_dict)

        # Place models on device
        student.model = student.model.to(self.device)
        teacher.model = teacher.model.to(self.device)
        teacher.model.eval()

        for p in student.model.parameters():
            p.requires_grad = True
        for p in teacher.model.parameters():
            p.requires_grad = False

        # 2. Initialize SSODDatasetHandler and PseudoLabelGenerator
        pseudo_generator = PseudoLabelGenerator(
            teacher_model=teacher,
            batch_size=self.config.batch_size * 2,
            half=self.config.pseudo_half and self.device.type == "cuda",
            workers=self.config.workers,
            conf=self.config.pseudo_conf,
            iou_threshold=self.config.pseudo_iou,
            device=self.device,
            logger=self.logger,
        )

        augment_cfg = SSODAugmentConfig(
            tail_ratio=self.config.tail_ratio,
            close_mosaic_ratio=self.config.close_mosaic_ratio,
        )

        dataset_handler = SSODDatasetHandler(
            gt_data_yaml=self.config.gt_data_yaml,
            pl_images_dir=self.config.pl_images_dir,
            pl_labels_dir=self.config.pl_labels_dir,
            imgsz=self.config.imgsz,
            batch_size=self.config.batch_size,
            workers=self.config.workers,
            seed=self.config.seed,
            augment_cfg=augment_cfg,
            logger=self.logger,
        )

        # 3. Setup Optimizer & Scheduler for Student
        optimizer = torch.optim.AdamW(
            student.model.parameters(),
            lr=self.config.lr0,
            betas=(self.config.momentum, 0.999),
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.ssl_epochs,
            eta_min=self.config.lr0 * self.config.lrf,
        )
        scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.config.amp and self.device.type == "cuda",
        )

        # 4. Results tracking
        results_csv = self.exp_dir / "results.csv"
        csv_header_written = False
        best_fitness = -1.0
        best_epoch = -1
        training_history: list[dict[str, Any]] = []

        # 5. Training Loop
        total_epochs = self.config.ssl_epochs
        for epoch in range(total_epochs):
            epoch_start_time = time.time()
            self.logger.info(f"\n--- Epoch {epoch + 1}/{total_epochs} ---")

            # A. Generate Pseudo-labels at epoch start using Teacher model
            self.logger.info(f"Generating pseudo-labels with Teacher (conf={self.config.pseudo_conf})...")
            gen_out = pseudo_generator.generate(
                images_dir=self.config.pl_images_dir,
                labels_dir=self.config.pl_labels_dir,
            )
            dataset_handler.reload_pseudo_labels(filtered_df=gen_out.get("filtered_df"))

            # B. Configure epoch (25% GT tail sampling, 10% mosaic closing)
            train_loader = dataset_handler.get_train_loader()
            dataset_handler.set_epoch(
                epoch=epoch,
                total_epochs=total_epochs,
                dataloader=train_loader,
            )

            # C. Student Training over Batches
            student.model.train()
            train_losses: list[float] = []

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{total_epochs} [Student Train]",
                leave=False,
            )

            for batch in pbar:
                # Move tensors to device & normalize image to float [0, 1]
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
                batch["img"] = batch["img"].float() / 255.0

                optimizer.zero_grad()

                with torch.amp.autocast(
                    self.device.type,
                    enabled=self.config.amp and self.device.type == "cuda",
                ):
                    loss, loss_items = compute_batch_loss(
                        model=student.model,
                        batch=batch,
                        unsupervised_weight=self.config.unsupervised_weight,
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()

                # D. Update Teacher EMA at EVERY STEP
                update_teacher_ema(
                    teacher_model=teacher.model,
                    student_model=student.model,
                    decay=self.config.ema_decay,
                )

                train_losses.append(loss.item())
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            scheduler.step()
            mean_train_loss = float(np.mean(train_losses)) if train_losses else 0.0

            # E. Evaluate Teacher Model on GT Validation split
            self.logger.info("Evaluating Teacher model on validation split...")
            teacher.model.eval()
            val_save_dir = self.exp_dir / f"val_epoch_{epoch + 1}"

            val_metrics = evaluate_model_coco(
                model=teacher,
                gt_data_yaml=self.config.gt_data_yaml,
                coco_gt_path=self.config.coco_gt_path,
                imgsz=self.config.imgsz,
                batch_size=self.config.batch_size * 2,
                device=self.device,
                save_dir=val_save_dir,
                logger=self.logger,
            )

            epoch_time = time.time() - epoch_start_time
            fitness = val_metrics.get("fitness", 0.0)
            map50 = val_metrics.get("metrics/mAP50(B)", 0.0)
            map50_95 = val_metrics.get("metrics/mAP50-95(B)", 0.0)

            self.logger.info(
                f"Epoch {epoch + 1} Result ({epoch_time:.1f}s): "
                f"Train Loss={mean_train_loss:.4f} | "
                f"Teacher mAP50={map50:.4f}, mAP50-95={map50_95:.4f}, Fitness={fitness:.4f}"
            )

            # Record epoch log
            record = {
                "epoch": epoch + 1,
                "train_loss": mean_train_loss,
                "epoch_time": epoch_time,
                **val_metrics,
            }
            training_history.append(record)

            # Append to results.csv
            fieldnames = list(record.keys())
            if not csv_header_written:
                with open(results_csv, "w", encoding="utf-8") as f:
                    f.write(",".join(fieldnames) + "\n")
                csv_header_written = True

            with open(results_csv, "a", encoding="utf-8") as f:
                f.write(",".join(str(record[k]) for k in fieldnames) + "\n")

            # Checkpoint management
            last_path = self.weights_dir / "last.pt"
            teacher.save(str(last_path))

            if fitness > best_fitness:
                best_fitness = fitness
                best_epoch = epoch + 1
                best_path = self.weights_dir / "best.pt"
                teacher.save(str(best_path))
                self.logger.info(
                    f"New best model saved to {best_path} (fitness={best_fitness:.4f} at epoch {best_epoch})."
                )

        self.logger.info(
            f"\n=== Training Complete ===\n"
            f"Best Fitness={best_fitness:.4f} observed at Epoch {best_epoch}.\n"
            f"Results saved to {self.exp_dir}"
        )
        return {
            "best_fitness": best_fitness,
            "best_epoch": best_epoch,
            "best_model": str(self.weights_dir / "best.pt"),
            "results_csv": str(results_csv),
            "history": training_history,
        }


@app.command()
def main(
    gt_yaml: Path = typer.Option(
        Path("code/data_sliced.yaml"), "--gt-yaml", help="Path to GT dataset YAML."
    ),
    pl_images: Path = typer.Option(
        Path("data/video_frames/images/all"), "--pl-images", help="Path to unlabeled images dir."
    ),
    pl_labels: Path = typer.Option(
        Path("data/video_frames/labels/all"), "--pl-labels", help="Path to pseudo-labels dir."
    ),
    coco_gt: Optional[Path] = typer.Option(
        None, "--coco-gt", help="Path to reference COCO annotations JSON."
    ),
    model: str = typer.Option("yolo26m.pt", "--model", help="Base model weights."),
    warmup_weights: Optional[Path] = typer.Option(
        None, "--warmup-weights", help="Optional path to existing warm-up checkpoint."
    ),
    warmup_epochs: int = typer.Option(10, "--warmup-epochs", help="Supervised warm-up epochs."),
    ssl_epochs: int = typer.Option(50, "--ssl-epochs", help="Mean Teacher SSL epochs."),
    batch: int = typer.Option(10, "--batch", help="Batch size."),
    imgsz: int = typer.Option(640, "--imgsz", help="Input image size."),
    workers: int = typer.Option(4, "--workers", help="DataLoader worker processes."),
    device: Optional[str] = typer.Option(None, "--device", help="PyTorch device."),
    ema_decay: float = typer.Option(0.999, "--ema-decay", help="Teacher EMA decay per step."),
    unsupervised_weight: float = typer.Option(1.0, "--unsupervised-weight", help="Weight lambda_u for pseudo loss."),
    project: Path = typer.Option(Path("runs/mean_teacher"), "--project", help="Output project directory."),
    name: str = typer.Option("exp", "--name", help="Experiment name."),
) -> None:
    """Run Mean Teacher semi-supervised training."""
    cfg = MeanTeacherConfig(
        model_name=model,
        warmup_weights=warmup_weights,
        gt_data_yaml=gt_yaml,
        pl_images_dir=pl_images,
        pl_labels_dir=pl_labels,
        coco_gt_path=coco_gt,
        warmup_epochs=warmup_epochs,
        ssl_epochs=ssl_epochs,
        batch_size=batch,
        imgsz=imgsz,
        workers=workers,
        device=device,
        ema_decay=ema_decay,
        unsupervised_weight=unsupervised_weight,
        runs_dir=project,
        experiment_name=name,
    )
    trainer = MeanTeacherTrainer(config=cfg)
    trainer.train_mean_teacher()


if __name__ == "__main__":
    app()
