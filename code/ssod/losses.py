"""
Per-box weighted YOLO detection loss for pseudo-labeled training.

`WeightedDetectionLoss` importance-weights each ground-truth box's foreground contribution
(classification BCE on its assigned anchors, IoU and DFL box terms) by `batch["pl_weight"]`.
Background anchors and the loss normalisation (`target_scores_sum`) are left untouched, so a
weight-0 box acts as an ignore region: its anchors are neither pushed to background nor to the box.
Without `pl_weight` in the batch it is numerically identical to `v8DetectionLoss`.
"""

from __future__ import annotations

from typing import Any

import torch
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


class WeightedDetectionLoss(v8DetectionLoss):
    """
    v8DetectionLoss with per-box importance weights.

    `get_assigned_targets_and_loss` mirrors ultralytics 8.4.41; the equivalence test in
    tests/test_losses.py guards against drift when Ultralytics is upgraded.
    """

    def get_assigned_targets_and_loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, Any]
    ) -> tuple:
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )

        # Targets
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Per-anchor importance weight: the assigned box's weight on foreground anchors, 1 elsewhere
        anchor_weight = None
        box_weights = batch.get("pl_weight")
        if box_weights is not None and box_weights.numel() > 0:
            padded = self.preprocess(
                torch.cat(
                    (batch_idx, box_weights.view(-1, 1), batch["bboxes"]), 1
                ).to(self.device),
                batch_size,
                scale_tensor=imgsz[[1, 0, 1, 0]],
            )[..., 0]  # (b, max_boxes), padded exactly like the targets
            if padded.shape[1] > 0:
                assigned = padded.gather(1, target_gt_idx)
                anchor_weight = torch.where(fg_mask, assigned, torch.ones_like(assigned))

        # Cls loss with optional class weighting
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # (bs, num_anchors, nc)
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        if anchor_weight is not None:
            bce_loss = bce_loss * anchor_weight.unsqueeze(-1).to(bce_loss.dtype)
        loss[1] = bce_loss.sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            box_target_scores = (
                target_scores
                if anchor_weight is None
                else target_scores * anchor_weight.unsqueeze(-1).to(target_scores.dtype)
            )
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                box_target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )  # loss(box, cls, dfl)
