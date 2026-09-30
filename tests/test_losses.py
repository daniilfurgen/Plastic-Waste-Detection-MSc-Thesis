"""
Unit tests for the per-box weighted YOLO detection loss.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.loss import E2ELoss

# Ensure code directory is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from ssod.losses import WeightedDetectionLoss  # noqa: E402


class TestWeightedDetectionLoss(unittest.TestCase):
    """Test equivalence with the stock loss and per-box importance weighting."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        yolo = YOLO("yolo26n.pt")
        cls.model = yolo.model
        cls.model.args = IterableSimpleNamespace(**dict(DEFAULT_CFG_DICT))
        cls.model.train()
        img = torch.rand(2, 3, 256, 256)
        cls.batch = {
            "img": img,
            "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
            "cls": torch.zeros(3, 1),
            "bboxes": torch.tensor(
                [[0.3, 0.3, 0.2, 0.2], [0.7, 0.6, 0.3, 0.25], [0.5, 0.5, 0.4, 0.4]]
            ),
        }
        with torch.no_grad():
            cls.preds = cls.model(img)

    def _loss(self, criterion, **extra):
        batch = {**self.batch, **extra}
        # A bare branch criterion takes the one-to-many head outputs; E2ELoss takes both heads
        preds = self.preds if isinstance(criterion, E2ELoss) else self.preds["one2many"]
        with torch.no_grad():
            return criterion(preds, batch)

    def test_matches_stock_loss_without_and_with_unit_weights(self):
        stock = self._loss(E2ELoss(self.model))
        weighted = E2ELoss(self.model, loss_fn=WeightedDetectionLoss)
        for extra in ({}, {"pl_weight": torch.ones(3)}):
            ours = self._loss(weighted, **extra)
            torch.testing.assert_close(ours[0], stock[0])
            torch.testing.assert_close(ours[1], stock[1])

    def test_zero_weights_remove_foreground_only(self):
        criterion = WeightedDetectionLoss(self.model, tal_topk=10)
        full = self._loss(criterion, pl_weight=torch.ones(3))[1]
        ignored = self._loss(criterion, pl_weight=torch.zeros(3))[1]
        # box (0) and dfl (2) terms vanish; background classification (1) remains
        self.assertEqual(ignored[0].item(), 0.0)
        self.assertEqual(ignored[2].item(), 0.0)
        self.assertGreater(ignored[1].item(), 0.0)
        self.assertLess(ignored[1].item(), full[1].item())

    def test_single_box_weight_is_monotonic(self):
        criterion = WeightedDetectionLoss(self.model, tal_topk=10)
        totals = [
            self._loss(criterion, pl_weight=torch.tensor([1.0, w, 1.0]))[1].sum().item()
            for w in (0.0, 0.5, 1.0)
        ]
        self.assertLess(totals[0], totals[1])
        self.assertLess(totals[1], totals[2])


if __name__ == "__main__":
    unittest.main()
