"""
Unit tests for the per-box weighted YOLO detection loss.
"""

from __future__ import annotations

import pytest
import torch
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.loss import E2ELoss

from ssod.losses import WeightedDetectionLoss


@pytest.fixture(scope="module")
def model(yolo26n_path):
    torch.manual_seed(0)
    net = YOLO(yolo26n_path).model
    net.args = IterableSimpleNamespace(**dict(DEFAULT_CFG_DICT))
    net.train()
    return net


@pytest.fixture(scope="module")
def batch():
    return {
        "img": torch.rand(2, 3, 256, 256),
        "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
        "cls": torch.zeros(3, 1),
        "bboxes": torch.tensor(
            [[0.3, 0.3, 0.2, 0.2], [0.7, 0.6, 0.3, 0.25], [0.5, 0.5, 0.4, 0.4]]
        ),
    }


@pytest.fixture(scope="module")
def preds(model, batch):
    with torch.no_grad():
        return model(batch["img"])


@pytest.fixture(scope="module")
def compute_loss(model, batch, preds):
    """Evaluate a criterion on the shared batch, with optional extra batch keys."""

    def _loss(criterion, **extra):
        # A bare branch criterion takes the one-to-many head outputs; E2ELoss takes both heads
        p = preds if isinstance(criterion, E2ELoss) else preds["one2many"]
        with torch.no_grad():
            return criterion(p, {**batch, **extra})

    return _loss


@pytest.fixture
def criterion(model):
    return WeightedDetectionLoss(model, tal_topk=10)


@pytest.mark.parametrize("extra", [{}, {"pl_weight": torch.ones(3)}], ids=["no-weights", "unit-weights"])
def test_matches_stock_loss_without_and_with_unit_weights(model, compute_loss, extra):
    stock = compute_loss(E2ELoss(model))
    ours = compute_loss(E2ELoss(model, loss_fn=WeightedDetectionLoss), **extra)
    torch.testing.assert_close(ours[0], stock[0])
    torch.testing.assert_close(ours[1], stock[1])


def test_zero_weights_remove_foreground_only(criterion, compute_loss):
    full = compute_loss(criterion, pl_weight=torch.ones(3))[1]
    ignored = compute_loss(criterion, pl_weight=torch.zeros(3))[1]
    # box (0) and dfl (2) terms vanish; background classification (1) remains
    assert ignored[0].item() == 0.0
    assert ignored[2].item() == 0.0
    assert 0.0 < ignored[1].item() < full[1].item()


def test_zero_box_weight_keeps_cls_and_drops_box_terms(criterion, compute_loss):
    weights = torch.tensor([1.0, 0.6, 0.8])
    weighted = compute_loss(criterion, pl_weight=weights)[1]
    cls_only = compute_loss(criterion, pl_weight=weights, box_weight=torch.zeros(3))[1]
    torch.testing.assert_close(cls_only[1], weighted[1])
    assert cls_only[0].item() == 0.0
    assert cls_only[2].item() == 0.0


def test_box_weight_equal_to_pl_weight_is_a_no_op(criterion, compute_loss):
    weights = torch.tensor([1.0, 0.6, 0.8])
    torch.testing.assert_close(
        compute_loss(criterion, pl_weight=weights, box_weight=weights.clone())[1],
        compute_loss(criterion, pl_weight=weights)[1],
    )


def test_single_box_weight_is_monotonic(criterion, compute_loss):
    totals = [
        compute_loss(criterion, pl_weight=torch.tensor([1.0, w, 1.0]))[1].sum().item()
        for w in (0.0, 0.5, 1.0)
    ]
    assert totals[0] < totals[1] < totals[2]
