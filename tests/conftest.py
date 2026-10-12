"""Shared pytest fixtures for the SSOD test suite."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
YOLO26N_PATH = REPO_ROOT / "code" / "yolo26n.pt"


@pytest.fixture(scope="session")
def yolo26n_path() -> str:
    """Absolute path to the small YOLO26 weights, independent of the working directory."""
    if not YOLO26N_PATH.exists():
        pytest.skip(f"{YOLO26N_PATH} is missing")
    return str(YOLO26N_PATH)


@pytest.fixture(scope="session")
def device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def make_image() -> Callable[..., None]:
    """Factory writing a random BGR image of the given size."""

    def _make(path: Path, width: int = 128, height: int = 128) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        cv2.imwrite(str(path), img)

    return _make


@pytest.fixture
def make_label() -> Callable[[Path, list[str]], None]:
    """Factory writing a YOLO label file from raw lines."""

    def _make(path: Path, lines: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(lines) + ("\n" if lines else "")
        path.write_text(content, encoding="utf-8")

    return _make
