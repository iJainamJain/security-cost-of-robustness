"""Canonical paths of the repository. Everything is relative to its root."""
from __future__ import annotations

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

SRC = ROOT / "src"
DATA = ROOT / "data"
BASELINES = ROOT / "baselines"
EXPERIMENTS = ROOT / "experiments"
NOTEBOOKS = ROOT / "notebooks"
RESULTS = ROOT / "results"

def ensure_results(*parts: str) -> pathlib.Path:
    """Return (and create) a subdirectory under results/."""
    p = RESULTS.joinpath(*parts) if parts else RESULTS
    p.mkdir(parents=True, exist_ok=True)
    return p
