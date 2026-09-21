"""Shared configuration loading and small helpers for the CLI scripts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

# Repository root (the directory that contains ``polycard/``).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "default.yaml")

# Allow ``python scripts/xxx.py`` from anywhere.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the YAML config, falling back to ``configs/default.yaml``."""
    path = path or DEFAULT_CONFIG
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML is required: pip install pyyaml") from exc
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=None, help="path to a YAML config file")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(REPO_ROOT, "data"),
        help="root of the data directory (default: <repo>/data)",
    )
    return parser


def resolve(path: str, data_dir: Optional[str] = None) -> str:
    """Make a (possibly relative) path absolute w.r.t. the repo root."""
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def path_in(data_dir: str, *parts: str) -> str:
    return os.path.join(data_dir, *parts)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def log_header(name: str) -> None:
    print("=" * 72)
    print(f"  {name}")
    print("=" * 72)


def save_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
