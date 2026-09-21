"""Evaluation metrics -- q-error and its percentiles."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

__all__ = [
    "ZERO_TOL",
    "qerror",
    "qerror_percentiles",
    "summarize_qerror",
    "format_summary",
]


ZERO_TOL = 1e-6

# Percentiles reported in Table 2 of the paper.
PERCENTILES = (25, 50, 75, 95, 99)


def qerror(preds, truths, zero_tol: float = ZERO_TOL) -> np.ndarray:
    """``max(pred, true) / min(pred, true)`` for every query.

    Queries whose true *or* estimated cardinality is zero are skipped,
    matching the evaluation protocol of the paper.
    """
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    truths = np.asarray(truths, dtype=np.float64).reshape(-1)
    mask = (np.abs(preds) > zero_tol) & (np.abs(truths) > zero_tol)
    preds = preds[mask]
    truths = truths[mask]
    if preds.size == 0:
        return np.empty(0, dtype=np.float64)
    return np.maximum(preds, truths) / np.minimum(preds, truths)


def qerror_percentiles(
    preds,
    truths,
    percentiles: Sequence[int] = PERCENTILES,
) -> Dict[int, float]:
    """q-error at the requested percentiles (25/50/75/95/99 by default)."""
    errors = qerror(preds, truths)
    if errors.size == 0:
        return {int(p): float("nan") for p in percentiles}
    return {int(p): float(np.percentile(errors, p)) for p in percentiles}


def summarize_qerror(
    preds,
    truths,
    percentiles: Sequence[int] = PERCENTILES,
) -> Dict[str, float]:
    """Percentiles plus mean/max and the number of evaluated queries."""
    errors = qerror(preds, truths)
    out = {f"{p}th": float(np.percentile(errors, p)) for p in percentiles}
    out["mean"] = float(np.mean(errors)) if errors.size else float("nan")
    out["max"] = float(np.max(errors)) if errors.size else float("nan")
    out["num_queries"] = int(errors.size)
    return out


def format_summary(summary: Dict[str, float]) -> str:
    """Render :func:`summarize_qerror` output as one readable line."""
    parts = [f"{k}={v:.2f}" for k, v in summary.items() if k != "num_queries"]
    return f"q-error [{'  '.join(parts)}]  n={summary['num_queries']}"
