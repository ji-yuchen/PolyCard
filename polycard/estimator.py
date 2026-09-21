"""One-line inference: load a checkpoint, estimate a cardinality.

    from polycard import PolyCardEstimator

    est = PolyCardEstimator.load("results/sports")
    est.estimate("POLYGON ((...))")       # -> 4213
    est.estimate_many([wkt1, wkt2])       # -> array([...])

The estimator owns everything needed for inference: the network weights,
the label normalization bounds, the featurization settings (vertex count,
sampling method) and the coordinate normalization bounds.  Both are read
from ``label_stats.json`` written by ``scripts/train.py``.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Optional, Union

import numpy as np
import torch

from .featurize import featurize_polygons
from .io_utils import wkt_to_coords
from .model import PolyCardNet

__all__ = ["PolyCardEstimator"]


class PolyCardEstimator:
    """Trained PolyCard model, ready to answer intersection queries."""

    def __init__(
        self,
        model: PolyCardNet,
        min_val: float,
        max_val: float,
        label_eps: float = 1.02,
        num_vertices: int = 10,
        method: str = "adaptive",
        dataset_index: int = 0,
        num_datasets: Optional[int] = None,
        norm_origin=None,
        norm_span=None,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.label_eps = float(label_eps)
        self.num_vertices = int(num_vertices)
        self.method = method
        self.dataset_index = int(dataset_index)
        self.num_datasets = int(num_datasets or max(1, self.dataset_index + 1))
        self.norm_origin = None if norm_origin is None else np.asarray(
            norm_origin, dtype=np.float64
        )
        self.norm_span = None if norm_span is None else np.asarray(
            norm_span, dtype=np.float64
        )

    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        checkpoint_dir: str,
        device: Optional[str] = None,
        weights: str = "polycard.pt",
    ) -> "PolyCardEstimator":
        """Restore an estimator from ``scripts/train.py``'s output."""
        stats_path = os.path.join(checkpoint_dir, "label_stats.json")
        with open(stats_path, "r", encoding="utf-8") as fh:
            stats = json.load(fh)

        dataset_feats = stats.get("dataset_feats", 1)
        model = PolyCardNet(
            dataset_feats=dataset_feats,
            polygon_feats=stats.get("polygon_feats", 2 * stats.get("num_vertices", 10)),
            hid_units=stats.get("hid_units", 256),
        )
        state = torch.load(
            os.path.join(checkpoint_dir, weights),
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state)

        return cls(
            model=model,
            min_val=stats["min_val"],
            max_val=stats["max_val"],
            label_eps=stats.get("label_eps", 1.02),
            num_vertices=stats.get("num_vertices", 10),
            method=stats.get("method", "adaptive"),
            dataset_index=stats.get("dataset_index", 0),
            num_datasets=dataset_feats,
            norm_origin=stats.get("norm_origin"),
            norm_span=stats.get("norm_span"),
            device=device,
        )

    # ------------------------------------------------------------------
    def _featurize(self, polygons) -> tuple:
        polys = [
            wkt_to_coords(p) if isinstance(p, str) else np.asarray(p, dtype=np.float64)
            for p in polygons
        ]
        vec = featurize_polygons(
            polys,
            num_vertices=self.num_vertices,
            method=self.method,
            origin=self.norm_origin,
            span=self.norm_span,
        )
        dvec = np.zeros((len(polys), self.num_datasets), dtype=np.float32)
        dvec[:, self.dataset_index] = 1.0
        return vec, dvec

    @torch.no_grad()
    def _forward(self, poly_vec: np.ndarray, dataset_vec: np.ndarray) -> np.ndarray:
        out = self.model(
            torch.from_numpy(dataset_vec).to(self.device),
            torch.from_numpy(poly_vec).to(self.device),
        )
        return out.detach().cpu().numpy().reshape(-1)

    def _to_cardinality(self, norm: np.ndarray) -> np.ndarray:
        logged = norm * (self.max_val - self.min_val) + self.min_val
        card = np.exp(logged) - self.label_eps
        return np.maximum(card, 0.0)

    # ------------------------------------------------------------------
    def estimate(self, polygon: Union[str, np.ndarray]) -> int:
        """Estimated cardinality of a single query polygon."""
        return int(round(float(self.estimate_many([polygon])[0])))

    def estimate_many(self, polygons: Iterable) -> np.ndarray:
        """Estimated cardinalities, one per input polygon (WKT or array)."""
        polygons = list(polygons)
        if not polygons:
            return np.empty(0, dtype=np.int64)
        poly_vec, dataset_vec = self._featurize(polygons)
        norm = self._forward(poly_vec, dataset_vec)
        return np.rint(self._to_cardinality(norm)).astype(np.int64)

    # ------------------------------------------------------------------
    def benchmark_latency(
        self,
        polygons: Optional[Iterable] = None,
        num_samples: int = 1000,
        batch_size: int = 1024,
        repeats: int = 5,
    ) -> float:
        """Mean per-query estimation time in **microseconds**.

        Includes featurization *and* the forward pass, i.e. the full cost
        of answering one cardinality question.
        """
        if polygons is None:
            rng = np.random.default_rng(0)
            polygons = [
                np.column_stack(
                    [
                        rng.uniform(0, 1, self.num_vertices),
                        rng.uniform(0, 1, self.num_vertices),
                    ]
                )
                for _ in range(num_samples)
            ]
        polygons = list(polygons)
        if not polygons:
            return float("nan")

        poly_vec, dataset_vec = self._featurize(polygons)
        x_d = torch.from_numpy(dataset_vec).to(self.device)
        x_p = torch.from_numpy(poly_vec).to(self.device)

        for _ in range(2):  # warm-up
            self.model(x_d, x_p)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        import time

        t0 = time.perf_counter()
        for _ in range(repeats):
            self.model(x_d, x_p)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return elapsed / (len(polygons) * repeats) * 1e6
