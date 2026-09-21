"""Training loop for PolyCard (Section 4.7 of the paper).

The loss is the **q-error** computed in *cardinality* space, i.e. the
normalized network output is first mapped back through
``exp( . ) - eps`` and only then compared with the true cardinality.
Training directly on the q-error is what keeps the high percentiles
(95th / 99th) low -- the metric the paper cares about.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional

import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from .dataset import PolyCardDataset, make_loaders, save_label_stats
from .featurize import LABEL_EPS
from .io_utils import ensure_dir
from .model import PolyCardNet

__all__ = ["qerror_loss", "train_polycard", "DEFAULT_HPARAMS"]


DEFAULT_HPARAMS = {
    "epochs": 100,
    "batch_size": 128,
    "hid_units": 256,
    "lr": 1e-3,
    "lr_gamma": 0.97,
    "eval_every": 2,
    "label_eps": LABEL_EPS,
}


def qerror_loss(
    preds_norm: torch.Tensor,
    targets_norm: torch.Tensor,
    min_val: float,
    max_val: float,
    eps: float = LABEL_EPS,
) -> torch.Tensor:
    """Mean q-error between predictions and targets (both normalized).

    Both tensors are un-normalized first so that the loss is the real
    q-error and not a distorted version of it.
    """
    preds = torch.exp(preds_norm * (max_val - min_val) + min_val) - eps
    targets = torch.exp(targets_norm * (max_val - min_val) + min_val) - eps
    # Guard against zero cardinalities (log(0 + eps) is finite, but the
    # round-trip can still land on 0 for extreme inputs).
    preds = torch.clamp(preds, min=1.0)
    targets = torch.clamp(targets, min=1.0)
    ratio = torch.max(preds, targets) / torch.min(preds, targets)
    return ratio.mean()


def _to_device(batch, device: torch.device):
    return [t.to(device) for t in batch]


@torch.no_grad()
def _evaluate_loss(
    model: PolyCardNet,
    loader: DataLoader,
    device: torch.device,
    min_val: float,
    max_val: float,
    eps: float,
) -> float:
    model.eval()
    total, count = 0.0, 0
    for dataset_vec, polygon_vec, targets in loader:
        dataset_vec = dataset_vec.to(device)
        polygon_vec = polygon_vec.to(device)
        targets = targets.to(device).reshape(-1, 1)
        out = model(dataset_vec, polygon_vec)
        total += qerror_loss(out, targets, min_val, max_val, eps).item()
        count += 1
    model.train()
    return total / max(count, 1)


def train_polycard(
    train_set: PolyCardDataset,
    test_set: Optional[PolyCardDataset] = None,
    min_val: float = 0.0,
    max_val: float = 1.0,
    epochs: int = DEFAULT_HPARAMS["epochs"],
    batch_size: int = DEFAULT_HPARAMS["batch_size"],
    hid_units: int = DEFAULT_HPARAMS["hid_units"],
    lr: float = DEFAULT_HPARAMS["lr"],
    lr_gamma: float = DEFAULT_HPARAMS["lr_gamma"],
    eval_every: int = DEFAULT_HPARAMS["eval_every"],
    label_eps: float = LABEL_EPS,
    device: Optional[str] = None,
    out_dir: Optional[str] = None,
    log: Callable[[str], None] = print,
    on_epoch: Optional[Callable[[int, dict], None]] = None,
    extra_stats: Optional[Dict] = None,
) -> Dict:
    """Train PolyCard and optionally checkpoint it.

    Returns a dict with the trained ``model``, the loss history and the
    label normalization bounds that :func:`polycard.evaluate` needs.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    train_loader, test_loader = make_loaders(train_set, test_set, batch_size=batch_size)

    model = PolyCardNet(
        dataset_feats=train_set.dataset_feats,
        polygon_feats=train_set.polygon_feats,
        hid_units=hid_units,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_gamma)

    log(
        "PolyCard: dataset_feats=%d polygon_feats=%d hid=%d params=%d device=%s"
        % (
            train_set.dataset_feats,
            train_set.polygon_feats,
            hid_units,
            model.count_parameters(),
            device,
        )
    )

    history: List[float] = []
    model.train()
    for epoch in range(epochs):
        t0 = time.time()
        running, seen = 0.0, 0
        for dataset_vec, polygon_vec, targets in train_loader:
            dataset_vec = dataset_vec.to(device)
            polygon_vec = polygon_vec.to(device)
            targets = targets.to(device).reshape(-1, 1)

            optimizer.zero_grad()
            out = model(dataset_vec, polygon_vec)
            loss = qerror_loss(out, targets, min_val, max_val, label_eps)
            loss.backward()
            optimizer.step()
            running += loss.item() * targets.shape[0]
            seen += targets.shape[0]
        scheduler.step()

        epoch_loss = running / max(seen, 1)
        history.append(epoch_loss)
        msg = f"epoch {epoch + 1}/{epochs}  loss={epoch_loss:.4f}  ({time.time() - t0:.1f}s)"
        if test_loader is not None and (epoch + 1) % eval_every == 0:
            val_loss = _evaluate_loss(
                model, test_loader, device, min_val, max_val, label_eps
            )
            msg += f"  val_loss={val_loss:.4f}"
        log(msg)
        if on_epoch is not None:
            on_epoch(epoch, {"loss": epoch_loss, "epoch": epoch})

    result = {
        "model": model,
        "history": history,
        "min_val": float(min_val),
        "max_val": float(max_val),
        "label_eps": float(label_eps),
        "hparams": {
            "epochs": epochs,
            "batch_size": batch_size,
            "hid_units": hid_units,
            "lr": lr,
            "lr_gamma": lr_gamma,
        },
    }

    if out_dir is not None:
        ensure_dir(out_dir)
        torch.save(model.state_dict(), os.path.join(out_dir, "polycard.pt"))
        # Everything inference needs to reproduce the *training-time*
        # featurization.  In particular the coordinate normalization bounds:
        # without them PolyCardEstimator would silently fall back to the
        # global lon/lat bounds and estimate garbage for models trained with
        # --norm-from-data.
        stats = {
            "min_val": float(min_val),
            "max_val": float(max_val),
            "label_eps": float(label_eps),
            "dataset_feats": train_set.dataset_feats,
            "polygon_feats": train_set.polygon_feats,
            "hid_units": hid_units,
            "history": history,
        }
        if extra_stats:
            stats.update(extra_stats)
        save_label_stats(os.path.join(out_dir, "label_stats.json"), stats)
        log(f"saved checkpoint to {out_dir}")

    return result
