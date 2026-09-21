"""The PolyCard neural network (Section 4.6 of the paper).

The architecture is a three-block MLP stack, adapted from the set-convolution
model of MSCN (Kipf et al., CIDR 2019):

    dataset vector  --MLP--> hid \
                                  concat --MLP--> sigmoid --> [0, 1]
    polygon vector  --MLP--> hid /

Each MLP is two fully connected layers with ReLU activations, i.e.
``MLP(x) = ReLU(ReLU(x W1 + b1) W2 + b2)``.  The output layer replaces the
final ReLU with a sigmoid because targets are normalized to ``[0, 1]``.

Fully connected layers are used throughout (no convolutions, no attention):
the input is only ``2n`` floats wide, so a small MLP both trains in minutes
and answers a query in a few microseconds.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MLP", "PolyCardNet"]


class MLP(nn.Module):
    """Two-layer ReLU MLP: ``MLP(x) = ReLU(ReLU(x W1 + b1) W2 + b2)``."""

    def __init__(self, in_features: int, hid_units: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hid_units)
        self.fc2 = nn.Linear(hid_units, hid_units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc2(F.relu(self.fc1(x))))


class PolyCardNet(nn.Module):
    """PolyCard estimator.

    Parameters
    ----------
    dataset_feats :
        Width of the dataset one-hot vector (number of source datasets).
    polygon_feats :
        Width of the polygon vector, i.e. ``2 * num_vertices``.
    hid_units :
        Hidden width of every MLP (256 in the paper).
    """

    def __init__(self, dataset_feats: int, polygon_feats: int, hid_units: int = 256):
        super().__init__()
        self.dataset_feats = int(dataset_feats)
        self.polygon_feats = int(polygon_feats)
        self.hid_units = int(hid_units)

        self.dataset_mlp = MLP(self.dataset_feats, self.hid_units)
        self.polygon_mlp = MLP(self.polygon_feats, self.hid_units)
        self.out_mlp1 = nn.Linear(self.hid_units * 2, self.hid_units)
        self.out_mlp2 = nn.Linear(self.hid_units, 1)

    def forward(self, dataset_vec: torch.Tensor, polygon_vec: torch.Tensor):
        """Return estimated (normalized) cardinality in ``[0, 1]``.

        Shapes: ``dataset_vec`` is ``(B, M)``, ``polygon_vec`` is
        ``(B, 2n)``; the output is ``(B, 1)``.
        """
        hid_dataset = self.dataset_mlp(dataset_vec)
        hid_polygon = self.polygon_mlp(polygon_vec)
        hid = torch.cat((hid_dataset, hid_polygon), dim=1)
        hid = F.relu(self.out_mlp1(hid))
        return torch.sigmoid(self.out_mlp2(hid))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
