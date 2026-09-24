"""
deep_model package
==================
QuantFormer – State-of-the-art directional prediction for financial panel time-series.
"""

from .model import QuantFormer, FocalLoss, TCNEncoder, DualPathTemporalEncoder, CrossModalFusion
from .dataset import QRTDataset, make_loaders, encode_groups

__all__ = [
    "QuantFormer",
    "FocalLoss",
    "TCNEncoder",
    "DualPathTemporalEncoder",
    "CrossModalFusion",
    "QRTDataset",
    "make_loaders",
    "encode_groups",
]
