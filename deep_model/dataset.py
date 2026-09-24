"""
dataset.py
==========
Custom PyTorch Dataset & DataLoader factory for the QRT Asset Allocation
Directional Prediction Challenge.

Raw data layout (flat pandas columns)
--------------------------------------
  - RET_1 … RET_20          : past returns (t-1 is most recent)
  - SIGNED_VOLUME_1 … _20   : signed volume  (same convention)
  - MEDIAN_DAILY_TURNOVER    : scalar static feature
  - GROUP                    : categorical (integer-encoded below)
  - TARGET                   : raw future return; we derive the binary label

Output tensors per sample
--------------------------
  x_seq   : (seq_len=20, n_temporal_feats=2)   float32
  x_static: (1,)                                float32
  x_cat   : ()                                  long  (scalar index)
  label   : ()                                  long  (0 or 1)
  aux_tgt : ()                                  float32  (past volatility for auxiliary loss)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: build column lists in chronological order (t-20 → t-1)
# ─────────────────────────────────────────────────────────────────────────────
SEQ_LEN = 20
RET_COLS     = [f"RET_{i}"            for i in range(SEQ_LEN, 0, -1)]   # t-20 … t-1
VOL_COLS     = [f"SIGNED_VOLUME_{i}"  for i in range(SEQ_LEN, 0, -1)]
STATIC_COLS  = ["MEDIAN_DAILY_TURNOVER"]
CAT_COL      = "GROUP"


# ─────────────────────────────────────────────────────────────────────────────
#  Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def encode_groups(
    df_train: pd.DataFrame,
    df_test:  Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], int]:
    """
    Integer-encode the GROUP column.
    Returns (df_train_encoded, df_test_encoded, num_groups).
    Unknown groups in test are mapped to a special <UNK> bucket (last index).
    """
    le = LabelEncoder()
    train_groups = df_train[CAT_COL].astype(str).values
    le.fit(train_groups)

    n_known = len(le.classes_)
    # Add 1 for the UNK embedding bucket
    num_groups = n_known + 1
    unk_idx    = n_known

    df_train = df_train.copy()
    df_train[CAT_COL] = le.transform(train_groups)

    if df_test is not None:
        df_test = df_test.copy()
        test_groups = df_test[CAT_COL].astype(str).values
        # map unknown to unk_idx
        encoded = []
        known_set = set(le.classes_)
        for g in test_groups:
            encoded.append(le.transform([g])[0] if g in known_set else unk_idx)
        df_test[CAT_COL] = encoded

    return df_train, df_test, num_groups


def robust_scale_series(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Row-wise robust scaling: subtract median, divide by IQR.
    Applied independently for each sample's time series.
    arr shape: (N, seq_len)
    """
    median = np.nanmedian(arr, axis=1, keepdims=True)
    q75, q25 = np.nanpercentile(arr, [75, 25], axis=1)
    iqr = (q75 - q25)[:, None] + eps
    return (arr - median) / iqr


def preprocess_dataframe(
    df: pd.DataFrame,
    fit_stats: Optional[dict] = None,
    is_train: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Extract and scale all feature arrays from the raw DataFrame.

    Returns
    -------
    ret_seq  : (N, 20)     – scaled return sequences
    vol_seq  : (N, 20)     – scaled signed-volume sequences
    static   : (N, 1)      – scaled MEDIAN_DAILY_TURNOVER
    aux_vol  : (N,)        – realised volatility of past returns (auxiliary target)
    stats    : dict        – scaling parameters (filled only if is_train=True)
    """
    N = len(df)

    # ── 1. Temporal features ──────────────────────────────────────────────
    ret_raw = df[RET_COLS].values.astype(np.float32)   # (N, 20)
    vol_raw = df[VOL_COLS].values.astype(np.float32)   # (N, 20)

    # Fill NaN with 0 (rare missing values)
    ret_raw = np.nan_to_num(ret_raw, nan=0.0)
    vol_raw = np.nan_to_num(vol_raw, nan=0.0)

    # Row-wise robust scaling (each sample independently → avoids look-ahead)
    ret_seq = robust_scale_series(ret_raw)
    vol_seq = robust_scale_series(vol_raw)

    # ── 2. Static feature ─────────────────────────────────────────────────
    turnover = df[STATIC_COLS].values.astype(np.float32)   # (N, 1)
    turnover = np.nan_to_num(turnover, nan=0.0)

    if is_train:
        to_mean = float(np.mean(turnover))
        to_std  = float(np.std(turnover) + 1e-8)
        stats = {"to_mean": to_mean, "to_std": to_std}
    else:
        assert fit_stats is not None, "fit_stats required for test/val preprocessing"
        to_mean = fit_stats["to_mean"]
        to_std  = fit_stats["to_std"]
        stats   = fit_stats

    static = (turnover - to_mean) / to_std   # (N, 1)

    # ── 3. Auxiliary target: realised volatility of past 20 returns ───────
    # log-scale to reduce skew; used as a regression head side-task
    aux_vol = np.std(ret_raw, axis=1).astype(np.float32)   # (N,)

    return ret_seq, vol_seq, static, aux_vol, stats


# ─────────────────────────────────────────────────────────────────────────────
#  PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class QRTDataset(Dataset):
    """
    Dataset for the QRT directional-prediction challenge.

    Parameters
    ----------
    df_X       : pd.DataFrame – feature dataframe (already group-encoded)
    df_y       : pd.Series / None – raw future returns (None → inference mode)
    fit_stats  : dict / None – scaling stats from training set
    is_train   : bool – if True, computes fit_stats from df_X
    """

    def __init__(
        self,
        df_X:      pd.DataFrame,
        df_y:      Optional[pd.Series] = None,
        fit_stats: Optional[dict]      = None,
        is_train:  bool                = True,
    ):
        super().__init__()

        self.inference_mode = (df_y is None)

        # ── Preprocess features ───────────────────────────────────────────
        ret_seq, vol_seq, static, aux_vol, self.fit_stats = preprocess_dataframe(
            df_X, fit_stats=fit_stats, is_train=is_train
        )

        # Stack temporal features: (N, 20, 2)
        self.x_seq    = np.stack([ret_seq, vol_seq], axis=-1).astype(np.float32)
        self.x_static = static.astype(np.float32)          # (N, 1)
        self.x_cat    = df_X[CAT_COL].values.astype(np.int64)  # (N,)
        self.aux_vol  = aux_vol                             # (N,)

        # ── Labels ────────────────────────────────────────────────────────
        if not self.inference_mode:
            # Binary label: 1 if future return > 0 else 0
            self.labels = (df_y.values > 0).astype(np.int64)  # (N,)
        else:
            self.labels = np.zeros(len(df_X), dtype=np.int64)  # placeholder

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.x_seq)

    def __getitem__(self, idx: int) -> dict:
        return {
            "x_seq":    torch.from_numpy(self.x_seq[idx]),          # (20, 2)
            "x_static": torch.from_numpy(self.x_static[idx]),       # (1,)
            "x_cat":    torch.tensor(self.x_cat[idx], dtype=torch.long),
            "label":    torch.tensor(self.labels[idx], dtype=torch.long),
            "aux_tgt":  torch.tensor(self.aux_vol[idx], dtype=torch.float32),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def make_loaders(
    df_X_train: pd.DataFrame,
    df_y_train: pd.Series,
    df_X_val:   pd.DataFrame,
    df_y_val:   pd.Series,
    batch_size: int  = 2048,
    num_workers: int = 4,
    val_ratio:  float = 0.15,     # kept for signature parity; split done upstream
) -> Tuple[DataLoader, DataLoader, dict, int]:
    """
    Build train & validation DataLoaders.
    Returns (train_loader, val_loader, fit_stats, num_groups).

    NOTE: Group encoding must be done BEFORE calling this function.
          Use encode_groups() on the full dataset first.
    """
    train_ds = QRTDataset(df_X_train, df_y_train, is_train=True)
    fit_stats  = train_ds.fit_stats

    val_ds   = QRTDataset(df_X_val,   df_y_val,   fit_stats=fit_stats, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,          # ensures stable batch norms
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    # Infer num_groups from categorical column (assumes already encoded)
    num_groups = int(df_X_train[CAT_COL].max()) + 2  # +2: UNK bucket

    return train_loader, val_loader, fit_stats, num_groups
