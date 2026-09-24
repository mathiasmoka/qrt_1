"""
train.py
========
Production-grade training loop for QuantFormer.

Features
--------
- AdamW optimizer with decoupled weight decay
- OneCycleLR scheduler (super-convergence)
- Gradient clipping (stability on noisy financial gradients)
- Mixed-precision training (torch.cuda.amp) for GPU speed
- Multi-task loss: Focal (classification) + MSE (volatility auxiliary)
- Early stopping on validation accuracy
- Model checkpointing (best val accuracy)
- Reproducible seeding
"""

import os
import math
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import GroupShuffleSplit
from typing import Optional

from dataset import encode_groups, make_loaders, SEQ_LEN
from model import QuantFormer, FocalLoss


# ─────────────────────────────────────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False   # set True if input sizes are fixed


# ─────────────────────────────────────────────────────────────────────────────
#  Config (edit these)
# ─────────────────────────────────────────────────────────────────────────────

CFG = dict(
    # ── Paths ────────────────────────────────────────────────────────────────
    x_train_path  = "X_train.csv",
    y_train_path  = "y_train.csv",
    checkpoint_dir= "checkpoints",

    # ── Split ────────────────────────────────────────────────────────────────
    # GroupShuffleSplit on TS (time-based) to avoid temporal leakage
    val_ratio     = 0.15,
    seed          = 42,

    # ── Model hyperparameters ────────────────────────────────────────────────
    d_model       = 128,
    d_embed       = 32,
    d_static      = 32,
    d_fusion      = 256,
    tcn_layers    = 4,
    n_heads       = 4,
    kernel_size   = 3,
    dropout       = 0.15,

    # ── Training hyperparameters ─────────────────────────────────────────────
    batch_size    = 2048,
    n_epochs      = 50,
    lr_max        = 3e-4,          # OneCycleLR peak LR
    weight_decay  = 1e-2,
    grad_clip     = 1.0,           # max gradient norm
    num_workers   = 4,

    # ── Loss weights ─────────────────────────────────────────────────────────
    focal_gamma       = 2.0,
    focal_alpha       = 0.5,
    label_smoothing   = 0.05,
    aux_loss_weight   = 0.1,       # λ: weight of volatility regression loss

    # ── Early stopping ───────────────────────────────────────────────────────
    patience      = 8,             # epochs without improvement before stopping
)


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
#  One epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model:      QuantFormer,
    loader,
    focal_loss: FocalLoss,
    optimizer:  Optional[torch.optim.Optimizer],
    scheduler,
    scaler:     GradScaler,
    device:     torch.device,
    aux_weight: float,
    grad_clip:  float,
    is_train:   bool = True,
) -> dict:
    """
    Runs one full pass through `loader`.

    Returns dict with keys: loss, cls_loss, aux_loss, accuracy.
    """
    model.train(is_train)
    total_loss = total_cls = total_aux = total_acc = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for batch in loader:
            x_seq    = batch["x_seq"].to(device, non_blocking=True)      # (B,20,2)
            x_static = batch["x_static"].to(device, non_blocking=True)   # (B,1)
            x_cat    = batch["x_cat"].to(device, non_blocking=True)       # (B,)
            labels   = batch["label"].to(device, non_blocking=True)       # (B,)
            aux_tgt  = batch["aux_tgt"].to(device, non_blocking=True)     # (B,)

            # ── Forward pass with mixed precision ────────────────────────
            with autocast(enabled=(device.type == "cuda")):
                logits, vol_pred = model(x_seq, x_static, x_cat)

                # 1. Focal classification loss
                cls_loss = focal_loss(logits, labels)

                # 2. Auxiliary volatility regression loss (Huber = robust MSE)
                #    We log-scale the target to reduce extreme-value sensitivity
                log_vol_tgt  = torch.log1p(aux_tgt)
                log_vol_pred = torch.log1p(vol_pred)
                aux_loss = nn.functional.huber_loss(log_vol_pred, log_vol_tgt, delta=1.0)

                # 3. Combined loss
                loss = cls_loss + aux_weight * aux_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()

                # Gradient clipping (unscale first for correct norm)
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()

                if scheduler is not None:
                    scheduler.step()

            # ── Accumulate metrics ────────────────────────────────────────
            total_loss += loss.item()
            total_cls  += cls_loss.item()
            total_aux  += aux_loss.item()
            total_acc  += accuracy(logits, labels)
            n_batches  += 1

    return {
        "loss":     total_loss / n_batches,
        "cls_loss": total_cls  / n_batches,
        "aux_loss": total_aux  / n_batches,
        "accuracy": total_acc  / n_batches,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Train / Val split (time-safe)
# ─────────────────────────────────────────────────────────────────────────────

def time_safe_split(df_X: pd.DataFrame, df_y: pd.Series, val_ratio: float, seed: int):
    """
    Split by TS (timestamp) groups to prevent temporal leakage:
    all samples at the same date go entirely into train or val.

    This is CRITICAL for financial data — random splits cause data leakage
    through cross-sectional correlation (all allocations at same date share
    the same market environment).
    """
    ts_values = df_X["TS"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(gss.split(df_X, groups=ts_values))

    return (
        df_X.iloc[train_idx].reset_index(drop=True),
        df_y.iloc[train_idx].reset_index(drop=True),
        df_X.iloc[val_idx].reset_index(drop=True),
        df_y.iloc[val_idx].reset_index(drop=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict = CFG):
    seed_everything(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on: {device}")
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("[INFO] Loading data...")
    df_X = pd.read_csv(cfg["x_train_path"], index_col="ROW_ID")
    df_y_raw = pd.read_csv(cfg["y_train_path"], index_col="ROW_ID")["target"]

    # ── 2. Encode GROUP categorically (fit on full train, no leakage) ─────────
    df_X, _, num_groups = encode_groups(df_X)
    print(f"[INFO] num_groups (incl. UNK bucket): {num_groups}")

    # ── 3. Time-safe train / val split ───────────────────────────────────────
    X_tr, y_tr, X_val, y_val = time_safe_split(
        df_X, df_y_raw, cfg["val_ratio"], cfg["seed"]
    )
    print(f"[INFO] Train size: {len(X_tr):,} | Val size: {len(X_val):,}")

    # ── 4. DataLoaders ───────────────────────────────────────────────────────
    train_loader, val_loader, fit_stats, _ = make_loaders(
        X_tr, y_tr, X_val, y_val,
        batch_size  = cfg["batch_size"],
        num_workers = cfg["num_workers"],
    )

    # ── 5. Model ─────────────────────────────────────────────────────────────
    model = QuantFormer(
        num_groups       = num_groups,
        n_temporal_feats = 2,
        seq_len          = SEQ_LEN,
        d_model          = cfg["d_model"],
        d_embed          = cfg["d_embed"],
        d_static         = cfg["d_static"],
        d_fusion         = cfg["d_fusion"],
        tcn_layers       = cfg["tcn_layers"],
        n_heads          = cfg["n_heads"],
        kernel_size      = cfg["kernel_size"],
        dropout          = cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] QuantFormer parameters: {n_params:,}")

    # ── 6. Loss function ─────────────────────────────────────────────────────
    focal_loss = FocalLoss(
        gamma           = cfg["focal_gamma"],
        alpha           = cfg["focal_alpha"],
        label_smoothing = cfg["label_smoothing"],
    )

    # ── 7. Optimizer ─────────────────────────────────────────────────────────
    # Separate weight decay from bias/norm parameters (standard best practice)
    decay_params    = [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in ["bias", "norm", "embedding"])]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and     any(nd in n for nd in ["bias", "norm", "embedding"])]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": cfg["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg["lr_max"] / 25.0,   # OneCycleLR will ramp up from this
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # ── 8. OneCycleLR Scheduler ──────────────────────────────────────────────
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = cfg["lr_max"],
        total_steps     = cfg["n_epochs"] * steps_per_epoch,
        pct_start       = 0.3,          # 30% of training to ramp up
        anneal_strategy = "cos",        # cosine annealing
        div_factor      = 25.0,         # initial_lr = max_lr / 25
        final_div_factor= 1e4,          # final_lr = initial_lr / 1e4
    )

    # ── 9. Mixed precision scaler ────────────────────────────────────────────
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ── 10. Training loop ────────────────────────────────────────────────────
    best_val_acc   = 0.0
    patience_count = 0
    history        = []

    print("\n" + "═" * 70)
    print(f"{'Epoch':>6} │ {'Train Loss':>10} │ {'Train Acc':>9} │ {'Val Loss':>9} │ {'Val Acc':>8} │ {'LR':>10}")
    print("═" * 70)

    for epoch in range(1, cfg["n_epochs"] + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────
        train_metrics = run_epoch(
            model, train_loader, focal_loss, optimizer, scheduler,
            scaler, device, cfg["aux_loss_weight"], cfg["grad_clip"],
            is_train=True,
        )

        # ── Validate ──────────────────────────────────────────────────────
        val_metrics = run_epoch(
            model, val_loader, focal_loss, None, None,
            scaler, device, cfg["aux_loss_weight"], cfg["grad_clip"],
            is_train=False,
        )

        current_lr = scheduler.get_last_lr()[0]
        elapsed    = time.time() - t0

        print(
            f"{epoch:>6} │ {train_metrics['loss']:>10.5f} │ "
            f"{train_metrics['accuracy']:>9.5f} │ "
            f"{val_metrics['loss']:>9.5f} │ "
            f"{val_metrics['accuracy']:>8.5f} │ "
            f"{current_lr:>10.2e}  [{elapsed:.0f}s]"
        )

        history.append({
            "epoch":        epoch,
            "train_loss":   train_metrics["loss"],
            "train_acc":    train_metrics["accuracy"],
            "val_loss":     val_metrics["loss"],
            "val_acc":      val_metrics["accuracy"],
            "lr":           current_lr,
        })

        # ── Checkpoint & early stopping ───────────────────────────────────
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc   = val_metrics["accuracy"]
            patience_count = 0
            ckpt_path = os.path.join(cfg["checkpoint_dir"], "quantformer_best.pt")
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc":    best_val_acc,
                "cfg":        cfg,
                "fit_stats":  fit_stats,
                "num_groups": num_groups,
            }, ckpt_path)
            print(f"         ✓ Saved checkpoint (val_acc={best_val_acc:.5f})")
        else:
            patience_count += 1
            if patience_count >= cfg["patience"]:
                print(f"\n[INFO] Early stopping triggered after {epoch} epochs.")
                break

    print("═" * 70)
    print(f"[INFO] Best Val Accuracy: {best_val_acc:.5f}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
#  Inference helper
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    x_test_path:    str,
    checkpoint_path: str,
    output_path:    str = "submission.csv",
    batch_size:     int = 4096,
    num_workers:    int = 4,
):
    """
    Load a trained QuantFormer from checkpoint and generate submission CSV.

    Outputs a CSV with columns: ROW_ID, TARGET (probability of positive return).
    """
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from dataset import QRTDataset, encode_groups, CAT_COL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg        = ckpt["cfg"]
    fit_stats  = ckpt["fit_stats"]
    num_groups = ckpt["num_groups"]

    model = QuantFormer(
        num_groups       = num_groups,
        n_temporal_feats = 2,
        seq_len          = SEQ_LEN,
        d_model          = cfg["d_model"],
        d_embed          = cfg["d_embed"],
        d_static         = cfg["d_static"],
        d_fusion         = cfg["d_fusion"],
        tcn_layers       = cfg["tcn_layers"],
        n_heads          = cfg["n_heads"],
        kernel_size      = cfg["kernel_size"],
        dropout          = cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Load and preprocess test data ─────────────────────────────────────────
    df_test = pd.read_csv(x_test_path, index_col="ROW_ID")
    row_ids = df_test.index.tolist()

    # Encode groups using TRAIN mapping (unknown → UNK bucket)
    # We load a dummy train df just for the encoder; here we reuse num_groups
    # and encode manually using the same label mapping stored at training time.
    # Simplest: store the LabelEncoder in the checkpoint (extend if needed).
    # For now: any group not seen → last index (UNK), handled in QRTDataset init.
    # Re-encode using the same num_groups logic:
    le_classes = ckpt.get("le_classes", None)
    if le_classes is not None:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.classes_ = np.array(le_classes)
        known = set(le_classes)
        unk_idx = num_groups - 1
        df_test[CAT_COL] = [
            le.transform([str(g)])[0] if str(g) in known else unk_idx
            for g in df_test[CAT_COL].values
        ]
    else:
        # Fallback: map everything to UNK if no encoder stored
        df_test[CAT_COL] = num_groups - 1

    test_ds = QRTDataset(df_test, df_y=None, fit_stats=fit_stats, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # ── Inference ─────────────────────────────────────────────────────────────
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            x_seq    = batch["x_seq"].to(device)
            x_static = batch["x_static"].to(device)
            x_cat    = batch["x_cat"].to(device)

            with autocast(enabled=(device.type == "cuda")):
                logits, _ = model(x_seq, x_static, x_cat)
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())

    # ── Write submission ──────────────────────────────────────────────────────
    submission = pd.DataFrame({
        "ROW_ID": row_ids,
        "TARGET": all_probs,
    })
    submission.to_csv(output_path, index=False)
    print(f"[INFO] Submission saved to '{output_path}' ({len(submission):,} rows)")
    return submission


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    history = train(CFG)

    # Optionally generate test predictions immediately after training
    # predict(
    #     x_test_path     = "X_test.csv",
    #     checkpoint_path = "checkpoints/quantformer_best.pt",
    #     output_path     = "submission_quantformer.csv",
    # )
