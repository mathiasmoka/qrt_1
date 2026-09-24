"""
model.py
========
QuantFormer – A production-grade Deep Learning architecture for directional
prediction of financial time series.

Architecture overview
─────────────────────
                   ┌──────────────────────────────────────────────┐
   x_seq (B,20,2)  │   TCN Temporal Encoder                       │
   ─────────────── │   Dilated causal convolutions + LayerNorm     │─→ h_tcn (B, d_model)
                   └──────────────────────────────────────────────┘
                              ↓
                   ┌──────────────────────────────────────────────┐
                   │   Intra-Sequence Self-Attention               │
                   │   (Multi-Head, causal mask optional)          │─→ h_attn (B, d_model)
                   └──────────────────────────────────────────────┘
                              ↓
   x_cat  (B,)  ──→ Embedding (B, d_embed)  ──┐
   x_static (B,1) ─→ MLP       (B, d_static) ─┤
                                               │
                   ┌───────────────────────────▼─────────────────┐
                   │   Cross-Modal Fusion (Gated Attention Pooling)│─→ h_fused (B, d_model)
                   └──────────────────────────────────────────────┘
                              ↓
              ┌───────────────┴──────────────┐
              │                              │
     ┌────────▼────────┐           ┌─────────▼────────┐
     │  Classifier Head│           │ Volatility Head  │
     │  (Focal Loss)   │           │ (MSE aux loss)   │
     └─────────────────┘           └──────────────────┘
         logits (B,2)                  vol_pred (B,)

Mathematical Motivation
-----------------------
1. TCN: receptive field = 2^(n_layers) * kernel_size covers all 20 lags with
   exponential dilation. Translation-equivariant → robust to phase shifts in
   signals. Causal: no look-ahead bias.

2. Self-Attention on temporal dim: learns which of the 20 days are most
   informative (momentum vs mean-reversion regimes). Positional Encoding
   prevents the permutation-invariance problem.

3. Entity Embedding for GROUP: maps categorical allocations to a dense manifold,
   allowing the model to learn shared structure between related groups. This is
   strictly superior to one-hot for high-cardinality categoricals.

4. Gated Fusion: a learned sigmoid gate controls how much each modality
   (temporal, static, categorical) contributes. Prevents any single modality
   from dominating in a noisy signal regime.

5. Auxiliary Volatility Task: predicting σ(past returns) shares a gradient
   path with the main binary task (volatility ↔ uncertainty ↔ directionality).
   Acts as a learned regulariser, preventing overfitting on directional noise.

6. Focal Loss: down-weights easy-to-classify samples (balanced classes near 50%)
   and focuses gradient on hard samples. α parameter re-weights class imbalance.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Temporal Convolutional Network (TCN) building blocks
# ─────────────────────────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """
    Causal dilated convolution: pads only on the left so that output[t]
    depends only on input[..., :t].

    Math: y[t] = Σ_k w[k] · x[t - d·k]   where d = dilation
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        dilation:     int = 1,
    ):
        super().__init__()
        # Left-pad to preserve causal constraint
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        x = F.pad(x, (self.pad, 0))   # causal left-pad
        return self.conv(x)


class TCNBlock(nn.Module):
    """
    One residual TCN block:
        x → CausalConv → LayerNorm → GELU → Dropout
          → CausalConv → LayerNorm → GELU → Dropout
          → + residual (1×1 conv if dim changes)

    Uses LayerNorm (over channel dim) rather than BatchNorm to be robust with
    small financial batches and variable sequence lengths.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int   = 3,
        dilation:     int   = 1,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels,  out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)

        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.GELU()

        # Projection shortcut if channel dimensions differ
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, L)
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)   # LayerNorm on C
        out = self.act(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = self.act(out)
        out = self.drop(out)

        return out + residual


class TCNEncoder(nn.Module):
    """
    Stack of TCN blocks with exponentially increasing dilation.

    Receptive field RF = 1 + 2 * (kernel_size - 1) * Σ_{i=0}^{n-1} 2^i
                       = 1 + 2 * (kernel_size - 1) * (2^n - 1)

    For kernel_size=3, n=4 layers: RF = 1 + 4*(15) = 61 >> 20 lags.
    Every past lag is in the receptive field of the final time step.
    """

    def __init__(
        self,
        input_size:  int,        # n_temporal_feats (2)
        d_model:     int = 64,
        n_layers:    int = 4,
        kernel_size: int = 3,
        dropout:     float = 0.1,
    ):
        super().__init__()
        layers = []
        in_ch = input_size
        for i in range(n_layers):
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, d_model, kernel_size, dilation, dropout))
            in_ch = d_model
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, seq_len, n_feats)
        out : (B, d_model)   – representation at the LAST time step (most recent)
        """
        x = x.permute(0, 2, 1)          # (B, n_feats, seq_len) for Conv1d
        x = self.net(x)                  # (B, d_model, seq_len)
        return x[:, :, -1]              # take t-1 (most recent) → (B, d_model)


# ─────────────────────────────────────────────────────────────────────────────
#  Intra-Sequence Self-Attention with Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embeddings (preferred over sinusoidal for short seq).
    pe(i) ∈ ℝ^d_model, i ∈ {0, ..., seq_len-1}
    """

    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model)
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pe(positions).unsqueeze(0)


class TemporalSelfAttention(nn.Module):
    """
    Multi-head self-attention over the time dimension.
    Learns which timesteps (lags) are most predictive.

    We do NOT apply a causal mask here: at inference time we observe all 20
    past lags simultaneously, so bidirectional attention is valid and more
    powerful.
    """

    def __init__(
        self,
        d_model:   int,
        n_heads:   int   = 4,
        dropout:   float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm  = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

        # Feed-forward sub-layer (standard Transformer FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, d_model) → (B, d_model)"""
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + self.drop(attn_out))

        # FFN with residual
        x = self.norm2(x + self.drop(self.ffn(x)))

        # Temporal pooling: weighted mean over time steps
        # (simple mean; alternatively learn an attention-based pooling)
        return x.mean(dim=1)   # (B, d_model)


class TemporalAttentionEncoder(nn.Module):
    """
    Project temporal features → d_model, add positional encoding,
    then apply self-attention pooling.
    """

    def __init__(
        self,
        input_size: int,
        seq_len:    int   = 20,
        d_model:    int   = 64,
        n_heads:    int   = 4,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = LearnedPositionalEncoding(seq_len, d_model)
        self.attn       = TemporalSelfAttention(d_model, n_heads, dropout)
        self.norm       = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, n_feats) → (B, d_model)"""
        x = self.input_proj(x)    # (B, 20, d_model)
        x = self.pos_enc(x)       # add positional info
        x = self.norm(x)
        return self.attn(x)       # (B, d_model)


# ─────────────────────────────────────────────────────────────────────────────
#  Dual-Path Temporal Encoder (TCN + Attention in parallel, then fused)
# ─────────────────────────────────────────────────────────────────────────────

class DualPathTemporalEncoder(nn.Module):
    """
    Runs TCN (local patterns) and Attention (global patterns) in PARALLEL,
    then fuses with a learned gate:

        g = sigmoid(W · [h_tcn, h_attn])
        h = g * h_tcn + (1-g) * h_attn

    This is strictly more expressive than either alone.
    """

    def __init__(
        self,
        input_size:   int,
        seq_len:      int   = 20,
        d_model:      int   = 64,
        tcn_layers:   int   = 4,
        n_heads:      int   = 4,
        kernel_size:  int   = 3,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.tcn  = TCNEncoder(input_size, d_model, tcn_layers, kernel_size, dropout)
        self.attn = TemporalAttentionEncoder(input_size, seq_len, d_model, n_heads, dropout)

        # Gating network: computes how much each path contributes
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 20, 2) → (B, d_model)"""
        h_tcn  = self.tcn(x)                   # (B, d_model)
        h_attn = self.attn(x)                   # (B, d_model)

        gate  = self.gate(torch.cat([h_tcn, h_attn], dim=-1))   # (B, d_model)
        fused = gate * h_tcn + (1 - gate) * h_attn              # (B, d_model)
        return self.out_norm(fused)


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-Modal Fusion with Gated Attention
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalFusion(nn.Module):
    """
    Fuses three modalities:
      (1) h_temporal  : (B, d_model)   – from DualPathTemporalEncoder
      (2) h_cat       : (B, d_embed)   – from GROUP embedding
      (3) h_static    : (B, d_static)  – from MEDIAN_DAILY_TURNOVER MLP

    Mechanism: Gated Residual Network (GRN) + cross-attention style
      - Project each modality to a common d_fusion dimension
      - Compute attention scores (each modality attends to the others)
      - Softmax-weighted sum → fused representation

    This is inspired by the Temporal Fusion Transformer (TFT) gating.
    """

    def __init__(
        self,
        d_model:  int,
        d_embed:  int,
        d_static: int,
        d_fusion: int  = 128,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.proj_temporal = nn.Linear(d_model,  d_fusion)
        self.proj_cat      = nn.Linear(d_embed,  d_fusion)
        self.proj_static   = nn.Linear(d_static, d_fusion)

        # Cross-modal attention: each modality computes a query
        self.q_proj = nn.Linear(d_fusion, d_fusion)
        self.k_proj = nn.Linear(d_fusion, d_fusion)

        self.norm    = nn.LayerNorm(d_fusion)
        self.dropout = nn.Dropout(dropout)

        # Gated output layer
        self.gate_proj = nn.Linear(d_fusion, d_fusion)
        self.main_proj = nn.Linear(d_fusion, d_fusion)
        self.out_norm  = nn.LayerNorm(d_fusion)

    def forward(
        self,
        h_temporal: torch.Tensor,
        h_cat:      torch.Tensor,
        h_static:   torch.Tensor,
    ) -> torch.Tensor:
        """→ (B, d_fusion)"""
        # Project all modalities to d_fusion
        p_t = F.gelu(self.proj_temporal(h_temporal))   # (B, d_fusion)
        p_c = F.gelu(self.proj_cat(h_cat))              # (B, d_fusion)
        p_s = F.gelu(self.proj_static(h_static))        # (B, d_fusion)

        # Stack as "token" sequence: (B, 3, d_fusion)
        tokens = torch.stack([p_t, p_c, p_s], dim=1)

        # Self-attention over the 3 modality tokens
        q = self.q_proj(tokens)   # (B, 3, d_fusion)
        k = self.k_proj(tokens)
        scale  = math.sqrt(q.size(-1))
        scores = torch.bmm(q, k.transpose(1, 2)) / scale   # (B, 3, 3)
        weights = torch.softmax(scores, dim=-1)             # (B, 3, 3)
        weights = self.dropout(weights)
        attended = torch.bmm(weights, tokens)               # (B, 3, d_fusion)

        # Residual + norm
        attended = self.norm(attended + tokens)

        # Pool over modality dimension → (B, d_fusion)
        pooled = attended.mean(dim=1)

        # Gated Residual Network output
        gate = torch.sigmoid(self.gate_proj(pooled))
        out  = gate * F.gelu(self.main_proj(pooled)) + (1 - gate) * pooled
        return self.out_norm(out)


# ─────────────────────────────────────────────────────────────────────────────
#  Loss functions
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification (Lin et al., 2017).

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    γ = 2 focuses on hard examples; α corrects class imbalance.
    For balanced classes (50/50) with label smoothing, γ=1–2 works well.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma          = gamma
        self.alpha          = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, 2)
        targets : (B,) long
        """
        # Label smoothing: soft_target = (1-ε)*y + ε/2
        probs = F.softmax(logits, dim=-1)
        B = logits.size(0)

        # Smooth labels
        eps = self.label_smoothing
        smooth_targets = torch.full_like(probs, eps / 2)
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - eps / 2)

        log_probs = torch.log(probs.clamp(min=1e-8))

        # Focal weight: (1 - p_t)^gamma
        p_t = (probs * F.one_hot(targets, 2)).sum(dim=-1, keepdim=True)   # (B, 1)
        focal_weight = (1 - p_t) ** self.gamma                             # (B, 1)

        # Alpha weighting
        alpha_t = torch.where(targets == 1,
                              torch.tensor(self.alpha,       device=logits.device),
                              torch.tensor(1.0 - self.alpha, device=logits.device))

        loss = -(focal_weight.squeeze() * alpha_t * (smooth_targets * log_probs).sum(dim=-1))
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
#  Main Model: QuantFormer
# ─────────────────────────────────────────────────────────────────────────────

class QuantFormer(nn.Module):
    """
    QuantFormer: End-to-end directional prediction model for financial
    panel time-series data.

    Args
    ----
    num_groups       : number of unique GROUP categories (including UNK)
    n_temporal_feats : 2 (RET + SIGNED_VOLUME)
    seq_len          : 20 time steps
    d_model          : hidden dim for temporal encoder
    d_embed          : GROUP embedding dimension
    d_static         : hidden dim for static feature MLP
    d_fusion         : dimension of cross-modal fusion layer
    tcn_layers       : number of TCN blocks
    n_heads          : attention heads
    kernel_size      : TCN kernel size
    dropout          : dropout probability throughout
    """

    def __init__(
        self,
        num_groups:        int,
        n_temporal_feats:  int   = 2,
        seq_len:           int   = 20,
        d_model:           int   = 128,
        d_embed:           int   = 32,
        d_static:          int   = 32,
        d_fusion:          int   = 256,
        tcn_layers:        int   = 4,
        n_heads:           int   = 4,
        kernel_size:       int   = 3,
        dropout:           float = 0.15,
    ):
        super().__init__()

        # ── 1. Temporal Encoder (TCN ‖ Attention) ─────────────────────────
        self.temporal_encoder = DualPathTemporalEncoder(
            input_size  = n_temporal_feats,
            seq_len     = seq_len,
            d_model     = d_model,
            tcn_layers  = tcn_layers,
            n_heads     = n_heads,
            kernel_size = kernel_size,
            dropout     = dropout,
        )

        # ── 2. GROUP Categorical Embedding ────────────────────────────────
        # padding_idx=0 would be wasted; UNK is the LAST index
        self.group_embedding = nn.Embedding(num_groups, d_embed)
        nn.init.normal_(self.group_embedding.weight, std=0.01)

        # ── 3. Static Feature MLP ─────────────────────────────────────────
        self.static_mlp = nn.Sequential(
            nn.Linear(1, d_static),
            nn.GELU(),
            nn.LayerNorm(d_static),
            nn.Dropout(dropout / 2),
            nn.Linear(d_static, d_static),
            nn.GELU(),
        )

        # ── 4. Cross-Modal Fusion ─────────────────────────────────────────
        self.fusion = CrossModalFusion(
            d_model  = d_model,
            d_embed  = d_embed,
            d_static = d_static,
            d_fusion = d_fusion,
            dropout  = dropout,
        )

        # ── 5a. Classification Head ───────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(d_fusion, d_fusion // 2),
            nn.GELU(),
            nn.LayerNorm(d_fusion // 2),
            nn.Dropout(dropout),
            nn.Linear(d_fusion // 2, d_fusion // 4),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_fusion // 4, 2),           # 2 logits: class 0 / class 1
        )

        # ── 5b. Auxiliary Volatility Regression Head ──────────────────────
        # Predicts realised volatility of past returns as a side task.
        # Gradient from this head regularises the temporal encoder.
        self.vol_head = nn.Sequential(
            nn.Linear(d_fusion, d_fusion // 4),
            nn.GELU(),
            nn.Linear(d_fusion // 4, 1),
            nn.Softplus(),                          # σ must be > 0
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform init for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(
        self,
        x_seq:    torch.Tensor,    # (B, 20, 2)
        x_static: torch.Tensor,    # (B, 1)
        x_cat:    torch.Tensor,    # (B,) long
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits   : (B, 2)   – classification logits
        vol_pred : (B,)     – predicted past volatility (auxiliary task)
        """
        # ── Temporal encoding ─────────────────────────────────────────────
        h_temporal = self.temporal_encoder(x_seq)        # (B, d_model)

        # ── Categorical embedding ─────────────────────────────────────────
        h_cat = self.group_embedding(x_cat)              # (B, d_embed)

        # ── Static MLP ────────────────────────────────────────────────────
        h_static = self.static_mlp(x_static)            # (B, d_static)

        # ── Cross-modal fusion ────────────────────────────────────────────
        h_fused = self.fusion(h_temporal, h_cat, h_static)   # (B, d_fusion)

        # ── Output heads ─────────────────────────────────────────────────
        logits   = self.classifier(h_fused)              # (B, 2)
        vol_pred = self.vol_head(h_fused).squeeze(-1)    # (B,)

        return logits, vol_pred

    @torch.no_grad()
    def predict_proba(
        self,
        x_seq:    torch.Tensor,
        x_static: torch.Tensor,
        x_cat:    torch.Tensor,
    ) -> torch.Tensor:
        """Returns P(label=1) – convenience method for inference."""
        logits, _ = self.forward(x_seq, x_static, x_cat)
        return F.softmax(logits, dim=-1)[:, 1]   # (B,)
