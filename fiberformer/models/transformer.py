"""
FiberFormer: footprint-sequence transformer over FiberHMM-called reads.

Key design choices (toggleable via init args):
  - `layer_kind='conformer'`: depthwise 1D conv (k=5) between attention and
    FFN. Captures nucleosome array phasing (177bp repeats) natively at
    token level -- no need for bp-resolution rasterization.
  - `positional='bp_sin_cos'`: continuous base-pair SinCos PE added to
    token embeddings, using per-token integer bp offset. Replaces token-
    index RoPE. Standard 10000 base spectrum gives 62.8kb lowest
    wavelength (fits 60kb read) and 6.3bp highest (nucleosome-resolution).
  - `pool='length_weighted' / 'length_weighted_sqrt'`: weight tokens by
    their physical bp size before mean-pool. Prevents unweighted mean from
    diluting rare massive protamine tokens. Mask-aware during training to
    prevent leaking reconstruction targets via pool weights.

Legacy paths preserved for ablation (`layer_kind='rope'`,
`positional='rope'`, `pool='cls'|'mean'`). Head A (cosine classifier) and
Head B (span-MAE) are pretraining objectives.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fiberformer.data.size_quantizer import SIZE_QUANTIZER, GAP_QUANTIZER


def apply_rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Apply Rotary Position Embeddings to Q or K.

    Args:
        x: (batch, nhead, seq_len, head_dim)
        positions: (batch, seq_len) int — token indices
    Returns:
        Rotated x with same shape.
    """
    batch, nhead, seq_len, head_dim = x.shape
    half = head_dim // 2

    freqs = 1.0 / (10000.0 ** (torch.arange(0, half, device=x.device).float() / half))
    angles = positions.unsqueeze(-1).float() * freqs.unsqueeze(0)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    x1 = x[..., :half]
    x2 = x[..., half:]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out


def sin_cos_bp_pe(bp_positions: torch.Tensor, d_model: int) -> torch.Tensor:
    """Continuous base-pair Sine/Cosine positional encoding (v7).

    Args:
        bp_positions: (B, S) int32 — per-token bp offset relative to read start.
                      CLS should be 0. Padding positions should be 0.
        d_model: embedding dim (must be even).

    Returns:
        PE: (B, S, d_model) float32.

    No position scaling. Lowest frequency wavelength 10000 * 2π ≈ 62.8 kb
    fits a 60kb read in one oscillation; highest ≈ 6.3 bp gives nucleosome
    resolution. Scaling pos/C would compress the finest wavelength to C*6.3
    bp — e.g., pos/200 → 1.2 kb minimum resolution, blinding the model.
    """
    half = d_model // 2
    freqs = 1.0 / (10000.0 ** (torch.arange(0, half, device=bp_positions.device).float() / half))
    # angles: (B, S, half)
    angles = bp_positions.unsqueeze(-1).float() * freqs
    pe = torch.zeros(*bp_positions.shape, d_model, device=bp_positions.device, dtype=torch.float32)
    pe[..., 0::2] = torch.sin(angles)
    pe[..., 1::2] = torch.cos(angles)
    return pe


class LogDistanceBias(nn.Module):
    """Tiny MLP mapping log10(D_ij+1) to per-head scalar bias."""

    def __init__(self, nhead: int, hidden: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, nhead),
        )

    def forward(self, log_dist: torch.Tensor) -> torch.Tensor:
        bias = self.mlp(log_dist.unsqueeze(-1))
        return bias.permute(0, 3, 1, 2)


class RoPETransformerLayer(nn.Module):
    """Single Transformer encoder layer with RoPE and distance bias.

    v7 adds optional `conv_kernel` parameter. When >0, inserts a masked
    depthwise 1D conv between attention residual and FFN (Conformer-style).
    When 0 (default), behaves identically to v5/v6.
    """

    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1,
                 conv_kernel: int = 0, use_bp_pe: bool = False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0
        self.conv_kernel = conv_kernel
        self.use_bp_pe = use_bp_pe  # when True, skip RoPE (bp-PE added elsewhere)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if conv_kernel > 0:
            assert conv_kernel % 2 == 1, "conv_kernel must be odd"
            self.conv_norm = nn.LayerNorm(d_model)
            self.depthwise_conv = nn.Conv1d(
                d_model, d_model,
                kernel_size=conv_kernel,
                padding=conv_kernel // 2,
                groups=d_model,
                bias=False,
            )
            # Dirac initialization: the center tap = 1, others = 0.
            # Makes the conv a no-op at init — stable warmup.
            with torch.no_grad():
                self.depthwise_conv.weight.zero_()
                self.depthwise_conv.weight[:, :, conv_kernel // 2] = 1.0
            self.conv_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        dist_bias: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, S, D = x.shape

        x_norm = self.norm1(x)
        qkv = self.qkv_proj(x_norm).reshape(B, S, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # v7 bp-PE path skips RoPE — positional info is already baked into
        # token embeddings via sin_cos_bp_pe(). RoPE + bp-PE would be
        # duplicative.
        if not self.use_bp_pe:
            q = apply_rope(q, positions)
            k = apply_rope(k, positions)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = attn + dist_bias

        key_mask = ~attn_mask
        attn = attn.masked_fill(key_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, S, D)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        x = x + out  # attention residual

        # --- v7 depthwise conv (Conformer block) ---
        if self.conv_kernel > 0:
            # Mask-aware conv: zero both BEFORE and AFTER to kill
            # (1) padding bleed from kernel pulling in valid neighbors
            # (2) valid-position corruption from kernel pulling in zeroed padding
            mask_expanded = attn_mask.float().unsqueeze(-1)  # (B, S, 1)
            c = self.conv_norm(x)
            c = c * mask_expanded                            # zero BEFORE
            c = c.transpose(1, 2)                             # (B, D, S)
            c = self.depthwise_conv(c)
            c = c.transpose(1, 2)                             # (B, S, D)
            c = F.gelu(c)
            c = c * mask_expanded                            # zero AFTER
            c = self.conv_dropout(c)
            x = x + c  # conv residual

        x = x + self.ffn(self.norm2(x))
        return x


class FootprintTransformer(nn.Module):
    """Foundation-model Footprint Transformer: span-MAE + cosine anchor classifier."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        n_type_ids: int = 5,        # v1-v7: CLS, PROT, NUC, SUB, MASK. v8: 7 (adds LACUNA + shifts MASK)
        n_continuous: int = 5,       # log_size, gap_to_next, is_first, is_last, masked_span
        n_size_bins: int = None,
        n_gap_bins: int = None,
        n_head_b_type_classes: int = 3,  # v1-v7: 3 (PROT, NUC, SUB). v8: 4 (+LACUNA)
        logit_scale_init: float = 10.0,
        pool: str = 'cls',           # 'cls' (v1-v5), 'mean' (v6), 'length_weighted' (v7)
        conv_kernel: int = 0,        # v7: >0 enables Conformer depthwise conv (e.g. 5)
        positional: str = 'rope',    # 'rope' (v1-v6) or 'bp_sin_cos' (v7)
    ):
        super().__init__()
        self.d_model = d_model
        self.pool = pool
        self.positional = positional

        if n_size_bins is None:
            n_size_bins = SIZE_QUANTIZER.num_bins
        if n_gap_bins is None:
            n_gap_bins = GAP_QUANTIZER.num_bins

        # --- Token Embedding (type + continuous → d_model) ---
        embed_half = d_model // 2
        self.type_embed = nn.Embedding(n_type_ids, embed_half)
        self.continuous_proj = nn.Linear(n_continuous, embed_half)
        self.embed_norm = nn.LayerNorm(d_model)

        # --- Positional: distance bias (shared across layers) ---
        self.dist_bias = LogDistanceBias(nhead)

        # --- Encoder ---
        self.layers = nn.ModuleList([
            RoPETransformerLayer(d_model, nhead, d_ff, dropout,
                                  conv_kernel=conv_kernel,
                                  use_bp_pe=(positional == 'bp_sin_cos'))
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # --- Head A: cosine classifier (no bias, will L2-norm its weight at forward) ---
        # Two weight vectors (d_model each): class-0 and class-1 directions on hypersphere.
        # Binary output = (cos(CLS, w1) - cos(CLS, w0)) * temperature.
        self.head_a = nn.Linear(d_model, 2, bias=False)
        # Learnable temperature, stored in log-space for numerical stability
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(float(logit_scale_init))))

        # --- Head B: masked reconstruction ---
        self.head_b_type = nn.Linear(d_model, n_head_b_type_classes)
        self.head_b_size = nn.Linear(d_model, n_size_bins)
        self.head_b_gap = nn.Linear(d_model, n_gap_bins)
        self.n_head_b_type_classes = n_head_b_type_classes

        # Head C is dropped entirely.

    def forward(
        self,
        type_ids: torch.Tensor,       # (B, S) long
        continuous: torch.Tensor,      # (B, S, 5) float
        attn_mask: torch.Tensor,       # (B, S) bool
        log_dist: torch.Tensor,        # (B, S, S) float
        summary: torch.Tensor = None,  # kept for loader-compat; unused
        bp_positions: torch.Tensor = None,   # v7: (B, S) int32 per-token bp offsets
        masked_positions: torch.Tensor = None,  # v7: (B, S) bool — for mask-aware LW pool
    ) -> dict:
        B, S = type_ids.shape

        # Token embedding
        type_emb = self.type_embed(type_ids)
        cont_emb = self.continuous_proj(continuous)
        x = torch.cat([type_emb, cont_emb], dim=-1)  # (B, S, d_model)
        x = self.embed_norm(x)

        # v7: add base-pair SinCos PE once, before layers (replaces RoPE).
        if self.positional == 'bp_sin_cos':
            if bp_positions is None:
                # Fallback: use token indices as bp positions (wrong magnitude
                # but avoids crash for backward-compat caller sites)
                bp_positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
            x = x + sin_cos_bp_pe(bp_positions, self.d_model)

        # Distance bias (shared across layers)
        dist_bias = self.dist_bias(log_dist)

        # Token-index positions for RoPE (unused if positional='bp_sin_cos')
        positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)

        # Encoder
        for layer in self.layers:
            x = layer(x, positions, dist_bias, attn_mask)
        x = self.final_norm(x)

        if self.pool in ('length_weighted', 'length_weighted_sqrt'):
            # Weight tokens by physical bp size. Mask-aware to prevent leak of
            # reconstruction targets through pool weights.
            # continuous[:,:,0] is log_sizes (masked tokens set to 0 by dataset,
            # so 10^0 = 1 bp — effectively zero weight after normalization
            # relative to real tokens).
            #   v7   used 'length_weighted'      → raw bp size weighting.
            #   FiberFormer uses 'length_weighted_sqrt' → sqrt(bp) weighting. Raw size
            #   gave a single 5kb token ~34× the weight of one 147bp
            #   nucleosome, creating length-correlated variance in the pooled
            #   norm. sqrt(size) preserves monotonicity but dampens length
            #   sensitivity ~5.8× in this example.
            size_bp = torch.pow(10.0, continuous[:, :, 0])  # (B, S)
            pool_mask = attn_mask.bool().clone()
            if self.training and masked_positions is not None:
                pool_mask = pool_mask & ~masked_positions.bool()
            pool_mask[:, 0] = False  # exclude CLS
            if self.pool == 'length_weighted_sqrt':
                # AMP-safe: clamp(min=1.0) because sqrt'(0) = inf would NaN
                # the AMP gradient scaler. 1e-9 underflows in fp16 (smallest
                # fp16 normal ≈ 6.1e-5); use 1.0 — masked sizes are 0 →
                # clamp to 1 → sqrt(1) = 1 → multiplied by pool_mask=0 → 0.
                valid_sizes = torch.sqrt(size_bp.clamp(min=1.0)) * pool_mask.float()
            else:
                # v7 raw-size weighting (backward-compatible for v7 checkpoints)
                valid_sizes = size_bp * pool_mask.float()
            denom = valid_sizes.sum(dim=1, keepdim=True).clamp(min=1e-9)
            w = (valid_sizes / denom).unsqueeze(-1)
            cls_emb = (x * w).sum(dim=1)  # (B, d_model)
        elif self.pool == 'mean':
            # Mean-pool over VALID non-CLS tokens. `attn_mask` is True where
            # tokens are valid (including CLS at position 0); exclude CLS.
            mask = attn_mask.float().clone()
            mask[:, 0] = 0
            mask_u = mask.unsqueeze(-1)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            cls_emb = (x * mask_u).sum(dim=1) / denom  # (B, d_model)
        else:
            cls_emb = x[:, 0, :]  # (B, d_model) — original CLS extraction

        # --- Head A: cosine classifier ---
        # Normalize CLS and classifier weights to the unit hypersphere; scale by learnable temperature.
        cls_norm = F.normalize(cls_emb, p=2, dim=-1)
        w_norm = F.normalize(self.head_a.weight, p=2, dim=-1)  # (2, d_model)
        cos_logits = F.linear(cls_norm, w_norm) * self.logit_scale.exp()  # (B, 2)
        # Convert 2-class cosine logits to a single BCE logit: class-1 vs class-0.
        logit_a = cos_logits[:, 1] - cos_logits[:, 0]

        # --- Head B: per-token masked reconstruction ---
        type_logits = self.head_b_type(x)
        size_logits = self.head_b_size(x)
        gap_logits = self.head_b_gap(x)

        return {
            'logit_a': logit_a,
            'type_logits': type_logits,
            'size_logits': size_logits,
            'gap_logits': gap_logits,
            'cls_emb': cls_emb,       # unnormalized, for downstream flexibility
            'cls_norm': cls_norm,     # normalized to hypersphere (pseudotime/PHATE use this)
            'token_emb': x,
        }

    def compute_head_a_loss(self, logit_a, labels, label_mask):
        """BCE over the `label_mask` subset only. `label_mask` excludes unlabeled
        reads (label=-1) AND chrX/chrY reads (sex-linked sperm biology that
        should not train Head A — still trains Head B)."""
        if label_mask.sum() == 0:
            return torch.tensor(0.0, device=logit_a.device)
        logits = logit_a[label_mask]
        targets = labels[label_mask].float()
        return F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')

    def compute_head_b_loss(self, type_logits, size_logits, gap_logits,
                            target_types, target_sizes, target_gaps, mask):
        """Cross-entropy reconstruction loss at masked positions only."""
        if mask.sum() == 0:
            # FiberFormer: keep Head B parameters in the autograd graph with a zero-
            # weighted term. DDP gradient sync crashes/hangs if any parameter
            # is "unused" in the backward pass (sparse-read guard at n_tokens<3
            # zeros the mask for those reads; rare but possible to get a batch
            # where every read is sparse-guarded). Sum of (logits * 0) keeps
            # all three head sub-modules connected with zero gradient.
            return ((type_logits * 0.0).sum()
                    + (size_logits * 0.0).sum()
                    + (gap_logits  * 0.0).sum())

        # v1-v7: 0=PROT, 1=NUC, 2=SUB. v8: adds 3=LACUNA.
        type_targets = target_types[mask] - 1
        type_targets = torch.clamp(type_targets, 0, self.n_head_b_type_classes - 1)
        loss_type = F.cross_entropy(type_logits[mask], type_targets)

        size_targets = torch.clamp(target_sizes[mask], 0, size_logits.shape[-1] - 1)
        loss_size = F.cross_entropy(size_logits[mask], size_targets)

        gap_targets = torch.clamp(target_gaps[mask], 0, gap_logits.shape[-1] - 1)
        loss_gap = F.cross_entropy(gap_logits[mask], gap_targets)

        return loss_type + loss_size + loss_gap
