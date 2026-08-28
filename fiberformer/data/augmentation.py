"""
On-the-fly augmentations for Transformer training.

All augmentations operate on TokenizedRead and return new TokenizedRead.
Never cached — applied fresh each epoch.

Augmentations:
    - sub_window_crop: Keep 60-90% contiguous token sub-window
    - orientation_flip: Reverse token order (PacBio strand-agnostic)
    - subnucleosomal_dropout: Drop 5-10% of <90bp tokens (FiberHMM noise)
    - protamine_fragmentation: Split large protamine into two (Hia5 noise)
    - create_contrastive_views: Two overlapping views for contrastive learning
"""

import numpy as np
from .tokenizer import (
    TokenizedRead, tokenize_read, classify_footprint,
    CLS, PROTAMINE, SUBNUCLEOSOMAL, PROTAMINE_LOG_CAP,
)
from .size_quantizer import SIZE_QUANTIZER, GAP_QUANTIZER
from .loader import FiberRead


def _rebuild_tokens(
    footprint_starts: np.ndarray,
    footprint_sizes: np.ndarray,
    read_start: int,
    read_end: int,
    read_id: str,
    chrom: str,
    label: int,
    summary_features: np.ndarray,
) -> TokenizedRead:
    """Rebuild a TokenizedRead from raw footprint arrays after augmentation."""
    read = FiberRead(
        chrom=chrom, start=read_start, end=read_end,
        read_id=read_id,
        footprint_starts=footprint_starts,
        footprint_sizes=footprint_sizes,
    )
    n = len(footprint_sizes)
    total = n + 1

    type_ids = np.zeros(total, dtype=np.int8)
    log_sizes = np.zeros(total, dtype=np.float32)
    gap_to_next = np.zeros(total, dtype=np.float32)
    is_first = np.zeros(total, dtype=bool)
    is_last = np.zeros(total, dtype=bool)
    raw_size_bins = np.zeros(total, dtype=np.int16)
    raw_gap_bins = np.zeros(total, dtype=np.int16)

    if n > 0:
        sizes = footprint_sizes
        gaps = read.gap_to_next()

        for i in range(n):
            type_ids[i + 1] = classify_footprint(int(sizes[i]))

        raw_log = np.log10(np.maximum(sizes, 1).astype(np.float32))
        capped = np.where(sizes > 200, np.minimum(raw_log, PROTAMINE_LOG_CAP), raw_log)
        log_sizes[1:] = capped
        gap_to_next[1:] = np.log10(gaps.astype(np.float32) + 1)
        is_first[1] = True
        is_last[-1] = True
        raw_size_bins[1:] = SIZE_QUANTIZER.encode_batch(sizes)
        raw_gap_bins[1:] = GAP_QUANTIZER.encode_batch(gaps)

    return TokenizedRead(
        read_id=read_id, n_tokens=n,
        type_ids=type_ids, log_sizes=log_sizes,
        gap_to_next=gap_to_next, is_first=is_first, is_last=is_last,
        raw_size_bins=raw_size_bins, raw_gap_bins=raw_gap_bins,
        summary_features=summary_features.copy(),
        footprint_starts=footprint_starts.copy(),
        footprint_sizes=footprint_sizes.copy(),
        label=label, chrom=chrom,
        read_start=read_start, read_end=read_end,
    )


def sub_window_crop(tok: TokenizedRead, rng: np.random.Generator,
                    min_frac: float = 0.6, max_frac: float = 0.9) -> TokenizedRead:
    """Keep a contiguous sub-window of 60-90% of tokens."""
    n = tok.n_tokens
    if n <= 2:
        return tok

    frac = rng.uniform(min_frac, max_frac)
    window_size = max(2, int(n * frac))
    max_start = n - window_size
    start = rng.integers(0, max_start + 1)

    new_starts = tok.footprint_starts[start:start + window_size].copy()
    new_sizes = tok.footprint_sizes[start:start + window_size].copy()

    # Adjust start positions relative to new read start
    new_read_start = tok.read_start + int(new_starts[0])
    new_starts = new_starts - new_starts[0]
    new_read_end = new_read_start + int(new_starts[-1] + new_sizes[-1]) + 1

    return _rebuild_tokens(
        new_starts, new_sizes, new_read_start, new_read_end,
        tok.read_id, tok.chrom, tok.label, tok.summary_features,
    )


def orientation_flip(tok: TokenizedRead) -> TokenizedRead:
    """Reverse token order (PacBio is strand-agnostic)."""
    n = tok.n_tokens
    if n <= 1:
        return tok

    sizes_rev = tok.footprint_sizes[::-1].copy()
    # Recompute starts from reversed sizes and gaps
    gaps = tok.footprint_starts[1:] - (tok.footprint_starts[:-1] + tok.footprint_sizes[:-1])
    gaps_rev = gaps[::-1]
    new_starts = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        new_starts[i] = new_starts[i - 1] + sizes_rev[i - 1] + gaps_rev[i - 1]

    return _rebuild_tokens(
        new_starts, sizes_rev, tok.read_start, tok.read_end,
        tok.read_id, tok.chrom, tok.label, tok.summary_features,
    )


def subnucleosomal_dropout(tok: TokenizedRead, rng: np.random.Generator,
                           drop_rate_min: float = 0.05,
                           drop_rate_max: float = 0.10) -> TokenizedRead:
    """Drop 5-10% of subnucleosomal (<90bp) tokens, merge flanking gaps.

    Simulates FiberHMM false-negative missed small footprints from sparse m6A tags.
    """
    n = tok.n_tokens
    if n <= 2:
        return tok

    subnuc_mask = tok.footprint_sizes < 90
    n_subnuc = subnuc_mask.sum()
    if n_subnuc == 0:
        return tok

    drop_rate = rng.uniform(drop_rate_min, drop_rate_max)
    n_drop = max(1, int(n_subnuc * drop_rate))
    subnuc_indices = np.where(subnuc_mask)[0]
    drop_indices = set(rng.choice(subnuc_indices, size=min(n_drop, len(subnuc_indices)),
                                  replace=False))

    keep_mask = np.array([i not in drop_indices for i in range(n)])
    new_starts = tok.footprint_starts[keep_mask].copy()
    new_sizes = tok.footprint_sizes[keep_mask].copy()

    if len(new_sizes) == 0:
        return tok

    return _rebuild_tokens(
        new_starts, new_sizes, tok.read_start, tok.read_end,
        tok.read_id, tok.chrom, tok.label, tok.summary_features,
    )


def protamine_fragmentation(tok: TokenizedRead, rng: np.random.Generator,
                            frag_rate: float = 0.015) -> TokenizedRead:
    """Split 1-2% of large protamine tokens (>1000bp) into two with a small gap.

    Simulates Hia5 processivity noise that artificially fragments large protamine blocks.
    """
    n = tok.n_tokens
    large_prot = np.where(tok.footprint_sizes > 1000)[0]
    if len(large_prot) == 0:
        return tok

    n_frag = max(0, int(len(large_prot) * frag_rate))
    if n_frag == 0 and rng.random() < frag_rate * len(large_prot):
        n_frag = 1
    if n_frag == 0:
        return tok

    frag_indices = set(rng.choice(large_prot, size=min(n_frag, len(large_prot)),
                                  replace=False))

    new_starts = []
    new_sizes = []
    for i in range(n):
        if i in frag_indices:
            size = tok.footprint_sizes[i]
            gap = rng.integers(10, 21)  # 10-20bp gap
            split_point = rng.integers(int(size * 0.3), int(size * 0.7) + 1)
            size1 = split_point
            size2 = size - split_point - gap
            if size2 < 50:
                new_starts.append(tok.footprint_starts[i])
                new_sizes.append(tok.footprint_sizes[i])
                continue
            new_starts.append(tok.footprint_starts[i])
            new_sizes.append(size1)
            new_starts.append(tok.footprint_starts[i] + size1 + gap)
            new_sizes.append(size2)
        else:
            new_starts.append(tok.footprint_starts[i])
            new_sizes.append(tok.footprint_sizes[i])

    return _rebuild_tokens(
        np.array(new_starts, dtype=np.int64),
        np.array(new_sizes, dtype=np.int64),
        tok.read_start, tok.read_end,
        tok.read_id, tok.chrom, tok.label, tok.summary_features,
    )


def augment_read(tok: TokenizedRead, rng: np.random.Generator,
                 do_crop: bool = True, do_flip: bool = True) -> TokenizedRead:
    """Apply standard augmentation pipeline."""
    if do_crop and rng.random() < 0.5:
        tok = sub_window_crop(tok, rng)
    if do_flip and rng.random() < 0.5:
        tok = orientation_flip(tok)
    tok = subnucleosomal_dropout(tok, rng)
    tok = protamine_fragmentation(tok, rng)
    return tok


def create_contrastive_views(
    tok: TokenizedRead, rng: np.random.Generator,
) -> tuple[TokenizedRead, TokenizedRead]:
    """Create two augmented views for contrastive learning.

    View 1: [0%, 75%] of tokens + flip + dropout + fragmentation
    View 2: [25%, 100%] of tokens + flip + dropout + fragmentation
    """
    n = tok.n_tokens
    if n <= 4:
        v1 = augment_read(tok, rng, do_crop=False)
        v2 = augment_read(tok, rng, do_crop=False)
        return v1, v2

    # View 1: first 75%
    end1 = max(2, int(n * 0.75))
    starts1 = tok.footprint_starts[:end1].copy()
    sizes1 = tok.footprint_sizes[:end1].copy()
    re1 = tok.read_start + int(starts1[-1] + sizes1[-1]) + 1
    v1 = _rebuild_tokens(starts1, sizes1, tok.read_start, re1,
                         tok.read_id, tok.chrom, tok.label, tok.summary_features)

    # View 2: last 75%
    start2 = max(0, n - int(n * 0.75))
    starts2 = tok.footprint_starts[start2:].copy()
    sizes2 = tok.footprint_sizes[start2:].copy()
    rs2 = tok.read_start + int(starts2[0])
    starts2 = starts2 - starts2[0]
    re2 = rs2 + int(starts2[-1] + sizes2[-1]) + 1
    v2 = _rebuild_tokens(starts2, sizes2, rs2, re2,
                         tok.read_id, tok.chrom, tok.label, tok.summary_features)

    # Apply augmentations (no crop — already cropped)
    v1 = augment_read(v1, rng, do_crop=False)
    v2 = augment_read(v2, rng, do_crop=False)

    return v1, v2
