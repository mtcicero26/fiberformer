"""
Tokenizer: converts FiberRead -> token sequence for the Transformer.

Each footprint = one token. NO gap tokens. CLS token prepended.

Per-token features:
    type_id: int (0=CLS, 1=PROTAMINE >200bp, 2=NUCLEOSOME 90-200bp, 3=SUBNUCLEOSOME <90bp)
    log_size: float32, log10(size_bp), capped at 3.3 for protamine
    gap_to_next: float32, log10(gap_bp + 1)
    is_first, is_last: bool truncation flags
    raw_size_bin, raw_gap_bin: int16, quantized targets for Head B
"""

import numpy as np
from dataclasses import dataclass
from .loader import FiberRead
from .features import compute_features
from .size_quantizer import SIZE_QUANTIZER, GAP_QUANTIZER

# Type IDs
CLS = 0
PROTAMINE = 1
NUCLEOSOME = 2
SUBNUCLEOSOMAL = 3
LACUNA = 4                 # v8: explicit token for nucleosome-masked gaps ≥ 300bp
MASK = 5                   # v8: shifted from 4 to make room for LACUNA

# Protamine log-size cap: log10(2000) ≈ 3.3
PROTAMINE_LOG_CAP = np.log10(2000)

# v8: lacuna detection threshold. Empirically bimodal split between
# "adjacent protamine trace gap" (<300bp) and real lacunae (300bp-10kb).
LACUNA_MIN_GAP_BP = 300


@dataclass
class TokenizedRead:
    """A tokenized Fiber-seq read ready for the Transformer."""
    read_id: str
    n_tokens: int               # excludes CLS
    type_ids: np.ndarray        # (n_tokens+1,) int8, includes CLS at [0]
    log_sizes: np.ndarray       # (n_tokens+1,) float32
    gap_to_next: np.ndarray     # (n_tokens+1,) float32
    is_first: np.ndarray        # (n_tokens+1,) bool
    is_last: np.ndarray         # (n_tokens+1,) bool
    raw_size_bins: np.ndarray   # (n_tokens+1,) int16
    raw_gap_bins: np.ndarray    # (n_tokens+1,) int16
    summary_features: np.ndarray  # (25,) float32
    # Original arrays for distance matrix and augmentation
    footprint_starts: np.ndarray  # (n_tokens,) int64
    footprint_sizes: np.ndarray   # (n_tokens,) int64
    label: int                    # 1=sperm, 0=somatic, -1=unlabeled
    chrom: str
    read_start: int
    read_end: int


def classify_footprint(size_bp: int) -> int:
    if size_bp > 200:
        return PROTAMINE
    elif size_bp >= 90:
        return NUCLEOSOME
    else:
        return SUBNUCLEOSOMAL


def _insert_lacuna_tokens(read: FiberRead, min_gap: int):
    """v8: build an augmented (starts, sizes, is_lacuna) stream by inserting
    synthetic LACUNA pseudo-footprints wherever the gap between consecutive
    real footprints is ≥ min_gap. LACUNA 'size' is the gap length in bp.

    Returns (new_starts, new_sizes, is_lacuna_bool) — all same length.
    Real footprints keep their original start+size; lacunae are inserted
    between them.
    """
    starts = read.footprint_starts
    sizes = read.footprint_sizes
    n = len(sizes)
    if n == 0:
        return (np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=bool))
    out_starts = []
    out_sizes = []
    out_lac = []
    ends = starts + sizes
    # Leading gap before first footprint
    if starts[0] >= min_gap:
        out_starts.append(0)
        out_sizes.append(int(starts[0]))
        out_lac.append(True)
    for i in range(n):
        out_starts.append(int(starts[i]))
        out_sizes.append(int(sizes[i]))
        out_lac.append(False)
        if i + 1 < n:
            gap = int(starts[i + 1] - ends[i])
            if gap >= min_gap:
                out_starts.append(int(ends[i]))
                out_sizes.append(gap)
                out_lac.append(True)
    # Trailing gap after last footprint (up to read.length)
    if read.length - ends[-1] >= min_gap:
        out_starts.append(int(ends[-1]))
        out_sizes.append(int(read.length - ends[-1]))
        out_lac.append(True)
    return (np.array(out_starts, dtype=np.int64),
            np.array(out_sizes, dtype=np.int64),
            np.array(out_lac, dtype=bool))


def tokenize_read(read: FiberRead, label: int = -1,
                   size_encoding: str = 'absolute',
                   use_lacuna_tokens: bool = False,
                   lacuna_min_gap: int = LACUNA_MIN_GAP_BP) -> TokenizedRead:
    """Convert a FiberRead into a TokenizedRead.

    Prepends CLS token. Computes all per-token features.

    size_encoding:
      'absolute'           : log10(size_bp), capped at 2kb for protamine (v1-v5 default).
      'fractional'         : log10(size / read_length). Read-length-aware, bounded
                             (-∞, 0]. v6 default. Warps physics — a 147bp
                             nucleosome looks different on different-length reads.
      'absolute_uncapped'  : log10(size_bp), uncapped — paired with v7's extended
                             SizeQuantizer (goes up to 60kb). Preserves physical
                             meaning: a nucleosome is 147bp regardless of read
                             context.

    use_lacuna_tokens (v8): when True, insert explicit LACUNA tokens at gaps
      ≥ lacuna_min_gap bp. Each lacuna appears as its own token with
      type_id=LACUNA and log_size=log10(lacuna_bp).
    """
    if use_lacuna_tokens:
        new_starts, new_sizes, is_lacuna = _insert_lacuna_tokens(read, lacuna_min_gap)
        n = len(new_sizes)
    else:
        new_starts = read.footprint_starts
        new_sizes = read.footprint_sizes
        is_lacuna = np.zeros(len(new_sizes), dtype=bool)
        n = read.n_footprints
    total = n + 1  # +1 for CLS

    type_ids = np.zeros(total, dtype=np.int8)
    log_sizes = np.zeros(total, dtype=np.float32)
    gap_to_next_arr = np.zeros(total, dtype=np.float32)
    is_first = np.zeros(total, dtype=bool)
    is_last = np.zeros(total, dtype=bool)
    raw_size_bins = np.zeros(total, dtype=np.int16)
    raw_gap_bins = np.zeros(total, dtype=np.int16)

    # CLS token at position 0
    type_ids[0] = CLS
    # CLS gets zeros for all continuous features

    if n > 0:
        sizes = new_sizes
        # Compute gap from each token to the start of the next token (same
        # semantics as before — just with possibly-inserted lacuna tokens
        # in the stream). Last token gets gap to read end.
        token_ends = new_starts + new_sizes
        gaps = np.zeros(n, dtype=np.int64)
        if n > 1:
            gaps[:-1] = new_starts[1:] - token_ends[:-1]
        gaps[-1] = max(0, read.length - int(token_ends[-1]))
        gaps = np.maximum(gaps, 0)

        # Type IDs — LACUNA if synthetic, else classify by size
        for i in range(n):
            if is_lacuna[i]:
                type_ids[i + 1] = LACUNA
            else:
                type_ids[i + 1] = classify_footprint(int(sizes[i]))

        # Log sizes
        if size_encoding == 'fractional':
            rl = max(read.length, 1)
            size_frac = np.clip(sizes.astype(np.float32) / rl, 1e-5, 1.0)
            log_sizes[1:] = np.log10(size_frac)
        elif size_encoding == 'absolute_uncapped':
            sizes_clip = np.clip(sizes.astype(np.float32), 1, 60_000)
            log_sizes[1:] = np.log10(sizes_clip)
        else:
            raw_log_sizes = np.log10(np.maximum(sizes, 1).astype(np.float32))
            capped_log_sizes = np.where(
                sizes > 200,
                np.minimum(raw_log_sizes, PROTAMINE_LOG_CAP),
                raw_log_sizes,
            )
            log_sizes[1:] = capped_log_sizes

        # Gap to next: log10(gap + 1)
        gap_to_next_arr[1:] = np.log10(gaps.astype(np.float32) + 1)

        # Truncation flags
        is_first[1] = True
        is_last[-1] = True

        # Quantized targets for Head B
        raw_size_bins[1:] = SIZE_QUANTIZER.encode_batch(sizes)
        raw_gap_bins[1:] = GAP_QUANTIZER.encode_batch(gaps)

    # Summary features (25 floats)
    summary = compute_features(read).astype(np.float32)

    return TokenizedRead(
        read_id=read.read_id,
        n_tokens=n,
        type_ids=type_ids,
        log_sizes=log_sizes,
        gap_to_next=gap_to_next_arr,
        is_first=is_first,
        is_last=is_last,
        raw_size_bins=raw_size_bins,
        raw_gap_bins=raw_gap_bins,
        summary_features=summary,
        # Store the (possibly lacuna-augmented) stream so distance matrices
        # downstream see LACUNA positions. Consumers that want the original
        # footprints should pass the FiberRead directly.
        footprint_starts=new_starts.copy().astype(np.int64),
        footprint_sizes=new_sizes.copy().astype(np.int64),
        label=label,
        chrom=read.chrom,
        read_start=read.start,
        read_end=read.end,
    )
