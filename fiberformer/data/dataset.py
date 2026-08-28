"""
HDF5-cached dataset for the Footprint Sequence Transformer.

Builds once from BED files, then loads from cache. Handles:
    - Tokenization and padding
    - Span-hint masking for Head B
    - Corrupted distance matrix for attention bias
    - Contrastive view generation
    - Dynamic padding + attention masks via custom collate_fn
"""

import os
import sys
import math
import hashlib
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from fiberformer.data.loader import parse_bed_file, HUMAN_CHROMS, FiberRead
from fiberformer.data.tokenizer import tokenize_read, TokenizedRead, CLS, MASK
from fiberformer.data.augmentation import augment_read, create_contrastive_views
from fiberformer.data.size_quantizer import SIZE_QUANTIZER, GAP_QUANTIZER
from fiberformer.data.features import compute_features

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BED_DIR = BASE_DIR / 'data' / 'bigbed'


def define_protamine_obligate_regions(
    sperm_filepath: str | Path,
    max_reads: int = 200000,
    min_coverage: int = 3,
    min_protamine_frac: float = 0.95,
    bin_size: int = 50000,
) -> set:
    """Define genomic bins that are >95% protaminated in purified sperm."""
    bin_protamine = {}
    for read in parse_bed_file(sperm_filepath, min_read_length=10000,
                               max_reads=max_reads, valid_chroms=HUMAN_CHROMS):
        key = (read.chrom, read.start // bin_size)
        if key not in bin_protamine:
            bin_protamine[key] = []
        bin_protamine[key].append(read.protamine_fraction())

    obligate = set()
    for key, fracs in bin_protamine.items():
        if len(fracs) >= min_coverage and np.mean(fracs) >= min_protamine_frac:
            obligate.add(key)
    return obligate


def build_hdf5_cache(
    cache_path: str | Path,
    sperm_path: str | Path,
    testes_path: str | Path,
    max_tokens: int = 128,
    n_sperm: int = 50000,
    n_somatic: int = 50000,
    n_unlabeled: int = 100000,
) -> None:
    """Build HDF5 cache from BED files.

    Labels: 1=sperm, 0=somatic (locus-aware zero-protamine), -1=unlabeled testes.
    Split by hash(read_id) % 1000: 0-699=train, 700-849=val, 850-999=test.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print("Building HDF5 cache...")
    print(f"  Sperm: {sperm_path} (n={n_sperm})")
    print(f"  Testes: {testes_path} (somatic={n_somatic}, unlabeled={n_unlabeled})")

    # Define obligate regions for locus-aware negative sampling
    print("  Defining protamine-obligate regions...")
    obligate = define_protamine_obligate_regions(sperm_path, max_reads=200000)
    print(f"    Found {len(obligate)} obligate bins")

    # Collect all reads
    all_tokens = []

    # 1. Sperm reads (label=1)
    print(f"  Loading sperm reads...")
    count = 0
    for read in parse_bed_file(sperm_path, min_read_length=10000,
                               max_reads=n_sperm, valid_chroms=HUMAN_CHROMS):
        tok = tokenize_read(read, label=1)
        all_tokens.append(tok)
        count += 1
    print(f"    Loaded {count} sperm reads")

    # 2. Somatic reads (label=0): zero-protamine, locus-aware
    print(f"  Loading somatic reads (locus-aware)...")
    count = 0
    for read in parse_bed_file(testes_path, min_read_length=10000,
                               max_reads=n_somatic * 5, valid_chroms=HUMAN_CHROMS):
        if read.large_protamine_fraction() > 0:
            continue
        if read.n_footprints < 10:
            continue
        bin_key = (read.chrom, read.start // 50000)
        if bin_key not in obligate:
            continue
        tok = tokenize_read(read, label=0)
        all_tokens.append(tok)
        count += 1
        if count >= n_somatic:
            break
    print(f"    Loaded {count} somatic reads")

    # 3. Unlabeled testes reads (label=-1)
    print(f"  Loading unlabeled testes reads...")
    count = 0
    seen_ids = {t.read_id for t in all_tokens}
    for read in parse_bed_file(testes_path, min_read_length=10000,
                               max_reads=n_unlabeled * 2, valid_chroms=HUMAN_CHROMS):
        if read.read_id in seen_ids:
            continue
        tok = tokenize_read(read, label=-1)
        all_tokens.append(tok)
        seen_ids.add(read.read_id)
        count += 1
        if count >= n_unlabeled:
            break
    print(f"    Loaded {count} unlabeled reads")

    total = len(all_tokens)
    print(f"  Total reads: {total}")

    # Write to HDF5
    seq_len = max_tokens + 1  # +1 for CLS

    with h5py.File(cache_path, 'w') as f:
        # Create datasets with chunking and compression
        dt_int8 = np.int8
        dt_int16 = np.int16
        dt_int32 = np.int32
        dt_int64 = np.int64
        dt_f32 = np.float32
        chunk = min(1000, total)

        ds_type_ids = f.create_dataset('type_ids', shape=(total, seq_len),
                                       dtype=dt_int8, chunks=(chunk, seq_len),
                                       compression='gzip', compression_opts=4)
        ds_log_sizes = f.create_dataset('log_sizes', shape=(total, seq_len),
                                        dtype=dt_f32, chunks=(chunk, seq_len),
                                        compression='gzip', compression_opts=4)
        ds_gap_to_next = f.create_dataset('gap_to_next', shape=(total, seq_len),
                                          dtype=dt_f32, chunks=(chunk, seq_len),
                                          compression='gzip', compression_opts=4)
        ds_is_first = f.create_dataset('is_first', shape=(total, seq_len),
                                       dtype=bool, chunks=(chunk, seq_len),
                                       compression='gzip', compression_opts=4)
        ds_is_last = f.create_dataset('is_last', shape=(total, seq_len),
                                      dtype=bool, chunks=(chunk, seq_len),
                                      compression='gzip', compression_opts=4)
        ds_size_bins = f.create_dataset('raw_size_bins', shape=(total, seq_len),
                                        dtype=dt_int16, chunks=(chunk, seq_len),
                                        compression='gzip', compression_opts=4)
        ds_gap_bins = f.create_dataset('raw_gap_bins', shape=(total, seq_len),
                                       dtype=dt_int16, chunks=(chunk, seq_len),
                                       compression='gzip', compression_opts=4)
        ds_n_tokens = f.create_dataset('n_tokens', shape=(total,),
                                       dtype=dt_int32)
        ds_summary = f.create_dataset('summary_features', shape=(total, 25),
                                      dtype=dt_f32, chunks=(chunk, 25),
                                      compression='gzip', compression_opts=4)
        ds_labels = f.create_dataset('labels', shape=(total,), dtype=dt_int8)

        # Store original starts and sizes for augmentation/distance matrix
        ds_orig_starts = f.create_dataset('orig_starts', shape=(total, max_tokens),
                                          dtype=dt_int64, chunks=(chunk, max_tokens),
                                          compression='gzip', compression_opts=4)
        ds_orig_sizes = f.create_dataset('orig_sizes', shape=(total, max_tokens),
                                         dtype=dt_int64, chunks=(chunk, max_tokens),
                                         compression='gzip', compression_opts=4)

        # Store read IDs and metadata
        dt_str = h5py.string_dtype()
        ds_read_ids = f.create_dataset('read_ids', shape=(total,), dtype=dt_str)
        ds_chroms = f.create_dataset('chroms', shape=(total,), dtype=dt_str)
        ds_read_starts = f.create_dataset('read_starts', shape=(total,), dtype=dt_int64)
        ds_read_ends = f.create_dataset('read_ends', shape=(total,), dtype=dt_int64)

        # Split assignment
        ds_split = f.create_dataset('split', shape=(total,), dtype=dt_int8)

        # Chunked writes
        write_chunk = 5000
        for chunk_start in range(0, total, write_chunk):
            chunk_end = min(chunk_start + write_chunk, total)
            if chunk_start % 20000 == 0:
                print(f"    Writing {chunk_start}/{total}...")

            for i in range(chunk_start, chunk_end):
                tok = all_tokens[i]
                n = min(tok.n_tokens, max_tokens)

                # Pad to seq_len
                type_ids = np.zeros(seq_len, dtype=dt_int8)
                log_sizes = np.zeros(seq_len, dtype=dt_f32)
                gap_to_next = np.zeros(seq_len, dtype=dt_f32)
                is_first = np.zeros(seq_len, dtype=bool)
                is_last = np.zeros(seq_len, dtype=bool)
                size_bins = np.zeros(seq_len, dtype=dt_int16)
                gap_bins = np.zeros(seq_len, dtype=dt_int16)
                orig_starts = np.zeros(max_tokens, dtype=dt_int64)
                orig_sizes = np.zeros(max_tokens, dtype=dt_int64)

                actual_len = n + 1  # CLS + n tokens
                type_ids[:actual_len] = tok.type_ids[:actual_len]
                log_sizes[:actual_len] = tok.log_sizes[:actual_len]
                gap_to_next[:actual_len] = tok.gap_to_next[:actual_len]
                is_first[:actual_len] = tok.is_first[:actual_len]
                is_last[:actual_len] = tok.is_last[:actual_len]
                size_bins[:actual_len] = tok.raw_size_bins[:actual_len]
                gap_bins[:actual_len] = tok.raw_gap_bins[:actual_len]
                orig_starts[:n] = tok.footprint_starts[:n]
                orig_sizes[:n] = tok.footprint_sizes[:n]

                ds_type_ids[i] = type_ids
                ds_log_sizes[i] = log_sizes
                ds_gap_to_next[i] = gap_to_next
                ds_is_first[i] = is_first
                ds_is_last[i] = is_last
                ds_size_bins[i] = size_bins
                ds_gap_bins[i] = gap_bins
                ds_n_tokens[i] = n
                ds_summary[i] = tok.summary_features
                ds_labels[i] = tok.label
                ds_orig_starts[i] = orig_starts
                ds_orig_sizes[i] = orig_sizes
                ds_read_ids[i] = tok.read_id
                ds_chroms[i] = tok.chrom
                ds_read_starts[i] = tok.read_start
                ds_read_ends[i] = tok.read_end

                # Split: hash(read_id) % 1000
                h = int(hashlib.md5(tok.read_id.encode()).hexdigest(), 16) % 1000
                if h < 700:
                    ds_split[i] = 0  # train
                elif h < 850:
                    ds_split[i] = 1  # val
                else:
                    ds_split[i] = 2  # test

    print(f"  Cache saved to {cache_path}")
    print(f"  Size: {cache_path.stat().st_size / 1e6:.1f} MB")


def _sample_reads_round_robin(
    paths,
    n_target,
    filter_fn=None,
    valid_chroms=None,
    min_read_length=10000,
    label=None,
    verbose_label="",
    seed=42,
):
    """Reservoir-sample ~n_target/len(paths) reads from each path.

    BigBed files are chromosome-sorted; naive "take first N" would pick only
    chr1. Instead we reservoir-sample over the first SCAN_CAP reads (after
    filter) so reads are drawn uniformly across the whole genome.
    """
    import random
    rng = random.Random(seed)
    per_sample = max(1, n_target // len(paths))
    # Cap total scan — big enough to span all chromosomes even for huge files,
    # but bounded so we don't walk 100M-row GM12878 end-to-end.
    SCAN_CAP = max(per_sample * 20, 500_000)

    collected = []
    for p in paths:
        reservoir = []      # of TokenizedRead, size ≤ per_sample
        n_seen = 0
        for read in parse_bed_file(p, min_read_length=min_read_length,
                                   max_reads=SCAN_CAP,
                                   valid_chroms=valid_chroms):
            if filter_fn and not filter_fn(read):
                continue
            tok = tokenize_read(read, label=label)
            if len(reservoir) < per_sample:
                reservoir.append(tok)
            else:
                # reservoir sampling: swap into random slot with probability per_sample/(n_seen+1)
                j = rng.randint(0, n_seen)
                if j < per_sample:
                    reservoir[j] = tok
            n_seen += 1

        collected.extend(reservoir)
        # Chromosome coverage sanity — just for logging
        chroms_seen = len(set(t.chrom for t in reservoir))
        print(f"    {verbose_label} {Path(p).name}: {len(reservoir):,}  "
              f"(from {n_seen:,} scanned, {chroms_seen} chroms)")
    return collected


def build_hdf5_cache_v2(
    cache_path,
    sperm_paths,         # list of BED/BigBed paths — purified sperm samples
    testes_paths,        # list of paths — testes (pooled background for locus-aware + unlabeled)
    somatic_paths,       # list of paths — explicit somatic (GM12878, SMaHT)
    max_tokens: int = 128,
    n_sperm: int = 80000,
    n_somatic_locus_aware: int = 40000,   # from testes at obligate loci
    n_somatic_explicit: int = 40000,      # from GM12878/SMaHT
    n_unlabeled: int = 120000,            # from testes
) -> None:
    """Multi-sample cache builder for hybrid negative-class training.

    Labels: 1=sperm, 0=somatic (locus-aware OR explicit), -1=unlabeled testes.
    Split by hash(read_id) % 1000: 0-699=train, 700-849=val, 850-999=test.

    Pools:
      - Sperm pool: pooled across `sperm_paths` (round-robin), full protamination
                    spectrum.
      - Negative pool (hybrid):
          half = locus-aware testes reads (zero-protamine at sperm-obligate bins)
          half = explicit somatic from GM12878 / SMaHT samples
      - Unlabeled testes: pooled across `testes_paths` (round-robin).
    """
    from pathlib import Path as _P
    sperm_paths = [_P(p) for p in sperm_paths]
    testes_paths = [_P(p) for p in testes_paths]
    somatic_paths = [_P(p) for p in somatic_paths]

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print("Building HDF5 cache v2 (multi-sample hybrid negative class)...")
    print(f"  Sperm samples:    {[p.name for p in sperm_paths]}  (target n={n_sperm})")
    print(f"  Testes samples:   {[p.name for p in testes_paths]}  (locus-aware n={n_somatic_locus_aware}, unlabeled n={n_unlabeled})")
    print(f"  Somatic samples:  {[p.name for p in somatic_paths]}  (explicit n={n_somatic_explicit})")

    # --- Define obligate regions from union of sperm samples ---
    print("  Defining protamine-obligate regions (union over sperm pool)...")
    obligate = set()
    per_sperm_obligate = max(50000, 200000 // len(sperm_paths))
    for p in sperm_paths:
        o = define_protamine_obligate_regions(p, max_reads=per_sperm_obligate)
        obligate |= o
        print(f"    {p.name}: contributed {len(o):,} bins (running union: {len(obligate):,})")

    all_tokens = []
    seen_ids = set()

    # --- 1. Sperm reads (label=1) — round-robin across all sperm samples ---
    print(f"  Loading sperm reads (pooled)...")
    sperm_tokens = _sample_reads_round_robin(
        sperm_paths, n_sperm,
        filter_fn=None, valid_chroms=HUMAN_CHROMS,
        label=1, verbose_label="sperm",
    )
    for t in sperm_tokens:
        all_tokens.append(t)
        seen_ids.add(t.read_id)
    print(f"    Total sperm: {len(sperm_tokens):,}")

    # --- 2a. Locus-aware negatives from testes (label=0) ---
    print(f"  Loading locus-aware somatic negatives from testes pool...")
    def _locus_aware_filter(read):
        if read.large_protamine_fraction() > 0:
            return False
        if read.n_footprints < 10:
            return False
        bin_key = (read.chrom, read.start // 50000)
        return bin_key in obligate

    la_tokens = _sample_reads_round_robin(
        testes_paths, n_somatic_locus_aware,
        filter_fn=_locus_aware_filter, valid_chroms=HUMAN_CHROMS,
        label=0, verbose_label="locus-aware",
    )
    for t in la_tokens:
        if t.read_id not in seen_ids:
            all_tokens.append(t)
            seen_ids.add(t.read_id)
    print(f"    Total locus-aware somatic: {len(la_tokens):,}")

    # --- 2b. Explicit somatic (GM12878 + SMaHT) — universal nucleosome grammar ---
    print(f"  Loading explicit somatic (GM12878 + SMaHT)...")
    # Cheap filter: don't keep degenerate reads
    def _explicit_filter(read):
        return read.n_footprints >= 10

    explicit_tokens = _sample_reads_round_robin(
        somatic_paths, n_somatic_explicit,
        filter_fn=_explicit_filter, valid_chroms=HUMAN_CHROMS,
        label=0, verbose_label="explicit-somatic",
    )
    for t in explicit_tokens:
        if t.read_id not in seen_ids:
            all_tokens.append(t)
            seen_ids.add(t.read_id)
    print(f"    Total explicit somatic: {len(explicit_tokens):,}")

    # --- 3. Unlabeled testes (label=-1) — pooled across all testes samples ---
    print(f"  Loading unlabeled testes reads (pooled)...")
    unlabeled_tokens = _sample_reads_round_robin(
        testes_paths, n_unlabeled,
        filter_fn=lambda r: True, valid_chroms=HUMAN_CHROMS,
        label=-1, verbose_label="unlabeled",
    )
    for t in unlabeled_tokens:
        if t.read_id not in seen_ids:
            all_tokens.append(t)
            seen_ids.add(t.read_id)
    print(f"    Total unlabeled: {len(unlabeled_tokens):,}")

    total = len(all_tokens)
    label_counts = {}
    for t in all_tokens:
        label_counts[t.label] = label_counts.get(t.label, 0) + 1
    print(f"  Total reads: {total:,}  (by label: {label_counts})")

    # --- Write to HDF5 (identical schema to build_hdf5_cache) ---
    seq_len = max_tokens + 1

    with h5py.File(cache_path, 'w') as f:
        dt_int8 = np.int8; dt_int16 = np.int16; dt_int32 = np.int32
        dt_int64 = np.int64; dt_f32 = np.float32
        chunk = min(1000, total)

        ds_type_ids = f.create_dataset('type_ids', shape=(total, seq_len),
                                       dtype=dt_int8, chunks=(chunk, seq_len),
                                       compression='gzip', compression_opts=4)
        ds_log_sizes = f.create_dataset('log_sizes', shape=(total, seq_len),
                                        dtype=dt_f32, chunks=(chunk, seq_len),
                                        compression='gzip', compression_opts=4)
        ds_gap_to_next = f.create_dataset('gap_to_next', shape=(total, seq_len),
                                          dtype=dt_f32, chunks=(chunk, seq_len),
                                          compression='gzip', compression_opts=4)
        ds_is_first = f.create_dataset('is_first', shape=(total, seq_len),
                                       dtype=bool, chunks=(chunk, seq_len),
                                       compression='gzip', compression_opts=4)
        ds_is_last = f.create_dataset('is_last', shape=(total, seq_len),
                                      dtype=bool, chunks=(chunk, seq_len),
                                      compression='gzip', compression_opts=4)
        ds_size_bins = f.create_dataset('raw_size_bins', shape=(total, seq_len),
                                        dtype=dt_int16, chunks=(chunk, seq_len),
                                        compression='gzip', compression_opts=4)
        ds_gap_bins = f.create_dataset('raw_gap_bins', shape=(total, seq_len),
                                       dtype=dt_int16, chunks=(chunk, seq_len),
                                       compression='gzip', compression_opts=4)
        ds_n_tokens = f.create_dataset('n_tokens', shape=(total,), dtype=dt_int32)
        ds_summary = f.create_dataset('summary_features', shape=(total, 25),
                                      dtype=dt_f32, chunks=(chunk, 25),
                                      compression='gzip', compression_opts=4)
        ds_labels = f.create_dataset('labels', shape=(total,), dtype=dt_int8)

        ds_orig_starts = f.create_dataset('orig_starts', shape=(total, max_tokens),
                                          dtype=dt_int64, chunks=(chunk, max_tokens),
                                          compression='gzip', compression_opts=4)
        ds_orig_sizes = f.create_dataset('orig_sizes', shape=(total, max_tokens),
                                         dtype=dt_int64, chunks=(chunk, max_tokens),
                                         compression='gzip', compression_opts=4)

        dt_str = h5py.string_dtype()
        ds_read_ids = f.create_dataset('read_ids', shape=(total,), dtype=dt_str)
        ds_chroms = f.create_dataset('chroms', shape=(total,), dtype=dt_str)
        ds_read_starts = f.create_dataset('read_starts', shape=(total,), dtype=dt_int64)
        ds_read_ends = f.create_dataset('read_ends', shape=(total,), dtype=dt_int64)

        ds_split = f.create_dataset('split', shape=(total,), dtype=dt_int8)

        write_chunk = 5000
        for chunk_start in range(0, total, write_chunk):
            chunk_end = min(chunk_start + write_chunk, total)
            if chunk_start % 20000 == 0:
                print(f"    Writing {chunk_start}/{total}...")

            for i in range(chunk_start, chunk_end):
                tok = all_tokens[i]
                n = min(tok.n_tokens, max_tokens)

                type_ids = np.zeros(seq_len, dtype=dt_int8)
                log_sizes = np.zeros(seq_len, dtype=dt_f32)
                gap_to_next = np.zeros(seq_len, dtype=dt_f32)
                is_first = np.zeros(seq_len, dtype=bool)
                is_last = np.zeros(seq_len, dtype=bool)
                size_bins = np.zeros(seq_len, dtype=dt_int16)
                gap_bins = np.zeros(seq_len, dtype=dt_int16)
                orig_starts = np.zeros(max_tokens, dtype=dt_int64)
                orig_sizes = np.zeros(max_tokens, dtype=dt_int64)

                actual_len = n + 1
                type_ids[:actual_len] = tok.type_ids[:actual_len]
                log_sizes[:actual_len] = tok.log_sizes[:actual_len]
                gap_to_next[:actual_len] = tok.gap_to_next[:actual_len]
                is_first[:actual_len] = tok.is_first[:actual_len]
                is_last[:actual_len] = tok.is_last[:actual_len]
                size_bins[:actual_len] = tok.raw_size_bins[:actual_len]
                gap_bins[:actual_len] = tok.raw_gap_bins[:actual_len]
                orig_starts[:n] = tok.footprint_starts[:n]
                orig_sizes[:n] = tok.footprint_sizes[:n]

                ds_type_ids[i] = type_ids
                ds_log_sizes[i] = log_sizes
                ds_gap_to_next[i] = gap_to_next
                ds_is_first[i] = is_first
                ds_is_last[i] = is_last
                ds_size_bins[i] = size_bins
                ds_gap_bins[i] = gap_bins
                ds_n_tokens[i] = n
                ds_summary[i] = tok.summary_features
                ds_labels[i] = tok.label
                ds_orig_starts[i] = orig_starts
                ds_orig_sizes[i] = orig_sizes
                ds_read_ids[i] = tok.read_id
                ds_chroms[i] = tok.chrom
                ds_read_starts[i] = tok.read_start
                ds_read_ends[i] = tok.read_end

                # Chromosome holdout: chr17→val, chr8→test, all else→train
                # Purpose: prevent genomic region leakage between train/val/test.
                # chr17 (~2.7%) is a modest-sized holdout; chr8 (~4.4%) is larger.
                # Sex chromosomes stay in train to expose model to sperm-specific
                # X/Y chromatin biology.
                if tok.chrom == 'chr17':
                    ds_split[i] = 1  # val
                elif tok.chrom == 'chr8':
                    ds_split[i] = 2  # test
                else:
                    ds_split[i] = 0  # train

    # Report split sizes
    with h5py.File(cache_path, 'r') as fcheck:
        splits = fcheck['split'][:]
        n_train = int((splits == 0).sum())
        n_val = int((splits == 1).sum())
        n_test = int((splits == 2).sum())
    print(f"  Cache saved to {cache_path}")
    print(f"  Size: {cache_path.stat().st_size / 1e6:.1f} MB")
    print(f"  Split sizes — train: {n_train:,} | val (chr17): {n_val:,} | test (chr8): {n_test:,}")


# =====================================================================
# v3 cache builder (for v5 foundation-model training):
#   - Per-sample labeling: m6a_200 -> label=1, PS00750-755 -> label=1 only
#     if protamine_fraction >= maturity_threshold (else -1), GM12878+SMaHT ->
#     label=0, all testes -> label=-1. Avoids locus-aware rule-based negatives
#     and the MIL label-noise of forcing stalled spermatids to the sperm anchor.
#   - pool_ids field (0=sperm, 1=testes, 2=somatic) so the stratified sampler
#     can compose 20/60/20 batches.
#   - Overflow logging (how many reads have n_tokens > max_tokens).
#   - Reservoir-sampled per source file for genome coverage.
#   - chr17 -> val, chr8 -> test, everything else -> train.
# =====================================================================

POOL_SPERM = 0
POOL_TESTES = 1
POOL_SOMATIC = 2


def _reservoir_sample_with_source(
    source_paths,
    n_per_source,
    valid_chroms=None,
    min_read_length=10000,
    seed=42,
    scan_cap=None,
):
    """Reservoir-sample up to n_per_source reads from EACH of source_paths.

    Yields (read, source_path) pairs, so the caller can apply per-source
    labeling rules (maturity filter, pool assignment, etc.).

    IMPORTANT: BigBed files are chrom-sorted. Reservoir sampling over a
    truncated scan (e.g. first 500K reads) gives a chr1-biased sample for
    large files (GM12878, testes, etc.). Default `scan_cap=None` scans the
    entire file for genome-wide coverage — slower, but correct.
    """
    import random
    rng = random.Random(seed)
    if scan_cap is None:
        scan_cap = None  # None means "no cap"; parse_bed_file streams the whole file

    for p in source_paths:
        reservoir = []
        n_seen = 0
        for read in parse_bed_file(p, min_read_length=min_read_length,
                                   max_reads=scan_cap,  # None = no cap → whole file
                                   valid_chroms=valid_chroms):
            if len(reservoir) < n_per_source:
                reservoir.append(read)
            else:
                j = rng.randint(0, n_seen)
                if j < n_per_source:
                    reservoir[j] = read
            n_seen += 1

        chroms_seen = len(set(r.chrom for r in reservoir))
        print(f"    {Path(p).name}: {len(reservoir):,} reads (scanned {n_seen:,}, {chroms_seen} chroms)")
        for read in reservoir:
            yield read, p


def build_hdf5_cache_v3(
    cache_path,
    sperm_paths,          # list of BED/BigBed — purified sperm (m6a_200 + PS00750-755)
    testes_paths,         # list of paths — testes (unlabeled)
    somatic_paths,        # list of paths — real somatic (GM12878 + SMaHT)
    pure_sperm_names=('m6a_200_fp',),    # stems treated as fully-mature → always label=1
    maturity_threshold: float = 0.90,    # PS00750-755 reads below this → label=-1
    max_tokens: int = 128,
    n_per_sperm: int = 12000,
    n_per_testes: int = 30000,
    n_per_somatic: int = 7000,
    min_footprints: int = 5,
    size_encoding: str = 'absolute',   # 'absolute' (v5) or 'fractional' (v6)
    use_lacuna_tokens: bool = False,   # v8: insert LACUNA tokens at gaps ≥ 300bp
) -> None:
    """v3 cache builder: per-sample labeling with maturity filter, pool_ids
    for stratified batching, chrom-held splits, overflow logging.

    Labels:
      1 = sperm anchor (m6a_200 all reads; PS00750-755 reads above maturity_threshold)
      0 = somatic anchor (all GM12878 + SMaHT reads)
     -1 = unlabeled (all testes reads; PS00750-755 reads below threshold)

    Pool IDs (for stratified sampler):
      0 = sperm (m6a_200 + PS00750-755), 1 = testes, 2 = somatic
    """
    from pathlib import Path as _P
    sperm_paths = [_P(p) for p in sperm_paths]
    testes_paths = [_P(p) for p in testes_paths]
    somatic_paths = [_P(p) for p in somatic_paths]

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print("Building HDF5 cache v3 (foundation-model / v5 proposal)...")
    print(f"  Sperm samples ({len(sperm_paths)}):   {[p.name for p in sperm_paths]}  (n_per_sample={n_per_sperm})")
    print(f"  Testes samples ({len(testes_paths)}): {[p.name for p in testes_paths]}  (n_per_sample={n_per_testes})")
    print(f"  Somatic samples ({len(somatic_paths)}): {[p.name for p in somatic_paths]}  (n_per_sample={n_per_somatic})")
    print(f"  Pure-sperm stems (always label=1): {list(pure_sperm_names)}")
    print(f"  Maturity threshold for non-pure sperm: protamine_fraction >= {maturity_threshold}")

    all_tokens = []
    pool_ids = []
    source_names = []
    seen_ids = set()
    overflow_n_tokens = 0  # count of reads with raw n_tokens > max_tokens
    per_source_total = 0

    def _is_pure_sperm(path):
        stem = Path(path).stem  # e.g. 'm6a_200_fp'
        return any(s in stem for s in pure_sperm_names)

    # --- 1. Sperm pool ---
    print("  Loading sperm pool (reservoir-sampled)...")
    n_sperm_label1 = 0
    n_sperm_label_neg = 0
    for read, src in _reservoir_sample_with_source(
            sperm_paths, n_per_sperm, valid_chroms=HUMAN_CHROMS):
        # Filter degenerate reads
        if read.n_footprints < min_footprints:
            continue
        if read.read_id in seen_ids:
            continue
        # Per-source labeling
        if _is_pure_sperm(src):
            label = 1
        else:
            if read.protamine_fraction() >= maturity_threshold:
                label = 1
            else:
                label = -1
        tok = tokenize_read(read, label=label, size_encoding=size_encoding,
                             use_lacuna_tokens=use_lacuna_tokens)
        if tok.n_tokens > max_tokens:
            overflow_n_tokens += 1
        all_tokens.append(tok)
        pool_ids.append(POOL_SPERM)
        source_names.append(src.name)
        seen_ids.add(tok.read_id)
        if label == 1:
            n_sperm_label1 += 1
        else:
            n_sperm_label_neg += 1
        per_source_total += 1
    print(f"    Sperm pool total: {n_sperm_label1:,} label=1, "
          f"{n_sperm_label_neg:,} sub-maturity (label=-1)")

    # Replace the obsolete n_sperm variable (collapsed into per-sample budget)
    #   -- nothing here, left intentionally blank

    # --- 2. Testes pool (all unlabeled) ---
    print("  Loading testes pool (reservoir-sampled)...")
    n_testes_total = 0
    for read, src in _reservoir_sample_with_source(
            testes_paths, n_per_testes, valid_chroms=HUMAN_CHROMS):
        if read.n_footprints < min_footprints:
            continue
        if read.read_id in seen_ids:
            continue
        tok = tokenize_read(read, label=-1, size_encoding=size_encoding,
                             use_lacuna_tokens=use_lacuna_tokens)
        if tok.n_tokens > max_tokens:
            overflow_n_tokens += 1
        all_tokens.append(tok)
        pool_ids.append(POOL_TESTES)
        source_names.append(src.name)
        seen_ids.add(tok.read_id)
        n_testes_total += 1
    print(f"    Testes pool total: {n_testes_total:,}")

    # --- 3. Somatic pool ---
    print("  Loading somatic pool (reservoir-sampled)...")
    n_somatic_total = 0
    for read, src in _reservoir_sample_with_source(
            somatic_paths, n_per_somatic, valid_chroms=HUMAN_CHROMS):
        if read.n_footprints < min_footprints:
            continue
        if read.read_id in seen_ids:
            continue
        tok = tokenize_read(read, label=0, size_encoding=size_encoding,
                             use_lacuna_tokens=use_lacuna_tokens)
        if tok.n_tokens > max_tokens:
            overflow_n_tokens += 1
        all_tokens.append(tok)
        pool_ids.append(POOL_SOMATIC)
        source_names.append(src.name)
        seen_ids.add(tok.read_id)
        n_somatic_total += 1
    print(f"    Somatic pool total: {n_somatic_total:,}")

    total = len(all_tokens)
    label_counts = {1: 0, 0: 0, -1: 0}
    for t in all_tokens:
        label_counts[t.label] += 1
    pool_counts = {POOL_SPERM: 0, POOL_TESTES: 0, POOL_SOMATIC: 0}
    for pid in pool_ids:
        pool_counts[pid] += 1
    overflow_pct = 100.0 * overflow_n_tokens / max(total, 1)
    print(f"  Total reads: {total:,}")
    print(f"  By label:  1={label_counts[1]:,}  0={label_counts[0]:,}  -1={label_counts[-1]:,}")
    print(f"  By pool:   sperm={pool_counts[POOL_SPERM]:,}  testes={pool_counts[POOL_TESTES]:,}  "
          f"somatic={pool_counts[POOL_SOMATIC]:,}")
    print(f"  Overflow (raw n_tokens > {max_tokens}): {overflow_n_tokens:,} "
          f"({overflow_pct:.2f}%) — these reads are truncated")

    # --- Write HDF5 (schema = v2 schema + pool_ids field) ---
    seq_len = max_tokens + 1

    with h5py.File(cache_path, 'w') as f:
        dt_int8 = np.int8; dt_int16 = np.int16; dt_int32 = np.int32
        dt_int64 = np.int64; dt_f32 = np.float32
        chunk = min(1000, total)

        ds_type_ids = f.create_dataset('type_ids', shape=(total, seq_len),
                                       dtype=dt_int8, chunks=(chunk, seq_len),
                                       compression='gzip', compression_opts=4)
        ds_log_sizes = f.create_dataset('log_sizes', shape=(total, seq_len),
                                        dtype=dt_f32, chunks=(chunk, seq_len),
                                        compression='gzip', compression_opts=4)
        ds_gap_to_next = f.create_dataset('gap_to_next', shape=(total, seq_len),
                                          dtype=dt_f32, chunks=(chunk, seq_len),
                                          compression='gzip', compression_opts=4)
        ds_is_first = f.create_dataset('is_first', shape=(total, seq_len),
                                       dtype=bool, chunks=(chunk, seq_len),
                                       compression='gzip', compression_opts=4)
        ds_is_last = f.create_dataset('is_last', shape=(total, seq_len),
                                      dtype=bool, chunks=(chunk, seq_len),
                                      compression='gzip', compression_opts=4)
        ds_size_bins = f.create_dataset('raw_size_bins', shape=(total, seq_len),
                                        dtype=dt_int16, chunks=(chunk, seq_len),
                                        compression='gzip', compression_opts=4)
        ds_gap_bins = f.create_dataset('raw_gap_bins', shape=(total, seq_len),
                                       dtype=dt_int16, chunks=(chunk, seq_len),
                                       compression='gzip', compression_opts=4)
        ds_n_tokens = f.create_dataset('n_tokens', shape=(total,), dtype=dt_int32)
        ds_summary = f.create_dataset('summary_features', shape=(total, 25),
                                      dtype=dt_f32, chunks=(chunk, 25),
                                      compression='gzip', compression_opts=4)
        ds_labels = f.create_dataset('labels', shape=(total,), dtype=dt_int8)
        ds_pool = f.create_dataset('pool_ids', shape=(total,), dtype=dt_int8)

        ds_orig_starts = f.create_dataset('orig_starts', shape=(total, max_tokens),
                                          dtype=dt_int64, chunks=(chunk, max_tokens),
                                          compression='gzip', compression_opts=4)
        ds_orig_sizes = f.create_dataset('orig_sizes', shape=(total, max_tokens),
                                         dtype=dt_int64, chunks=(chunk, max_tokens),
                                         compression='gzip', compression_opts=4)

        dt_str = h5py.string_dtype()
        ds_read_ids = f.create_dataset('read_ids', shape=(total,), dtype=dt_str)
        ds_chroms = f.create_dataset('chroms', shape=(total,), dtype=dt_str)
        ds_sources = f.create_dataset('source', shape=(total,), dtype=dt_str)
        ds_read_starts = f.create_dataset('read_starts', shape=(total,), dtype=dt_int64)
        ds_read_ends = f.create_dataset('read_ends', shape=(total,), dtype=dt_int64)

        ds_split = f.create_dataset('split', shape=(total,), dtype=dt_int8)
        # Mask for whether this read's loss should contribute to Head A.
        # True iff labeled (label != -1) AND chrom not in {chrX, chrY}.
        ds_head_a_mask = f.create_dataset('head_a_mask', shape=(total,), dtype=bool)

        write_chunk = 5000
        for chunk_start in range(0, total, write_chunk):
            chunk_end = min(chunk_start + write_chunk, total)
            if chunk_start % 20000 == 0:
                print(f"    Writing {chunk_start}/{total}...")

            for i in range(chunk_start, chunk_end):
                tok = all_tokens[i]
                n = min(tok.n_tokens, max_tokens)

                type_ids = np.zeros(seq_len, dtype=dt_int8)
                log_sizes = np.zeros(seq_len, dtype=dt_f32)
                gap_to_next = np.zeros(seq_len, dtype=dt_f32)
                is_first = np.zeros(seq_len, dtype=bool)
                is_last = np.zeros(seq_len, dtype=bool)
                size_bins = np.zeros(seq_len, dtype=dt_int16)
                gap_bins = np.zeros(seq_len, dtype=dt_int16)
                orig_starts = np.zeros(max_tokens, dtype=dt_int64)
                orig_sizes = np.zeros(max_tokens, dtype=dt_int64)

                actual_len = n + 1
                type_ids[:actual_len] = tok.type_ids[:actual_len]
                log_sizes[:actual_len] = tok.log_sizes[:actual_len]
                gap_to_next[:actual_len] = tok.gap_to_next[:actual_len]
                is_first[:actual_len] = tok.is_first[:actual_len]
                is_last[:actual_len] = tok.is_last[:actual_len]
                size_bins[:actual_len] = tok.raw_size_bins[:actual_len]
                gap_bins[:actual_len] = tok.raw_gap_bins[:actual_len]
                orig_starts[:n] = tok.footprint_starts[:n]
                orig_sizes[:n] = tok.footprint_sizes[:n]

                ds_type_ids[i] = type_ids
                ds_log_sizes[i] = log_sizes
                ds_gap_to_next[i] = gap_to_next
                ds_is_first[i] = is_first
                ds_is_last[i] = is_last
                ds_size_bins[i] = size_bins
                ds_gap_bins[i] = gap_bins
                ds_n_tokens[i] = n
                ds_summary[i] = tok.summary_features
                ds_labels[i] = tok.label
                ds_pool[i] = pool_ids[i]
                ds_orig_starts[i] = orig_starts
                ds_orig_sizes[i] = orig_sizes
                ds_read_ids[i] = tok.read_id
                ds_chroms[i] = tok.chrom
                ds_sources[i] = source_names[i]
                ds_read_starts[i] = tok.read_start
                ds_read_ends[i] = tok.read_end

                # Chromosome holdout
                if tok.chrom == 'chr17':
                    ds_split[i] = 1  # val
                elif tok.chrom == 'chr8':
                    ds_split[i] = 2  # test
                else:
                    ds_split[i] = 0  # train

                # Head A mask: labeled AND not sex chromosome
                is_labeled = tok.label != -1
                is_autosome_or_other = tok.chrom not in ('chrX', 'chrY')
                ds_head_a_mask[i] = bool(is_labeled and is_autosome_or_other)

    # Report final stats
    with h5py.File(cache_path, 'r') as fcheck:
        splits = fcheck['split'][:]
        pools = fcheck['pool_ids'][:]
        ha_mask = fcheck['head_a_mask'][:]
        labels = fcheck['labels'][:]
        n_train = int((splits == 0).sum())
        n_val = int((splits == 1).sum())
        n_test = int((splits == 2).sum())
        print(f"  Cache saved to {cache_path}")
        print(f"  Size: {cache_path.stat().st_size / 1e6:.1f} MB")
        print(f"  Splits — train: {n_train:,}  val (chr17): {n_val:,}  test (chr8): {n_test:,}")
        print(f"  Head A trainable (labeled + non-X/Y): {int(ha_mask.sum()):,}")
        for pid, pname in [(POOL_SPERM, 'sperm'), (POOL_TESTES, 'testes'), (POOL_SOMATIC, 'somatic')]:
            n_p = int((pools == pid).sum())
            n_p_train = int(((pools == pid) & (splits == 0)).sum())
            print(f"    Pool {pname}: total={n_p:,}  train={n_p_train:,}")


def compute_corrupted_distance_matrix(
    type_ids: torch.Tensor,       # (seq_len,) int
    log_sizes: torch.Tensor,      # (seq_len,) float
    gap_to_next: torch.Tensor,    # (seq_len,) float
    mask: torch.Tensor,           # (seq_len,) bool — True for masked tokens
    masked_spans: torch.Tensor,   # (seq_len,) float — noisy spans for masked tokens
    n_tokens: int,
) -> torch.Tensor:
    """Compute log-distance matrix from corrupted positions.

    For unmasked tokens: use actual size + gap (in bp).
    For masked tokens: use the noisy masked_span hint.
    Build pseudo-positions via cumsum, compute D_ij from those.

    Returns: (seq_len, seq_len) float32 tensor of log10(D_ij + 1).
    """
    seq_len = type_ids.shape[0]
    actual_len = n_tokens + 1  # CLS + tokens

    # Convert log features back to linear bp
    # log_sizes: log10(size_bp), gap_to_next: log10(gap_bp + 1)
    sizes_bp = torch.pow(10.0, log_sizes)  # approximate bp
    gaps_bp = torch.pow(10.0, gap_to_next) - 1.0
    gaps_bp = torch.clamp(gaps_bp, min=0.0)

    # For each token, its span = size + gap_to_next
    spans = sizes_bp + gaps_bp  # (seq_len,)

    # Replace masked token spans with noisy hints
    spans = torch.where(mask, masked_spans, spans)

    # CLS at position 0 has span 0
    spans[0] = 0.0
    # Zero out padding
    spans[actual_len:] = 0.0

    # Build pseudo-positions via cumsum
    positions = torch.cumsum(spans, dim=0)  # (seq_len,)
    # Shift so CLS is at 0
    positions = positions - positions[0]

    # Pairwise distances: D_ij = |pos_i - pos_j|
    D = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))  # (seq_len, seq_len)

    # Log-transform
    log_D = torch.log10(D + 1.0)

    return log_D


class FiberSeqDataset(Dataset):
    """Dataset loading from HDF5 cache with on-the-fly masking and augmentation."""

    def __init__(
        self,
        cache_path: str | Path,
        split: str = 'train',       # 'train', 'val', 'test'
        max_tokens: int = 128,
        mask_rate: float = 0.40,    # v5 default bumped 0.20 → 0.40 for span-MAE
        span_min: int = 3,          # v5: contiguous span length range
        span_max: int = 5,
        contrastive: bool = False,
        augment: bool = True,
        seed: int = 42,
    ):
        self.cache_path = str(cache_path)
        self.max_tokens = max_tokens
        self.seq_len = max_tokens + 1  # +1 for CLS
        self.mask_rate = mask_rate
        self.span_min = span_min
        self.span_max = span_max
        self.contrastive = contrastive
        self.augment = augment and split == 'train'
        self.rng = np.random.default_rng(seed)

        # FiberFormer: 9 kb footprint-size clamp + ±0.02 log10 chemistry jitter +
        # 50kb absolute bp-position shift. See PLAN docs for derivation.
        self.LOG_SIZE_CAP = math.log10(9000.0)              # ≈ 3.954
        self.CLAMP_BIN = int(SIZE_QUANTIZER.encode(9000.0)) # discrete bin for 9kb
        self.BP_JITTER_MAX = 50_000                         # uniform [0, 50_000) shift
        self.CHEM_JITTER_LOG = 0.02                         # log10 space, ≈ ±5% bp

        split_map = {'train': 0, 'val': 1, 'test': 2}
        split_id = split_map[split]

        # Load indices for this split (lazy file opening handled by worker_init_fn)
        with h5py.File(self.cache_path, 'r') as f:
            splits = f['split'][:]
            keep = splits == split_id
            sidecar = Path(self.cache_path + '.mapq_keep.npy')
            if sidecar.exists():
                mapq_keep = np.load(sidecar)
                if len(mapq_keep) != len(splits):
                    raise RuntimeError(
                        f"MAPQ sidecar length {len(mapq_keep):,} != cache rows {len(splits):,}"
                    )
                excluded = int((~mapq_keep & keep).sum())
                keep &= mapq_keep
                print(f"[FiberSeqDataset {split}] MAPQ blacklist from {sidecar.name}: "
                      f"excluded {excluded:,} of {(splits == split_id).sum():,} "
                      f"({100.0*excluded/max((splits==split_id).sum(),1):.2f}%)")
            self.indices = np.where(keep)[0]
            self.total_size = len(self.indices)

        # File handle will be opened per-worker
        self._file = None

    def _open_file(self):
        if self._file is None:
            self._file = h5py.File(self.cache_path, 'r')

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        self._open_file()
        real_idx = int(self.indices[idx])

        # Load from HDF5
        type_ids = torch.from_numpy(self._file['type_ids'][real_idx].astype(np.int64))
        log_sizes = torch.from_numpy(self._file['log_sizes'][real_idx].astype(np.float32))
        gap_to_next = torch.from_numpy(self._file['gap_to_next'][real_idx].astype(np.float32))
        is_first = torch.from_numpy(self._file['is_first'][real_idx])
        is_last = torch.from_numpy(self._file['is_last'][real_idx])
        raw_size_bins = torch.from_numpy(self._file['raw_size_bins'][real_idx].astype(np.int64))
        raw_gap_bins = torch.from_numpy(self._file['raw_gap_bins'][real_idx].astype(np.int64))
        n_tokens = int(self._file['n_tokens'][real_idx])
        summary = torch.from_numpy(self._file['summary_features'][real_idx].astype(np.float32))
        label = int(self._file['labels'][real_idx])
        # v7: per-token bp positions (relative to read start). Available in v3+ caches
        # as `orig_starts` (size max_tokens). CLS goes at position 0, then orig_starts.
        if 'orig_starts' in self._file:
            orig_starts = self._file['orig_starts'][real_idx].astype(np.int32)
            bp_positions = np.zeros(self.seq_len, dtype=np.int32)
            bp_positions[1:1 + min(n_tokens, self.max_tokens)] = orig_starts[:min(n_tokens, self.max_tokens)]
            bp_positions_t = torch.from_numpy(bp_positions)
        else:
            bp_positions_t = torch.arange(self.seq_len, dtype=torch.int32)
        # v3+ cache fields: head_a_mask (chrom + label aware). Fallback to label != -1 for v2 caches.
        if 'head_a_mask' in self._file:
            head_a_mask = bool(self._file['head_a_mask'][real_idx])
        else:
            head_a_mask = (label != -1)

        actual_len = n_tokens + 1  # CLS + n tokens

        # --- FiberFormer Change 2: bp_position jitter (length-leak via PE) ---
        # Uniform shift on ALL positions including CLS so relative distances are
        # preserved (Conv1D and distance bias depend on relative coords; absolute
        # read-start anchor is the only thing that should change).
        if self.augment:
            shift = int(self.rng.integers(0, self.BP_JITTER_MAX))
            bp_positions_t = bp_positions_t + shift

        # --- FiberFormer Changes 3+4: chemistry jitter (size + gap) THEN 9kb clamp ---
        # Order matters: jitter first so trivial reads can drift slightly, then
        # the clamp collapses every drifted trivial read to the same hard ceiling.
        # Inverting the order would re-introduce continuous variation on
        # already-clamped tokens.
        if self.augment and n_tokens > 0:
            size_shift = float(self.rng.uniform(-self.CHEM_JITTER_LOG,
                                                 self.CHEM_JITTER_LOG))
            # Shift only real tokens (skip CLS at index 0). For gap_to_next, the
            # last token's gap is a sentinel for "end of read" — preserve it.
            log_sizes[1:1 + n_tokens] = log_sizes[1:1 + n_tokens] + size_shift
            if n_tokens >= 2:
                # gap_to_next[i] for i in [1, n_tokens-1] points to the next
                # real footprint; gap_to_next[n_tokens] has no successor.
                gap_to_next[1:n_tokens] = gap_to_next[1:n_tokens] + size_shift

        # Hard clamp on continuous + discrete size features. Trivial-prot reads
        # (n_fps=1, single 10-60kb footprint) collapse to the same token here.
        log_sizes = torch.clamp(log_sizes, max=self.LOG_SIZE_CAP)
        raw_size_bins = torch.clamp(raw_size_bins, max=self.CLAMP_BIN)

        # --- Span masking for Head B ---
        # Sample contiguous token spans until cumulative coverage ~mask_rate.
        # Span length is uniform in [self.span_min, self.span_max].
        # Random per-token masking is trivially solved by neighbor-copy given
        # chromatin's spatial autocorrelation; spans force distant-context
        # reconstruction.
        mask = torch.zeros(self.seq_len, dtype=torch.bool)
        masked_spans = torch.zeros(self.seq_len, dtype=torch.float32)

        # FiberFormer Change 6a: sparse-read guard. With span_min=3, a read with
        # n_tokens<3 would have every footprint masked, leaving the encoder
        # with a pure-MASK sequence. Head B targets become pathological. Skip
        # masking for these reads — mask stays all-False, masked_spans stays
        # zero. Head A still trains on the unmasked tokens.
        if n_tokens >= 3:
            target_n = max(1, int(round(n_tokens * self.mask_rate)))
            footprint_start = 1  # skip CLS
            footprint_end = 1 + n_tokens
            n_masked = 0
            attempts = 0
            # Guard against pathological inputs / short reads
            max_attempts = max(10, 4 * target_n)
            while n_masked < target_n and attempts < max_attempts:
                span_len = int(self.rng.integers(self.span_min, self.span_max + 1))
                span_len = min(span_len, n_tokens)  # clamp for short reads
                start = int(self.rng.integers(footprint_start, footprint_end - span_len + 1))
                end = start + span_len
                # Count only NEW tokens this span would add
                newly_masked = (~mask[start:end]).sum().item()
                if newly_masked == 0:
                    attempts += 1
                    continue
                mask[start:end] = True
                n_masked += newly_masked
                attempts += 1

            # Compute noisy span hints (size + gap) for each masked position.
            # Vectorized: extract log features at mask positions, reconstruct bp, add noise.
            mask_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if mask_indices.numel() > 0:
                sizes_bp = torch.pow(10.0, log_sizes[mask_indices])
                gaps_bp = torch.clamp(torch.pow(10.0, gap_to_next[mask_indices]) - 1.0, min=0.0)
                true_span = sizes_bp + gaps_bp
                noise = torch.from_numpy(
                    1.0 + self.rng.normal(0, 0.1, size=mask_indices.numel())
                ).float()
                masked_spans[mask_indices] = true_span * noise

            # Replace masked token features with MASK values
            type_ids_masked = type_ids.clone()
            log_sizes_masked = log_sizes.clone()
            gap_to_next_masked = gap_to_next.clone()
            type_ids_masked[mask] = MASK
            log_sizes_masked[mask] = 0.0
            gap_to_next_masked[mask] = 0.0
        else:
            type_ids_masked = type_ids.clone()
            log_sizes_masked = log_sizes.clone()
            gap_to_next_masked = gap_to_next.clone()

        # --- Corrupted distance matrix ---
        log_dist = compute_corrupted_distance_matrix(
            type_ids_masked, log_sizes_masked, gap_to_next_masked,
            mask, masked_spans, n_tokens,
        )

        # --- Attention mask (1=attend, 0=ignore) ---
        attn_mask = torch.zeros(self.seq_len, dtype=torch.bool)
        attn_mask[:actual_len] = True

        # Pack continuous features: [log_size, gap_to_next, is_first, is_last, masked_span]
        continuous = torch.stack([
            log_sizes_masked,
            gap_to_next_masked,
            is_first.float(),
            is_last.float(),
            masked_spans,
        ], dim=-1)  # (seq_len, 5)

        result = {
            'type_ids': type_ids_masked,           # (seq_len,)
            'continuous': continuous,               # (seq_len, 5)
            'attn_mask': attn_mask,                 # (seq_len,)
            'log_dist': log_dist,                   # (seq_len, seq_len)
            'summary': summary,                     # (25,)
            'label': torch.tensor(label, dtype=torch.long),
            'head_a_mask': torch.tensor(head_a_mask, dtype=torch.bool),
            # Targets for Head B (only at masked positions)
            'mask': mask,                           # (seq_len,) — masked_positions
            'target_type_ids': type_ids,            # (seq_len,) — original types
            'target_size_bins': raw_size_bins,      # (seq_len,)
            'target_gap_bins': raw_gap_bins,        # (seq_len,)
            'n_tokens': n_tokens,
            # v7: per-token bp offset relative to read start, for bp-SinCos PE
            'bp_positions': bp_positions_t,         # (seq_len,) int32
        }

        return result

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


def worker_init_fn(worker_id):
    """Re-seed RNG per worker to avoid duplicate augmentations."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        dataset._file = None  # Force re-open in this worker
        dataset.rng = np.random.default_rng(worker_info.seed % (2**32))


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate: dynamic padding to max actual length in batch."""
    # Find max actual length
    max_len = max(item['n_tokens'] + 1 for item in batch)  # +1 for CLS

    # Keys that have sequence dimension (should be trimmed to max_len)
    seq_keys = {'type_ids', 'continuous', 'attn_mask', 'log_dist',
                'mask', 'target_type_ids', 'target_size_bins', 'target_gap_bins',
                'bp_positions'}

    result = {}
    for key in batch[0]:
        if key == 'n_tokens':
            result[key] = torch.tensor([item[key] for item in batch])
            continue
        tensors = [item[key] for item in batch]
        if tensors[0].dim() == 0:
            result[key] = torch.stack(tensors)
        elif tensors[0].dim() == 1:
            if key in seq_keys:
                result[key] = torch.stack([t[:max_len] for t in tensors])
            else:
                result[key] = torch.stack(tensors)
        elif tensors[0].dim() == 2:
            if key in seq_keys:
                result[key] = torch.stack([t[:max_len, :max_len] for t in tensors])
            else:
                result[key] = torch.stack(tensors)

    return result


class StratifiedBatchSampler:
    """Yields batches stratified by pool_id (0=sperm, 1=testes, 2=somatic).

    Each batch is composed according to `pool_ratios` (default 20/60/20).
    Within each pool, indices are shuffled every epoch. Epoch length is
    determined by the largest pool; smaller pools are cycled (reshuffled)
    as needed so every batch hits the target composition.

    Works with PyTorch's DataLoader(batch_sampler=...). Do NOT also pass
    batch_size or shuffle when using this.
    """

    def __init__(self, dataset, cache_path, batch_size: int,
                 pool_ratios: dict, seed: int = 42):
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

        # Load pool_ids for the dataset's split
        with h5py.File(cache_path, 'r') as f:
            if 'pool_ids' not in f:
                raise KeyError("Cache does not contain 'pool_ids' (not a v3 cache)")
            all_pool_ids = f['pool_ids'][:]

        # dataset.indices holds the subset for this split
        split_pool_ids = all_pool_ids[dataset.indices]

        self.pool_indices = {}
        for pool_name, pool_id in [('sperm', POOL_SPERM),
                                    ('testes', POOL_TESTES),
                                    ('somatic', POOL_SOMATIC)]:
            # These are positions into the Dataset (i.e. what __getitem__ consumes).
            idx = np.where(split_pool_ids == pool_id)[0]
            if len(idx) == 0:
                raise ValueError(f"No reads in split for pool={pool_name} — cannot stratify")
            self.pool_indices[pool_name] = idx

        # Ratio → per-batch counts
        self.per_batch = {}
        remaining = batch_size
        names = list(pool_ratios.keys())
        for name in names[:-1]:
            n = int(round(batch_size * pool_ratios[name]))
            self.per_batch[name] = n
            remaining -= n
        self.per_batch[names[-1]] = remaining

        # Epoch length: enough batches to cover the largest pool at its target rate
        # (smaller pools will cycle)
        limiting = max(len(self.pool_indices[n]) / self.per_batch[n] for n in names)
        self.n_batches = int(limiting)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1

        # Shuffle each pool, keep a cursor; reshuffle on cycle.
        pools = {name: rng.permutation(idx).copy()
                 for name, idx in self.pool_indices.items()}
        cursors = {name: 0 for name in pools}

        for _ in range(self.n_batches):
            batch = []
            for name, n in self.per_batch.items():
                pool = pools[name]
                start = cursors[name]
                end = start + n
                if end > len(pool):
                    # Reshuffle and wrap around
                    wrap = end - len(pool)
                    batch.extend(pool[start:].tolist())
                    pools[name] = rng.permutation(pool)
                    pool = pools[name]
                    batch.extend(pool[:wrap].tolist())
                    cursors[name] = wrap
                else:
                    batch.extend(pool[start:end].tolist())
                    cursors[name] = end
            yield batch

    def __len__(self):
        return self.n_batches


def create_dataloaders(
    cache_path: str | Path,
    batch_size: int = 512,
    num_workers: int = 4,
    max_tokens: int = 128,
    mask_rate: float = 0.40,
    span_min: int = 3,
    span_max: int = 5,
    seed: int = 42,
    pool_ratios: dict = None,  # {'sperm': 0.2, 'testes': 0.6, 'somatic': 0.2}
) -> dict:
    """Create train/val/test DataLoaders.

    If `pool_ratios` is provided and the cache has a 'pool_ids' field, the
    train loader uses a StratifiedBatchSampler; val/test still use plain
    shuffle=False sequential sampling.
    """
    loaders = {}
    for split in ['train', 'val', 'test']:
        ds = FiberSeqDataset(
            cache_path=cache_path,
            split=split,
            max_tokens=max_tokens,
            mask_rate=mask_rate,
            span_min=span_min,
            span_max=span_max,
            augment=(split == 'train'),
            seed=seed,
        )

        loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

        if split == 'train' and pool_ratios is not None:
            try:
                sampler = StratifiedBatchSampler(
                    ds, cache_path=cache_path, batch_size=batch_size,
                    pool_ratios=pool_ratios, seed=seed,
                )
                loader_kwargs['batch_sampler'] = sampler
                # batch_sampler is mutually exclusive with batch_size/shuffle/drop_last
                loader_kwargs.pop('batch_size')
            except (KeyError, ValueError) as e:
                print(f"[warn] StratifiedBatchSampler unavailable ({e}); falling back to shuffle")
                loader_kwargs['shuffle'] = True
                loader_kwargs['drop_last'] = True
        else:
            loader_kwargs['shuffle'] = (split == 'train')
            loader_kwargs['drop_last'] = (split == 'train')

        loaders[split] = DataLoader(ds, **loader_kwargs)
    return loaders


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build HDF5 cache')
    parser.add_argument('--cache-path', type=str,
                        default=str(Path(__file__).parent.parent / 'outputs' / 'cache.h5'))
    parser.add_argument('--n-sperm', type=int, default=50000)
    parser.add_argument('--n-somatic', type=int, default=50000)
    parser.add_argument('--n-unlabeled', type=int, default=100000)
    parser.add_argument('--max-tokens', type=int, default=128)
    args = parser.parse_args()

    sperm_path = BED_DIR / 'm6a_200_fp.bed'
    testes_path = BED_DIR / 'PS30065_fp.bed'

    build_hdf5_cache(
        cache_path=args.cache_path,
        sperm_path=sperm_path,
        testes_path=testes_path,
        max_tokens=args.max_tokens,
        n_sperm=args.n_sperm,
        n_somatic=args.n_somatic,
        n_unlabeled=args.n_unlabeled,
    )
