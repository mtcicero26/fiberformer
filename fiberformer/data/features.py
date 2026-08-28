"""
Summary statistics computation for XGBoost baselines.

Computes 25 engineered features per FiberRead, split into:
- Features 1-13: All features (including protamine mass)
- Features 14-25: Lacuna-only features (for Baseline 2)
"""

import numpy as np
from .loader import FiberRead


# Feature names for reference
ALL_FEATURE_NAMES = [
    # 1-13: General features (includes protamine mass)
    'read_length',
    'n_footprints',
    'n_gaps',
    'total_footprint_coverage',
    'protamine_coverage',
    'nucleosome_coverage',
    'subnucleosomal_coverage',
    'mean_footprint_size',
    'median_footprint_size',
    'max_footprint_size',
    'footprint_size_std',
    'mean_gap_size',
    'median_gap_size',
    # 14-25: Lacuna-only features
    'n_protamine_lacunae',
    'mean_lacuna_size',
    'nucleosomes_per_kb_in_lacunae',
    'footprint_size_entropy',
    'n_cenpa_footprints',
    'n_canonical_nuc_footprints',
    'n_dinucleosome_footprints',
    'cenpa_to_canonical_ratio',
    'nucleosome_spacing_regularity',
    'fraction_large_protamine',
    'n_footprint_boundaries',
    'mean_protamine_block_size',
]

LACUNA_FEATURE_NAMES = ALL_FEATURE_NAMES[13:]  # features 14-25

# Indices for the two XGBoost baselines
NAIVE_FEATURE_INDICES = list(range(25))
LACUNA_FEATURE_INDICES = list(range(13, 25))


def compute_features(read: FiberRead) -> np.ndarray:
    """Compute all 25 summary features for a single read.

    Returns:
        np.ndarray of shape (25,) with feature values
    """
    sizes = read.footprint_sizes
    length = read.length
    gaps = read.gaps()

    # Size class masks
    is_protamine = sizes > 200
    is_nucleosome = (sizes >= 90) & (sizes <= 200)
    is_subnuc = sizes < 90
    is_cenpa = (sizes >= 100) & (sizes <= 135)
    is_canonical = (sizes > 135) & (sizes <= 170)
    is_dinuc = (sizes >= 240) & (sizes <= 290)
    is_large_prot = sizes > 5000

    # Coverage fractions
    total_cov = sizes.sum() / length if length > 0 else 0
    prot_cov = sizes[is_protamine].sum() / length if length > 0 else 0
    nuc_cov = sizes[is_nucleosome].sum() / length if length > 0 else 0
    sub_cov = sizes[is_subnuc].sum() / length if length > 0 else 0

    # Gap statistics
    mean_gap = gaps.mean() if len(gaps) > 0 else 0
    median_gap = np.median(gaps) if len(gaps) > 0 else 0

    # --- Lacuna features ---

    # Protamine lacunae: gaps between protamine footprints >100bp
    prot_indices = np.where(is_protamine)[0]
    lacunae = []
    if len(prot_indices) > 1:
        for i in range(len(prot_indices) - 1):
            idx_a = prot_indices[i]
            idx_b = prot_indices[i + 1]
            # gap between end of protamine A and start of protamine B
            gap = read.footprint_starts[idx_b] - (read.footprint_starts[idx_a] + sizes[idx_a])
            if gap > 100:
                lacunae.append({
                    'size': gap,
                    'start_idx': idx_a + 1,
                    'end_idx': idx_b,
                })

    n_lacunae = len(lacunae)
    mean_lacuna_size = np.mean([l['size'] for l in lacunae]) if lacunae else 0

    # Nucleosomes per kb within lacunae
    total_lacuna_bp = sum(l['size'] for l in lacunae)
    nucs_in_lacunae = 0
    for lac in lacunae:
        for idx in range(lac['start_idx'], lac['end_idx']):
            if idx < len(sizes) and is_nucleosome[idx]:
                nucs_in_lacunae += 1
    nuc_per_kb_lacunae = (nucs_in_lacunae / (total_lacuna_bp / 1000)) if total_lacuna_bp > 0 else 0

    # Footprint size entropy
    if len(sizes) > 1:
        # Bin sizes into categories for entropy
        bins = np.array([0, 90, 200, 2000, np.inf])
        counts = np.histogram(sizes, bins=bins)[0]
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
    else:
        entropy = 0

    # CENP-A and canonical nucleosome counts
    n_cenpa = int(is_cenpa.sum())
    n_canonical = int(is_canonical.sum())
    n_dinuc = int(is_dinuc.sum())
    cenpa_ratio = n_cenpa / n_canonical if n_canonical > 0 else 0

    # Nucleosome spacing regularity
    nuc_indices = np.where(is_nucleosome)[0]
    if len(nuc_indices) > 1:
        # Inter-nucleosome distances (center-to-center)
        nuc_centers = read.footprint_starts[nuc_indices] + sizes[nuc_indices] / 2
        spacings = np.diff(nuc_centers)
        spacing_regularity = np.std(spacings) if len(spacings) > 0 else 0
    else:
        spacing_regularity = 0

    # Large protamine fraction
    frac_large_prot = sizes[is_large_prot].sum() / length if length > 0 else 0

    # Number of footprint boundaries (transitions)
    n_boundaries = 2 * read.n_footprints  # each footprint has a start and end boundary

    # Mean protamine block size
    prot_sizes = sizes[is_protamine]
    mean_prot_size = prot_sizes.mean() if len(prot_sizes) > 0 else 0

    features = np.array([
        # 1-13: General
        length,
        read.n_footprints,
        len(gaps),
        total_cov,
        prot_cov,
        nuc_cov,
        sub_cov,
        sizes.mean() if len(sizes) > 0 else 0,
        np.median(sizes) if len(sizes) > 0 else 0,
        sizes.max() if len(sizes) > 0 else 0,
        sizes.std() if len(sizes) > 1 else 0,
        mean_gap,
        median_gap,
        # 14-25: Lacuna-only
        n_lacunae,
        mean_lacuna_size,
        nuc_per_kb_lacunae,
        entropy,
        n_cenpa,
        n_canonical,
        n_dinuc,
        cenpa_ratio,
        spacing_regularity,
        frac_large_prot,
        n_boundaries,
        mean_prot_size,
    ], dtype=np.float64)

    return features


def compute_features_batch(reads: list[FiberRead]) -> np.ndarray:
    """Compute features for a list of reads.

    Returns:
        np.ndarray of shape (n_reads, 25)
    """
    return np.array([compute_features(r) for r in reads])
