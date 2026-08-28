"""
Size and gap quantization for masked reconstruction targets (Head B).

Size bins (~155): fine resolution in nucleosome range, coarser for protamine.
Gap bins (~70): linear at short range, log-scaled for long-range sparsity.
"""

import numpy as np


class SizeQuantizer:
    """Quantizes footprint sizes into discrete bins for cross-entropy targets.

    Bin layout:
        0-80bp:       8bp steps          (10 bins)
        80-300bp:     2bp steps          (110 bins) — resolves CENP-A vs canonical
        300-2000bp:   50bp steps         (34 bins)
        2000-60000bp: log-spaced         (30 bins) — v7 tail, protamine range
        >60000bp:     overflow           (1 bin)
    Total: 185 bins (v7) — v6/prior used 155 bins with 2kb overflow.
    """

    def __init__(self):
        edges_0_80 = np.arange(0, 80, 8)           # 0, 8, 16, ..., 72 (10 edges)
        edges_80_300 = np.arange(80, 300, 2)        # 80, 82, ..., 298 (110 edges)
        edges_300_2000 = np.arange(300, 2000, 50)   # 300, 350, ..., 1950 (34 edges)
        # Log-spaced tail for protamine and ultra-long footprints (v7 addition).
        # geomspace(2000, 60000, 30) gives ~10% growth per bin — matches how
        # sizes are distributed in log space, avoids the 580-bin bloat of
        # linear 100bp steps, and keeps most of the mass in small bins.
        edges_2000_60000 = np.geomspace(2000, 60000, 30, endpoint=False).astype(np.int64)
        # Ensure strictly increasing after rounding
        edges_2000_60000 = np.unique(edges_2000_60000)
        self.edges = np.concatenate([edges_0_80, edges_80_300, edges_300_2000,
                                     edges_2000_60000])
        self.n_bins = len(self.edges) + 1  # +1 for >60kb overflow
        # Centers for decoding
        self._centers = np.zeros(self.n_bins, dtype=np.float32)
        for i in range(len(self.edges) - 1):
            self._centers[i] = (self.edges[i] + self.edges[i + 1]) / 2
        # Last real bin: mid between last edge and 60kb
        self._centers[len(self.edges) - 1] = (self.edges[-1] + 60_000) / 2
        self._centers[-1] = 70_000.0  # >60kb overflow center

    def encode(self, size_bp: float) -> int:
        return int(np.searchsorted(self.edges, size_bp, side='right') - 1)

    def encode_batch(self, sizes: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.edges, sizes, side='right').astype(np.int16) - 1

    def decode_center(self, bin_idx: int) -> float:
        return float(self._centers[bin_idx])

    @property
    def num_bins(self) -> int:
        return self.n_bins


class GapQuantizer:
    """Quantizes gap sizes with log-scaled bins for long-range coverage.

    Bin layout:
        0-50bp:      10bp steps  (5 bins)
        50-200bp:    5bp steps   (30 bins)
        200-2000bp:  ~15 log-spaced bins
        2000-25000bp: ~20 log-spaced bins
    Total: ~70 bins
    """

    def __init__(self):
        edges_0_50 = np.arange(0, 50, 10)          # 0, 10, 20, 30, 40 (5 edges)
        edges_50_200 = np.arange(50, 200, 5)       # 50, 55, ..., 195 (30 edges)
        # Log-scaled from 200 to 2000 (~15 bins)
        edges_200_2000 = np.unique(np.round(
            np.logspace(np.log10(200), np.log10(2000), 16)
        ).astype(np.int64))
        # Log-scaled from 2000 to 25000 (~20 bins)
        edges_2000_25000 = np.unique(np.round(
            np.logspace(np.log10(2000), np.log10(25000), 21)
        ).astype(np.int64))
        # Remove duplicates at boundaries
        edges_200_2000 = edges_200_2000[edges_200_2000 > 200]
        edges_2000_25000 = edges_2000_25000[edges_2000_25000 > 2000]
        self.edges = np.concatenate([
            edges_0_50, edges_50_200, edges_200_2000, edges_2000_25000
        ]).astype(np.float64)
        self.n_bins = len(self.edges) + 1
        # Centers for decoding
        self._centers = np.zeros(self.n_bins, dtype=np.float32)
        for i in range(len(self.edges) - 1):
            self._centers[i] = (self.edges[i] + self.edges[i + 1]) / 2
        self._centers[len(self.edges) - 1] = (self.edges[-1] + 25000) / 2
        self._centers[-1] = 30000.0

    def encode(self, gap_bp: float) -> int:
        return int(np.searchsorted(self.edges, gap_bp, side='right') - 1)

    def encode_batch(self, gaps: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.edges, gaps, side='right').astype(np.int16) - 1

    def decode_center(self, bin_idx: int) -> float:
        return float(self._centers[bin_idx])

    @property
    def num_bins(self) -> int:
        return self.n_bins


# Module-level singletons
SIZE_QUANTIZER = SizeQuantizer()
GAP_QUANTIZER = GapQuantizer()
