"""
BED/BigBed parser for FiberHMM footprint files.

Supports two formats:
  - BigBed (.bb): standard BED12 output from FiberHMM v2+ extract_tags.py
      cols: chrom, start, end, name, score, strand, thickStart, thickEnd,
            itemRgb, blockCount, blockSizes, blockStarts
  - Legacy BED (.bed): pre-2.0 FiberHMM 10-column format
      cols: chrom, start, end, name, thickStart, thickEnd, itemRgb,
            blockCount, blockStarts, blockSizes   (starts/sizes SWAPPED vs BED12)

parse_bed_file() accepts either extension and handles both transparently.
"""

import os
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterator, Optional
from dataclasses import dataclass
import io

# bigBedToBed may not be in PATH (installed to ~/bigBedToBed)
def _find_tool(name: str) -> str:
    import shutil
    found = shutil.which(name)
    if found:
        return found
    candidate = os.path.expanduser(f'~/{name}')
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    raise FileNotFoundError(f"{name} not found in PATH or ~/. Install from UCSC: http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/{name}")


@dataclass
class FiberRead:
    """A single Fiber-seq read with its footprint calls."""
    chrom: str
    start: int
    end: int
    read_id: str
    footprint_starts: np.ndarray   # relative to read start
    footprint_sizes: np.ndarray

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def n_footprints(self) -> int:
        return len(self.footprint_sizes)

    @property
    def footprint_ends(self) -> np.ndarray:
        return self.footprint_starts + self.footprint_sizes

    def gaps(self) -> np.ndarray:
        """Compute gap sizes between consecutive footprints."""
        if self.n_footprints <= 1:
            return np.array([], dtype=np.int64)
        return self.footprint_starts[1:] - self.footprint_ends[:-1]

    def gap_to_next(self) -> np.ndarray:
        """Gap after each footprint (last footprint gets gap to read end)."""
        if self.n_footprints == 0:
            return np.array([], dtype=np.int64)
        ends = self.footprint_ends
        gaps = np.zeros(self.n_footprints, dtype=np.int64)
        if self.n_footprints > 1:
            gaps[:-1] = self.footprint_starts[1:] - ends[:-1]
        gaps[-1] = self.length - ends[-1]
        return np.maximum(gaps, 0)

    def protamine_fraction(self) -> float:
        """Fraction of read covered by >200bp footprints."""
        mask = self.footprint_sizes > 200
        if self.length == 0:
            return 0.0
        return float(self.footprint_sizes[mask].sum()) / self.length

    def large_protamine_fraction(self) -> float:
        """Fraction of read covered by >2kb footprints (preprint heuristic)."""
        mask = self.footprint_sizes > 2000
        if self.length == 0:
            return 0.0
        return float(self.footprint_sizes[mask].sum()) / self.length


def _parse_bed12_row(row, chrom_col, start_col, end_col, name_col,
                     sizes_col, starts_col, min_read_length, valid_chroms):
    """Parse one row of BED12 into a FiberRead. Returns None if filtered."""
    chrom = str(row[chrom_col])
    if valid_chroms and chrom not in valid_chroms:
        return None
    try:
        start = int(row[start_col])
        end = int(row[end_col])
    except (ValueError, TypeError):
        return None
    if end - start < min_read_length:
        return None

    starts_str = str(row[starts_col])
    sizes_str = str(row[sizes_col])
    try:
        starts = np.array([int(x) for x in starts_str.split(',') if x.strip()], dtype=np.int64)
        sizes = np.array([int(x) for x in sizes_str.split(',') if x.strip()], dtype=np.int64)
    except ValueError:
        return None

    if len(starts) != len(sizes) or len(starts) == 0:
        return None
    if len(starts) > 1 and not np.all(np.diff(starts) >= 0):
        return None

    return FiberRead(
        chrom=chrom,
        start=start,
        end=end,
        read_id=str(row[name_col]),
        footprint_starts=starts,
        footprint_sizes=sizes,
    )


def _iter_bigbed(filepath: Path, min_read_length: int, max_reads: Optional[int],
                 valid_chroms: Optional[set]) -> Iterator[FiberRead]:
    """Stream a BigBed file via bigBedToBed, yielding FiberRead objects.

    BigBed is standard BED12:
      col0=chrom, col1=start, col2=end, col3=name, col4=score, col5=strand,
      col6=thickStart, col7=thickEnd, col8=itemRgb, col9=blockCount,
      col10=blockSizes, col11=blockStarts
    """
    cmd = [_find_tool('bigBedToBed'), str(filepath), 'stdout']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=65536)
    yielded = 0
    try:
        for line in io.TextIOWrapper(proc.stdout, encoding='utf-8'):
            if line.startswith('#') or not line.strip():
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 12:
                continue
            read = _parse_bed12_row(
                fields, 0, 1, 2, 3,
                sizes_col=10, starts_col=11,
                min_read_length=min_read_length,
                valid_chroms=valid_chroms,
            )
            if read is None:
                continue
            yield read
            yielded += 1
            if max_reads and yielded >= max_reads:
                proc.terminate()
                return
    finally:
        proc.wait()


def _iter_legacy_bed(filepath: Path, min_read_length: int, max_reads: Optional[int],
                     valid_chroms: Optional[set], chunksize: int = 100000) -> Iterator[FiberRead]:
    """Stream a legacy 10-column FiberHMM BED file.

    Legacy format:
      col0=chrom, col1=start, col2=end, col3=name, col4=thickStart,
      col5=thickEnd, col6=itemRgb, col7=blockCount,
      col8=blockStarts, col9=blockSizes   (NOTE: starts before sizes)

    GM12878 has a header row; others don't.
    """
    # Detect header
    with open(filepath, 'r') as f:
        first_line = f.readline()
    has_header = first_line.lower().startswith('chrom') or first_line.startswith('#')
    skiprows = 1 if has_header else 0

    yielded = 0
    for chunk in pd.read_csv(
        filepath, sep='\t', header=None, skiprows=skiprows,
        usecols=[0, 1, 2, 3, 8, 9],
        chunksize=chunksize, dtype={0: str}, low_memory=False,
    ):
        for _, row in chunk.iterrows():
            read = _parse_bed12_row(
                row, 0, 1, 2, 3,
                sizes_col=9, starts_col=8,   # legacy: col8=starts, col9=sizes
                min_read_length=min_read_length,
                valid_chroms=valid_chroms,
            )
            if read is None:
                continue
            yield read
            yielded += 1
            if max_reads and yielded >= max_reads:
                return


def parse_bed_file(
    filepath: str | Path,
    min_read_length: int = 10000,
    max_reads: Optional[int] = None,
    valid_chroms: Optional[set] = None,
    chunksize: int = 100000,
) -> Iterator[FiberRead]:
    """Parse a FiberHMM footprint file and yield FiberRead objects.

    Accepts both BigBed (.bb) and legacy BED (.bed) formats.
    """
    filepath = Path(filepath)
    if filepath.suffix == '.bb':
        yield from _iter_bigbed(filepath, min_read_length, max_reads, valid_chroms)
    else:
        yield from _iter_legacy_bed(filepath, min_read_length, max_reads, valid_chroms, chunksize)


def load_reads_as_list(
    filepath: str | Path,
    min_read_length: int = 10000,
    max_reads: Optional[int] = None,
    valid_chroms: Optional[set] = None,
) -> list[FiberRead]:
    """Load all reads from a BED or BigBed file into a list."""
    return list(parse_bed_file(filepath, min_read_length, max_reads, valid_chroms))


# Standard human autosomes + X
HUMAN_CHROMS = {f'chr{i}' for i in range(1, 23)} | {'chrX'}
