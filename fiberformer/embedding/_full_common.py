"""Shared loader / helper for the full-embed downstream re-run figures.

Datasets (from full_rerun_master.py):
  PS30065_full     — single testis, 8M reads
  all_testis_full  — all 4 testis pooled
  m6a_200_full     — single sperm
  all_sperm_full   — all 18 sperm pooled

Each pooled dataset lives at:
  outputs/embeddings/pooled_v7_2_full/<name>/
with arrays: chroms, starts, ends, lengths, n_fps, prot_fracs, perplexity,
scores, pseudotime, umap_2d, sample_id, sample_names,
fire_has, fire_open, fire_tf, fire_nuc, fire_prot.

Annotation overlaps live at:
  outputs/embeddings/region_overlays_v7_2_full/<name>.npz

Figures are written to outputs/figures/v7/full/<name>/.

Sector projection: nearest-centroid on the pooled umap_2d, using the
PS30065 mapq60 k=6 centroids with UMAP1 flipped to match the pipeline's
convention (somatic left → sperm right).
"""
from pathlib import Path
import sys, warnings, gzip
warnings.filterwarnings('ignore')
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(Path(__file__).parent))

REF_DIR = PROJECT_DIR.parent / 'Reference'
REV_DIR = PROJECT_DIR.parent / 'Revisions'
V7_2_TESTES_REF = PROJECT_DIR / 'outputs' / 'embeddings' / 'all_testes_v7_2' / 'PS30065'
V7_2_UMAP = PROJECT_DIR / 'outputs' / 'embeddings' / 'umap_v7_2_ps30065_mapq60'
SECTOR_DIR = PROJECT_DIR / 'outputs' / 'embeddings' / 'sectors_ps30065_mapq60_v7_2_k6'
POOLED = PROJECT_DIR / 'outputs' / 'embeddings' / 'pooled_v7_2_full'
CACHE_DIR = PROJECT_DIR / 'outputs' / 'embeddings' / 'region_overlays_v7_2_full'
FIG_ROOT = PROJECT_DIR / 'outputs' / 'figures' / 'v7' / 'full'

TESTIS_SAMPLES = ['PS30065', 'PS30135', 'PS30330', 'PS30549']
SPERM_SAMPLES = [
    'm6a_200', 'PS00335', 'PS00750', 'PS00751', 'PS00752', 'PS00753',
    'PS00754', 'PS00755', 'PS00876', 'PS00877', 'PS00878', 'PS00879',
    'PS00880', 'PS00881', 'PS00882', 'PS00883', 'PS00884', 'PS00885',
]

DATASETS = {
    'PS30065_full':    {'kind': 'testis', 'sources': ['PS30065']},
    'all_testis_full': {'kind': 'testis', 'sources': TESTIS_SAMPLES},
    'm6a_200_full':    {'kind': 'sperm',  'sources': ['m6a_200']},
    'all_sperm_full':  {'kind': 'sperm',  'sources': SPERM_SAMPLES},
}

K = 6


def dataset_kind(name):
    return DATASETS[name]['kind']


def dataset_sources(name):
    return list(DATASETS[name]['sources'])


def fig_dir(name):
    out = FIG_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_dataset(name, arrays=None, mmap=True):
    """Load pooled fields for a dataset by name.

    arrays: list of field names to load; None => all standard fields.
    Returns dict[str, np.ndarray].
    """
    if arrays is None:
        arrays = ['chroms', 'starts', 'ends', 'lengths', 'n_fps', 'prot_fracs',
                  'perplexity', 'scores', 'pseudotime', 'umap_2d', 'sample_id',
                  'sample_names', 'umap_dist']
    d = POOLED / name
    if not d.exists():
        raise FileNotFoundError(f"Pooled dataset not ready: {d}")
    out = {}
    for a in arrays:
        p = d / f'{a}.npy'
        if not p.exists():
            out[a] = None; continue
        m = 'r' if mmap and a != 'chroms' and a != 'sample_names' else None
        if a in ('chroms', 'sample_names'):
            out[a] = np.load(p, allow_pickle=True)
        else:
            out[a] = np.load(p, mmap_mode=m)
    return out


def ood_keep_mask(name, quantile=0.95):
    """Return a bool mask keeping the (1-quantile) reads with the smallest kNN
    cosine distance to the reference set. Reads with distance > pQ are the
    out-of-distribution ones that drive the projection streak artifact.

    Falls back to all-True if the distance array isn't cached yet.
    """
    d = POOLED / name
    p = d / 'umap_dist.npy'
    if not p.exists():
        n = np.load(d / 'chroms.npy', allow_pickle=True).shape[0]
        return np.ones(n, dtype=bool)
    dist = np.load(p)
    thr = float(np.quantile(dist, quantile))
    return dist < thr


def load_fire_state(name):
    d = POOLED / name
    out = {}
    for k in ('fire_has', 'fire_open', 'fire_tf', 'fire_nuc', 'fire_prot'):
        p = d / f'{k}.npy'
        out[k] = np.load(p) if p.exists() else None
    return out


def load_annotation_cache(name):
    p = CACHE_DIR / f'{name}.npz'
    if not p.exists():
        raise FileNotFoundError(f"Annotation cache not ready: {p}")
    z = np.load(p)
    return {k: z[k] for k in z.files}


def project_sectors(umap_2d, flip=True):
    """Project pooled umap_2d onto PS30065 k=6 sector centroids.

    The pipeline's umap_2d is FLIPPED on UMAP1 (somatic left → sperm right)
    but the stored centroids come from the ORIGINAL (unflipped) coords.
    We flip the centroids to match the pooled orientation, then apply
    nearest-centroid classification.
    """
    centroids = np.load(SECTOR_DIR / 'centroids.npy').copy()
    if flip:
        centroids[:, 0] = -centroids[:, 0]
    u = np.asarray(umap_2d, dtype=np.float32)
    d2 = ((u[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    return d2.argmin(axis=1).astype(np.int16), centroids


def sector_medians_prot(prot_fracs, sector_id):
    return np.array([float(np.median(prot_fracs[sector_id == s])) for s in range(K)])


def sector_palette_by_prot(prot_meds):
    """Delegates to v7_2_scope_regen.sector_palette_by_prot_rank (blue→green→orange)."""
    from v7_2_scope_regen import sector_palette_by_prot_rank
    return sector_palette_by_prot_rank(prot_meds)


# ---------------- annotation helpers ----------------

def load_bed_per_chrom(bed_path, expand_bp=0, skip_header=False):
    if not bed_path.exists(): return {}
    opener = gzip.open if str(bed_path).endswith('.gz') else open
    per = {}
    with opener(bed_path, 'rt') as f:
        first = True
        for line in f:
            if first and skip_header:
                first = False; continue
            first = False
            if line.startswith(('#', 'track', 'browser')): continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3: continue
            try:
                s = int(parts[1]) - expand_bp; e = int(parts[2]) + expand_bp
            except ValueError:
                continue
            per.setdefault(parts[0], []).append((max(0, s), e))
    out = {}
    for c, ivs in per.items():
        ivs.sort()
        m_s, m_e = [], []
        cs, ce = ivs[0]
        for s, e in ivs[1:]:
            if s <= ce: ce = max(ce, e)
            else: m_s.append(cs); m_e.append(ce); cs, ce = s, e
        m_s.append(cs); m_e.append(ce)
        out[c] = (np.array(m_s, dtype=np.int64), np.array(m_e, dtype=np.int64))
    return out


def add_bivalent_and_prom(overlaps):
    """Add composite annotations that combine already-cached booleans."""
    K4 = overlaps.get('K4me3'); K27 = overlaps.get('K27me3')
    if K4 is not None and K27 is not None:
        overlaps['bivalent'] = K4 & K27
        overlaps['K4_only']  = K4 & ~K27
        overlaps['K27_only'] = ~K4 & K27
    return overlaps


# ---------------- 2D binning helpers ----------------

def binned_density(x, y, mask, xlim, ylim, bins=(200, 200)):
    if mask.sum() == 0:
        return np.zeros(bins, dtype=np.float64)
    h, _, _ = np.histogram2d(y[mask], x[mask], bins=bins,
                              range=[ylim, xlim])
    return h


# Canonical axes: PS30065_full 0.5–99.5 percentile + 1 unit of padding on all
# sides. Used everywhere so sperm-only datasets show up against the same
# somatic→sperm reference frame as testis datasets, and testis panels are not
# pinched at the edges.
CANONICAL_UMAP_LIMS = ((-14.7, 1.8), (-1.2, 10.7))


def compute_umap_lims(u2d, pctl=(0.5, 99.5), canonical=True):
    """Return the canonical UMAP lims by default, so every dataset renders on
    the same axes as the testis reference. Pass canonical=False to recompute
    per-dataset percentiles (e.g. for a dataset-only zoom)."""
    if canonical:
        return CANONICAL_UMAP_LIMS
    xlo, xhi = np.percentile(u2d[:, 0], pctl)
    ylo, yhi = np.percentile(u2d[:, 1], pctl)
    return (float(xlo), float(xhi)), (float(ylo), float(yhi))


def draw_shape_bg(ax, u_all, xlim=None, ylim=None, bins=(220, 220),
                    sigma=1.2, cmap='Greys', alpha=0.35, log=True):
    """Render the overall UMAP shape as a grey backdrop so subset overlays sit
    on top of the manifold outline. Call this BEFORE the subset imshow.

    u_all is the pooled dataset's umap_2d (all reads). log=True compresses
    dynamic range so the outline is visible even where subset density peaks.
    """
    from scipy.ndimage import gaussian_filter
    from matplotlib import pyplot as plt
    if xlim is None or ylim is None:
        xlim, ylim = CANONICAL_UMAP_LIMS
    h = np.histogram2d(u_all[:, 1], u_all[:, 0], bins=bins,
                        range=[ylim, xlim])[0]
    h = gaussian_filter(h, sigma)
    if log:
        h = np.log1p(h)
    if not (h > 0).any():
        return
    vmax = float(np.percentile(h[h > 0], 99.9))
    ax.imshow(h, origin='lower', extent=[*xlim, *ylim], aspect='auto',
              cmap=cmap, vmin=0, vmax=vmax, alpha=alpha, zorder=0)


def savefig_dual(fig, path_no_ext):
    from matplotlib import pyplot as plt
    for ext in ('png', 'pdf'):
        fig.savefig(f'{path_no_ext}.{ext}',
                    dpi=150 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)


def paper_title_for(name):
    return {
        'PS30065_full':    'PS30065 (single testis donor)',
        'all_testis_full': 'All 4 testis donors pooled',
        'm6a_200_full':    'm6a_200 (single sperm donor)',
        'all_sperm_full':  'All 18 sperm donors pooled',
    }.get(name, name)
