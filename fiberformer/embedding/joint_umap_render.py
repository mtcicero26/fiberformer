"""Render density comparisons of the joint UMAP fit vs the per-sample UMAP fit
for PS30065 + m6a_200.

Uses the 1M fitted training embeddings directly (500K per sample). Full-cohort
projection through umap-learn/pynndescent is slow off large read cohorts, and
500K per sample is more than enough for density plotting.

Also loads pre-computed per-sample umap coords for the same reads (matched by
fit_indices) as a comparison reference.
"""
import sys, pickle, warnings
warnings.filterwarnings('ignore')
sys.modules['tensorflow'] = None
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(Path(__file__).parent))
from _full_common import draw_shape_bg, savefig_dual

OUT_DIR = PROJECT_DIR / 'outputs/embeddings/joint_ps30065_m6a200'
FIG_DIR = PROJECT_DIR / 'outputs/figures/joint_ps30065_m6a200'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading joint UMAP embedding...", flush=True)
    with open(OUT_DIR / 'umap_model.pkl', 'rb') as f:
        pkg = pickle.load(f)
    umap = pkg['umap']; flip = pkg['flip']
    train_embed = np.asarray(umap.embedding_, dtype=np.float32).copy()
    if flip: train_embed[:, 0] = -train_embed[:, 0]
    fi = np.load(OUT_DIR / 'fit_indices.npz')
    idx_ps = fi['ps30065']; idx_m6 = fi['m6a_200']
    n_ps = len(idx_ps); n_m6 = len(idx_m6)
    u_ps_joint = train_embed[:n_ps]
    u_m6_joint = train_embed[n_ps:n_ps + n_m6]
    print(f"  PS30065 joint {u_ps_joint.shape}, m6a_200 joint {u_m6_joint.shape}",
          flush=True)

    # Save split embeddings for later
    np.save(OUT_DIR / 'PS30065_umap_train.npy', u_ps_joint)
    np.save(OUT_DIR / 'm6a_200_umap_train.npy', u_m6_joint)
    np.savez(OUT_DIR / 'joint_umap_lims.npz',
             xlim=(float(train_embed[:, 0].min()) - 1,
                   float(train_embed[:, 0].max()) + 1),
             ylim=(float(train_embed[:, 1].min()) - 1,
                   float(train_embed[:, 1].max()) + 1))

    # per-sample coords for the same training indices
    v72_ps = np.load(PROJECT_DIR /
        'outputs/embeddings/testes/PS30065/umap_2d.npy',
        mmap_mode='r')
    v72_m6 = np.load(PROJECT_DIR /
        'outputs/embeddings/sperm/m6a_200/umap_2d.npy',
        mmap_mode='r')
    u_ps_v72 = np.array(v72_ps[idx_ps])
    u_m6_v72 = np.array(v72_m6[idx_m6])

    def _bin(u, xlim, ylim, bins=(220, 220), sigma=1.2):
        h = np.histogram2d(u[:, 1], u[:, 0], bins=bins,
                            range=[ylim, xlim])[0]
        return gaussian_filter(h, sigma)

    # Joint UMAP axis range
    j_xlim = (float(train_embed[:, 0].min()) - 1,
              float(train_embed[:, 0].max()) + 1)
    j_ylim = (float(train_embed[:, 1].min()) - 1,
              float(train_embed[:, 1].max()) + 1)

    # per-sample canonical axis range
    from _full_common import CANONICAL_UMAP_LIMS
    v_xlim, v_ylim = CANONICAL_UMAP_LIMS

    # ---------------- Figure: 2x2 density comparison ----------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax_row, (label, u_joint, u_v72) in enumerate([
        ('PS30065 (n=500K training subset)', u_ps_joint, u_ps_v72),
        ('m6a_200 (n=500K training subset)', u_m6_joint, u_m6_v72),
    ]):
        # per-sample side
        ax = axes[ax_row, 0]
        draw_shape_bg(ax, u_ps_v72 if ax_row == 0 else np.vstack([u_ps_v72, u_m6_v72]),
                       v_xlim, v_ylim)
        h = _bin(u_v72, v_xlim, v_ylim)
        vmax = float(np.percentile(h[h > 0], 99)) if (h > 0).any() else 1.0
        im = ax.imshow(np.ma.masked_where(h <= 0, h), origin='lower',
                        extent=[*v_xlim, *v_ylim], aspect='auto',
                        cmap='magma', vmin=0, vmax=vmax, zorder=1)
        plt.colorbar(im, ax=ax, fraction=0.045, label='reads / bin')
        ax.set_xlim(*v_xlim); ax.set_ylim(*v_ylim)
        ax.set_title(f'per-sample UMAP — {label}',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')

        # joint side
        ax = axes[ax_row, 1]
        draw_shape_bg(ax, train_embed, j_xlim, j_ylim)
        h = _bin(u_joint, j_xlim, j_ylim)
        vmax = float(np.percentile(h[h > 0], 99)) if (h > 0).any() else 1.0
        im = ax.imshow(np.ma.masked_where(h <= 0, h), origin='lower',
                        extent=[*j_xlim, *j_ylim], aspect='auto',
                        cmap='magma', vmin=0, vmax=vmax, zorder=1)
        plt.colorbar(im, ax=ax, fraction=0.045, label='reads / bin')
        ax.set_xlim(*j_xlim); ax.set_ylim(*j_ylim)
        ax.set_title(f'joint-fit UMAP — {label}', fontsize=11, fontweight='bold')
        ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')

    fig.suptitle('per-sample (PS30065-only fit) vs joint-fit (PS30065+m6a_200) UMAPs',
                 fontsize=13, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    savefig_dual(fig, str(FIG_DIR / 'joint_vs_v72_density'))
    print(f"saved joint_vs_v72_density.png/.pdf", flush=True)


if __name__ == '__main__':
    main()
