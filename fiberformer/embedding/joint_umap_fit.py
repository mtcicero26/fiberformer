"""Fit UMAP jointly on PS30065 + m6a_200 (all reads).

Fits on a stratified subsample (~1M each), then transforms all reads for both
samples. Saves the fitted UMAP model + per-sample projections.

Outputs:
  outputs/embeddings/joint_ps30065_m6a200/
    umap_model.pkl
    fit_indices.npz             (which reads were used for training)
    PS30065_umap.npy            (all 8M PS30065 reads projected)
    m6a_200_umap.npy            (all 4.5M m6a_200 reads projected)

Uses umap-learn on CPU. Subsample size chosen so fit completes in ~30 min.
"""
import sys, time, pickle, warnings, gzip
warnings.filterwarnings('ignore')
sys.modules['tensorflow'] = None
from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

TESTIS_CLS = PROJECT_DIR / 'outputs/embeddings/all_testes_v7_2_full/PS30065/cls_norm.npy'
SPERM_CLS  = PROJECT_DIR / 'outputs/embeddings/sperm_motility_v7_2_full/m6a_200/cls_norm.npy'
OUT_DIR    = PROJECT_DIR / 'outputs/embeddings/joint_ps30065_m6a200'
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_TRAIN_PER_SAMPLE = 500_000     # 1M total training reads
CHUNK = 200_000                   # transform batch


def main():
    print("Loading PS30065 + m6a_200 cls_norm...", flush=True)
    t0 = time.time()
    cls_ps = np.load(TESTIS_CLS, mmap_mode='r').astype(np.float32)
    cls_m6 = np.load(SPERM_CLS, mmap_mode='r').astype(np.float32)
    N_ps, N_m6 = len(cls_ps), len(cls_m6)
    print(f"  PS30065 {cls_ps.shape}, m6a_200 {cls_m6.shape} loaded in {time.time()-t0:.0f}s",
          flush=True)

    rng = np.random.default_rng(42)
    idx_ps = rng.choice(N_ps, size=min(N_TRAIN_PER_SAMPLE, N_ps), replace=False)
    idx_m6 = rng.choice(N_m6, size=min(N_TRAIN_PER_SAMPLE, N_m6), replace=False)
    idx_ps.sort(); idx_m6.sort()
    train = np.vstack([cls_ps[idx_ps], cls_m6[idx_m6]])
    print(f"Training set: {train.shape}", flush=True)
    np.savez(OUT_DIR / 'fit_indices.npz', ps30065=idx_ps, m6a_200=idx_m6)

    print("Fitting UMAP (metric=cosine, n_neighbors=25, min_dist=0.05)...",
          flush=True)
    from umap.umap_ import UMAP
    t0 = time.time()
    umap = UMAP(n_components=2, metric='cosine',
                 n_neighbors=25, min_dist=0.05,
                 random_state=42, low_memory=True, verbose=True)
    train_embed = umap.fit_transform(train)
    print(f"  UMAP fit in {time.time()-t0:.0f}s. train_embed shape: {train_embed.shape}",
          flush=True)

    # Flip UMAP1 so somatic-left / sperm-right (match paper convention)
    # Determine flip by comparing PS30065 vs m6a_200 median UMAP1
    n_train_ps = len(idx_ps)
    med_ps = float(np.median(train_embed[:n_train_ps, 0]))
    med_m6 = float(np.median(train_embed[n_train_ps:, 0]))
    if med_m6 < med_ps:
        print(f"  sperm side is LEFT (m6a_200 median {med_m6:.2f} < PS30065 {med_ps:.2f}); flipping",
              flush=True)
        flip = True
    else:
        flip = False

    with open(OUT_DIR / 'umap_model.pkl', 'wb') as f:
        pickle.dump({'umap': umap, 'flip': flip}, f)
    print(f"  saved umap_model.pkl", flush=True)

    def _transform(cls, name):
        N = len(cls)
        out = np.empty((N, 2), dtype=np.float32)
        t0 = time.time()
        for i in range(0, N, CHUNK):
            j = min(i + CHUNK, N)
            out[i:j] = umap.transform(cls[i:j]).astype(np.float32)
            elapsed = time.time() - t0
            rem = (N - j) / max(j, 1) * elapsed
            print(f'    {name} transform {j:,}/{N:,}  ({elapsed:.0f}s, ETA {rem:.0f}s)',
                  flush=True)
        if flip:
            out[:, 0] = -out[:, 0]
        return out

    print("Transforming all PS30065 reads...", flush=True)
    u_ps = _transform(cls_ps, 'PS30065')
    np.save(OUT_DIR / 'PS30065_umap.npy', u_ps)

    print("Transforming all m6a_200 reads...", flush=True)
    u_m6 = _transform(cls_m6, 'm6a_200')
    np.save(OUT_DIR / 'm6a_200_umap.npy', u_m6)

    print(f"\nDone. Coordinates saved to {OUT_DIR}", flush=True)


if __name__ == '__main__':
    main()
