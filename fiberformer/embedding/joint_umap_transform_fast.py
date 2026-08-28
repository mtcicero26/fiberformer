"""Fast transform stage for the joint UMAP.

umap-learn's .transform() is prohibitively slow for millions of queries. This
uses pynndescent kNN on the 1M training embeddings instead: for each query,
find its nearest neighbors among the training cls vectors and weight-average
their fitted UMAP coords. Same principle as the paper's original 'Option C'
projection.
"""
import sys, time, pickle, warnings
warnings.filterwarnings('ignore')
sys.modules['tensorflow'] = None
from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(Path(__file__).parent))
from _gpu_project import CosineANNIndex

TESTIS_CLS = PROJECT_DIR / 'outputs/embeddings/testes/PS30065/cls_norm.npy'
SPERM_CLS  = PROJECT_DIR / 'outputs/embeddings/sperm/m6a_200/cls_norm.npy'
OUT_DIR    = PROJECT_DIR / 'outputs/embeddings/joint_ps30065_m6a200'


def main():
    print("Loading UMAP model + training data...", flush=True)
    t0 = time.time()
    with open(OUT_DIR / 'umap_model.pkl', 'rb') as f:
        pkg = pickle.load(f)
    umap = pkg['umap']; flip = pkg['flip']
    train_embed = np.asarray(umap.embedding_, dtype=np.float32)
    if flip:
        # embedding_ is unflipped; store the flip so we apply it consistently
        train_embed = train_embed.copy()
        train_embed[:, 0] = -train_embed[:, 0]
    print(f"  train_embed {train_embed.shape}, flip={flip}", flush=True)

    fi = np.load(OUT_DIR / 'fit_indices.npz')
    idx_ps = fi['ps30065']; idx_m6 = fi['m6a_200']

    # Sequential chunked read + gather. Random access via mmap fancy indexing
    def sequential_gather(path, needed_idx):
        t0 = time.time()
        needed_sorted = np.sort(needed_idx)
        mm = np.load(path, mmap_mode='r')
        N = len(mm); D = mm.shape[1]
        out = np.empty((len(needed_sorted), D), dtype=np.float32)
        CHUNK = 500_000
        needed_ptr = 0
        for i in range(0, N, CHUNK):
            j = min(i + CHUNK, N)
            # copy chunk into memory
            chunk = np.array(mm[i:j], dtype=np.float32)
            # gather any needed rows falling in [i, j)
            while (needed_ptr < len(needed_sorted)
                   and needed_sorted[needed_ptr] < j):
                out[needed_ptr] = chunk[needed_sorted[needed_ptr] - i]
                needed_ptr += 1
            print(f'    ...{j:,}/{N:,}  gathered {needed_ptr:,}/{len(needed_sorted):,}'
                  f'  ({time.time()-t0:.0f}s)', flush=True)
            if needed_ptr >= len(needed_sorted):
                break
        del mm
        return out

    print("Sequential-gather PS30065 training rows...", flush=True)
    train_ps = sequential_gather(TESTIS_CLS, idx_ps)
    print(f"  {train_ps.shape}", flush=True)

    print("Sequential-gather m6a_200 training rows...", flush=True)
    train_m6 = sequential_gather(SPERM_CLS, idx_m6)
    print(f"  {train_m6.shape}", flush=True)

    train_cls = np.vstack([train_ps, train_m6])
    del train_ps, train_m6
    print(f"  train_cls {train_cls.shape}", flush=True)

    print("Building ANN index over 1M train cls...", flush=True)
    t0 = time.time()
    ann = CosineANNIndex(train_cls, train_embed)
    print(f"  ANN built in {time.time()-t0:.0f}s", flush=True)

    def _project(cls_path, name):
        print(f"Loading {name} cls_norm...", flush=True)
        t0 = time.time()
        cls_mm = np.load(cls_path, mmap_mode='r')
        N = len(cls_mm)
        # Copy in chunks to avoid one giant mmap-to-array copy
        print(f"Projecting {N:,} {name} reads...", flush=True)
        t0 = time.time()
        out = np.empty((N, 2), dtype=np.float32)
        BATCH = 200_000
        for i in range(0, N, BATCH):
            j = min(i + BATCH, N)
            chunk = np.ascontiguousarray(cls_mm[i:j], dtype=np.float32)
            out[i:j] = ann.project(chunk, k=15, batch=BATCH, verbose=False)
            elapsed = time.time() - t0
            rem = (N - j) / max(j, 1) * elapsed
            print(f'    {name} {j:,}/{N:,}  ({elapsed:.0f}s, ETA {rem:.0f}s)',
                  flush=True)
        del cls_mm
        print(f"  {name} projected in {time.time()-t0:.0f}s", flush=True)
        np.save(OUT_DIR / f'{name}_umap.npy', out)

    _project(TESTIS_CLS, 'PS30065')
    _project(SPERM_CLS, 'm6a_200')
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
