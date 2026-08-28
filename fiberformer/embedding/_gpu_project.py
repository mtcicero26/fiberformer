"""Fast cosine kNN projector for the full-embed UMAP + pt.

Two backends:
  - gpu_cosine_knn_project: torch batched matmul + topk (fastest on 5090)
  - cpu_ann_cosine_project: pynndescent approximate cosine kNN
    (falls back cleanly when the GPU is contended by another workload)

sklearn's cosine KNeighborsRegressor is brute-force on the CPU and takes
~10 min per 500K queries on our workstation. Both backends here are
orders of magnitude faster.
"""
import numpy as np
import torch


def gpu_cosine_knn_project(ref_norm, ref_coords, queries, k=15, batch=4096,
                            device=None, verbose=False):
    """Project unit-normalized query vectors onto ref_coords via cosine kNN.

    ref_norm:    (N_ref, D) float32, unit-normalized rows.
    ref_coords:  (N_ref, C) float32 or float64.
    queries:     (N_q,   D) float32 (unit-normalized recommended).
    Returns (N_q, C) float32.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tref = torch.as_tensor(ref_norm, dtype=torch.float32, device=device)
    tcoord = torch.as_tensor(ref_coords, dtype=torch.float32, device=device)
    N_q = queries.shape[0]
    C = tcoord.shape[1]
    out = np.empty((N_q, C), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, N_q, batch):
            j = min(i + batch, N_q)
            q = torch.as_tensor(queries[i:j], dtype=torch.float32, device=device)
            # ensure unit norm (should already be)
            qn = q.norm(dim=1, keepdim=True).clamp(min=1e-9)
            q = q / qn
            sim = q @ tref.T           # cosine similarity
            vals, idx = sim.topk(k, dim=1)
            d = 1.0 - vals              # cosine distance
            w = 1.0 / (d + 1e-8)
            w = w / w.sum(dim=1, keepdim=True)
            gather = tcoord[idx]        # (BS, k, C)
            proj = (gather * w.unsqueeze(-1)).sum(dim=1)
            out[i:j] = proj.cpu().numpy()
            if verbose and (i // batch) % 8 == 0:
                pct = 100.0 * j / N_q
                print(f'    gpu-knn  {j:,}/{N_q:,}  ({pct:.1f}%)', flush=True)
    return out


class CosineANNIndex:
    """Pynndescent-backed approximate cosine kNN over unit vectors.

    Fit once on reference; project queries in chunks. Suitable when the
    GPU is contended or unavailable.
    """
    def __init__(self, ref_norm, ref_coords, n_trees=None, n_neighbors_build=30):
        from pynndescent import NNDescent
        self.ref_coords = np.asarray(ref_coords, dtype=np.float32)
        self.ref_norm = np.ascontiguousarray(ref_norm, dtype=np.float32)
        self.index = NNDescent(self.ref_norm, metric='cosine',
                                n_neighbors=n_neighbors_build,
                                n_trees=n_trees, low_memory=True,
                                random_state=42)
        # Trigger prepare so first query is fast.
        _ = self.index.prepare()

    def project(self, queries, k=15, batch=200_000, verbose=False,
                 return_distance=False):
        import time
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        N_q = queries.shape[0]
        C = self.ref_coords.shape[1]
        out = np.empty((N_q, C), dtype=np.float32)
        med_dist = np.empty(N_q, dtype=np.float32) if return_distance else None
        t0 = time.time()
        for i in range(0, N_q, batch):
            j = min(i + batch, N_q)
            idx, dist = self.index.query(queries[i:j], k=k)
            if k > 1:
                w = 1.0 / (dist + 1e-8)
                w /= w.sum(axis=1, keepdims=True)
                gather = self.ref_coords[idx]           # (BS, k, C)
                out[i:j] = (gather * w[:, :, None]).sum(axis=1).astype(np.float32)
            else:
                # k=1: snap to nearest ref point exactly
                out[i:j] = self.ref_coords[idx[:, 0]].astype(np.float32)
            if return_distance:
                med_dist[i:j] = dist.mean(axis=1)
            if verbose:
                elapsed = time.time() - t0
                rem = (N_q - j) / max(j, 1) * elapsed
                print(f'    ann-cos  {j:,}/{N_q:,}  '
                      f'({100.0*j/N_q:.1f}%, {elapsed:.0f}s, ETA {rem:.0f}s)',
                      flush=True)
        if return_distance:
            return out, med_dist
        return out


def cpu_ann_cosine_project(ref_norm, ref_coords, queries, k=15,
                            batch=200_000, verbose=False):
    """Convenience one-shot wrapper around CosineANNIndex."""
    idx = CosineANNIndex(ref_norm, ref_coords)
    return idx.project(queries, k=k, batch=batch, verbose=verbose)
