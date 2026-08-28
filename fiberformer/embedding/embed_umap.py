"""Fit a two-dimensional UMAP to FiberFormer read embeddings."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit UMAP to a fiberformer-embed output directory"
    )
    parser.add_argument("embedding_dir", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="output directory (default: the embedding directory)",
    )
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.3)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--point-size", type=float, default=1.0)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # Numba normally caches compiled UMAP kernels beside its installed source.
    # That location is often read-only in shared conda/module environments.
    numba_cache = Path(tempfile.gettempdir()) / "fiberformer-numba-cache"
    numba_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache))

    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    embedding_dir = args.embedding_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None else embedding_dir
    )
    input_path = embedding_dir / "cls_norm.npy"
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Expected normalized embeddings at {input_path}; run fiberformer-embed first"
        )

    embeddings = np.load(input_path, mmap_mode="r")
    if embeddings.ndim != 2 or embeddings.shape[1] != 256:
        raise ValueError(
            f"Expected an (n_reads, 256) array, found {embeddings.shape}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fitting UMAP to {len(embeddings):,} embeddings ...", flush=True)
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
    )
    coordinates = reducer.fit_transform(np.asarray(embeddings))
    np.save(output_dir / "umap_2d.npy", coordinates.astype(np.float32))
    joblib.dump(reducer, output_dir / "umap_model.joblib")

    if not args.no_plot:
        overlays = [
            ("sperm_scores.npy", "Head A sperm score", "viridis"),
            ("protamine_fractions.npy", "Per-read protamine fraction", "magma"),
            ("reconstruction_nll.npy", "Head B reconstruction NLL", "plasma"),
        ]
        available = [item for item in overlays if (embedding_dir / item[0]).is_file()]
        if not available:
            available = [(None, "FiberFormer UMAP", None)]
        figure, axes = plt.subplots(
            1, len(available), figsize=(6 * len(available), 5), squeeze=False
        )
        for axis, (filename, label, colormap) in zip(axes[0], available):
            if filename is None:
                axis.scatter(
                    coordinates[:, 0], coordinates[:, 1],
                    s=args.point_size, alpha=0.5, rasterized=True,
                )
            else:
                values = np.load(embedding_dir / filename)
                if len(values) != len(coordinates):
                    raise ValueError(
                        f"{filename} has {len(values):,} rows; expected {len(coordinates):,}"
                    )
                scatter = axis.scatter(
                    coordinates[:, 0], coordinates[:, 1], c=values,
                    cmap=colormap, s=args.point_size, alpha=0.6, rasterized=True,
                )
                figure.colorbar(scatter, ax=axis, label=label)
            axis.set(title=label, xlabel="UMAP1", ylabel="UMAP2")
        figure.tight_layout()
        figure.savefig(output_dir / "umap.png", dpi=200, bbox_inches="tight")
        plt.close(figure)

    print(f"Saved UMAP outputs to {output_dir}")


if __name__ == "__main__":
    main()
