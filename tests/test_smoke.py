import json

import numpy as np

from fiberformer.checkpoint import DEFAULT_CHECKPOINT_PATH, load_model
from fiberformer.data.loader import FiberRead, load_reads_as_list
from fiberformer.embedding.embed import embed_reads, save_result
from fiberformer.embedding.embed_umap import main as umap_main


def synthetic_reads():
    return [
        FiberRead(
            chrom="chr1",
            start=1_000,
            end=12_100,
            read_id="synthetic-1",
            footprint_starts=np.array([20, 250, 510, 850], dtype=np.int64),
            footprint_sizes=np.array([147, 35, 260, 145], dtype=np.int64),
        ),
        FiberRead(
            chrom="chr2",
            start=2_000,
            end=14_300,
            read_id="synthetic-2",
            footprint_starts=np.array([10, 300, 750, 1_600, 5_200], dtype=np.int64),
            footprint_sizes=np.array([60, 146, 500, 150, 4_000], dtype=np.int64),
        ),
    ]


def test_bundled_checkpoint_embedding_smoke(tmp_path):
    model, config, checkpoint, device = load_model(device="cpu")
    result = embed_reads(
        model,
        synthetic_reads(),
        device,
        config=config,
        batch_size=2,
        compute_reconstruction=True,
        seed=1,
    )

    assert checkpoint["epoch"] == 93
    assert result.cls_norm.shape == (2, 256)
    assert result.cls_embeddings.shape == (2, 256)
    assert result.sperm_scores.shape == (2,)
    assert result.reconstruction_nll.shape == (2,)
    assert np.all(np.isfinite(result.reconstruction_nll))
    assert np.all(result.reconstruction_perplexity > 0)
    np.testing.assert_allclose(
        np.linalg.norm(result.cls_norm, axis=1), np.ones(2), atol=1e-5
    )
    assert np.all((result.sperm_scores >= 0) & (result.sperm_scores <= 1))

    output = tmp_path / "embedding"
    save_result(
        result,
        output,
        input_path=tmp_path / "synthetic.bed",
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        checkpoint=checkpoint,
        settings={"smoke_test": True},
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["n_reads"] == 2
    assert manifest["embedding_dimensions"] == 256
    assert len(manifest["checkpoint_sha256"]) == 64
    assert (output / "cls_norm.npy").is_file()
    assert (output / "sperm_scores.npy").is_file()
    assert (output / "reconstruction_nll.npy").is_file()


def test_legacy_bed_fixture():
    reads = load_reads_as_list(
        "tests/data/synthetic_legacy.bed",
        min_read_length=10_000,
    )
    assert [read.read_id for read in reads] == ["synthetic-bed-1", "synthetic-bed-2"]
    assert reads[0].footprint_sizes.tolist() == [147, 50, 300, 145]


def test_umap_cli_smoke(tmp_path):
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(30, 256)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(tmp_path / "cls_norm.npy", vectors)

    umap_main([
        str(tmp_path),
        "--n-neighbors", "5",
        "--min-dist", "0.2",
        "--no-plot",
    ])

    coordinates = np.load(tmp_path / "umap_2d.npy")
    assert coordinates.shape == (30, 2)
    assert (tmp_path / "umap_model.joblib").is_file()
