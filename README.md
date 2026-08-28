# FiberFormer

FiberFormer is a transformer-based encoder for single-molecule Fiber-seq
footprint reads. It converts the ordered footprints on each read into a
256-dimensional representation of that read's chromatin state. The model was
developed to resolve intermediate states in the histone-to-protamine transition
that are not captured by bulk per-read summaries such as protamine fraction.

This repository contains the model, training and inference code, the reported
pretrained checkpoint, and selected paper-specific workflows. It does not
contain the underlying Fiber-seq datasets. Software version and citation
metadata are recorded in `pyproject.toml` and `CITATION.cff`.

## Start here

If you have a FiberHMM footprint BED or BigBed and want embeddings, the complete
workflow is:

```bash
git clone https://github.com/mtcicero26/fiberformer.git
cd fiberformer

# Create/activate a Python 3.10 environment first. For GPU use, install the
# PyTorch build appropriate for the workstation or cluster, then:
python -m pip install -e .

# Embed a reproducible reservoir sample of 10,000 eligible reads.
fiberformer-embed path/to/sample.bb outputs/embeddings/sample

# Fit and render a UMAP from the normalized read embeddings.
fiberformer-umap outputs/embeddings/sample
```

The pretrained checkpoint is already included at
`fiberformer/weights/best_model.pt`; there is no separate model download.

For a fast CPU-only check, use a small sample and skip the reconstruction head:

```bash
fiberformer-embed tests/data/synthetic_legacy.bed /tmp/fiberformer-smoke \
  --max-reads 20 --device cpu --skip-reconstruction
```

Use `--max-reads 0` to process every eligible read. A CUDA GPU is strongly
recommended for more than a small smoke test.

The checked-in test suite exercises checkpoint loading, CPU embedding,
serialization, the legacy BED parser, and UMAP:

```bash
python -m pytest -q
```

On the release-preparation host (Linux x86-64 under WSL2, Python 3.10.14,
PyTorch 2.10.0+cu128 with CPU execution), the three tests passed in 17.23 s and
the two-read command above completed in 1.90 s wall time on 22 August 2026.
These timings are hardware-specific and do not estimate full-dataset runtime.

## What problem does it solve?

Simple read-level features readily separate mature sperm and somatic Fiber-seq
reads. The difficult and biologically useful cases are testis reads partway
through chromatin remodeling. Two reads can have the same overall protamine
fraction while differing in whether the remaining non-protamine sequence is
densely nucleosomal, sparsely occupied, or organized into a different spatial
pattern.

FiberFormer learns from the full ordered footprint sequence. Reads with similar
sequence context and spatial organization are placed near one another in a
continuous embedding space. That embedding is the primary output used for UMAP,
trajectory analysis, clustering, and genomic-context overlays. The supervised
sperm/somatic score is an anchoring objective, not the main purpose of the
model.

## Inputs

### Supported footprint files

`fiberformer-embed` accepts either:

- A FiberHMM BigBed (`.bb`) using standard BED12 block columns.
- A legacy FiberHMM BED (`.bed`) using the older 10-column layout.

The standard BED12 interpretation is:

| Column | Meaning |
| --- | --- |
| 1-3 | chromosome, read start, read end |
| 4 | read identifier |
| 10 | footprint/block count |
| 11 | comma-separated footprint sizes |
| 12 | comma-separated footprint starts relative to the read start |

For a legacy 10-column BED, column 9 contains relative starts and column 10
contains sizes. See `fiberformer/data/loader.py` for the exact parser.

BigBed input requires UCSC `bigBedToBed` on `PATH` (or as an executable at
`~/bigBedToBed`). Plain BED input does not require UCSC tools.

### Default filtering

The generic embedding command:

- keeps reads at least 10 kb long;
- keeps `chr1`-`chr22` and `chrX` by default;
- takes a deterministic reservoir sample of 10,000 reads with seed 42;
- truncates each read to the first 128 footprint tokens.

Change these settings with `--min-read-length`, `--all-chromosomes`,
`--max-reads`, `--seed`, and `--max-tokens`. FiberFormer does not inspect BAM
MAPQ itself; perform read-quality and mapping filters before producing the
footprint file. Use `--all-chromosomes` for non-human genomes or to retain chrY.

### Footprint tokenization

Each FiberHMM footprint becomes one token:

- subnucleosomal: less than 90 bp;
- nucleosome-sized: 90-200 bp;
- protamine-sized: greater than 200 bp.

Each token carries its type, `log10(size_bp)`, `log10(gap_to_next_bp + 1)`,
first/last flags, and its offset from the beginning of the read. `[CLS]` is
prepended at position zero. The pretrained model uses physical, uncapped sizes
at tokenization followed by the same 9 kb inference clamp used during training.

These size classes are model features derived from FiberHMM footprints; the
Head A score should not be interpreted as an independently calibrated clinical
or diagnostic measurement.

## Embedding a new sample

The supported command is:

```bash
fiberformer-embed INPUT.{bb,bed} OUTPUT_DIRECTORY [options]
```

Common examples:

```bash
# Default: reservoir-sample 10,000 human autosomal/chrX reads.
fiberformer-embed data/bigbed/new_sample.bb outputs/embeddings/new_sample

# Embed all eligible reads on GPU, 1,024 reads per inference batch.
fiberformer-embed data/bigbed/new_sample.bb outputs/embeddings/new_sample_all \
  --max-reads 0 --device cuda --batch-size 1024

# Include every chromosome and use a non-default checkpoint.
fiberformer-embed data/bigbed/new_sample.bb outputs/embeddings/new_sample \
  --all-chromosomes --checkpoint path/to/checkpoint.pt

# Skip masked reconstruction when only the 256-D embedding and anchor score
# are needed. This roughly halves model inference work.
fiberformer-embed data/bigbed/new_sample.bb outputs/embeddings/new_sample \
  --skip-reconstruction
```

Run `fiberformer-embed --help` for the full option list.

### Output files

Each output directory contains NumPy arrays in the same read order:

| File | Shape | Description |
| --- | --- | --- |
| `cls_norm.npy` | `(n, 256)` | L2-normalized read embedding; preferred for cosine k-NN and UMAP |
| `cls_embeddings.npy` | `(n, 256)` | Unnormalized pooled read embedding |
| `sperm_scores.npy` | `(n,)` | sigmoid-transformed Head A sperm-vs-somatic anchor score |
| `reconstruction_nll.npy` | `(n,)` | mean joint Head B reconstruction negative log-likelihood on masked tokens |
| `reconstruction_perplexity.npy` | `(n,)` | `exp(reconstruction_nll)`; can be very large for out-of-distribution reads |
| `read_ids.npy` | `(n,)` | source read identifiers |
| `chroms.npy`, `starts.npy`, `ends.npy` | `(n,)` | genomic coordinates |
| `lengths.npy` | `(n,)` | read lengths in bp |
| `n_footprints.npy` | `(n,)` | untruncated number of called footprints |
| `protamine_fractions.npy` | `(n,)` | fraction of read covered by footprints greater than 200 bp |
| `footprint_coverages.npy` | `(n,)` | total called-footprint coverage fraction |
| `manifest.json` | - | input, settings, checkpoint metrics, and checkpoint SHA-256 |

The reconstruction arrays are omitted with `--skip-reconstruction`. Sparse
reads with fewer than three usable footprint tokens receive `NaN` reconstruction
values because the model's span-masking objective cannot be applied safely.

### Python API

```python
from fiberformer.checkpoint import load_model
from fiberformer.data.loader import load_reads_as_list
from fiberformer.embedding.embed import embed_reads

model, config, checkpoint, device = load_model(device="auto")
reads = load_reads_as_list(
    "data/bigbed/new_sample.bb",
    min_read_length=10_000,
    max_reads=1_000,
)
result = embed_reads(
    model,
    reads,
    device,
    config=config,
    batch_size=256,
    compute_reconstruction=False,
)

normalized_embeddings = result.cls_norm
anchor_scores = result.sperm_scores
```

`load_reads_as_list(..., max_reads=N)` takes the first N eligible reads. The
command-line tool instead uses reservoir sampling so every eligible read has an
equal chance of selection.

## UMAP

After embedding:

```bash
fiberformer-umap outputs/embeddings/new_sample
```

This reads `cls_norm.npy` and writes:

- `umap_2d.npy`: two-dimensional coordinates;
- `umap_model.joblib`: the fitted reducer for later transforms;
- `umap.png`: available biological overlays from the embedding directory.

Relevant controls include `--n-neighbors`, `--min-dist`, `--metric`, and
`--seed`. UMAP coordinates are run-specific; compare samples in a shared fit or
project them through the same saved reducer rather than comparing independently
fit coordinate systems.

## How the model works

### Encoder

FiberFormer contains six Conformer-style blocks with hidden dimension 256,
eight attention heads, feed-forward dimension 1,024, and a depthwise convolution
with kernel size five. Each layer combines:

- multi-head self-attention across all footprints on a read;
- a learned bias based on `log10(bp separation + 1)`;
- local one-dimensional convolution over neighboring footprint tokens;
- a position-wise feed-forward network.

The model adds a continuous base-pair sine/cosine positional encoding using the
footprint's offset from the read start. Its wavelengths span local footprint
resolution through approximately 63 kb, matching the scale of the input reads.

### Per-read pooling

After the encoder, FiberFormer produces one contextual vector per token. The
reported model collapses these into a single read vector using a square-root-bp
weighted mean over non-CLS tokens. A 4 kb footprint therefore has more influence
than a 147 bp footprint, without receiving the approximately 27-fold weight it
would have under raw-length weighting. The resulting 256-D vector is then
L2-normalized for the primary `cls_norm` output.

### Training heads

Two objectives are optimized together:

1. **Head A: supervised sperm/somatic anchoring.** A cosine classifier places
   the pooled embedding relative to learned sperm and somatic directions.
   Unlabeled testis reads do not contribute to this loss.
2. **Head B: masked footprint reconstruction.** Forty percent of footprint
   tokens are masked in spans of three to five. The model predicts footprint
   type, size bin, and gap bin from the remaining context. This objective also
   trains on unlabeled testis reads and supplies most of the intermediate-state
   structure used downstream.

The contrastive Head C code remains as an experimental scaffold but its loss
weight is zero in the reported configuration.

## Reported checkpoint

The reviewed checkpoint bundled with this repository has the following
provenance:

| Field | Value |
| --- | --- |
| Path | `fiberformer/weights/best_model.pt` |
| Size | 19,317,950 bytes |
| SHA-256 | `10a1aad108c6403b57042ffbb05ba5fab427cf316a168615f7549ddb61b87ed6` |
| Training epoch | 93 |
| Trainable parameters | 4,819,164 |
| Validation Head B loss | 4.758927 |
| Validation Head A ROC AUC | 0.999084 |
| Pooling | square-root-bp weighted |
| Position encoding | continuous bp sine/cosine |
| Maximum footprints/read | 128 |

The validation metrics describe the held-out training design below; they are
not a performance guarantee on a new cohort.

## Training data and split design

The reported model used three pools of reads at least 10 kb long:

- **Sperm pool:** `m6a_200` and PS00750-PS00755. The in-vitro reference is
  always a sperm anchor; other sperm-pool reads receive Head A label 1 only
  when their per-read protamine fraction is at least 0.90.
- **Unlabeled testis pool:** PS30065, PS30135, PS30330, and PS30549. These reads
  train Head B but do not receive a sperm/somatic label.
- **Somatic pool:** GM12878, PS30010, PS30040, PS30071, PS30076, and PS30086,
  with Head A label 0.

Training batches use a 20% sperm, 60% testis, 20% somatic pool ratio. chr17 is
held out for validation, chr8 for testing, and other retained chromosomes for
training. A cache-adjacent `*.mapq_keep.npy` sidecar can exclude reads failing
the upstream MAPQ/blacklist filter when dataloaders are constructed.

Training-only augmentation includes a random uniform bp-position shift, a
small log-space footprint chemistry jitter, and a 9 kb footprint-size clamp.
The clamp is retained at inference; the random augmentations are not.

## Reproducing training

### Data layout

Place the configured footprint files under `data/bigbed/`:

```text
data/bigbed/
├── m6a_200_fp.bb
├── PS00750_fp.bb
├── ...
├── PS30065.sorted_footprint.bb
├── ...
└── GM12878_fp.bb
```

The complete list and per-pool sampling budgets are in
`fiberformer/configs/fiberformer.yaml`. To use another directory, edit
`data.data_dir` or pass `--data-dir` to the cache builder.

### Build the HDF5 cache

```bash
fiberformer-build-cache \
  --config fiberformer/configs/fiberformer.yaml \
  --data-dir data/bigbed
```

The default cache is `cache/fiberformer.h5`. The builder will not overwrite an
existing cache. Move or remove an obsolete cache explicitly before rebuilding
it. On the reported dataset, cache construction takes substantially longer and
uses much more memory than a small inference run.

### Train

```bash
# Install the optional TensorBoard dependency if wanted.
python -m pip install -e '.[training]'

fiberformer-train --config fiberformer/configs/fiberformer.yaml
```

The supplied configuration uses batch size 1,024, 16 data-loader workers,
mixed precision, up to 150 epochs, and early stopping. It was run on a 32 GB
CUDA GPU. Reduce `training.batch_size` and `training.num_workers` for smaller
hardware. Checkpoints and logs are written beneath `outputs/` and ignored by
Git.

## Repository layout

```text
fiberformer/
├── fiberformer/
│   ├── checkpoint.py          # checkpoint discovery and model loading
│   ├── configs/               # reported YAML configuration
│   ├── data/                  # parsers, tokenization, cache, augmentation
│   ├── embedding/
│   │   ├── embed.py           # supported generic inference command
│   │   ├── embed_umap.py      # supported generic UMAP command
│   │   └── ...                # fixed-cohort paper workflows
│   ├── models/transformer.py  # encoder and Heads A/B
│   ├── scripts/build_cache.py # cache CLI
│   ├── training/train.py      # training CLI
│   └── weights/best_model.pt  # reported pretrained checkpoint
├── tests/                     # checkpoint and inference smoke tests
├── pyproject.toml
└── README.md
```

The `embed_sperm.py`, `embed_testes.py`, `joint_umap_*`, and
`tss_trajectory.py` modules preserve sample-specific workflows used during the
project. They assume the named cohorts and intermediate output layout. For a
new lab sample, start with `fiberformer-embed` and `fiberformer-umap` rather
than editing those scripts.

## Installation notes and troubleshooting

### CUDA or CPU

`--device auto` selects CUDA when `torch.cuda.is_available()` and CPU otherwise.
If CUDA is requested but unavailable, the command fails immediately. Install a
PyTorch build compatible with the machine's driver before installing this
repository; do not assume a wheel built for one workstation matches a cluster.

### `bigBedToBed` not found

Either put the UCSC binary on `PATH`, place an executable copy at
`~/bigBedToBed`, or convert the input to BED12 first. Confirm discovery with:

```bash
command -v bigBedToBed
```

### Empty input after filtering

Check chromosome naming (`chr1` rather than `1`), the 10 kb minimum, BED block
columns, and whether the dataset is non-human. Use `--all-chromosomes` or lower
`--min-read-length` when scientifically appropriate.

### Out-of-memory errors

Reduce `--batch-size`. For UMAP, fit a scientifically chosen subset when the
full embedding matrix exceeds available RAM, then project the remainder through
the saved reducer. The generic UMAP command currently fits all supplied rows.

### Reproducibility

Keep `manifest.json` with every exported embedding directory. It records the
checkpoint hash and sampling settings needed to identify exactly which model
and read subset produced the arrays. Do not combine embeddings made from
different checkpoints without explicitly validating alignment.

## Versioning and archive

The canonical source URL is
`https://github.com/mtcicero26/fiberformer`. Releases are identified by semantic
version tags and include the bundled checkpoint used by the corresponding
software version.

## Data availability, citation, and license

Raw Fiber-seq and FiberHMM footprint inputs are not redistributed here. Follow
the accompanying manuscript's data-availability statement and access controls.

Please cite the accompanying manuscript and the software version used for the
analysis. Software citation metadata are provided in `CITATION.cff`.

The code is available under the MIT License; see `LICENSE`. Restrictions on the
underlying human genomic data apply independently of the source-code license.
