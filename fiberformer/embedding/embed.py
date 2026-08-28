"""Generic BED/BigBed-to-FiberFormer embedding command.

This is the supported inference entry point for new samples. The older
sample-specific modules in this package preserve the paper workflows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from fiberformer.checkpoint import DEFAULT_CHECKPOINT_PATH, load_model
from fiberformer.data.loader import FiberRead, HUMAN_CHROMS, parse_bed_file
from fiberformer.data.size_quantizer import SIZE_QUANTIZER
from fiberformer.data.tokenizer import MASK, tokenize_read


DEFAULT_SIZE_CAP_BP = 9_000


@dataclass
class EmbeddingResult:
    """Arrays produced by :func:`embed_reads`."""

    cls_norm: np.ndarray
    cls_embeddings: np.ndarray
    sperm_scores: np.ndarray
    reconstruction_nll: np.ndarray | None
    reconstruction_perplexity: np.ndarray | None
    read_ids: np.ndarray
    chroms: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    lengths: np.ndarray
    n_footprints: np.ndarray
    protamine_fractions: np.ndarray
    footprint_coverages: np.ndarray


def reservoir_sample_reads(
    input_path: str | Path,
    max_reads: int | None,
    *,
    min_read_length: int = 10_000,
    valid_chroms: set[str] | None = HUMAN_CHROMS,
    seed: int = 42,
) -> list[FiberRead]:
    """Load all eligible reads or take a deterministic reservoir sample."""
    path = Path(input_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input footprint file not found: {path}")
    if max_reads is not None and max_reads <= 0:
        raise ValueError("max_reads must be positive or None")

    stream = parse_bed_file(
        path,
        min_read_length=min_read_length,
        max_reads=None,
        valid_chroms=valid_chroms,
    )
    if max_reads is None:
        return list(stream)

    rng = random.Random(seed)
    reservoir: list[FiberRead] = []
    for index, read in enumerate(stream):
        if len(reservoir) < max_reads:
            reservoir.append(read)
            continue
        replacement = rng.randint(0, index)
        if replacement < max_reads:
            reservoir[replacement] = read
    return reservoir


def _prepare_read(
    read: FiberRead,
    *,
    max_tokens: int,
    size_encoding: str,
    use_lacuna_tokens: bool,
    size_cap_bp: int,
) -> dict[str, np.ndarray | int | float | str]:
    tokenized = tokenize_read(
        read,
        size_encoding=size_encoding,
        use_lacuna_tokens=use_lacuna_tokens,
    )
    n_tokens = min(tokenized.n_tokens, max_tokens)
    length = n_tokens + 1
    size_cap_log = math.log10(float(size_cap_bp))
    size_cap_bin = int(SIZE_QUANTIZER.encode(size_cap_bp))

    return {
        "type_ids": tokenized.type_ids[:length].astype(np.int64),
        "log_sizes": np.minimum(
            tokenized.log_sizes[:length], size_cap_log
        ).astype(np.float32),
        "gap_to_next": tokenized.gap_to_next[:length].astype(np.float32),
        "is_first": tokenized.is_first[:length].astype(np.float32),
        "is_last": tokenized.is_last[:length].astype(np.float32),
        "raw_size_bins": np.minimum(
            tokenized.raw_size_bins[:length], size_cap_bin
        ).astype(np.int64),
        "raw_gap_bins": tokenized.raw_gap_bins[:length].astype(np.int64),
        "summary": tokenized.summary_features.astype(np.float32),
        # Training uses the footprint's true offset from the read start for
        # bp-scale sinusoidal positional encoding. CLS remains at offset zero.
        "bp_positions": np.concatenate(
            [np.zeros(1, dtype=np.int64), tokenized.footprint_starts[:n_tokens]]
        ),
        "n_tokens": n_tokens,
        "read_id": read.read_id,
        "chrom": read.chrom,
        "start": read.start,
        "end": read.end,
        "length": read.length,
        "n_footprints": read.n_footprints,
        "protamine_fraction": read.protamine_fraction(),
        "footprint_coverage": float(read.footprint_sizes.sum()) / max(read.length, 1),
    }


def _pad_batch(
    prepared: Sequence[dict[str, np.ndarray | int | float | str]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch_size = len(prepared)
    max_len = max(int(item["n_tokens"]) + 1 for item in prepared)

    def padded(name: str, dtype: np.dtype) -> np.ndarray:
        result = np.zeros((batch_size, max_len), dtype=dtype)
        for row, item in enumerate(prepared):
            values = np.asarray(item[name])
            result[row, : len(values)] = values
        return result

    type_ids = torch.from_numpy(padded("type_ids", np.int64)).to(device)
    log_sizes = torch.from_numpy(padded("log_sizes", np.float32)).to(device)
    gap_to_next = torch.from_numpy(padded("gap_to_next", np.float32)).to(device)
    is_first = torch.from_numpy(padded("is_first", np.float32)).to(device)
    is_last = torch.from_numpy(padded("is_last", np.float32)).to(device)
    raw_size_bins = torch.from_numpy(padded("raw_size_bins", np.int64)).to(device)
    raw_gap_bins = torch.from_numpy(padded("raw_gap_bins", np.int64)).to(device)
    bp_positions = torch.from_numpy(padded("bp_positions", np.int64)).to(device)
    n_tokens = torch.tensor(
        [int(item["n_tokens"]) for item in prepared], dtype=torch.long, device=device
    )
    summary = torch.from_numpy(
        np.stack([np.asarray(item["summary"]) for item in prepared]).astype(np.float32)
    ).to(device)

    positions = torch.arange(max_len, device=device).unsqueeze(0)
    attention_mask = positions < (n_tokens.unsqueeze(1) + 1)
    zeros = torch.zeros_like(log_sizes)
    continuous = torch.stack(
        [log_sizes, gap_to_next, is_first, is_last, zeros], dim=-1
    )
    log_dist = _distance_matrix(log_sizes, gap_to_next, attention_mask)
    return {
        "type_ids": type_ids,
        "log_sizes": log_sizes,
        "gap_to_next": gap_to_next,
        "continuous": continuous,
        "attention_mask": attention_mask,
        "log_dist": log_dist,
        "summary": summary,
        "bp_positions": bp_positions,
        "n_tokens": n_tokens,
        "raw_size_bins": raw_size_bins,
        "raw_gap_bins": raw_gap_bins,
    }


def _distance_matrix(
    log_sizes: torch.Tensor,
    gap_to_next: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    masked_positions: torch.Tensor | None = None,
    masked_spans: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the pseudo-position distance matrix used during training."""
    sizes_bp = torch.pow(10.0, log_sizes)
    gaps_bp = torch.clamp(torch.pow(10.0, gap_to_next) - 1.0, min=0.0)
    spans = sizes_bp + gaps_bp
    if masked_positions is not None and masked_spans is not None:
        spans = torch.where(masked_positions, masked_spans, spans)
    spans[:, 0] = 0.0
    spans = spans * attention_mask.float()
    pseudo_positions = torch.cumsum(spans, dim=1)
    distances = torch.abs(
        pseudo_positions.unsqueeze(2) - pseudo_positions.unsqueeze(1)
    )
    return torch.log10(distances + 1.0)


def _span_mask(
    n_tokens: torch.Tensor,
    max_len: int,
    *,
    mask_rate: float,
    span_min: int,
    span_max: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    mask = np.zeros((len(n_tokens), max_len), dtype=bool)
    for row, value in enumerate(n_tokens.detach().cpu().numpy()):
        n = int(value)
        if n < span_min:
            continue
        target = max(1, int(round(n * mask_rate)))
        masked = 0
        attempts = 0
        while masked < target and attempts < max(10, 4 * target):
            span_length = min(int(rng.integers(span_min, span_max + 1)), n)
            start = int(rng.integers(1, 2 + n - span_length))
            end = start + span_length
            newly_masked = int((~mask[row, start:end]).sum())
            if newly_masked:
                mask[row, start:end] = True
                masked += newly_masked
            attempts += 1
    return torch.from_numpy(mask).to(n_tokens.device)


def _reconstruction_nll(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    mask_rate: float,
    span_min: int,
    span_max: int,
    rng: np.random.Generator,
    amp_enabled: bool,
) -> np.ndarray:
    max_len = batch["type_ids"].shape[1]
    mask = _span_mask(
        batch["n_tokens"],
        max_len,
        mask_rate=mask_rate,
        span_min=span_min,
        span_max=span_max,
        rng=rng,
    )
    masked_spans = torch.zeros_like(batch["log_sizes"])
    true_spans = torch.pow(10.0, batch["log_sizes"]) + torch.clamp(
        torch.pow(10.0, batch["gap_to_next"]) - 1.0, min=0.0
    )
    noise = torch.from_numpy(
        1.0 + rng.normal(0.0, 0.1, size=mask.shape)
    ).to(device=mask.device, dtype=true_spans.dtype)
    masked_spans[mask] = (true_spans * noise)[mask]

    masked_type_ids = batch["type_ids"].clone()
    masked_type_ids[mask] = MASK
    masked_continuous = batch["continuous"].clone()
    masked_continuous[mask] = 0.0
    masked_continuous[:, :, 4] = masked_spans
    corrupted_dist = _distance_matrix(
        batch["log_sizes"],
        batch["gap_to_next"],
        batch["attention_mask"],
        masked_positions=mask,
        masked_spans=masked_spans,
    )

    with autocast("cuda", enabled=amp_enabled):
        output = model(
            masked_type_ids,
            masked_continuous,
            batch["attention_mask"],
            corrupted_dist,
            batch["summary"],
            bp_positions=batch["bp_positions"],
            masked_positions=mask,
        )

    type_targets = (batch["type_ids"] - 1).clamp(
        0, output["type_logits"].shape[-1] - 1
    )
    size_targets = batch["raw_size_bins"].clamp(
        0, output["size_logits"].shape[-1] - 1
    )
    gap_targets = batch["raw_gap_bins"].clamp(
        0, output["gap_logits"].shape[-1] - 1
    )
    token_nll = (
        F.cross_entropy(
            output["type_logits"].float().transpose(1, 2),
            type_targets,
            reduction="none",
        )
        + F.cross_entropy(
            output["size_logits"].float().transpose(1, 2),
            size_targets,
            reduction="none",
        )
        + F.cross_entropy(
            output["gap_logits"].float().transpose(1, 2),
            gap_targets,
            reduction="none",
        )
    )
    counts = mask.sum(dim=1)
    nll = torch.full(
        (len(mask),), float("nan"), device=mask.device, dtype=torch.float32
    )
    valid = counts > 0
    nll[valid] = (token_nll[valid] * mask[valid].float()).sum(dim=1) / counts[valid]
    return nll.detach().cpu().numpy()


def embed_reads(
    model: torch.nn.Module,
    reads: Sequence[FiberRead],
    device: torch.device,
    *,
    config: dict,
    batch_size: int = 256,
    max_tokens: int | None = None,
    compute_reconstruction: bool = True,
    seed: int = 42,
) -> EmbeddingResult:
    """Embed a collection of already parsed Fiber-seq reads."""
    if not reads:
        raise ValueError("No eligible reads were found in the input file")
    data_cfg = config.get("data", {})
    max_tokens = max_tokens or int(data_cfg.get("max_tokens", 128))
    size_encoding = data_cfg.get("size_encoding", "absolute_uncapped")
    use_lacuna_tokens = bool(data_cfg.get("use_lacuna_tokens", False))
    amp_enabled = device.type == "cuda"
    rng = np.random.default_rng(seed)

    cls_norm_parts: list[np.ndarray] = []
    cls_embedding_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    nll_parts: list[np.ndarray] = []
    prepared_all: list[dict[str, np.ndarray | int | float | str]] = []

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(reads), batch_size):
            prepared = [
                _prepare_read(
                    read,
                    max_tokens=max_tokens,
                    size_encoding=size_encoding,
                    use_lacuna_tokens=use_lacuna_tokens,
                    size_cap_bp=DEFAULT_SIZE_CAP_BP,
                )
                for read in reads[start : start + batch_size]
            ]
            prepared_all.extend(prepared)
            batch = _pad_batch(prepared, device)
            with autocast("cuda", enabled=amp_enabled):
                output = model(
                    batch["type_ids"],
                    batch["continuous"],
                    batch["attention_mask"],
                    batch["log_dist"],
                    batch["summary"],
                    bp_positions=batch["bp_positions"],
                    masked_positions=None,
                )
            cls_norm_parts.append(output["cls_norm"].float().cpu().numpy())
            cls_embedding_parts.append(output["cls_emb"].float().cpu().numpy())
            score_parts.append(
                torch.sigmoid(output["logit_a"]).float().cpu().numpy()
            )
            if compute_reconstruction:
                nll_parts.append(
                    _reconstruction_nll(
                        model,
                        batch,
                        mask_rate=float(data_cfg.get("mask_rate", 0.40)),
                        span_min=int(data_cfg.get("span_min", 3)),
                        span_max=int(data_cfg.get("span_max", 5)),
                        rng=rng,
                        amp_enabled=amp_enabled,
                    )
                )

    nll = np.concatenate(nll_parts).astype(np.float32) if nll_parts else None
    perplexity = None if nll is None else np.exp(np.minimum(nll, 80.0)).astype(np.float32)

    def metadata(name: str, dtype: np.dtype | None = None) -> np.ndarray:
        values = [item[name] for item in prepared_all]
        return np.asarray(values, dtype=dtype)

    return EmbeddingResult(
        cls_norm=np.concatenate(cls_norm_parts).astype(np.float32),
        cls_embeddings=np.concatenate(cls_embedding_parts).astype(np.float32),
        sperm_scores=np.concatenate(score_parts).astype(np.float32),
        reconstruction_nll=nll,
        reconstruction_perplexity=perplexity,
        read_ids=metadata("read_id", str),
        chroms=metadata("chrom", str),
        starts=metadata("start", np.int64),
        ends=metadata("end", np.int64),
        lengths=metadata("length", np.int32),
        n_footprints=metadata("n_footprints", np.int32),
        protamine_fractions=metadata("protamine_fraction", np.float32),
        footprint_coverages=metadata("footprint_coverage", np.float32),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_result(
    result: EmbeddingResult,
    output_dir: str | Path,
    *,
    input_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint: dict,
    settings: dict,
) -> None:
    """Write embedding arrays plus a provenance manifest."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "cls_norm": result.cls_norm,
        "cls_embeddings": result.cls_embeddings,
        "sperm_scores": result.sperm_scores,
        "read_ids": result.read_ids,
        "chroms": result.chroms,
        "starts": result.starts,
        "ends": result.ends,
        "lengths": result.lengths,
        "n_footprints": result.n_footprints,
        "protamine_fractions": result.protamine_fractions,
        "footprint_coverages": result.footprint_coverages,
    }
    if result.reconstruction_nll is not None:
        arrays["reconstruction_nll"] = result.reconstruction_nll
        arrays["reconstruction_perplexity"] = result.reconstruction_perplexity
    for name, array in arrays.items():
        np.save(output / f"{name}.npy", array)

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    manifest = {
        "input": str(Path(input_path).expanduser().resolve()),
        "output": str(output),
        "n_reads": int(len(result.sperm_scores)),
        "embedding_dimensions": int(result.cls_norm.shape[1]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_auc_roc": checkpoint.get("val_auc_roc"),
        "checkpoint_val_loss_b": checkpoint.get("val_loss_b"),
        "settings": settings,
        "files": sorted(f"{name}.npy" for name in arrays),
    }
    with (output / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed FiberHMM footprint BED/BigBed reads with FiberFormer"
    )
    parser.add_argument("input", type=Path, help="FiberHMM .bb or legacy .bed file")
    parser.add_argument("output", type=Path, help="Directory for .npy outputs")
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH,
        help=f"checkpoint path (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--max-reads", type=int, default=10_000,
        help="reservoir sample size; use 0 to embed every eligible read (default: 10000)",
    )
    parser.add_argument("--min-read-length", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--all-chromosomes", action="store_true",
        help="do not restrict input to chr1-chr22 and chrX",
    )
    parser.add_argument(
        "--skip-reconstruction", action="store_true",
        help="skip Head B NLL/perplexity (roughly halves inference work)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    max_reads = None if args.max_reads == 0 else args.max_reads
    if args.max_reads < 0:
        raise SystemExit("--max-reads must be non-negative")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    start = time.time()
    valid_chroms = None if args.all_chromosomes else HUMAN_CHROMS
    print(f"Reading {args.input} ...", flush=True)
    reads = reservoir_sample_reads(
        args.input,
        max_reads,
        min_read_length=args.min_read_length,
        valid_chroms=valid_chroms,
        seed=args.seed,
    )
    print(f"Selected {len(reads):,} eligible reads", flush=True)

    model, config, checkpoint, device = load_model(args.checkpoint, args.device)
    print(
        f"Loaded epoch {checkpoint.get('epoch', 'unknown')} checkpoint on {device}",
        flush=True,
    )
    result = embed_reads(
        model,
        reads,
        device,
        config=config,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        compute_reconstruction=not args.skip_reconstruction,
        seed=args.seed,
    )
    settings = {
        "max_reads": max_reads,
        "min_read_length": args.min_read_length,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens or config.get("data", {}).get("max_tokens", 128),
        "seed": args.seed,
        "device": str(device),
        "human_chromosomes_only": not args.all_chromosomes,
        "compute_reconstruction": not args.skip_reconstruction,
    }
    save_result(
        result,
        args.output,
        input_path=args.input,
        checkpoint_path=args.checkpoint,
        checkpoint=checkpoint,
        settings=settings,
    )
    print(
        f"Saved {len(reads):,} x {result.cls_norm.shape[1]} embeddings to "
        f"{args.output} in {time.time() - start:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
