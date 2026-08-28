"""
Embed the sperm motility series (m6a_200 + PS00750-755) through FiberFormer.

~10K reads per sample, MAX_TOKENS=128. Output: cls_norm,
cls_embeddings, scores, perplexity, per-read meta.
"""

import argparse
import gc
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
FP_DIR = PROJECT_DIR / 'data' / 'bigbed'

MAX_TOKENS = 128
SEQ_LEN = MAX_TOKENS + 1
BATCH_SIZE = 4096       # sperm reads are short, model is small — GPU has headroom
MASK_RATE = 0.40
MASK_TYPE_ID = 5   # FiberFormer / current tokenizer (v8 added LACUNA, shifting MASK to 5)
SPAN_MIN = 3
SPAN_MAX = 5
NUM_WORKERS = 8

# (sample_name, motility_pct, bigbed_stem)
SAMPLES = [
    ('m6a_200', 88, 'm6a_200_fp.bb'),
    ('PS00750', 62, 'PS00750_fp.bb'),
    ('PS00751', 82, 'PS00751_fp.bb'),
    ('PS00752', 65, 'PS00752_fp.bb'),
    ('PS00753', 10, 'PS00753_fp.bb'),
    ('PS00754',  2, 'PS00754_fp.bb'),
    ('PS00755',  1, 'PS00755_fp.bb'),
]

N_PER_SAMPLE = 10000  # reservoir sample size
from fiberformer.checkpoint import DEFAULT_CHECKPOINT_PATH
MODEL_PATH = DEFAULT_CHECKPOINT_PATH
OUT_DIR = PROJECT_DIR / 'outputs' / 'embeddings' / 'sperm'


def _tok_init():
    global _tokenize_read
    from fiberformer.data.tokenizer import tokenize_read as _tokenize_read


# FiberFormer: 9kb clamp applied at inference too (matches FiberSeqDataset training-time clamp)
import math
from fiberformer.data.size_quantizer import SIZE_QUANTIZER
_LOG_SIZE_CAP = math.log10(9000.0)
_CLAMP_BIN = int(SIZE_QUANTIZER.encode(9000.0))


def _tok_one(read):
    tok = _tokenize_read(read, size_encoding='absolute_uncapped')
    n = min(tok.n_tokens, MAX_TOKENS)
    al = n + 1
    type_ids = np.zeros(SEQ_LEN, dtype=np.int16)
    log_sizes = np.zeros(SEQ_LEN, dtype=np.float32)
    gap_to_next = np.zeros(SEQ_LEN, dtype=np.float32)
    is_first = np.zeros(SEQ_LEN, dtype=np.float32)
    is_last = np.zeros(SEQ_LEN, dtype=np.float32)
    raw_size_bins = np.zeros(SEQ_LEN, dtype=np.int16)
    raw_gap_bins = np.zeros(SEQ_LEN, dtype=np.int16)
    type_ids[:al] = tok.type_ids[:al].astype(np.int16)
    log_sizes[:al] = np.minimum(tok.log_sizes[:al], _LOG_SIZE_CAP)
    gap_to_next[:al] = tok.gap_to_next[:al]
    is_first[:al] = tok.is_first[:al].astype(np.float32)
    is_last[:al] = tok.is_last[:al].astype(np.float32)
    raw_size_bins[:al] = np.minimum(tok.raw_size_bins[:al].astype(np.int16), _CLAMP_BIN)
    raw_gap_bins[:al] = tok.raw_gap_bins[:al].astype(np.int16)
    summary = tok.summary_features.astype(np.float32)
    return (type_ids, log_sizes, gap_to_next, is_first, is_last,
            raw_size_bins, raw_gap_bins, summary, np.int32(n),
            read.chrom, read.start, read.end, read.length,
            read.n_footprints, float(read.protamine_fraction()),
            float(read.footprint_sizes.sum()) / max(read.length, 1))


def batch_distance_and_positions(log_sizes_t, gap_to_next_t, n_tokens_t, max_len):
    sizes_bp = torch.pow(10.0, log_sizes_t)
    gaps_bp = torch.clamp(torch.pow(10.0, gap_to_next_t) - 1.0, min=0.0)
    spans = sizes_bp + gaps_bp
    spans[:, 0] = 0.0
    idx = torch.arange(max_len, device=log_sizes_t.device).unsqueeze(0)
    valid = idx < (n_tokens_t.unsqueeze(1) + 1)
    spans = spans * valid.float()
    positions = torch.cumsum(spans, dim=1)
    positions = positions - positions[:, 0:1]
    D = torch.abs(positions.unsqueeze(2) - positions.unsqueeze(1))
    return torch.log10(D + 1.0), positions.long()


def _load_model(device):
    from fiberformer.checkpoint import load_model
    model, cfg, _, _ = load_model(MODEL_PATH, device)
    return model, cfg['model']['d_model']


def _span_mask_batch(n_tokens_np, max_len, mask_rate, rng):
    B = n_tokens_np.shape[0]
    mask = np.zeros((B, max_len), dtype=bool)
    for r in range(B):
        n = int(n_tokens_np[r])
        if n <= 0:
            continue
        target_n = max(1, int(round(n * mask_rate)))
        footprint_start = 1
        footprint_end = 1 + n
        n_masked = 0
        attempts = 0
        max_attempts = max(10, 4 * target_n)
        while n_masked < target_n and attempts < max_attempts:
            span_len = int(rng.integers(SPAN_MIN, SPAN_MAX + 1))
            span_len = min(span_len, n)
            start = int(rng.integers(footprint_start, footprint_end - span_len + 1))
            end = start + span_len
            new = (~mask[r, start:end]).sum()
            if new == 0:
                attempts += 1
                continue
            mask[r, start:end] = True
            n_masked += int(new)
            attempts += 1
    return mask


def reservoir_sample(bb_path, n_target, seed=42):
    import random
    from fiberformer.data.loader import parse_bed_file, HUMAN_CHROMS
    rng = random.Random(seed)
    reservoir = []
    for i, read in enumerate(parse_bed_file(bb_path, min_read_length=10000,
                                             max_reads=None,
                                             valid_chroms=HUMAN_CHROMS)):
        if len(reservoir) < n_target:
            reservoir.append(read)
        else:
            j = rng.randint(0, i)
            if j < n_target:
                reservoir[j] = read
    return reservoir


def embed_sample(sample, motility, bb_stem, model, d_model, device):
    out_dir = OUT_DIR / sample
    out_dir.mkdir(parents=True, exist_ok=True)

    bb_path = FP_DIR / bb_stem
    t0 = time.time()
    print(f"[{sample}] reservoir-sampling up to {N_PER_SAMPLE:,} from {bb_path.name}...")
    reads = reservoir_sample(bb_path, N_PER_SAMPLE)
    print(f"[{sample}]   got {len(reads):,} reads in {time.time()-t0:.1f}s")

    # Tokenize in parallel
    t0 = time.time()
    with Pool(NUM_WORKERS, initializer=_tok_init) as pool:
        results = pool.map(_tok_one, reads, chunksize=256)
    print(f"[{sample}]   tokenized in {time.time()-t0:.1f}s")
    del reads

    type_ids = np.array([r[0] for r in results], dtype=np.int16)
    log_sizes = np.array([r[1] for r in results], dtype=np.float32)
    gap_to_next = np.array([r[2] for r in results], dtype=np.float32)
    is_first = np.array([r[3] for r in results], dtype=np.float32)
    is_last = np.array([r[4] for r in results], dtype=np.float32)
    raw_size_bins = np.array([r[5] for r in results], dtype=np.int16)
    raw_gap_bins = np.array([r[6] for r in results], dtype=np.int16)
    summary = np.array([r[7] for r in results], dtype=np.float32)
    n_tokens = np.array([r[8] for r in results], dtype=np.int32)
    chroms = np.array([r[9] for r in results])
    starts = np.array([r[10] for r in results], dtype=np.int64)
    ends = np.array([r[11] for r in results], dtype=np.int64)
    lengths = np.array([r[12] for r in results], dtype=np.int32)
    n_fps = np.array([r[13] for r in results], dtype=np.int32)
    prot_fracs = np.array([r[14] for r in results], dtype=np.float32)
    coverages = np.array([r[15] for r in results], dtype=np.float32)
    del results

    total = len(n_tokens)
    cls_norm_arr = np.zeros((total, d_model), dtype=np.float32)
    cls_emb_arr = np.zeros((total, d_model), dtype=np.float16)
    scores_arr = np.zeros(total, dtype=np.float32)
    perp_arr = np.zeros(total, dtype=np.float32)

    rng = np.random.default_rng(0)
    t0 = time.time()
    with torch.no_grad():
        for bi in range(0, total, BATCH_SIZE):
            be = min(bi + BATCH_SIZE, total)
            bn = be - bi
            n_tok = n_tokens[bi:be]
            max_len = int(n_tok.max()) + 1

            tid = torch.from_numpy(type_ids[bi:be, :max_len].astype(np.int64)).to(device)
            ls = torch.from_numpy(log_sizes[bi:be, :max_len]).to(device)
            gtn = torch.from_numpy(gap_to_next[bi:be, :max_len]).to(device)
            ifirst = torch.from_numpy(is_first[bi:be, :max_len]).to(device)
            ilast = torch.from_numpy(is_last[bi:be, :max_len]).to(device)
            n_tok_t = torch.from_numpy(n_tok.astype(np.int64)).to(device)
            summ = torch.from_numpy(summary[bi:be]).to(device)

            zeros = torch.zeros(bn, max_len, device=device)
            continuous = torch.stack([ls, gtn, ifirst, ilast, zeros], dim=-1)
            idx = torch.arange(max_len, device=device).unsqueeze(0)
            attn_mask = idx < (n_tok_t.unsqueeze(1) + 1)
            log_dist, bp_positions = batch_distance_and_positions(ls, gtn, n_tok_t, max_len)

            with autocast('cuda'):
                out = model(tid, continuous, attn_mask, log_dist, summ,
                            bp_positions=bp_positions, masked_positions=None)
            cls_norm_arr[bi:be] = out['cls_norm'].float().cpu().numpy()
            cls_emb_arr[bi:be] = out['cls_emb'].half().cpu().numpy()
            scores_arr[bi:be] = torch.sigmoid(out['logit_a']).float().cpu().numpy().flatten()

            mask_np = _span_mask_batch(n_tok, max_len, MASK_RATE, rng)
            to_mask = torch.from_numpy(mask_np).to(device)
            tid_m = tid.clone(); tid_m[to_mask] = MASK_TYPE_ID
            cont_m = continuous.clone(); cont_m[to_mask] = 0.0

            with autocast('cuda'):
                out_m = model(tid_m, cont_m, attn_mask, log_dist, summ,
                              bp_positions=bp_positions, masked_positions=to_mask)

            type_target = (tid - 1).clamp(0, 2)
            size_target = torch.from_numpy(raw_size_bins[bi:be, :max_len].astype(np.int64)).to(device).clamp(0, out_m['size_logits'].shape[-1]-1)
            gap_target = torch.from_numpy(raw_gap_bins[bi:be, :max_len].astype(np.int64)).to(device).clamp(0, out_m['gap_logits'].shape[-1]-1)

            type_logp = F.log_softmax(out_m['type_logits'].float(), dim=-1)
            size_logp = F.log_softmax(out_m['size_logits'].float(), dim=-1)
            gap_logp = F.log_softmax(out_m['gap_logits'].float(), dim=-1)
            type_ce = -type_logp.gather(-1, type_target.unsqueeze(-1)).squeeze(-1)
            size_ce = -size_logp.gather(-1, size_target.unsqueeze(-1)).squeeze(-1)
            gap_ce = -gap_logp.gather(-1, gap_target.unsqueeze(-1)).squeeze(-1)
            ce_per_token = type_ce + size_ce + gap_ce

            mask_f = to_mask.float()
            mask_count = mask_f.sum(dim=1).clamp(min=1.0)
            perp_arr[bi:be] = ((ce_per_token * mask_f).sum(dim=1) / mask_count).float().cpu().numpy()
    print(f"[{sample}]   inference in {time.time()-t0:.1f}s")

    np.save(out_dir / 'cls_norm.npy', cls_norm_arr)
    np.save(out_dir / 'cls_embeddings.npy', cls_emb_arr)
    np.save(out_dir / 'scores.npy', scores_arr)
    np.save(out_dir / 'perplexity.npy', perp_arr)
    np.save(out_dir / 'chroms.npy', chroms)
    np.save(out_dir / 'starts.npy', starts)
    np.save(out_dir / 'ends.npy', ends)
    np.save(out_dir / 'lengths.npy', lengths)
    np.save(out_dir / 'n_fps.npy', n_fps)
    np.save(out_dir / 'prot_fracs.npy', prot_fracs)
    np.save(out_dir / 'coverages.npy', coverages)

    with open(out_dir / 'meta.txt', 'w') as f:
        f.write(f"sample={sample}\nmotility_pct={motility}\nn_reads={total}\n")
    norms = np.linalg.norm(cls_norm_arr, axis=1)
    print(f"[{sample}] ({motility}% motility) "
          f"score_median={np.median(scores_arr):.3f} "
          f"perp_median={np.median(perp_arr):.2f} "
          f"cls_norm_mean_L2={norms.mean():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model, d_model = _load_model(device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = [(s, m, b) for s, m, b in SAMPLES if args.sample is None or s == args.sample]
    for s, m, b in samples:
        print(f"\n=== {s} ({m}% motility) ===", flush=True)
        embed_sample(s, m, b, model, d_model, device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
