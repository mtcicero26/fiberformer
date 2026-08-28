"""
Embed 4 testes samples through the FiberFormer transformer.

Key inference-time settings (loaded from checkpoint config):
  - MASK_TYPE_ID = 5 (current tokenizer, with LACUNA slot reserved)
  - n_type_ids = 7
  - pool = 'length_weighted_sqrt'

Two-phase pipeline:
  Phase 1: CPU tokenize (multiprocessing) + write to ./cache/tok_{sample}/
  Phase 2: GPU inference -> cls_norm, cls_emb, scores, perplexity

Outputs under outputs/embeddings/testes/{sample}/.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
FP_DIR = PROJECT_DIR / 'data' / 'bigbed'

MAX_TOKENS = 128
SEQ_LEN = MAX_TOKENS + 1
BATCH_SIZE = 1024
MASK_RATE = 0.40
MASK_TYPE_ID = 5   # FiberFormer / current tokenizer (v8 added LACUNA, shifting MASK to 5)
SPAN_MIN = 3
SPAN_MAX = 5
NUM_WORKERS = 10
N_PER_SAMPLE = 200000
TOK_CHUNK = 50000
MIN_READ_LENGTH = 10_000

SAMPLES = {
    'PS30065': FP_DIR / 'PS30065.sorted_footprint.bb',
    'PS30135': FP_DIR / 'PS30135.sorted_footprint.bb',
    'PS30330': FP_DIR / 'PS30330.sorted_footprint.bb',
    'PS30549': FP_DIR / 'PS30549.sorted_footprint.bb',
}
OUT_DIR = PROJECT_DIR / 'outputs' / 'embeddings' / 'testes'
from fiberformer.checkpoint import DEFAULT_CHECKPOINT_PATH
MODEL_PATH = DEFAULT_CHECKPOINT_PATH


def _tok_dir(sample):
    return PROJECT_DIR / 'cache' / f'tok_{sample}'


def _out_dir(sample):
    d = OUT_DIR / sample
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------- Tokenize (fractional, MAX_TOKENS=128) ----------
def _tok_init():
    global _tokenize_read
    from fiberformer.data.tokenizer import tokenize_read as _tokenize_read


# FiberFormer: must apply the same 9kb size clamp the dataset applied at train time,
# otherwise inference feeds the model out-of-distribution >9kb sizes and
# trivial-prot reads (n_fps=1, single 10-60kb footprint) refuse to collapse.
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


def reservoir_sample(bb_path, n_target, seed=0):
    from fiberformer.data.loader import parse_bed_file, HUMAN_CHROMS
    rng = random.Random(seed)
    reservoir = []
    for i, r in enumerate(parse_bed_file(bb_path, min_read_length=MIN_READ_LENGTH,
                                          max_reads=None,
                                          valid_chroms=HUMAN_CHROMS)):
        if len(reservoir) < n_target:
            reservoir.append(r)
        else:
            j = rng.randint(0, i)
            if j < n_target:
                reservoir[j] = r
    return reservoir


def phase_tokenize(sample):
    tok_dir = _tok_dir(sample)
    tok_dir.mkdir(parents=True, exist_ok=True)
    state_path = tok_dir / 'state.json'
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        if state.get('done', 0) >= state.get('total', 0) and state.get('total', 0) > 0:
            print(f"[{sample}] tokenize done per state.json (N={state['total']:,})")
            return state['total']

    bb_path = SAMPLES[sample]
    print(f"[{sample}] reservoir-sampling {N_PER_SAMPLE:,} from {bb_path.name}...")
    t0 = time.time()
    reads = reservoir_sample(bb_path, N_PER_SAMPLE)
    print(f"[{sample}]   sampled {len(reads):,} in {time.time()-t0:.0f}s")

    print(f"[{sample}] Tokenizing (fractional encoding)...")
    # Collect meta first (small); split tokens into chunks via multiprocessing
    meta_chroms, meta_starts, meta_ends = [], [], []
    meta_lengths, meta_n_fps, meta_prot_fracs, meta_coverages = [], [], [], []

    total = len(reads)
    pool = Pool(NUM_WORKERS, initializer=_tok_init)
    t_start = time.time()
    tc = 0
    done = 0
    for ci in range(0, total, TOK_CHUNK):
        chunk = reads[ci:ci + TOK_CHUNK]
        results = pool.map(_tok_one, chunk, chunksize=256)
        for r in chunk:
            pass  # meta captured inside _tok_one
        meta_chroms.extend([r[9] for r in results])
        meta_starts.extend([r[10] for r in results])
        meta_ends.extend([r[11] for r in results])
        meta_lengths.extend([r[12] for r in results])
        meta_n_fps.extend([r[13] for r in results])
        meta_prot_fracs.extend([r[14] for r in results])
        meta_coverages.extend([r[15] for r in results])

        type_ids = np.array([r[0] for r in results], dtype=np.int16)
        log_sizes = np.array([r[1] for r in results], dtype=np.float32)
        gap_to_next = np.array([r[2] for r in results], dtype=np.float32)
        is_first = np.array([r[3] for r in results], dtype=np.float32)
        is_last = np.array([r[4] for r in results], dtype=np.float32)
        raw_size_bins = np.array([r[5] for r in results], dtype=np.int16)
        raw_gap_bins = np.array([r[6] for r in results], dtype=np.int16)
        summary = np.array([r[7] for r in results], dtype=np.float32)
        n_tokens = np.array([r[8] for r in results], dtype=np.int32)
        np.savez(tok_dir / f'tok_{tc:04d}.npz',
                 type_ids=type_ids, log_sizes=log_sizes, gap_to_next=gap_to_next,
                 is_first=is_first, is_last=is_last,
                 raw_size_bins=raw_size_bins, raw_gap_bins=raw_gap_bins,
                 summary=summary, n_tokens=n_tokens)
        tc += 1
        done += len(results)
        rate = done / max(time.time() - t_start, 0.1)
        print(f"[{sample}]   tok {done:,}/{total:,} | {rate:>5.0f} r/s", flush=True)

    pool.close(); pool.join()

    np.save(tok_dir / 'chroms.npy', np.array(meta_chroms))
    np.save(tok_dir / 'starts.npy', np.array(meta_starts, dtype=np.int64))
    np.save(tok_dir / 'ends.npy', np.array(meta_ends, dtype=np.int64))
    np.save(tok_dir / 'lengths.npy', np.array(meta_lengths, dtype=np.int32))
    np.save(tok_dir / 'n_fps.npy', np.array(meta_n_fps, dtype=np.int32))
    np.save(tok_dir / 'prot_fracs.npy', np.array(meta_prot_fracs, dtype=np.float32))
    np.save(tok_dir / 'coverages.npy', np.array(meta_coverages, dtype=np.float32))

    with open(state_path, 'w') as f:
        json.dump({'done': total, 'total': total, 'chunks': tc}, f)
    print(f"[{sample}] tokenize done in {(time.time()-t_start)/60:.1f} min — {tc} chunks")
    return total


# ---------- GPU inference ----------
def batch_distance_and_positions(log_sizes_t, gap_to_next_t, n_tokens_t, max_len):
    """Compute BOTH the log-distance matrix AND per-token bp positions.
    v7 uses bp_positions as input to the model's bp-SinCos PE."""
    sizes_bp = torch.pow(10.0, log_sizes_t)
    gaps_bp = torch.clamp(torch.pow(10.0, gap_to_next_t) - 1.0, min=0.0)
    spans = sizes_bp + gaps_bp
    spans[:, 0] = 0.0
    idx = torch.arange(max_len, device=log_sizes_t.device).unsqueeze(0)
    valid = idx < (n_tokens_t.unsqueeze(1) + 1)
    spans = spans * valid.float()
    positions = torch.cumsum(spans, dim=1)
    positions = positions - positions[:, 0:1]  # (B, S) float — bp offsets
    D = torch.abs(positions.unsqueeze(2) - positions.unsqueeze(1))
    log_dist = torch.log10(D + 1.0)
    # bp_positions as int (for sin_cos_bp_pe)
    bp_positions = positions.long()
    return log_dist, bp_positions


def _load_model(device):
    from fiberformer.checkpoint import load_model
    model, cfg, _, _ = load_model(MODEL_PATH, device)
    return model, cfg['model']['d_model']


def _span_mask_batch(n_tokens_np, max_len, rng):
    B = n_tokens_np.shape[0]
    mask = np.zeros((B, max_len), dtype=bool)
    for r in range(B):
        n = int(n_tokens_np[r])
        if n <= 0:
            continue
        target_n = max(1, int(round(n * MASK_RATE)))
        n_masked = 0
        attempts = 0
        max_attempts = max(10, 4 * target_n)
        while n_masked < target_n and attempts < max_attempts:
            span_len = int(rng.integers(SPAN_MIN, SPAN_MAX + 1))
            span_len = min(span_len, n)
            start = int(rng.integers(1, 2 + n - span_len))
            end = start + span_len
            new = (~mask[r, start:end]).sum()
            if new == 0:
                attempts += 1; continue
            mask[r, start:end] = True
            n_masked += int(new)
            attempts += 1
    return mask


def phase_infer(sample, model=None, d_model=None, device=None):
    tok_dir = _tok_dir(sample)
    out_dir = _out_dir(sample)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model is None:
        model, d_model = _load_model(device)

    tok_chunks = sorted(tok_dir.glob('tok_*.npz'))
    chunk_sizes = []
    for tc in tok_chunks:
        with np.load(tc) as d:
            chunk_sizes.append(len(d['n_tokens']))
    chunk_offsets = np.cumsum([0] + chunk_sizes)
    total = int(chunk_offsets[-1])
    print(f"[{sample}] Inference on {total:,} reads, {len(tok_chunks)} chunks")

    cls_norm_arr = np.zeros((total, d_model), dtype=np.float32)
    cls_emb_arr = np.zeros((total, d_model), dtype=np.float16)
    scores_arr = np.zeros(total, dtype=np.float32)
    perp_arr = np.zeros(total, dtype=np.float32)
    rng = np.random.default_rng(0)
    t_start = time.time()
    bi_global = 0

    with torch.no_grad():
        for ci, tc in enumerate(tok_chunks):
            with np.load(tc) as d:
                c_type_ids = np.asarray(d['type_ids'])
                c_log_sizes = np.asarray(d['log_sizes'])
                c_gap_to_next = np.asarray(d['gap_to_next'])
                c_is_first = np.asarray(d['is_first'])
                c_is_last = np.asarray(d['is_last'])
                c_raw_size_bins = np.asarray(d['raw_size_bins'])
                c_raw_gap_bins = np.asarray(d['raw_gap_bins'])
                c_summary = np.asarray(d['summary'])
                c_n_tokens = np.asarray(d['n_tokens'])
            chunk_n = len(c_n_tokens)
            for li in range(0, chunk_n, BATCH_SIZE):
                le = min(li + BATCH_SIZE, chunk_n)
                bn = le - li
                n_tok = c_n_tokens[li:le]
                max_len = int(n_tok.max()) + 1

                tid = torch.from_numpy(c_type_ids[li:le, :max_len].astype(np.int64)).to(device)
                ls = torch.from_numpy(c_log_sizes[li:le, :max_len]).to(device)
                gtn = torch.from_numpy(c_gap_to_next[li:le, :max_len]).to(device)
                ifirst = torch.from_numpy(c_is_first[li:le, :max_len]).to(device)
                ilast = torch.from_numpy(c_is_last[li:le, :max_len]).to(device)
                n_tok_t = torch.from_numpy(n_tok.astype(np.int64)).to(device)
                summ = torch.from_numpy(c_summary[li:le]).to(device)
                zeros = torch.zeros(bn, max_len, device=device)
                continuous = torch.stack([ls, gtn, ifirst, ilast, zeros], dim=-1)
                idx = torch.arange(max_len, device=device).unsqueeze(0)
                attn_mask = idx < (n_tok_t.unsqueeze(1) + 1)
                log_dist, bp_positions = batch_distance_and_positions(ls, gtn, n_tok_t, max_len)

                with autocast('cuda'):
                    out = model(tid, continuous, attn_mask, log_dist, summ,
                                bp_positions=bp_positions, masked_positions=None)
                bi = bi_global + li
                be = bi_global + le
                cls_norm_arr[bi:be] = out['cls_norm'].float().cpu().numpy()
                cls_emb_arr[bi:be] = out['cls_emb'].half().cpu().numpy()
                scores_arr[bi:be] = torch.sigmoid(out['logit_a']).float().cpu().numpy().flatten()

                mask_np = _span_mask_batch(n_tok, max_len, rng)
                to_mask = torch.from_numpy(mask_np).to(device)
                tid_m = tid.clone(); tid_m[to_mask] = MASK_TYPE_ID
                cont_m = continuous.clone(); cont_m[to_mask] = 0.0
                with autocast('cuda'):
                    out_m = model(tid_m, cont_m, attn_mask, log_dist, summ,
                                  bp_positions=bp_positions, masked_positions=to_mask)
                type_target = (tid - 1).clamp(0, 2)
                size_target = torch.from_numpy(c_raw_size_bins[li:le, :max_len].astype(np.int64)).to(device).clamp(0, out_m['size_logits'].shape[-1] - 1)
                gap_target = torch.from_numpy(c_raw_gap_bins[li:le, :max_len].astype(np.int64)).to(device).clamp(0, out_m['gap_logits'].shape[-1] - 1)
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
            bi_global += chunk_n
            if ci % 3 == 0:
                print(f"[{sample}]   {bi_global:>7,}/{total:,} | ETA {((total-bi_global)/max(bi_global, 1))*(time.time()-t_start)/60:.1f}min", flush=True)

    # Save
    np.save(out_dir / 'cls_norm.npy', cls_norm_arr)
    np.save(out_dir / 'cls_embeddings.npy', cls_emb_arr)
    np.save(out_dir / 'scores.npy', scores_arr)
    np.save(out_dir / 'perplexity.npy', perp_arr)
    # Copy meta arrays from tok_dir
    for k in ['chroms', 'starts', 'ends', 'lengths', 'n_fps', 'prot_fracs', 'coverages']:
        src = tok_dir / f'{k}.npy'
        if src.exists():
            arr = np.load(src, allow_pickle=(k == 'chroms'))
            np.save(out_dir / f'{k}.npy', arr)
    norms = np.linalg.norm(cls_norm_arr, axis=1)
    print(f"[{sample}] done. cls_norm L2: mean={norms.mean():.4f}, "
          f"score_med={np.median(scores_arr):.3f}, perp_med={np.median(perp_arr):.2f}")


def run_all():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model, d_model = _load_model(device)
    for s in SAMPLES:
        print(f"\n=== {s} ===", flush=True)
        phase_tokenize(s)
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        phase_infer(s, model=model, d_model=d_model, device=device)
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', default=None, choices=list(SAMPLES) + [None])
    args = parser.parse_args()
    if args.sample:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, d_model = _load_model(device)
        phase_tokenize(args.sample)
        phase_infer(args.sample, model=model, d_model=d_model, device=device)
    else:
        run_all()
