"""
Training loop for Footprint Sequence Transformer.

Curriculum training:
    Stage 1: lambda_A=0, lambda_B=1.0, lambda_C=1.0 (self-supervised only)
             Runs until val reconstruction loss plateaus for N epochs.
    Transition: lambda_A anneals 0->target via cosine over N epochs.
    Stage 2: lambda_A=0.5, lambda_B=1.0, lambda_C=0.5
             Early stops on val AUC-ROC.
"""

import sys
import math
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from pathlib import Path
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from fiberformer.models.transformer import FootprintTransformer
from fiberformer.data.dataset import (create_dataloaders, build_hdf5_cache,
                           build_hdf5_cache_v2, FiberSeqDataset)
from fiberformer.training.contrastive import combined_contrastive_loss

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, scheduler, scaler, cfg,
                lambda_a, lambda_b, lambda_c, device, global_step, writer=None,
                epoch_num=None, ckpt_fn=None):
    """Train one epoch. Returns dict of mean losses and updated global_step.

    If `ckpt_fn` is provided, it is called every `intraepoch_ckpt_every_n_steps`
    with the current global_step, so long epochs survive crashes.
    """
    model.train()
    losses = {'total': 0, 'a': 0, 'b': 0, 'c': 0}
    n_batches = 0
    n_total_batches = len(loader)
    log_every = cfg['logging']['log_every_n_steps']
    ckpt_every = cfg['logging'].get('intraepoch_ckpt_every_n_steps', 0)
    t_epoch_start = time.time()
    t_window_start = t_epoch_start
    window_loss = {'total': 0, 'a': 0, 'b': 0, 'c': 0}
    window_n = 0

    for batch in loader:
        # Move to device
        type_ids = batch['type_ids'].to(device)
        continuous = batch['continuous'].to(device)
        attn_mask = batch['attn_mask'].to(device)
        log_dist = batch['log_dist'].to(device)
        summary = batch['summary'].to(device)
        labels = batch['label'].to(device)
        mask = batch['mask'].to(device)
        target_types = batch['target_type_ids'].to(device)
        target_sizes = batch['target_size_bins'].to(device)
        target_gaps = batch['target_gap_bins'].to(device)
        # v7: per-token bp positions for bp-SinCos PE
        bp_positions = batch['bp_positions'].to(device) if 'bp_positions' in batch else None
        # v3+ cache stores head_a_mask (label != -1 AND chrom not in {X,Y}).
        # Fallback for older caches: labels != -1.
        if 'head_a_mask' in batch:
            label_mask = batch['head_a_mask'].to(device)
        else:
            label_mask = labels >= 0

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=cfg['training']['mixed_precision']):
            outputs = model(type_ids, continuous, attn_mask, log_dist, summary,
                            bp_positions=bp_positions, masked_positions=mask)

            # Head A: classification on the labeled + non-X/Y subset
            loss_a = model.compute_head_a_loss(outputs['logit_a'], labels, label_mask)

            # Head B: masked reconstruction
            loss_b = model.compute_head_b_loss(
                outputs['type_logits'], outputs['size_logits'], outputs['gap_logits'],
                target_types, target_sizes, target_gaps, mask,
            )

            # Head C: contrastive — only if model still exposes proj_c
            if lambda_c > 0 and 'proj_c' in outputs:
                loss_c = combined_contrastive_loss(
                    outputs['proj_c'], labels,
                    temperature=cfg['contrastive']['temperature'],
                )
            else:
                loss_c = torch.zeros((), device=device)

            # Combined loss
            loss = lambda_a * loss_a + lambda_b * loss_b + lambda_c * loss_c

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_total_item = loss.item()
        loss_a_item = loss_a.item()
        loss_b_item = loss_b.item()
        loss_c_item = loss_c.item()

        losses['total'] += loss_total_item
        losses['a'] += loss_a_item
        losses['b'] += loss_b_item
        losses['c'] += loss_c_item
        window_loss['total'] += loss_total_item
        window_loss['a'] += loss_a_item
        window_loss['b'] += loss_b_item
        window_loss['c'] += loss_c_item
        window_n += 1
        n_batches += 1
        global_step += 1

        # Intraepoch stdout progress
        if n_batches % log_every == 0:
            t_now = time.time()
            window_dt = t_now - t_window_start
            # loader.batch_size is None when using a BatchSampler; fall back to
            # the actual batch's first-dim size.
            bs = loader.batch_size or type_ids.shape[0]
            samples = window_n * bs
            sps = samples / max(window_dt, 1e-6)
            pct = 100.0 * n_batches / n_total_batches
            ep_str = f"ep{epoch_num}" if epoch_num is not None else ""
            print(f"  [{ep_str} batch {n_batches}/{n_total_batches} ({pct:.1f}%)] "
                  f"loss={window_loss['total']/window_n:.4f} "
                  f"(A={window_loss['a']/window_n:.4f} "
                  f"B={window_loss['b']/window_n:.4f} "
                  f"C={window_loss['c']/window_n:.4f}) | "
                  f"{sps:.0f} samples/s | lr={scheduler.get_last_lr()[0]:.2e}",
                  flush=True)
            for k in window_loss:
                window_loss[k] = 0
            window_n = 0
            t_window_start = t_now

        if writer and global_step % log_every == 0:
            writer.add_scalar('train/loss_total', loss_total_item, global_step)
            writer.add_scalar('train/loss_a', loss_a_item, global_step)
            writer.add_scalar('train/loss_b', loss_b_item, global_step)
            writer.add_scalar('train/loss_c', loss_c_item, global_step)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)

        # Intraepoch checkpointing
        if ckpt_fn is not None and ckpt_every > 0 and n_batches % ckpt_every == 0:
            ckpt_fn(global_step, n_batches, losses_so_far=losses)

    for k in losses:
        losses[k] /= max(n_batches, 1)

    return losses, global_step


@torch.no_grad()
def validate(model, loader, cfg, lambda_a, lambda_b, lambda_c, device):
    """Validate. Returns dict of mean losses + AUC-ROC if labeled data exists."""
    model.eval()
    losses = {'total': 0, 'a': 0, 'b': 0, 'c': 0}
    all_logits = []
    all_labels = []
    n_batches = 0

    for batch in loader:
        type_ids = batch['type_ids'].to(device)
        continuous = batch['continuous'].to(device)
        attn_mask = batch['attn_mask'].to(device)
        log_dist = batch['log_dist'].to(device)
        summary = batch['summary'].to(device)
        labels = batch['label'].to(device)
        mask = batch['mask'].to(device)
        target_types = batch['target_type_ids'].to(device)
        target_sizes = batch['target_size_bins'].to(device)
        target_gaps = batch['target_gap_bins'].to(device)
        bp_positions = batch['bp_positions'].to(device) if 'bp_positions' in batch else None
        if 'head_a_mask' in batch:
            label_mask = batch['head_a_mask'].to(device)
        else:
            label_mask = labels >= 0

        with autocast('cuda', enabled=cfg['training']['mixed_precision']):
            outputs = model(type_ids, continuous, attn_mask, log_dist, summary,
                            bp_positions=bp_positions, masked_positions=mask)

            loss_a = model.compute_head_a_loss(outputs['logit_a'], labels, label_mask)
            loss_b = model.compute_head_b_loss(
                outputs['type_logits'], outputs['size_logits'], outputs['gap_logits'],
                target_types, target_sizes, target_gaps, mask,
            )
            if lambda_c > 0 and 'proj_c' in outputs:
                loss_c = combined_contrastive_loss(
                    outputs['proj_c'], labels,
                    temperature=cfg['contrastive']['temperature'],
                )
            else:
                loss_c = torch.zeros((), device=device)
            loss = lambda_a * loss_a + lambda_b * loss_b + lambda_c * loss_c

        losses['total'] += loss.item()
        losses['a'] += loss_a.item()
        losses['b'] += loss_b.item()
        losses['c'] += loss_c.item()
        n_batches += 1

        # Collect for AUC
        if label_mask.sum() > 0:
            all_logits.append(outputs['logit_a'][label_mask].cpu())
            all_labels.append(labels[label_mask].cpu())

    for k in losses:
        losses[k] /= max(n_batches, 1)

    # AUC-ROC
    if all_logits:
        logits = torch.cat(all_logits)
        labs = torch.cat(all_labels)
        probs = torch.sigmoid(logits).numpy()
        try:
            losses['auc_roc'] = roc_auc_score(labs.numpy(), probs)
        except ValueError:
            losses['auc_roc'] = 0.0
    else:
        losses['auc_roc'] = 0.0

    return losses


def train(cfg):
    """Full training pipeline with curriculum."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Relative data/cache/output paths are anchored to the directory from which
    # the CLI is run. This works for both editable clones and installed wheels.
    project_dir = Path.cwd().resolve()
    configured_cache = Path(cfg['data']['cache_path']).expanduser()
    cache_path = (configured_cache if configured_cache.is_absolute()
                  else project_dir / configured_cache)

    # Build cache if needed
    if not cache_path.exists():
        print("Cache not found, building...")
        configured_data_dir = Path(cfg['data'].get('data_dir', './data/bigbed')).expanduser()
        bed_dir = (configured_data_dir if configured_data_dir.is_absolute()
                   else project_dir / configured_data_dir)
        # v3 builder expected for the foundation-model / v5 config
        if 'sperm_paths' in cfg['data'] and cfg['data'].get('cache_version', 'v3') == 'v3':
            from fiberformer.data.dataset import build_hdf5_cache_v3
            sperm_paths = [bed_dir / p for p in cfg['data']['sperm_paths']]
            testes_paths = [bed_dir / p for p in cfg['data']['testes_paths']]
            somatic_paths = [bed_dir / p for p in cfg['data']['somatic_paths']]
            build_hdf5_cache_v3(
                cache_path=cache_path,
                sperm_paths=sperm_paths,
                testes_paths=testes_paths,
                somatic_paths=somatic_paths,
                pure_sperm_names=tuple(cfg['data'].get('pure_sperm_names', ['m6a_200_fp'])),
                maturity_threshold=cfg['data'].get('maturity_threshold', 0.90),
                max_tokens=cfg['data']['max_tokens'],
                n_per_sperm=cfg['data'].get('n_per_sperm', 12000),
                n_per_testes=cfg['data'].get('n_per_testes', 30000),
                n_per_somatic=cfg['data'].get('n_per_somatic', 7000),
                size_encoding=cfg['data'].get('size_encoding', 'absolute'),
                use_lacuna_tokens=cfg['data'].get('use_lacuna_tokens', False),
            )
        elif 'sperm_paths' in cfg['data']:
            sperm_paths = [bed_dir / p for p in cfg['data']['sperm_paths']]
            testes_paths = [bed_dir / p for p in cfg['data']['testes_paths']]
            somatic_paths = [bed_dir / p for p in cfg['data']['somatic_paths']]
            build_hdf5_cache_v2(
                cache_path=cache_path,
                sperm_paths=sperm_paths,
                testes_paths=testes_paths,
                somatic_paths=somatic_paths,
                max_tokens=cfg['data']['max_tokens'],
                n_sperm=cfg['data']['n_sperm'],
                n_somatic_locus_aware=cfg['data']['n_somatic_locus_aware'],
                n_somatic_explicit=cfg['data']['n_somatic_explicit'],
                n_unlabeled=cfg['data']['n_unlabeled'],
            )
        else:
            build_hdf5_cache(
                cache_path=cache_path,
                sperm_path=bed_dir / 'm6a_200_fp.bed',
                testes_path=bed_dir / 'PS30065_fp.bed',
                max_tokens=cfg['data']['max_tokens'],
                n_sperm=cfg['data']['n_sperm'],
                n_somatic=cfg['data']['n_somatic'],
                n_unlabeled=cfg['data']['n_unlabeled'],
            )

    # Create dataloaders
    loaders = create_dataloaders(
        cache_path=cache_path,
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['training']['num_workers'],
        max_tokens=cfg['data']['max_tokens'],
        mask_rate=cfg['data']['mask_rate'],
        span_min=cfg['data'].get('span_min', 3),
        span_max=cfg['data'].get('span_max', 5),
        seed=cfg['training']['seed'],
        pool_ratios=cfg['data'].get('pool_ratios'),  # None → plain shuffle
    )
    print(f"Train: {len(loaders['train'].dataset)} | Val: {len(loaders['val'].dataset)} | "
          f"Test: {len(loaders['test'].dataset)}")

    # Model
    # label_smoothing arg is v1-only (removed in v5's cosine classifier Head A)
    model_kwargs = dict(
        d_model=cfg['model']['d_model'],
        nhead=cfg['model']['nhead'],
        n_layers=cfg['model']['n_layers'],
        d_ff=cfg['model']['d_ff'],
        dropout=cfg['model']['dropout'],
    )
    if 'logit_scale_init' in cfg['model']:
        model_kwargs['logit_scale_init'] = cfg['model']['logit_scale_init']
    if 'pool' in cfg['model']:
        model_kwargs['pool'] = cfg['model']['pool']
    # v7 options
    if 'conv_kernel' in cfg['model']:
        model_kwargs['conv_kernel'] = cfg['model']['conv_kernel']
    if 'positional' in cfg['model']:
        model_kwargs['positional'] = cfg['model']['positional']
    # v8 options: LACUNA adds a token-id class (5→7) and Head B type class (3→4)
    if 'n_type_ids' in cfg['model']:
        model_kwargs['n_type_ids'] = cfg['model']['n_type_ids']
    if 'n_head_b_type_classes' in cfg['model']:
        model_kwargs['n_head_b_type_classes'] = cfg['model']['n_head_b_type_classes']
    model = FootprintTransformer(**model_kwargs).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay'],
    )

    # Scheduler
    total_steps = cfg['training']['max_epochs'] * len(loaders['train'])
    scheduler = get_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=cfg['training']['warmup_steps'],
        total_steps=total_steps,
    )

    scaler = GradScaler('cuda', enabled=cfg['training']['mixed_precision'])

    # Tensorboard
    writer = None
    if cfg.get('logging', {}).get('tensorboard', True):
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = project_dir / cfg['logging']['log_dir']
            log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(log_dir))
        except Exception as error:
            print(f"[warn] TensorBoard disabled: {error}")

    # Save directory — configurable per run so v2 doesn't trample v1 checkpoints
    save_subdir = cfg.get('logging', {}).get('save_dir', 'outputs/models')
    save_dir = project_dir / save_subdir
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Training mode: curriculum (v1) or flat (v5 / foundation-model) ---
    # If cfg has a `curriculum` block, v1 curriculum runs. If not, fall back
    # to fixed loss weights from cfg['training'] — the v5 default.
    global_step = 0
    best_val_loss_b = float('inf')
    best_val_auc = 0.0
    plateau_count = 0
    stage = 1
    transition_start_epoch = None
    start_epoch = 0

    use_curriculum = 'curriculum' in cfg
    if use_curriculum:
        cur = cfg['curriculum']
    else:
        cur = None
        # v5: fixed lambdas from cfg['training'] (single-stage training).
        # Support lambda_c even though Head C is dropped — allows 0 to skip.
        flat_lambda_a = cfg['training'].get('lambda_a', 1.0)
        flat_lambda_b = cfg['training'].get('lambda_b', 1.0)
        flat_lambda_c = cfg['training'].get('lambda_c', 0.0)
        stage = 'flat'

    # Resume from checkpoint if it exists
    resume_path = save_dir / 'checkpoint_latest.pt'
    if resume_path.exists():
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']
        stage = ckpt['stage']
        best_val_loss_b = ckpt['best_val_loss_b']
        best_val_auc = ckpt['best_val_auc']
        plateau_count = ckpt['plateau_count']
        transition_start_epoch = ckpt.get('transition_start_epoch')
        print(f"  Resumed at epoch {start_epoch}, stage={stage}, "
              f"best_val_B={best_val_loss_b:.4f}, best_AUC={best_val_auc:.4f}")

    for epoch in range(start_epoch, cfg['training']['max_epochs']):
        t0 = time.time()

        # Determine lambdas — curriculum (v1) or flat (v5)
        if not use_curriculum:
            lambda_a = flat_lambda_a
            lambda_b = flat_lambda_b
            lambda_c = flat_lambda_c
        elif stage == 1:
            lambda_a = cur['stage1']['lambda_a']
            lambda_b = cur['stage1']['lambda_b']
            lambda_c = cur['stage1']['lambda_c']
        elif stage == 'transition':
            # Cosine anneal lambda_a from 0 to stage2 target
            progress = (epoch - transition_start_epoch) / max(1, cur['transition']['n_epochs'])
            progress = min(1.0, progress)
            target_a = cur['stage2']['lambda_a']
            lambda_a = target_a * 0.5 * (1 - math.cos(math.pi * progress))
            lambda_b = cur['stage2']['lambda_b']
            lambda_c = cur['stage2']['lambda_c']
        else:  # stage 2
            lambda_a = cur['stage2']['lambda_a']
            lambda_b = cur['stage2']['lambda_b']
            lambda_c = cur['stage2']['lambda_c']

        # Intraepoch checkpoint closure — saves a "in-progress" checkpoint so a
        # mid-epoch crash can resume from the same epoch's latest known step.
        # Note: we mark epoch-1 so that on resume, the epoch loop will re-run
        # this epoch (simple & safe — we prefer replaying a partial epoch over
        # inventing per-step dataloader resume).
        def _intraepoch_ckpt(gs, n_batches_done, losses_so_far):
            torch.save({
                'epoch': epoch - 1,          # so resume re-runs THIS epoch
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'global_step': gs,
                'stage': stage,
                'best_val_loss_b': best_val_loss_b,
                'best_val_auc': best_val_auc,
                'plateau_count': plateau_count,
                'transition_start_epoch': transition_start_epoch,
                'intraepoch_batch': n_batches_done,  # informational
            }, save_dir / 'checkpoint_latest.pt')

        # Train
        train_losses, global_step = train_epoch(
            model, loaders['train'], optimizer, scheduler, scaler, cfg,
            lambda_a, lambda_b, lambda_c, device, global_step, writer,
            epoch_num=epoch + 1, ckpt_fn=_intraepoch_ckpt,
        )

        # Validate
        val_losses = validate(model, loaders['val'], cfg, lambda_a, lambda_b, lambda_c, device)

        dt = time.time() - t0
        print(f"Epoch {epoch+1:3d} [{stage}] "
              f"train_loss={train_losses['total']:.4f} "
              f"(A={train_losses['a']:.4f} B={train_losses['b']:.4f} C={train_losses['c']:.4f}) | "
              f"val_loss={val_losses['total']:.4f} "
              f"(B={val_losses['b']:.4f}) "
              f"AUC={val_losses['auc_roc']:.4f} | "
              f"λ_A={lambda_a:.3f} | {dt:.1f}s")

        if writer:
            writer.add_scalar('val/loss_total', val_losses['total'], epoch)
            writer.add_scalar('val/loss_a', val_losses['a'], epoch)
            writer.add_scalar('val/loss_b', val_losses['b'], epoch)
            writer.add_scalar('val/loss_c', val_losses['c'], epoch)
            writer.add_scalar('val/auc_roc', val_losses['auc_roc'], epoch)
            writer.add_scalar('curriculum/lambda_a', lambda_a, epoch)
            writer.add_scalar('curriculum/stage', 1 if stage == 1 else (1.5 if stage == 'transition' else 2), epoch)

        # --- Stage transitions (curriculum only) ---
        if not use_curriculum:
            # v5 flat training: save best-by-val-loss-b; early-stop on patience.
            if val_losses['b'] < best_val_loss_b - 1e-4:
                best_val_loss_b = val_losses['b']
                plateau_count = 0
                # Also track best AUC if labeled val data exists, for logging.
                if val_losses['auc_roc'] > best_val_auc:
                    best_val_auc = val_losses['auc_roc']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss_b': val_losses['b'],
                    'val_auc_roc': val_losses['auc_roc'],
                    'config': cfg,
                }, save_dir / 'best_model.pt')
                print(f"  -> Saved best model (val_B={best_val_loss_b:.4f}, AUC={val_losses['auc_roc']:.4f})")
            else:
                plateau_count += 1
            early_stop_patience = cfg['training'].get('early_stop_patience', 15)
            if plateau_count >= early_stop_patience:
                print(f"  -> Early stopping at epoch {epoch+1}")
                break

        elif stage == 1:
            # Check for plateau in val reconstruction loss
            if val_losses['b'] < best_val_loss_b - 1e-4:
                best_val_loss_b = val_losses['b']
                plateau_count = 0
            else:
                plateau_count += 1

            if plateau_count >= cur['stage1']['patience']:
                print(f"  -> Stage 1 plateau detected. Starting transition.")
                stage = 'transition'
                transition_start_epoch = epoch + 1
                # Save Stage 1 checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'stage': 'stage1_complete',
                }, save_dir / 'checkpoint_stage1.pt')

        elif stage == 'transition':
            if epoch - transition_start_epoch >= cur['transition']['n_epochs']:
                print(f"  -> Transition complete. Entering Stage 2.")
                stage = 2
                best_val_auc = val_losses['auc_roc']
                plateau_count = 0

        elif stage == 2:
            # Early stopping on val AUC-ROC
            if val_losses['auc_roc'] > best_val_auc + 1e-5:
                best_val_auc = val_losses['auc_roc']
                plateau_count = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc_roc': best_val_auc,
                    'stage': 'stage2_best',
                }, save_dir / 'best_model.pt')
                print(f"  -> Saved best model (AUC={best_val_auc:.4f})")
            else:
                plateau_count += 1

            if plateau_count >= cur['stage2']['early_stop_patience']:
                print(f"  -> Early stopping at epoch {epoch+1}")
                break

        # Periodic checkpoint every epoch (for crash recovery)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'global_step': global_step,
            'stage': stage,
            'best_val_loss_b': best_val_loss_b,
            'best_val_auc': best_val_auc,
            'plateau_count': plateau_count,
            'transition_start_epoch': transition_start_epoch,
        }, save_dir / 'checkpoint_latest.pt')

    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': cfg,
    }, save_dir / 'final_model.pt')
    print(f"\nTraining complete. Best val AUC: {best_val_auc:.4f}")

    if writer:
        writer.close()

    # --- Final evaluation on test set ---
    print("\n--- Test Set Evaluation ---")
    if use_curriculum:
        la, lb, lc = cur['stage2']['lambda_a'], cur['stage2']['lambda_b'], cur['stage2']['lambda_c']
    else:
        la, lb, lc = flat_lambda_a, flat_lambda_b, flat_lambda_c
    test_losses = validate(model, loaders['test'], cfg, la, lb, lc, device)
    print(f"Test loss: {test_losses['total']:.4f} | "
          f"Head B: {test_losses['b']:.4f} | "
          f"AUC-ROC: {test_losses['auc_roc']:.4f}")

    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train Footprint Sequence Transformer')
    parser.add_argument('--config', type=str,
                        default=str(Path(__file__).resolve().parents[1]
                                    / 'configs' / 'fiberformer.yaml'))
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg['training']['seed'])
    np.random.seed(cfg['training']['seed'])

    train(cfg)


if __name__ == '__main__':
    main()
