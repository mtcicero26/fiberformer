"""
TSS chromatin remodeling along the spermatogenesis trajectory.

Uses model sperm score for trajectory binning with protamine-fraction gating
to exclude the off-trajectory island (Sertoli-like cells with intermediate
scores but zero protamine). Fork split uses UMAP1 median on mature sperm reads.

At TSSs, both fork branches are nearly fully protaminated (~99%), unlike CTCF
sites where the nuc-retaining fork shows clear nucleosome retention.

Produces:
  - tss_profiles_v4.pkl: per-bin profiles (small/nuc/prot/coverage)
  - tss_trajectory_lineplot.png: dual-axis lineplot with fork split
"""

import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from fiberformer.data.loader import parse_bed_file, HUMAN_CHROMS

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
BED_DIR = PROJECT_DIR / 'data' / 'bigbed'  # per-sample bigBed files; configure via project data dir
EMB_DIR = PROJECT_DIR / 'outputs' / 'embeddings' / 'all_testes' / 'merged'
OUT_DIR = PROJECT_DIR / 'outputs' / 'embeddings' / 'all_testes'
FIG_DIR = PROJECT_DIR / 'outputs' / 'figures'

HALF_WINDOW = 5000
TSS_CENTER_HALF = 250  # +/-250bp = 500bp window

# Score bins for trajectory
SCORE_BINS = [
    ('Somatic',         0.00, 0.05),
    ('0.05\u20130.15',       0.05, 0.15),
    ('0.15\u20130.30',       0.15, 0.30),
    ('0.30\u20130.50',       0.30, 0.50),
    ('0.50\u20130.70',       0.50, 0.70),
    ('0.70\u20130.85',       0.70, 0.85),
    ('0.85\u20130.95',       0.85, 0.95),
    ('0.95\u20131.00',       0.95, 1.01),
]

# Off-trajectory island filter: non-somatic reads must have prot_frac >= this
MIN_PROT_FRAC = 0.10

# Fork: split mature sperm reads by UMAP1 median
FORK_SCORE_THRESH = 0.85


def load_tss(path='./data/protein_coding_tss.tsv'):
    tss_list = []
    with open(path) as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            tss_list.append((parts[0], int(parts[1]), parts[2], parts[3]))
    return tss_list


def extract_tss_from_gtf(gtf_path):
    acc_to_chr = {}
    for i in range(1, 23):
        acc_to_chr[f'NC_0000{i:02d}'] = f'chr{i}'
    acc_to_chr['NC_000023'] = 'chrX'
    acc_to_chr['NC_000024'] = 'chrY'
    seen, tss_list = set(), []
    with open(gtf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9 or parts[2] != 'gene':
                continue
            if 'gene_biotype "protein_coding"' not in parts[8]:
                continue
            gene_name = None
            for attr in parts[8].split(';'):
                attr = attr.strip()
                if attr.startswith('gene_id'):
                    gene_name = attr.split('"')[1]
            acc = parts[0].rsplit('.', 1)[0]
            chrom = acc_to_chr.get(acc)
            if chrom is None or chrom not in HUMAN_CHROMS:
                continue
            strand = parts[6]
            start, end = int(parts[3]) - 1, int(parts[4])
            tss = start if strand == '+' else end
            key = (chrom, tss, strand)
            if key in seen:
                continue
            seen.add(key)
            tss_list.append((chrom, tss, strand, gene_name or ''))
    return tss_list


def find_tss_overlaps(chroms, starts, ends, tss_list):
    margin = HALF_WINDOW
    tss_by_chrom = {}
    for chrom, pos, strand, gene in tss_list:
        tss_by_chrom.setdefault(chrom, []).append(pos)
    for chrom in tss_by_chrom:
        tss_by_chrom[chrom] = np.array(sorted(tss_by_chrom[chrom]), dtype=np.int64)
    read_indices, tss_centers = [], []
    for i in range(len(chroms)):
        c = chroms[i]
        if c not in tss_by_chrom:
            continue
        arr = tss_by_chrom[c]
        lo = np.searchsorted(arr, starts[i] + margin, side='left')
        hi = np.searchsorted(arr, ends[i] - margin, side='right')
        for j in range(lo, hi):
            read_indices.append(i)
            tss_centers.append(arr[j])
        if i % 1_000_000 == 0 and i > 0:
            print(f"  Scanned {i:,}/{len(chroms):,}, {len(read_indices):,} pairs")
    return np.array(read_indices, dtype=np.int64), np.array(tss_centers, dtype=np.int64)


def split_fork_umap(read_indices, scores, prot_fracs, umap_2d):
    """Split mature sperm reads into compacted vs nuc-retaining using UMAP1 median.

    Returns dict mapping read index -> 'compacted' or 'nuc_retaining'.
    """
    sperm_mask = scores[read_indices] >= FORK_SCORE_THRESH
    sperm_ri = np.unique(read_indices[sperm_mask])
    print(f"\nFork split: {len(sperm_ri):,} unique mature sperm reads (score >= {FORK_SCORE_THRESH})")

    if len(sperm_ri) < 100:
        print("  Too few reads for fork split")
        return {}

    # Get UMAP coordinates
    u1 = umap_2d[sperm_ri, 0]
    valid = ~np.isnan(u1)
    u1_valid = u1[valid]
    ri_valid = sperm_ri[valid]
    print(f"  {len(ri_valid):,} with UMAP coordinates")

    if len(ri_valid) < 100:
        print("  Too few reads with UMAP coords")
        return {}

    u1_median = np.median(u1_valid)
    left = u1_valid < u1_median
    right = ~left

    pf_left = prot_fracs[ri_valid[left]].mean()
    pf_right = prot_fracs[ri_valid[right]].mean()
    print(f"  UMAP1 median split at {u1_median:.1f}")
    print(f"  Left:  n={left.sum():,}, mean prot_frac={pf_left:.4f}")
    print(f"  Right: n={right.sum():,}, mean prot_frac={pf_right:.4f}")

    # Higher prot_frac = compacted
    if pf_left >= pf_right:
        left_label, right_label = 'compacted', 'nuc_retaining'
    else:
        left_label, right_label = 'nuc_retaining', 'compacted'

    read_to_fork = {}
    for i, ri in enumerate(ri_valid):
        read_to_fork[ri] = left_label if left[i] else right_label

    n_comp = sum(1 for v in read_to_fork.values() if v == 'compacted')
    n_nuc = sum(1 for v in read_to_fork.values() if v == 'nuc_retaining')
    print(f"  Compacted: {n_comp:,}, Nuc-retaining: {n_nuc:,}")

    return read_to_fork


def compute_profiles(read_indices, tss_centers, bin_assignments, bed_path):
    needed_reads = set()
    for idx_arr in bin_assignments.values():
        needed_reads.update(read_indices[idx_arr].tolist())
    print(f"\nNeed {len(needed_reads):,} unique reads")

    read_to_pairs = {}
    for bname, pair_indices in bin_assignments.items():
        for pi in pair_indices:
            read_to_pairs.setdefault(read_indices[pi], []).append((bname, tss_centers[pi]))

    window = 2 * HALF_WINDOW
    profiles = {bname: {k: np.zeros(window, dtype=np.float64)
                        for k in ('small', 'nuc', 'prot', 'coverage')}
                for bname in bin_assignments}

    print("Streaming through BED file...")
    read_idx, processed = 0, 0
    for read in parse_bed_file(bed_path, min_read_length=10000, max_reads=None,
                               valid_chroms=HUMAN_CHROMS):
        if read_idx not in read_to_pairs:
            read_idx += 1
            continue
        fp_starts_abs = read.start + read.footprint_starts
        fp_ends_abs = fp_starts_abs + read.footprint_sizes
        fp_sizes = read.footprint_sizes
        for bname, tss_center in read_to_pairs[read_idx]:
            prof = profiles[bname]
            ws = tss_center - HALF_WINDOW
            we = tss_center + HALF_WINDOW
            cs = max(read.start, ws) - ws
            ce = min(read.start + read.length, we) - ws
            if cs < ce:
                prof['coverage'][cs:ce] += 1
            for fi in range(len(fp_starts_abs)):
                s = max(fp_starts_abs[fi], ws) - ws
                e = min(fp_ends_abs[fi], we) - ws
                if s >= e:
                    continue
                sz = fp_sizes[fi]
                if sz < 90:
                    prof['small'][s:e] += 1
                elif sz < 200:
                    prof['nuc'][s:e] += 1
                else:
                    prof['prot'][s:e] += 1
        processed += 1
        read_idx += 1
        if processed % 50000 == 0:
            print(f"  Processed {processed:,}/{len(needed_reads):,}")
    print(f"  Done: {processed:,} reads")

    for bname in profiles:
        cov_safe = np.maximum(profiles[bname]['coverage'], 1)
        for key in ('small', 'nuc', 'prot'):
            profiles[bname][key] /= cov_safe
    return profiles


def plot_umap_bins(read_indices, umap_2d, bin_assignments, bin_info, fig_path):
    """Debug UMAP showing which reads fall into each trajectory/fork bin."""
    traj_names = bin_info['trajectory_names']
    fork_names = bin_info['fork_names']
    all_names = traj_names + fork_names

    # Trajectory color ramp: blue -> green -> yellow -> red
    traj_colors = plt.cm.RdYlBu_r(np.linspace(0.05, 0.95, len(traj_names)))
    fork_colors = {'fork_compacted': '#CC3333', 'fork_nuc_retaining': '#993333'}

    n_panels = len(all_names)
    cols = 4
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = axes.flatten()

    for idx, bname in enumerate(all_names):
        ax = axes[idx]
        if bname not in bin_assignments or len(bin_assignments[bname]) == 0:
            ax.set_title(f'{bname}\n(0 reads)', fontsize=8)
            ax.set_visible(False)
            continue

        pair_idx = bin_assignments[bname]
        ri = np.unique(read_indices[pair_idx])
        u = umap_2d[ri]
        valid = ~np.isnan(u[:, 0])
        u = u[valid]

        # Background: all TSS-overlapping reads
        all_ri = np.unique(read_indices)
        u_all = umap_2d[all_ri]
        v_all = ~np.isnan(u_all[:, 0])
        u_all = u_all[v_all]

        ax.scatter(u_all[:, 0], u_all[:, 1], c='lightgray', s=0.3, alpha=0.1,
                   rasterized=True, zorder=1)

        if idx < len(traj_names):
            color = traj_colors[idx]
        else:
            color = fork_colors.get(bname, 'red')

        ax.scatter(u[:, 0], u[:, 1], c=[color], s=1.5, alpha=0.3,
                   rasterized=True, zorder=3)
        ax.set_title(f'{bname}\n({len(u):,} reads)', fontsize=8, fontweight='bold')
        ax.set_xlabel('UMAP1', fontsize=7)
        ax.set_ylabel('UMAP2', fontsize=7)
        ax.tick_params(labelsize=6)

    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('TSS trajectory bins on UMAP', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved UMAP debug to {fig_path.name}")
    plt.close()


def extract_metrics(profiles, bin_info, center_lo, center_hi, bg_lo=None, bg_hi=None):
    """Extract per-bin summary metrics from profile arrays.

    center_lo/hi: slice of the profile array for the center metric.
    bg_lo/hi: optional flank region for background subtraction.
    """
    traj_names = bin_info['trajectory_names']
    fork_names = bin_info['fork_names']
    metrics = {}
    for bname in traj_names + fork_names:
        if bname not in profiles:
            continue
        p = profiles[bname]
        nuc_c = np.mean(p['nuc'][center_lo:center_hi])
        prot_c = np.mean(p['prot'][center_lo:center_hi])
        small_c = np.mean(p['small'][center_lo:center_hi])
        n = int(np.mean(p['coverage'][center_lo:center_hi]))
        m = {'nuc': nuc_c, 'prot': prot_c, 'small': small_c, 'n': n}
        if bg_lo is not None:
            # Background from flanking regions
            m['nuc_bg'] = np.mean(np.concatenate([p['nuc'][bg_lo[0]:bg_lo[1]],
                                                   p['nuc'][bg_hi[0]:bg_hi[1]]]))
            m['prot_bg'] = np.mean(np.concatenate([p['prot'][bg_lo[0]:bg_lo[1]],
                                                    p['prot'][bg_hi[0]:bg_hi[1]]]))
            m['small_bg'] = np.mean(np.concatenate([p['small'][bg_lo[0]:bg_lo[1]],
                                                     p['small'][bg_hi[0]:bg_hi[1]]]))
        metrics[bname] = m
    return metrics


def _make_lineplot(x_pos, n_traj, valid_traj, metrics, fork_names,
                   y1_vals, y1_label, y1_color,
                   y2_vals, y2_label, y2_color,
                   fork_y1, fork_y2, title, fig_path):
    """Generic dual-axis lineplot with fork."""
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    ax1.plot(x_pos[:n_traj], y1_vals, 'o-', color=y1_color, linewidth=2.5, markersize=7, zorder=5)
    ax1.set_ylabel(y1_label, fontsize=11, color=y1_color)
    ax1.tick_params(axis='y', labelcolor=y1_color)
    ax1.spines['top'].set_visible(False)

    for i, b in enumerate(valid_traj):
        ax1.annotate(f'n={metrics[b]["n"]:,}', (x_pos[i], y1_vals[i]),
                     textcoords="offset points", xytext=(-5, 10),
                     fontsize=5.5, color='gray', alpha=0.7)

    ax2 = ax1.twinx()
    ax2.plot(x_pos[:n_traj], y2_vals, 'o-', color=y2_color, linewidth=2.5, markersize=7, zorder=4)
    ax2.set_ylabel(y2_label, fontsize=11, color=y2_color)
    ax2.tick_params(axis='y', labelcolor=y2_color)
    ax2.spines['top'].set_visible(False)

    # Fork branches
    fork_x = x_pos[n_traj]
    branch_idx = n_traj - 1
    fork_styles = {
        'fork_compacted': ('#CC3333', 'D', 'Compacted sperm'),
        'fork_nuc_retaining': ('#993333', 's', 'Nuc-retaining sperm'),
    }
    legend_fork = []
    for fb, (fc, fm, fl) in fork_styles.items():
        if fb not in fork_y1 or metrics.get(fb, {}).get('n', 0) < 10:
            continue
        ax1.plot([x_pos[branch_idx], fork_x], [y1_vals[branch_idx], fork_y1[fb]],
                 '--', color=fc, linewidth=1.5, alpha=0.7)
        ax1.plot(fork_x, fork_y1[fb], fm, color=fc, markersize=9, zorder=6)
        ax2.plot([x_pos[branch_idx], fork_x], [y2_vals[branch_idx], fork_y2[fb]],
                 '--', color=fc, linewidth=1.5, alpha=0.5)
        ax2.plot(fork_x, fork_y2[fb], fm, color=fc, markersize=9, zorder=6, alpha=0.6)
        ax1.annotate(f'n={metrics[fb]["n"]:,}', (fork_x, fork_y1[fb]),
                     textcoords="offset points", xytext=(5, 5), fontsize=6.5, color=fc)
        legend_fork.append(Line2D([0], [0], marker=fm, color=fc, linestyle='--',
                                  markersize=8, label=fl))

    x_labels = list(valid_traj) + ['Mature\nsperm']
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, fontsize=8, rotation=30, ha='right')
    ax1.set_xlabel('Spermatogenesis stage (model sperm score)', fontsize=11)

    handles = [
        Line2D([0], [0], marker='o', color=y1_color, linewidth=2, markersize=7,
               label=y1_label.split('(')[0].strip()),
        Line2D([0], [0], marker='o', color=y2_color, linewidth=2, markersize=7,
               label=y2_label.split('(')[0].strip()),
    ] + legend_fork
    ax1.legend(handles=handles, fontsize=8, loc='center left', framealpha=0.9)

    plt.title(title, fontsize=13, fontweight='bold', pad=15)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved {fig_path.name}")
    plt.close()


def plot_all_variants(profiles, bin_info, fig_dir):
    """Generate all lineplot variants from cached profiles."""
    center = HALF_WINDOW
    c_lo = center - TSS_CENTER_HALF
    c_hi = center + TSS_CENTER_HALF
    # Flank regions for background: 4-5kb from center (both sides)
    flank_left = (0, 1000)       # positions 0-1000 = 4-5kb upstream
    flank_right = (9000, 10000)  # positions 9000-10000 = 4-5kb downstream

    traj_names = bin_info['trajectory_names']
    fork_names = bin_info['fork_names']

    # --- TSS metrics ---
    metrics_tss = extract_metrics(profiles, bin_info, c_lo, c_hi,
                                  bg_lo=flank_left, bg_hi=flank_right)
    # --- Flank (non-TSS) metrics: use the flank region as the "center" ---
    # Combine both flanks into one metric
    metrics_flank = {}
    for bname in list(traj_names) + list(fork_names):
        if bname not in profiles:
            continue
        p = profiles[bname]
        nuc_f = np.mean(np.concatenate([p['nuc'][0:1000], p['nuc'][9000:10000]]))
        prot_f = np.mean(np.concatenate([p['prot'][0:1000], p['prot'][9000:10000]]))
        small_f = np.mean(np.concatenate([p['small'][0:1000], p['small'][9000:10000]]))
        n_f = int(np.mean(np.concatenate([p['coverage'][0:1000], p['coverage'][9000:10000]])))
        metrics_flank[bname] = {'nuc': nuc_f, 'prot': prot_f, 'small': small_f, 'n': n_f}

    valid_traj = [b for b in traj_names if b in metrics_tss and metrics_tss[b]['n'] >= 50]
    n_traj = len(valid_traj)
    x_pos = np.arange(n_traj + 1)

    # Print debug info
    print("\nTSS center metrics (500bp):")
    for b in valid_traj:
        m = metrics_tss[b]
        non_prot = max(1 - m['prot'], 1e-8)
        nd = m['nuc'] / non_prot
        print(f"  {b:>12}: nuc={m['nuc']:.4f}, prot={m['prot']:.4f}, nuc_density={nd:.4f}, "
              f"prot_depl={m['prot_bg']-m['prot']:.4f}, n={m['n']:,}")
    for fb in fork_names:
        if fb in metrics_tss:
            m = metrics_tss[fb]
            non_prot = max(1 - m['prot'], 1e-8)
            nd = m['nuc'] / non_prot
            print(f"  {fb:>20}: nuc={m['nuc']:.4f}, prot={m['prot']:.4f}, nuc_density={nd:.4f}, "
                  f"prot_depl={m['prot_bg']-m['prot']:.4f}, n={m['n']:,}")
    print("\nFlank (non-TSS) metrics (4-5kb from center):")
    for b in valid_traj:
        m = metrics_flank[b]
        non_prot = max(1 - m['prot'], 1e-8)
        nd = m['nuc'] / non_prot
        print(f"  {b:>12}: nuc={m['nuc']:.4f}, prot={m['prot']:.4f}, nuc_density={nd:.4f}, n={m['n']:,}")
    for fb in fork_names:
        if fb in metrics_flank:
            m = metrics_flank[fb]
            non_prot = max(1 - m['prot'], 1e-8)
            nd = m['nuc'] / non_prot
            print(f"  {fb:>20}: nuc={m['nuc']:.4f}, prot={m['prot']:.4f}, nuc_density={nd:.4f}, n={m['n']:,}")

    # ===========================
    # Plot 1: Both going down — nuc occupancy + protamine depletion at TSS
    # ===========================
    y_nuc = [metrics_tss[b]['nuc'] for b in valid_traj]
    y_prot_depl = [metrics_tss[b]['prot_bg'] - metrics_tss[b]['prot'] for b in valid_traj]
    fork_y1 = {fb: metrics_tss[fb]['nuc'] for fb in fork_names if fb in metrics_tss}
    fork_y2 = {fb: metrics_tss[fb].get('prot_bg', 0) - metrics_tss[fb]['prot']
               for fb in fork_names if fb in metrics_tss}

    _make_lineplot(x_pos, n_traj, valid_traj, metrics_tss, fork_names,
                   y_nuc, 'Nucleosome occupancy at TSS (\u00b1250bp)', '#33AA33',
                   y_prot_depl, 'Protamine depletion at TSS (bg \u2212 center)', '#CC6600',
                   fork_y1, fork_y2,
                   'Chromatin remodeling at TSSs during spermatogenesis',
                   fig_dir / 'tss_trajectory_both_down.png')

    # ===========================
    # Plot 2: Nucleosome density — nuc / (1 - prot) at TSS
    # ===========================
    y_nuc_dens = []
    for b in valid_traj:
        m = metrics_tss[b]
        non_prot = max(1 - m['prot'], 1e-8)
        y_nuc_dens.append(m['nuc'] / non_prot)
    y_prot_tss = [metrics_tss[b]['prot'] for b in valid_traj]
    fork_y1_nd = {}
    fork_y2_nd = {}
    for fb in fork_names:
        if fb in metrics_tss:
            m = metrics_tss[fb]
            non_prot = max(1 - m['prot'], 1e-8)
            fork_y1_nd[fb] = m['nuc'] / non_prot
            fork_y2_nd[fb] = m['prot']

    _make_lineplot(x_pos, n_traj, valid_traj, metrics_tss, fork_names,
                   y_nuc_dens, 'Nucleosome density (nuc / non-protamine)', '#33AA33',
                   y_prot_tss, 'Protamine occupancy at TSS (\u00b1250bp)', '#CC6600',
                   fork_y1_nd, fork_y2_nd,
                   'Nucleosome density at TSSs during spermatogenesis',
                   fig_dir / 'tss_trajectory_nuc_density.png')

    # ===========================
    # Plot 3: Non-TSS (flanks) — nuc occupancy + protamine occupancy
    # ===========================
    valid_flank = [b for b in valid_traj if b in metrics_flank]
    y_nuc_f = [metrics_flank[b]['nuc'] for b in valid_flank]
    y_prot_f = [metrics_flank[b]['prot'] for b in valid_flank]
    fork_y1_f = {fb: metrics_flank[fb]['nuc'] for fb in fork_names if fb in metrics_flank}
    fork_y2_f = {fb: metrics_flank[fb]['prot'] for fb in fork_names if fb in metrics_flank}

    _make_lineplot(x_pos, n_traj, valid_flank, metrics_flank, fork_names,
                   y_nuc_f, 'Nucleosome occupancy (non-TSS flanks)', '#33AA33',
                   y_prot_f, 'Protamine occupancy (non-TSS flanks)', '#CC6600',
                   fork_y1_f, fork_y2_f,
                   'Chromatin remodeling at non-TSS regions during spermatogenesis',
                   fig_dir / 'tss_trajectory_nontss.png')

    # ===========================
    # Plot 4: Non-TSS nucleosome density
    # ===========================
    y_nuc_dens_f = []
    for b in valid_flank:
        m = metrics_flank[b]
        non_prot = max(1 - m['prot'], 1e-8)
        y_nuc_dens_f.append(m['nuc'] / non_prot)
    fork_y1_ndf = {}
    fork_y2_ndf = {}
    for fb in fork_names:
        if fb in metrics_flank:
            m = metrics_flank[fb]
            non_prot = max(1 - m['prot'], 1e-8)
            fork_y1_ndf[fb] = m['nuc'] / non_prot
            fork_y2_ndf[fb] = m['prot']

    _make_lineplot(x_pos, n_traj, valid_flank, metrics_flank, fork_names,
                   y_nuc_dens_f, 'Nucleosome density (nuc / non-protamine)', '#33AA33',
                   y_prot_f, 'Protamine occupancy (non-TSS flanks)', '#CC6600',
                   fork_y1_ndf, fork_y2_ndf,
                   'Nucleosome density at non-TSS regions during spermatogenesis',
                   fig_dir / 'tss_trajectory_nontss_nuc_density.png')


def main():
    tss_path = './data/protein_coding_tss.tsv'
    gtf_path = PROJECT_DIR / 'data' / 'reference' / 'hg38.gtf'
    if Path(tss_path).exists():
        tss_list = load_tss(tss_path)
    else:
        tss_list = extract_tss_from_gtf(gtf_path)
        with open(tss_path, 'w') as f:
            f.write('chrom\ttss\tstrand\tgene\n')
            for c, p, s, g in tss_list:
                f.write(f'{c}\t{p}\t{s}\t{g}\n')
    print(f"{len(tss_list)} TSSs")

    # Load metadata
    print("Loading metadata...")
    scores = np.load(EMB_DIR / 'scores.npy').flatten()
    chroms_raw = np.load(EMB_DIR / 'chroms.npy')
    starts = np.load(EMB_DIR / 'starts.npy').flatten()
    ends = np.load(EMB_DIR / 'ends.npy').flatten()
    prot_fracs = np.load(EMB_DIR / 'prot_fracs.npy').flatten()
    chroms = [str(c) if chroms_raw.dtype.kind in ('U', 'S', 'O') else
              (c.decode() if isinstance(c, bytes) else str(c))
              for c in chroms_raw]
    print(f"  {len(scores):,} reads")

    # TSS overlaps
    pairs_path = OUT_DIR / 'tss_pairs.npz'
    if pairs_path.exists():
        data = np.load(pairs_path)
        read_indices, tss_centers = data['read_indices'], data['tss_centers']
    else:
        read_indices, tss_centers = find_tss_overlaps(chroms, starts, ends, tss_list)
        np.savez(pairs_path, read_indices=read_indices, tss_centers=tss_centers)
    print(f"  {len(read_indices):,} read-TSS pairs")

    # --- Fork split using UMAP1 median ---
    umap_path = OUT_DIR / 'tss_umap2d.npy'
    umap_2d = np.load(umap_path)
    read_to_fork = split_fork_umap(read_indices, scores, prot_fracs, umap_2d)

    # --- Assign pairs to score bins (with island filter) ---
    bin_assignments = {}
    pair_scores = scores[read_indices]
    pair_pf = prot_fracs[read_indices]

    for bname, blo, bhi in SCORE_BINS:
        score_mask = (pair_scores >= blo) & (pair_scores < bhi)
        if blo >= 0.05:
            # Filter off-trajectory island: require minimum protamine
            score_mask = score_mask & (pair_pf >= MIN_PROT_FRAC)
        bin_assignments[bname] = np.where(score_mask)[0]

    # Fork bins: use the UMAP-based labels
    fork_comp_idx = []
    fork_nuc_idx = []
    for pi in range(len(read_indices)):
        ri = read_indices[pi]
        if ri in read_to_fork:
            if read_to_fork[ri] == 'compacted':
                fork_comp_idx.append(pi)
            else:
                fork_nuc_idx.append(pi)
    bin_assignments['fork_compacted'] = np.array(fork_comp_idx, dtype=np.int64)
    bin_assignments['fork_nuc_retaining'] = np.array(fork_nuc_idx, dtype=np.int64)

    print(f"\nBin sizes (after island filter, prot_frac >= {MIN_PROT_FRAC} for non-somatic):")
    for bname, idx in bin_assignments.items():
        print(f"  {bname}: {len(idx):,}")

    bin_info = {
        'trajectory_names': [b[0] for b in SCORE_BINS],
        'fork_names': ['fork_compacted', 'fork_nuc_retaining'],
    }

    # --- Compute profiles ---
    profiles_path = OUT_DIR / 'tss_profiles_v4.pkl'
    if profiles_path.exists():
        print(f"\nLoading cached profiles...")
        with open(profiles_path, 'rb') as f:
            profiles = pickle.load(f)['profiles']
    else:
        bed_path = Path('./data/bed/PS30065_fp.bed')
        if not bed_path.exists():
            bed_path = BED_DIR / 'PS30065_fp.bed'
        profiles = compute_profiles(read_indices, tss_centers, bin_assignments, bed_path)
        with open(profiles_path, 'wb') as f:
            pickle.dump({'profiles': profiles, 'bin_info': bin_info}, f)

    # --- UMAP debug plot: show which bin each read falls into ---
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_umap_bins(read_indices, umap_2d, bin_assignments, bin_info,
                   FIG_DIR / 'tss_trajectory_umap_bins.png')

    # --- All lineplot variants ---
    plot_all_variants(profiles, bin_info, FIG_DIR)


if __name__ == '__main__':
    main()
