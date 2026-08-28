"""
Contrastive losses for Head C.

- SupCon: Supervised contrastive on labeled reads
- SimCLR: InfoNCE between augmented view pairs
- Combined: SupCon on labeled + SimCLR on all
"""

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    embeddings: torch.Tensor,   # (N, D) L2-normalized
    labels: torch.Tensor,       # (N,) int: 0=somatic, 1=sperm
    temperature: float = 0.07,
) -> torch.Tensor:
    """SupCon loss: pull same-class together, push different-class apart.

    Only operates on labeled reads (label >= 0).
    """
    device = embeddings.device
    N = embeddings.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=device)

    # Similarity matrix
    sim = torch.matmul(embeddings, embeddings.t()) / temperature  # (N, N)

    # Same-class mask (excludes self)
    label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
    self_mask = ~torch.eye(N, dtype=torch.bool, device=device)
    pos_mask = label_mask & self_mask

    # For numerical stability
    sim_max = sim.max(dim=1, keepdim=True).values
    sim = sim - sim_max.detach()

    # Log-sum-exp over all negatives + positives (excluding self)
    exp_sim = torch.exp(sim) * self_mask.float()
    log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # Mean of log-prob over positive pairs
    log_prob = sim - log_sum_exp  # (N, N)

    # Average over positive pairs per anchor
    n_pos = pos_mask.float().sum(dim=1)  # (N,)
    valid = n_pos > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    loss = -(log_prob * pos_mask.float()).sum(dim=1) / (n_pos + 1e-8)
    return loss[valid].mean()


def simclr_loss(
    z1: torch.Tensor,   # (N, D) L2-normalized, view 1
    z2: torch.Tensor,   # (N, D) L2-normalized, view 2
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE loss between augmented view pairs."""
    device = z1.device
    N = z1.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=device)

    # Concatenate views: [z1_0, z1_1, ..., z2_0, z2_1, ...]
    z = torch.cat([z1, z2], dim=0)  # (2N, D)
    sim = torch.matmul(z, z.t()) / temperature  # (2N, 2N)

    # Positive pairs: (i, i+N) and (i+N, i)
    labels = torch.arange(N, device=device)
    labels = torch.cat([labels + N, labels])  # positive pair indices

    # Exclude self-similarity
    mask = ~torch.eye(2 * N, dtype=torch.bool, device=device)
    sim = sim.masked_fill(~mask, float('-inf'))

    return F.cross_entropy(sim, labels)


def combined_contrastive_loss(
    proj_c: torch.Tensor,           # (B, D) all projections
    labels: torch.Tensor,           # (B,) -1/0/1
    temperature: float = 0.07,
) -> torch.Tensor:
    """SupCon on labeled reads within the batch.

    For full contrastive with augmented views, use simclr_loss separately.
    """
    # Filter labeled reads
    labeled_mask = labels >= 0
    if labeled_mask.sum() < 2:
        return torch.tensor(0.0, device=proj_c.device)

    labeled_emb = proj_c[labeled_mask]
    labeled_labels = labels[labeled_mask]

    return supervised_contrastive_loss(labeled_emb, labeled_labels, temperature)
