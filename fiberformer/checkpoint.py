"""Utilities for loading FiberFormer checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fiberformer.models.transformer import FootprintTransformer


DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parent / "weights" / "best_model.pt"


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA when available and CPU otherwise."""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return resolved


def model_kwargs_from_config(config: dict[str, Any], *, inference: bool = True) -> dict[str, Any]:
    """Translate the checkpoint model config into constructor arguments."""
    model_cfg = config["model"]
    kwargs: dict[str, Any] = {
        "d_model": model_cfg["d_model"],
        "nhead": model_cfg["nhead"],
        "n_layers": model_cfg["n_layers"],
        "d_ff": model_cfg["d_ff"],
        "dropout": 0.0 if inference else model_cfg.get("dropout", 0.1),
        "logit_scale_init": model_cfg.get("logit_scale_init", 10.0),
        "pool": model_cfg.get("pool", "cls"),
        "conv_kernel": model_cfg.get("conv_kernel", 0),
        "positional": model_cfg.get("positional", "rope"),
    }
    for optional_key in ("n_type_ids", "n_head_b_type_classes"):
        if optional_key in model_cfg:
            kwargs[optional_key] = model_cfg[optional_key]
    return kwargs


def load_model(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    device: str | torch.device = "auto",
) -> tuple[FootprintTransformer, dict[str, Any], dict[str, Any], torch.device]:
    """Load a checkpoint and return ``(model, config, checkpoint, device)``.

    The model is put in evaluation mode and dropout is disabled. The complete
    checkpoint is returned so callers can record training metadata such as the
    epoch and validation metrics.
    """
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"FiberFormer checkpoint not found: {path}")

    resolved_device = resolve_device(device)
    checkpoint = torch.load(path, map_location=resolved_device, weights_only=False)
    if "model_state_dict" not in checkpoint or "config" not in checkpoint:
        raise ValueError(
            f"Checkpoint {path} must contain 'model_state_dict' and 'config'"
        )

    config = checkpoint["config"]
    model = FootprintTransformer(**model_kwargs_from_config(config)).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, checkpoint, resolved_device
