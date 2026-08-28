"""Build the HDF5 cache used to train FiberFormer."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from fiberformer.data.dataset import build_hdf5_cache_v3


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_DIR / "configs" / "fiberformer.yaml"


def _resolve_from_project(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd().resolve() / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the FiberFormer HDF5 cache")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="override config data.data_dir (directory containing footprint files)",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="override config data.cache_path",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    data_config = config["data"]

    data_dir = _resolve_from_project(
        args.data_dir if args.data_dir is not None
        else data_config.get("data_dir", "./data/bigbed")
    )
    cache_path = _resolve_from_project(
        args.cache_path if args.cache_path is not None else data_config["cache_path"]
    )
    if cache_path.exists():
        print(
            f"Cache already exists at {cache_path} "
            f"({cache_path.stat().st_size / 1e9:.2f} GB). "
            "Move or delete it explicitly before rebuilding."
        )
        return

    sperm_paths = [data_dir / name for name in data_config["sperm_paths"]]
    testes_paths = [data_dir / name for name in data_config["testes_paths"]]
    somatic_paths = [data_dir / name for name in data_config["somatic_paths"]]
    missing = [path for path in sperm_paths + testes_paths + somatic_paths
               if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing {len(missing)} input file(s):\n{formatted}")

    print(f"Configuration: {config_path}")
    print(f"Input directory: {data_dir}")
    print(f"Output cache: {cache_path}")
    build_hdf5_cache_v3(
        cache_path=cache_path,
        sperm_paths=sperm_paths,
        testes_paths=testes_paths,
        somatic_paths=somatic_paths,
        pure_sperm_names=tuple(data_config.get("pure_sperm_names", ["m6a_200_fp"])),
        maturity_threshold=data_config.get("maturity_threshold", 0.90),
        max_tokens=data_config["max_tokens"],
        n_per_sperm=data_config["n_per_sperm"],
        n_per_testes=data_config["n_per_testes"],
        n_per_somatic=data_config["n_per_somatic"],
        size_encoding=data_config["size_encoding"],
        use_lacuna_tokens=data_config.get("use_lacuna_tokens", False),
    )
    print(f"Done. Cache written to {cache_path}")


if __name__ == "__main__":
    main()
