"""Shared CLI helpers."""

import argparse
import sys
from pathlib import Path

import yaml


def add_repo_to_path():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    return p


def apply_overrides(cfg: dict, args) -> dict:
    """Apply a few common command-line overrides onto the config dict."""
    if getattr(args, "smoke", False):
        cfg["training"]["smoke"] = True
        cfg["preprocess"]["max_volumes_per_dataset"] = max(
            1, cfg["preprocess"].get("max_volumes_per_dataset") or 2
        )
        cfg["generation"]["num_patients"] = min(4, cfg["generation"]["num_patients"])
        cfg["generation"]["slices_per_volume"] = min(16, cfg["generation"]["slices_per_volume"])
        cfg["generation"]["sampler_steps"] = min(10, cfg["generation"]["sampler_steps"])
    if getattr(args, "max_per_dataset", None) is not None:
        cfg["preprocess"]["max_volumes_per_dataset"] = args.max_per_dataset
    if getattr(args, "device", None):
        cfg["training"]["device"] = args.device
    if getattr(args, "cvae_epochs", None) is not None:
        cfg["training"]["cvae"]["epochs"] = int(args.cvae_epochs)
    if getattr(args, "diff_epochs", None) is not None:
        cfg["training"]["diffusion"]["epochs"] = int(args.diff_epochs)
    if getattr(args, "cvae_batch", None) is not None:
        cfg["training"]["cvae"]["batch_size"] = int(args.cvae_batch)
    if getattr(args, "diff_batch", None) is not None:
        cfg["training"]["diffusion"]["batch_size"] = int(args.diff_batch)
    return cfg
