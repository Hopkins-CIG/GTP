from pathlib import Path

import yaml

from src.inference_config import project_root


def load_test_config(
    method: str,
    task: str,
    n_views: int,
    results_root: str | Path | None = None,
) -> dict:
    """Load a public MR-to-CT config from the version-controlled manifest."""
    manifest_path = project_root() / "configs" / "testing" / "mr_ct.yml"
    with manifest_path.open() as manifest_file:
        manifest = yaml.safe_load(manifest_file) or {}
    common = manifest.get("common", {}).get(method)
    overrides = manifest.get("settings", {}).get(task, {}).get(n_views, {}).get(method, {})
    if common is None or not isinstance(overrides, dict):
        raise FileNotFoundError(
            f"No public test config found for {method} ({task}, {n_views} views)"
        )
    config = {**common, **overrides, "n_views": n_views, "task": task, "snr": 50.0}
    return config


def load_pet_test_config(
    method: str,
    total_counts: float,
    results_root: str | Path | None = None,
) -> dict:
    """Load a public CT-to-PET config from the version-controlled manifest."""
    manifest_path = project_root() / "configs" / "testing" / "pet.yml"
    with manifest_path.open() as manifest_file:
        manifest = yaml.safe_load(manifest_file) or {}
    config = manifest.get("settings", {}).get(int(total_counts), {}).get(method)
    if config is None:
        raise FileNotFoundError(
            f"No public test config found for {method} ({total_counts:g} counts)"
        )
    return {**config, "total_counts": total_counts}
