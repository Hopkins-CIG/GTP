from pathlib import Path

import torch


VALID_EVALUATION_SETS = ("test", "val")


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_root(path: str | Path | None, default: Path) -> Path:
    root = Path(path).expanduser() if path is not None else default
    return root.resolve()


def resolve_inference_roots(
    data_root: str | Path | None = None,
    checkpoint_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    root = project_root()
    return (
        resolve_root(data_root, root / "data"),
        resolve_root(checkpoint_root, root / "checkpoints"),
        resolve_root(output_root, root / "results"),
    )


def resolve_output_subdir(output_root: Path, save_dir: str | Path) -> Path:
    """Resolve a run save directory under output_root without duplicating a leading 'results/' segment."""
    subdir = Path(save_dir).expanduser()
    if subdir.is_absolute():
        return subdir.resolve()
    if subdir.parts[:1] == ("results",):
        subdir = Path(*subdir.parts[1:])
    return (output_root / subdir).resolve()


def validate_evaluation_set(evaluation_set: str) -> str:
    if evaluation_set not in VALID_EVALUATION_SETS:
        choices = ", ".join(VALID_EVALUATION_SETS)
        raise ValueError(f"evaluation_set must be one of: {choices}")
    return evaluation_set


def resolve_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {device!r} was requested, but CUDA is unavailable. "
                "Use a CUDA-enabled environment and driver."
            )
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {device!r} is unavailable; "
                f"{torch.cuda.device_count()} CUDA device(s) detected."
            )
    return resolved


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {description}: {path}\n"
            "Download the required inference assets and follow the documented layout."
        )
    return path


def data_file(data_root: Path, dataset: str, evaluation_set: str) -> Path:
    validate_evaluation_set(evaluation_set)
    return require_file(
        data_root / dataset / f"{evaluation_set}.pt",
        f"{dataset} {evaluation_set} data",
    )


def checkpoint_file(checkpoint_root: Path, filename: str) -> Path:
    return require_file(checkpoint_root / filename, f"checkpoint {filename}")


def psnr(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    peak_value: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Compute per-sample PSNR with an explicit image peak value."""
    if predictions.shape != targets.shape:
        raise ValueError(
            f"PSNR inputs must have the same shape, got {predictions.shape} and {targets.shape}"
        )
    if predictions.ndim < 2:
        raise ValueError("PSNR inputs must include a batch dimension")
    mse = torch.mean(
        (predictions - targets) ** 2,
        dim=tuple(range(1, predictions.ndim)),
    )
    peak_value = torch.as_tensor(peak_value, dtype=mse.dtype, device=mse.device)
    return 10 * torch.log10(peak_value.square() / mse)


def psnr_neg_one_one(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute PSNR after converting diffusion images from [-1, 1] to [0, 1]."""
    return psnr((predictions.clamp(-1, 1) + 1) / 2, (targets.clamp(-1, 1) + 1) / 2)
