import argparse
from collections.abc import Callable, Sequence

from configs.testing import load_test_config


def run_mr_method(
    run_experiment: Callable[..., None],
    method: str,
    task: str,
    views: Sequence[int],
    description: str,
) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--device', default='cuda:0', help='CUDA device to use')
    parser.add_argument('--data-root', default=None, help='Root containing the downloaded datasets')
    parser.add_argument('--checkpoint-root', default=None, help='Directory containing model checkpoints')
    parser.add_argument('--output-root', default=None, help='Root for inference results')
    args = parser.parse_args()

    for n_views in views:
        config = load_test_config(method, task, n_views)
        config.update(
            evaluation_set='test',
            device=args.device,
            data_root=args.data_root,
            checkpoint_root=args.checkpoint_root,
            output_root=args.output_root,
        )
        run_experiment(**config)
