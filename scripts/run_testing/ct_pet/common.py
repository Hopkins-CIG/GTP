import argparse
from collections.abc import Callable, Sequence

from configs.testing import load_pet_test_config


def run_pet_method(
    run_experiment: Callable[..., None],
    method: str,
    counts: Sequence[float],
    description: str,
    include_checkpoint_root: bool = False,
) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--device', default='cuda:0', help='CUDA device to use')
    parser.add_argument('--data-root', default=None, help='Root containing the downloaded datasets')
    if include_checkpoint_root:
        parser.add_argument('--checkpoint-root', default=None, help='Directory containing CT-to-PET checkpoints')
    parser.add_argument('--output-root', default=None, help='Root for inference results')
    parser.add_argument('--evaluation-set', choices=('test', 'val'), default='test')
    args = parser.parse_args()

    for total_counts in counts:
        config = load_pet_test_config(method, total_counts)
        config.update(
            device=args.device,
            data_root=args.data_root,
            output_root=args.output_root,
            evaluation_set=args.evaluation_set,
        )
        if include_checkpoint_root:
            config['checkpoint_root'] = args.checkpoint_root
        run_experiment(**config)