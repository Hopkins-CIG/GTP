import argparse
from scripts.test_reconstruction.test_recon_mr_ct_fbp import run_experiment

def main() -> None:
    parser = argparse.ArgumentParser(description='Run limited-angle MR-to-CT FBP inference')
    parser.add_argument('--device', type=str, default='cuda:0', help='CUDA device to use')
    parser.add_argument('--data-root', type=str, default=None, help='Root containing the downloaded datasets')
    parser.add_argument('--output-root', type=str, default=None, help='Root for inference results')
    parser.add_argument('--limit-batches', type=int, default=None, help='Maximum number of samples per view count')
    args = parser.parse_args()

    for n_views in (30, 60, 90, 120):
        run_experiment(
            task='la',
            evaluation_set='test',
            n_views=n_views,
            device=args.device,
            data_root=args.data_root,
            output_root=args.output_root,
            limit_batches=args.limit_batches,
        )


if __name__ == '__main__':
    main()

