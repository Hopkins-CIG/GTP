from scripts.run_testing.mr_ct.common import run_mr_method
from scripts.test_reconstruction.test_recon_mr_ct_dds_gtp import run_experiment

if __name__ == '__main__':
    run_mr_method(run_experiment, 'dds_gtp', 'sv', (4, 8, 16, 32), 'Run sparse-view MR-to-CT DDS-GTP inference')
