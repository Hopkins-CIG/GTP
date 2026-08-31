from scripts.run_testing.mr_ct.common import run_mr_method
from scripts.test_reconstruction.test_recon_mr_ct_daps_gtp import run_experiment

if __name__ == '__main__':
    run_mr_method(run_experiment, 'daps_gtp', 'la', (30, 60, 90, 120), 'Run limited-angle MR-to-CT DAPS-GTP inference')
