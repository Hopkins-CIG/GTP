from scripts.run_testing.mr_ct.common import run_mr_method
from scripts.test_reconstruction.test_recon_mr_ct_prox_gtp_unpaired import run_experiment

if __name__ == '__main__':
    run_mr_method(run_experiment, 'prox_gtp_unpaired', 'la', (30, 60, 90, 120), 'Run limited-angle unpaired MR-to-CT Prox-GTP inference')
