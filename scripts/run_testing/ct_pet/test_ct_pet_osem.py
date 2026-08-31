from scripts.run_testing.ct_pet.common import run_pet_method
from scripts.test_reconstruction.test_recon_ct_pet_osem import run_experiment
if __name__ == '__main__':
    run_pet_method(
        run_experiment,
        'osem',
        (1e5, 5e5, 1e6),
        'Run CT-to-PET OSEM inference',
    )
