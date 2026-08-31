from scripts.run_testing.ct_pet.common import run_pet_method
from scripts.test_reconstruction.test_recon_ct_pet_dds import run_experiment
if __name__ == '__main__':
    run_pet_method(
        run_experiment,
        'pet_dds',
        (1e5, 5e5, 1e6),
        'Run CT-to-PET PET-DDS inference',
        include_checkpoint_root=True,
    )
