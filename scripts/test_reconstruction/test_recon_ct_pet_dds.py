import torch
import numpy as np
import os
import glob
from src.models.ddbm import CMSI
from src.models.karras_diffusion import karras_sample
from src.inference_config import psnr as compute_psnr
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from src.physics import pet
from src.physics.pet import PET2D
import time
from jsonargparse import auto_cli
from src.inference_config import checkpoint_file, data_file, resolve_device, resolve_inference_roots

'''
Implements PET-DDS from Singh, Imraj RD, et al. "Score-Based Generative Models for PET Image Reconstruction."
Machine Learning for Biomedical Imaging 2. Special Issue for Generative Models (2024): 547-585.
'''

torch.manual_seed(123)

def run_experiment(
        # Data saving and loading arguments
        save_dir: str = None,
        evaluation_set: str = 'test',
        limit_batches: int = None,
        # Model and evaluation arguments 
        use_ct: bool = True, # Whether to use conditional model
        # Task arguments
        total_counts: float = 1e5, # Total counts for PET simulation
        # Diffusion sampling arguments
        num_steps: int = 200,
        alpha: float = 1e-4, # Data consistency step size
        lamda: float = 1, # Regularization parameter
        prox_steps: int = 50,
        # Device
        device: str = 'cuda:0',
        data_root: str = None,
        checkpoint_root: str = None,
        output_root: str = None,
        ):

    data_root, checkpoint_root, default_output_root = resolve_inference_roots(
        data_root, checkpoint_root, output_root
    )
    device = resolve_device(device)
    save_dir = str(save_dir or default_output_root / 'pet_recon') + '/'
    
    config = {
        'save_dir': save_dir,
        'evaluation_set': evaluation_set,
        'limit_batches': limit_batches,
        'use_ct': use_ct,
        'total_counts': total_counts,
        'num_steps': num_steps,
        'alpha': alpha,
        'lamda': lamda,
        'prox_steps': prox_steps,
        'device': device
    }

    method = 'pet_dds'

    save_dir = f'{save_dir}{evaluation_set}/total_counts_{total_counts}/{method}/{prox_steps}_prox_steps/alpha_{alpha}/lamda_{lamda}/'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f'{save_dir}config.yml', 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    # -------- Loading model --------

    ckpt_path = checkpoint_file(checkpoint_root, 'ct_conditional_pet.ckpt')

    model = CMSI.load_from_checkpoint(ckpt_path).to(device)
    model.eval()

    # -------- Loading test data --------

    test_ct_slices = torch.load(data_file(data_root, 'ct_pet/ct', evaluation_set)).to(device)
    test_pet_slices = torch.load(data_file(data_root, 'ct_pet/pet', evaluation_set)).to(device)

    # -------- Evaluation --------
    psnr_values = []
    ssim_values = []
    runtimes = []

    # -------- Batchify inference --------
    batch_size = 1
    num_samples = test_ct_slices.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    if limit_batches is not None:
        num_batches = min(num_batches, limit_batches)

    reconstructions = []

    def data_consistency(x, y_meas, diffusion, denoiser, y, sigma_next, x_T, ld, alpha, op, **kwargs):
        x_denoised = denoiser(x, y, sigma_next.unsqueeze(0).to(device))

        x_denoised = (x_denoised + 1) / 2
        x_denoised = x_denoised / ld
        x_dc = x_denoised.clone().clamp(1e-12)

        A_T_ones = op.A_T(torch.ones_like(y_meas))
        
        for _ in range(prox_steps):
            grad_nll = op.A_T(1 - y_meas / op.A(x_dc))
            grad_prior = lamda / 2 * (x_dc - x_denoised)
            step_size = alpha * x_dc / A_T_ones
            x_dc = x_dc - step_size * (grad_nll + grad_prior)
            x_dc = x_dc.clamp(1e-12)

        x_dc = x_dc * ld
        x_dc = 2 * x_dc.clamp(0) - 1
        x_dc  = diffusion.bridge_sample(x_dc, torch.randn_like(x_T), sigma_next.unsqueeze(0).to(device)) # Using torch.randn_like(x_T) works much better than using x_T
        return x_dc


    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Evaluating Batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            
            if use_ct:
                init_cond = torch.randn_like(test_pet_slices[start_idx:end_idx]).unsqueeze(1)
                y = test_ct_slices[start_idx:end_idx].unsqueeze(1)
            else:
                init_cond = torch.randn_like(test_pet_slices[start_idx:end_idx]).unsqueeze(1)
                y = init_cond

            op = PET2D(y, device=device)
            total_counts = float(total_counts)
            pet_img = test_pet_slices[start_idx:end_idx].unsqueeze(0).to(device)
            ld = op.A(pet_img).sum().item()/total_counts
            y_meas = torch.poisson(op.A(1 / ld * pet_img).clamp(0))

            start_time = time.perf_counter()

            sample, path, nfe = karras_sample(model.denoiser, model.unet,
                                            x_T=init_cond,
                                            y=y,
                                            steps=200,
                                            churn_step_ratio=0.33,
                                            dc_type='xt',
                                            dc_step=data_consistency,
                                            dc_step_kwargs={'y_meas': y_meas, 'ld': ld, 'alpha': alpha, 'op': op},
                                            clip_denoised=False
                                            )
            
            end_time = time.perf_counter()
            runtimes.append(end_time - start_time)
            
            reconstructions.append(((sample + 1) / 2).clamp(0))

    for recon, target in zip(reconstructions, test_pet_slices):
        psnr_value = compute_psnr(
            recon.cpu(),
            target.cpu().unsqueeze(0).unsqueeze(0),
            peak_value=target.max(),
        )
        psnr_values.append(psnr_value.item())
        ssim = SSIM()(recon.cpu(), target.cpu().unsqueeze(0).unsqueeze(0))
        ssim_values.append(ssim.item())

    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    avg_runtime = sum(runtimes) / len(runtimes)

    print(f"Average PSNR: {avg_psnr:.3f} dB")
    print(f"Average SSIM: {avg_ssim:.3f}")

    torch.save(torch.tensor(psnr_values), f'{save_dir}psnr_values.pt')
    torch.save(torch.tensor(avg_psnr), f'{save_dir}avg_psnr.pt')
    torch.save(torch.tensor(ssim_values), f'{save_dir}ssim_values.pt')
    torch.save(torch.tensor(avg_ssim), f'{save_dir}avg_ssim.pt')

    torch.save(torch.stack(reconstructions).cpu(), f'{save_dir}reconstructions.pt')
    torch.save(test_ct_slices.cpu(), f'{save_dir}target_slices.pt')
    torch.save(torch.tensor(runtimes), f'{save_dir}runtimes.pt')
    torch.save(torch.tensor(avg_runtime), f'{save_dir}avg_runtime.pt')

    # Log mean PSNR to a text file
    with open(f'{save_dir}metrics.txt', 'w') as f:
        f.write(f"Average PSNR: {avg_psnr:.3f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.3f}\n")
        f.write(f"Average Runtime: {avg_runtime:.3f} seconds\n")
        f.write("Metric values for each slice:\n")
        for i, (psnr, ssim, runtime) in enumerate(zip(psnr_values, ssim_values, runtimes)):
            f.write(f"Case {i}: PSNR = {psnr:.2f} dB, SSIM = {ssim:.4f}, Runtime = {runtime:.3f} seconds\n")

    # Save reconstructions
    for i, recon in enumerate(reconstructions):
        slice_idx = i % 3
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(test_ct_slices[i].cpu().squeeze(), cmap='gray')
        plt.title(f'Input - Case {i}\n')
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.imshow(recon.cpu().squeeze(), cmap='hot', vmin=0, vmax=5)
        plt.title(f'Reconstructed - Case {i}\nPSNR: {psnr_values[i]:.2f} dB')
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.imshow(test_pet_slices[i].cpu().squeeze(), cmap='hot', vmin=0, vmax=5)
        plt.title(f'Target - Case {i}\n')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(f"{save_dir}recon_case{i}.png")
        plt.close()

if __name__ == "__main__":
    auto_cli(run_experiment)