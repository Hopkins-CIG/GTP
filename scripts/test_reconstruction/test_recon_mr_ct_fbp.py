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
from src.physics.pbct import PBCT
import time
from jsonargparse import auto_cli
from src.inference_config import data_file, resolve_device, resolve_inference_roots

torch.manual_seed(123)

'''
Script for getting the FBP reconstruction metrics
'''

def run_experiment(
        # Data saving and loading arguments
        save_dir: str = None,
        evaluation_set: str = 'test',
        limit_batches: int = None,
        # Task arguments
        n_views: int = 8, # Number of views for CT operator
        task: str = 'sv', # Task type: sparse-view (sv) or limited-angle (la)
        snr: float = 50.0, # Signal-to-noise ratio for measurement noise in dB
        # Device
        device: str = 'cuda:0',
        data_root: str = None,
        output_root: str = None,
        ):

    data_root, _, default_output_root = resolve_inference_roots(data_root, output_root=output_root)
    device = resolve_device(device)
    save_dir = str(save_dir or default_output_root / 'ct_recon') + '/'
    
    # Save the config used for this experiment as a yml file
    config = {
        'save_dir': save_dir,
        'evaluation_set': evaluation_set,
        'n_views': n_views,
        'task': task,
        'snr': snr,
        'device': device
    }
    
    # Set save_dir name based on task and config
    method = 'fbp'
    save_dir = f'{save_dir}{evaluation_set}/{task}/{n_views}_views_snr_{snr}_dB/{method}/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f'{save_dir}config.yml', 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")


    # -------- Loading test data --------

    test_ct_slices = (torch.load(data_file(data_root, 'mr_ct/ct', evaluation_set)).to(device) + 1) / 2

    # -------- Batchify inference --------
    batch_size = 1
    num_samples = test_ct_slices.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    if limit_batches is not None:
        num_batches = min(num_batches, limit_batches)

    reconstructions = []
    runtimes = []

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Evaluating Batches"):

            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            
            if task == 'sv':
                ct_op = PBCT(n_views, 1, 256, device=device)
            elif task == 'la':
                ct_op = PBCT(n_views, 1, 256, device=device, angles=np.arange(n_views, dtype=np.float32))
            test_slc = test_ct_slices[start_idx:end_idx].unsqueeze(1)
            y_meas = ct_op.A(test_slc)
            
            avg_signal_power = torch.mean(y_meas ** 2)
            noise_power = avg_signal_power / (10 ** (snr / 10))
            sigma_meas = torch.sqrt(noise_power)
            noise = sigma_meas * torch.randn_like(y_meas)
            y_meas = y_meas + noise

            start_time = time.perf_counter()

            fbp_recon = ct_op.A_pinv(y_meas)
            
            end_time = time.perf_counter()
            runtimes.append(end_time - start_time)
            
            reconstructions.append(fbp_recon)

    # Compute metrics
    psnr_values = []
    ssim_values = []

    for recon, target in zip(reconstructions, test_ct_slices):
        psnr_value = compute_psnr(
            recon.cpu().clamp(0, 1),
            target.cpu().clamp(0, 1).unsqueeze(0).unsqueeze(0),
        )
        ssim = SSIM()(recon.cpu(), target.cpu().unsqueeze(0).unsqueeze(0))
        psnr_values.append(psnr_value.item())
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
    torch.save(torch.tensor(avg_runtime), f'{save_dir}avg_runtime.pt')
    torch.save(torch.tensor(runtimes), f'{save_dir}runtimes.pt')


    torch.save(torch.stack(reconstructions).cpu(), f'{save_dir}reconstructions.pt')
    torch.save(test_ct_slices.cpu(), f'{save_dir}target_slices.pt')

    # Log mean PSNR to a text file
    with open(f'{save_dir}metrics.txt', 'w') as f:
        f.write(f"Average PSNR: {avg_psnr:.3f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.3f}\n")
        f.write(f"Average Runtime: {avg_runtime:.5f} seconds\n")

        f.write("Metrics for each slice:\n")
        for i, psnr in enumerate(psnr_values):

            f.write(f"Case {i}: {psnr:.2f} dB, SSIM: {ssim_values[i]:.3f}, Runtime: {runtimes[i]:.5f} s\n")


    # Save reconstructions
    for i, recon in enumerate(reconstructions):

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(recon.cpu().squeeze(), cmap='gray')
        plt.title(f'FBP - Case {i}\n')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(test_ct_slices[i].cpu().squeeze(), cmap='gray')
        plt.title(f'Target - Case {i}\n')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(f"{save_dir}recon_case{i}.png")
        plt.close()


if __name__ == "__main__":
    auto_cli(run_experiment)