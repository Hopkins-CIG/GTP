import torch
import numpy as np
import os
import glob
from src.models.ddbm import CMSI
from src.models.karras_diffusion import karras_sample
from src.inference_config import psnr_neg_one_one as compute_psnr
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from src.physics.pbct import PBCT
import time
from jsonargparse import auto_cli
from src.utils import get_project_root
from src.inference_config import checkpoint_file, data_file, resolve_device, resolve_inference_roots, resolve_output_subdir

torch.manual_seed(123)

'''
Script for evaluating reconstruction with gradient-based data consistency for the MR -> CT task
I.e. this runs grad-GTP when the translation model is used or DPS when a diffusion model is used
'''

def run_experiment(
        # Data saving and loading arguments
        save_dir: str = 'results/ct_recon/',
        evaluation_set: str = 'test',
        # Task arguments
        n_views: int = 8, # Number of views for CT operator
        task: str = 'sv', # Task type: sparse-view (sv) or limited-angle (la)
        snr: float = 50.0, # Signal-to-noise ratio for measurement noise in dB
        # Diffusion sampling arguments
        num_steps: int = 200,
        lamda: float = 1e-1,
        num_samples: int = 1,
        limit_batches: int = None, # For debugging, limit the number of batches to run
        # Device
        device: str = 'cuda:0',
        data_root: str = None,
        checkpoint_root: str = None,
        output_root: str = None
        ):

    data_root, checkpoint_root, default_output_root = resolve_inference_roots(data_root, checkpoint_root, output_root)
    device = resolve_device(device)
    
    # Save the config used for this experiment as a yml file
    config = {
        'save_dir': save_dir,
        'evaluation_set': evaluation_set,
        'limit_batches': limit_batches,
        'n_views': n_views,
        'task': task,
        'snr': snr,
        'num_steps': num_steps,
        'lamda': lamda,
        'num_samples': num_samples,
        'device': device
    }
    
    # Set save_dir name based on task and config
    method = 'prox_gtp_rigorous'
    save_dir = str(resolve_output_subdir(default_output_root, save_dir)) + '/'
    save_dir = f'{save_dir}{evaluation_set}/{task}/{n_views}_views_snr_{snr}_dB/{method}/lamda_{lamda}/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f'{save_dir}config.yml', 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    # -------- Loading model --------

    ckpt_path = checkpoint_file(checkpoint_root, 'mr_ct_translation.ckpt')

    model = CMSI.load_from_checkpoint(ckpt_path).to(device)
    model.eval()

    # -------- Loading test data --------

    test_ct_slices = torch.load(data_file(data_root, 'mr_ct/ct', evaluation_set)).to(device)
    test_mr_slices = torch.load(data_file(data_root, 'mr_ct/mr', evaluation_set)).to(device)

    # -------- Batchify inference --------
    batch_size = 1
    num_batches = test_mr_slices.shape[0]

    if limit_batches is not None:
        num_batches = min(num_batches, limit_batches)

    reconstructions = []
    fbps = []
    runtimes = []

    def data_consistency(x, y_meas, diffusion, denoiser, y, sigma, sigma_next, x_T, lamda, dt, sig_t, sigma_meas, **kwargs):
        
        with torch.set_grad_enabled(True):
            x_opt = x.clone().detach().requires_grad_(True)
            opt = torch.optim.SGD([x_opt], lr=1e-2)

            for _ in range(5):
                opt.zero_grad()
                x_0_hat = denoiser(x_opt, y, sigma_next.unsqueeze(0).to(device))

                n = x.shape[0]
                diff = y_meas - ct_op.A(x_0_hat)

                prox_obj = lamda * (1 - sigma) * 1/(2*sigma_meas**2) * torch.linalg.norm(diff)**2 + 1/2 * torch.linalg.norm(x_opt - x)**2
                prox_obj.backward()

                x_opt.grad.clamp_(-1, 1)

                opt.step()
                opt.zero_grad()
            
        L_each = (1 - sigma) * 1/(2 * sigma_meas**2) * torch.linalg.norm(diff.flatten(1), dim=1)**2
        L_grad = 1/(lamda) * (x - x_opt)

        sig_sq = 2 * lamda / dt.abs()

        x = x - (sig_sq / 2) * dt.abs() * L_grad

        return x.detach()
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Evaluating Batches"):
            
            init_cond = test_mr_slices[batch_idx].unsqueeze(0).repeat(num_samples, 1, 1, 1)
            y = test_mr_slices[batch_idx].unsqueeze(0).repeat(num_samples, 1, 1, 1)

            if task == 'sv':
                ct_op = PBCT(n_views, 1, 256, device=device, batch_size=num_samples)
            elif task == 'la':
                ct_op = PBCT(n_views, 1, 256, device=device, angles=np.arange(n_views, dtype=np.float32), batch_size=num_samples)
            y_meas = ct_op.A(test_ct_slices[batch_idx].unsqueeze(0).repeat(num_samples, 1, 1, 1))
            
            avg_signal_power = torch.mean(y_meas ** 2)
            noise_power = avg_signal_power / (10 ** (snr / 10))
            sigma_meas = torch.sqrt(noise_power)
            noise = sigma_meas * torch.randn_like(y_meas)
            y_meas = y_meas + noise


            start_time = time.perf_counter()
            
            sample, path, nfe = karras_sample(model.denoiser, model.unet,
                                                x_T=init_cond,
                                                y=y,
                                                steps=num_steps,
                                                churn_step_ratio=0.33,
                                                dc_type='xt',
                                                dc_step=data_consistency,
                                                dc_step_kwargs={'y_meas': y_meas, 'lamda': lamda, 'sigma_meas': sigma_meas},
                                                clip_denoised=False
                                                )
            
            end_time = time.perf_counter()
            runtimes.append(end_time - start_time)
            
            reconstructions.append(sample)

    # Compute metrics
    psnr_values = []
    ssim_values = []
    if num_samples > 1:
        psnr_sample_mean_values = []
        ssim_sample_mean_values = []
        nll_values = []

    for recon, target in zip(reconstructions, test_ct_slices):
        psnr_value = compute_psnr(recon.cpu(), target.cpu().unsqueeze(0).unsqueeze(0).repeat(recon.shape[0], 1, 1, 1))
        ssim = SSIM()(recon.cpu(), target.cpu().unsqueeze(0).unsqueeze(0).repeat(recon.shape[0], 1, 1, 1))
        psnr_values.append(psnr_value.item())
        ssim_values.append(ssim.item())
        if num_samples > 1:
            # Compute PSNR and SSIM for the sample mean reconstruction as well
            # Also compute the std of the reconstructions across samples and compute NLL of the target under a Gaussian with that mean and std
            recon_mean = recon.mean(dim=0, keepdim=True)
            recon_std = recon.std(dim=0, keepdim=True)
            psnr_mean = compute_psnr(recon_mean.cpu(), target.cpu().unsqueeze(0).unsqueeze(0))
            ssim_mean = SSIM()(recon_mean.cpu(), target.cpu().unsqueeze(0).unsqueeze(0))
            psnr_sample_mean_values.append(psnr_mean.item())
            ssim_sample_mean_values.append(ssim_mean.item())
            # Compute NLL
            nll = 0.5 * torch.log(2 * torch.pi * recon_std**2) + 0.5 * (target.unsqueeze(0).unsqueeze(0) - recon_mean)**2 / (recon_std**2)
            nll = nll.mean().item()
            nll_values.append(nll)

    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    avg_runtime = sum(runtimes) / len(runtimes)

    if num_samples > 1:
        avg_psnr_sample_mean = sum(psnr_sample_mean_values) / len(psnr_sample_mean_values)
        avg_ssim_sample_mean = sum(ssim_sample_mean_values) / len(ssim_sample_mean_values)
        avg_nll = sum(nll_values) / len(nll_values)

    print(f"Average PSNR: {avg_psnr:.3f} dB")
    print(f"Average SSIM: {avg_ssim:.3f}")

    if num_samples > 1:
        print(f"Average PSNR (Sample Mean): {avg_psnr_sample_mean:.3f} dB")
        print(f"Average SSIM (Sample Mean): {avg_ssim_sample_mean:.3f}")
        print(f"Average NLL: {avg_nll:.3f}")

    torch.save(torch.tensor(psnr_values), f'{save_dir}psnr_values.pt')
    torch.save(torch.tensor(avg_psnr), f'{save_dir}avg_psnr.pt')
    torch.save(torch.tensor(ssim_values), f'{save_dir}ssim_values.pt')
    torch.save(torch.tensor(avg_ssim), f'{save_dir}avg_ssim.pt')
    torch.save(torch.tensor(avg_runtime), f'{save_dir}avg_runtime.pt')
    torch.save(torch.tensor(runtimes), f'{save_dir}runtimes.pt')

    if num_samples > 1:
        torch.save(torch.tensor(avg_psnr_sample_mean), f'{save_dir}avg_psnr_sample_mean.pt')
        torch.save(torch.tensor(avg_ssim_sample_mean), f'{save_dir}avg_ssim_sample_mean.pt')
        torch.save(torch.tensor(avg_nll), f'{save_dir}avg_nll.pt')

    torch.save(torch.stack(reconstructions).cpu(), f'{save_dir}reconstructions.pt')
    torch.save(test_ct_slices.cpu(), f'{save_dir}target_slices.pt')

    # Log mean PSNR to a text file
    with open(f'{save_dir}metrics.txt', 'w') as f:
        f.write(f"Average PSNR: {avg_psnr:.3f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.3f}\n")

        f.write(f"Average Runtime: {avg_runtime:.2f} seconds\n")
        if num_samples > 1:
            f.write(f"Average PSNR (Sample Mean): {avg_psnr_sample_mean:.3f} dB\n")
            f.write(f"Average SSIM (Sample Mean): {avg_ssim_sample_mean:.3f}\n")
            f.write(f"Average NLL: {avg_nll:.3f}\n")
        
        f.write("Metrics for each slice:\n")
        for i, psnr in enumerate(psnr_values):
            f.write(f"Case {i}: {psnr:.2f} dB, SSIM: {ssim_values[i]:.3f}, Runtime: {runtimes[i]:.2f} s")
            if num_samples > 1:
                f.write(f", PSNR (Sample Mean): {psnr_sample_mean_values[i]:.2f} dB, SSIM (Sample Mean): {ssim_sample_mean_values[i]:.3f}, NLL: {nll_values[i]:.3f}")
            f.write("\n")


    # Save reconstructions
    for i, recon in enumerate(reconstructions):

        plt.figure(figsize=(20, 5))

        plt.subplot(1, 4, 1)
        plt.imshow(test_mr_slices[i].cpu().squeeze(), cmap='gray')
        plt.title(f'Input - Case {i}\n')
        plt.colorbar()

        plt.subplot(1, 4, 3)
        plt.imshow(recon[0].cpu().squeeze(), cmap='gray')
        plt.title(f'Reconstructed - Case {i}\nPSNR: {psnr_values[i]:.2f} dB, SSIM: {ssim_values[i]:.3f}')
        plt.colorbar()

        plt.subplot(1, 4, 4)
        plt.imshow(test_ct_slices[i].cpu().squeeze(), cmap='gray')
        plt.title(f'Target - Case {i}\n')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(f"{save_dir}recon_case{i}.png")
        plt.close()

if __name__ == "__main__":
    auto_cli(run_experiment)