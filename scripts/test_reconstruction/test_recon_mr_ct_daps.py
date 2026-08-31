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
Script for evaluating reconstruction with DAPS for the MR -> CT task
In principle, we can use either the translation model or diffusion model
'''

def run_experiment(
        # Data saving and loading arguments
        save_dir: str = 'results/ct_recon/',
        evaluation_set: str = 'test',
        limit_batches: int = None,
        # Task arguments
        n_views: int = 8, # Number of views for CT operator
        task: str = 'sv', # Task type: sparse-view (sv) or limited-angle (la)
        snr: float = 50.0, # Signal-to-noise ratio for measurement noise in dB
        # Diffusion sampling arguments
        num_steps: int = 200,
        langevin_lr: float = 1e-5,
        num_langevin_steps: int = 100,
        meas_grad_weight: float = 1.0,
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
        'n_views': n_views,
        'task': task,
        'snr': snr,
        'num_steps': num_steps,
        'langevin_lr': langevin_lr,
        'meas_grad_weight': meas_grad_weight,
        'num_langevin_steps': num_langevin_steps,
        'device': device
    }
    
    # Set save_dir name based on task and config
    method = 'daps'
    save_dir = str(resolve_output_subdir(default_output_root, save_dir)) + '/'
    save_dir = f'{save_dir}{evaluation_set}/{task}/{n_views}_views_snr_{snr}_dB/{method}/lr_{langevin_lr}/meas_grad_weight_{meas_grad_weight}/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f'{save_dir}config.yml', 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    # -------- Loading model --------

    ckpt_path = checkpoint_file(checkpoint_root, 'unconditional_ct.ckpt')

    model = CMSI.load_from_checkpoint(ckpt_path).to(device)
    model.eval()

    # -------- Loading test data --------

    test_ct_slices = torch.load(data_file(data_root, 'mr_ct/ct', evaluation_set)).to(device)
    test_mr_slices = torch.load(data_file(data_root, 'mr_ct/mr', evaluation_set)).to(device)

    # -------- Batchify inference --------
    batch_size = 1
    num_samples = test_ct_slices.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    if limit_batches is not None:
        num_batches = min(num_batches, limit_batches)

    reconstructions = []
    runtimes = []

    # DAPS implementation adapted from InverseBench (see https://github.com/devzhk/InverseBench/blob/main/algo/daps.py)
    class LangevinDynamics:
        """
        Langevin Dynamics sampling method.
        """

        def __init__(self, num_steps, lr, lr_min_ratio=0.01):
            """
                Initializes the Langevin dynamics sampler with the given parameters.

                Parameters:
                    num_steps (int): Number of steps in the sampling process.
                    lr (float): Learning rate.
                    lr_min_ratio (float): Minimum learning rate ratio.
            """
            super().__init__()
            self.num_steps = num_steps
            self.lr = lr
            self.lr_min_ratio = lr_min_ratio

        def sample(self, x0hat, ct_op, measurement, ratio, dc_step_kwargs, verbose=False):
            """
                Samples using Langevin dynamics.

                Parameters:
                    x0hat (torch.Tensor): Initial state.
                    ct_op (Operator): CT operator.
                    measurement (torch.Tensor): Measurement tensor.
                    sigma (float): Current sigma value.
                    ratio (float): Current step ratio.
                    dc_step_kwargs (dict): Keyword arguments for the data consistency steps.
                    verbose (bool): Whether to display progress bar.

                Returns:
                    torch.Tensor: The final sampled state.
            """
            with torch.set_grad_enabled(True):
                pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
                lr = self.get_lr(ratio)
                x0hat = x0hat.detach()
                x = x0hat.clone().detach().requires_grad_(True)
                optimizer = torch.optim.SGD([x], lr)
                for _ in pbar:
                    optimizer.zero_grad()

                    measurement = dc_step_kwargs['y_meas']
                    tau = dc_step_kwargs['sigma_meas']

                    dc_loss = torch.mean((measurement - ct_op.A(x)) ** 2) / (2 * tau ** 2)
                    dc_loss.backward()
                    gradient = meas_grad_weight * x.grad.detach().clone()

                    # In DDBM, the distribution p(x_0 | x_t, x_T) is assumed to be a Gaussian
                    # The gradient is computed as the derivative of the log probability of the Gaussian distribution
                    x_T = dc_step_kwargs['x_T']
                    x_t = dc_step_kwargs['x']
                    a_t, b_t, std_t = dc_step_kwargs['denoiser'].bridge_sample_coeffs(dc_step_kwargs['sigma_next'])
                    gradient += b_t * (x_t - a_t * x_T - b_t * x) / (std_t ** 2)
                    # gradient += (x - x0hat) / sigma ** 2 # Grad for diffusion model
                    x.grad = gradient

                    optimizer.step()
                    with torch.no_grad():
                        epsilon = torch.randn_like(x)
                        x.data = x.data + np.sqrt(2 * lr) * epsilon

                    # early stopping with NaN
                    if torch.isnan(x).any():
                        return torch.zeros_like(x)

            return x.detach()
        
        def get_lr(self, ratio):
            """
                Computes the learning rate based on the given ratio.
            """
            p = 1
            multiplier = (1 ** (1 / p) + ratio * (self.lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
            return multiplier * self.lr

    def data_consistency(x, y_meas, diffusion, denoiser, y, sigma, sigma_next, x_T, sigma_meas, i, sig_t, **kwargs):
        langevin = LangevinDynamics(lr=langevin_lr, num_steps=num_langevin_steps)
        if sigma_next.item() > 1e-5:
            x0hat, _, _   = karras_sample(model.denoiser,
                                            model.unet,
                                            x_T=x.clone(),
                                            y=y,
                                            steps=20,
                                            sigma_max=sigma_next,
                                            clip_denoised=False,
                                            dc_type=None)

            x_dc = langevin.sample(x0hat, ct_op, y_meas, ratio=i/num_steps,
                                dc_step_kwargs={'y_meas': y_meas, 'sigma_meas': sigma_meas, 'x_T': x_T, 'x': x, 'denoiser': model.denoiser, 'sigma_next': sigma_next})

            x_dc = diffusion.bridge_sample(x_dc, x_T, sigma_next.to(device))
        else:
            x_dc = x
        return x_dc

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Evaluating Batches"):

            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            
            init_cond = torch.randn_like(test_ct_slices[start_idx:end_idx]).unsqueeze(1)
            y = init_cond

            if task == 'sv':
                ct_op = PBCT(n_views, 1, 256, device=device)
            elif task == 'la':
                ct_op = PBCT(n_views, 1, 256, device=device, angles=np.arange(n_views, dtype=np.float32))
            y_meas = ct_op.A(test_ct_slices[start_idx:end_idx].unsqueeze(1))
            
            avg_signal_power = torch.mean(y_meas ** 2)
            noise_power = avg_signal_power / (10 ** (snr / 10))
            sigma_meas = torch.sqrt(noise_power)
            noise = sigma_meas * torch.randn_like(y_meas)
            y_meas = y_meas + noise


            start_time = time.perf_counter()
            
            sample, path, nfe  = karras_sample(model.denoiser, model.unet,
                                                x_T=init_cond,
                                                y=y,
                                                steps=num_steps,
                                                dc_type='xt',
                                                dc_step=data_consistency,
                                                dc_step_kwargs={'y_meas': y_meas, 'sigma_meas': sigma_meas},
                                                sigma_min=1e-3,
                                                )
            
            end_time = time.perf_counter()
            runtimes.append(end_time - start_time)
            
            reconstructions.append(sample)

    # Compute metrics
    psnr_values = []
    ssim_values = []

    for recon, target in zip(reconstructions, test_ct_slices):
        psnr_value = compute_psnr(recon.cpu(), target.cpu().unsqueeze(0).unsqueeze(0))
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
        f.write(f"Average Runtime: {avg_runtime:.2f} seconds\n")

        f.write("\n")
        f.write("Metrics for each slice:\n")
        for i, psnr in enumerate(psnr_values):
            f.write(f"Case {i}: {psnr:.2f} dB, SSIM: {ssim_values[i]:.3f}, Runtime: {runtimes[i]:.2f} s\n")


    # Save reconstructions
    for i, recon in enumerate(reconstructions):

        plt.figure(figsize=(20, 5))

        plt.subplot(1, 4, 1)
        plt.imshow(test_mr_slices[i].cpu().squeeze(), cmap='gray')
        plt.title(f'Input - Case {i}\n')
        plt.colorbar()

        plt.subplot(1, 4, 3)
        plt.imshow(recon.cpu().squeeze(), cmap='gray')
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