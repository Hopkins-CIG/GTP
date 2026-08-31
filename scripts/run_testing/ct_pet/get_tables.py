import glob
import torch
import numpy as np
from pathlib import Path
import os
import yaml
from src.utils import get_project_root

def get_bottom_dirs(root_path):
    bottom_dirs = []
    for root, dirs, files in os.walk(root_path):
        # If the list of subdirectories (dirs) is empty, this is a terminal node
        if not dirs:
            bottom_dirs.append(root)
    return bottom_dirs


root_dir = get_project_root()
result_dirs = get_bottom_dirs(f'{root_dir}/results/pet_recon/test/')

results = []

for result_dir in result_dirs:
    # Read the config yaml file
    config_path = os.path.join(result_dir, 'config.yml')
    if not os.path.exists(config_path):
        print(f"Config file not found in {result_dir}, skipping.")
        continue
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'osem' in result_dir:
        method = 'osem'
    elif 'pet_dds' in result_dir:
        method = 'pet_dds'
    elif 'prox_gtp' in result_dir:
        method = 'prox_gtp'
    else:
        print(f"Unknown method in {result_dir}, skipping.")
        continue

    avg_psnr = torch.load(os.path.join(result_dir, 'avg_psnr.pt'))
    config['avg_psnr'] = avg_psnr
    avg_ssim = torch.load(os.path.join(result_dir, 'avg_ssim.pt'))
    config['avg_ssim'] = avg_ssim
    config['recons'] = torch.load(os.path.join(result_dir, 'reconstructions.pt'))
    config['targets'] = torch.load(os.path.join(result_dir, 'target_slices.pt'))
    config['avg_runtime'] = torch.load(os.path.join(result_dir, 'avg_runtime.pt'))
    config['method'] = method

    results.append(config)

best_results_osem = {}
best_results_pet_dds = {}
best_results_prox_gtp = {}

for result in results:
    key = (result['total_counts'])
    if result['method'] == 'osem':
        if key not in best_results_osem or result['avg_psnr'] > best_results_osem[key]['avg_psnr']:
            best_results_osem[key] = result
    elif result['method'] == 'pet_dds':
        if key not in best_results_pet_dds or result['avg_psnr'] > best_results_pet_dds[key]['avg_psnr']:
            best_results_pet_dds[key] = result
    elif result['method'] == 'prox_gtp':
        if key not in best_results_prox_gtp or result['avg_psnr'] > best_results_prox_gtp[key]['avg_psnr']:
            best_results_prox_gtp[key] = result

print('Reconstruction results:')
for method in ['osem', 'pet_dds', 'prox_gtp']:
    row = ''
    for n_counts in [1e5, 5e5, 1e6]:
        if method == 'osem':
            result = best_results_osem.get(n_counts)
        elif method == 'pet_dds':
            result = best_results_pet_dds.get(n_counts)
        elif method == 'prox_gtp':
            result = best_results_prox_gtp.get(n_counts)
        
        if result is not None:
            row += f"{result['avg_psnr']:.2f} & {result['avg_ssim']:.3f} & "
        else:
            row += "N/A & N/A & "
    print(f"{method} & {row[:-2]} \\\\")

print('\n\n')
print('Runtime results:')
for method in ['osem', 'pet_dds', 'prox_gtp']:
    row = ''
    for n_counts in [1e5, 5e5, 1e6]:
        if method == 'osem':
            result = best_results_osem.get(n_counts)
        elif method == 'pet_dds':
            result = best_results_pet_dds.get(n_counts)
        elif method == 'prox_gtp':
            result = best_results_prox_gtp.get(n_counts)
        
        if result is not None:
            row += f"{result['avg_runtime']:.2f} & "
        else:
            row += "N/A & "
    print(f"{method} & {row[:-2]} \\\\")

