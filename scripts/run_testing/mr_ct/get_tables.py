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
result_dirs = get_bottom_dirs(f'{root_dir}/results/ct_recon/test/')

results = []

for result_dir in result_dirs:
    # Read the config yaml file
    config_path = os.path.join(result_dir, 'config.yml')
    if not os.path.exists(config_path):
        print(f"Config file not found in {result_dir}, skipping.")
        continue
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'daps_gtp' in result_dir:
        method = 'daps_gtp'
    elif 'dds_gtp' in result_dir:
        method = 'dds_gtp'
    elif 'dds' in result_dir:
        method = 'dds'
    elif 'grad_gtp_unpaired' in result_dir:
        method = 'grad_gtp_unpaired'
    elif 'grad_gtp' in result_dir:
        method = 'grad_gtp'
    elif 'dps' in result_dir:
        method = 'dps'
    elif 'fbp' in result_dir:
        method = 'fbp'
    elif 'jtv' in result_dir:
        method = 'jtv'
    elif 'tv' in result_dir:
        method = 'tv'
    elif 'daps' in result_dir:
        method = 'daps'
    elif 'dds_gtp' in result_dir:
        method = 'dds_gtp'
    elif 'dds' in result_dir:
        method = 'dds'
    elif 'prox_gtp_rigorous' in result_dir:
        method = 'prox_gtp_rigorous'
    elif 'prox_gtp_unpaired' in result_dir:
        method = 'prox_gtp_unpaired'
    elif 'prox_gtp' in result_dir:
        method = 'prox_gtp'
    else:
        print(f"Unknown method in {result_dir}, skipping.")
        continue

    if method not in ['fbp', 'jtv', 'tv', 'dps', 'daps', 'dds', 'grad_gtp', 'grad_gtp_unpaired', 'daps_gtp', 'dds_gtp', 'prox_gtp', 'prox_gtp_unpaired', 'prox_gtp_rigorous']:
        continue

    # Check if the result files exist
    avg_psnr = torch.load(os.path.join(result_dir, 'avg_psnr.pt')) if os.path.exists(os.path.join(result_dir, 'avg_psnr.pt')) else None
    config['avg_psnr'] = avg_psnr
    avg_ssim = torch.load(os.path.join(result_dir, 'avg_ssim.pt')) if os.path.exists(os.path.join(result_dir, 'avg_ssim.pt')) else None
    config['avg_ssim'] = avg_ssim
    config['recons'] = torch.load(os.path.join(result_dir, 'reconstructions.pt')) if os.path.exists(os.path.join(result_dir, 'reconstructions.pt')) else None
    config['targets'] = torch.load(os.path.join(result_dir, 'target_slices.pt')) if os.path.exists(os.path.join(result_dir, 'target_slices.pt')) else None
    config['avg_runtime'] = torch.load(os.path.join(result_dir, 'avg_runtime.pt')) if os.path.exists(os.path.join(result_dir, 'avg_runtime.pt')) else None

    config['method'] = method

    results.append(config)

best_results_dds = {}
best_results_prox_gtp = {}
best_results_prox_gtp_unpaired = {}
best_results_prox_gtp_rigorous = {}
best_results_grad_gtp = {}
best_results_grad_gtp_unpaired = {}
best_results_dps = {}
best_results_daps = {}
best_results_daps_gtp = {}
best_results_dds_gtp = {}

def has_result(method, task, n_views):
    if method in ['fbp', 'tv', 'jtv']:
        return any(r['method'] == method and r['task'] == task and r['n_views'] == n_views
                   for r in results)
    result_maps = {
        'dds': best_results_dds,
        'prox_gtp': best_results_prox_gtp,
        'prox_gtp_unpaired': best_results_prox_gtp_unpaired,
        'prox_gtp_rigorous': best_results_prox_gtp_rigorous,
        'grad_gtp': best_results_grad_gtp,
        'grad_gtp_unpaired': best_results_grad_gtp_unpaired,
        'dps': best_results_dps,
        'daps': best_results_daps,
        'daps_gtp': best_results_daps_gtp,
        'dds_gtp': best_results_dds_gtp,
    }
    return (task, n_views) in result_maps[method]


for result in results:
    key = (result['task'], result['n_views'])
    if result['method'] == 'dds':
        if key not in best_results_dds or result['avg_psnr'] > best_results_dds[key]['avg_psnr']:
            best_results_dds[key] = result
    elif result['method'] == 'prox_gtp':
        if key not in best_results_prox_gtp or result['avg_psnr'] > best_results_prox_gtp[key]['avg_psnr']:
            best_results_prox_gtp[key] = result
    elif result['method'] == 'prox_gtp_rigorous':
        if key not in best_results_prox_gtp_rigorous or result['avg_psnr'] > best_results_prox_gtp_rigorous[key]['avg_psnr']:
            best_results_prox_gtp_rigorous[key] = result
    elif result['method'] == 'grad_gtp':
        if (key not in best_results_grad_gtp or result['avg_psnr'] > best_results_grad_gtp[key]['avg_psnr']):
            best_results_grad_gtp[key] = result
    elif result['method'] == 'grad_gtp_unpaired':
        if (key not in best_results_grad_gtp_unpaired or result['avg_psnr'] > best_results_grad_gtp_unpaired[key]['avg_psnr']):
            best_results_grad_gtp_unpaired[key] = result
    elif result['method'] == 'prox_gtp_unpaired':
        if (key not in best_results_prox_gtp_unpaired or result['avg_psnr'] > best_results_prox_gtp_unpaired[key]['avg_psnr']):
            best_results_prox_gtp_unpaired[key] = result
    elif result['method'] == 'dps':
        if (key not in best_results_dps or result['avg_psnr'] > best_results_dps[key]['avg_psnr']):
            best_results_dps[key] = result
    elif result['method'] == 'daps':
        if (key not in best_results_daps or result['avg_psnr'] > best_results_daps[key]['avg_psnr']):
            best_results_daps[key] = result
    elif result['method'] == 'daps_gtp':
        if (key not in best_results_daps_gtp or result['avg_psnr'] > best_results_daps_gtp[key]['avg_psnr']):
            best_results_daps_gtp[key] = result
    elif result['method'] == 'dds_gtp':
        if (key not in best_results_dds_gtp or result['avg_psnr'] > best_results_dds_gtp[key]['avg_psnr']):
            best_results_dds_gtp[key] = result
    

print('SV-CT Results:')
for method in ['fbp', 'tv', 'jtv', 'dps', 'daps','daps_gtp','dds', 'dds_gtp', 'grad_gtp', 'grad_gtp_unpaired', 'prox_gtp_rigorous', 'prox_gtp', 'prox_gtp_unpaired']:
    if not all(has_result(method, 'sv', n_views) for n_views in [4, 8, 16, 32]):
        continue
    row = ''
    for n_views in [4, 8, 16, 32]:
        if method == 'prox_gtp':
            result = best_results_prox_gtp[('sv', n_views)]
            name = 'Prox-GTP'
        elif method == 'prox_gtp_rigorous':
            result = best_results_prox_gtp_rigorous[('sv', n_views)]
            name = 'Prox-GTP (strict)'
        elif method == 'prox_gtp_unpaired':
            result = best_results_prox_gtp_unpaired[('sv', n_views)]
            name = 'Prox-GTP (unpaired)'
        elif method == 'dds':
            result = best_results_dds[('sv', n_views)]
            name = 'DDS'
        elif method == 'dds_gtp':
            result = best_results_dds_gtp[('sv', n_views)]
            name = 'DDS-GTP'
        elif method == 'grad_gtp':
            result = best_results_grad_gtp[('sv', n_views)]
            name = 'Grad-GTP'
        elif method == 'grad_gtp_unpaired':
            result = best_results_grad_gtp_unpaired[('sv', n_views)]
            name = 'Grad-GTP (unpaired)'
        elif method == 'dps':
            result = best_results_dps[('sv', n_views)]
            name = 'DPS'
        elif method == 'daps':
            result = best_results_daps[('sv', n_views)]
            name = 'DAPS'
        elif method == 'daps_gtp':
            result = best_results_daps_gtp[('sv', n_views)]
            name = 'DAPS-GTP'
        elif method == 'fbp':
            result = next(r for r in results if r['method'] == 'fbp' and r['task'] == 'sv' and r['n_views'] == n_views)
            name = 'FBP'
        elif method == 'tv':
            result = next(r for r in results if r['method'] == 'tv' and r['task'] == 'sv' and r['n_views'] == n_views)
            name = 'TV'
        elif method == 'jtv':
            result = next(r for r in results if r['method'] == 'jtv' and r['task'] == 'sv' and r['n_views'] == n_views)
            name = 'JTV'
        if row == '':
            row = name
        psnr = f"{result['avg_psnr']:.2f}"
        ssim = f"{result['avg_ssim']:.3f}" if 'avg_ssim' in result else '--'
        row += f" & {psnr} & {ssim}"
    row += " \\\\"
    print(row)

print('\n\n')

print('LA-CT Results:')
# Now do LA-CT results. Include PSNR and SSIM for each if available.
for method in ['fbp', 'tv', 'jtv', 'dps', 'daps', 'daps_gtp', 'dds', 'dds_gtp', 'grad_gtp', 'grad_gtp_unpaired', 'prox_gtp_rigorous', 'prox_gtp', 'prox_gtp_unpaired']:
    if not all(has_result(method, 'la', n_views) for n_views in [30, 60, 90, 120]):
        continue
    row = ''
    for n_views in [30, 60, 90, 120]:
        if method == 'prox_gtp':
            result = best_results_prox_gtp[('la', n_views)]
            name = 'Prox-GTP'
        elif method == 'prox_gtp_rigorous':
            result = best_results_prox_gtp_rigorous[('la', n_views)]
            name = 'Prox-GTP (strict)'
        elif method == 'prox_gtp_unpaired':
            result = best_results_prox_gtp_unpaired[('la', n_views)]
            name = 'Prox-GTP (unpaired)'
        elif method == 'dds':
            result = best_results_dds[('la', n_views)]
            name = 'DDS'
        elif method == 'dds_gtp':
            result = best_results_dds_gtp[('la', n_views)]
            name = 'DDS-GTP'
        elif method == 'grad_gtp':
            result = best_results_grad_gtp[('la', n_views)]
            name = 'Grad-GTP'
        elif method == 'grad_gtp_unpaired':
            result = best_results_grad_gtp_unpaired[('la', n_views)]
            name = 'Grad-GTP (unpaired)'
        elif method == 'dps':
            result = best_results_dps[('la', n_views)]
            name = 'DPS'
        elif method == 'daps':
            result = best_results_daps[('la', n_views)]
            name = 'DAPS'
        elif method == 'daps_gtp':
            result = best_results_daps_gtp[('la', n_views)]
            name = 'DAPS-GTP'
        elif method == 'fbp':
            result = next(r for r in results if r['method'] == 'fbp' and r['task'] == 'la' and r['n_views'] == n_views)
            name = 'FBP'
        elif method == 'tv':
            result = next(r for r in results if r['method'] == 'tv' and r['task'] == 'la' and r['n_views'] == n_views)
            name = 'TV'
        elif method == 'jtv':
            result = next(r for r in results if r['method'] == 'jtv' and r['task'] == 'la' and r['n_views'] == n_views)
            name = 'JTV'
        if row == '':
            row = name
        psnr = f"{result['avg_psnr']:.2f}"
        ssim = f"{result['avg_ssim']:.3f}" if 'avg_ssim' in result else '--'
        row += f" & {psnr} & {ssim}"
    row += " \\\\"
    print(row)

# Get runtimes for each method for 8-view SV-CT and 90-degree LA-CT. Print them in a table.

print('\n\n')
print('Runtime results:')
for method in ['fbp', 'tv', 'jtv', 'dps', 'daps', 'daps_gtp', 'dds', 'dds_gtp', 'grad_gtp', 'grad_gtp_unpaired', 'prox_gtp_rigorous', 'prox_gtp', 'prox_gtp_unpaired']:
    if not has_result(method, 'sv', 8) or not has_result(method, 'la', 90):
        continue
    if method == 'prox_gtp':
        result_sv = best_results_prox_gtp[('sv', 8)]
        result_la = best_results_prox_gtp[('la', 90)]
        name = 'Prox-GTP'
    elif method == 'prox_gtp_rigorous':
        result_sv = best_results_prox_gtp_rigorous[('sv', 8)]
        result_la = best_results_prox_gtp_rigorous[('la', 90)]
        name = 'Prox-GTP (strict)'
    elif method == 'prox_gtp_unpaired':
        result_sv = best_results_prox_gtp_unpaired[('sv', 8)]
        result_la = best_results_prox_gtp_unpaired[('la', 90)]
        name = 'Prox-GTP (unpaired)'
    elif method == 'dds':
        result_sv = best_results_dds[('sv', 8)]
        result_la = best_results_dds[('la', 90)]
        name = 'DDS'
    elif method == 'dds_gtp':
        result_sv = best_results_dds_gtp[('sv', 8)]
        result_la = best_results_dds_gtp[('la', 90)]
        name = 'DDS-GTP'
    elif method == 'grad_gtp':
        result_sv = best_results_grad_gtp[('sv', 8)]
        result_la = best_results_grad_gtp[('la', 90)]
        name = 'Grad-GTP'
    elif method == 'grad_gtp_unpaired':
        result_sv = best_results_grad_gtp_unpaired[('sv', 8)]
        result_la = best_results_grad_gtp_unpaired[('la', 90)]
        name = 'Grad-GTP (unpaired)'
    elif method == 'dps':
        result_sv = best_results_dps[('sv', 8)]
        result_la = best_results_dps[('la', 90)]
        name = 'DPS'
    elif method == 'daps':
        result_sv = best_results_daps[('sv', 8)]
        result_la = best_results_daps[('la', 90)]
        name = 'DAPS'
    elif method == 'daps_gtp':
        result_sv = best_results_daps_gtp[('sv', 8)]
        result_la = best_results_daps_gtp[('la', 90)]
        name = 'DAPS-GTP'
    elif method == 'fbp':
        result_sv = next(r for r in results if r['method'] == 'fbp' and r['task'] == 'sv' and r['n_views'] == 8)
        result_la = next(r for r in results if r['method'] == 'fbp' and r['task'] == 'la' and r['n_views'] == 90)
        name = 'FBP'
    elif method == 'tv':
        result_sv = next(r for r in results if r['method'] == 'tv' and r['task'] == 'sv' and r['n_views'] == 8)
        result_la = next(r for r in results if r['method'] == 'tv' and r['task'] == 'la' and r['n_views'] == 90)
        name = 'TV'
    elif method == 'jtv':
        result_sv = next(r for r in results if r['method'] == 'jtv' and r['task'] == 'sv' and r['n_views'] == 8)
        result_la = next(r for r in results if r['method'] == 'jtv' and r['task'] == 'la' and r['n_views'] == 90)
        name = 'JTV'
    runtime_sv = f"{result_sv['avg_runtime']:.2f}"
    runtime_la = f"{result_la['avg_runtime']:.2f}"
    print(f"{name} & {runtime_sv} & {runtime_la} \\\\")
