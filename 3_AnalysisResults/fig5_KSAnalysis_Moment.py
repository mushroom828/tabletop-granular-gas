# SPDX-License-Identifier: AGPL-3.0-only
# See ../LICENSE for the full license terms.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import argparse
from pathlib import Path
from scipy.stats import kstest
from scipy.constants import k


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['pdf.fonttype'] = 42

M = 0.0036  # Particle mass (kg)
     
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / '2_GetResults' / 'processed_data'
BALL_SIZE_CSV = DATA_DIR / 'ball_sizes_means.csv'

# Theoretical CDF
def theoretical_cdf(v, Tg):
    return 1.0 - np.exp(-M * v**2 / (2 * k * Tg))

# Keep valid speeds without a statistical cutoff on their magnitude.
def load_and_clean_data(csv_path, conversion_factor):
    df = pd.read_csv(csv_path)
    df = df[df['obj_id'] != 'container'].copy()
    df = df.sort_values('frame')
    vx = df['vx'].values * conversion_factor
    vy = df['vy'].values * conversion_factor
    v = np.sqrt(vx**2 + vy**2)
    return v[np.isfinite(v) & (v > 0)]

# Autocorrelation calculation
def compute_autocorrelation(csv_path, max_lag_frames=500):
    df = pd.read_csv(csv_path)
    df = df[df['obj_id'] != 'container'].copy()
    if df.empty:
        return None, 0, None
    acf_dot_sum = np.zeros(max_lag_frames)
    acf_count = np.zeros(max_lag_frames)
    for obj_id, group in df.groupby('obj_id'):
        group = group.sort_values('frame')
        min_t, max_t = group['frame'].min(), group['frame'].max()
        group = group.set_index('frame').reindex(range(min_t, max_t + 1))
        vx = group['vx'].values
        vy = group['vy'].values
        for lag in range(max_lag_frames):
            if lag >= len(vx):
                break
            if lag == 0:
                dot = vx**2 + vy**2
            else:
                dot = vx[:-lag] * vx[lag:] + vy[:-lag] * vy[lag:]
            valid = ~np.isnan(dot)
            acf_dot_sum[lag] += np.sum(dot[valid])
            acf_count[lag] += np.sum(valid)
    valid_lags = acf_count > 0
    if not np.any(valid_lags):
        return None, 0, None
    acf = np.zeros(max_lag_frames)
    acf[valid_lags] = acf_dot_sum[valid_lags] / acf_count[valid_lags]
    if acf[0] > 0:
        acf = acf / acf[0]
    else:
        return None, 0, None
    threshold = np.exp(-1)
    decay_indices = np.where(valid_lags & (acf < threshold))[0]
    tau_frames = decay_indices[0] if len(decay_indices) > 0 else max_lag_frames
    return acf, tau_frames, acf_count

def batch_compute_autocorrelation(data_dir, rpm_list, ball_list, max_lag_frames=500):
    all_files = []
    for f in os.listdir(data_dir):
        if f.lower().endswith('.csv'):
            match = re.search(r'vx_vy_data_(\d+)r_(\d+)\.csv', f)
            if match:
                rpm = int(match.group(1))
                balls = int(match.group(2))
                if rpm in rpm_list and balls in ball_list:
                    all_files.append((rpm, balls, os.path.join(data_dir, f)))
    results = {}
    for rpm, balls, path in all_files:
        _, tau, _ = compute_autocorrelation(path, max_lag_frames)
        if tau is not None:
            results[(rpm, balls)] = tau
            print(f"{rpm}r, {balls} balls: tau = {tau} frames")
        else:
            print(f"{rpm}r, {balls} balls: computation failed")
    return results

# KS test
def analyze_subset_ks(v_subset):
    if len(v_subset) < 10:
        return np.nan, np.nan
    mean_sq_v = np.mean(v_subset**2)
    Tg = M * mean_sq_v / (2.0 * k)
    stat_D, _ = kstest(v_subset, lambda v: theoretical_cdf(v, Tg))
    return stat_D, Tg

# Load pixel conversion factors
def load_conversion_factors(csv_path):
    df = pd.read_csv(csv_path)
    factor_dict = {}
    for _, row in df.iterrows():
        match = re.search(r'(\d+)r_(\d+)_ball_sizes', row['filename'])
        if match:
            rpm = int(match.group(1))
            balls = int(match.group(2))
            mean_diameter = (row['mean_width_px'] + row['mean_height_px']) / 2.0
            factor = 0.014 / mean_diameter
            factor_dict[(rpm, balls)] = factor
    return factor_dict

# Plotting function
def plot_ks_vs_scaled_sample_count(
    data_dir, tau_dict, conversion_dict, target_rpm,
    balls=10, pdf_dir=SCRIPT_DIR / 'figures'
):
    key = (target_rpm, balls)
    csv_name = f'vx_vy_data_{target_rpm}r_{balls}.csv'
    csv_path = os.path.join(data_dir, csv_name)

    if not os.path.exists(csv_path):
        print(f"Error: file not found {csv_path}")
        return
    if key not in tau_dict:
        print(f"Error: tau_dict missing ({target_rpm}, {balls})")
        return
    if key not in conversion_dict:
        print(f"Error: conversion_dict missing ({target_rpm}, {balls})")
        return

    tau_first_crossing = tau_dict[key]
    if tau_first_crossing <= 0:
        tau_first_crossing = 1
    fac = conversion_dict[key]

    v_all = load_and_clean_data(csv_path, fac)
    total_samples = len(v_all)
    print(f"{target_rpm}r, {balls} balls: total valid speed points = {total_samples}")

    steps = np.linspace(100, total_samples, 100, dtype=int)
    scaled_count_list = []
    D_list = []

    for step in steps:
        v_sub = v_all[:step]
        scaled_count = len(v_sub) / tau_first_crossing
        D_stat, _ = analyze_subset_ks(v_sub)
        if not np.isnan(D_stat):
            scaled_count_list.append(scaled_count)
            D_list.append(D_stat)

    if len(D_list) < 10:
        print("Warning: too few valid data points to plot a meaningful curve.")
        return

    # Smooth curve (rolling average)
    D_smooth = pd.Series(D_list).rolling(window=3, min_periods=1).mean().values
    scaled_counts = np.array(scaled_count_list)

    # Plot
    fig = plt.figure(figsize=(8, 5))
    color = '#396B93'
    plt.plot(scaled_counts, D_smooth, color=color, lw=2, label=f'{balls} spheres')

    # Mark all data points (scatter)
    plt.scatter(scaled_counts, D_smooth, color="#254B9C", s=8, zorder=5, label='Data points')

    # Stable reference line (mean of the last 30% of data)
    tail_start = int(0.7 * len(D_smooth))
    stable_value = np.mean(D_smooth[tail_start:])
    plt.axhline(y=stable_value, color='gray', linestyle='--', lw=1.5,
                label=f'Stable level ≈ {stable_value:.4f}')

    # Decorations
    plt.xlabel(r'Effective Sample Size, $N_{\mathrm{eff}}$', fontsize=18)
    plt.ylabel(r'KS Statistic, $D$', fontsize=18)
    plt.tick_params(axis='x', which='major', labelsize=18)
    plt.tick_params(axis='y', which='major', labelsize=18)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=18)
    plt.tight_layout()
    pdf_dir.mkdir(parents=True, exist_ok=True)
    base = pdf_dir / f'fig5_ks_moment_{target_rpm}r_{balls}.pdf'
    for ext in ('pdf', 'png'):
        plt.savefig(base.with_suffix(f'.{ext}'),
                    format=ext, dpi=300, bbox_inches='tight')
        print(f'Saved: {base.with_suffix(f".{ext}")}')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Export the figure as PDF')
    parser.add_argument('--pdf-dir', type=Path, default=SCRIPT_DIR / 'figures',
                        help='PDF output directory (default: figures beside this script)')
    args = parser.parse_args()
    pdf_dir = args.pdf_dir.expanduser()
    if not pdf_dir.is_absolute():
        pdf_dir = SCRIPT_DIR / pdf_dir
    rpm_list = [40, 60, 80, 100, 120, 140, 180, 200]
    ball_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    target_rpm = 120
    target_balls = 10

    print("Step 1: finding each VACF's first 1/e crossing...")
    tau_dict = batch_compute_autocorrelation(DATA_DIR, rpm_list, ball_list, max_lag_frames=500)

    print("\nStep 2: loading pixel conversion factors...")
    conversion_dict = load_conversion_factors(BALL_SIZE_CSV)

    print(f"\nStep 3: plotting KS convergence for {target_rpm} rpm, {target_balls} balls...")
    plot_ks_vs_scaled_sample_count(
        DATA_DIR, tau_dict, conversion_dict,
        target_rpm=target_rpm,
        balls=target_balls,
        pdf_dir=pdf_dir
    )
    print("All done!")

if __name__ == '__main__':
    main()
