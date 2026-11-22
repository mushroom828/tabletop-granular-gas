# SPDX-License-Identifier: AGPL-3.0-only
# See ../LICENSE for the full license terms.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker      
import os
import re
import argparse
from pathlib import Path


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['pdf.fonttype'] = 42

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / '2_GetResults' / 'processed_data'

# Autocorrelation core function
def compute_autocorrelation(csv_path, max_lag_frames=500):
    """
    Compute the velocity autocorrelation function for a single CSV file,
    return the number of frames tau_frames required to decay to 1/e.
    """
    df = pd.read_csv(csv_path)
    # Filter out 'container'
    df = df[df['obj_id'] != 'container'].copy()
    if df.empty:
        return None, 0

    acf_dot_sum = np.zeros(max_lag_frames)
    acf_count = np.zeros(max_lag_frames)

    # Group by obj_id (to handle SORT ID fragmentation)
    for obj_id, group in df.groupby('obj_id'):
        group = group.sort_values('frame')
        # Ensure frame numbers are continuous, fill gaps with NaN.
        min_t, max_t = group['frame'].min(), group['frame'].max()
        group = group.set_index('frame').reindex(range(min_t, max_t + 1))
        vx = group['vx'].values
        vy = group['vy'].values

        # Compute dot product for each lag
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

    # Normalize ACF
    valid_lags = acf_count > 0
    if not np.any(valid_lags):
        return None, 0
    acf = np.zeros(max_lag_frames)
    acf[valid_lags] = acf_dot_sum[valid_lags] / acf_count[valid_lags]
    if acf[0] > 0:
        acf = acf / acf[0]
    else:
        return None, 0

    # Find first lag where ACF drops below 1/e
    threshold = np.exp(-1)
    decay_indices = np.where(valid_lags & (acf < threshold))[0]
    if len(decay_indices) > 0:
        tau_frames = decay_indices[0]
    else:
        tau_frames = max_lag_frames  # Not decayed to threshold, take max frames

    return tau_frames, acf  # Return frame count and full ACF (optional)

# Batch processing
def batch_compute_autocorrelation(data_dir, rpm_list, ball_list, max_lag_frames=500):
    """
    Scan all CSVs in data_dir and compute tau_frames for each (rpm, balls) combination.
    Return a dictionary {(rpm, balls): tau_frames}
    """
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
        tau, _ = compute_autocorrelation(path, max_lag_frames)
        if tau is not None:
            results[(rpm, balls)] = tau
            print(f"{rpm}r, {balls} balls: tau = {tau} frames")
        else:
            print(f"{rpm}r, {balls} balls: computation failed")
    return results

# Plot heatmap
def plot_autocorrelation_heatmap(results, rpm_list, ball_list, pdf_dir=SCRIPT_DIR / 'figures'):
    """
    Plot and export a heatmap based on the results dictionary.
    The x-axis shows angular velocity ω (rad/min) with π notation (e.g., 80π).
    """
    # Build matrix
    data_matrix = np.full((len(ball_list), len(rpm_list)), np.nan)
    for i, balls in enumerate(ball_list):
        for j, rpm in enumerate(rpm_list):
            key = (rpm, balls)
            if key in results:
                data_matrix[i, j] = results[key]  # frame count

    # Plot heatmap (YlGnBu colormap, linear normalization, 2 balls at bottom, 10 at top)
    fig, ax = plt.subplots(figsize=(10, 6))
    # Key: origin='lower' places the first row (2 balls) at the bottom
    im = ax.imshow(data_matrix, cmap='YlGnBu', aspect='auto', origin='lower')

    # Generate x-axis labels with π, e.g., 40 rpm -> 80π, 60 -> 120π, ... 200 -> 400π
    omega_labels = [r'${}\pi$'.format(2*r) for r in rpm_list]

    # Set ticks
    ax.set_xticks(np.arange(len(rpm_list)))
    ax.set_xticklabels(omega_labels)
    ax.set_yticks(np.arange(len(ball_list)))
    ax.set_yticklabels(ball_list)
    ax.tick_params(axis='x', which='major', labelsize=22)
    ax.tick_params(axis='y', which='major', labelsize=22)
    ax.set_xlabel(r'$\omega$ (rad/min)', fontsize=22)
    ax.set_ylabel('Number of Spheres', fontsize=22)

    # Display values in each cell
    for i in range(len(ball_list)):
        for j in range(len(rpm_list)):
            val = data_matrix[i, j]
            if not np.isnan(val):
                # Adjust text color based on value (relative to median)
                text_color = 'white' if val > np.nanmedian(data_matrix) else 'black'
                ax.text(j, i, f'{int(val)}', ha='center', va='center',
                        color=text_color, fontsize=20)

    # Rectangular colorbar (no arrow)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\tau_{\mathrm{ac}}$ (frames)', fontsize=22)
    cbar.ax.tick_params(labelsize=19)

    plt.tight_layout()
    pdf_dir.mkdir(parents=True, exist_ok=True)
    base = pdf_dir / 'fig6_vacf_heatmap.pdf'
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
    # Parameter settings
    rpm_list = [40, 60, 80, 100, 120, 140, 180, 200]
    ball_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    max_lag_frames = 500   # Adjust as needed

    print("Finding each VACF's first 1/e crossing...")
    results = batch_compute_autocorrelation(DATA_DIR, rpm_list, ball_list, max_lag_frames)

    if not results:
        print("No data successfully computed. Please check file paths and filename format.")
        return

    print("\nPlotting heatmap...")
    plot_autocorrelation_heatmap(results, rpm_list, ball_list, pdf_dir=pdf_dir)

if __name__ == '__main__':
    main()
