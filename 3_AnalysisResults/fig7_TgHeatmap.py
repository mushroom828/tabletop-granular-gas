# SPDX-License-Identifier: AGPL-3.0-only
# See ../LICENSE for the full license terms.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import LogLocator, MaxNLocator
import pandas as pd
import os
from scipy.optimize import curve_fit
from scipy.stats import kstest
import re
import argparse
from pathlib import Path
from scipy.constants import k
from velocity_samples_filter import load_lab_components


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['pdf.fonttype'] = 42

M = 0.0036                       # Particle mass (kg)
FIT_COLOR = '#144A74'            # Fit curve color
    
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / '2_GetResults' / 'processed_data'
BALL_SIZE_CSV = DATA_DIR / 'ball_sizes_means.csv'

# New model functions
def maxwell_speed_distribution(v, Tg):
    alpha = M / (2.0 * k * Tg)
    return 2.0 * alpha * v * np.exp(-alpha * v**2)

def maxwell_speed_cdf(v, Tg):
    alpha = M / (2.0 * k * Tg)
    return 1.0 - np.exp(-alpha * v**2)

# Fitting and KS test
def fit_and_ks(velocities, upper_bound):
    if len(velocities) < 50:
        return None, 0, 1.0, 0.0

    n_bins = 300
    hist, bin_edges = np.histogram(velocities, bins=n_bins,
                                   density=True, range=(0, upper_bound))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_max = np.max(hist)
    valid_mask = hist > hist_max * 0.01
    bin_centers_valid = bin_centers[valid_mask]
    hist_valid = hist[valid_mask]
    if len(bin_centers_valid) < 5:
        bin_centers_valid, hist_valid = bin_centers, hist

    mean_sq = np.mean(velocities ** 2)
    Tg_initial = M * mean_sq / (2.0 * k)

    initial_guesses = np.logspace(np.log10(Tg_initial*0.1), np.log10(Tg_initial*10), 20)
    best_r2, best_Tg = -np.inf, None
    for p0 in initial_guesses:
        try:
            popt, _ = curve_fit(maxwell_speed_distribution, bin_centers_valid, hist_valid,
                                p0=[p0], bounds=(1e-10, np.inf), maxfev=5000)
            Tg_cur = popt[0]
            y_pred = maxwell_speed_distribution(bin_centers_valid, Tg_cur)
            ss_res = np.sum((hist_valid - y_pred) ** 2)
            ss_tot = np.sum((hist_valid - np.mean(hist_valid)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
            if r2 > best_r2 and Tg_cur > 0:
                best_r2, best_Tg = r2, Tg_cur
        except Exception:
            continue

    if best_Tg is None:
        raise RuntimeError("Maxwell speed fit failed; no fitted temperature is available")

    D_stat, p_value = kstest(velocities, lambda v: maxwell_speed_cdf(v, best_Tg))
    return best_Tg, best_r2, D_stat, p_value

# File scanning
def get_all_data_files(data_dir):
    if not os.path.exists(data_dir):
        print(f'Error: directory {data_dir} does not exist')
        return []
    csv_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.csv')]
    def extract_key(filename):
        match = re.search(r'vx_vy_data_(\d+)r_(\d+)\.csv', filename)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return (9999, 9999)
    csv_files.sort(key=extract_key)
    return [os.path.join(data_dir, f) for f in csv_files]

# Conversion factor loading
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

# Color mapping 
def get_color_for_Tg(Tg, vmin, vmax):
    norm = (np.log10(Tg) - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))
    norm = max(0.0, min(1.0, norm))
    base_color = plt.cm.coolwarm(norm)
    adjusted_color = [base_color[0], base_color[1], base_color[2], 0.5]
    return adjusted_color

# Plot 5x8 combined figure
def create_combined_plot(data_files, conversion_dict, pdf_dir=SCRIPT_DIR / 'figures'):
    rpm_list = [40, 60, 80, 100, 120, 140, 180, 200]
    ball_list = [2, 4, 6, 8, 10]

    file_dict = {}
    for fp in data_files:
        fname = os.path.basename(fp)
        match = re.search(r'vx_vy_data_(\d+)r_(\d+)\.csv', fname)
        if match:
            r = int(match.group(1))
            b = int(match.group(2))
            if r in rpm_list and b in ball_list:
                file_dict[(r, b)] = fp

    fig = plt.figure(figsize=(26, 12), facecolor='white')
    gs = plt.GridSpec(5, 9, figure=fig, width_ratios=[1]*8 + [0.1])

    print("Phase 1: fitting all data, collecting Tg range...")
    tg_list = []
    temp_results = []
    for balls in ball_list:
        for rpm in rpm_list:
            key = (rpm, balls)
            if key not in file_dict:
                temp_results.append((key, None, None, None, None, None))
                continue
            file = file_dict[key]
            try:
                fac = conversion_dict.get(key, None)
                if fac is None:
                    temp_results.append((key, None, None, None, None, None))
                    continue
                camera_components = load_lab_components(file, fac)
                v_clean = np.linalg.norm(camera_components, axis=1)
                if len(v_clean) == 0 or not np.any(v_clean > 0):
                    temp_results.append((key, None, None, None, None, None))
                    continue
                upper = float(np.max(v_clean))
                Tg, r2, D, p = fit_and_ks(v_clean, upper)
                if Tg is not None:
                    tg_list.append(Tg)
                    temp_results.append((key, r2, Tg, D, p, v_clean))
                else:
                    temp_results.append((key, None, None, None, None, None))
            except Exception as e:
                print(f"  Error processing {rpm}r_{balls}: {e}")
                temp_results.append((key, None, None, None, None, None))

    if len(tg_list) == 0:
        print("No valid data!")
        return []

    # Zoom the main panels without discarding samples or renormalizing the PDF.
    # Share a range covering at least 99.9% of each condition at the same RPM.
    display_upper = {}
    full_upper = {}
    for key, r2, Tg, D, p, samples in temp_results:
        if samples is not None and len(samples):
            rpm = key[0]
            display_upper[rpm] = max(display_upper.get(rpm, 0.0),
                                     float(np.quantile(samples, 0.999)))
            full_upper[rpm] = max(full_upper.get(rpm, 0.0), float(np.max(samples)))

    tg_min = np.min(tg_list)
    tg_max = np.max(tg_list)
    print(f"Tg range: {tg_min:.6e} ~ {tg_max:.6e} K")

    # Build quick lookup dictionary
    results_dict = {item[0]: item[1:] if item[1] is not None else None for item in temp_results}

    print("Phase 2: drawing subplots (bottom: 2 balls, top: 10 balls)...")
    for i, balls in enumerate(reversed(ball_list)):
        for j, rpm in enumerate(rpm_list):
            ax = fig.add_subplot(gs[i, j])
            display_limit = display_upper.get(rpm)
            if display_limit is not None:
                ax.set_xlim(0, display_limit)
            key = (rpm, balls)
            if key not in file_dict:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                continue
            data_pack = results_dict.get(key)
            if data_pack is None:
                ax.text(0.5, 0.5, 'Fail', ha='center', va='center', transform=ax.transAxes)
                continue
            r2, Tg, D, p, v_clean = data_pack
            plot_upper = display_upper[rpm]
            color = get_color_for_Tg(Tg, tg_min, tg_max)
            # Include off-screen tails in the density normalization.
            bin_width = plot_upper / 300
            edges = np.arange(int(np.ceil(full_upper[rpm] / bin_width)) + 1) * bin_width
            ax.hist(v_clean, bins=edges, density=True,
                    color=color, alpha=1, edgecolor='none')
            v_smooth = np.linspace(0, full_upper[rpm], 1000)
            ax.plot(v_smooth, maxwell_speed_distribution(v_smooth, Tg), color=FIT_COLOR, lw=1.2)
            ax.text(0.98, 0.97, f'$D$ = {D:.3f}', transform=ax.transAxes,
                    fontsize=35, ha='right', va='top', color='black', zorder=10,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=0.5))
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=12)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            print(f"{rpm}r_{balls}: data points {len(v_clean)}, R^2={r2:.4f}, Tg={Tg:.6e} K, D={D:.4f}")

    # Colorbar (fixed 5 main ticks)
    cax = fig.add_subplot(gs[:, 8])
    norm = mcolors.LogNorm(vmin=tg_min, vmax=tg_max)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)

    # Fixed ticks specified by user
    custom_ticks = [3e17, 1e18, 3e18, 6e18, 8.5e18]
    custom_labels = ['3×10¹⁷', '1×10¹⁸', '3×10¹⁸', '6×10¹⁸', '8.5×10¹⁸']
    cbar.set_ticks(custom_ticks)
    cbar.set_ticklabels(custom_labels)

    cbar.set_label(r'$T_{\mathrm{g}}$ (K)', fontsize=30)
    cax.tick_params(labelsize=35)

    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    base = pdf_dir / 'fig7_camera_allcondition.pdf'
    for ext in ('pdf', 'png'):
            plt.savefig(base.with_suffix(f'.{ext}'),
                        format=ext, dpi=300, bbox_inches='tight')
            print(f'Saved: {base.with_suffix(f".{ext}")}')
    plt.show()

    results_list = []
    for key, r2, Tg, D, p, _ in temp_results:
        if Tg is not None:
            results_list.append((f"{key[0]}r_{key[1]}", r2, Tg, D, p))
    return results_list


def main():
    parser = argparse.ArgumentParser(description='Export the figure as PDF')
    parser.add_argument('--pdf-dir', type=Path, default=SCRIPT_DIR / 'figures',
                        help='PDF output directory (default: figures beside this script)')
    args = parser.parse_args()
    pdf_dir = args.pdf_dir.expanduser()
    if not pdf_dir.is_absolute():
        pdf_dir = SCRIPT_DIR / pdf_dir
    print("Loading pixel conversion factors...")
    conversion_dict = load_conversion_factors(BALL_SIZE_CSV)
    print(f'Loaded {len(conversion_dict)} conversion factors')

    data_files = get_all_data_files(DATA_DIR)
    if not data_files:
        print('No CSV files found!')
        return

    print('Starting to plot 5x8 combined figure (Tg heatmap + KS test)...')
    results = create_combined_plot(data_files, conversion_dict, pdf_dir=pdf_dir)

    if results:
        tg_values = [r[2] for r in results if r[2] is not None]
        print(f'Final Tg range: [{min(tg_values):.6e}, {max(tg_values):.6e}] K')
    print(f'Done, processed {len(results)} conditions')

if __name__ == '__main__':
    main()

