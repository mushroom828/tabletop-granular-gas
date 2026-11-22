# SPDX-License-Identifier: AGPL-3.0-only
# See ../LICENSE for the full license terms.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import kstest
from scipy.special import erf
import csv
import pandas as pd
import re
import argparse
from pathlib import Path
from scipy.constants import k          # Boltzmann constant
from velocity_samples_filter import load_lab_components


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['pdf.fonttype'] = 42

# Particle mass (kg)
M = 0.0036


# Choose one condition for both Figure 4 scripts; override with --condition 120r_10.
DEFAULT_CONDITION = '100r_10'
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / '2_GetResults' / 'processed_data'
parser = argparse.ArgumentParser(description='Plot Figure 4 for one RPM and particle count')
parser.add_argument('--condition', default=DEFAULT_CONDITION,
                    help='Condition such as 100r_10 or 120r_10')
parser.add_argument('--pdf-dir', type=Path, default=SCRIPT_DIR / 'figures',
                    help='Directory for the exported Figure 4 PDF (default: figures beside this script)')
args = parser.parse_args()
match = re.fullmatch(r'(\d+)r_(\d+)', args.condition)
if match is None:
    parser.error(f'Invalid condition {args.condition!r}; expected e.g. 100r_10')
sample_key = (int(match.group(1)), int(match.group(2)))
condition = f'{sample_key[0]}r_{sample_key[1]}'
CSV_FILE = DATA_DIR / f'vx_vy_data_{condition}.csv'
BALL_SIZE_CSV = DATA_DIR / 'ball_sizes_means.csv'
PDF_DIR = args.pdf_dir.expanduser()
if not PDF_DIR.is_absolute():
    PDF_DIR = SCRIPT_DIR / PDF_DIR
if not CSV_FILE.is_file():
    parser.error(f'CSV file not found: {CSV_FILE}')

# Conversion Factor Reading
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

# Match the selected data file to its own pixel-to-metre calibration.
conversion_dict = load_conversion_factors(BALL_SIZE_CSV)
if sample_key not in conversion_dict:
    raise ValueError(f'Conversion factor for {condition} not found in {BALL_SIZE_CSV}')
FAC = conversion_dict[sample_key]
print(f'Condition {condition}: conversion factor {FAC:.6f} m/px')


def format_tex_sci(value, precision=2):
    if value == 0:
        return '0'
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0 ** exponent)
    return f'{mantissa:.{precision}f}\\times 10^{{{exponent}}}'

# Laboratory-frame statistics use all finite ball velocities.
camera_components = load_lab_components(CSV_FILE, FAC)
if len(camera_components) == 0:
    raise ValueError("No finite laboratory-frame ball velocity samples")
vx_list = camera_components[:, 0]
vy_list = camera_components[:, 1]
abs_speed_list = np.linalg.norm(camera_components, axis=1)

# Distribution Functions
def maxwell_speed_distribution(v, Tg):
    alpha = M / (2.0 * k * Tg)
    return 2.0 * alpha * v * np.exp(-alpha * v**2)

def maxwell_velocity_component_distribution(v, Tg):
    alpha = M / (2.0 * k * Tg)
    return np.sqrt(alpha / np.pi) * np.exp(-alpha * v**2)

def maxwell_speed_cdf(v, Tg):
    alpha = M / (2.0 * k * Tg)
    return 1.0 - np.exp(-alpha * v**2)

def maxwell_component_cdf(v, Tg):
    alpha = M / (2.0 * k * Tg)
    return 0.5 * (1.0 + erf(np.sqrt(alpha) * v))

# Core Fitting Function
def fit_and_ks(velocities, upper_bound):
    if len(velocities) < 50:
        return None, 0, 1.0, 0.0

    # Histogram
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

    # Initial guess
    mean_sq = np.mean(velocities ** 2)
    Tg_initial = M * mean_sq / (2.0 * k)

    # Multi-start fitting
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

    # KS test
    D_stat, p_value = kstest(velocities, lambda v: maxwell_speed_cdf(v, best_Tg))

    return best_Tg, best_r2, D_stat, p_value


v_clean = np.array(abs_speed_list)
upper = float(np.max(v_clean))  # Set the histogram and plot range from retained data.
print(f"Number of valid points for speed fitting: {len(v_clean)}")

# Perform fitting
Tg_speed, r2_speed, ks_speed, p_speed = fit_and_ks(v_clean, upper)
print(f"Tg = {Tg_speed:.6e} K")
print(f"R^2 = {r2_speed:.6f}")
print(f"KS statistic D = {ks_speed:.6f}, p-value = {p_speed:.6f}")

# Generate histogram for plotting (no low-frequency filtering)
n_bins = 300
hist_speed, bin_edges_speed = np.histogram(v_clean, bins=n_bins, density=True, range=(0, upper))
bin_centers_speed = (bin_edges_speed[:-1] + bin_edges_speed[1:]) / 2
v_smooth = np.linspace(0, upper, 1000)
fitted_speed = maxwell_speed_distribution(v_smooth, Tg_speed)


def fit_velocity_component_distribution(velocity_data):
    n_bins = 300
    bin_counts, bin_edges = np.histogram(velocity_data, bins=n_bins, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    mean_sq = np.mean(np.array(velocity_data)**2)
    if mean_sq == 0:
        initial_guess = [1e6]
    else:
        initial_guess = [M * mean_sq / k]
    try:
        params, _ = curve_fit(maxwell_velocity_component_distribution, bin_centers, bin_counts,
                              p0=initial_guess, maxfev=10000)
    except RuntimeError as exc:
        raise RuntimeError("Maxwell component fit failed") from exc
    Tg = params[0]
    ss_res = np.sum((bin_counts - maxwell_velocity_component_distribution(bin_centers, Tg)) ** 2)
    ss_tot = np.sum((bin_counts - np.mean(bin_counts)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    ks_stat, ks_p = kstest(velocity_data, maxwell_component_cdf, args=(Tg,))
    return bin_edges, bin_centers, bin_counts, params, r_squared, ks_stat, ks_p

print("\n=== vx Component Distribution Fitting (Tg parameter) ===")
bin_edges_vx, bin_centers_vx, bin_counts_vx, params_vx, r_squared_vx, ks_vx, p_vx = fit_velocity_component_distribution(vx_list)
Tg_vx = params_vx[0]
fitted_vx = maxwell_velocity_component_distribution(bin_centers_vx, Tg_vx)
print(f"Tg = {Tg_vx:.6e} K, R^2 = {r_squared_vx:.6f}, D = {ks_vx:.6f}")

print("\n=== vy Component Distribution Fitting (Tg parameter) ===")
bin_edges_vy, bin_centers_vy, bin_counts_vy, params_vy, r_squared_vy, ks_vy, p_vy = fit_velocity_component_distribution(vy_list)
Tg_vy = params_vy[0]
fitted_vy = maxwell_velocity_component_distribution(bin_centers_vy, Tg_vy)
print(f"Tg = {Tg_vy:.6e} K, R^2 = {r_squared_vy:.6f}, D = {ks_vy:.6f}")


speed_eq = r'$f(v) = \frac{m}{k_B T_{\mathrm{g}}} v \exp\left(-\frac{m v^2}{2 k_B T_{\mathrm{g}}}\right)$'
vx_eq = r'$g(v_x) = \sqrt{\frac{m}{2\pi k_B T_{\mathrm{g}}}} \exp\left(-\frac{m v_x^2}{2 k_B T_{\mathrm{g}}}\right)$'
vy_eq = r'$g(v_y) = \sqrt{\frac{m}{2\pi k_B T_{\mathrm{g}}}} \exp\left(-\frac{m v_y^2}{2 k_B T_{\mathrm{g}}}\right)$'

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Speed distribution plot
axes[0].hist(v_clean, bins=n_bins, density=True, alpha=0.8, label='Speed',
             color="#82aed2")
axes[0].plot(v_smooth, fitted_speed, "#396B93", linewidth=2,
            label=f'$T_{{\\mathrm{{g}}}} = {format_tex_sci(Tg_speed)}$ K\n$D = {ks_speed:.4f}$')
axes[0].set_xlabel(r'$v$ (m/s)', fontsize=25)
axes[0].set_ylabel('Probability density', fontsize=25)
# axes[0].set_title(r'Speed distribution $f(v)$', fontsize=25)
axes[0].set_title(r'(a) Speed', fontsize=25)
axes[0].set_xlim(0, 0.3)
axes[0].set_ylim(0,15)
axes[0].tick_params(axis='both', which='major', labelsize=24)
axes[0].legend(fontsize=20, loc='upper right')
axes[0].grid(True, alpha=0.3)

# vx component
axes[1].hist(vx_list, bins=bin_edges_vx, density=True, alpha=0.8, label=r'$v_x$', color="#cbdfecc6")
axes[1].plot(bin_centers_vx, fitted_vx, "#7C7F90", linewidth=2,
            label=f'$T_{{\\mathrm{{g}}}} = {format_tex_sci(Tg_vx)}$ K\n$D = {ks_vx:.4f}$')
axes[1].set_xlabel(r'$v_x$ (m/s)', fontsize=25)
# axes[1].set_title(r'$v_x$ component distribution $g(v_x)$', fontsize=25)
axes[1].set_title(r'(b) $v_x$', fontsize=25)
component_limit = max(0.288, np.max(np.abs(vx_list)) * 1.02,
                      np.max(np.abs(vy_list)) * 1.02)
axes[1].set_xlim(-component_limit, component_limit)
axes[1].set_ylim(0,15)
axes[1].tick_params(axis='both', which='major', labelsize=24)
axes[1].legend(fontsize=20, loc='upper right')
axes[1].grid(True, alpha=0.3)

# vy component
axes[2].hist(vy_list, bins=bin_edges_vy, density=True, alpha=0.8, label=r'$v_y$', color="#f19780")
axes[2].plot(bin_centers_vy, fitted_vy, "#CB2B1F", linewidth=2,
            label=f'$T_{{\\mathrm{{g}}}} = {format_tex_sci(Tg_vy)}$ K\n$D = {ks_vy:.4f}$')
axes[2].set_xlabel(r'$v_y$ (m/s)', fontsize=25)
# axes[2].set_title(r'$v_y$ component distribution $g(v_y)$', fontsize=25)
axes[2].set_title(r'(c) $v_y$', fontsize=25)
axes[2].set_xlim(-component_limit, component_limit)
axes[2].set_ylim(0,15)
axes[2].tick_params(axis='both', which='major', labelsize=24)
axes[2].legend(fontsize=20, loc='upper right')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
PDF_DIR.mkdir(parents=True, exist_ok=True)
base = PDF_DIR / f'fig4_camera_{condition}'
for ext in ('pdf', 'png'):
    fig.savefig(base.with_suffix(f'.{ext}'),
                format=ext, dpi=300, bbox_inches='tight')
    print(f'Saved: {base.with_suffix(f".{ext}")}')
plt.show()

# High-energy tail zoom plot
# Use the paper's display window for each representative condition.
# This does not change histogram normalization, fitting, or KS samples.
tail_windows = {'100r_10': (0.10, 0.25), '120r_10': (0.12, 0.30)}
v_tail_start, v_tail_end = tail_windows.get(condition, (0.10, 0.30))

tail_mask = (bin_centers_speed >= v_tail_start) & (bin_centers_speed <= v_tail_end)
v_tail_centers = bin_centers_speed[tail_mask]
counts_tail = hist_speed[tail_mask]

v_tail_smooth = np.linspace(v_tail_start, v_tail_end, 300)
fitted_tail = maxwell_speed_distribution(v_tail_smooth, Tg_speed)

fig_tail = plt.figure(figsize=(8, 5))
ax_tail = fig_tail.add_subplot(111)

bin_width = np.diff(bin_edges_speed)[0]
ax_tail.bar(v_tail_centers, counts_tail, width=bin_width*0.9,
            alpha=0.8, color="#6ea6d4")
ax_tail.plot(v_tail_smooth, fitted_tail, "#396B93", linewidth=2)

ax_tail.set_xlim(v_tail_start, v_tail_end)
tail_density_max = max(np.max(counts_tail, initial=0), np.max(fitted_tail, initial=0))
ax_tail.set_ylim(bottom=0, top=tail_density_max * 1.15 if tail_density_max > 0 else 1)
ax_tail.tick_params(axis='x', which='major', labelsize=35)
ax_tail.tick_params(axis='y', which='major', labelsize=35)

plt.tight_layout()
PDF_DIR.mkdir(parents=True, exist_ok=True)
base = PDF_DIR / f'fig4_camera_tail_{condition}'
for ext in ('pdf', 'png'):
    fig_tail.savefig(base.with_suffix(f'.{ext}'),
                format=ext, dpi=300, bbox_inches='tight')
    print(f'Saved: {base.with_suffix(f".{ext}")}')
plt.show()

