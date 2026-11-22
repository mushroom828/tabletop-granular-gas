# SPDX-License-Identifier: AGPL-3.0-only
# See ../LICENSE for the full license terms.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import re
import argparse
from pathlib import Path
from scipy.constants import k          # Boltzmann constant
from velocity_samples_filter import load_lab_components, load_paired_components


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


# Container-frame velocities require finite ball/container samples from the same frame.
_, relative_components = load_paired_components(CSV_FILE, FAC)
if len(relative_components) == 0:
    raise ValueError("No paired camera/container velocity samples")
vx_list = relative_components[:, 0]
vy_list = relative_components[:, 1]
abs_speed_list = np.linalg.norm(relative_components, axis=1)

# Distribution Functions
def maxwell_speed_distribution(v, Tg):
    alpha = M / (2.0 * k * Tg)
    return 2.0 * alpha * v * np.exp(-alpha * v**2)

def maxwell_velocity_component_distribution(v, Tg):
    alpha = M / (2.0 * k * Tg)
    return np.sqrt(alpha / np.pi) * np.exp(-alpha * v**2)

# Preprocess speeds. These second moments include any phase-coherent motion.
v_clean = np.array(abs_speed_list)
upper = float(np.max(v_clean))  # Retain the full range for histograms and reference curves.
print(f"Number of valid points for speed analysis: {len(v_clean)}")
Tg_speed = M * np.mean(v_clean ** 2) / (2.0 * k)
Tg_vx = M * np.mean(vx_list ** 2) / k
Tg_vy = M * np.mean(vy_list ** 2) / k
print(f"Raw second-moment scales (including coherent motion): "
      f"speed={Tg_speed:.6e} K, vx={Tg_vx:.6e} K, vy={Tg_vy:.6e} K")

# Generate histogram for plotting
n_bins = 300
bin_edges_speed = np.histogram_bin_edges(v_clean, bins=n_bins, range=(0, upper))
v_smooth = np.linspace(0, upper, 600)
# Match the laboratory-frame display range for this condition without changing
# the paired relative-velocity samples used above for the second moments.
lab_display_components = load_lab_components(CSV_FILE, FAC)
component_limit = max(0.288, np.max(np.abs(lab_display_components[:, 0])) * 1.02,
                       np.max(np.abs(lab_display_components[:, 1])) * 1.02)
vx_grid = np.linspace(-component_limit, component_limit, 600)
vy_grid = np.linspace(-component_limit, component_limit, 600)
bin_edges_vx = np.histogram_bin_edges(vx_list, bins=n_bins)
bin_edges_vy = np.histogram_bin_edges(vy_list, bins=n_bins)

# Construct Gaussian and Maxwell reference curves from the measured second moments.
reference_speed = maxwell_speed_distribution(v_smooth, Tg_speed)
reference_vx = maxwell_velocity_component_distribution(vx_grid, Tg_vx)
reference_vy = maxwell_velocity_component_distribution(vy_grid, Tg_vy)

# Plot histograms with moment-based reference curves.
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Speed distribution plot
axes[0].hist(v_clean, bins=bin_edges_speed, density=True, alpha=0.8, label='Speed',
             color="#82aed2")
axes[0].plot(v_smooth, reference_speed, color="#62666E", linestyle='--',
             linewidth=1.3,
             label=f'Maxwellian reference\n$T_{{\\mathrm{{g}}}} = {format_tex_sci(Tg_speed)}$ K')
axes[0].set_xlabel(r'$v$ (m/s)', fontsize=25)
axes[0].set_ylabel('Probability density', fontsize=25)
# axes[0].set_title(r'Speed distribution $f(v)$', fontsize=25)
axes[0].set_title(r'(d) Speed', fontsize=25)
axes[0].set_xlim(0, 0.3)
axes[0].set_ylim(0,15)
axes[0].tick_params(axis='both', which='major', labelsize=24)
axes[0].legend(fontsize=20, loc='upper right', framealpha=0.6)
axes[0].grid(True, alpha=0.3)

# vx component
axes[1].hist(vx_list, bins=bin_edges_vx, density=True, alpha=0.8, label=r'$v_x$', color="#cbdfecc6")
axes[1].plot(vx_grid, reference_vx, color="#62666E", linestyle='--',
             linewidth=1.3,
             label=f'Gaussian reference\n$T_{{\\mathrm{{g}}}} = {format_tex_sci(Tg_vx)}$ K')
axes[1].set_xlabel(r'$v_x$ (m/s)', fontsize=25)
# axes[1].set_title(r'$v_x$ component distribution $g(v_x)$', fontsize=25)
axes[1].set_title(r'(e) $v_x$', fontsize=25)
axes[1].set_xlim(-component_limit, component_limit)
axes[1].set_ylim(0,7.5)
axes[1].tick_params(axis='both', which='major', labelsize=24)
axes[1].legend(fontsize=20, loc='upper right', framealpha=0.6)
axes[1].grid(True, alpha=0.3)

# vy component
axes[2].hist(vy_list, bins=bin_edges_vy, density=True, alpha=0.8, label=r'$v_y$', color="#f19780")
axes[2].plot(vy_grid, reference_vy, color="#62666E", linestyle='--',
             linewidth=1.3,
             label=f'Gaussian reference\n$T_{{\\mathrm{{g}}}} = {format_tex_sci(Tg_vy)}$ K')
axes[2].set_xlabel(r'$v_y$ (m/s)', fontsize=25)
# axes[2].set_title(r'$v_y$ component distribution $g(v_y)$', fontsize=25)
axes[2].set_title(r'(f) $v_y$', fontsize=25)
axes[2].set_xlim(-component_limit, component_limit)
axes[2].set_ylim(0,7.5)
axes[2].tick_params(axis='both', which='major', labelsize=24)
axes[2].legend(fontsize=20, loc='upper right', framealpha=0.6)
axes[2].grid(True, alpha=0.3)

for ax in axes:
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.tight_layout()
PDF_DIR.mkdir(parents=True, exist_ok=True)
base = PDF_DIR / f'fig4_container_{condition}'
for ext in ('pdf', 'png'):
    fig.savefig(base.with_suffix(f'.{ext}'),
                format=ext, dpi=300, bbox_inches='tight')
    print(f'Saved: {base.with_suffix(f".{ext}")}')
plt.show()
