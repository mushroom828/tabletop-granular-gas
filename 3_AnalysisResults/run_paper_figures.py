"""Generate the main-text Figures 4–7 in sequence using the existing scripts."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_PLOTS = (
    ("Figure 4: 100r_10", "fig4_VelocityDistribution.py", ("--condition", "100r_10")),
    ("Figure 4: 120r_10", "fig4_VelocityDistribution.py", ("--condition", "120r_10")),
    ("Figure 5: KS convergence", "fig5_KSAnalysis.py", ()),
    ("Figure 6: velocity autocorrelation", "fig6_AutocorrelationHeatmap.py", ()),
    ("Figure 7: laboratory-frame speed distributions", "fig7_TgHeatmap.py", ()),
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate the main-text Figures 4–7 sequentially, without opening plot windows."
    )
    parser.add_argument(
        "--pdf-dir", type=Path, default=SCRIPT_DIR / "figures",
        help="PDF/PNG output directory (default: figures beside this script); "
             "relative paths are resolved from the analysis directory",
    )
    args = parser.parse_args(argv)
    output_dir = args.pdf_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = SCRIPT_DIR / output_dir
    output_dir = output_dir.resolve()

    for _, filename, _ in PAPER_PLOTS:
        script_path = SCRIPT_DIR / filename
        if not script_path.is_file():
            parser.error(f"Plotting script not found: {script_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"  # Save plots without blocking on plt.show().
    env["PYTHONUNBUFFERED"] = "1"
    # Hide only the expected warning from plt.show() on the noninteractive backend.
    warning_filter = "ignore:FigureCanvasAgg is non-interactive:UserWarning"
    env["PYTHONWARNINGS"] = f"{env.get('PYTHONWARNINGS', '')},{warning_filter}".lstrip(",")
    print(f"Figure output directory: {output_dir}", flush=True)
    for index, (label, filename, options) in enumerate(PAPER_PLOTS, 1):
        print(f"\n[{index}/{len(PAPER_PLOTS)}] {label}", flush=True)
        command = [sys.executable, str(SCRIPT_DIR / filename), *options,
                   "--pdf-dir", str(output_dir)]
        result = subprocess.run(command, cwd=SCRIPT_DIR, env=env, check=False)
        if result.returncode:
            print(f"Stopped: {filename} failed with exit code {result.returncode}.",
                  file=sys.stderr, flush=True)
            return 1

    print(f"\nFinished all {len(PAPER_PLOTS)} plotting steps. Figures saved to {output_dir}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
