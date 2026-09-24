"""Plot results from the NOAA SST experiment.

Reads noaa_sst_results.csv and generates a summary figure.
This script also re-generates the decomposition and reconstruction
plots from the saved NOAA data and coefficient files if available.

For standalone use, this script reads the CSV output from the
NOAA SST experiment and prints a formatted summary.

Usage:
    python utils/plot_noaa_sst.py [path/to/noaa_sst_results.csv]

If no path is provided, reads from noaa_sst_results.csv in the current
working directory.
"""

import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_results(csv_path):
    """Load NOAA SST experiment results from CSV.

    Returns a list of dicts with keys: metric, centralized, merged,
    west_only, east_only.
    """
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(dict(row))
    return results


def plot_results(results, output_path='noaa_sst_summary.png'):
    """Generate a summary bar chart from the NOAA SST results.

    Parameters
    ----------
    results : list of dict
        Loaded CSV results.
    output_path : str
        Output file path for the figure.
    """
    # Extract RMSE values where available
    metrics = []
    values = []
    for row in results:
        metric = row.get('metric', '')
        central_val = row.get('centralized', '')
        merged_val = row.get('merged', '')
        if central_val:
            metrics.append(f"{metric} (centralized)")
            values.append(float(central_val))
        if merged_val:
            metrics.append(f"{metric} (merged)")
            values.append(float(merged_val))

    if not values:
        print("No plottable values found in results.")
        return

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1a1a2e')
    ax.set_facecolor('#0d1117')

    x = np.arange(len(metrics))
    colors = ['#4ade80' if 'centralized' in m else '#c792ea' for m in metrics]
    ax.bar(x, values, color=colors, edgecolor='#30363d')
    ax.set_xlabel('Configuration', color='white')
    ax.set_ylabel('RMSE (deg C)', color='white')
    ax.set_title('NOAA SST Reconstruction RMSE', color='white',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha='right', fontsize=9)
    ax.tick_params(colors='#666')
    for spine in ax.spines.values():
        spine.set_color('#30363d')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#1a1a2e',
                bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = 'noaa_sst_results.csv'

    if not os.path.exists(csv_path):
        print(f"ERROR: Results file not found: {csv_path}")
        print("Run experiment_noaa_sst.py first to generate results.")
        sys.exit(1)

    results = load_results(csv_path)
    print(f"Loaded {len(results)} rows from {csv_path}")

    plot_results(results)

    # Print summary table
    print("\nNOAA SST Results:")
    print(f"{'Metric':<20} {'Centralized':>14} {'Merged':>14} "
          f"{'West only':>14} {'East only':>14}")
    print(f"{'-'*80}")
    for row in results:
        metric = row.get('metric', '')
        central = row.get('centralized', '-')
        merged = row.get('merged', '-')
        west = row.get('west_only', '-')
        east = row.get('east_only', '-')
        print(f"{metric:<20} {central:>14} {merged:>14} "
              f"{west:>14} {east:>14}")


if __name__ == "__main__":
    main()
