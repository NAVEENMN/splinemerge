"""Plot results from the wave equation experiment.

Reads merge_pde_wave.csv and generates a summary bar chart comparing
reconstruction RMSE and recovered wave speed across experimental
configurations.

Usage:
    python utils/plot_wave.py [path/to/merge_pde_wave.csv]

If no path is provided, reads from merge_pde_wave.csv in the current
working directory.
"""

import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_results(csv_path):
    """Load wave experiment results from CSV.

    Returns a list of dicts with keys: experiment, n_obs, noise,
    recon_rmse, c_hat, c_err, pde_rmse.
    """
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'experiment': row['experiment'],
                'n_obs': int(row['n_obs']),
                'noise': float(row['noise']),
                'recon_rmse': float(row['recon_rmse']),
                'c_hat': float(row['c_hat']),
                'c_err': float(row['c_err']),
                'pde_rmse': float(row['pde_rmse']),
            })
    return results


def plot_results(results, output_path='wave_summary.png'):
    """Generate a summary figure for the wave experiment results.

    Parameters
    ----------
    results : list of dict
        Loaded CSV results.
    output_path : str
        Output file path for the figure.
    """
    C_TRUE = 1.0
    names = [r['experiment'] for r in results]
    recon_rmses = [r['recon_rmse'] for r in results]
    c_hats = [r['c_hat'] for r in results]
    c_errs = [r['c_err'] for r in results]

    x = np.arange(len(names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    facecolor='#1a1a2e')

    # Left panel: Reconstruction RMSE
    ax1.set_facecolor('#0d1117')
    bars1 = ax1.bar(x, recon_rmses, width, color='#4ade80', edgecolor='#30363d')
    ax1.set_xlabel('Experiment', color='white')
    ax1.set_ylabel('Reconstruction RMSE', color='white')
    ax1.set_title('Field Reconstruction Error', color='white',
                  fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax1.tick_params(colors='#666')
    for spine in ax1.spines.values():
        spine.set_color('#30363d')

    # Right panel: Recovered wave speed
    ax2.set_facecolor('#0d1117')
    bars2 = ax2.bar(x, c_hats, width, color='#c792ea', edgecolor='#30363d')
    ax2.axhline(y=C_TRUE, color='#ff6464', linewidth=1.5, linestyle='--',
                label=f'True c = {C_TRUE}')
    ax2.set_xlabel('Experiment', color='white')
    ax2.set_ylabel('Recovered c', color='white')
    ax2.set_title('Wave Speed Recovery', color='white', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax2.legend(facecolor='#1a1a2e', edgecolor='#30363d', labelcolor='#cccccc')
    ax2.tick_params(colors='#666')
    for spine in ax2.spines.values():
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
        csv_path = 'merge_pde_wave.csv'

    if not os.path.exists(csv_path):
        print(f"ERROR: Results file not found: {csv_path}")
        print("Run experiment_wave.py first to generate results.")
        sys.exit(1)

    results = load_results(csv_path)
    print(f"Loaded {len(results)} experiment configurations from {csv_path}")

    plot_results(results)

    # Print summary table
    print(f"\n{'Experiment':<20} {'Recon RMSE':>11} {'c_hat':>8} {'c_err':>8}")
    print(f"{'-'*50}")
    for r in results:
        print(f"{r['experiment']:<20} {r['recon_rmse']:>11.6f} "
              f"{r['c_hat']:>8.4f} {r['c_err']:>8.4f}")


if __name__ == "__main__":
    main()
