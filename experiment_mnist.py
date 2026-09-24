"""
MNIST Class-Average Merge: Gram merge across ALL images per digit class.

For each digit class (0-9), ALL training images contribute observations
to a 2D spline field on [0,1]^2. Three data holders each observe a
different horizontal strip of every image:
  - Holder 1: rows 0-9   (top)
  - Holder 2: rows 10-18 (middle, disjoint)
  - Holder 3: rows 19-27 (bottom)

Each holder accumulates Gram statistics across all images in the class.
Since all 28x28 images share the same coordinate grid, the design matrix
Phi_strip is identical for every image -- only pixel values change:

    G_s = N_images * Phi_strip^T Phi_strip
    h_s = Phi_strip^T (sum_images y_s,i)

The merged field represents a smooth class-average spatial structure:
a "prototype" digit reconstructed from distributed partial observations.

This differs from image stitching because thousands of different images
contribute observations, each holder sees only a strip of each image,
and the result is a smooth spline fit -- not pixel averaging.

Basis: 2D tensor-product cubic B-splines, grid_size=6 per axis,
giving K_x = K_y = 9 and P = 81 total coefficients.

Dependencies: torch, numpy, matplotlib, torchvision, inkan
"""

import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets

from inkan.basis import bspline_basis_eager


# ============================================================================
# B-spline basis utilities
# ============================================================================

def make_basis_params(grid_size, grid_range):
    """Create grid_starts, inv_h, n_bases for 1D cubic B-spline basis."""
    n_bases = grid_size + 3  # cubic spline order = 3
    h = (grid_range[1] - grid_range[0]) / grid_size
    inv_h = 1.0 / h
    grid_starts = (torch.arange(n_bases, dtype=torch.float64) * h
                   + grid_range[0] - 3 * h)
    return grid_starts, inv_h, n_bases


def eval_1d_basis(x, grid_starts, inv_h):
    """Evaluate 1D B-spline basis. x: [N,1] -> [N, K] in float64."""
    n_bases = len(grid_starts)
    try:
        return bspline_basis_eager(x, grid_starts, inv_h, n_bases).squeeze(1).double()
    except TypeError:
        try:
            return bspline_basis_eager(x, grid_starts, inv_h, bounded=False).squeeze(1).double()
        except TypeError:
            return bspline_basis_eager(x, grid_starts, inv_h).squeeze(1).double()


@torch.no_grad()
def build_2d_features(coords_x, coords_y, bp_x, bp_y):
    """Build 2D tensor-product feature matrix.

    Phi[n, i*Ky + j] = B_i(x_n) * B_j(y_n)
    Returns [N, Kx*Ky] in float64.
    """
    gs_x, ih_x, Kx = bp_x
    gs_y, ih_y, Ky = bp_y

    bx = eval_1d_basis(coords_x, gs_x, ih_x)  # [N, Kx]
    by = eval_1d_basis(coords_y, gs_y, ih_y)  # [N, Ky]

    # Outer product: [N, Kx, Ky] -> [N, P]
    return (bx.unsqueeze(2) * by.unsqueeze(1)).reshape(-1, Kx * Ky)


# ============================================================================
# Coordinate utilities
# ============================================================================

def strip_coordinates(row_start, row_end):
    """Return normalized (x, y) coordinates for a horizontal strip.

    Pixel coordinates are normalized to [0, 1].
    Row 0 -> y near 1 (top), row 27 -> y near 0 (bottom).

    Returns:
        coords_x: [N, 1] float64 tensor
        coords_y: [N, 1] float64 tensor
        row_indices: flat array of row indices into 28x28 image
        col_indices: flat array of column indices into 28x28 image
    """
    rows = np.arange(row_start, row_end + 1)
    cols = np.arange(28)
    rr, cc = np.meshgrid(rows, cols, indexing='ij')
    rr_flat = rr.flatten()
    cc_flat = cc.flatten()

    x_norm = cc_flat / 27.0
    y_norm = 1.0 - rr_flat / 27.0

    coords_x = torch.tensor(x_norm, dtype=torch.float64).unsqueeze(1)
    coords_y = torch.tensor(y_norm, dtype=torch.float64).unsqueeze(1)

    return coords_x, coords_y, rr_flat, cc_flat


def extract_strip_pixels(image_np, row_indices, col_indices):
    """Extract pixel values from a 28x28 uint8 image at given indices.

    Returns [N] float64 numpy array normalized to [0, 1].
    """
    return image_np[row_indices, col_indices].astype(np.float64) / 255.0


# ============================================================================
# Fitting and merging
# ============================================================================

@torch.no_grad()
def gram_merge(G_list, h_list, reg=1e-4):
    """Merge Gram statistics from multiple holders.

    c_merged = (sum(G_s) + reg I)^{-1} sum(h_s)
    """
    G_sum = sum(G_list)
    h_sum = sum(h_list)
    P = G_sum.shape[0]
    c = torch.linalg.solve(G_sum + reg * torch.eye(P, dtype=torch.float64),
                           h_sum)
    return c


# ============================================================================
# Reconstruction
# ============================================================================

def reconstruct_image(c, bp_x, bp_y):
    """Evaluate the spline field on the full 28x28 pixel grid.

    Returns a 28x28 numpy array.
    """
    cols = np.arange(28)
    rows = np.arange(28)
    rr, cc = np.meshgrid(rows, cols, indexing='ij')

    x_norm = cc.flatten() / 27.0
    y_norm = 1.0 - rr.flatten() / 27.0

    coords_x = torch.tensor(x_norm, dtype=torch.float64).unsqueeze(1)
    coords_y = torch.tensor(y_norm, dtype=torch.float64).unsqueeze(1)

    Phi = build_2d_features(coords_x, coords_y, bp_x, bp_y)
    recon = (Phi @ c).numpy().reshape(28, 28)
    return recon


# ============================================================================
# Main experiment
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")

    # Strip definitions (row ranges, inclusive)
    strips = [
        (0, 9),    # Holder 1: rows 0-9 (top)
        (10, 18),  # Holder 2: rows 10-18 (middle, disjoint)
        (19, 27),  # Holder 3: rows 19-27 (bottom)
    ]

    # Basis parameters: grid_size=6 per axis -> K=9 per axis, P=81
    grid_size = 6
    bp_x = make_basis_params(grid_size, (0.0, 1.0))
    bp_y = make_basis_params(grid_size, (0.0, 1.0))
    Kx, Ky = bp_x[2], bp_y[2]
    P = Kx * Ky
    reg = 1e-4

    print("=" * 70)
    print("MNIST Class-Average Merge: Gram Merge Across ALL Images Per Class")
    print("=" * 70)
    print(f"  Basis: 2D tensor-product cubic B-splines")
    print(f"  Grid size: {grid_size} per axis")
    print(f"  Basis functions: Kx={Kx}, Ky={Ky}, P={P}")
    print(f"  Regularization: lambda={reg}")
    print(f"  Strips: {strips}")
    print()

    # Load MNIST
    print("Loading MNIST dataset...")
    mnist = datasets.MNIST(root=data_dir, train=True, download=True)
    images = mnist.data.numpy()   # [60000, 28, 28] uint8
    labels = mnist.targets.numpy()  # [60000]

    # ------------------------------------------------------------------
    # Precompute strip coordinates and design matrices (shared by all images)
    # ------------------------------------------------------------------
    print("Precomputing strip design matrices...")
    strip_info = []
    for s_idx, (row_start, row_end) in enumerate(strips):
        cx, cy, ri, ci = strip_coordinates(row_start, row_end)
        Phi_strip = build_2d_features(cx, cy, bp_x, bp_y)
        GtG_strip = Phi_strip.T @ Phi_strip  # [P, P]
        n_pixels = Phi_strip.shape[0]
        strip_info.append({
            'Phi': Phi_strip,
            'GtG': GtG_strip,
            'row_indices': ri,
            'col_indices': ci,
            'n_pixels': n_pixels,
        })
        print(f"  Strip {s_idx + 1} (rows {row_start}-{row_end}): "
              f"{n_pixels} pixels per image")

    # Full-image coordinates and design matrix
    cx_full, cy_full, ri_full, ci_full = strip_coordinates(0, 27)
    Phi_full = build_2d_features(cx_full, cy_full, bp_x, bp_y)
    GtG_full = Phi_full.T @ Phi_full

    print()

    # ------------------------------------------------------------------
    # Process each digit class using ALL training images
    # ------------------------------------------------------------------
    results = []

    for digit in range(10):
        digit_mask = labels == digit
        digit_images = images[digit_mask]  # [N_d, 28, 28] uint8
        N_d = digit_images.shape[0]

        print(f"--- Digit {digit}: {N_d} training images ---")

        # Compute mean image (simple pixel average, for reference)
        mean_image = digit_images.astype(np.float64).mean(axis=0) / 255.0

        # ----------------------------------------------------------
        # Efficient accumulation: sum pixel values across all images
        # ----------------------------------------------------------
        # For each strip, sum y_s,i across all images -> [N_pixels]
        # G_s = N_d * Phi_strip^T Phi_strip
        # h_s = Phi_strip^T (sum_images y_s,i)

        G_list = []
        h_list = []
        strip_mean_views = []

        for s_idx, sinfo in enumerate(strip_info):
            Phi_s = sinfo['Phi']
            ri = sinfo['row_indices']
            ci = sinfo['col_indices']
            n_pix = sinfo['n_pixels']

            # Sum pixel values across all images for this strip
            pixel_sum = np.zeros(n_pix, dtype=np.float64)
            for img in digit_images:
                pixel_sum += extract_strip_pixels(img, ri, ci)

            pixel_sum_t = torch.tensor(pixel_sum, dtype=torch.float64)

            G_s = N_d * sinfo['GtG']         # [P, P]
            h_s = Phi_s.T @ pixel_sum_t       # [P]

            G_list.append(G_s)
            h_list.append(h_s)

            total_obs = N_d * n_pix
            print(f"  Holder {s_idx + 1}: {n_pix} pixels/image x {N_d} images "
                  f"= {total_obs:,} total observations")

            # Mean strip view (average pixel values in the strip region)
            strip_mean = np.zeros((28, 28), dtype=np.float64)
            rs, re = strips[s_idx]
            strip_mean[rs:re + 1, :] = mean_image[rs:re + 1, :]
            strip_mean_views.append(strip_mean)

        # Merged reconstruction
        c_merged = gram_merge(G_list, h_list, reg=reg)

        # Centralized fit: uses all pixels from all images
        pixel_sum_full = np.zeros(Phi_full.shape[0], dtype=np.float64)
        for img in digit_images:
            pixel_sum_full += extract_strip_pixels(img, ri_full, ci_full)

        pixel_sum_full_t = torch.tensor(pixel_sum_full, dtype=torch.float64)
        G_central = N_d * GtG_full
        h_central = Phi_full.T @ pixel_sum_full_t
        c_central = torch.linalg.solve(
            G_central + reg * torch.eye(P, dtype=torch.float64), h_central)

        # Reconstruct 28x28 images
        recon_merged = reconstruct_image(c_merged, bp_x, bp_y)
        recon_central = reconstruct_image(c_central, bp_x, bp_y)

        # Metrics: compare merged vs centralized (not vs original,
        # since there is no single original for the class average)
        delta_max = np.max(np.abs(recon_merged - recon_central))
        rmse_vs_central = np.sqrt(np.mean((recon_merged - recon_central) ** 2))
        rmse_merged_vs_mean = np.sqrt(np.mean((recon_merged - mean_image) ** 2))
        rmse_central_vs_mean = np.sqrt(
            np.mean((recon_central - mean_image) ** 2))

        print(f"  RMSE merged vs centralized:   {rmse_vs_central:.2e}")
        print(f"  delta_max (merged vs central): {delta_max:.2e}")
        print(f"  RMSE merged vs pixel mean:     {rmse_merged_vs_mean:.6f}")
        print(f"  RMSE centralized vs pixel mean: {rmse_central_vs_mean:.6f}")

        results.append({
            'digit': digit,
            'n_images': N_d,
            'rmse_vs_central': rmse_vs_central,
            'delta_max': delta_max,
            'rmse_merged_vs_mean': rmse_merged_vs_mean,
            'rmse_central_vs_mean': rmse_central_vs_mean,
            'mean_image': mean_image,
            'strip_mean_views': strip_mean_views,
            'recon_merged': recon_merged,
            'recon_central': recon_central,
        })

    # ================================================================
    # Summary statistics
    # ================================================================
    avg_rmse_vs_central = np.mean([r['rmse_vs_central'] for r in results])
    avg_delta_max = np.mean([r['delta_max'] for r in results])
    avg_rmse_merged_vs_mean = np.mean(
        [r['rmse_merged_vs_mean'] for r in results])
    avg_rmse_central_vs_mean = np.mean(
        [r['rmse_central_vs_mean'] for r in results])

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Digit':<8} {'N_images':>8} {'RMSE m-vs-c':>14} "
          f"{'delta_max':>14} {'RMSE m-vs-mean':>14} {'RMSE c-vs-mean':>14}")
    print(f"  {'-' * 76}")
    for r in results:
        print(f"  {r['digit']:<8} {r['n_images']:>8} "
              f"{r['rmse_vs_central']:>14.2e} {r['delta_max']:>14.2e} "
              f"{r['rmse_merged_vs_mean']:>14.6f} "
              f"{r['rmse_central_vs_mean']:>14.6f}")
    print(f"  {'-' * 76}")
    print(f"  {'Average':<8} {'':>8} {avg_rmse_vs_central:>14.2e} "
          f"{avg_delta_max:>14.2e} {avg_rmse_merged_vs_mean:>14.6f} "
          f"{avg_rmse_central_vs_mean:>14.6f}")

    # ================================================================
    # Save CSV
    # ================================================================
    csv_path = os.path.join(script_dir, "mnist_image_merge.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['digit', 'n_images', 'rmse_merged_vs_central',
                         'delta_max', 'rmse_merged_vs_mean',
                         'rmse_central_vs_mean'])
        for r in results:
            writer.writerow([
                r['digit'], r['n_images'],
                f"{r['rmse_vs_central']:.8e}",
                f"{r['delta_max']:.8e}",
                f"{r['rmse_merged_vs_mean']:.8f}",
                f"{r['rmse_central_vs_mean']:.8f}",
            ])
        writer.writerow([
            'average', '',
            f"{avg_rmse_vs_central:.8e}",
            f"{avg_delta_max:.8e}",
            f"{avg_rmse_merged_vs_mean:.8f}",
            f"{avg_rmse_central_vs_mean:.8f}",
        ])
    print(f"\nSaved: {csv_path}")

    # ================================================================
    # Visualization
    # ================================================================
    print("\nGenerating visualization...")

    n_digits = 10
    n_rows = 7  # mean image, 3 strip means, merged, centralized, error
    fig, axes = plt.subplots(n_rows, n_digits, figsize=(20, 14),
                             facecolor='white')

    row_labels = [
        "Pixel Mean\n(reference)",
        "Holder 1 Mean\n(rows 0-9)",
        "Holder 2 Mean\n(rows 10-18)",
        "Holder 3 Mean\n(rows 19-27)",
        "Merged Spline\nReconstruction",
        "Centralized Spline\nReconstruction",
        "|Merged - Central|",
    ]

    for col, r in enumerate(results):
        # Row 0: Mean image (pixel average across all images of this class)
        axes[0, col].imshow(r['mean_image'], cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(
            f"Digit {r['digit']}\n({r['n_images']} images)",
            fontsize=8, fontweight='bold')

        # Rows 1-3: Mean of each holder's strip
        for s_idx in range(3):
            axes[s_idx + 1, col].imshow(r['strip_mean_views'][s_idx],
                                        cmap='gray', vmin=0, vmax=1)

        # Row 4: Merged spline reconstruction
        axes[4, col].imshow(r['recon_merged'], cmap='gray', vmin=0, vmax=1)

        # Row 5: Centralized spline reconstruction
        axes[5, col].imshow(r['recon_central'], cmap='gray', vmin=0, vmax=1)

        # Row 6: |Merged - Centralized| error
        err = np.abs(r['recon_merged'] - r['recon_central'])
        axes[6, col].imshow(err, cmap='hot', vmin=0,
                            vmax=max(err.max(), 1e-10))

    # Row labels on the left
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(label, fontsize=7, fontweight='bold',
                                    rotation=0, labelpad=80, va='center')

    # Clean up axes
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    # Add a colorbar for the error row
    cbar_ax = fig.add_axes([0.92, 0.04, 0.015, 0.1])
    max_err = max(np.abs(r['recon_merged'] - r['recon_central']).max()
                  for r in results)
    sm = plt.cm.ScalarMappable(cmap='hot',
                               norm=plt.Normalize(0, max(max_err, 1e-10)))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='|error|')

    fig.suptitle(
        "Gram Merge on MNIST: Class-Average Prototype from ALL Training Images\n"
        "3 Data Holders with Disjoint Horizontal Strips    "
        f"Basis: grid_size={grid_size}, P={P}    "
        f"Avg delta_max={avg_delta_max:.2e}",
        fontsize=10, fontweight='bold', y=0.99)

    plt.tight_layout(rect=[0.09, 0.0, 0.91, 0.96])
    png_path = os.path.join(script_dir, "mnist_image_merge.png")
    plt.savefig(png_path, dpi=150, facecolor='white', bbox_inches='tight')
    print(f"Saved: {png_path}")
    plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
