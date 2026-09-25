"""
Synthetic PDE Experiment: Pure Diffusion Equation

  u_t = D * (u_xx + u_yy)

Known ground truth: D = 0.05 (diffusion coefficient).
Periodic BCs on [0, 2pi]^2 x [0, T_max].

Exact spectral solution: each Fourier mode decays as
  u_hat(k, t) = u_hat(k, 0) * exp(-D * |k|^2 * t).

The experiment fits a 3D tensor-product B-spline field from scattered
observations, then recovers the diffusion coefficient from finite-difference
derivatives. Distributed merge via Gram statistics demonstrates that
physics can be recovered from spatially partitioned data holders.
"""

import torch
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from inkan.basis import bspline_basis_eager


# ============================================================================
# Exact diffusion solver (spectral)
# ============================================================================

def solve_diffusion_spectral(u0_grid, D, Lx, Ly, times):
    """Exact solution for u_t = D * (u_xx + u_yy).

    Periodic BCs on [0, Lx] x [0, Ly]. No discretization error.
    Each Fourier mode decays as u_hat(k, t) = u_hat(k, 0) * exp(-D |k|^2 t).
    Returns [Nt, Nx, Ny].
    """
    Nx, Ny = u0_grid.shape
    dx, dy = Lx / Nx, Ly / Ny

    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    # Decay rate for each mode: -D * (kx^2 + ky^2)
    decay = -D * (KX**2 + KY**2)

    u0_hat = np.fft.fft2(u0_grid)
    result = np.zeros((len(times), Nx, Ny))
    for i, t in enumerate(times):
        result[i] = np.real(np.fft.ifft2(u0_hat * np.exp(decay * t)))

    return result


# ============================================================================
# 3D tensor-product B-spline features
# ============================================================================

def make_basis_params(grid_size, grid_range):
    """Create grid_starts and inv_h for 1D B-spline basis."""
    n_bases = grid_size + 3  # cubic
    h = (grid_range[1] - grid_range[0]) / grid_size
    inv_h = 1.0 / h
    grid_starts = torch.arange(n_bases, dtype=torch.float64) * h + grid_range[0] - 3 * h
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
def build_3d_features(x, y, t, bp_x, bp_y, bp_t, batch_size=2000):
    """Build 3D tensor-product feature matrix.

    Phi[n, i*Ky*Kt + j*Kt + l] = B_i(x_n) * B_j(y_n) * T_l(t_n)
    Returns [N, Kx*Ky*Kt].
    """
    gs_x, ih_x, Kx = bp_x
    gs_y, ih_y, Ky = bp_y
    gs_t, ih_t, Kt = bp_t
    N = len(x)
    P = Kx * Ky * Kt
    parts = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        bx = eval_1d_basis(x[start:end], gs_x, ih_x)  # [B, Kx]
        by = eval_1d_basis(y[start:end], gs_y, ih_y)  # [B, Ky]
        bt = eval_1d_basis(t[start:end], gs_t, ih_t)  # [B, Kt]
        # Outer product: [B, Kx, Ky, Kt] -> [B, P]
        phi = (bx[:, :, None, None] * by[:, None, :, None] *
               bt[:, None, None, :]).reshape(-1, P)
        parts.append(phi)

    return torch.cat(parts, dim=0)


# ============================================================================
# Field fitting and evaluation
# ============================================================================

@torch.no_grad()
def fit_field(Phi, u_obs, reg=1e-4):
    """Solve C = (Phi^T Phi + reg I)^{-1} Phi^T u."""
    G = Phi.T @ Phi
    h = Phi.T @ u_obs.double()
    P = G.shape[0]
    C = torch.linalg.solve(G + reg * torch.eye(P, dtype=torch.float64), h)
    if not torch.isfinite(C).all():
        raise ValueError(f"fit_field produced non-finite coefficients")
    return C, G


@torch.no_grad()
def eval_field_on_grid(C, x_grid, y_grid, t_grid, bp_x, bp_y, bp_t):
    """Evaluate fitted field on a regular 3D grid.

    Returns u_hat[t_idx, x_idx, y_idx].
    """
    Nt, Nx, Ny = len(t_grid), len(x_grid), len(y_grid)
    u_hat = np.zeros((Nt, Nx, Ny))

    for ti in range(Nt):
        xx, yy = torch.meshgrid(x_grid, y_grid, indexing='ij')
        xx_flat = xx.reshape(-1, 1)
        yy_flat = yy.reshape(-1, 1)
        tt_flat = torch.full_like(xx_flat, t_grid[ti].item())

        Phi = build_3d_features(xx_flat, yy_flat, tt_flat, bp_x, bp_y, bp_t)
        vals = (Phi @ C).numpy().reshape(Nx, Ny)
        if not np.isfinite(vals).all():
            raise ValueError(f"eval_field_on_grid produced non-finite values at t={t_grid[ti]}")
        u_hat[ti] = vals

    return u_hat


# ============================================================================
# Derivative computation (finite differences on evaluation grid)
# ============================================================================

def compute_diffusion_derivatives(u_grid, dx, dy, dt):
    """Central finite differences for u_t and Laplacian.

    Returns derivatives on interior points: [Nt-2, Nx-2, Ny-2].
    """
    # Time derivative (central)
    u_t = (u_grid[2:, 1:-1, 1:-1] - u_grid[:-2, 1:-1, 1:-1]) / (2 * dt)

    # Spatial second derivatives (Laplacian components)
    u_xx = (u_grid[1:-1, 2:, 1:-1] - 2 * u_grid[1:-1, 1:-1, 1:-1] +
            u_grid[1:-1, :-2, 1:-1]) / dx**2
    u_yy = (u_grid[1:-1, 1:-1, 2:] - 2 * u_grid[1:-1, 1:-1, 1:-1] +
            u_grid[1:-1, 1:-1, :-2]) / dy**2

    return u_t, u_xx, u_yy


# ============================================================================
# Physics recovery
# ============================================================================

def recover_diffusion_coeff(u_t, u_xx, u_yy):
    """Recover D from u_t = D * (u_xx + u_yy).

    Single parameter: D = (laplacian^T u_t) / (laplacian^T laplacian).
    """
    laplacian = (u_xx + u_yy).flatten()
    b = u_t.flatten()

    valid = np.isfinite(laplacian) & np.isfinite(b)
    if valid.sum() < 10:
        return float('nan'), float('nan')
    laplacian, b = laplacian[valid], b[valid]

    # Single-parameter ridge regression
    ATA = np.dot(laplacian, laplacian) + 1e-8
    ATb = np.dot(laplacian, b)
    D_hat = ATb / ATA
    if D_hat < 0:
        print(f"  WARNING: Negative D = {D_hat:.6f}, physically inadmissible")

    residual = b - D_hat * laplacian
    rmse = np.sqrt(np.mean(residual**2))

    return D_hat, rmse


# ============================================================================
# Gram merge
# ============================================================================

@torch.no_grad()
def gram_merge(G_a, h_a, G_b, h_b, reg=1e-4):
    """Merge two independently fitted fields via Gram statistics.

    C_M = (G_A + G_B + reg*I)^{-1} (h_A + h_B)
    where G_s = Phi_s^T Phi_s, h_s = Phi_s^T u_s.
    """
    P = G_a.shape[0]
    G = G_a + G_b + reg * torch.eye(P, dtype=torch.float64)
    h = h_a + h_b
    C = torch.linalg.solve(G, h)
    if not torch.isfinite(C).all():
        raise ValueError("gram_merge produced non-finite coefficients")
    return C


# ============================================================================
# Main experiment
# ============================================================================

def main():
    # Ground truth parameters
    D_TRUE = 0.05
    Lx, Ly = 2 * math.pi, 2 * math.pi
    T_max = 2.0

    # PDE grid (fine, for ground truth)
    Nx_pde, Ny_pde = 64, 64
    Nt_pde = 50
    x_pde = np.linspace(0, Lx, Nx_pde, endpoint=False)
    y_pde = np.linspace(0, Ly, Ny_pde, endpoint=False)
    t_pde = np.linspace(0, T_max, Nt_pde)

    # Initial condition: multi-scale smooth function
    XX, YY = np.meshgrid(x_pde, y_pde, indexing='ij')
    u0 = (np.sin(XX) * np.cos(YY)
          + 0.5 * np.exp(-3 * ((XX - math.pi)**2 + (YY - math.pi)**2))
          + 0.3 * np.sin(2 * XX + YY))

    print("=" * 70)
    print("Synthetic PDE: Pure Diffusion Equation")
    print(f"  u_t = {D_TRUE} * (u_xx + u_yy)")
    print(f"  Domain: [0, 2pi]^2 x [0, {T_max}]")
    print(f"  PDE grid: {Nx_pde}x{Ny_pde}x{Nt_pde}")
    print("=" * 70)

    # Solve PDE exactly
    print("\nSolving diffusion equation (spectral, exact)...")
    u_true = solve_diffusion_spectral(u0, D_TRUE, Lx, Ly, t_pde)
    print(f"  Solution range: [{u_true.min():.3f}, {u_true.max():.3f}]")

    # Spline basis parameters
    gs_x, gs_y, gs_t = 8, 8, 8
    bp_x = make_basis_params(gs_x, (0, Lx))
    bp_y = make_basis_params(gs_y, (0, Ly))
    bp_t = make_basis_params(gs_t, (0, T_max))
    Kx, Ky, Kt = bp_x[2], bp_y[2], bp_t[2]
    P = Kx * Ky * Kt
    print(f"\n  Spline basis: Kx={Kx}, Ky={Ky}, Kt={Kt}, total features={P}")

    # Evaluation grid (avoid boundaries for finite differences)
    Nx_eval, Ny_eval, Nt_eval = 40, 40, 30
    x_eval = torch.linspace(0.3, Lx - 0.3, Nx_eval, dtype=torch.float64)
    y_eval = torch.linspace(0.3, Ly - 0.3, Ny_eval, dtype=torch.float64)
    t_eval = torch.linspace(0.2, T_max - 0.2, Nt_eval, dtype=torch.float64)
    dx_eval = (x_eval[1] - x_eval[0]).item()
    dy_eval = (y_eval[1] - y_eval[0]).item()
    dt_eval = (t_eval[1] - t_eval[0]).item()

    # Ground truth on evaluation grid via interpolation from PDE grid
    def eval_exact_on_grid(x_grid, y_grid, t_val):
        """Interpolate exact spectral solution onto the evaluation grid."""
        Nx_g, Ny_g = len(x_grid), len(y_grid)
        XX_g, YY_g = np.meshgrid(x_grid, y_grid, indexing='ij')
        u_sol = solve_diffusion_spectral(u0, D_TRUE, Lx, Ly, [t_val])[0]
        # Extend grid for periodic interpolation
        x_ext = np.append(x_pde, Lx)
        u_ext = np.concatenate([u_sol, u_sol[0:1, :]], axis=0)
        y_ext = np.append(y_pde, Ly)
        u_ext = np.concatenate([u_ext, u_ext[:, 0:1]], axis=1)
        interp = RegularGridInterpolator((x_ext, y_ext), u_ext,
                                          method='cubic',
                                          bounds_error=False,
                                          fill_value=None)
        xx_w = XX_g.flatten() % Lx
        yy_w = YY_g.flatten() % Ly
        pts = np.stack([xx_w, yy_w], axis=1)
        return interp(pts).reshape(Nx_g, Ny_g)

    print("\n  Computing ground truth on evaluation grid...")
    u_true_eval = np.zeros((Nt_eval, Nx_eval, Ny_eval))
    for ti in range(Nt_eval):
        u_true_eval[ti] = eval_exact_on_grid(
            x_eval.numpy(), y_eval.numpy(), t_eval[ti].item())

    # Ground truth derivatives and sanity check
    u_t_true, u_xx_true, u_yy_true = compute_diffusion_derivatives(
        u_true_eval, dx_eval, dy_eval, dt_eval)

    D_gt, rmse_gt = recover_diffusion_coeff(u_t_true, u_xx_true, u_yy_true)
    print(f"\n  Ground truth physics recovery (sanity check):")
    print(f"    D:    {D_gt:.6f} (true={D_TRUE})")
    print(f"    RMSE: {rmse_gt:.6e}")

    # Helper to get observation values from exact solution
    def get_obs(obs_x, obs_y, obs_t):
        """Sample the exact diffusion solution at scattered observation points."""
        t_list = obs_t[:, 0].tolist()
        u_sols = solve_diffusion_spectral(u0, D_TRUE, Lx, Ly, t_list)
        n = len(obs_x)
        vals = np.zeros(n)
        for i in range(n):
            u_sol = u_sols[i]
            # Extend grid for periodic interpolation
            x_ext = np.append(x_pde, Lx)
            u_ext = np.concatenate([u_sol, u_sol[0:1, :]], axis=0)
            y_ext = np.append(y_pde, Ly)
            u_ext = np.concatenate([u_ext, u_ext[:, 0:1]], axis=1)
            interp = RegularGridInterpolator(
                (x_ext, y_ext), u_ext, method='linear',
                bounds_error=False)
            xi = obs_x[i, 0].item() % Lx
            yi = obs_y[i, 0].item() % Ly
            vals[i] = interp([[xi, yi]])[0]
        return vals

    # ================================================================
    # Experiment loop: vary observation density and noise
    # ================================================================
    results = {}

    for exp_name, n_obs, noise_std in [
        ("Dense (10K)", 10000, 0.0),
        ("Moderate (3K)", 3000, 0.0),
        ("Sparse (1K)", 1000, 0.0),
        ("Dense+noise", 10000, 0.02),
        ("Sparse+noise", 1000, 0.02),
    ]:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name} (n={n_obs}, noise={noise_std})")
        print(f"{'='*60}")

        # Sample random observation points
        torch.manual_seed(42)
        obs_x = torch.rand(n_obs, 1, dtype=torch.float64) * Lx
        obs_y = torch.rand(n_obs, 1, dtype=torch.float64) * Ly
        obs_t = torch.rand(n_obs, 1, dtype=torch.float64) * T_max

        # Get true values at observation points
        obs_u = get_obs(obs_x, obs_y, obs_t)
        if noise_std > 0:
            obs_u += np.random.default_rng(42).normal(0, noise_std, n_obs)
        obs_u_tensor = torch.tensor(obs_u, dtype=torch.float64)

        # Build features and fit
        print("  Building features...")
        Phi = build_3d_features(obs_x, obs_y, obs_t, bp_x, bp_y, bp_t)
        print(f"  Phi shape: {Phi.shape}")

        C, G = fit_field(Phi, obs_u_tensor, reg=1e-4)

        # Evaluate on grid
        print("  Evaluating on grid...")
        u_hat = eval_field_on_grid(C, x_eval, y_eval, t_eval,
                                    bp_x, bp_y, bp_t)

        # Reconstruction RMSE
        recon_rmse = np.sqrt(np.mean((u_hat - u_true_eval)**2))

        # Derivative recovery
        u_t_hat, u_xx_hat, u_yy_hat = compute_diffusion_derivatives(
            u_hat, dx_eval, dy_eval, dt_eval)

        deriv_rmse_ut = np.sqrt(np.mean((u_t_hat - u_t_true)**2))
        deriv_rmse_lap = np.sqrt(np.mean(
            (u_xx_hat + u_yy_hat - u_xx_true - u_yy_true)**2))

        # Physics recovery
        D_hat, pde_rmse = recover_diffusion_coeff(u_t_hat, u_xx_hat, u_yy_hat)
        D_err = abs(D_hat - D_TRUE)

        print(f"\n  Reconstruction RMSE:    {recon_rmse:.6f}")
        print(f"  Derivative u_t RMSE:    {deriv_rmse_ut:.6f}")
        print(f"  Derivative Lap u RMSE:  {deriv_rmse_lap:.6f}")
        print(f"  Physics recovery:")
        print(f"    D:   {D_hat:.6f} (true={D_TRUE}, err={D_err:.6f})")
        print(f"    PDE residual RMSE: {pde_rmse:.6e}")

        results[exp_name] = {
            'n_obs': n_obs, 'noise': noise_std,
            'recon_rmse': recon_rmse,
            'deriv_ut_rmse': deriv_rmse_ut,
            'deriv_lap_rmse': deriv_rmse_lap,
            'D_hat': D_hat, 'D_err': D_err,
            'pde_rmse': pde_rmse,
        }

    # ================================================================
    # Distributed merge experiment
    # ================================================================
    print(f"\n{'='*60}")
    print("Distributed Merge (left/right spatial split at x=pi)")
    print(f"{'='*60}")

    n_per_sensor = 5000
    torch.manual_seed(42)

    # Sensor A: x < pi (left half)
    obs_x_a = torch.rand(n_per_sensor, 1, dtype=torch.float64) * math.pi
    obs_y_a = torch.rand(n_per_sensor, 1, dtype=torch.float64) * Ly
    obs_t_a = torch.rand(n_per_sensor, 1, dtype=torch.float64) * T_max

    # Sensor B: x >= pi (right half)
    obs_x_b = torch.rand(n_per_sensor, 1, dtype=torch.float64) * math.pi + math.pi
    obs_y_b = torch.rand(n_per_sensor, 1, dtype=torch.float64) * Ly
    obs_t_b = torch.rand(n_per_sensor, 1, dtype=torch.float64) * T_max

    print("  Collecting sensor A observations...")
    u_obs_a = torch.tensor(get_obs(obs_x_a, obs_y_a, obs_t_a),
                           dtype=torch.float64)
    print("  Collecting sensor B observations...")
    u_obs_b = torch.tensor(get_obs(obs_x_b, obs_y_b, obs_t_b),
                           dtype=torch.float64)

    # Centralized (all 10K points)
    obs_x_all = torch.cat([obs_x_a, obs_x_b])
    obs_y_all = torch.cat([obs_y_a, obs_y_b])
    obs_t_all = torch.cat([obs_t_a, obs_t_b])
    u_obs_all = torch.cat([u_obs_a, u_obs_b])

    print("  Building features and fitting...")
    Phi_a = build_3d_features(obs_x_a, obs_y_a, obs_t_a, bp_x, bp_y, bp_t)
    Phi_b = build_3d_features(obs_x_b, obs_y_b, obs_t_b, bp_x, bp_y, bp_t)
    Phi_all = build_3d_features(obs_x_all, obs_y_all, obs_t_all,
                                 bp_x, bp_y, bp_t)

    # Independent fits
    C_a, G_a = fit_field(Phi_a, u_obs_a)
    C_b, G_b = fit_field(Phi_b, u_obs_b)
    h_a = Phi_a.T @ u_obs_a
    h_b = Phi_b.T @ u_obs_b

    # Centralized fit
    C_central, _ = fit_field(Phi_all, u_obs_all)

    # Gram merge
    C_merged = gram_merge(G_a, h_a, G_b, h_b, reg=1e-4)

    # Evaluate all variants
    print("  Evaluating fields...")
    u_central = eval_field_on_grid(C_central, x_eval, y_eval, t_eval,
                                    bp_x, bp_y, bp_t)
    u_merged = eval_field_on_grid(C_merged, x_eval, y_eval, t_eval,
                                   bp_x, bp_y, bp_t)
    u_left = eval_field_on_grid(C_a, x_eval, y_eval, t_eval,
                                 bp_x, bp_y, bp_t)
    u_right = eval_field_on_grid(C_b, x_eval, y_eval, t_eval,
                                  bp_x, bp_y, bp_t)

    # Reconstruction comparison
    for label, u_hat in [("Centralized", u_central), ("Merged", u_merged),
                          ("Left only", u_left), ("Right only", u_right)]:
        rmse = np.sqrt(np.mean((u_hat - u_true_eval)**2))
        mid_idx = Nx_eval // 2
        rmse_left = np.sqrt(np.mean((u_hat[:, :mid_idx, :] -
                                      u_true_eval[:, :mid_idx, :])**2))
        rmse_right = np.sqrt(np.mean((u_hat[:, mid_idx:, :] -
                                       u_true_eval[:, mid_idx:, :])**2))
        print(f"  {label:<16} RMSE: {rmse:.6f}  "
              f"(left={rmse_left:.6f}, right={rmse_right:.6f})")

    # Physics recovery from merged and centralized fields
    print("\n  Physics recovery comparison:")
    for label, u_hat in [("Centralized", u_central), ("Merged", u_merged)]:
        u_t_h, u_xx_h, u_yy_h = compute_diffusion_derivatives(
            u_hat, dx_eval, dy_eval, dt_eval)
        D_h, pde_r = recover_diffusion_coeff(u_t_h, u_xx_h, u_yy_h)
        print(f"  {label:<16} D={D_h:.6f} (err={abs(D_h - D_TRUE):.6f}, "
              f"residual={pde_r:.4e})")

    # ================================================================
    # Summary table
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"\n  True diffusion coefficient: D = {D_TRUE}")
    print(f"\n  {'Experiment':<20} {'Recon RMSE':>11} {'D_hat':>9} {'D_err':>11}")
    print(f"  {'-'*55}")

    for name, r in results.items():
        print(f"  {name:<20} {r['recon_rmse']:>11.6f} {r['D_hat']:>9.6f} "
              f"{r['D_err']:>11.6f}")

    # ================================================================
    # Visualization
    # ================================================================
    print("\nGenerating plots...")

    # Reconstruct merged field (already computed above)
    u_merged_grid = u_merged

    fig, axes = plt.subplots(3, 4, figsize=(18, 12), facecolor='#1a1a2e')
    fig.suptitle(f'Pure Diffusion: u_t = {D_TRUE} (u_xx + u_yy)\n'
                 f'Distributed Merge: 2 data holders, Gram statistics',
                 fontsize=14, fontweight='bold', color='white', y=0.98)

    t_indices = [0, Nt_eval // 2, Nt_eval - 1]
    plot_data = [("True", u_true_eval), ("Centralized", u_central),
                 ("Merged", u_merged_grid), ("|Error|", None)]

    for row, ti in enumerate(t_indices):
        for col, (label, data) in enumerate(plot_data):
            ax = axes[row, col]
            ax.set_facecolor('#0d1117')

            if label == "|Error|":
                err = np.abs(u_merged_grid[ti] - u_true_eval[ti])
                im = ax.imshow(err.T, origin='lower', cmap='hot',
                               extent=[x_eval[0], x_eval[-1],
                                       y_eval[0], y_eval[-1]],
                               aspect='auto')
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                im = ax.imshow(data[ti].T, origin='lower', cmap='viridis',
                               extent=[x_eval[0], x_eval[-1],
                                       y_eval[0], y_eval[-1]],
                               aspect='auto',
                               vmin=u_true_eval.min(), vmax=u_true_eval.max())
                if col == 0:
                    plt.colorbar(im, ax=ax, fraction=0.046)

            if col == 2:  # Merged: show split boundary
                ax.axvline(x=math.pi, color='white', linewidth=0.5,
                           linestyle='--', alpha=0.5)

            t_val = t_eval[ti].item()
            ax.set_title(f'{label} (t={t_val:.1f})', fontsize=10,
                         fontweight='bold', color='white')
            ax.tick_params(colors='#666', labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#30363d')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('merge_pde_diffusion.png', dpi=150, facecolor='#1a1a2e',
                bbox_inches='tight')
    print("  Saved: merge_pde_diffusion.png")
    plt.close()

    # Save CSV
    with open('merge_pde_diffusion.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['experiment', 'n_obs', 'noise', 'recon_rmse',
                     'D_hat', 'D_err', 'pde_rmse'])
        for name, r in results.items():
            w.writerow([name, r['n_obs'], r['noise'],
                        f"{r['recon_rmse']:.8f}",
                        f"{r['D_hat']:.6f}", f"{r['D_err']:.6f}",
                        f"{r['pde_rmse']:.8e}"])
    print("  Saved: merge_pde_diffusion.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
