"""
Synthetic PDE Experiment: Heat Equation with Spatially Varying Source

  u_t = D * (u_xx + u_yy) + S(x, y)

Known ground truth: D = 0.03 (diffusion coefficient).
Source S(x, y) = 0.3 + 0.2*cos(x) + 0.1*sin(2*y)
Periodic source with clean Fourier transform on [0, 2pi]^2 x [0, T_max].

Exact Fourier solution (no time-stepping, no snapshot snapping):
  For each mode k with a_k = D*|k|^2:
    |k| > 0: u_hat_k(t) = exp(-a_k*t)*u_hat_k(0) + (1 - exp(-a_k*t))/a_k * S_hat_k
    k = 0:   u_hat_0(t) = u_hat_0(0) + t * S_hat_0

The experiment fits a 3D tensor-product B-spline field from scattered
observations, then recovers the diffusion coefficient by subtracting
the known source from u_t before regressing against the Laplacian.
Distributed merge via Gram statistics demonstrates that physics can
be recovered from spatially partitioned data holders.
"""

import torch
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from inkan.basis import bspline_basis_eager


# ============================================================================
# Source function
# ============================================================================

def source_field(x, y):
    """Spatially varying source S(x, y): periodic, clean Fourier transform.

    S(x, y) = 0.3 + 0.2*cos(x) + 0.1*sin(2*y)
    Only a few nonzero Fourier modes.
    """
    return 0.3 + 0.2 * np.cos(x) + 0.1 * np.sin(2.0 * y)


# ============================================================================
# Exact Fourier solver: u_t = D*(u_xx + u_yy) + S(x, y)
# ============================================================================

def solve_heat_source_exact(u0_grid, S_grid, D, Lx, Ly, times):
    """Exact Fourier solution for u_t = D*(u_xx + u_yy) + S(x, y).

    Periodic BCs on [0, Lx] x [0, Ly]. No time-stepping error.
    For each Fourier mode k with a_k = D*|k|^2:
      |k| > 0: u_hat_k(t) = exp(-a_k*t)*u_hat_k(0) + (1 - exp(-a_k*t))/a_k * S_hat_k
      k = 0:   u_hat_0(t) = u_hat_0(0) + t * S_hat_0

    Parameters
    ----------
    u0_grid : ndarray [Nx, Ny]  -- initial condition on PDE grid
    S_grid  : ndarray [Nx, Ny]  -- source field on PDE grid
    D       : float             -- diffusion coefficient
    Lx, Ly  : float             -- domain lengths
    times   : array-like        -- arbitrary evaluation times

    Returns [len(times), Nx, Ny].
    """
    Nx, Ny = u0_grid.shape
    dx, dy = Lx / Nx, Ly / Ny

    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    k_sq = KX**2 + KY**2       # |k|^2 for each mode
    a_k = D * k_sq              # decay rate

    u0_hat = np.fft.fft2(u0_grid)
    S_hat = np.fft.fft2(S_grid)

    result = np.zeros((len(times), Nx, Ny))
    for i, t in enumerate(times):
        # Compute u_hat(t) for all modes at once
        exp_decay = np.exp(-a_k * t)

        # Steady-state contribution: (1 - exp(-a_k*t)) / a_k * S_hat
        # Handle k=0 mode separately to avoid division by zero
        steady = np.zeros_like(S_hat)
        nonzero = k_sq > 0
        steady[nonzero] = (1.0 - exp_decay[nonzero]) / a_k[nonzero] * S_hat[nonzero]
        # k=0 mode: u_hat_0(t) = u_hat_0(0) + t * S_hat_0
        steady[~nonzero] = t * S_hat[~nonzero]

        # Homogeneous part decays, source drives steady state
        # For nonzero modes: u_hat = exp(-a_k*t)*u0_hat + steady
        # For zero mode: u_hat = u0_hat + t*S_hat (exp_decay=1 for k=0)
        u_hat_t = exp_decay * u0_hat + steady

        result[i] = np.real(np.fft.ifft2(u_hat_t))

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
        raise ValueError("fit_field produced non-finite coefficients")
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

def compute_derivatives(u_grid, dx, dy, dt):
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

def recover_diffusion_coeff(u_t, u_xx, u_yy, S_interior):
    """Recover D from u_t = D * (u_xx + u_yy) + S(x, y).

    Subtract the known source: u_t_adjusted = u_t - S.
    Then: D = (laplacian^T u_t_adjusted) / (laplacian^T laplacian).
    """
    laplacian = (u_xx + u_yy).flatten()
    u_t_adjusted = (u_t - S_interior).flatten()

    valid = np.isfinite(laplacian) & np.isfinite(u_t_adjusted)
    if valid.sum() < 10:
        return float('nan'), float('nan')
    laplacian, u_t_adjusted = laplacian[valid], u_t_adjusted[valid]

    # Single-parameter ridge regression
    ATA = np.dot(laplacian, laplacian) + 1e-8
    ATb = np.dot(laplacian, u_t_adjusted)
    D_hat = ATb / ATA
    if D_hat < 0:
        print(f"  WARNING: Negative D = {D_hat:.6f}, physically inadmissible")

    residual = u_t_adjusted - D_hat * laplacian
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
    D_TRUE = 0.03
    Lx, Ly = 2 * math.pi, 2 * math.pi
    T_max = 2.0

    # PDE grid (for exact Fourier solution and interpolation)
    Nx_pde, Ny_pde = 64, 64
    x_pde = np.linspace(0, Lx, Nx_pde, endpoint=False)
    y_pde = np.linspace(0, Ly, Ny_pde, endpoint=False)

    # Initial condition
    XX, YY = np.meshgrid(x_pde, y_pde, indexing='ij')
    u0 = 0.5 * np.sin(XX) * np.cos(YY) + 0.5

    # Source field on PDE grid
    S_grid = source_field(XX, YY)

    print("=" * 70)
    print("Synthetic PDE: Heat Equation with Spatially Varying Source")
    print(f"  u_t = {D_TRUE} * (u_xx + u_yy) + S(x, y)")
    print(f"  S(x, y) = 0.3 + 0.2*cos(x) + 0.1*sin(2*y)")
    print(f"  Domain: [0, 2pi]^2 x [0, {T_max}]")
    print(f"  PDE grid: {Nx_pde}x{Ny_pde} (exact Fourier, no time-stepping)")
    print("=" * 70)

    # Verify exact solution at a few times
    test_times = [0.0, 1.0, 2.0]
    u_test = solve_heat_source_exact(u0, S_grid, D_TRUE, Lx, Ly, test_times)
    print(f"\n  Exact solution check:")
    for i, t in enumerate(test_times):
        print(f"    t={t:.1f}: range [{u_test[i].min():.4f}, {u_test[i].max():.4f}]")

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

    # Source on the interior evaluation grid (for physics recovery)
    # Interior points after central-difference trimming: [Nt-2, Nx-2, Ny-2]
    x_interior = x_eval[1:-1].numpy()
    y_interior = y_eval[1:-1].numpy()
    XX_int, YY_int = np.meshgrid(x_interior, y_interior, indexing='ij')
    S_interior_2d = source_field(XX_int, YY_int)
    # Broadcast over time dimension: same source at every time step
    Nt_int = Nt_eval - 2
    S_interior = np.broadcast_to(S_interior_2d[None, :, :],
                                 (Nt_int, len(x_interior), len(y_interior))).copy()

    # Ground truth on evaluation grid via exact Fourier + periodic interpolation
    def eval_exact_on_grid(x_grid, y_grid, t_val):
        """Evaluate exact Fourier solution on an arbitrary spatial grid."""
        Nx_g, Ny_g = len(x_grid), len(y_grid)
        XX_g, YY_g = np.meshgrid(x_grid, y_grid, indexing='ij')
        u_sol = solve_heat_source_exact(u0, S_grid, D_TRUE, Lx, Ly, [t_val])[0]
        # Extend grid for periodic cubic interpolation
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
    u_t_true, u_xx_true, u_yy_true = compute_derivatives(
        u_true_eval, dx_eval, dy_eval, dt_eval)

    D_gt, rmse_gt = recover_diffusion_coeff(u_t_true, u_xx_true, u_yy_true,
                                            S_interior)
    print(f"\n  Ground truth physics recovery (sanity check):")
    print(f"    D:    {D_gt:.6f} (true={D_TRUE})")
    print(f"    RMSE: {rmse_gt:.6e}")

    # Helper to get observation values from exact Fourier solution
    def get_obs(obs_x, obs_y, obs_t):
        """Sample the exact Fourier solution at scattered (x, y, t) points.

        Uses exact solution at each observation time (no snapshot snapping),
        then periodic cubic interpolation for spatial coordinates.
        """
        t_list = obs_t[:, 0].tolist()
        # Compute exact Fourier solution at all unique observation times
        u_sols = solve_heat_source_exact(u0, S_grid, D_TRUE, Lx, Ly, t_list)
        n = len(obs_x)
        vals = np.zeros(n)
        for i in range(n):
            u_sol = u_sols[i]
            # Extend grid for periodic cubic interpolation
            x_ext = np.append(x_pde, Lx)
            u_ext = np.concatenate([u_sol, u_sol[0:1, :]], axis=0)
            y_ext = np.append(y_pde, Ly)
            u_ext = np.concatenate([u_ext, u_ext[:, 0:1]], axis=1)
            interp = RegularGridInterpolator(
                (x_ext, y_ext), u_ext, method='cubic',
                bounds_error=False, fill_value=None)
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
    ]:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name} (n={n_obs}, noise={noise_std})")
        print(f"{'='*60}")

        # Sample random observation points
        torch.manual_seed(42)
        obs_x = torch.rand(n_obs, 1, dtype=torch.float64) * Lx
        obs_y = torch.rand(n_obs, 1, dtype=torch.float64) * Ly
        obs_t = torch.rand(n_obs, 1, dtype=torch.float64) * T_max

        # Get true values at observation points (exact, no snapping)
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
        u_t_hat, u_xx_hat, u_yy_hat = compute_derivatives(
            u_hat, dx_eval, dy_eval, dt_eval)

        deriv_rmse_ut = np.sqrt(np.mean((u_t_hat - u_t_true)**2))
        deriv_rmse_lap = np.sqrt(np.mean(
            (u_xx_hat + u_yy_hat - u_xx_true - u_yy_true)**2))

        # Physics recovery (subtract source, then regress)
        D_hat, pde_rmse = recover_diffusion_coeff(u_t_hat, u_xx_hat, u_yy_hat,
                                                  S_interior)
        D_err = abs(D_hat - D_TRUE)
        D_err_pct = 100.0 * D_err / D_TRUE

        print(f"\n  Reconstruction RMSE:    {recon_rmse:.6f}")
        print(f"  Derivative u_t RMSE:    {deriv_rmse_ut:.6f}")
        print(f"  Derivative Lap u RMSE:  {deriv_rmse_lap:.6f}")
        print(f"  Physics recovery:")
        print(f"    D:   {D_hat:.6f} (true={D_TRUE}, err={D_err:.6f}, {D_err_pct:.2f}%)")
        print(f"    PDE residual RMSE: {pde_rmse:.6e}")

        results[exp_name] = {
            'n_obs': n_obs, 'noise': noise_std,
            'recon_rmse': recon_rmse,
            'deriv_ut_rmse': deriv_rmse_ut,
            'deriv_lap_rmse': deriv_rmse_lap,
            'D_hat': D_hat, 'D_err': D_err,
            'D_err_pct': D_err_pct,
            'pde_rmse': pde_rmse,
            'delta_max': '',
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

    # delta_max: prediction-space (max |u_merged - u_central| on eval grid)
    delta_max = np.max(np.abs(u_merged - u_central))
    print(f"\n  Merge delta_max (max|u_merged - u_central| on eval grid): {delta_max:.2e}")

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
    merge_results = {}
    for label, u_hat in [("Centralized", u_central), ("Merged", u_merged)]:
        u_t_h, u_xx_h, u_yy_h = compute_derivatives(
            u_hat, dx_eval, dy_eval, dt_eval)
        D_h, pde_r = recover_diffusion_coeff(u_t_h, u_xx_h, u_yy_h,
                                             S_interior)
        D_e = abs(D_h - D_TRUE)
        D_e_pct = 100.0 * D_e / D_TRUE
        print(f"  {label:<16} D={D_h:.6f} (err={D_e:.6f}, {D_e_pct:.2f}%, "
              f"residual={pde_r:.4e})")
        merge_results[label] = {
            'D_hat': D_h, 'D_err': D_e, 'D_err_pct': D_e_pct,
            'pde_rmse': pde_r,
            'recon_rmse': np.sqrt(np.mean((u_hat - u_true_eval)**2)),
        }

    # Store merge experiment results (n_obs = total 10000, not per-holder)
    results["Merge-Central"] = {
        'n_obs': 10000, 'noise': 0.0,
        'recon_rmse': merge_results["Centralized"]['recon_rmse'],
        'D_hat': merge_results["Centralized"]['D_hat'],
        'D_err': merge_results["Centralized"]['D_err'],
        'D_err_pct': merge_results["Centralized"]['D_err_pct'],
        'pde_rmse': merge_results["Centralized"]['pde_rmse'],
        'delta_max': '',
    }
    results["Merge-Gram"] = {
        'n_obs': 10000, 'noise': 0.0,
        'recon_rmse': merge_results["Merged"]['recon_rmse'],
        'D_hat': merge_results["Merged"]['D_hat'],
        'D_err': merge_results["Merged"]['D_err'],
        'D_err_pct': merge_results["Merged"]['D_err_pct'],
        'pde_rmse': merge_results["Merged"]['pde_rmse'],
        'delta_max': delta_max,
    }

    # ================================================================
    # Summary table
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"\n  True diffusion coefficient: D = {D_TRUE}")
    print(f"\n  {'Experiment':<20} {'Recon RMSE':>11} {'D_hat':>9} "
          f"{'D_err':>11} {'err%':>8}")
    print(f"  {'-'*63}")

    for name, r in results.items():
        print(f"  {name:<20} {r['recon_rmse']:>11.6f} {r['D_hat']:>9.6f} "
              f"{r['D_err']:>11.6f} {r.get('D_err_pct', 0):>7.2f}%")

    # ================================================================
    # Visualization: 3 rows (time steps) x 4 columns
    # ================================================================
    print("\nGenerating plots...")

    fig, axes = plt.subplots(3, 4, figsize=(18, 12), facecolor='white')
    fig.suptitle(
        f'Heat with Source: $u_t = {D_TRUE}\\,(u_{{xx}} + u_{{yy}}) + S(x,y)$\n'
        f'Distributed Merge: 2 data holders, Gram statistics, '
        f'$\\Delta_{{max}}$={delta_max:.2e}',
        fontsize=13, fontweight='bold', color='black', y=0.98)

    t_indices = [0, Nt_eval // 2, Nt_eval - 1]
    plot_data = [("True", u_true_eval), ("Centralized", u_central),
                 ("Merged", u_merged), ("|Error|", None)]

    for row, ti in enumerate(t_indices):
        for col, (label, data) in enumerate(plot_data):
            ax = axes[row, col]
            ax.set_facecolor('white')

            if label == "|Error|":
                err = np.abs(u_merged[ti] - u_true_eval[ti])
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
                         fontweight='bold', color='black')
            ax.tick_params(labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#cccccc')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('merge_pde_heat_source.png', dpi=150, facecolor='white',
                bbox_inches='tight')
    print("  Saved: merge_pde_heat_source.png")
    plt.close()

    # ================================================================
    # Save CSV
    # ================================================================
    with open('merge_pde_heat_source.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['experiment', 'n_obs', 'noise', 'recon_rmse',
                     'D_hat', 'D_err', 'D_err_pct',
                     'pde_rmse', 'delta_max'])
        for name, r in results.items():
            w.writerow([name, r['n_obs'], r['noise'],
                        f"{r['recon_rmse']:.8f}",
                        f"{r['D_hat']:.6f}",
                        f"{r['D_err']:.6f}",
                        f"{r.get('D_err_pct', 0):.4f}",
                        f"{r['pde_rmse']:.8e}",
                        r.get('delta_max', '')])
    print("  Saved: merge_pde_heat_source.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
