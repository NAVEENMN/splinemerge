"""
Noise Sensitivity Study for PDE Parameter Recovery

Systematic cross of observation density with noise levels across multiple
seeds to produce a reliability table. Covers both diffusion (u_t = D Laplacian u)
and wave (u_tt = c^2 Laplacian u) equations on [0, 2pi]^2 with periodic BCs.

Conditions:
    Observation counts: 1000, 3000, 10000
    Noise std (relative to clean field std): 0%, 1%, 5%
    Seeds: 42, 43, 44, 45, 46

For each condition, records field RMSE, recovered parameter, parameter
relative error (%), and max prediction difference between distributed
merge and centralized fit.

Output:
    Console summary table (mean +/- std across seeds)
    noise_sensitivity_results.csv
"""

import torch
import numpy as np
import math
import csv
from scipy.interpolate import RegularGridInterpolator

from inkan.basis import bspline_basis_eager


# ============================================================================
# Exact PDE solvers
# ============================================================================

def solve_diffusion_spectral(u0_grid, D, Lx, Ly, times):
    """Exact solution for u_t = D * (u_xx + u_yy).

    Periodic BCs on [0, Lx] x [0, Ly]. Each Fourier mode decays as
    u_hat(k, t) = u_hat(k, 0) * exp(-D |k|^2 t).
    Returns [Nt, Nx, Ny].
    """
    Nx, Ny = u0_grid.shape
    dx, dy = Lx / Nx, Ly / Ny

    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    decay = -D * (KX**2 + KY**2)

    u0_hat = np.fft.fft2(u0_grid)
    result = np.zeros((len(times), Nx, Ny))
    for i, t in enumerate(times):
        result[i] = np.real(np.fft.ifft2(u0_hat * np.exp(decay * t)))

    return result


def eval_wave_analytic(x, y, t, c):
    """Exact analytic solution for the wave initial condition.

    Closed-form evaluation; no interpolation needed.
    """
    return (np.sin(x) * np.cos(y) * np.cos(np.sqrt(2) * c * t)
            + 0.3 * np.sin(2 * x + y) * np.cos(np.sqrt(5) * c * t))


def solve_wave_spectral(u0_grid, ut0_grid, c, Lx, Ly, times):
    """Exact solution for u_tt = c^2 (u_xx + u_yy).

    Periodic BCs. Initial conditions: u(0) = u0, u_t(0) = ut0.
    Returns [Nt, Nx, Ny].
    """
    Nx, Ny = u0_grid.shape
    dx, dy = Lx / Nx, Ly / Ny

    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    omega = c * np.sqrt(KX**2 + KY**2)
    omega_safe = np.where(omega == 0, 1.0, omega)

    u0_hat = np.fft.fft2(u0_grid)
    ut0_hat = np.fft.fft2(ut0_grid)

    result = np.zeros((len(times), Nx, Ny))
    for i, t in enumerate(times):
        u_hat_t = (u0_hat * np.cos(omega * t) +
                   ut0_hat * np.sin(omega * t) / omega_safe)
        u_hat_t[0, 0] = u0_hat[0, 0] + ut0_hat[0, 0] * t
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
        bx = eval_1d_basis(x[start:end], gs_x, ih_x)
        by = eval_1d_basis(y[start:end], gs_y, ih_y)
        bt = eval_1d_basis(t[start:end], gs_t, ih_t)
        phi = (bx[:, :, None, None] * by[:, None, :, None] *
               bt[:, None, None, :]).reshape(-1, P)
        parts.append(phi)

    return torch.cat(parts, dim=0)


# ============================================================================
# Field fitting, evaluation, and merge
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
            raise ValueError(
                f"eval_field_on_grid produced non-finite values at t={t_grid[ti]}")
        u_hat[ti] = vals

    return u_hat


@torch.no_grad()
def gram_merge(G_a, h_a, G_b, h_b, reg=1e-4):
    """Merge two independently fitted fields via Gram statistics.

    C_M = (G_A + G_B + reg*I)^{-1} (h_A + h_B)
    """
    P = G_a.shape[0]
    G = G_a + G_b + reg * torch.eye(P, dtype=torch.float64)
    h = h_a + h_b
    C = torch.linalg.solve(G, h)
    if not torch.isfinite(C).all():
        raise ValueError("gram_merge produced non-finite coefficients")
    return C


# ============================================================================
# Derivative computation (finite differences on evaluation grid)
# ============================================================================

def compute_diffusion_derivatives(u_grid, dx, dy, dt):
    """Central finite differences for u_t and Laplacian.

    Returns derivatives on interior points: [Nt-2, Nx-2, Ny-2].
    """
    u_t = (u_grid[2:, 1:-1, 1:-1] - u_grid[:-2, 1:-1, 1:-1]) / (2 * dt)

    u_xx = (u_grid[1:-1, 2:, 1:-1] - 2 * u_grid[1:-1, 1:-1, 1:-1] +
            u_grid[1:-1, :-2, 1:-1]) / dx**2
    u_yy = (u_grid[1:-1, 1:-1, 2:] - 2 * u_grid[1:-1, 1:-1, 1:-1] +
            u_grid[1:-1, 1:-1, :-2]) / dy**2

    return u_t, u_xx, u_yy


def compute_wave_derivatives(u_grid, dx, dy, dt):
    """Central finite differences including u_tt.

    Returns on interior: [Nt-2, Nx-2, Ny-2].
    """
    u_tt = (u_grid[2:, 1:-1, 1:-1] - 2 * u_grid[1:-1, 1:-1, 1:-1] +
            u_grid[:-2, 1:-1, 1:-1]) / dt**2

    u_xx = (u_grid[1:-1, 2:, 1:-1] - 2 * u_grid[1:-1, 1:-1, 1:-1] +
            u_grid[1:-1, :-2, 1:-1]) / dx**2
    u_yy = (u_grid[1:-1, 1:-1, 2:] - 2 * u_grid[1:-1, 1:-1, 1:-1] +
            u_grid[1:-1, 1:-1, :-2]) / dy**2

    return u_tt, u_xx, u_yy


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
        return float('nan')
    laplacian, b = laplacian[valid], b[valid]

    ATA = np.dot(laplacian, laplacian) + 1e-8
    ATb = np.dot(laplacian, b)
    D_hat = ATb / ATA

    return D_hat


def recover_wave_speed(u_tt, u_xx, u_yy):
    """Recover c from u_tt = c^2 (u_xx + u_yy).

    Returns (c_hat, valid_flag).  If c^2 < 0 the recovery is invalid
    and c_hat is NaN.
    """
    laplacian = (u_xx + u_yy).flatten()
    b = u_tt.flatten()

    valid = np.isfinite(laplacian) & np.isfinite(b)
    if valid.sum() < 10:
        return float('nan'), False
    laplacian, b = laplacian[valid], b[valid]

    ATA = np.dot(laplacian, laplacian) + 1e-8
    ATb = np.dot(laplacian, b)
    c_sq = ATb / ATA

    if c_sq < 0:
        return np.nan, False
    c_hat = np.sqrt(c_sq)
    return c_hat, True


# ============================================================================
# Observation helpers
# ============================================================================

def get_diffusion_obs(obs_x, obs_y, obs_t, u0, D_TRUE, Lx, Ly, x_pde, y_pde):
    """Sample the exact diffusion solution at scattered observation points.

    Uses periodic grid extension for interpolation.
    """
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


def get_wave_obs(obs_x, obs_y, obs_t, c):
    """Sample the exact wave solution at scattered observation points.

    Uses closed-form eval_wave_analytic (no interpolation).
    """
    return eval_wave_analytic(
        obs_x[:, 0].numpy(), obs_y[:, 0].numpy(),
        obs_t[:, 0].numpy(), c)


# ============================================================================
# Ground truth on evaluation grid
# ============================================================================

def diffusion_true_on_eval_grid(
        x_eval, y_eval, t_eval, u0, D_TRUE, Lx, Ly, x_pde, y_pde):
    """Compute exact diffusion solution on evaluation grid.

    Uses periodic grid extension for interpolation.
    """
    Nt_eval = len(t_eval)
    Nx_eval = len(x_eval)
    Ny_eval = len(y_eval)
    u_true_eval = np.zeros((Nt_eval, Nx_eval, Ny_eval))
    XX_g, YY_g = np.meshgrid(x_eval, y_eval, indexing='ij')

    for ti in range(Nt_eval):
        t_val = t_eval[ti].item() if hasattr(t_eval[ti], 'item') else t_eval[ti]
        u_sol = solve_diffusion_spectral(u0, D_TRUE, Lx, Ly, [t_val])[0]
        # Extend grid for periodic interpolation
        x_ext = np.append(x_pde, Lx)
        u_ext = np.concatenate([u_sol, u_sol[0:1, :]], axis=0)
        y_ext = np.append(y_pde, Ly)
        u_ext = np.concatenate([u_ext, u_ext[:, 0:1]], axis=1)
        interp = RegularGridInterpolator(
            (x_ext, y_ext), u_ext, method='cubic',
            bounds_error=False, fill_value=None)
        xx_w = XX_g.flatten() % Lx
        yy_w = YY_g.flatten() % Ly
        pts = np.stack([xx_w, yy_w], axis=1)
        u_true_eval[ti] = interp(pts).reshape(Nx_eval, Ny_eval)

    return u_true_eval


def wave_true_on_eval_grid(x_eval, y_eval, t_eval, c):
    """Compute exact wave solution on evaluation grid.

    Uses closed-form eval_wave_analytic.
    """
    Nt_eval = len(t_eval)
    Nx_eval = len(x_eval)
    Ny_eval = len(y_eval)
    x_np = x_eval.numpy() if hasattr(x_eval, 'numpy') else np.asarray(x_eval)
    y_np = y_eval.numpy() if hasattr(y_eval, 'numpy') else np.asarray(y_eval)
    XX_e, YY_e = np.meshgrid(x_np, y_np, indexing='ij')
    u_true_eval = np.zeros((Nt_eval, Nx_eval, Ny_eval))
    for ti in range(Nt_eval):
        t_val = t_eval[ti].item() if hasattr(t_eval[ti], 'item') else t_eval[ti]
        u_true_eval[ti] = eval_wave_analytic(XX_e, YY_e, t_val, c)
    return u_true_eval


# ============================================================================
# Single-condition runner
# ============================================================================

def run_condition(pde, n_obs, noise_rel, seed, pde_config):
    """Run one (pde, n_obs, noise_rel, seed) condition.

    Uses a single shared dataset: n_obs/2 in the left region (x < pi),
    n_obs/2 in the right region (x >= pi).  The concatenation is used
    for the centralized fit; the left/right partition is used for the
    distributed merge.  ALL metrics come from the same data.

    Returns dict with field_rmse, param_rec, param_rel_error,
    param_signed_error, merge_delta_max, and wave_valid flag.
    """
    cfg = pde_config[pde]
    bp_x, bp_y, bp_t = cfg['bp_x'], cfg['bp_y'], cfg['bp_t']
    x_eval, y_eval, t_eval = cfg['x_eval'], cfg['y_eval'], cfg['t_eval']
    dx_eval, dy_eval, dt_eval = cfg['dx_eval'], cfg['dy_eval'], cfg['dt_eval']
    u_true_eval = cfg['u_true_eval']
    param_true = cfg['param_true']
    Lx, Ly, T_max = cfg['Lx'], cfg['Ly'], cfg['T_max']
    field_std = cfg['field_std']        # fixed noise scale (Fix 3)

    # ------------------------------------------------------------------
    # Generate ONE shared dataset: left (x < pi) + right (x >= pi)
    # ------------------------------------------------------------------
    n_half = n_obs // 2

    torch.manual_seed(seed)
    # Left region: x in [0, pi)
    obs_x_a = torch.rand(n_half, 1, dtype=torch.float64) * math.pi
    obs_y_a = torch.rand(n_half, 1, dtype=torch.float64) * Ly
    obs_t_a = torch.rand(n_half, 1, dtype=torch.float64) * T_max

    # Right region: x in [pi, 2*pi)
    obs_x_b = torch.rand(n_half, 1, dtype=torch.float64) * math.pi + math.pi
    obs_y_b = torch.rand(n_half, 1, dtype=torch.float64) * Ly
    obs_t_b = torch.rand(n_half, 1, dtype=torch.float64) * T_max

    if pde == 'diffusion':
        u_obs_a = get_diffusion_obs(
            obs_x_a, obs_y_a, obs_t_a,
            cfg['u0'], cfg['param_true'], Lx, Ly,
            cfg['x_pde'], cfg['y_pde'])
        u_obs_b = get_diffusion_obs(
            obs_x_b, obs_y_b, obs_t_b,
            cfg['u0'], cfg['param_true'], Lx, Ly,
            cfg['x_pde'], cfg['y_pde'])
    else:
        u_obs_a = get_wave_obs(obs_x_a, obs_y_a, obs_t_a, cfg['param_true'])
        u_obs_b = get_wave_obs(obs_x_b, obs_y_b, obs_t_b, cfg['param_true'])

    # Fixed noise scale from dense reference grid (Fix 3)
    noise_abs = noise_rel * field_std

    if noise_abs > 0:
        rng_a = np.random.default_rng(seed)
        rng_b = np.random.default_rng(seed + 1000)
        u_obs_a = u_obs_a + rng_a.normal(0, noise_abs, n_half)
        u_obs_b = u_obs_b + rng_b.normal(0, noise_abs, n_half)

    u_obs_a_t = torch.tensor(u_obs_a, dtype=torch.float64)
    u_obs_b_t = torch.tensor(u_obs_b, dtype=torch.float64)

    # ------------------------------------------------------------------
    # Concatenate for centralized fit (same data as merge)
    # ------------------------------------------------------------------
    obs_x_all = torch.cat([obs_x_a, obs_x_b])
    obs_y_all = torch.cat([obs_y_a, obs_y_b])
    obs_t_all = torch.cat([obs_t_a, obs_t_b])
    u_obs_all = torch.cat([u_obs_a_t, u_obs_b_t])

    Phi_all = build_3d_features(
        obs_x_all, obs_y_all, obs_t_all, bp_x, bp_y, bp_t)
    C_central, _ = fit_field(Phi_all, u_obs_all, reg=1e-4)

    u_hat_central = eval_field_on_grid(
        C_central, x_eval, y_eval, t_eval, bp_x, bp_y, bp_t)

    # Field RMSE
    field_rmse = np.sqrt(np.mean((u_hat_central - u_true_eval)**2))

    # Recover parameter
    wave_valid = True
    if pde == 'diffusion':
        u_t, u_xx, u_yy = compute_diffusion_derivatives(
            u_hat_central, dx_eval, dy_eval, dt_eval)
        param_rec = recover_diffusion_coeff(u_t, u_xx, u_yy)
    else:
        u_tt, u_xx, u_yy = compute_wave_derivatives(
            u_hat_central, dx_eval, dy_eval, dt_eval)
        param_rec, wave_valid = recover_wave_speed(u_tt, u_xx, u_yy)

    param_rel_error = abs(param_rec - param_true) / abs(param_true) * 100.0
    param_signed_error = param_rec - param_true

    # ------------------------------------------------------------------
    # Distributed merge using the same left/right partition
    # ------------------------------------------------------------------
    Phi_a = build_3d_features(obs_x_a, obs_y_a, obs_t_a, bp_x, bp_y, bp_t)
    Phi_b = build_3d_features(obs_x_b, obs_y_b, obs_t_b, bp_x, bp_y, bp_t)

    C_a, G_a = fit_field(Phi_a, u_obs_a_t, reg=1e-4)
    C_b, G_b = fit_field(Phi_b, u_obs_b_t, reg=1e-4)
    h_a = Phi_a.T @ u_obs_a_t
    h_b = Phi_b.T @ u_obs_b_t

    # Gram merge
    C_merged = gram_merge(G_a, h_a, G_b, h_b, reg=1e-4)

    # Evaluate centralized and merged on the same grid
    u_merged = eval_field_on_grid(
        C_merged, x_eval, y_eval, t_eval, bp_x, bp_y, bp_t)

    merge_delta_max = np.max(np.abs(u_merged - u_hat_central))

    return {
        'field_rmse': field_rmse,
        'param_rec': param_rec,
        'param_rel_error': param_rel_error,
        'param_signed_error': param_signed_error,
        'merge_delta_max': merge_delta_max,
        'wave_valid': wave_valid,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    # ------------------------------------------------------------------
    # Experimental design
    # ------------------------------------------------------------------
    OBS_COUNTS = [1000, 3000, 10000]
    NOISE_RELS = [0.0, 0.01, 0.05]
    SEEDS = [42, 43, 44, 45, 46]
    PDE_NAMES = ['diffusion', 'wave']

    Lx, Ly = 2 * math.pi, 2 * math.pi

    # ------------------------------------------------------------------
    # Diffusion configuration
    # ------------------------------------------------------------------
    D_TRUE = 0.05
    T_max_diff = 2.0

    Nx_pde, Ny_pde = 64, 64
    x_pde_diff = np.linspace(0, Lx, Nx_pde, endpoint=False)
    y_pde_diff = np.linspace(0, Ly, Ny_pde, endpoint=False)
    XX_d, YY_d = np.meshgrid(x_pde_diff, y_pde_diff, indexing='ij')
    u0_diff = (np.sin(XX_d) * np.cos(YY_d)
               + 0.5 * np.exp(-3 * ((XX_d - math.pi)**2 + (YY_d - math.pi)**2))
               + 0.3 * np.sin(2 * XX_d + YY_d))

    gs_x_d, gs_y_d, gs_t_d = 8, 8, 8
    bp_x_d = make_basis_params(gs_x_d, (0, Lx))
    bp_y_d = make_basis_params(gs_y_d, (0, Ly))
    bp_t_d = make_basis_params(gs_t_d, (0, T_max_diff))

    Nx_eval_d, Ny_eval_d, Nt_eval_d = 40, 40, 30
    x_eval_d = torch.linspace(0.3, Lx - 0.3, Nx_eval_d)
    y_eval_d = torch.linspace(0.3, Ly - 0.3, Ny_eval_d)
    t_eval_d = torch.linspace(0.2, T_max_diff - 0.2, Nt_eval_d)
    dx_eval_d = (x_eval_d[1] - x_eval_d[0]).item()
    dy_eval_d = (y_eval_d[1] - y_eval_d[0]).item()
    dt_eval_d = (t_eval_d[1] - t_eval_d[0]).item()

    print("Computing diffusion ground truth on evaluation grid...")
    u_true_eval_d = diffusion_true_on_eval_grid(
        x_eval_d.numpy(), y_eval_d.numpy(), t_eval_d,
        u0_diff, D_TRUE, Lx, Ly, x_pde_diff, y_pde_diff)

    # Fixed noise scale: compute field std on dense 100x100x50 reference grid
    print("  Computing fixed diffusion field_std on dense reference grid...")
    ref_x_d = np.linspace(0, Lx, 100, endpoint=False)
    ref_y_d = np.linspace(0, Ly, 100, endpoint=False)
    ref_t_d = np.linspace(0, T_max_diff, 50)
    ref_vals_d = diffusion_true_on_eval_grid(
        ref_x_d, ref_y_d, torch.tensor(ref_t_d),
        u0_diff, D_TRUE, Lx, Ly, x_pde_diff, y_pde_diff)
    field_std_d = float(np.std(ref_vals_d))
    print(f"  field_std_diffusion = {field_std_d:.6f}")
    print("  Done.")

    # ------------------------------------------------------------------
    # Wave configuration
    # ------------------------------------------------------------------
    C_TRUE = 1.0
    T_max_wave = 3.0

    gs_x_w, gs_y_w, gs_t_w = 8, 8, 10
    bp_x_w = make_basis_params(gs_x_w, (0, Lx))
    bp_y_w = make_basis_params(gs_y_w, (0, Ly))
    bp_t_w = make_basis_params(gs_t_w, (0, T_max_wave))

    Nx_eval_w, Ny_eval_w, Nt_eval_w = 40, 40, 40
    x_eval_w = torch.linspace(0.3, Lx - 0.3, Nx_eval_w)
    y_eval_w = torch.linspace(0.3, Ly - 0.3, Ny_eval_w)
    t_eval_w = torch.linspace(0.2, T_max_wave - 0.2, Nt_eval_w)
    dx_eval_w = (x_eval_w[1] - x_eval_w[0]).item()
    dy_eval_w = (y_eval_w[1] - y_eval_w[0]).item()
    dt_eval_w = (t_eval_w[1] - t_eval_w[0]).item()

    print("Computing wave ground truth on evaluation grid...")
    u_true_eval_w = wave_true_on_eval_grid(
        x_eval_w, y_eval_w, t_eval_w, C_TRUE)

    # Fixed noise scale: compute field std on dense 100x100x50 reference grid
    print("  Computing fixed wave field_std on dense reference grid...")
    ref_x_w = np.linspace(0, Lx, 100, endpoint=False)
    ref_y_w = np.linspace(0, Ly, 100, endpoint=False)
    ref_t_w = np.linspace(0, T_max_wave, 50)
    ref_vals_w = wave_true_on_eval_grid(
        torch.tensor(ref_x_w), torch.tensor(ref_y_w),
        torch.tensor(ref_t_w), C_TRUE)
    field_std_w = float(np.std(ref_vals_w))
    print(f"  field_std_wave = {field_std_w:.6f}")
    print("  Done.")

    # ------------------------------------------------------------------
    # Pack configs
    # ------------------------------------------------------------------
    pde_config = {
        'diffusion': {
            'bp_x': bp_x_d, 'bp_y': bp_y_d, 'bp_t': bp_t_d,
            'x_eval': x_eval_d, 'y_eval': y_eval_d, 't_eval': t_eval_d,
            'dx_eval': dx_eval_d, 'dy_eval': dy_eval_d, 'dt_eval': dt_eval_d,
            'u_true_eval': u_true_eval_d,
            'param_true': D_TRUE,
            'Lx': Lx, 'Ly': Ly, 'T_max': T_max_diff,
            'u0': u0_diff,
            'x_pde': x_pde_diff, 'y_pde': y_pde_diff,
            'field_std': field_std_d,
        },
        'wave': {
            'bp_x': bp_x_w, 'bp_y': bp_y_w, 'bp_t': bp_t_w,
            'x_eval': x_eval_w, 'y_eval': y_eval_w, 't_eval': t_eval_w,
            'dx_eval': dx_eval_w, 'dy_eval': dy_eval_w, 'dt_eval': dt_eval_w,
            'u_true_eval': u_true_eval_w,
            'param_true': C_TRUE,
            'Lx': Lx, 'Ly': Ly, 'T_max': T_max_wave,
            'field_std': field_std_w,
        },
    }

    # ------------------------------------------------------------------
    # Run all conditions
    # ------------------------------------------------------------------
    all_rows = []
    total = len(PDE_NAMES) * len(OBS_COUNTS) * len(NOISE_RELS) * len(SEEDS)
    count = 0

    print(f"\n{'='*74}")
    print("Noise Sensitivity Study")
    print(f"  PDEs:            {PDE_NAMES}")
    print(f"  Observation counts: {OBS_COUNTS}")
    print(f"  Noise levels:    {[f'{100*n:.0f}%' for n in NOISE_RELS]}")
    print(f"  Seeds:           {SEEDS}")
    print(f"  Total conditions: {total}")
    print(f"{'='*74}\n")

    for pde in PDE_NAMES:
        for n_obs in OBS_COUNTS:
            for noise_rel in NOISE_RELS:
                for seed in SEEDS:
                    count += 1
                    tag = (f"[{count}/{total}] {pde} n={n_obs} "
                           f"noise={100*noise_rel:.0f}% seed={seed}")
                    print(f"  {tag} ...", end="", flush=True)

                    result = run_condition(
                        pde, n_obs, noise_rel, seed, pde_config)

                    row = {
                        'pde': pde,
                        'n_obs': n_obs,
                        'noise_rel': noise_rel,
                        'seed': seed,
                        'field_rmse': result['field_rmse'],
                        'param_rec': result['param_rec'],
                        'param_rel_error': result['param_rel_error'],
                        'param_signed_error': result['param_signed_error'],
                        'merge_delta_max': result['merge_delta_max'],
                        'wave_valid': result['wave_valid'],
                    }
                    all_rows.append(row)

                    valid_tag = "" if result['wave_valid'] else "  [c^2<0]"
                    print(f"  RMSE={result['field_rmse']:.6f}  "
                          f"param={result['param_rec']:.6f}  "
                          f"rel_err={result['param_rel_error']:.2f}%  "
                          f"merge_delta={result['merge_delta_max']:.2e}"
                          f"{valid_tag}")

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    csv_path = 'noise_sensitivity_results.csv'
    csv_cols = ['pde', 'n_obs', 'noise_rel', 'seed',
                'field_rmse', 'param_rec', 'param_rel_error',
                'param_signed_error', 'merge_delta_max', 'wave_valid']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({
                'pde': row['pde'],
                'n_obs': row['n_obs'],
                'noise_rel': row['noise_rel'],
                'seed': row['seed'],
                'field_rmse': f"{row['field_rmse']:.8f}",
                'param_rec': f"{row['param_rec']:.8f}",
                'param_rel_error': f"{row['param_rel_error']:.4f}",
                'param_signed_error': f"{row['param_signed_error']:.8f}",
                'merge_delta_max': f"{row['merge_delta_max']:.8e}",
                'wave_valid': row['wave_valid'],
            })
    print(f"\nSaved: {csv_path}")

    # ------------------------------------------------------------------
    # Summary table: mean +/- std across seeds (ddof=1 for sample std)
    # ------------------------------------------------------------------
    print(f"\n{'='*120}")
    print("SUMMARY TABLE (mean +/- std across 5 seeds, sample std with ddof=1)")
    print(f"{'='*120}")

    header = (f"  {'PDE':<11} {'n_obs':>6} {'noise':>6} | "
              f"{'field_rmse':>18} {'param_rec':>18} "
              f"{'rel_err(%)':>18} {'merge_delta':>18} "
              f"{'fail':>5}")
    print(header)
    print(f"  {'-'*110}")

    for pde in PDE_NAMES:
        param_label = 'D' if pde == 'diffusion' else 'c_w'
        param_true = D_TRUE if pde == 'diffusion' else C_TRUE
        for n_obs in OBS_COUNTS:
            for noise_rel in NOISE_RELS:
                subset = [r for r in all_rows
                          if r['pde'] == pde
                          and r['n_obs'] == n_obs
                          and r['noise_rel'] == noise_rel]

                n_fail = sum(1 for r in subset if not r['wave_valid'])
                rmses = np.array([r['field_rmse'] for r in subset])
                params = np.array([r['param_rec'] for r in subset])
                rel_errs = np.array([r['param_rel_error'] for r in subset])
                deltas = np.array([r['merge_delta_max'] for r in subset])

                noise_pct = f"{100*noise_rel:.0f}%"
                fail_str = f"{n_fail}/{len(subset)}" if n_fail > 0 else ""
                print(f"  {pde:<11} {n_obs:>6} {noise_pct:>6} | "
                      f"{np.nanmean(rmses):.6f} +/- {np.nanstd(rmses, ddof=1):.6f} "
                      f"{np.nanmean(params):.6f} +/- {np.nanstd(params, ddof=1):.6f} "
                      f"{np.nanmean(rel_errs):.2f} +/- {np.nanstd(rel_errs, ddof=1):.2f} "
                      f"{np.nanmean(deltas):.2e} +/- {np.nanstd(deltas, ddof=1):.2e} "
                      f"{fail_str:>5}")

        print(f"  {'-'*110}")

    # ------------------------------------------------------------------
    # Per-PDE parameter summary (with signed bias)
    # ------------------------------------------------------------------
    for pde in PDE_NAMES:
        param_label = 'D' if pde == 'diffusion' else 'c_w'
        param_true = D_TRUE if pde == 'diffusion' else C_TRUE
        print(f"\n  {pde.upper()} -- true {param_label} = {param_true}")
        print(f"  {'n_obs':>6} {'noise':>6} | "
              f"{param_label+'_mean':>10} {param_label+'_std':>10} "
              f"{'rel_err_mean':>12} {'rel_err_std':>12} "
              f"{'signed_bias':>12} {'fail':>6}")
        print(f"  {'-'*80}")
        for n_obs in OBS_COUNTS:
            for noise_rel in NOISE_RELS:
                subset = [r for r in all_rows
                          if r['pde'] == pde
                          and r['n_obs'] == n_obs
                          and r['noise_rel'] == noise_rel]
                n_fail = sum(1 for r in subset if not r['wave_valid'])
                params = np.array([r['param_rec'] for r in subset])
                rel_errs = np.array([r['param_rel_error'] for r in subset])
                signed_errs = np.array([r['param_signed_error'] for r in subset])
                noise_pct = f"{100*noise_rel:.0f}%"
                fail_str = f"{n_fail}/{len(subset)}" if n_fail > 0 else ""
                print(f"  {n_obs:>6} {noise_pct:>6} | "
                      f"{np.nanmean(params):>10.6f} {np.nanstd(params, ddof=1):>10.6f} "
                      f"{np.nanmean(rel_errs):>12.4f} {np.nanstd(rel_errs, ddof=1):>12.4f} "
                      f"{np.nanmean(signed_errs):>12.6f} "
                      f"{fail_str:>6}")

    # ------------------------------------------------------------------
    # Wave failure rate summary
    # ------------------------------------------------------------------
    wave_rows = [r for r in all_rows if r['pde'] == 'wave']
    total_wave = len(wave_rows)
    total_fail = sum(1 for r in wave_rows if not r['wave_valid'])
    if total_fail > 0:
        print(f"\n  WAVE c^2 < 0 FAILURE SUMMARY: {total_fail}/{total_wave} "
              f"({100*total_fail/total_wave:.1f}%) seeds produced negative c^2")
        print(f"  {'n_obs':>6} {'noise':>6} | {'failures':>8}")
        print(f"  {'-'*30}")
        for n_obs in OBS_COUNTS:
            for noise_rel in NOISE_RELS:
                subset = [r for r in wave_rows
                          if r['n_obs'] == n_obs
                          and r['noise_rel'] == noise_rel]
                nf = sum(1 for r in subset if not r['wave_valid'])
                if nf > 0:
                    noise_pct = f"{100*noise_rel:.0f}%"
                    print(f"  {n_obs:>6} {noise_pct:>6} | {nf}/{len(subset)}")
    else:
        print(f"\n  WAVE: all {total_wave} conditions produced valid c^2 >= 0")

    print(f"\n{'='*120}")
    print("Done.")


if __name__ == "__main__":
    main()
