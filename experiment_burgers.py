"""
Synthetic PDE Experiment: Viscous Burgers Equation (Nonlinear)

  u_t + u * u_x = kappa * u_xx    (1D viscous Burgers equation)

Known ground truth: kappa = 0.1 (viscosity).
Periodic BCs on [0, 2*pi], time domain [0, 1.0].
Initial condition: u0(x) = sin(x) + 0.5*sin(2x).

This is a NONLINEAR PDE -- the key test requested by the senior reviewer.
No closed-form general solution; ground truth computed via high-resolution
pseudospectral solver (dealiased 3/2 rule, RK4 time stepping, N=256).

The PDE u_t + u*u_x = kappa*u_xx is nonlinear in u, but LINEAR in the
unknown parameter kappa when the PDE structure is known. Rearranging:
  kappa = sum(u_xx * (u_t + u*u_x)) / sum(u_xx^2)

The experiment fits a 2D tensor-product B-spline field (x, t) from
scattered observations, then recovers the viscosity from finite-difference
derivatives. Distributed merge via Gram statistics demonstrates that
physics can be recovered from spatially partitioned data holders.
"""

import torch
import numpy as np
import math
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from inkan.basis import bspline_basis_eager


# ============================================================================
# Pseudospectral Burgers solver (dealiased, RK4)
# ============================================================================

def solve_burgers_spectral(u0, kappa, L, dt, n_steps, save_every=1):
    """Pseudospectral solver for u_t + u*u_x = kappa*u_xx.

    Periodic BCs on [0, L]. Dealiasing via 3/2 rule for the nonlinear
    term u*u_x. Time stepping: classical RK4 with step size dt.

    Args:
        u0: Initial condition on N uniform grid points, shape [N].
        kappa: Viscosity coefficient.
        L: Domain length.
        dt: Time step size.
        n_steps: Number of time steps.
        save_every: Save a snapshot every this many steps.

    Returns:
        results: Array [Nt_saved, N] of solution snapshots.
        times: Array [Nt_saved] of corresponding times.
    """
    N = len(u0)
    dx = L / N
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi  # wavenumbers

    # Dealiasing grid (3/2 rule)
    N_dealias = 3 * N // 2
    k_d = np.fft.fftfreq(N_dealias, d=L / N_dealias) * 2 * np.pi

    def rhs(u_hat):
        """Compute RHS of du_hat/dt = -F[u*u_x] - kappa*k^2*u_hat."""
        # Diffusion in Fourier space
        diffusion = -kappa * k**2 * u_hat

        # Nonlinear term u*u_x via dealiased pseudospectral:
        # Pad u_hat to 3/2 N modes, transform to physical space,
        # compute u*u_x, transform back, truncate.
        u_hat_padded = np.zeros(N_dealias, dtype=complex)
        u_hat_padded[:N // 2] = u_hat[:N // 2]
        u_hat_padded[-(N // 2 - 1):] = u_hat[-(N // 2 - 1):]

        u_phys = np.fft.ifft(u_hat_padded).real * (N_dealias / N)

        # u_x in physical space (dealiased grid)
        ux_hat_padded = 1j * k_d * u_hat_padded
        # Zero high modes for dealiasing
        ux_hat_padded[N_dealias // 3: 2 * N_dealias // 3] = 0.0
        u_hat_padded_clean = u_hat_padded.copy()
        u_hat_padded_clean[N_dealias // 3: 2 * N_dealias // 3] = 0.0

        u_phys = np.fft.ifft(u_hat_padded_clean).real * (N_dealias / N)
        ux_phys = np.fft.ifft(ux_hat_padded).real * (N_dealias / N)

        # Product in physical space
        product = u_phys * ux_phys

        # Transform back and truncate
        product_hat_full = np.fft.fft(product) * (N / N_dealias)
        nonlinear = np.zeros(N, dtype=complex)
        nonlinear[:N // 2] = product_hat_full[:N // 2]
        nonlinear[-(N // 2 - 1):] = product_hat_full[-(N // 2 - 1):]

        return diffusion - nonlinear

    u_hat = np.fft.fft(u0)
    results = [u0.copy()]
    times = [0.0]

    for step in range(1, n_steps + 1):
        # Classical RK4
        k1 = dt * rhs(u_hat)
        k2 = dt * rhs(u_hat + 0.5 * k1)
        k3 = dt * rhs(u_hat + 0.5 * k2)
        k4 = dt * rhs(u_hat + k3)
        u_hat = u_hat + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        if step % save_every == 0:
            u_phys = np.fft.ifft(u_hat).real
            if not np.isfinite(u_phys).all():
                raise ValueError(
                    f"Burgers solver produced non-finite values at step {step}")
            results.append(u_phys.copy())
            times.append(step * dt)

    return np.array(results), np.array(times)


# ============================================================================
# 2D tensor-product B-spline features (x, t)
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
def build_2d_features(x, t, bp_x, bp_t, batch_size=2000):
    """Build 2D tensor-product feature matrix.

    Phi[n, i*Kt + l] = B_i(x_n) * T_l(t_n)
    Returns [N, Kx*Kt].
    """
    gs_x, ih_x, Kx = bp_x
    gs_t, ih_t, Kt = bp_t
    N = len(x)
    P = Kx * Kt
    parts = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        bx = eval_1d_basis(x[start:end], gs_x, ih_x)  # [B, Kx]
        bt = eval_1d_basis(t[start:end], gs_t, ih_t)  # [B, Kt]
        # Outer product: [B, Kx, Kt] -> [B, P]
        phi = (bx[:, :, None] * bt[:, None, :]).reshape(-1, P)
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
def eval_field_on_grid(C, x_grid, t_grid, bp_x, bp_t):
    """Evaluate fitted field on a regular 2D grid.

    Returns u_hat[t_idx, x_idx].
    """
    Nt, Nx = len(t_grid), len(x_grid)
    u_hat = np.zeros((Nt, Nx))

    for ti in range(Nt):
        xx = x_grid.reshape(-1, 1)
        tt = torch.full_like(xx, t_grid[ti].item())

        Phi = build_2d_features(xx, tt, bp_x, bp_t)
        vals = (Phi @ C).numpy().reshape(Nx)
        if not np.isfinite(vals).all():
            raise ValueError(
                f"eval_field_on_grid produced non-finite values at t={t_grid[ti]}")
        u_hat[ti] = vals

    return u_hat


# ============================================================================
# Derivative computation (finite differences on evaluation grid)
# ============================================================================

def compute_burgers_derivatives(u_grid, dx, dt):
    """Central finite differences for u_t, u_x, u_xx.

    u_grid: shape [Nt, Nx].
    Returns derivatives on interior points: [Nt-2, Nx-2].
    """
    # Time derivative (central)
    u_t = (u_grid[2:, 1:-1] - u_grid[:-2, 1:-1]) / (2 * dt)

    # First spatial derivative (central)
    u_x = (u_grid[1:-1, 2:] - u_grid[1:-1, :-2]) / (2 * dx)

    # Second spatial derivative (central)
    u_xx = (u_grid[1:-1, 2:] - 2 * u_grid[1:-1, 1:-1] +
            u_grid[1:-1, :-2]) / dx**2

    # Field values on the interior
    u_interior = u_grid[1:-1, 1:-1]

    return u_t, u_x, u_xx, u_interior


# ============================================================================
# Physics recovery
# ============================================================================

def recover_viscosity(u_t, u_x, u_xx, u_interior):
    """Recover kappa from u_t + u*u_x = kappa * u_xx.

    Rearranged: kappa = sum(u_xx * (u_t + u*u_x)) / sum(u_xx^2).
    """
    residual_lhs = (u_t + u_interior * u_x).flatten()
    laplacian = u_xx.flatten()

    valid = np.isfinite(residual_lhs) & np.isfinite(laplacian)
    if valid.sum() < 10:
        return float('nan'), float('nan')
    residual_lhs = residual_lhs[valid]
    laplacian = laplacian[valid]

    # Least-squares: kappa = (u_xx^T * rhs) / (u_xx^T * u_xx)
    ATA = np.dot(laplacian, laplacian) + 1e-8
    ATb = np.dot(laplacian, residual_lhs)
    kappa_hat = ATb / ATA

    # PDE residual: u_t + u*u_x - kappa*u_xx
    pde_residual = residual_lhs - kappa_hat * laplacian
    rmse = np.sqrt(np.mean(pde_residual**2))

    return kappa_hat, rmse


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
    KAPPA_TRUE = 0.1
    L = 2 * math.pi
    T_MAX = 1.0

    # PDE grid (high resolution for ground truth)
    N_pde = 256
    dt_pde = 0.001
    n_steps = int(T_MAX / dt_pde)
    save_every = max(1, n_steps // 100)  # save ~100 snapshots

    x_pde = np.linspace(0, L, N_pde, endpoint=False)

    # Initial condition: u0(x) = sin(x) + 0.5*sin(2x)
    u0 = np.sin(x_pde) + 0.5 * np.sin(2 * x_pde)

    print("=" * 70)
    print("Synthetic PDE: Viscous Burgers Equation (Nonlinear)")
    print(f"  u_t + u * u_x = {KAPPA_TRUE} * u_xx")
    print(f"  Domain: [0, 2*pi] x [0, {T_MAX}]")
    print(f"  PDE grid: N={N_pde}, dt={dt_pde}, steps={n_steps}")
    print("=" * 70)

    # Solve PDE via pseudospectral method
    print("\nSolving Burgers equation (pseudospectral, dealiased RK4)...")
    u_snapshots, t_snapshots = solve_burgers_spectral(
        u0, KAPPA_TRUE, L, dt_pde, n_steps, save_every=save_every)
    Nt_snap = len(t_snapshots)
    print(f"  Saved {Nt_snap} snapshots, t in [{t_snapshots[0]:.3f}, "
          f"{t_snapshots[-1]:.3f}]")
    print(f"  Solution range: [{u_snapshots.min():.4f}, "
          f"{u_snapshots.max():.4f}]")

    # Spline basis parameters: 2D (x, t)
    gs_x, gs_t = 12, 10
    bp_x = make_basis_params(gs_x, (0, L))
    bp_t = make_basis_params(gs_t, (0, T_MAX))
    Kx, Kt = bp_x[2], bp_t[2]
    P = Kx * Kt
    print(f"\n  Spline basis: Kx={Kx}, Kt={Kt}, total features P={P}")

    # Evaluation grid (avoid boundaries for finite differences)
    Nx_eval, Nt_eval = 80, 60
    x_eval = torch.linspace(0.2, L - 0.2, Nx_eval, dtype=torch.float64)
    t_eval = torch.linspace(0.05, T_MAX - 0.05, Nt_eval, dtype=torch.float64)
    dx_eval = (x_eval[1] - x_eval[0]).item()
    dt_eval = (t_eval[1] - t_eval[0]).item()

    # Ground truth on evaluation grid via interpolation from PDE snapshots
    def eval_true_on_grid(x_grid_np, t_grid_np):
        """Interpolate spectral solution onto the evaluation grid.

        Returns u_true[Nt_eval, Nx_eval].
        """
        # Extend x_pde for periodic interpolation
        x_ext = np.append(x_pde, L)
        u_ext = np.concatenate(
            [u_snapshots, u_snapshots[:, 0:1]], axis=1)  # [Nt_snap, N+1]

        interp = RegularGridInterpolator(
            (t_snapshots, x_ext), u_ext, method='cubic',
            bounds_error=False, fill_value=None)

        Nt_e, Nx_e = len(t_grid_np), len(x_grid_np)
        TT, XX = np.meshgrid(t_grid_np, x_grid_np, indexing='ij')
        pts = np.stack([TT.flatten(), XX.flatten() % L], axis=1)
        return interp(pts).reshape(Nt_e, Nx_e)

    print("\n  Computing ground truth on evaluation grid...")
    u_true_eval = eval_true_on_grid(x_eval.numpy(), t_eval.numpy())
    if not np.isfinite(u_true_eval).all():
        raise ValueError("Ground truth interpolation produced non-finite values")
    print(f"  u_true_eval range: [{u_true_eval.min():.4f}, "
          f"{u_true_eval.max():.4f}]")

    # Ground truth derivatives and sanity check
    u_t_true, u_x_true, u_xx_true, u_int_true = compute_burgers_derivatives(
        u_true_eval, dx_eval, dt_eval)
    kappa_gt, rmse_gt = recover_viscosity(
        u_t_true, u_x_true, u_xx_true, u_int_true)
    print(f"\n  Ground truth physics recovery (sanity check):")
    print(f"    kappa: {kappa_gt:.6f} (true={KAPPA_TRUE})")
    print(f"    RMSE:  {rmse_gt:.6e}")

    # Helper to sample observations from the solved PDE
    def get_obs(obs_x, obs_t):
        """Sample the Burgers solution at scattered observation points.

        Args:
            obs_x: [N, 1] tensor of x coordinates.
            obs_t: [N, 1] tensor of t coordinates.

        Returns:
            vals: [N] numpy array of solution values.
        """
        x_ext = np.append(x_pde, L)
        u_ext = np.concatenate(
            [u_snapshots, u_snapshots[:, 0:1]], axis=1)

        interp = RegularGridInterpolator(
            (t_snapshots, x_ext), u_ext, method='cubic',
            bounds_error=False, fill_value=None)

        n = len(obs_x)
        x_np = obs_x[:, 0].numpy() % L
        t_np = obs_t[:, 0].numpy()
        # Clamp t to valid range
        t_np = np.clip(t_np, t_snapshots[0], t_snapshots[-1])
        pts = np.stack([t_np, x_np], axis=1)
        vals = interp(pts)
        if not np.isfinite(vals).all():
            raise ValueError("get_obs produced non-finite values")
        return vals

    # ================================================================
    # Experiment loop: vary observation density and noise
    # ================================================================
    results = {}

    for exp_name, n_obs, noise_std in [
        ("Dense (5K)", 5000, 0.0),
        ("Moderate (2K)", 2000, 0.0),
        ("Sparse (500)", 500, 0.0),
        ("Dense+noise", 5000, 0.02),
    ]:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name} (n={n_obs}, noise={noise_std})")
        print(f"{'='*60}")

        # Sample random observation points
        torch.manual_seed(42)
        obs_x = (torch.rand(n_obs, 1) * L).double()
        obs_t = (torch.rand(n_obs, 1) * T_MAX).double()

        # Get true values at observation points
        obs_u = get_obs(obs_x, obs_t)
        if noise_std > 0:
            obs_u = obs_u + np.random.default_rng(42).normal(
                0, noise_std, n_obs)
        obs_u_tensor = torch.tensor(obs_u, dtype=torch.float64)

        # Build features and fit
        print("  Building features...")
        Phi = build_2d_features(obs_x, obs_t, bp_x, bp_t)
        print(f"  Phi shape: {Phi.shape}")

        C, G = fit_field(Phi, obs_u_tensor, reg=1e-4)

        # Evaluate on grid
        print("  Evaluating on grid...")
        u_hat = eval_field_on_grid(C, x_eval, t_eval, bp_x, bp_t)

        # Reconstruction RMSE
        recon_rmse = np.sqrt(np.mean((u_hat - u_true_eval)**2))

        # Derivative recovery
        u_t_hat, u_x_hat, u_xx_hat, u_int_hat = compute_burgers_derivatives(
            u_hat, dx_eval, dt_eval)

        deriv_rmse_ut = np.sqrt(np.mean((u_t_hat - u_t_true)**2))
        deriv_rmse_uxx = np.sqrt(np.mean((u_xx_hat - u_xx_true)**2))

        # Physics recovery
        kappa_hat, pde_rmse = recover_viscosity(
            u_t_hat, u_x_hat, u_xx_hat, u_int_hat)
        kappa_err = abs(kappa_hat - KAPPA_TRUE)
        kappa_err_pct = 100.0 * kappa_err / KAPPA_TRUE

        print(f"\n  Reconstruction RMSE:      {recon_rmse:.6f}")
        print(f"  Derivative u_t RMSE:      {deriv_rmse_ut:.6f}")
        print(f"  Derivative u_xx RMSE:     {deriv_rmse_uxx:.6f}")
        print(f"  Physics recovery:")
        print(f"    kappa: {kappa_hat:.6f} (true={KAPPA_TRUE}, "
              f"err={kappa_err:.6f}, {kappa_err_pct:.2f}%)")
        print(f"    PDE residual RMSE: {pde_rmse:.6e}")

        results[exp_name] = {
            'n_obs': n_obs, 'noise': noise_std,
            'recon_rmse': recon_rmse,
            'deriv_ut_rmse': deriv_rmse_ut,
            'deriv_uxx_rmse': deriv_rmse_uxx,
            'kappa_hat': kappa_hat, 'kappa_err': kappa_err,
            'kappa_err_pct': kappa_err_pct,
            'pde_rmse': pde_rmse,
        }

    # ================================================================
    # Distributed merge experiment
    # ================================================================
    print(f"\n{'='*60}")
    print("Distributed Merge (spatial split at x=pi)")
    print(f"  2500 obs per holder, noiseless")
    print(f"{'='*60}")

    n_per_holder = 2500
    torch.manual_seed(42)

    # Holder A: x in [0, pi)
    obs_x_a = (torch.rand(n_per_holder, 1) * math.pi).double()
    obs_t_a = (torch.rand(n_per_holder, 1) * T_MAX).double()

    # Holder B: x in [pi, 2*pi)
    obs_x_b = (torch.rand(n_per_holder, 1) * math.pi + math.pi).double()
    obs_t_b = (torch.rand(n_per_holder, 1) * T_MAX).double()

    print("  Collecting holder A observations (x < pi)...")
    u_obs_a = torch.tensor(get_obs(obs_x_a, obs_t_a), dtype=torch.float64)
    print("  Collecting holder B observations (x >= pi)...")
    u_obs_b = torch.tensor(get_obs(obs_x_b, obs_t_b), dtype=torch.float64)

    # Centralized (all 5K points)
    obs_x_all = torch.cat([obs_x_a, obs_x_b])
    obs_t_all = torch.cat([obs_t_a, obs_t_b])
    u_obs_all = torch.cat([u_obs_a, u_obs_b])

    print("  Building features and fitting...")
    Phi_a = build_2d_features(obs_x_a, obs_t_a, bp_x, bp_t)
    Phi_b = build_2d_features(obs_x_b, obs_t_b, bp_x, bp_t)
    Phi_all = build_2d_features(obs_x_all, obs_t_all, bp_x, bp_t)

    # Independent fits
    C_a, G_a = fit_field(Phi_a, u_obs_a)
    C_b, G_b = fit_field(Phi_b, u_obs_b)
    h_a = Phi_a.T @ u_obs_a
    h_b = Phi_b.T @ u_obs_b

    # Centralized fit
    C_central, _ = fit_field(Phi_all, u_obs_all)

    # Gram merge
    C_merged = gram_merge(G_a, h_a, G_b, h_b, reg=1e-4)

    # Merge delta_max: maximum absolute difference between coefficients
    delta_max = torch.max(torch.abs(C_merged - C_central)).item()
    print(f"\n  Merge delta_max (||C_merged - C_central||_inf): {delta_max:.2e}")

    # Evaluate all variants
    print("  Evaluating fields...")
    u_central = eval_field_on_grid(C_central, x_eval, t_eval, bp_x, bp_t)
    u_merged = eval_field_on_grid(C_merged, x_eval, t_eval, bp_x, bp_t)
    u_left = eval_field_on_grid(C_a, x_eval, t_eval, bp_x, bp_t)
    u_right = eval_field_on_grid(C_b, x_eval, t_eval, bp_x, bp_t)

    # Reconstruction comparison
    mid_idx = Nx_eval // 2
    print("\n  Reconstruction comparison:")
    for label, u_hat in [("Centralized", u_central), ("Merged", u_merged),
                          ("Left only", u_left), ("Right only", u_right)]:
        rmse = np.sqrt(np.mean((u_hat - u_true_eval)**2))
        rmse_left = np.sqrt(np.mean((u_hat[:, :mid_idx] -
                                      u_true_eval[:, :mid_idx])**2))
        rmse_right = np.sqrt(np.mean((u_hat[:, mid_idx:] -
                                       u_true_eval[:, mid_idx:])**2))
        print(f"    {label:<16} RMSE: {rmse:.6f}  "
              f"(left={rmse_left:.6f}, right={rmse_right:.6f})")

    # Physics recovery from merged and centralized fields
    print("\n  Physics recovery comparison:")
    merge_results = {}
    for label, u_hat in [("Centralized", u_central), ("Merged", u_merged)]:
        u_t_h, u_x_h, u_xx_h, u_int_h = compute_burgers_derivatives(
            u_hat, dx_eval, dt_eval)
        kappa_h, pde_r = recover_viscosity(u_t_h, u_x_h, u_xx_h, u_int_h)
        kappa_e = abs(kappa_h - KAPPA_TRUE)
        kappa_e_pct = 100.0 * kappa_e / KAPPA_TRUE
        print(f"    {label:<16} kappa={kappa_h:.6f} (err={kappa_e:.6f}, "
              f"{kappa_e_pct:.2f}%, residual={pde_r:.4e})")
        merge_results[label] = {
            'kappa_hat': kappa_h, 'kappa_err': kappa_e,
            'kappa_err_pct': kappa_e_pct, 'pde_rmse': pde_r,
            'recon_rmse': np.sqrt(np.mean((u_hat - u_true_eval)**2)),
        }

    # Store merge experiment results
    results["Merge-Central"] = {
        'n_obs': 5000, 'noise': 0.0,
        'recon_rmse': merge_results["Centralized"]['recon_rmse'],
        'kappa_hat': merge_results["Centralized"]['kappa_hat'],
        'kappa_err': merge_results["Centralized"]['kappa_err'],
        'kappa_err_pct': merge_results["Centralized"]['kappa_err_pct'],
        'pde_rmse': merge_results["Centralized"]['pde_rmse'],
        'delta_max': '',
    }
    results["Merge-Gram"] = {
        'n_obs': 5000, 'noise': 0.0,
        'recon_rmse': merge_results["Merged"]['recon_rmse'],
        'kappa_hat': merge_results["Merged"]['kappa_hat'],
        'kappa_err': merge_results["Merged"]['kappa_err'],
        'kappa_err_pct': merge_results["Merged"]['kappa_err_pct'],
        'pde_rmse': merge_results["Merged"]['pde_rmse'],
        'delta_max': delta_max,
    }

    # ================================================================
    # Summary table
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"\n  True viscosity: kappa = {KAPPA_TRUE}")
    print(f"\n  {'Experiment':<20} {'RMSE':>10} {'kappa_hat':>10} "
          f"{'err%':>8} {'PDE res':>12}")
    print(f"  {'-'*64}")

    for name, r in results.items():
        print(f"  {name:<20} {r['recon_rmse']:>10.6f} "
              f"{r['kappa_hat']:>10.6f} "
              f"{r.get('kappa_err_pct', 0):>7.2f}% "
              f"{r['pde_rmse']:>12.4e}")

    # ================================================================
    # Visualization: 3 rows (time steps) x 4 columns
    # ================================================================
    print("\nGenerating plots...")

    fig, axes = plt.subplots(3, 4, figsize=(16, 9), facecolor='white')
    fig.suptitle(
        r'Viscous Burgers: $u_t + u\,u_x = \kappa\,u_{xx}$'
        f'  ($\\kappa$={KAPPA_TRUE})\n'
        f'Distributed Merge: 2 data holders, Gram statistics, '
        f'$\\Delta_{{max}}$={delta_max:.2e}',
        fontsize=13, fontweight='bold', color='black', y=0.99)

    t_indices = [0, Nt_eval // 2, Nt_eval - 1]
    x_eval_np = x_eval.numpy()

    plot_data = [("True", u_true_eval), ("Centralized", u_central),
                 ("Merged", u_merged), ("|Error|", None)]

    vmin_global = min(u_true_eval.min(), u_central.min(), u_merged.min())
    vmax_global = max(u_true_eval.max(), u_central.max(), u_merged.max())

    for row, ti in enumerate(t_indices):
        t_val = t_eval[ti].item()
        for col, (label, data) in enumerate(plot_data):
            ax = axes[row, col]
            ax.set_facecolor('white')

            if label == "|Error|":
                err = np.abs(u_merged[ti] - u_true_eval[ti])
                ax.fill_between(x_eval_np, 0, err, color='#d32f2f',
                                alpha=0.4, linewidth=0)
                ax.plot(x_eval_np, err, color='#d32f2f', linewidth=1.2)
                ax.set_ylabel('|Error|', fontsize=8)
                ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
            else:
                ax.plot(x_eval_np, data[ti], color='#1565c0', linewidth=1.5)
                ax.set_ylim(vmin_global - 0.05, vmax_global + 0.05)

            if col == 2:  # Merged: show split boundary
                ax.axvline(x=math.pi, color='gray', linewidth=0.8,
                           linestyle='--', alpha=0.6)

            ax.set_title(f'{label} (t={t_val:.2f})', fontsize=9,
                         fontweight='bold', color='black')
            ax.tick_params(labelsize=7)
            ax.set_xlim(x_eval_np[0], x_eval_np[-1])
            ax.grid(True, alpha=0.2, linewidth=0.5)

            if row == 2:
                ax.set_xlabel('x', fontsize=8)
            if col == 0:
                ax.set_ylabel(f'u(x, t={t_val:.2f})', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('merge_pde_burgers.png', dpi=150, facecolor='white',
                bbox_inches='tight')
    print("  Saved: merge_pde_burgers.png")
    plt.close()

    # ================================================================
    # Save CSV
    # ================================================================
    with open('merge_pde_burgers.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['experiment', 'n_obs', 'noise', 'recon_rmse',
                     'kappa_hat', 'kappa_err', 'kappa_err_pct',
                     'pde_rmse', 'delta_max'])
        for name, r in results.items():
            w.writerow([name, r['n_obs'], r['noise'],
                        f"{r['recon_rmse']:.8f}",
                        f"{r['kappa_hat']:.6f}",
                        f"{r['kappa_err']:.6f}",
                        f"{r.get('kappa_err_pct', 0):.4f}",
                        f"{r['pde_rmse']:.8e}",
                        r.get('delta_max', '')])
    print("  Saved: merge_pde_burgers.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
