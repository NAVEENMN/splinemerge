"""
Synthetic PDE Experiment #2: Wave Equation

  u_tt = c^2 (u_xx + u_yy)

Known ground truth: c = 1.0 (wave speed).
Introduces second time derivative, energy-preserving propagation.

Same pipeline: distributed sensors -> spline field -> derivatives -> physics.
Key test: can the spline field's temporal resolution support u_tt recovery?
"""

import torch
import numpy as np
import copy
import math
import csv
import matplotlib.pyplot as plt

from inkan.basis import bspline_basis_eager


def eval_wave_analytic(x, y, t, c):
    """Exact analytic solution for the wave initial condition."""
    return (np.sin(x) * np.cos(y) * np.cos(np.sqrt(2) * c * t)
            + 0.3 * np.sin(2 * x + y) * np.cos(np.sqrt(5) * c * t))


# ============================================================================
# Exact wave equation solver (spectral)
# ============================================================================

def solve_wave_spectral(u0_grid, ut0_grid, c, Lx, Ly, times):
    """Exact solution for u_tt = c^2 (u_xx + u_yy).

    Periodic BCs. Initial conditions: u(0) = u0, u_t(0) = ut0.
    û(t) = û(0) cos(wt) + ût(0) sin(wt)/w, where w = c*|k|.
    Returns [Nt, Nx, Ny].
    """
    Nx, Ny = u0_grid.shape
    dx, dy = Lx / Nx, Ly / Ny

    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    omega = c * np.sqrt(KX**2 + KY**2)
    # Avoid division by zero at k=0
    omega_safe = np.where(omega == 0, 1.0, omega)

    u0_hat = np.fft.fft2(u0_grid)
    ut0_hat = np.fft.fft2(ut0_grid)

    result = np.zeros((len(times), Nx, Ny))
    for i, t in enumerate(times):
        u_hat_t = (u0_hat * np.cos(omega * t) +
                   ut0_hat * np.sin(omega * t) / omega_safe)
        # Fix k=0 mode: u_hat(0,t) = u0_hat(0) + ut0_hat(0)*t
        u_hat_t[0, 0] = u0_hat[0, 0] + ut0_hat[0, 0] * t
        result[i] = np.real(np.fft.ifft2(u_hat_t))

    return result


# ============================================================================
# 3D tensor-product B-spline features (reused from advdiff)
# ============================================================================

def make_basis_params(grid_size, grid_range):
    n_bases = grid_size + 3
    h = (grid_range[1] - grid_range[0]) / grid_size
    inv_h = 1.0 / h
    grid_starts = torch.arange(n_bases).float() * h + grid_range[0] - 3 * h
    return grid_starts, inv_h, n_bases


def eval_1d_basis(x, grid_starts, inv_h):
    try:
        return bspline_basis_eager(x, grid_starts, inv_h, bounded=False).squeeze(1).double()
    except TypeError:
        return bspline_basis_eager(x, grid_starts, inv_h).squeeze(1).double()


@torch.no_grad()
def build_3d_features(x, y, t, bp_x, bp_y, bp_t, batch_size=2000):
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


@torch.no_grad()
def fit_field(Phi, u_obs, reg=1e-4):
    G = Phi.T @ Phi
    h = Phi.T @ u_obs.double()
    P = G.shape[0]
    C = torch.linalg.solve(G + reg * torch.eye(P, dtype=torch.float64), h)
    if not torch.isfinite(C).all():
        raise ValueError(f"fit_field produced non-finite coefficients")
    return C, G


@torch.no_grad()
def eval_field_on_grid(C, x_grid, y_grid, t_grid, bp_x, bp_y, bp_t):
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
# Derivative computation (wave equation needs u_tt)
# ============================================================================

def compute_wave_derivatives(u_grid, dx, dy, dt):
    """Central finite differences including u_tt.

    Returns on interior: [Nt-2, Nx-2, Ny-2].
    """
    # u_tt (second time derivative)
    u_tt = (u_grid[2:, 1:-1, 1:-1] - 2 * u_grid[1:-1, 1:-1, 1:-1] +
            u_grid[:-2, 1:-1, 1:-1]) / dt**2

    # Laplacian
    u_xx = (u_grid[1:-1, 2:, 1:-1] - 2 * u_grid[1:-1, 1:-1, 1:-1] +
            u_grid[1:-1, :-2, 1:-1]) / dx**2
    u_yy = (u_grid[1:-1, 1:-1, 2:] - 2 * u_grid[1:-1, 1:-1, 1:-1] +
            u_grid[1:-1, 1:-1, :-2]) / dy**2

    return u_tt, u_xx, u_yy


def recover_wave_speed(u_tt, u_xx, u_yy):
    """Recover c^2 from u_tt = c^2 (u_xx + u_yy).

    Single parameter: u_tt = a * laplacian, a = c^2.
    """
    laplacian = (u_xx + u_yy).flatten()
    b = u_tt.flatten()

    valid = np.isfinite(laplacian) & np.isfinite(b)
    if valid.sum() < 10:
        return float('nan'), float('nan')
    laplacian, b = laplacian[valid], b[valid]

    # Ridge regression for single parameter
    ATA = np.dot(laplacian, laplacian) + 1e-8
    ATb = np.dot(laplacian, b)
    c_sq = ATb / ATA
    if c_sq < 0:
        print(f"  WARNING: Negative c^2 = {c_sq:.6f}, physically inadmissible")
    c_hat = np.sqrt(abs(c_sq))
    residual = b - c_sq * laplacian
    rmse = np.sqrt(np.mean(residual**2))

    return c_hat, rmse


@torch.no_grad()
def gram_merge(G_a, h_a, G_b, h_b, reg=1e-4):
    P = G_a.shape[0]
    G = G_a + G_b + reg * torch.eye(P, dtype=torch.float64)
    h = h_a + h_b
    C = torch.linalg.solve(G, h)
    if not torch.isfinite(C).all():
        raise ValueError("gram_merge produced non-finite coefficients")
    return C


# ============================================================================
# Main
# ============================================================================

def main():
    C_TRUE = 1.0  # wave speed
    Lx, Ly = 2 * math.pi, 2 * math.pi
    T_max = 3.0  # shorter than advdiff (waves are faster)

    Nx_pde, Ny_pde = 64, 64
    x_pde = np.linspace(0, Lx, Nx_pde, endpoint=False)
    y_pde = np.linspace(0, Ly, Ny_pde, endpoint=False)

    XX, YY = np.meshgrid(x_pde, y_pde, indexing='ij')
    u0 = np.sin(XX) * np.cos(YY) + 0.3 * np.sin(2 * XX + YY)
    ut0 = np.zeros_like(u0)  # start from rest

    print("=" * 70)
    print("Synthetic PDE #2: Wave Equation")
    print(f"  u_tt = {C_TRUE}^2 * (u_xx + u_yy)")
    print(f"  Domain: [0, 2pi]^2 x [0, {T_max}]")
    print(f"  Initial: u_t(0) = 0 (start from rest)")
    print("=" * 70)

    # Solve exactly
    print("\nSolving wave equation (spectral, exact)...")
    Nt_pde = 80  # more time steps for u_tt accuracy
    t_pde = np.linspace(0, T_max, Nt_pde)
    u_true_pde = solve_wave_spectral(u0, ut0, C_TRUE, Lx, Ly, t_pde)
    print(f"  Solution range: [{u_true_pde.min():.3f}, {u_true_pde.max():.3f}]")

    # Spline basis
    gs_x, gs_y, gs_t = 8, 8, 10  # more temporal resolution for u_tt
    bp_x = make_basis_params(gs_x, (0, Lx))
    bp_y = make_basis_params(gs_y, (0, Ly))
    bp_t = make_basis_params(gs_t, (0, T_max))
    Kx, Ky, Kt = bp_x[2], bp_y[2], bp_t[2]
    P = Kx * Ky * Kt
    print(f"  Basis: Kx={Kx}, Ky={Ky}, Kt={Kt}, total features={P}")

    # Evaluation grid
    Nx_eval, Ny_eval, Nt_eval = 40, 40, 40
    x_eval = torch.linspace(0.3, Lx - 0.3, Nx_eval)
    y_eval = torch.linspace(0.3, Ly - 0.3, Ny_eval)
    t_eval = torch.linspace(0.2, T_max - 0.2, Nt_eval)
    dx_e = (x_eval[1] - x_eval[0]).item()
    dy_e = (y_eval[1] - y_eval[0]).item()
    dt_e = (t_eval[1] - t_eval[0]).item()

    # Ground truth on eval grid (analytic)
    print("  Computing ground truth on eval grid...")
    XX_e, YY_e = np.meshgrid(x_eval.numpy(), y_eval.numpy(), indexing='ij')
    u_true_eval = np.zeros((Nt_eval, Nx_eval, Ny_eval))
    for ti in range(Nt_eval):
        u_true_eval[ti] = eval_wave_analytic(XX_e, YY_e, t_eval[ti].item(), C_TRUE)

    # Ground truth derivatives and sanity check
    u_tt_true, u_xx_true, u_yy_true = compute_wave_derivatives(
        u_true_eval, dx_e, dy_e, dt_e)
    c_gt, rmse_gt = recover_wave_speed(u_tt_true, u_xx_true, u_yy_true)
    print(f"\n  Ground truth recovery: c={c_gt:.4f} (true={C_TRUE}), "
          f"RMSE={rmse_gt:.4e}")

    # Helper to get observation values (analytic)
    def get_obs(obs_x, obs_y, obs_t):
        return eval_wave_analytic(obs_x[:, 0].numpy(), obs_y[:, 0].numpy(),
                                  obs_t[:, 0].numpy(), C_TRUE)

    # ================================================================
    # Experiments
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
        print(f"{exp_name} (n={n_obs}, noise={noise_std})")
        print(f"{'='*60}")

        torch.manual_seed(42)
        obs_x = torch.rand(n_obs, 1) * Lx
        obs_y = torch.rand(n_obs, 1) * Ly
        obs_t = torch.rand(n_obs, 1) * T_max

        obs_u = get_obs(obs_x, obs_y, obs_t)
        if noise_std > 0:
            obs_u += np.random.default_rng(42).normal(0, noise_std, n_obs)
        obs_u_t = torch.tensor(obs_u, dtype=torch.float64)

        Phi = build_3d_features(obs_x, obs_y, obs_t, bp_x, bp_y, bp_t)
        C, G = fit_field(Phi, obs_u_t)

        u_hat = eval_field_on_grid(C, x_eval, y_eval, t_eval,
                                    bp_x, bp_y, bp_t)

        recon_rmse = np.sqrt(np.mean((u_hat - u_true_eval)**2))

        u_tt_hat, u_xx_hat, u_yy_hat = compute_wave_derivatives(
            u_hat, dx_e, dy_e, dt_e)
        deriv_rmse_utt = np.sqrt(np.mean((u_tt_hat - u_tt_true)**2))

        c_hat, pde_rmse = recover_wave_speed(u_tt_hat, u_xx_hat, u_yy_hat)
        c_err = abs(c_hat - C_TRUE)

        print(f"  Reconstruction RMSE: {recon_rmse:.6f}")
        print(f"  u_tt RMSE:          {deriv_rmse_utt:.6f}")
        print(f"  Wave speed: c={c_hat:.4f} (true={C_TRUE}, err={c_err:.4f})")
        print(f"  PDE residual RMSE:  {pde_rmse:.4e}")

        results[exp_name] = {
            'n_obs': n_obs, 'noise': noise_std,
            'recon_rmse': recon_rmse,
            'deriv_utt_rmse': deriv_rmse_utt,
            'c_hat': c_hat, 'c_err': c_err,
            'pde_rmse': pde_rmse,
        }

    # ================================================================
    # Distributed merge
    # ================================================================
    print(f"\n{'='*60}")
    print("Distributed Merge (left/right split)")
    print(f"{'='*60}")

    n_per = 5000
    torch.manual_seed(42)
    obs_x_a = torch.rand(n_per, 1) * math.pi
    obs_y_a = torch.rand(n_per, 1) * Ly
    obs_t_a = torch.rand(n_per, 1) * T_max
    obs_x_b = torch.rand(n_per, 1) * math.pi + math.pi
    obs_y_b = torch.rand(n_per, 1) * Ly
    obs_t_b = torch.rand(n_per, 1) * T_max

    u_a = torch.tensor(get_obs(obs_x_a, obs_y_a, obs_t_a), dtype=torch.float64)
    u_b = torch.tensor(get_obs(obs_x_b, obs_y_b, obs_t_b), dtype=torch.float64)

    Phi_a = build_3d_features(obs_x_a, obs_y_a, obs_t_a, bp_x, bp_y, bp_t)
    Phi_b = build_3d_features(obs_x_b, obs_y_b, obs_t_b, bp_x, bp_y, bp_t)

    C_a, G_a = fit_field(Phi_a, u_a)
    C_b, G_b = fit_field(Phi_b, u_b)
    h_a = Phi_a.T @ u_a
    h_b = Phi_b.T @ u_b

    # Centralized
    obs_x_all = torch.cat([obs_x_a, obs_x_b])
    obs_y_all = torch.cat([obs_y_a, obs_y_b])
    obs_t_all = torch.cat([obs_t_a, obs_t_b])
    u_all = torch.cat([u_a, u_b])
    Phi_all = build_3d_features(obs_x_all, obs_y_all, obs_t_all,
                                 bp_x, bp_y, bp_t)
    C_central, _ = fit_field(Phi_all, u_all)

    # Merge
    C_merged = gram_merge(G_a, h_a, G_b, h_b)

    print("\n  Evaluating...")
    for label, C_test in [("Centralized", C_central), ("Merged", C_merged),
                           ("Left only", C_a), ("Right only", C_b)]:
        u_h = eval_field_on_grid(C_test, x_eval, y_eval, t_eval,
                                  bp_x, bp_y, bp_t)
        rmse = np.sqrt(np.mean((u_h - u_true_eval)**2))
        mid = Nx_eval // 2
        rmse_l = np.sqrt(np.mean((u_h[:, :mid, :] - u_true_eval[:, :mid, :])**2))
        rmse_r = np.sqrt(np.mean((u_h[:, mid:, :] - u_true_eval[:, mid:, :])**2))
        print(f"  {label:<16} RMSE: {rmse:.6f} "
              f"(left={rmse_l:.6f}, right={rmse_r:.6f})")

    print("\n  Physics recovery:")
    for label, C_test in [("Centralized", C_central), ("Merged", C_merged)]:
        u_h = eval_field_on_grid(C_test, x_eval, y_eval, t_eval,
                                  bp_x, bp_y, bp_t)
        utt_h, uxx_h, uyy_h = compute_wave_derivatives(u_h, dx_e, dy_e, dt_e)
        c_h, r = recover_wave_speed(utt_h, uxx_h, uyy_h)
        print(f"  {label:<16} c={c_h:.4f} (err={abs(c_h-C_TRUE):.4f}, "
              f"residual={r:.4e})")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  True wave speed: c = {C_TRUE}")
    print(f"\n  {'Experiment':<20} {'Recon RMSE':>11} {'c_hat':>8} {'c_err':>8}")
    print(f"  {'-'*50}")
    for name, r in results.items():
        print(f"  {name:<20} {r['recon_rmse']:>11.6f} {r['c_hat']:>8.4f} "
              f"{r['c_err']:>8.4f}")

    # ================================================================
    # Visualization
    # ================================================================
    print("\nGenerating plots...")

    # Reconstruct merged field for plotting
    u_merged_grid = eval_field_on_grid(C_merged, x_eval, y_eval, t_eval,
                                        bp_x, bp_y, bp_t)

    fig, axes = plt.subplots(3, 4, figsize=(18, 12), facecolor='#1a1a2e')
    fig.suptitle(f'Wave Equation: u_tt = {C_TRUE}^2 (u_xx + u_yy)\n'
                 f'Distributed Merge: 2 sensors, Gram statistics',
                 fontsize=14, fontweight='bold', color='white', y=0.98)

    t_indices = [0, Nt_eval // 2, Nt_eval - 1]
    for row, ti in enumerate(t_indices):
        for col, (label, data) in enumerate([
            ("True", u_true_eval), ("Centralized", eval_field_on_grid(
                C_central, x_eval, y_eval, t_eval, bp_x, bp_y, bp_t)),
            ("Merged", u_merged_grid),
            ("|Error|", None),
        ]):
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
                vmin, vmax = u_true_eval.min(), u_true_eval.max()
                im = ax.imshow(data[ti].T, origin='lower', cmap='RdBu_r',
                               extent=[x_eval[0], x_eval[-1],
                                       y_eval[0], y_eval[-1]],
                               aspect='auto', vmin=vmin, vmax=vmax)
                if col == 0:
                    plt.colorbar(im, ax=ax, fraction=0.046)
            if col == 2:
                ax.axvline(x=math.pi, color='white', linewidth=0.5,
                           linestyle=':', alpha=0.5)
            ax.set_title(f'{label} (t={t_eval[ti]:.1f})', fontsize=10,
                         fontweight='bold', color='white')
            ax.tick_params(colors='#666', labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#30363d')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('merge_pde_wave.png', dpi=150, facecolor='#1a1a2e',
                bbox_inches='tight')
    print("  Saved: merge_pde_wave.png")
    plt.close()

    # CSV
    with open('merge_pde_wave.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['experiment', 'n_obs', 'noise', 'recon_rmse',
                     'c_hat', 'c_err', 'pde_rmse'])
        for name, r in results.items():
            w.writerow([name, r['n_obs'], r['noise'],
                        f"{r['recon_rmse']:.8f}", f"{r['c_hat']:.6f}",
                        f"{r['c_err']:.6f}", f"{r['pde_rmse']:.8e}"])
    print("  Saved: merge_pde_wave.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
