"""
Derivative verification: separate FD-grid error from spline-fit error.

The paper claims sub-percent parameter recovery. This script decomposes
the total error into two independent sources:

  1. FD-grid error -- applying finite-difference stencils to the EXACT
     analytic field on the evaluation grid (no spline involved).
  2. Full pipeline error -- FD stencils applied to the spline-fitted
     field (what the paper actually reports).

The difference between (2) and (1) isolates the spline contribution.

Uses identical grid parameters, linspace ranges, and FD stencils as
merge_pde_wave.py and merge_pde_diffusion.py.  Self-contained: no
imports from src/ or inkan.
"""

import numpy as np
import math


# =====================================================================
# Wave equation: closed-form analytic solution
# =====================================================================

def eval_wave_analytic(x, y, t, c=1.0):
    """Exact wave solution: two standing-wave modes, zero initial velocity.

    u(x,y,t) = sin(x)*cos(y)*cos(sqrt(2)*c*t)
              + 0.3*sin(2x+y)*cos(sqrt(5)*c*t)
    """
    return (np.sin(x) * np.cos(y) * np.cos(np.sqrt(2) * c * t)
            + 0.3 * np.sin(2 * x + y) * np.cos(np.sqrt(5) * c * t))


# =====================================================================
# Diffusion equation: spectral (FFT-decay) exact solver
# =====================================================================

def solve_diffusion_spectral(u0_grid, D, Lx, Ly, times):
    """Exact diffusion solution via Fourier-mode exponential decay.

    Periodic BCs on [0, Lx] x [0, Ly].
    Each mode: u_hat(k, t) = u_hat(k, 0) * exp(-D |k|^2 t).
    Returns array of shape [len(times), Nx, Ny].
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


def eval_diffusion_on_grid(x_eval, y_eval, t_eval, u0_grid, D, Lx, Ly,
                           x_pde, y_pde):
    """Evaluate the exact diffusion field on an arbitrary eval grid.

    The spectral solver lives on the PDE grid (x_pde, y_pde).  Cubic
    interpolation maps from that grid onto the eval coordinates.
    Periodic wrapping is applied.
    """
    from scipy.interpolate import RegularGridInterpolator

    Nt = len(t_eval)
    Nx_e, Ny_e = len(x_eval), len(y_eval)
    XX_e, YY_e = np.meshgrid(x_eval, y_eval, indexing='ij')

    u_out = np.zeros((Nt, Nx_e, Ny_e))
    for ti in range(Nt):
        u_sol = solve_diffusion_spectral(u0_grid, D, Lx, Ly,
                                         [t_eval[ti]])[0]
        # Extend grid for periodic interpolation
        x_ext = np.append(x_pde, Lx)
        u_ext = np.concatenate([u_sol, u_sol[0:1, :]], axis=0)
        y_ext = np.append(y_pde, Ly)
        u_ext = np.concatenate([u_ext, u_ext[:, 0:1]], axis=1)

        interp = RegularGridInterpolator(
            (x_ext, y_ext), u_ext, method='cubic',
            bounds_error=False, fill_value=None)
        xx_w = XX_e.flatten() % Lx
        yy_w = YY_e.flatten() % Ly
        pts = np.stack([xx_w, yy_w], axis=1)
        u_out[ti] = interp(pts).reshape(Nx_e, Ny_e)
    return u_out


# =====================================================================
# Finite-difference stencils (identical to experiment scripts)
# =====================================================================

def fd_wave_derivatives(u, dx, dy, dt):
    """Second-order central differences for the wave equation.

    u_tt = (u[t+1] - 2*u[t] + u[t-1]) / dt**2
    u_xx = (u[x+1] - 2*u[x] + u[x-1]) / dx**2
    u_yy = (u[y+1] - 2*u[y] + u[y-1]) / dy**2

    Interior points only -> shape [Nt-2, Nx-2, Ny-2].
    """
    u_tt = (u[2:, 1:-1, 1:-1] - 2 * u[1:-1, 1:-1, 1:-1]
            + u[:-2, 1:-1, 1:-1]) / dt**2
    u_xx = (u[1:-1, 2:, 1:-1] - 2 * u[1:-1, 1:-1, 1:-1]
            + u[1:-1, :-2, 1:-1]) / dx**2
    u_yy = (u[1:-1, 1:-1, 2:] - 2 * u[1:-1, 1:-1, 1:-1]
            + u[1:-1, 1:-1, :-2]) / dy**2
    return u_tt, u_xx, u_yy


def fd_diffusion_derivatives(u, dx, dy, dt):
    """Second-order central differences for the diffusion equation.

    u_t  = (u[t+1] - u[t-1]) / (2*dt)
    u_xx = (u[x+1] - 2*u[x] + u[x-1]) / dx**2
    u_yy = (u[y+1] - 2*u[y] + u[y-1]) / dy**2

    Interior points only -> shape [Nt-2, Nx-2, Ny-2].
    """
    u_t = (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2 * dt)
    u_xx = (u[1:-1, 2:, 1:-1] - 2 * u[1:-1, 1:-1, 1:-1]
            + u[1:-1, :-2, 1:-1]) / dx**2
    u_yy = (u[1:-1, 1:-1, 2:] - 2 * u[1:-1, 1:-1, 1:-1]
            + u[1:-1, 1:-1, :-2]) / dy**2
    return u_t, u_xx, u_yy


# =====================================================================
# Parameter recovery (single-parameter ridge regression)
# =====================================================================

def recover_wave_speed(u_tt, u_xx, u_yy):
    """Recover c from  u_tt = c^2 (u_xx + u_yy).

    Returns (c_hat, residual_rmse).
    """
    lap = (u_xx + u_yy).flatten()
    b = u_tt.flatten()
    valid = np.isfinite(lap) & np.isfinite(b)
    lap, b = lap[valid], b[valid]

    ATA = np.dot(lap, lap) + 1e-8
    ATb = np.dot(lap, b)
    c_sq = ATb / ATA
    c_hat = np.sqrt(abs(c_sq))

    residual = b - c_sq * lap
    rmse = np.sqrt(np.mean(residual**2))
    return c_hat, rmse


def recover_diffusion_coeff(u_t, u_xx, u_yy):
    """Recover D from  u_t = D (u_xx + u_yy).

    Returns (D_hat, residual_rmse).
    """
    lap = (u_xx + u_yy).flatten()
    b = u_t.flatten()
    valid = np.isfinite(lap) & np.isfinite(b)
    lap, b = lap[valid], b[valid]

    ATA = np.dot(lap, lap) + 1e-8
    ATb = np.dot(lap, b)
    D_hat = ATb / ATA

    residual = b - D_hat * lap
    rmse = np.sqrt(np.mean(residual**2))
    return D_hat, rmse


# =====================================================================
# Main verification
# =====================================================================

def main():
    Lx = Ly = 2 * math.pi

    # =================================================================
    # 1. WAVE EQUATION -- exact analytic field on the eval grid
    # =================================================================
    C_TRUE_W = 1.0
    T_max_w = 3.0

    # Evaluation grid (same as merge_pde_wave.py)
    Nx_eval, Ny_eval, Nt_eval_w = 40, 40, 40
    x_eval = np.linspace(0.3, Lx - 0.3, Nx_eval)
    y_eval = np.linspace(0.3, Ly - 0.3, Ny_eval)
    t_eval_w = np.linspace(0.2, T_max_w - 0.2, Nt_eval_w)

    dx_e = x_eval[1] - x_eval[0]
    dy_e = y_eval[1] - y_eval[0]
    dt_e_w = t_eval_w[1] - t_eval_w[0]

    print("=" * 72)
    print("DERIVATIVE VERIFICATION: Separating FD-grid error from spline error")
    print("=" * 72)

    # -- Build exact wave field on eval grid --
    print("\n[1] Wave equation  u_tt = c^2 (u_xx + u_yy),  c_true = 1.0")
    print(f"    Eval grid: Nx={Nx_eval}, Ny={Ny_eval}, Nt={Nt_eval_w}")
    print(f"    x in [{x_eval[0]:.2f}, {x_eval[-1]:.2f}], "
          f"y in [{y_eval[0]:.2f}, {y_eval[-1]:.2f}], "
          f"t in [{t_eval_w[0]:.2f}, {t_eval_w[-1]:.2f}]")
    print(f"    dx={dx_e:.5f}, dy={dy_e:.5f}, dt={dt_e_w:.5f}")

    XX, YY = np.meshgrid(x_eval, y_eval, indexing='ij')
    u_exact_w = np.zeros((Nt_eval_w, Nx_eval, Ny_eval))
    for ti in range(Nt_eval_w):
        u_exact_w[ti] = eval_wave_analytic(XX, YY, t_eval_w[ti], C_TRUE_W)

    # -- FD on exact field --
    u_tt_ex, u_xx_ex, u_yy_ex = fd_wave_derivatives(u_exact_w, dx_e, dy_e,
                                                      dt_e_w)
    c_fd_exact, rmse_fd_exact = recover_wave_speed(u_tt_ex, u_xx_ex, u_yy_ex)
    c_err_fd_exact = abs(c_fd_exact - C_TRUE_W)
    c_pct_fd_exact = c_err_fd_exact / C_TRUE_W * 100

    print(f"\n    FD on exact analytic field:")
    print(f"      c_hat      = {c_fd_exact:.6f}")
    print(f"      abs error  = {c_err_fd_exact:.6f}")
    print(f"      pct error  = {c_pct_fd_exact:.4f}%")
    print(f"      residual   = {rmse_fd_exact:.4e}")
    n_interior_w = (Nt_eval_w - 2) * (Nx_eval - 2) * (Ny_eval - 2)
    print(f"      interior pts used: {n_interior_w}")

    # =================================================================
    # 2. DIFFUSION EQUATION -- spectral exact field on the eval grid
    # =================================================================
    D_TRUE = 0.05
    T_max_d = 2.0
    Nt_eval_d = 30

    t_eval_d = np.linspace(0.2, T_max_d - 0.2, Nt_eval_d)
    dt_e_d = t_eval_d[1] - t_eval_d[0]

    # PDE grid for spectral solve (same as merge_pde_diffusion.py)
    Nx_pde, Ny_pde = 64, 64
    x_pde = np.linspace(0, Lx, Nx_pde, endpoint=False)
    y_pde = np.linspace(0, Ly, Ny_pde, endpoint=False)
    XX_pde, YY_pde = np.meshgrid(x_pde, y_pde, indexing='ij')

    # Initial condition (same as merge_pde_diffusion.py)
    u0_diff = (np.sin(XX_pde) * np.cos(YY_pde)
               + 0.5 * np.exp(-3 * ((XX_pde - math.pi)**2
                                     + (YY_pde - math.pi)**2))
               + 0.3 * np.sin(2 * XX_pde + YY_pde))

    print(f"\n[2] Diffusion equation  u_t = D (u_xx + u_yy),  D_true = {D_TRUE}")
    print(f"    Eval grid: Nx={Nx_eval}, Ny={Ny_eval}, Nt={Nt_eval_d}")
    print(f"    x in [{x_eval[0]:.2f}, {x_eval[-1]:.2f}], "
          f"y in [{y_eval[0]:.2f}, {y_eval[-1]:.2f}], "
          f"t in [{t_eval_d[0]:.2f}, {t_eval_d[-1]:.2f}]")
    print(f"    dx={dx_e:.5f}, dy={dy_e:.5f}, dt={dt_e_d:.5f}")

    # -- Build exact diffusion field on eval grid --
    u_exact_d = eval_diffusion_on_grid(
        x_eval, y_eval, t_eval_d, u0_diff, D_TRUE, Lx, Ly, x_pde, y_pde)

    # -- FD on exact field --
    u_t_ex, u_xx_d_ex, u_yy_d_ex = fd_diffusion_derivatives(
        u_exact_d, dx_e, dy_e, dt_e_d)
    D_fd_exact, rmse_d_exact = recover_diffusion_coeff(
        u_t_ex, u_xx_d_ex, u_yy_d_ex)
    D_err_fd_exact = abs(D_fd_exact - D_TRUE)
    D_pct_fd_exact = D_err_fd_exact / D_TRUE * 100

    print(f"\n    FD on exact spectral field:")
    print(f"      D_hat      = {D_fd_exact:.6f}")
    print(f"      abs error  = {D_err_fd_exact:.6f}")
    print(f"      pct error  = {D_pct_fd_exact:.4f}%")
    print(f"      residual   = {rmse_d_exact:.4e}")
    n_interior_d = (Nt_eval_d - 2) * (Nx_eval - 2) * (Ny_eval - 2)
    print(f"      interior pts used: {n_interior_d}")

    # =================================================================
    # 3. COMPARISON TABLE
    # =================================================================

    # Placeholder values for the spline-fitted pipeline.
    # These come from the experiment scripts (merge_pde_wave.py and
    # merge_pde_diffusion.py Dense 10K noiseless experiments).  If
    # the experiment CSVs are available, read them; otherwise print
    # the grid-only results and note that pipeline values should be
    # filled from the experiment outputs.

    c_spline, D_spline = None, None
    c_spline_label = "(not available -- run merge_pde_wave.py)"
    D_spline_label = "(not available -- run merge_pde_diffusion.py)"

    import os
    wave_csv = os.path.join(os.path.dirname(__file__), "..",
                            "..", "merge_pde_wave.csv")
    diff_csv = os.path.join(os.path.dirname(__file__), "..",
                            "..", "merge_pde_diffusion.csv")

    if os.path.isfile(wave_csv):
        import csv
        with open(wave_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "Dense (10K)" in row.get("experiment", ""):
                    try:
                        c_spline = float(row["c_hat"])
                        c_spline_label = f"{c_spline:.6f}"
                    except (KeyError, ValueError):
                        pass
                    break

    if os.path.isfile(diff_csv):
        import csv
        with open(diff_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "Dense (10K)" in row.get("experiment", ""):
                    try:
                        D_spline = float(row["D_hat"])
                        D_spline_label = f"{D_spline:.6f}"
                    except (KeyError, ValueError):
                        pass
                    break

    # -- Print comparison table --
    print("\n" + "=" * 72)
    print("COMPARISON TABLE: Error-source decomposition")
    print("=" * 72)

    hdr = (f"  {'Source':<38} {'Param':>8} {'True':>8} "
           f"{'Recovered':>10} {'Err%':>8}")
    sep = "  " + "-" * 68
    print(hdr)
    print(sep)

    # Wave: FD grid only
    print(f"  {'Wave: FD on exact field (grid err)':<38} {'c':>8} "
          f"{C_TRUE_W:>8.4f} {c_fd_exact:>10.6f} "
          f"{c_pct_fd_exact:>7.4f}%")

    # Wave: full pipeline (spline + FD)
    if c_spline is not None:
        c_pct_spline = abs(c_spline - C_TRUE_W) / C_TRUE_W * 100
        print(f"  {'Wave: FD on spline fit (pipeline err)':<38} {'c':>8} "
              f"{C_TRUE_W:>8.4f} {c_spline:>10.6f} "
              f"{c_pct_spline:>7.4f}%")
        c_spline_contrib = c_pct_spline - c_pct_fd_exact
        print(f"  {'  -> spline contribution':<38} {'':>8} "
              f"{'':>8} {'':>10} "
              f"{c_spline_contrib:>+7.4f}%")
    else:
        print(f"  {'Wave: FD on spline fit (pipeline err)':<38} {'c':>8} "
              f"{C_TRUE_W:>8.4f} {c_spline_label:>10}")

    print(sep)

    # Diffusion: FD grid only
    print(f"  {'Diff: FD on exact field (grid err)':<38} {'D':>8} "
          f"{D_TRUE:>8.4f} {D_fd_exact:>10.6f} "
          f"{D_pct_fd_exact:>7.4f}%")

    # Diffusion: full pipeline (spline + FD)
    if D_spline is not None:
        D_pct_spline = abs(D_spline - D_TRUE) / D_TRUE * 100
        print(f"  {'Diff: FD on spline fit (pipeline err)':<38} {'D':>8} "
              f"{D_TRUE:>8.4f} {D_spline:>10.6f} "
              f"{D_pct_spline:>7.4f}%")
        D_spline_contrib = D_pct_spline - D_pct_fd_exact
        print(f"  {'  -> spline contribution':<38} {'':>8} "
              f"{'':>8} {'':>10} "
              f"{D_spline_contrib:>+7.4f}%")
    else:
        print(f"  {'Diff: FD on spline fit (pipeline err)':<38} {'D':>8} "
              f"{D_TRUE:>8.4f} {D_spline_label:>10}")

    print(sep)

    # -- Interpretation --
    print("\nInterpretation:")
    print(f"  The FD grid alone introduces {c_pct_fd_exact:.4f}% error "
          f"in wave-speed recovery")
    print(f"  and {D_pct_fd_exact:.4f}% error in diffusion-coefficient "
          f"recovery.")
    print("  Any reported sub-percent error from the full pipeline must")
    print("  be understood as a combination of these two distinct sources:")
    print("    (a) finite-difference truncation error on the evaluation grid")
    print("    (b) spline approximation error in the fitted field")
    print("  The table above separates them.")


if __name__ == "__main__":
    main()
