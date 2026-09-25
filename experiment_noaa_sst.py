"""
NOAA SST Field Reconstruction: real spatiotemporal data.

Applies the validated PDE pipeline to NOAA OI SST V2 monthly data.
Region: tropical Pacific (30S-30N, 120E-280E).
Hybrid basis: spatial B-splines x temporal Fourier+trend.

Decomposition:
  SST(lat, lon, t) = mean(lat,lon)
                    + A_1(lat,lon) cos(w_yr t) + B_1(lat,lon) sin(w_yr t)
                    + A_2(lat,lon) cos(2 w_yr t) + B_2(lat,lon) sin(2 w_yr t)
                    + trend(lat,lon) * t + accel(lat,lon) * t^2

Merge: split west Pacific (120-200E) vs east Pacific (200-280E).
"""

import torch
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import netCDF4 as nc

from inkan.basis import bspline_basis_eager


# ============================================================================
# Basis helpers
# ============================================================================

def make_basis_params(grid_size, grid_range):
    n_bases = grid_size + 3
    h = (grid_range[1] - grid_range[0]) / grid_size
    inv_h = 1.0 / h
    grid_starts = torch.arange(n_bases, dtype=torch.float64) * h + grid_range[0] - 3 * h
    return grid_starts, inv_h, n_bases


def eval_1d_basis(x, grid_starts, inv_h):
    return bspline_basis_eager(x, grid_starts, inv_h).squeeze(1).double()


# ============================================================================
# Temporal features (Fourier + trend)
# ============================================================================

def temporal_features(t_months, t_max=None):
    """Compute temporal feature vector for each time point.

    Parameters
    ----------
    t_months : array-like
        Month indices (0-based from start of record).
    t_max : float, optional
        Fixed normalisation denominator for the trend terms.  When ``None``
        (the default), ``max(t.max(), 1)`` is used, which makes the
        normalisation data-dependent.  Callers that need reproducible
        features across different subsets should pass a fixed value
        computed once from the full time range.

    Returns
    -------
    ndarray, shape [N_t, 7]
        Columns: [1, cos(wt), sin(wt), cos(2wt), sin(2wt), t_norm, t_norm**2]
    """
    t = np.asarray(t_months, dtype=np.float64)
    w = 2 * np.pi / 12.0  # annual cycle in months
    if t_max is None:
        t_max = max(t.max(), 1)
    t_norm = t / t_max

    feats = np.stack([
        np.ones_like(t),         # mean/bias
        np.cos(w * t),           # annual cos
        np.sin(w * t),           # annual sin
        np.cos(2 * w * t),       # semi-annual cos
        np.sin(2 * w * t),       # semi-annual sin
        t_norm,                  # linear trend
        t_norm**2,               # quadratic trend
    ], axis=1)
    return feats  # [N_t, 7]


# ============================================================================
# Build spatial x temporal features (efficient batch by time step)
# ============================================================================

@torch.no_grad()
def build_spatial_features(lat_pts, lon_pts, bp_lat, bp_lon):
    """Precompute spatial B-spline features for all ocean points.

    Returns: [N_ocean, K_lat * K_lon] in float64.
    """
    lat_t = torch.tensor(lat_pts, dtype=torch.float32).unsqueeze(1)
    lon_t = torch.tensor(lon_pts, dtype=torch.float32).unsqueeze(1)

    b_lat = eval_1d_basis(lat_t, bp_lat[0], bp_lat[1])  # [N, K_lat]
    b_lon = eval_1d_basis(lon_t, bp_lon[0], bp_lon[1])  # [N, K_lon]

    # Outer product: [N, K_lat, K_lon] -> [N, K_lat*K_lon]
    phi_s = (b_lat[:, :, None] * b_lon[:, None, :]).reshape(len(lat_pts), -1)
    return phi_s


def accumulate_gram(phi_s, t_feats, sst_values, ocean_mask_per_t):
    """Accumulate Gram matrix and RHS across time steps.

    phi_s: [N_spatial_max, K_s] precomputed spatial features
    t_feats: [N_t, K_t] temporal features
    sst_values: [N_t, N_lat, N_lon] SST data
    ocean_mask_per_t: [N_t, N_spatial_max] boolean (valid ocean points)

    Features: phi(lat,lon,t) = phi_s(lat,lon) ⊗ phi_t(t)
    Total features P = K_s * K_t
    """
    K_s = phi_s.shape[1]
    K_t = t_feats.shape[1]
    P = K_s * K_t

    G = np.zeros((P, P), dtype=np.float64)
    h = np.zeros(P, dtype=np.float64)
    n_obs_total = 0

    N_t = len(t_feats)
    for ti in range(N_t):
        mask = ocean_mask_per_t[ti]
        if mask.sum() == 0:
            continue

        phi_s_valid = phi_s[mask].numpy()  # [n_valid, K_s]
        sst_valid = sst_values[ti][mask]   # [n_valid]
        ft = t_feats[ti]                   # [K_t]

        # Full feature: phi_s ⊗ ft -> [n_valid, P]
        phi_full = (phi_s_valid[:, :, None] * ft[None, None, :]).reshape(-1, P)

        G += phi_full.T @ phi_full
        h += phi_full.T @ sst_valid
        n_obs_total += mask.sum()

        if (ti + 1) % 100 == 0:
            print(f"    {ti+1}/{N_t} time steps processed "
                  f"({n_obs_total:,} obs)")

    print(f"    Total: {n_obs_total:,} observations")
    return G, h, n_obs_total


def solve_field(G, h, reg=1e-2):
    """Solve (G + reg*I) C = h."""
    P = G.shape[0]
    C = np.linalg.solve(G + reg * np.eye(P), h)
    return C


def eval_field(C, phi_s, t_feat, K_s, K_t):
    """Evaluate field at all spatial points for one time step.

    Returns: [N_spatial] values.
    """
    P = K_s * K_t
    phi_s_np = phi_s.numpy() if torch.is_tensor(phi_s) else phi_s
    phi_full = (phi_s_np[:, :, None] * t_feat[None, None, :]).reshape(-1, P)
    return phi_full @ C


# ============================================================================
# Decomposition extraction
# ============================================================================

def extract_seasonal_maps(C, phi_s, K_s, K_t):
    """Extract interpretable spatial fields from coefficient vector.

    C is ordered as [s0*t0, s0*t1, ..., s0*tK, s1*t0, ...] where
    the temporal features are:
      0: bias (mean)
      1: cos(wt) -> annual cos amplitude
      2: sin(wt) -> annual sin amplitude
      3: cos(2wt) -> semi-annual cos
      4: sin(2wt) -> semi-annual sin
      5: linear trend
      6: quadratic trend

    Returns dict of spatial maps.
    """
    phi_s_np = phi_s.numpy() if torch.is_tensor(phi_s) else phi_s
    C_reshaped = C.reshape(K_s, K_t)  # [K_s, K_t]

    maps = {}
    labels = ['mean', 'annual_cos', 'annual_sin',
              'semiannual_cos', 'semiannual_sin',
              'linear_trend', 'quadratic_trend']

    for j, label in enumerate(labels):
        maps[label] = phi_s_np @ C_reshaped[:, j]  # [N_spatial]

    # Derived: annual amplitude and phase
    A1 = maps['annual_cos']
    B1 = maps['annual_sin']
    maps['annual_amplitude'] = np.sqrt(A1**2 + B1**2)
    maps['annual_phase'] = np.arctan2(B1, A1)  # radians, convert to month
    maps['annual_phase_month'] = maps['annual_phase'] * 12 / (2 * np.pi)

    return maps


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("NOAA SST: Tropical Pacific Field Reconstruction")
    print("  Hybrid basis: spatial B-splines x temporal Fourier+trend")
    print("=" * 70)

    # Load data
    print("\nLoading NOAA OI SST V2...")
    ds = nc.Dataset("data/noaa/sst.mnmean.nc")
    lat = ds.variables['lat'][:].data
    lon = ds.variables['lon'][:].data
    # Preserve the netCDF masked-array so fill-values become NaN
    sst_full = np.ma.asarray(
        ds.variables['sst'][:], dtype=np.float64
    ).filled(np.nan)  # [N_t_full, 180, 360]
    times = nc.num2date(ds.variables['time'][:],
                        ds.variables['time'].units)
    ds.close()

    # Load authoritative NOAA land-sea mask
    # (NOAA OI SST V2 fills land cells by interpolation, so finite
    # values alone do not distinguish ocean from land.)
    print("  Loading land-sea mask (lsmask.nc)...")
    ds_mask = nc.Dataset("data/noaa/lsmask.nc")
    lsmask_full = ds_mask.variables['mask'][0]  # [180, 360], 1=ocean 0=land
    ds_mask.close()

    # Select tropical Pacific
    lat_min, lat_max = -30, 30
    lon_min, lon_max = 120, 280  # 120E to 80W
    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    lon_mask = (lon >= lon_min) & (lon <= lon_max)

    lat_sel = lat[lat_mask]
    lon_sel = lon[lon_mask]
    sst_region = sst_full[:, lat_mask, :][:, :, lon_mask]  # [N_t, N_lat, N_lon]
    lsmask_region = lsmask_full[lat_mask, :][:, lon_mask]  # [N_lat, N_lon]

    N_t, N_lat, N_lon = sst_region.shape
    print(f"  Region: lat [{lat_sel[0]:.1f}, {lat_sel[-1]:.1f}], "
          f"lon [{lon_sel[0]:.1f}, {lon_sel[-1]:.1f}]")
    print(f"  Shape: {N_t} months x {N_lat} lat x {N_lon} lon")
    print(f"  Time: {times[0]} to {times[-1]}")

    # Build ocean mask from the land-sea mask AND finite SST check
    ocean_mask_2d = (lsmask_region == 1) & np.isfinite(sst_region[0])
    n_ocean = ocean_mask_2d.sum()
    print(f"  Ocean points: {n_ocean} of {N_lat * N_lon} "
          f"(land-sea mask filtered)")

    # Flatten spatial coordinates for ocean points
    LAT_grid, LON_grid = np.meshgrid(lat_sel, lon_sel, indexing='ij')
    ocean_lats = LAT_grid[ocean_mask_2d]  # [n_ocean]
    ocean_lons = LON_grid[ocean_mask_2d]

    # Per-time-step ocean mask (some points may become NaN in winter)
    sst_flat = sst_region.reshape(N_t, N_lat * N_lon)  # [N_t, N_lat*N_lon]
    ocean_mask_flat_2d = ocean_mask_2d.flatten()  # [N_lat*N_lon]

    # For each time step, which ocean points are valid
    ocean_idx = np.where(ocean_mask_flat_2d)[0]
    ocean_mask_per_t = np.zeros((N_t, n_ocean), dtype=bool)
    sst_ocean = np.zeros((N_t, n_ocean))
    for ti in range(N_t):
        vals = sst_flat[ti, ocean_idx]
        valid = np.isfinite(vals)
        ocean_mask_per_t[ti] = valid
        sst_ocean[ti, valid] = vals[valid]

    print(f"  Valid obs per time step: min={ocean_mask_per_t.sum(1).min()}, "
          f"max={ocean_mask_per_t.sum(1).max()}")

    # Basis parameters
    gs_lat = 10  # grid_size for latitude
    gs_lon = 20  # grid_size for longitude (wider range)
    bp_lat = make_basis_params(gs_lat, (lat_min, lat_max))
    bp_lon = make_basis_params(gs_lon, (lon_min, lon_max))
    K_lat, K_lon = bp_lat[2], bp_lon[2]
    K_s = K_lat * K_lon
    K_t = 7  # temporal features
    P = K_s * K_t
    print(f"\n  Basis: K_lat={K_lat}, K_lon={K_lon}, K_s={K_s}, K_t={K_t}")
    print(f"  Total features: {P}")

    # Precompute spatial features for ocean points
    print("  Computing spatial features...")
    phi_s = build_spatial_features(ocean_lats, ocean_lons, bp_lat, bp_lon)
    print(f"  Spatial features: {phi_s.shape}")

    # Temporal features -- use a fixed t_max so normalisation is
    # deterministic regardless of which subset of months is evaluated.
    month_indices = np.arange(N_t, dtype=np.float64)
    t_max_global = float(max(month_indices.max(), 1))
    t_feats = temporal_features(month_indices, t_max=t_max_global)  # [N_t, 7]
    print(f"  Temporal features: {t_feats.shape} (t_max={t_max_global:.0f})")

    # ================================================================
    # Fit centralized (all data)
    # ================================================================
    print("\n--- Centralized fit (all months, full region) ---")
    G_all, h_all, n_all = accumulate_gram(phi_s, t_feats, sst_ocean,
                                           ocean_mask_per_t)
    C_all = solve_field(G_all, h_all, reg=1e-1)
    print(f"  Coefficients range: [{C_all.min():.2f}, {C_all.max():.2f}]")

    # Reconstruction error
    recon_errors = []
    for ti in range(N_t):
        mask = ocean_mask_per_t[ti]
        if mask.sum() == 0:
            continue
        pred = eval_field(C_all, phi_s[mask], t_feats[ti], K_s, K_t)
        true = sst_ocean[ti, mask]
        recon_errors.append(np.mean((pred - true)**2))
    rmse_all = np.sqrt(np.mean(recon_errors))
    print(f"  Reconstruction RMSE: {rmse_all:.4f} deg C")

    # ================================================================
    # Distributed merge: west vs east Pacific
    # ================================================================
    print("\n--- Distributed merge: West (120-200E) vs East (200-280E) ---")
    lon_split = 200.0

    west_mask = ocean_lons < lon_split  # [n_ocean] boolean
    east_mask = ~west_mask
    print(f"  West ocean points: {west_mask.sum()}, East: {east_mask.sum()}")

    # West sensor: only sees west ocean points
    phi_s_w = phi_s[west_mask]
    ocean_mask_w = ocean_mask_per_t[:, west_mask]
    sst_w = sst_ocean[:, west_mask]

    # East sensor
    phi_s_e = phi_s[east_mask]
    ocean_mask_e = ocean_mask_per_t[:, east_mask]
    sst_e = sst_ocean[:, east_mask]

    print("  Fitting west sensor...")
    G_w, h_w, n_w = accumulate_gram(phi_s_w, t_feats, sst_w, ocean_mask_w)
    C_w = solve_field(G_w, h_w, reg=1e-1)

    print("  Fitting east sensor...")
    G_e, h_e, n_e = accumulate_gram(phi_s_e, t_feats, sst_e, ocean_mask_e)
    C_e = solve_field(G_e, h_e, reg=1e-1)

    # Gram merge
    C_merged = solve_field(G_w + G_e, h_w + h_e, reg=1e-1)

    # Evaluate all three
    print("\n  Reconstruction RMSE (deg C):")
    for label, C_test in [("Centralized", C_all), ("Merged", C_merged),
                           ("West only", C_w), ("East only", C_e)]:
        errors_overall, errors_west, errors_east = [], [], []
        for ti in range(N_t):
            mask = ocean_mask_per_t[ti]
            if mask.sum() == 0:
                continue
            pred = eval_field(C_test, phi_s[mask], t_feats[ti], K_s, K_t)
            true = sst_ocean[ti, mask]
            err = (pred - true)**2

            errors_overall.append(err.mean())

            # Split by region
            w_sub = west_mask[mask]
            e_sub = east_mask[mask]
            if w_sub.sum() > 0:
                errors_west.append(err[w_sub].mean())
            if e_sub.sum() > 0:
                errors_east.append(err[e_sub].mean())

        rmse_o = np.sqrt(np.mean(errors_overall))
        rmse_w = np.sqrt(np.mean(errors_west))
        rmse_e = np.sqrt(np.mean(errors_east))
        print(f"  {label:<16} overall={rmse_o:.4f}  "
              f"west={rmse_w:.4f}  east={rmse_e:.4f}")

    # ================================================================
    # Seasonal decomposition
    # ================================================================
    print("\n--- Seasonal decomposition ---")
    maps_all = extract_seasonal_maps(C_all, phi_s, K_s, K_t)
    maps_merged = extract_seasonal_maps(C_merged, phi_s, K_s, K_t)

    print(f"  Mean SST range: [{maps_all['mean'].min():.1f}, "
          f"{maps_all['mean'].max():.1f}] deg C")
    print(f"  Annual amplitude range: [{maps_all['annual_amplitude'].min():.2f}, "
          f"{maps_all['annual_amplitude'].max():.2f}] deg C")
    print(f"  Linear trend range: [{maps_all['linear_trend'].min():.3f}, "
          f"{maps_all['linear_trend'].max():.3f}] deg C")

    # ================================================================
    # Visualization
    # ================================================================
    print("\nGenerating plots...")

    def scatter_map(ax, lats, lons, values, title, cmap='viridis',
                    vmin=None, vmax=None):
        sc = ax.scatter(lons, lats, c=values, cmap=cmap, s=1.5,
                        vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.tick_params(labelsize=7)
        return sc

    # Plot 1: Seasonal decomposition
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), facecolor='white')
    fig.suptitle('NOAA SST: Tropical Pacific Seasonal Decomposition\n'
                 'Spatial B-splines x Temporal Fourier+Trend',
                 fontsize=14, fontweight='bold', y=0.98)

    sc = scatter_map(axes[0, 0], ocean_lats, ocean_lons, maps_all['mean'],
                     'Mean SST (deg C)', cmap='RdYlBu_r')
    plt.colorbar(sc, ax=axes[0, 0], fraction=0.046)

    sc = scatter_map(axes[0, 1], ocean_lats, ocean_lons,
                     maps_all['annual_amplitude'],
                     'Annual Amplitude (deg C)', cmap='hot', vmin=0)
    plt.colorbar(sc, ax=axes[0, 1], fraction=0.046)

    sc = scatter_map(axes[0, 2], ocean_lats, ocean_lons,
                     maps_all['annual_phase_month'],
                     'Annual Phase (peak month)', cmap='hsv',
                     vmin=-6, vmax=6)
    plt.colorbar(sc, ax=axes[0, 2], fraction=0.046)

    sc = scatter_map(axes[1, 0], ocean_lats, ocean_lons,
                     maps_all['linear_trend'],
                     'Linear Trend (deg C / record)', cmap='RdBu_r',
                     vmin=-2, vmax=2)
    plt.colorbar(sc, ax=axes[1, 0], fraction=0.046)

    # Semi-annual amplitude
    sa_amp = np.sqrt(maps_all['semiannual_cos']**2 +
                     maps_all['semiannual_sin']**2)
    sc = scatter_map(axes[1, 1], ocean_lats, ocean_lons, sa_amp,
                     'Semi-annual Amplitude (deg C)', cmap='hot', vmin=0)
    plt.colorbar(sc, ax=axes[1, 1], fraction=0.046)

    # Merge vs centralized difference
    diff_mean = maps_merged['mean'] - maps_all['mean']
    sc = scatter_map(axes[1, 2], ocean_lats, ocean_lons, diff_mean,
                     'Merged - Centralized (mean SST)',
                     cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    plt.colorbar(sc, ax=axes[1, 2], fraction=0.046)
    axes[1, 2].axvline(x=lon_split, color='black', linewidth=1,
                        linestyle=':', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('noaa_sst_decomposition.png', dpi=150,
                facecolor='white', bbox_inches='tight')
    print("  Saved: noaa_sst_decomposition.png")
    plt.close()

    # Plot 2: Reconstruction at specific time steps
    fig, axes = plt.subplots(2, 3, figsize=(20, 8), facecolor='white')
    fig.suptitle('NOAA SST: Product Values, Merged Reconstruction, and Residual\n'
                 f'Split at {lon_split}E (dashed line)',
                 fontsize=14, fontweight='bold', y=0.98)

    # Pick January and July from a representative year (2010)
    sample_months = [337, 343]  # roughly Jan 2010, Jul 2010
    sample_months = [min(m, N_t - 1) for m in sample_months]

    for row, ti in enumerate(sample_months):
        mask = ocean_mask_per_t[ti]
        lats_m = ocean_lats[mask]
        lons_m = ocean_lons[mask]
        true_vals = sst_ocean[ti, mask]

        pred_central = eval_field(C_all, phi_s[mask], t_feats[ti], K_s, K_t)
        pred_merged = eval_field(C_merged, phi_s[mask], t_feats[ti], K_s, K_t)

        t_label = str(times[ti])[:7]

        sc = scatter_map(axes[row, 0], lats_m, lons_m, true_vals,
                         f'Product ({t_label})', cmap='RdYlBu_r',
                         vmin=15, vmax=32)
        if row == 0:
            plt.colorbar(sc, ax=axes[row, 0], fraction=0.046)

        sc = scatter_map(axes[row, 1], lats_m, lons_m, pred_merged,
                         f'Merged ({t_label})', cmap='RdYlBu_r',
                         vmin=15, vmax=32)
        axes[row, 1].axvline(x=lon_split, color='black', linewidth=1,
                              linestyle=':', alpha=0.5)

        diff = pred_merged - true_vals
        sc = scatter_map(axes[row, 2], lats_m, lons_m, diff,
                         f'Residual ({t_label})', cmap='RdBu_r',
                         vmin=-3, vmax=3)
        plt.colorbar(sc, ax=axes[row, 2], fraction=0.046)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('noaa_sst_reconstruction.png', dpi=150,
                facecolor='white', bbox_inches='tight')
    print("  Saved: noaa_sst_reconstruction.png")
    plt.close()

    # Plot 3: Time series at a specific point (Nino 3.4 region center)
    nino34_lat, nino34_lon = 0.0, 190.0  # ~170W
    idx_closest = np.argmin((ocean_lats - nino34_lat)**2 +
                            (ocean_lons - nino34_lon)**2)
    print(f"\n  Nino 3.4 center: lat={ocean_lats[idx_closest]:.1f}, "
          f"lon={ocean_lons[idx_closest]:.1f}")

    phi_s_point = phi_s[idx_closest:idx_closest+1]
    ts_true = sst_ocean[:, idx_closest]
    ts_central = np.array([eval_field(C_all, phi_s_point, t_feats[ti],
                                       K_s, K_t)[0] for ti in range(N_t)])
    ts_merged = np.array([eval_field(C_merged, phi_s_point, t_feats[ti],
                                      K_s, K_t)[0] for ti in range(N_t)])

    # Seasonal component only
    C_seasonal = C_all.copy().reshape(K_s, K_t)
    C_seasonal[:, 5:] = 0  # zero out trend terms
    C_seasonal_flat = C_seasonal.flatten()
    ts_seasonal = np.array([eval_field(C_seasonal_flat, phi_s_point,
                                        t_feats[ti], K_s, K_t)[0]
                            for ti in range(N_t)])

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), facecolor='white')
    fig.suptitle(f'SST Time Series at Nino 3.4 Region Center '
                 f'({ocean_lats[idx_closest]:.1f}N, '
                 f'{ocean_lons[idx_closest]:.1f}E)',
                 fontsize=14, fontweight='bold', y=0.98)

    years = [1982 + i/12 for i in range(N_t)]

    ax = axes[0]
    ax.plot(years, ts_true, color='#999', linewidth=0.5, alpha=0.7,
            label='Observed')
    ax.plot(years, ts_central, color='#2ca02c', linewidth=1.5,
            label='Centralized')
    ax.plot(years, ts_merged, color='#9467bd', linewidth=1.5, linestyle='--',
            label='Merged')
    ax.set_ylabel('SST (deg C)')
    ax.set_title('Full reconstruction', fontsize=11)
    ax.legend()

    ax = axes[1]
    anomaly = ts_true - ts_seasonal
    ax.fill_between(years, anomaly, 0, where=anomaly > 0,
                    color='#d62728', alpha=0.5, label='Warm anomaly')
    ax.fill_between(years, anomaly, 0, where=anomaly < 0,
                    color='#1f77b4', alpha=0.5, label='Cold anomaly')
    ax.axhline(y=0, color='#999', linewidth=0.5)
    ax.set_ylabel('SST Anomaly (deg C)')
    ax.set_xlabel('Year')
    ax.set_title('Anomaly (observed - seasonal fit)', fontsize=11)
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('noaa_sst_timeseries.png', dpi=150,
                facecolor='white', bbox_inches='tight')
    print("  Saved: noaa_sst_timeseries.png")
    plt.close()

    # Save results
    with open('noaa_sst_results.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'centralized', 'merged', 'west_only', 'east_only'])
        w.writerow(['rmse_overall', f'{rmse_all:.4f}', '', '', ''])
    print("  Saved: noaa_sst_results.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
