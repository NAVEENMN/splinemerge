"""
NOAA SST Held-Out Evaluation: spatial and temporal interpolation quality.

Extends the NOAA SST experiment with two held-out tests against the
analyzed gridded product (NOAA OI SST V2).  This measures interpolation
quality of the fitted field, not prediction of raw sensor observations.

Test 1 -- Spatial held-out:
    Randomly withhold 20% of ocean grid points (fixed seed=42) across
    ALL months.  Fit on the remaining 80%, evaluate on the withheld 20%.
    Run both centralized and distributed (west/east merge) fits.

Test 2 -- Temporal held-out:
    Withhold every 6th month (indices 0, 6, 12, 18, ...) across all
    ocean points.  Fit on the remaining months, evaluate on the withheld
    months.  Run both centralized and distributed (west/east merge) fits.

Region: tropical Pacific (30S-30N, 120E-280E).
Hybrid basis: spatial B-splines x temporal Fourier+trend.
"""

import torch
import numpy as np
import csv
import netCDF4 as nc

from inkan.basis import bspline_basis_eager


# ============================================================================
# Basis helpers  (reused from merge_noaa_sst.py)
# ============================================================================

def make_basis_params(grid_size, grid_range):
    """Create grid_starts, inv_h, n_bases for 1D cubic B-spline basis."""
    n_bases = grid_size + 3
    h = (grid_range[1] - grid_range[0]) / grid_size
    inv_h = 1.0 / h
    grid_starts = torch.arange(n_bases).float() * h + grid_range[0] - 3 * h
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
# Build spatial features
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


# ============================================================================
# Gram matrix accumulation and solve
# ============================================================================

def accumulate_gram(phi_s, t_feats, sst_values, ocean_mask_per_t):
    """Accumulate Gram matrix and RHS across time steps.

    Parameters
    ----------
    phi_s : torch.Tensor or ndarray, shape [N_spatial, K_s]
        Precomputed spatial features for the points in this partition.
    t_feats : ndarray, shape [N_t, K_t]
        Temporal features (may be a subset of all time steps).
    sst_values : ndarray, shape [N_t, N_spatial]
        SST values for the points in this partition.
    ocean_mask_per_t : ndarray, shape [N_t, N_spatial]
        Boolean mask indicating valid observations per time step.

    Returns
    -------
    G : ndarray [P, P]
    h : ndarray [P]
    n_obs_total : int
    """
    K_s = phi_s.shape[1]
    K_t = t_feats.shape[1]
    P = K_s * K_t

    G = np.zeros((P, P), dtype=np.float64)
    h = np.zeros(P, dtype=np.float64)
    n_obs_total = 0

    phi_s_np = phi_s.numpy() if torch.is_tensor(phi_s) else phi_s

    N_t = len(t_feats)
    for ti in range(N_t):
        mask = ocean_mask_per_t[ti]
        if mask.sum() == 0:
            continue

        phi_s_valid = phi_s_np[mask]           # [n_valid, K_s]
        sst_valid = sst_values[ti][mask]       # [n_valid]
        ft = t_feats[ti]                       # [K_t]

        # Full feature: phi_s x ft -> [n_valid, P]
        phi_full = (phi_s_valid[:, :, None] * ft[None, None, :]).reshape(-1, P)

        G += phi_full.T @ phi_full
        h += phi_full.T @ sst_valid
        n_obs_total += mask.sum()

        if (ti + 1) % 100 == 0:
            print(f"    {ti+1}/{N_t} time steps processed "
                  f"({n_obs_total:,} obs)")

    print(f"    Total: {n_obs_total:,} observations")
    return G, h, n_obs_total


def solve_field(G, h, reg=1e-4):
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
# RMSE evaluation helpers
# ============================================================================

def compute_rmse(C, phi_s, t_feats, sst_values, ocean_mask_per_t, K_s, K_t):
    """Compute RMSE over specified time steps and spatial points.

    Parameters
    ----------
    C : ndarray [P]
        Coefficient vector.
    phi_s : torch.Tensor or ndarray [N_spatial, K_s]
        Spatial features for the evaluation points.
    t_feats : ndarray [N_t, K_t]
        Temporal features for the evaluation time steps.
    sst_values : ndarray [N_t, N_spatial]
        Ground truth SST for the evaluation points.
    ocean_mask_per_t : ndarray [N_t, N_spatial]
        Boolean mask for valid observations.
    K_s, K_t : int
        Spatial and temporal basis counts.

    Returns
    -------
    float
        RMSE in degrees C.
    """
    errors = []
    for ti in range(len(t_feats)):
        mask = ocean_mask_per_t[ti]
        if mask.sum() == 0:
            continue
        pred = eval_field(C, phi_s[mask], t_feats[ti], K_s, K_t)
        true = sst_values[ti, mask]
        errors.append(np.mean((pred - true) ** 2))
    if len(errors) == 0:
        return float('nan')
    return np.sqrt(np.mean(errors))


# ============================================================================
# Data loading
# ============================================================================

def load_data(sst_path, mask_path):
    """Load NOAA OI SST V2 monthly data and land-sea mask.

    Returns a dict with all arrays needed for the experiments.
    """
    print("Loading NOAA OI SST V2...")
    ds = nc.Dataset(sst_path)
    lat = ds.variables['lat'][:].data
    lon = ds.variables['lon'][:].data
    sst_full = np.ma.asarray(
        ds.variables['sst'][:], dtype=np.float64
    ).filled(np.nan)  # [N_t_full, 180, 360]
    times = nc.num2date(ds.variables['time'][:],
                        ds.variables['time'].units)
    ds.close()

    print("  Loading land-sea mask (lsmask.nc)...")
    ds_mask = nc.Dataset(mask_path)
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

    # Per-time-step ocean mask
    sst_flat = sst_region.reshape(N_t, N_lat * N_lon)  # [N_t, N_lat*N_lon]
    ocean_mask_flat_2d = ocean_mask_2d.flatten()  # [N_lat*N_lon]

    ocean_idx = np.where(ocean_mask_flat_2d)[0]
    ocean_mask_per_t = np.zeros((N_t, n_ocean), dtype=bool)
    sst_ocean = np.zeros((N_t, n_ocean), dtype=np.float64)
    for ti in range(N_t):
        vals = sst_flat[ti, ocean_idx]
        valid = np.isfinite(vals)
        ocean_mask_per_t[ti] = valid
        sst_ocean[ti, valid] = vals[valid]

    print(f"  Valid obs per time step: min={ocean_mask_per_t.sum(1).min()}, "
          f"max={ocean_mask_per_t.sum(1).max()}")

    return {
        'ocean_lats': ocean_lats,
        'ocean_lons': ocean_lons,
        'sst_ocean': sst_ocean,
        'ocean_mask_per_t': ocean_mask_per_t,
        'n_ocean': n_ocean,
        'N_t': N_t,
        'lat_min': lat_min, 'lat_max': lat_max,
        'lon_min': lon_min, 'lon_max': lon_max,
    }


# ============================================================================
# Distributed merge helpers
# ============================================================================

def run_distributed_merge(phi_s, t_feats, sst_ocean, ocean_mask_per_t,
                          ocean_lons, lon_split, reg):
    """Fit west and east partitions, merge via Gram summation.

    Parameters
    ----------
    phi_s : torch.Tensor [N_ocean, K_s]
    t_feats : ndarray [N_t, K_t]
    sst_ocean : ndarray [N_t, N_ocean]
    ocean_mask_per_t : ndarray [N_t, N_ocean]
    ocean_lons : ndarray [N_ocean]
    lon_split : float
    reg : float

    Returns
    -------
    C_merged : ndarray [P]
    """
    west_mask = ocean_lons < lon_split
    east_mask = ~west_mask

    # West partition
    phi_s_w = phi_s[west_mask]
    ocean_mask_w = ocean_mask_per_t[:, west_mask]
    sst_w = sst_ocean[:, west_mask]

    print("  Fitting west partition...")
    G_w, h_w, _ = accumulate_gram(phi_s_w, t_feats, sst_w, ocean_mask_w)

    # East partition
    phi_s_e = phi_s[east_mask]
    ocean_mask_e = ocean_mask_per_t[:, east_mask]
    sst_e = sst_ocean[:, east_mask]

    print("  Fitting east partition...")
    G_e, h_e, _ = accumulate_gram(phi_s_e, t_feats, sst_e, ocean_mask_e)

    # Gram merge
    C_merged = solve_field(G_w + G_e, h_w + h_e, reg=reg)
    return C_merged


# ============================================================================
# Test 1: Spatial held-out
# ============================================================================

def test_spatial_holdout(data, phi_s, t_feats, bp_lat, bp_lon,
                         K_s, K_t, reg, lon_split):
    """Randomly withhold 20% of ocean grid points (seed=42).

    The withheld points are excluded across ALL months.
    Fit on the remaining 80%, evaluate on the withheld 20%.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Spatial Held-Out (20% ocean points withheld)")
    print("=" * 70)

    ocean_lats = data['ocean_lats']
    ocean_lons = data['ocean_lons']
    sst_ocean = data['sst_ocean']
    ocean_mask_per_t = data['ocean_mask_per_t']
    n_ocean = data['n_ocean']

    # Split ocean points: 80% train, 20% held-out
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_ocean)
    n_holdout = n_ocean // 5
    n_train = n_ocean - n_holdout

    holdout_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]

    # Sort indices for reproducibility
    holdout_idx = np.sort(holdout_idx)
    train_idx = np.sort(train_idx)

    print(f"  Train points: {n_train}, Held-out points: {n_holdout}")

    # Partition spatial features and SST
    phi_s_train = phi_s[train_idx]
    phi_s_holdout = phi_s[holdout_idx]

    sst_train = sst_ocean[:, train_idx]
    sst_holdout = sst_ocean[:, holdout_idx]

    mask_train = ocean_mask_per_t[:, train_idx]
    mask_holdout = ocean_mask_per_t[:, holdout_idx]

    # -- Centralized fit on 80% --
    print("\n  --- Centralized fit on training set ---")
    G_train, h_train, n_train_obs = accumulate_gram(
        phi_s_train, t_feats, sst_train, mask_train)
    C_central = solve_field(G_train, h_train, reg=reg)

    rmse_train_central = compute_rmse(
        C_central, phi_s_train, t_feats, sst_train, mask_train, K_s, K_t)
    rmse_holdout_central = compute_rmse(
        C_central, phi_s_holdout, t_feats, sst_holdout, mask_holdout, K_s, K_t)

    print(f"  Training RMSE:  {rmse_train_central:.4f} deg C")
    print(f"  Held-out RMSE:  {rmse_holdout_central:.4f} deg C")

    # -- Distributed merge on 80% training set --
    print("\n  --- Distributed merge on training set ---")
    train_lons = ocean_lons[train_idx]
    C_merged = run_distributed_merge(
        phi_s_train, t_feats, sst_train, mask_train,
        train_lons, lon_split, reg)

    rmse_train_merged = compute_rmse(
        C_merged, phi_s_train, t_feats, sst_train, mask_train, K_s, K_t)
    rmse_holdout_merged = compute_rmse(
        C_merged, phi_s_holdout, t_feats, sst_holdout, mask_holdout, K_s, K_t)

    print(f"  Training RMSE:  {rmse_train_merged:.4f} deg C")
    print(f"  Held-out RMSE:  {rmse_holdout_merged:.4f} deg C")

    # -- Check merged vs centralized match --
    coeff_diff = np.max(np.abs(C_merged - C_central))
    match = coeff_diff < 1e-8
    print(f"\n  Merged vs centralized max |coeff diff|: {coeff_diff:.2e}")
    print(f"  Exact match (tol=1e-8): {match}")

    return {
        'train_rmse_central': rmse_train_central,
        'holdout_rmse_central': rmse_holdout_central,
        'train_rmse_merged': rmse_train_merged,
        'holdout_rmse_merged': rmse_holdout_merged,
        'coeff_max_diff': coeff_diff,
        'merged_matches_centralized': match,
    }


# ============================================================================
# Test 2: Temporal held-out
# ============================================================================

def test_temporal_holdout(data, phi_s, t_feats_full, month_indices_full,
                          t_max_global, bp_lat, bp_lon, K_s, K_t,
                          reg, lon_split):
    """Withhold every 6th month (indices 0, 6, 12, 18, ...).

    Fit on the remaining months, evaluate on the withheld months.
    The temporal features use the same t_max from the full time range.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Temporal Held-Out (every 6th month withheld)")
    print("=" * 70)

    ocean_lons = data['ocean_lons']
    sst_ocean = data['sst_ocean']
    ocean_mask_per_t = data['ocean_mask_per_t']
    N_t = data['N_t']

    # Split time steps: every 6th month is held out
    all_t_idx = np.arange(N_t)
    holdout_t_idx = all_t_idx[::6]   # 0, 6, 12, 18, ...
    train_t_idx = np.array([i for i in all_t_idx if i not in holdout_t_idx])

    n_train_t = len(train_t_idx)
    n_holdout_t = len(holdout_t_idx)

    print(f"  Train months: {n_train_t}, Held-out months: {n_holdout_t}")

    # Temporal features for subsets (same t_max from full range)
    t_feats_train = t_feats_full[train_t_idx]
    t_feats_holdout = t_feats_full[holdout_t_idx]

    # SST and masks for subsets
    sst_train = sst_ocean[train_t_idx]
    sst_holdout = sst_ocean[holdout_t_idx]

    mask_train = ocean_mask_per_t[train_t_idx]
    mask_holdout = ocean_mask_per_t[holdout_t_idx]

    # -- Centralized fit on training months --
    print("\n  --- Centralized fit on training months ---")
    G_train, h_train, n_train_obs = accumulate_gram(
        phi_s, t_feats_train, sst_train, mask_train)
    C_central = solve_field(G_train, h_train, reg=reg)

    rmse_train_central = compute_rmse(
        C_central, phi_s, t_feats_train, sst_train, mask_train, K_s, K_t)
    rmse_holdout_central = compute_rmse(
        C_central, phi_s, t_feats_holdout, sst_holdout, mask_holdout, K_s, K_t)

    print(f"  Training RMSE:  {rmse_train_central:.4f} deg C")
    print(f"  Held-out RMSE:  {rmse_holdout_central:.4f} deg C")

    # -- Distributed merge on training months --
    print("\n  --- Distributed merge on training months ---")
    C_merged = run_distributed_merge(
        phi_s, t_feats_train, sst_train, mask_train,
        ocean_lons, lon_split, reg)

    rmse_train_merged = compute_rmse(
        C_merged, phi_s, t_feats_train, sst_train, mask_train, K_s, K_t)
    rmse_holdout_merged = compute_rmse(
        C_merged, phi_s, t_feats_holdout, sst_holdout, mask_holdout, K_s, K_t)

    print(f"  Training RMSE:  {rmse_train_merged:.4f} deg C")
    print(f"  Held-out RMSE:  {rmse_holdout_merged:.4f} deg C")

    # -- Check merged vs centralized match --
    coeff_diff = np.max(np.abs(C_merged - C_central))
    match = coeff_diff < 1e-8
    print(f"\n  Merged vs centralized max |coeff diff|: {coeff_diff:.2e}")
    print(f"  Exact match (tol=1e-8): {match}")

    return {
        'train_rmse_central': rmse_train_central,
        'holdout_rmse_central': rmse_holdout_central,
        'train_rmse_merged': rmse_train_merged,
        'holdout_rmse_merged': rmse_holdout_merged,
        'coeff_max_diff': coeff_diff,
        'merged_matches_centralized': match,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("NOAA SST: Held-Out Evaluation")
    print("  Spatial and temporal interpolation quality")
    print("  Evaluation against the analyzed gridded product (OI SST V2)")
    print("=" * 70)

    # Paths
    sst_path = "data/noaa/sst.mnmean.nc"
    mask_path = "data/noaa/lsmask.nc"

    # Load data
    data = load_data(sst_path, mask_path)

    ocean_lats = data['ocean_lats']
    ocean_lons = data['ocean_lons']
    N_t = data['N_t']
    lat_min, lat_max = data['lat_min'], data['lat_max']
    lon_min, lon_max = data['lon_min'], data['lon_max']

    # Basis parameters: K_lat=13, K_lon=23
    gs_lat = 10
    gs_lon = 20
    bp_lat = make_basis_params(gs_lat, (lat_min, lat_max))
    bp_lon = make_basis_params(gs_lon, (lon_min, lon_max))
    K_lat, K_lon = bp_lat[2], bp_lon[2]
    K_s = K_lat * K_lon
    K_t = 7  # temporal features
    P = K_s * K_t
    print(f"\n  Basis: K_lat={K_lat}, K_lon={K_lon}, K_s={K_s}, K_t={K_t}")
    print(f"  Total parameters: {P}")

    # Precompute spatial features for ALL ocean points
    print("  Computing spatial features...")
    phi_s = build_spatial_features(ocean_lats, ocean_lons, bp_lat, bp_lon)
    print(f"  Spatial features: {phi_s.shape}")

    # Temporal features with FIXED t_max from the full time range
    month_indices = np.arange(N_t, dtype=np.float64)
    t_max_global = float(max(month_indices.max(), 1))
    t_feats_full = temporal_features(month_indices, t_max=t_max_global)
    print(f"  Temporal features: {t_feats_full.shape} "
          f"(t_max={t_max_global:.0f})")

    # Regularization
    reg = 1e-4
    lon_split = 200.0

    # ----------------------------------------------------------------
    # Run tests
    # ----------------------------------------------------------------
    results_spatial = test_spatial_holdout(
        data, phi_s, t_feats_full, bp_lat, bp_lon, K_s, K_t, reg, lon_split)

    results_temporal = test_temporal_holdout(
        data, phi_s, t_feats_full, month_indices, t_max_global,
        bp_lat, bp_lon, K_s, K_t, reg, lon_split)

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    header = (f"{'Test':<22} {'Method':<14} {'Train RMSE':>12} "
              f"{'Holdout RMSE':>14} {'Match':>8}")
    print(header)
    print("-" * len(header))

    rows = []

    # Spatial held-out
    rows.append({
        'test': 'spatial_holdout',
        'method': 'centralized',
        'train_rmse': results_spatial['train_rmse_central'],
        'holdout_rmse': results_spatial['holdout_rmse_central'],
        'merged_match': '',
    })
    rows.append({
        'test': 'spatial_holdout',
        'method': 'merged',
        'train_rmse': results_spatial['train_rmse_merged'],
        'holdout_rmse': results_spatial['holdout_rmse_merged'],
        'merged_match': str(results_spatial['merged_matches_centralized']),
    })

    # Temporal held-out
    rows.append({
        'test': 'temporal_holdout',
        'method': 'centralized',
        'train_rmse': results_temporal['train_rmse_central'],
        'holdout_rmse': results_temporal['holdout_rmse_central'],
        'merged_match': '',
    })
    rows.append({
        'test': 'temporal_holdout',
        'method': 'merged',
        'train_rmse': results_temporal['train_rmse_merged'],
        'holdout_rmse': results_temporal['holdout_rmse_merged'],
        'merged_match': str(results_temporal['merged_matches_centralized']),
    })

    for r in rows:
        print(f"{r['test']:<22} {r['method']:<14} "
              f"{r['train_rmse']:>12.4f} {r['holdout_rmse']:>14.4f} "
              f"{r['merged_match']:>8}")

    print(f"\n  Spatial coeff max diff:  "
          f"{results_spatial['coeff_max_diff']:.2e}")
    print(f"  Temporal coeff max diff: "
          f"{results_temporal['coeff_max_diff']:.2e}")

    # ----------------------------------------------------------------
    # Save to CSV
    # ----------------------------------------------------------------
    csv_path = 'noaa_held_out_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['test', 'method', 'train_rmse', 'holdout_rmse',
                     'merged_matches_centralized', 'coeff_max_diff'])
        for r in rows:
            w.writerow([
                r['test'],
                r['method'],
                f"{r['train_rmse']:.6f}",
                f"{r['holdout_rmse']:.6f}",
                r['merged_match'],
                '',
            ])
        # Add coeff diff rows
        w.writerow(['spatial_holdout', 'coeff_diff', '', '',
                     str(results_spatial['merged_matches_centralized']),
                     f"{results_spatial['coeff_max_diff']:.2e}"])
        w.writerow(['temporal_holdout', 'coeff_diff', '', '',
                     str(results_temporal['merged_matches_centralized']),
                     f"{results_temporal['coeff_max_diff']:.2e}"])

    print(f"\n  Saved: {csv_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
