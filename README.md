# SplineMerge: Distributed Spatiotemporal Field Reconstruction via Composable Spline Statistics

Source code for reproducing the experiments in the ICLR 2027 submission.

## Installation

```bash
pip install -r requirements.txt
```

The key dependencies are:

- `torch` -- tensor operations and linear algebra
- `numpy`, `scipy` -- numerical routines
- `matplotlib` -- plotting
- `inkan` -- B-spline basis evaluation
- `netCDF4` -- reading NOAA SST data (required only for the NOAA experiment)

## Project structure

```
src/                Core library
  basis.py            B-spline basis construction and evaluation
  merge.py            Gram-based field fitting and merge operations
  solvers.py          PDE solvers for synthetic ground truth generation
experiments/        Reproducible experiment scripts
  experiment_diffusion.py   Diffusion equation (u_t = D Lap u)
  experiment_wave.py        Wave equation (u_tt = c^2 Lap u)
  experiment_burgers.py     Viscous Burgers equation (u_t + u u_x = kappa u_xx, nonlinear)
  experiment_heat_source.py Heat equation with source (u_t = D Lap u + S(x,y))
  experiment_mnist.py       MNIST class-average image reconstruction (2D fields)
  experiment_noaa_sst.py    NOAA OI SST V2 real-data reconstruction
  noaa_held_out_eval.py     NOAA spatial and temporal held-out evaluation
  noise_sensitivity_study.py Noise sensitivity (3 densities x 3 noise x 5 seeds)
  verify_derivatives.py     Derivative verification (FD-grid vs spline error)
utils/              Plotting and data utilities
  plot_diffusion.py         Plot diffusion experiment results
  plot_wave.py              Plot wave experiment results
  plot_noaa_sst.py          Plot NOAA SST experiment results
  download_noaa_data.py     Download NOAA SST data
```

## Obtaining the NOAA SST data

The NOAA experiment requires the NOAA OI SST V2 monthly mean dataset
(`sst.mnmean.nc`) and its land-sea mask (`lsmask.nc`). Both files are
freely available from the NOAA Physical Sciences Laboratory (PSL).

**Option A -- Automatic download:**

```bash
python utils/download_noaa_data.py
```

This downloads both files and places them in `data/noaa/`.

**Option B -- Manual download:**

1. Visit https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.html
2. Download `sst.mnmean.nc` from https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2/sst.mnmean.nc
3. Download `lsmask.nc` from https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2/lsmask.nc
4. Place both files at `data/noaa/` (relative to this directory)

## Running the experiments

All commands should be run from this directory (the project root).

### Experiment 1: Diffusion equation

```bash
python experiments/experiment_diffusion.py
```

This runs the pure diffusion PDE experiment with varying observation
densities and noise levels, followed by a distributed merge (left/right
spatial split). No external data is required -- ground truth is computed
via exact spectral solution.

**Expected outputs:**
- `merge_pde_diffusion.png` -- field reconstruction visualization
- `merge_pde_diffusion.csv` -- tabulated RMSE and recovered D values

### Experiment 2: Wave equation

```bash
python experiments/experiment_wave.py
```

This runs the wave equation experiment with the same pipeline. Ground
truth is again computed exactly via spectral methods.

**Expected outputs:**
- `merge_pde_wave.png` -- field reconstruction visualization
- `merge_pde_wave.csv` -- tabulated RMSE and recovered wave speed

### Experiment 3: NOAA SST reconstruction

```bash
python utils/download_noaa_data.py   # first time only (downloads sst.mnmean.nc and lsmask.nc)
python experiments/experiment_noaa_sst.py
```

This applies the splinemerge pipeline to real NOAA OI SST V2 monthly data
over the tropical Pacific (30S-30N, 120E-280E), using spatial B-splines
crossed with temporal Fourier+trend features. The distributed merge
splits the domain into west Pacific (120-200E) and east Pacific
(200-280E). The land-sea mask (`lsmask.nc`) is used to filter out land
cells that the SST product fills by interpolation.

**Expected outputs:**
- `noaa_sst_decomposition.png` -- seasonal decomposition maps (mean, amplitude, phase, trend)
- `noaa_sst_reconstruction.png` -- observed vs merged reconstruction at sample months
- `noaa_sst_timeseries.png` -- SST time series at the Nino 3.4 region center
- `noaa_sst_results.csv` -- tabulated RMSE results

### Derivative verification

```bash
python experiments/verify_derivatives.py
```

This script decomposes the total parameter-recovery error into two
independent sources: (1) finite-difference truncation error on the
evaluation grid (applied to the exact analytic/spectral field), and
(2) the full pipeline error (FD stencils applied to the spline-fitted
field). The difference isolates the spline contribution. It uses
identical grid parameters and FD stencils as the diffusion and wave
experiment scripts. Self-contained: no external data or `inkan`
dependency required.

## Generating summary plots from saved results

After running an experiment, the corresponding plot utility can regenerate
figures from the saved CSV:

```bash
python utils/plot_diffusion.py merge_pde_diffusion.csv
python utils/plot_wave.py merge_pde_wave.csv
python utils/plot_noaa_sst.py noaa_sst_results.csv
```

## Runtime estimates

- Diffusion experiment: approximately 5-10 minutes on CPU
- Wave experiment: approximately 5-10 minutes on CPU
- NOAA SST experiment: approximately 2-5 minutes on CPU (after data download)

## License

This code is provided for review purposes.
