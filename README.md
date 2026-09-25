# SplineMerge: Exact Distributed Field Reconstruction via Composable Spline Statistics

Source code for reproducing the experiments in the ICLR 2027 submission.

## Installation

```bash
pip install -r requirements.txt
```

Requires `inkan>=0.4.2`, `torch`, `numpy`, `scipy`, `matplotlib`, `torchvision`, `netCDF4`.

## Project structure

```
experiment_diffusion.py       Diffusion equation (u_t = D Lap u)
experiment_wave.py            Wave equation (u_tt = c^2 Lap u)
experiment_burgers.py         Viscous Burgers (u_t + u u_x = kappa u_xx, nonlinear)
experiment_heat_source.py     Heat with source (u_t = D Lap u + S(x,y))
experiment_mnist.py           MNIST class-average image reconstruction (2D fields)
experiment_noaa_sst.py        NOAA OI SST V2 real-data reconstruction
data/
  download_noaa_data.py       Download NOAA SST data and land-sea mask
evaluations/
  noaa_held_out_eval.py       NOAA spatial and temporal held-out evaluation
  noise_sensitivity_study.py  Noise sensitivity (3 densities x 3 noise x 5 seeds)
  verify_derivatives.py       Derivative verification (FD-grid error controls)
utils/
  plot_diffusion.py           Plot diffusion results from CSV
  plot_wave.py                Plot wave results from CSV
  plot_noaa_sst.py            Plot NOAA SST results from CSV
```

## Obtaining the NOAA SST data

```bash
python data/download_noaa_data.py
```

This downloads `sst.mnmean.nc` and `lsmask.nc` into `data/noaa/`.

Manual download: https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.html

## Running the experiments

All commands run from this directory.

```bash
# Synthetic PDEs (no external data needed)
python experiment_diffusion.py
python experiment_wave.py
python experiment_burgers.py
python experiment_heat_source.py

# MNIST (downloads automatically)
python experiment_mnist.py

# NOAA SST (requires data download first)
python data/download_noaa_data.py
python experiment_noaa_sst.py

# Evaluations
python evaluations/noise_sensitivity_study.py
python evaluations/noaa_held_out_eval.py
python evaluations/verify_derivatives.py

# Plotting from saved CSVs
python utils/plot_diffusion.py merge_pde_diffusion.csv
python utils/plot_wave.py merge_pde_wave.csv
python utils/plot_noaa_sst.py noaa_sst_results.csv
```

## Runtime estimates

- Synthetic PDE experiments: 5-10 minutes each on CPU
- MNIST: 2-3 minutes on CPU
- NOAA SST: 2-5 minutes on CPU (after data download)
- Noise sensitivity study: 30-60 minutes on CPU

## License

This code is provided for review purposes.
