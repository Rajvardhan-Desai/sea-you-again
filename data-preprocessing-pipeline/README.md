# MM-MARAS Preprocessing Pipeline

> **M**ulti-**M**odal **M**issingness-**A**ware **R**econstruction **A**nd **S**upervised forecasting of Chlorophyll-a in the Bay of Bengal.

This repository contains the full data preprocessing pipeline for MM-MARAS — a deep learning framework for satellite Chl-a reconstruction and short-range forecasting in the Bay of Bengal, using multi-source oceanographic and atmospheric forcing data.

---

## Table of Contents

- [Overview](#overview)
- [Domain](#domain)
- [Data Sources](#data-sources)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Authentication](#authentication)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Output Structure](#output-structure)
- [Patch Format](#patch-format)
- [ERA5 Download Methods](#era5-download-methods)
- [Resumability](#resumability)
- [Verifying Output](#verifying-output)
- [Troubleshooting](#troubleshooting)

---

## Overview

The pipeline downloads, aligns, normalizes, and patches five multi-source oceanographic datasets into spatiotemporal training samples for the MM-MARAS model.

```
BGC (Chl-a + nutrients)  ─┐
Physics (SST, currents)  ─┤
ERA5 (wind, pressure,    ─┼─► Align to 0.25° grid ─► Normalize ─► Patch ─► .npz
      precipitation)      │    (BGC target grid)
GloFAS (discharge)       ─┤
GEBCO (bathymetry)       ─┘
```

**Pipeline steps:**
1. Download raw data from CMEMS, CDS, CEMS
2. Standardize coordinates and clip to domain
3. Regrid all modalities to the BGC 0.25° grid
4. Align time axes to Chl-a reference
5. Build observation, land, bloom, MCAR, and MNAR masks
6. Normalize all variables (log1p + z-score or min-max)
7. Assemble static context (bathymetry + distance-to-coast)
8. Extract spatiotemporal patches with train/val/test split

---

## Domain

**Bay of Bengal** — IHO official definition (IHO Publication S-23)

| Boundary | Coordinate | Reference point |
|---|---|---|
| West | 79.5°E | Point Calimere, India (79°53'E) |
| East | 95.5°E | Pulau Breueh, Sumatra (95°02'E) |
| South | 5.5°N | Dondra Head, Sri Lanka (5°55'N) |
| North | 22.5°N | Northern Bangladesh coast |

**Temporal coverage:** 2021-01-01 → 2025-12-31 (5 years, daily)

---

## Data Sources

### 1. BGC — Chlorophyll-a and Nutrients
| Property | Value |
|---|---|
| Product | `GLOBAL_MULTIYEAR_BGC_001_029` |
| Dataset | `cmems_mod_glo_bgc_my_0.25deg_P1D-m` |
| Source | Copernicus Marine Service (CMEMS) |
| Resolution | 0.25°, daily |
| Surface depth | 0.51 m |
| Variables | `chl`, `o2`, `no3`, `po4`, `si`, `nppv` |

### 2. Physics — Ocean State
| Property | Value |
|---|---|
| Product | `GLOBAL_MULTIYEAR_PHY_001_030` |
| Dataset | `cmems_mod_glo_phy_my_0.083deg_P1D-m` |
| Source | Copernicus Marine Service (CMEMS) |
| Resolution | 0.083°, daily |
| Surface depth | 0.494 m |
| Variables | `thetao`, `uo`, `vo`, `mlotst`, `zos`, `so` |

### 3. ERA5 — Atmospheric Forcing
| Property | Value |
|---|---|
| Dataset | `derived-era5-single-levels-daily-statistics` |
| Source | Copernicus Climate Data Store (CDS) |
| Resolution | 0.25°, daily |
| Wind variables | `u10`, `v10` (daily mean) |
| Pressure | `msl` (daily mean) |
| Precipitation | `tp` (daily sum) |

### 4. GloFAS — Freshwater Forcing
| Property | Value |
|---|---|
| Dataset | `cems-glofas-historical` |
| Source | Copernicus Emergency Management Service (CEMS) |
| Resolution | 0.05°, daily |
| Variables | `dis24`, `ro`, `swvl` |
| Format | NetCDF-4 |

### 5. Bathymetry
| Property | Value |
|---|---|
| Product | GEBCO 2025 Global Grid |
| Source | [download.gebco.net](https://download.gebco.net) |
| Resolution | 15 arc-second (~450 m) |
| Variable | `elevation` (negative = ocean) |

**GEBCO download settings:**
- North: 22.5 · South: 5.5 · West: 79.5 · East: 95.5
- Layer: Bathymetry
- Format: NetCDF (Data)

---

## Repository Structure

```
blame-the-ocean/
├── pipeline.py        # Main orchestration — run this
├── config.py          # All settings, paths, variable lists, norm stats
├── loader.py          # Format detection, downloaders for all 4 sources
├── aligner.py         # Coordinate standardization, regridding, time alignment
├── masker.py          # Observation, land, bloom, MCAR, MNAR mask generation
├── normalizer.py      # log1p, z-score, min-max normalization
├── patcher.py         # Spatiotemporal patch extraction, train/val/test split
├── dataset.py         # PyTorch Dataset, DataLoader factory, sanity check
└── data/
    ├── raw/           # Downloaded source files (auto-created)
    ├── patches/
    │   ├── train/     # train_000000.npz ...
    │   ├── val/       # val_000000.npz ...
    │   └── test/      # test_000000.npz ...
    └── stats/
        └── norm_stats_bay_of_bengal.json
```

---

## Installation

```bash
pip install copernicusmarine cdsapi xarray netCDF4 numpy scipy \
            rioxarray h5py zarr dask pandas torch
```

Optional — for conservative regridding of discharge/precipitation:
```bash
pip install xesmf
```

---

## Authentication

### Copernicus Marine Service (CMEMS)
```bash
copernicusmarine login
```
Register at [marine.copernicus.eu](https://marine.copernicus.eu). Credentials are saved to `~/.copernicusmarine/`.

### Copernicus Climate Data Store (CDS)
Create `~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api
key: YOUR-API-KEY-HERE
```
Get your key from [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu) → Profile → API key.

---

## Configuration

All settings live in `config.py`. Key options:

### Date range
```python
DATE_START = "2021-01-01"
DATE_END   = "2025-12-31"
```

### Domain
```python
ACTIVE_DOMAIN = "bay_of_bengal"
```

### ERA5 download method
```python
# "daily_stats" — recommended: smaller downloads, pre-aggregated server-side
# "hourly"      — alternative: single combined request, 24x larger files
ERA5_DOWNLOAD_METHOD = "daily_stats"
```

### Bathymetry path (optional hardcode)
```python
# Set this to skip --bathy flag on every run
BATHY_PATH = "data/raw/gebco_2025_n22.5_s5.5_w79.5_e95.5.nc"
```

### Patch settings
```python
PATCH_SIZE       = 64     # spatial pixels H × W
PATCH_STRIDE     = 32     # overlap stride
TIME_WINDOW      = 10     # input timesteps T
FORECAST_HORIZON = 5      # forecast steps H
```

---

## Running the Pipeline

### Full run (downloads everything)

```bash
python pipeline.py --bathy data/raw/gebco_2025_n22.5_s5.5_w79.5_e95.5.nc
```

### Skip download — use existing local files

Required files differ by `ERA5_DOWNLOAD_METHOD` in `config.py`.

**`daily_stats` mode (default) — 3 separate ERA5 files:**
```bash
python pipeline.py \
    --no-download \
    --chl         data/raw/cmems_mod_glo_bgc_my_0.25deg_P1D-m_2021-01-01_2025-12-31.nc \
    --physics     data/raw/cmems_mod_glo_phy_my_0.083deg_P1D-m_2021-01-01_2025-12-31.nc \
    --era5-wind   data/raw/era5_wind_2021-01-01_2025-12-31.nc \
    --era5-msl    data/raw/era5_msl_2021-01-01_2025-12-31.nc \
    --era5-precip data/raw/era5_precip_2021-01-01_2025-12-31.nc \
    --discharge   data/raw/glofas_2021-01-01_2025-12-31.nc \
    --bathy       data/raw/gebco_2025_n22.5_s5.5_w79.5_e95.5.nc
```

**`hourly` mode — 1 combined ERA5 file (u10, v10, msl, tp together):**
```bash
python pipeline.py \
    --no-download \
    --chl       data/raw/cmems_mod_glo_bgc_my_0.25deg_P1D-m_2021-01-01_2025-12-31.nc \
    --physics   data/raw/cmems_mod_glo_phy_my_0.083deg_P1D-m_2021-01-01_2025-12-31.nc \
    --era5-wind data/raw/era5_hourly_2021-01-01_2025-12-31.nc \
    --discharge data/raw/glofas_2021-01-01_2025-12-31.nc \
    --bathy     data/raw/gebco_2025_n22.5_s5.5_w79.5_e95.5.nc
```
Note: `--era5-msl` and `--era5-precip` are not needed in hourly mode.
The combined file is identified by `--era5-wind` and contains all ERA5 variables.

### Reuse existing normalization stats

```bash
python pipeline.py --bathy data/raw/gebco_2025_n22.5_s5.5_w79.5_e95.5.nc --load-stats
```

### All CLI options

| Flag | Description |
|---|---|
| `--domain` | Domain preset from `config.py` (default: `bay_of_bengal`) |
| `--bathy` | Path to GEBCO bathymetry NetCDF |
| `--no-download` | Skip all downloads, use local files |
| `--chl` | Path to local BGC NetCDF |
| `--physics` | Path to local physics NetCDF |
| `--era5-wind` | ERA5 wind file. `daily_stats`: u10/v10 only. `hourly`: combined u10/v10/msl/tp |
| `--era5-msl` | ERA5 msl file (`daily_stats` mode only — not needed for `hourly`) |
| `--era5-precip` | ERA5 precip file (`daily_stats` mode only — not needed for `hourly`) |
| `--discharge` | Path to local GloFAS discharge NetCDF |
| `--load-stats` | Load existing norm stats instead of recomputing |

---

## Output Structure

### Patch tensors per `.npz` file

| Key | Shape | Description |
|---|---|---|
| `chl_obs` | `(T=10, H=64, W=64)` | Log-normalized Chl-a, NaN where missing |
| `obs_mask` | `(T=10, H=64, W=64)` | 1 = valid satellite observation |
| `mcar_mask` | `(T=10, H=64, W=64)` | 1 = MCAR missing pixel |
| `mnar_mask` | `(T=10, H=64, W=64)` | 1 = MNAR missing pixel |
| `physics` | `(T=10, C=6, H=64, W=64)` | thetao, uo, vo, mlotst, zos, so |
| `wind` | `(T=10, C=4, H=64, W=64)` | u10, v10, msl, tp |
| `discharge` | `(T=10, C=3, H=64, W=64)` | dis24, ro, swvl |
| `bgc_aux` | `(T=10, C=5, H=64, W=64)` | o2, no3, po4, si, nppv |
| `static` | `(C=2, H=64, W=64)` | bathymetry, distance-to-coast |
| `bloom_mask` | `(T=10, H=64, W=64)` | 1 = bloom event (ERI supervision) |
| `target_chl` | `(H=5, H=64, W=64)` | Future Chl-a (forecast target) |

**Derived at load time (not stored on disk):**

| Key | Shape | Description |
|---|---|---|
| `land_mask` | `(H=64, W=64)` | 1 = land pixel |
| `target_mask` | `(H=5, H=64, W=64)` | 1 = valid supervisable ocean pixel |

### Temporal split

| Split | Fraction | Period (approx.) |
|---|---|---|
| Train | 70% | 2021-01 → 2024-04 |
| Val | 15% | 2024-04 → 2024-10 |
| Test | 15% | 2024-10 → 2025-12 |

---

## ERA5 Download Methods

The pipeline supports two ERA5 download strategies, toggled via `config.py`:

### `daily_stats` (default — recommended)

Dataset: `derived-era5-single-levels-daily-statistics`

- Pre-aggregated daily values from the server — no post-processing needed
- Wind + pressure (u10, v10, msl): **daily mean**
- Precipitation (tp): **daily sum**
- Three separate CDS requests per month (each ≤2 variables = small-request category)
- Lighter queue, smaller file sizes

### `hourly` (alternative)

Dataset: `reanalysis-era5-single-levels`

- Raw hourly data — one combined request per month (u10, v10, msl, tp)
- Precipitation accumulated to daily totals after download
- Wind and msl daily-meaned from 24 hourly steps
- One CDS request per month — simpler but 24× larger files

---

## Resumability

All download functions are **fully resumable**. If the pipeline is interrupted at any point:

```bash
# Just rerun the same command — completed files are automatically skipped
python pipeline.py --bathy data/raw/gebco_2025_n22.5_s5.5_w79.5_e95.5.nc
```

Completed files are detected by filename. Corrupt or incomplete downloads are detected by attempting to open them with xarray and re-downloaded automatically.

---

## Verifying Output

Run the built-in sanity check after the pipeline completes:

```bash
python dataset.py
```

Expected output:
```
--- TRAIN ---
  chl_obs        (4, 10, 64, 64)          dtype=torch.float32
  obs_mask       (4, 10, 64, 64)          dtype=torch.float32
  mcar_mask      (4, 10, 64, 64)          dtype=torch.float32
  mnar_mask      (4, 10, 64, 64)          dtype=torch.float32
  physics        (4, 10, 6, 64, 64)       dtype=torch.float32
  wind           (4, 10, 4, 64, 64)       dtype=torch.float32
  discharge      (4, 10, 3, 64, 64)       dtype=torch.float32
  bgc_aux        (4, 10, 5, 64, 64)       dtype=torch.float32
  static         (4, 2, 64, 64)           dtype=torch.float32
  bloom_mask     (4, 10, 64, 64)          dtype=torch.float32
  target_chl     (4, 5, 64, 64)           dtype=torch.float32
  land_mask      (4, 64, 64)              dtype=torch.float32  (xx.x% land)
  target_mask    (4, 5, 64, 64)           dtype=torch.float32  (xx.x% supervisable)

Sanity check passed.
```

---

## Troubleshooting

### `CoordinatesOutOfDatasetBounds` on physics download
The depth range `0.0 → 1.0` clips to the nearest available level automatically. If this error appears, confirm `config.py` has:
```python
PHY_DEPTH_MIN = 0.0
PHY_DEPTH_MAX = 1.0
```

### `cost limits exceeded` from CDS
ERA5 daily-stats requests are size-limited. Ensure `ERA5_DOWNLOAD_METHOD = "daily_stats"` and each request covers only 1 month. Alternatively switch to `"hourly"` which has different (less strict) size limits.

### ERA5 CDS queue is slow
Normal — the derived daily-statistics dataset has a shared queue. Each request takes 4–8 minutes. With 60 monthly wind + 60 msl + 60 precip = 180 requests at ~5 min each, allow 15 hours. The pipeline runs unattended.

### `FutureWarning` from xarray on merge
Harmless deprecation notice. Does not affect data correctness. Will be resolved in a future xarray release.

### Normalization stats are placeholders
On the first run, `recompute_stats=True` (default) will recompute all stats from the training split and overwrite `data/stats/norm_stats_bay_of_bengal.json`. The `NORM_STATS` values in `config.py` are only used if `--load-stats` is passed before a stats file exists.

### Pipeline crashes mid-way through patching
Rerun the command. Downloads are cached. To restart only from Step 6 (patching), use `--no-download --load-stats`.

---

## Approximate Runtimes

| Step | Estimated Time |
|---|---|
| BGC download | 10–20 min |
| Physics download | 15–30 min |
| ERA5 wind (60 months) | 5–8 hours |
| ERA5 msl (60 months) | 5–8 hours |
| ERA5 precip (60 months) | 5–8 hours |
| GloFAS (5 years) | 30–60 min |
| Alignment + regridding | 10–20 min |
| Mask generation | 15–30 min |
| Normalization | 2–5 min |
| Patch extraction | 15–30 min |
| **Total** | **~1–2 days** |

> Leaving the pipeline running overnight (or across multiple days) is normal.
> Every step is resumable — no work is repeated on restart.