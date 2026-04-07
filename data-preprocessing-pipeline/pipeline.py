"""
pipeline.py
-----------
Main preprocessing pipeline for MM-MARAS.

Orchestrates all preprocessing steps in order:
    1.  Download / load raw data
    2.  Standardize coordinates
    3.  Clip to spatial domain
    4.  Temporal alignment
    5.  Spatial regridding to common Chl-a grid
    6.  Log-transform and normalization
    7.  Mask generation (obs, land, bloom, MCAR, MNAR)
    8.  Static context assembly (bathymetry auto-downloaded from CMEMS)
    9.  Patch extraction and train/val/test split
    10. Save to disk

Run from the command line:
    python pipeline.py

Or import and call run_pipeline() programmatically.
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

import config as cfg
from aligner import align_all_modalities
from loader import (
    load,
    load_time_series,
    download_copernicus,
    download_copernicus_static,
    download_glofas,
    download_era5_wind,
    download_era5_msl,
    download_era5_precip,
    download_era5_wind_hourly,
    accumulate_era5_precip_to_daily,
    load_glofas,
    print_dataset_summary,
)
from masker import build_all_masks
from normalizer import compute_stats, save_stats, load_stats, normalize_dataset
from patcher import PatchExtractor, save_patches, temporal_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Step 1: Data acquisition
# ---------------------------------------------------------------------------

def step_download(domain: dict, bathy_path_override: Optional[str] = None) -> dict:
    """
    Download all required data.

    Sources
    -------
    BGC Chl-a + aux  → CMEMS  cmems_mod_glo_bgc_my_0.25deg_P1D-m
    Physics          → CMEMS  cmems_mod_glo_phy_my_0.083deg_P1D-m
    Bathymetry       → CMEMS  cmems_mod_glo_phy_my_0.083deg_static  (auto-downloaded)
    ERA5 wind + msl  → CDS    derived-era5-single-levels-daily-statistics (daily_mean)
    ERA5 precip      → CDS    derived-era5-single-levels-daily-statistics (daily_sum)
    GloFAS discharge → CEMS   cems-glofas-historical (NetCDF-4)

    Parameters
    ----------
    bathy_path_override : if provided, skip CMEMS bathymetry download and use
                          this local file directly.
    """
    logger.info("STEP 1: Downloading data")
    paths = {}

    # BGC — all 6 variables; surface only (depth 0.51 m)
    paths["chl"] = download_copernicus(
        dataset_id  = cfg.COPERNICUS_BGC_DATASET,
        variables   = cfg.BGC_VARIABLES,
        date_start  = cfg.DATE_START,
        date_end    = cfg.DATE_END,
        lon_min     = domain["lon_min"],
        lon_max     = domain["lon_max"],
        lat_min     = domain["lat_min"],
        lat_max     = domain["lat_max"],
        depth_min   = cfg.BGC_DEPTH_MIN,
        depth_max   = cfg.BGC_DEPTH_MAX,
        output_dir  = cfg.RAW_DATA_DIR,
        skip_depth  = False,
    )

    # Physics — 6 variables (added so); surface only (depth 0.49 m)
    paths["physics"] = download_copernicus(
        dataset_id  = cfg.COPERNICUS_PHY_DATASET,
        variables   = cfg.PHY_VARIABLES,
        date_start  = cfg.DATE_START,
        date_end    = cfg.DATE_END,
        lon_min     = domain["lon_min"],
        lon_max     = domain["lon_max"],
        lat_min     = domain["lat_min"],
        lat_max     = domain["lat_max"],
        depth_min   = cfg.PHY_DEPTH_MIN,
        depth_max   = cfg.PHY_DEPTH_MAX,
        output_dir  = cfg.RAW_DATA_DIR,
        skip_depth  = False,
    )

    # Bathymetry -- resolution order:
    #   1. CLI/caller override (--bathy flag)
    #   2. cfg.BATHY_PATH constant (point to your GEBCO file in config.py)
    #   3. Auto-download CMEMS static bathymetry (fallback)
    resolved_bathy = bathy_path_override or cfg.BATHY_PATH
    if resolved_bathy and Path(str(resolved_bathy)).exists():
        paths["bathy"] = Path(str(resolved_bathy))
        logger.info(f"Using bathymetry: {paths['bathy']} (source: {cfg.BATHY_SOURCE})")
    else:
        logger.info("No GEBCO file provided -- downloading CMEMS static bathymetry as fallback")
        paths["bathy"] = download_copernicus_static(
            dataset_id  = cfg.COPERNICUS_PHY_BATHY_DATASET,
            variables   = [cfg.COPERNICUS_PHY_BATHY_VAR],
            lon_min     = domain["lon_min"],
            lon_max     = domain["lon_max"],
            lat_min     = domain["lat_min"],
            lat_max     = domain["lat_max"],
            output_dir  = cfg.RAW_DATA_DIR,
        )

    # ERA5 — method controlled by cfg.ERA5_DOWNLOAD_METHOD
    # "daily_stats" : 3 separate small requests (u10/v10, msl, tp) — default
    # "hourly"      : 1 combined request per month (u10, v10, tp together, raw hourly)
    if cfg.ERA5_DOWNLOAD_METHOD == "hourly":
        logger.info("ERA5 download method: hourly (reanalysis-era5-single-levels)")
        # Single combined download — wind + precip together, raw hourly
        paths["era5_wind"]   = download_era5_wind_hourly(
            date_start = cfg.DATE_START,
            date_end   = cfg.DATE_END,
            lon_min    = domain["lon_min"],
            lon_max    = domain["lon_max"],
            lat_min    = domain["lat_min"],
            lat_max    = domain["lat_max"],
            output_dir = cfg.RAW_DATA_DIR,
        )
        # Hourly method combines wind + precip in one file; msl not included
        paths["era5_msl"]    = None
        paths["era5_precip"] = paths["era5_wind"]   # same file contains tp
    else:
        logger.info("ERA5 download method: daily_stats (derived-era5-single-levels-daily-statistics)")
        # u10, v10 — daily_mean, <=2 variables = CDS small request
        paths["era5_wind"] = download_era5_wind(
            date_start = cfg.DATE_START,
            date_end   = cfg.DATE_END,
            lon_min    = domain["lon_min"],
            lon_max    = domain["lon_max"],
            lat_min    = domain["lat_min"],
            lat_max    = domain["lat_max"],
            output_dir = cfg.RAW_DATA_DIR,
        )
        # msl — separate request so wind stays <=2 variables
        paths["era5_msl"] = download_era5_msl(
            date_start = cfg.DATE_START,
            date_end   = cfg.DATE_END,
            lon_min    = domain["lon_min"],
            lon_max    = domain["lon_max"],
            lat_min    = domain["lat_min"],
            lat_max    = domain["lat_max"],
            output_dir = cfg.RAW_DATA_DIR,
        )
        # tp — daily_sum, separate request
        paths["era5_precip"] = download_era5_precip(
            date_start = cfg.DATE_START,
            date_end   = cfg.DATE_END,
            lon_min    = domain["lon_min"],
            lon_max    = domain["lon_max"],
            lat_min    = domain["lat_min"],
            lat_max    = domain["lat_max"],
            output_dir = cfg.RAW_DATA_DIR,
        )

    # GloFAS discharge — NetCDF-4 (dis24, rowe)
    paths["discharge"] = download_glofas(
        date_start = cfg.DATE_START,
        date_end   = cfg.DATE_END,
        lon_min    = domain["lon_min"],
        lon_max    = domain["lon_max"],
        lat_min    = domain["lat_min"],
        lat_max    = domain["lat_max"],
        output_dir = cfg.RAW_DATA_DIR,
    )

    return paths


# ---------------------------------------------------------------------------
# Step 2: Load and align all modalities
# ---------------------------------------------------------------------------

def step_load_and_align(paths: dict, domain: dict) -> dict:
    """
    Load raw files and align all modalities to the Chl-a grid and time axis.
    Returns dict of aligned xr.Datasets.

    Key structure:
        "chl"       → xr.Dataset  with chl, o2, no3, po4, si, nppv
        "physics"   → xr.Dataset  with thetao, uo, vo, mlotst, zos, so
        "wind"      → xr.Dataset  with u10, v10, msl
        "precip"    → xr.Dataset  with tp
        "discharge" → xr.Dataset  with dis24, ro, swvl
    """
    logger.info("STEP 2: Loading and aligning modalities")

    # Full BGC dataset — all 6 variables; defines the target grid (0.25°)
    chl_ds = load(paths["chl"], variables=cfg.BGC_VARIABLES)

    physics_ds = load(paths["physics"], variables=cfg.PHY_VARIABLES)

    # ERA5 loading — branches on cfg.ERA5_DOWNLOAD_METHOD
    if cfg.ERA5_DOWNLOAD_METHOD == "hourly":
        # One file contains u10, v10, tp (raw hourly) — split and accumulate
        era5_hourly_ds = load(paths["era5_wind"])    # same file holds everything
        # Hourly file contains u10, v10, msl, tp at hourly resolution.
        # Wind and msl need daily-mean; precip needs daily-sum accumulation.
        wind_ds = era5_hourly_ds[["u10", "v10", "msl"]].resample(time="1D").mean(skipna=True)
        precip_ds = accumulate_era5_precip_to_daily(
            era5_hourly_ds[["tp"]], var="tp"
        )
    else:
        # Three separate files: wind (u10,v10), msl, precip (tp) — merge wind+msl
        wind_uv_ds = load(paths["era5_wind"], variables=["u10", "v10"])
        msl_ds     = load(paths["era5_msl"],  variables=["msl"])
        wind_ds    = xr.merge([wind_uv_ds, msl_ds])   # combined (u10, v10, msl)
        precip_ds  = load(paths["era5_precip"], variables=cfg.ERA5_PRECIP_VARIABLES)
        # No accumulation needed — daily statistics dataset provides daily sums

    discharge_ds = load_glofas(paths["discharge"])

    print_dataset_summary(chl_ds,       label="Raw BGC (all 6 variables)")
    print_dataset_summary(physics_ds,   label="Raw Physics (6 variables incl. so)")
    print_dataset_summary(wind_ds,      label="Raw ERA5 Wind+MSL (daily mean)")
    print_dataset_summary(precip_ds,    label="Raw ERA5 Precipitation (daily sum)")
    print_dataset_summary(discharge_ds, label="Raw GloFAS (dis24, rowe)")

    aligned = align_all_modalities(
        chl_ds       = chl_ds,
        physics_ds   = physics_ds,
        wind_ds      = wind_ds,
        discharge_ds = discharge_ds,
        precip_ds    = precip_ds,
        domain       = domain,
        regrid_method = cfg.REGRID_METHOD,
    )

    return aligned


# ---------------------------------------------------------------------------
# Step 3: Build masks
# ---------------------------------------------------------------------------

def step_build_masks(aligned: dict) -> xr.Dataset:
    """
    Generate all binary masks from the Chl-a DataArray.
    aligned["chl"] now contains all BGC variables — chl is still present.
    """
    logger.info("STEP 3: Building masks")

    chl_da = aligned["chl"]["chl"]
    if "depth" in chl_da.dims:
        chl_da = chl_da.isel(depth=0, drop=True)

    mask_ds = build_all_masks(
        chl             = chl_da,
        valid_min       = cfg.CHL_VALID_MIN,
        valid_max       = cfg.CHL_VALID_MAX,
        bloom_threshold = cfg.CHL_BLOOM_THRESH,
        mcar_threshold  = cfg.MCAR_SPATIAL_CORR_THRESHOLD,
        mnar_threshold  = cfg.MNAR_CHL_CORR_THRESHOLD,
    )
    return mask_ds


# ---------------------------------------------------------------------------
# Step 4: Normalize
# ---------------------------------------------------------------------------

def step_normalize(
    aligned: dict,
    mask_ds: xr.Dataset,
    stats_path: str,
    recompute_stats: bool = True,
) -> dict:
    """
    Compute normalization statistics from training data only, then normalize
    all modalities.

    Returns a dict with keys:
        "chl"        → normalized chl only (target variable)
        "bgc_aux"    → normalized o2, no3, po4, si, nppv
        "physics"    → normalized thetao, uo, vo, mlotst, zos, so
        "wind"       → normalized u10, v10, msl
        "precip"     → normalized tp
        "discharge"  → normalized dis24, rowe
        "stats"      → the normalization statistics dict
    """
    logger.info("STEP 4: Normalizing")

    if recompute_stats:
        T_total = aligned["chl"].sizes["time"]
        train_slice, _, _ = temporal_split(T_total)
        obs_mask_train = mask_ds["obs_mask"].isel(time=train_slice)

        # Split BGC dataset into target and auxiliary for separate stat computation
        def _surface(ds, var):
            da = ds[var]
            return da.isel(depth=0, drop=True) if "depth" in da.dims else da

        chl_ds_surface = xr.Dataset({"chl": _surface(aligned["chl"], "chl")})
        bgc_aux_ds     = xr.Dataset({v: _surface(aligned["chl"], v) for v in cfg.BGC_AUX_VARIABLES})

        chl_stats       = compute_stats(chl_ds_surface.isel(time=train_slice),
                                        [cfg.BGC_TARGET_VARIABLE], obs_mask=obs_mask_train)
        bgc_aux_stats   = compute_stats(bgc_aux_ds.isel(time=train_slice),
                                        cfg.BGC_AUX_VARIABLES)
        phy_stats       = compute_stats(aligned["physics"].isel(time=train_slice),
                                        cfg.PHY_VARIABLES)
        wind_stats      = compute_stats(aligned["wind"].isel(time=train_slice),
                                        cfg.ERA5_WIND_ALL_VARS)   # u10, v10, msl
        precip_stats    = compute_stats(aligned["precip"].isel(time=train_slice),
                                        cfg.ERA5_PRECIP_VARIABLES)
        discharge_stats = compute_stats(aligned["discharge"].isel(time=train_slice),
                                        cfg.DISCHARGE_VARIABLES)

        all_stats = {
            **chl_stats, **bgc_aux_stats, **phy_stats,
            **wind_stats, **precip_stats, **discharge_stats,
        }
        os.makedirs(cfg.STATS_DIR, exist_ok=True)
        save_stats(all_stats, stats_path)

    else:
        all_stats = load_stats(stats_path)

    # Helper: strip depth if present, then normalize
    def _prep_and_norm(ds, var_list):
        out = {}
        for v in var_list:
            da = ds[v]
            if "depth" in da.dims:
                da = da.isel(depth=0, drop=True)
            out[v] = da
        return normalize_dataset(xr.Dataset(out), all_stats, var_list)

    chl_norm       = _prep_and_norm(aligned["chl"],       [cfg.BGC_TARGET_VARIABLE])
    bgc_aux_norm   = _prep_and_norm(aligned["chl"],       cfg.BGC_AUX_VARIABLES)
    physics_norm   = _prep_and_norm(aligned["physics"],   cfg.PHY_VARIABLES)
    wind_norm      = _prep_and_norm(aligned["wind"],      cfg.ERA5_WIND_ALL_VARS)  # u10, v10, msl
    precip_norm    = _prep_and_norm(aligned["precip"],    cfg.ERA5_PRECIP_VARIABLES)
    discharge_norm = _prep_and_norm(aligned["discharge"], cfg.DISCHARGE_VARIABLES)

    return {
        "chl":       chl_norm,
        "bgc_aux":   bgc_aux_norm,
        "physics":   physics_norm,
        "wind":      wind_norm,
        "precip":    precip_norm,
        "discharge": discharge_norm,
        "stats":     all_stats,
    }


# ---------------------------------------------------------------------------
# Step 5: Build static context arrays
# ---------------------------------------------------------------------------

def step_build_static(
    chl_ds: xr.Dataset,
    bathy_path: Optional[str] = None,
) -> np.ndarray:
    """
    Assemble static context channels: bathymetry and distance-to-coast.

    Bathymetry source priority
    --------------------------
    1. GEBCO NetCDF passed via --bathy or cfg.BATHY_PATH  (cfg.BATHY_SOURCE = "gebco")
       GEBCO elevation: negative = ocean, positive = land  --> NO negation applied.
    2. CMEMS static sub-dataset (auto-downloaded fallback) (cfg.BATHY_SOURCE = "cmems")
       deptho: positive = depth below surface --> NEGATED so ocean < 0.
    3. Zeros placeholder if neither is available.

    masker.py build_land_mask convention: ocean < 0, land >= 0.

    Returns
    -------
    static_arr : (2, H, W) float32, min-max normalized per channel
    """
    logger.info("STEP 5: Building static context")

    H = chl_ds.sizes["lat"]
    W = chl_ds.sizes["lon"]

    # Resolve bathy path: CLI/caller arg > config constant > None
    resolved_bathy = bathy_path or cfg.BATHY_PATH

    if resolved_bathy and Path(resolved_bathy).exists():
        bathy_ds = load(resolved_bathy)
        from aligner import standardize_coords, regrid_to_target
        bathy_ds = standardize_coords(bathy_ds)
        bathy_ds = regrid_to_target(bathy_ds, chl_ds, method="bilinear")

        source = cfg.BATHY_SOURCE.lower()
        var_name = cfg.GEBCO_VAR if source == "gebco" else cfg.COPERNICUS_PHY_BATHY_VAR
        if var_name not in bathy_ds:
            var_name = list(bathy_ds.data_vars)[0]
            logger.warning(f"Expected bathy variable not found; falling back to '{var_name}'")

        bathy_da = bathy_ds[var_name]
        if "time" in bathy_da.dims:
            bathy_da = bathy_da.isel(time=0)
        bathy_arr = bathy_da.values.astype(np.float32)

        if source == "gebco":
            # GEBCO elevation is already negative for ocean -- no change needed
            logger.info(f"GEBCO bathymetry loaded: {resolved_bathy} (no negation)")
        else:
            # CMEMS deptho is positive for ocean -- negate to match masker convention
            bathy_arr = -bathy_arr
            logger.info(f"CMEMS bathymetry loaded: {resolved_bathy} (negated)")
    else:
        logger.warning("No bathymetry file found; using zeros as placeholder.")
        bathy_arr = np.zeros((H, W), dtype=np.float32)

    # Distance-to-coast from land mask (Euclidean distance transform)
    from masker import build_land_mask
    chl_da_static = chl_ds["chl"]
    if "depth" in chl_da_static.dims:
        chl_da_static = chl_da_static.isel(depth=0, drop=True)
    land_mask  = build_land_mask(chl_da_static)
    ocean_bool = land_mask.values.astype(bool)
    from scipy import ndimage
    dist_arr = ndimage.distance_transform_edt(ocean_bool).astype(np.float32)

    static_arr = np.stack([bathy_arr, dist_arr], axis=0)   # (2, H, W)

    # Min-max normalize each channel independently
    for c in range(static_arr.shape[0]):
        ch     = static_arr[c]
        finite = ch[np.isfinite(ch)]
        if len(finite) > 0:
            vmin, vmax = finite.min(), finite.max()
            static_arr[c] = (ch - vmin) / max(vmax - vmin, 1e-8)

    logger.info(f"Static context shape: {static_arr.shape}")
    return static_arr


# ---------------------------------------------------------------------------
# Step 6: Extract and save patches
# ---------------------------------------------------------------------------

def step_extract_patches(
    normalized: dict,
    mask_ds: xr.Dataset,
    static_arr: np.ndarray,
) -> None:
    """
    Stack all modalities into NumPy arrays, split into train/val/test,
    and save patches as .npz files.

    Tensor layout
    -------------
    physics  : (T, 6, H, W)  — thetao, uo, vo, mlotst, zos, so
    wind     : (T, 4, H, W)  — u10, v10, msl, tp
    discharge: (T, 2, H, W)  — dis24, rowe
    bgc_aux  : (T, 5, H, W)  — o2, no3, po4, si, nppv   ← new tensor
    """
    logger.info("STEP 6: Extracting and saving patches")

    chl_var = normalized["chl"]["chl"]
    if "depth" in chl_var.dims:
        chl_var = chl_var.isel(depth=0, drop=True)

    chl_np   = chl_var.values
    obs_np   = mask_ds["obs_mask"].values
    mcar_np  = mask_ds["mcar_mask"].values
    mnar_np  = mask_ds["mnar_mask"].values
    bloom_np = mask_ds["bloom_mask"].values
    land_np  = mask_ds["land_mask"].values

    # Physics: (T, 6, H, W)
    physics_np = np.stack(
        [normalized["physics"][v].values for v in cfg.PHY_VARIABLES],
        axis=1,
    ).astype(np.float32)

    # Wind tensor: (T, 4, H, W)  — u10, v10, msl (ERA5 daily_mean) + tp (ERA5 daily_sum)
    wind_np = np.stack([
        normalized["wind"]["u10"].values,
        normalized["wind"]["v10"].values,
        normalized["wind"]["msl"].values,
        normalized["precip"]["tp"].values,
    ], axis=1).astype(np.float32)

    # Discharge tensor: (T, 2, H, W)  — dis24, rowe
    discharge_np = np.stack(
        [normalized["discharge"][v].values for v in cfg.DISCHARGE_VARIABLES],
        axis=1,
    ).astype(np.float32)

    # BGC auxiliary tensor: (T, 5, H, W)  — o2, no3, po4, si, nppv
    bgc_aux_np = np.stack(
        [normalized["bgc_aux"][v].values for v in cfg.BGC_AUX_VARIABLES],
        axis=1,
    ).astype(np.float32)

    lats  = chl_var.lat.values
    lons  = chl_var.lon.values
    times = chl_var.time.values

    T_total = len(times)
    train_sl, val_sl, test_sl = temporal_split(T_total)

    extractor = PatchExtractor(
        patch_size       = cfg.PATCH_SIZE,
        stride           = cfg.PATCH_STRIDE,
        time_window      = cfg.TIME_WINDOW,
        forecast_horizon = cfg.FORECAST_HORIZON,
        min_valid_frac   = cfg.MIN_VALID_FRACTION,
        land_mask        = land_np,
    )

    os.makedirs(cfg.PATCHES_DIR, exist_ok=True)

    for split, sl in [("train", train_sl), ("val", val_sl), ("test", test_sl)]:
        logger.info(f"Extracting {split} patches...")
        count = save_patches(
            extractor    = extractor,
            chl_norm     = chl_np[sl],
            obs_mask     = obs_np[sl],
            mcar_mask    = mcar_np[sl],
            mnar_mask    = mnar_np[sl],
            physics      = physics_np[sl],
            wind         = wind_np[sl],
            discharge    = discharge_np[sl],
            bgc_aux      = bgc_aux_np[sl],
            static       = static_arr,
            bloom_mask   = bloom_np[sl],
            lats         = lats,
            lons         = lons,
            times        = times[sl],
            output_dir   = cfg.PATCHES_DIR,
            split        = split,
        )
        logger.info(f"{split}: {count} patches saved")


# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    domain_name: str = cfg.ACTIVE_DOMAIN,
    bathymetry_path: Optional[str] = None,
    download: bool = True,
    chl_path: Optional[str] = None,
    physics_path: Optional[str] = None,
    era5_wind_path: Optional[str] = None,
    era5_msl_path: Optional[str] = None,
    era5_precip_path: Optional[str] = None,
    discharge_path: Optional[str] = None,
    recompute_stats: bool = True,
) -> None:
    """
    Run the full MM-MARAS preprocessing pipeline.

    Parameters
    ----------
    domain_name       : key in cfg.DOMAINS
    bathymetry_path   : optional local bathymetry file; auto-downloaded from
                        CMEMS if not provided
    download          : if True, download all data from remote sources
    chl_path          : local BGC file (required when download=False)
    physics_path      : local physics file (required when download=False)
    era5_wind_path    : local ERA5 wind file.
                        daily_stats mode: u10/v10 only (required when download=False)
                        hourly mode    : combined u10/v10/msl/tp file (required)
    era5_msl_path     : local ERA5 msl file (daily_stats mode only — not needed for hourly)
    era5_precip_path  : local ERA5 precip file (daily_stats mode only — not needed for hourly)
    discharge_path    : local GloFAS file (required when download=False)
    recompute_stats   : if True, recompute normalization stats from training data
    """
    domain = cfg.DOMAINS[domain_name]
    logger.info(
        f"Starting MM-MARAS preprocessing pipeline | "
        f"domain: {domain_name} | dates: {cfg.DATE_START} to {cfg.DATE_END}"
    )

    if download:
        paths = step_download(domain, bathy_path_override=bathymetry_path)
    else:
        # Required paths differ by ERA5 download method:
        #   "daily_stats" : era5_wind (u10/v10) + era5_msl + era5_precip (3 files)
        #   "hourly"      : era5_wind (u10/v10/msl/tp combined — 1 file)
        #                   era5_msl and era5_precip are not needed
        missing = []
        if not chl_path:       missing.append("chl_path")
        if not physics_path:   missing.append("physics_path")
        if not era5_wind_path: missing.append("era5_wind_path")
        if not discharge_path: missing.append("discharge_path")

        if cfg.ERA5_DOWNLOAD_METHOD != "hourly":
            # daily_stats needs separate msl and precip files
            if not era5_msl_path:    missing.append("era5_msl_path")
            if not era5_precip_path: missing.append("era5_precip_path")

        if missing:
            raise ValueError(
                f"When download=False (method={cfg.ERA5_DOWNLOAD_METHOD!r}), "
                f"you must provide: {', '.join(missing)}"
            )

        if cfg.ERA5_DOWNLOAD_METHOD == "hourly":
            # Single combined file — era5_wind holds u10, v10, msl, tp
            paths = {
                "chl":         chl_path,
                "physics":     physics_path,
                "bathy":       bathymetry_path,
                "era5_wind":   era5_wind_path,   # combined file
                "era5_msl":    None,              # not needed — msl is inside era5_wind
                "era5_precip": era5_wind_path,    # same combined file contains tp
                "discharge":   discharge_path,
            }
        else:
            paths = {
                "chl":         chl_path,
                "physics":     physics_path,
                "bathy":       bathymetry_path,
                "era5_wind":   era5_wind_path,
                "era5_msl":    era5_msl_path,
                "era5_precip": era5_precip_path,
                "discharge":   discharge_path,
            }

    aligned    = step_load_and_align(paths, domain)
    mask_ds    = step_build_masks(aligned)

    stats_path = os.path.join(cfg.STATS_DIR, f"norm_stats_{domain_name}.json")
    normalized = step_normalize(aligned, mask_ds, stats_path, recompute_stats)

    static_arr = step_build_static(aligned["chl"], paths.get("bathy"))

    step_extract_patches(normalized, mask_ds, static_arr)

    logger.info("Preprocessing pipeline complete.")
    logger.info(f"Patches saved to: {cfg.PATCHES_DIR}")
    logger.info(f"Normalization stats at: {stats_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MM-MARAS Data Preprocessing Pipeline")
    parser.add_argument("--domain",       type=str, default=cfg.ACTIVE_DOMAIN,
                        choices=list(cfg.DOMAINS.keys()))
    parser.add_argument("--no-download",  action="store_true",
                        help="Skip download; use local files")
    parser.add_argument("--chl",          type=str, default=None)
    parser.add_argument("--physics",      type=str, default=None)
    parser.add_argument("--era5-wind",    type=str, default=None,
                        help="Path to ERA5 wind NetCDF. "
                             "daily_stats: u10/v10 only. "
                             "hourly: combined u10/v10/msl/tp file.")
    parser.add_argument("--era5-msl",     type=str, default=None,
                        help="Path to ERA5 msl NetCDF (daily_stats mode only — not needed for hourly)")
    parser.add_argument("--era5-precip",  type=str, default=None,
                        help="Path to ERA5 precipitation NetCDF (daily_stats mode only — not needed for hourly)")
    parser.add_argument("--discharge",    type=str, default=None,
                        help="Path to GloFAS discharge NetCDF")
    parser.add_argument("--bathy",        type=str, default=None,
                        help="Path to bathymetry file (auto-downloaded if omitted)")
    parser.add_argument("--load-stats",   action="store_true",
                        help="Load existing normalization stats instead of recomputing")
    args = parser.parse_args()

    run_pipeline(
        domain_name      = args.domain,
        bathymetry_path  = args.bathy,
        download         = not args.no_download,
        chl_path         = args.chl,
        physics_path     = args.physics,
        era5_wind_path   = args.era5_wind,
        era5_msl_path    = args.era5_msl,
        era5_precip_path = args.era5_precip,
        discharge_path   = args.discharge,
        recompute_stats  = not args.load_stats,
    )