"""
config.py
---------
Central configuration for the MM-MARAS preprocessing pipeline.
All paths, variable names, normalization stats, and spatial/temporal
settings are defined here so every other module stays clean.
"""

from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Copernicus Marine Service (CMEMS) — BGC
# Product : GLOBAL_MULTIYEAR_BGC_001_029
# Dataset : cmems_mod_glo_bgc_my_0.25deg_P1D-m
# Coverage: 01/01/1993 → 31/01/2026  (fully covers 2020-2024, no interim needed)
# Res     : 0.25°, daily
# API tool: copernicusmarine  |  Auth: copernicusmarine login
# ---------------------------------------------------------------------------
COPERNICUS_BGC_DATASET = "cmems_mod_glo_bgc_my_0.25deg_P1D-m"

# BGC surface depth — request 0.0 to 1.0 to safely bracket the shallowest
# level regardless of its exact internal coordinate (nominally 0.51 m).
# CMEMS returns only the nearest available level within the requested range.
BGC_DEPTH_MIN = 0.0
BGC_DEPTH_MAX = 1.0

# All 6 BGC variables — full biogeochemical state vector for BoB
# o2   : tightly coupled to phytoplankton photosynthesis/respiration
# no3  : primary limiting nutrient; upwelling NO₃ directly triggers blooms
# po4  : co-limits growth alongside NO₃, especially near BoB river mouths
# si   : controls diatom blooms; BoB post-monsoon blooms are often diatom-dominated
# nppv : net primary production — mechanistically upstream of Chl-a accumulation
BGC_TARGET_VARIABLE = "chl"
BGC_AUX_VARIABLES   = ["o2", "no3", "po4", "si", "nppv"]
BGC_VARIABLES       = [BGC_TARGET_VARIABLE] + BGC_AUX_VARIABLES   # all 6 for download


# ---------------------------------------------------------------------------
# Copernicus Marine Service (CMEMS) — Physics
# Product : GLOBAL_MULTIYEAR_PHY_001_030
# Dataset : cmems_mod_glo_phy_my_0.083deg_P1D-m
# Coverage: 01/01/1993 → 24/02/2026  (fully covers 2020-2024, no interim needed)
# Res     : 0.083°, daily
# API tool: copernicusmarine  |  Auth: copernicusmarine login
# ---------------------------------------------------------------------------
COPERNICUS_PHY_DATASET = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

# Physics surface depth — request 0.0 to 1.0 to safely bracket the shallowest
# level regardless of its exact internal coordinate (0.49402... m).
# CMEMS returns only the nearest available level within the requested range.
PHY_DEPTH_MIN = 0.0
PHY_DEPTH_MAX = 1.0

# 5 original variables + salinity (so).
# so   : BoB has one of the world's strongest haloclines (Ganges/Brahmaputra/Irrawaddy
#        discharge). The halocline — not the thermocline — controls MLD and nutrient
#        access. Without so the model is blind to this dominant physical mechanism.
# Excluded: siconc, sithick, usi, vsi (sea ice — tropical domain, always zero)
#           bottomT (no surface-Chl signal at BoB depths)
PHY_VARIABLES = ["thetao", "uo", "vo", "mlotst", "zos", "so"]

# ---------------------------------------------------------------------------
# Bathymetry -- GEBCO 2025 Global Grid (primary source)
# Source : https://www.gebco.net/data_and_products/gridded_bathymetry_data/
# Tool   : GEBCO Grid Subsetting App (download as NetCDF)
# Res    : 15 arc-second (~450 m) -- regridded to 0.25 degrees by pipeline
# Subset : N=22.5  S=5.5  W=79.5  E=95.5  (matches ACTIVE_DOMAIN exactly)
#
# GEBCO elevation convention:
#   elevation < 0 = ocean (depth in metres, negative)
#   elevation > 0 = land
#   --> already matches masker.py (ocean < 0). NO negation applied.
#   Differs from CMEMS deptho (positive = depth) which DID require negation.
#
# How to download:
#   1. Go to https://download.gebco.net
#   2. Enter N=22.5  S=5.5  W=79.5  E=95.5
#   3. Select "GEBCO 2025 Global" -> NetCDF -> Download
#   4. Pass the file with --bathy, or set BATHY_PATH below.
# ---------------------------------------------------------------------------
BATHY_SOURCE = "gebco"         # "gebco" | "cmems" -- controls sign convention in pipeline
GEBCO_VAR    = "elevation"     # variable name inside the GEBCO NetCDF
BATHY_PATH   = None            # hardcode a path here to skip --bathy flag
                               # e.g. BATHY_PATH = "data/raw/gebco_2025_bob.nc"

# CMEMS static bathymetry (auto-downloaded fallback when no GEBCO file given)
# deptho: positive = depth below surface --> pipeline negates --> ocean < 0
COPERNICUS_PHY_BATHY_DATASET = "cmems_mod_glo_phy_my_0.083deg_static"
COPERNICUS_PHY_BATHY_LAYER   = "bathy"
COPERNICUS_PHY_BATHY_VAR     = "deptho"


# ---------------------------------------------------------------------------
# Copernicus Climate Data Store (CDS) — ERA5 daily statistics
# Dataset: derived-era5-single-levels-daily-statistics
# Coverage: 1940 → present
# Res     : 0.25°, daily (pre-aggregated server-side)
# API tool: cdsapi  |  Auth: ~/.cdsapirc
#
# KEY CHANGE from previous version:
#   Old dataset: reanalysis-era5-single-levels (raw hourly)
#     → required manual 24-step accumulation (accumulate_era5_precip_to_daily)
#     → 24× larger downloads
#   New dataset: derived-era5-single-levels-daily-statistics
#     → aggregation done server-side; no "time" key in request
#     → wind + pressure use daily_mean; precipitation uses daily_sum
#     → these CANNOT share one request (different daily_statistic)
#     → accumulate_era5_precip_to_daily() is deleted
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ERA5 download method toggle
# ---------------------------------------------------------------------------
# "daily_stats" : derived-era5-single-levels-daily-statistics (recommended)
#   + Pre-aggregated server-side — 24x smaller downloads
#   + No manual accumulation needed
#   - CDS queue can be slow (use <=2 variables per request to stay "small")
#   - Requests split: wind(u10,v10) + msl + precip(tp) = 3 separate downloads
#
# "hourly"      : reanalysis-era5-single-levels (original method)
#   + Single combined download for wind + precip
#   + Fewer CDS requests (1 per month covers all variables)
#   - 24x larger downloads (raw hourly data)
#   - Requires accumulate_era5_precip_to_daily() post-processing in pipeline
#   - Heavier on disk and memory during processing
# ---------------------------------------------------------------------------
ERA5_DOWNLOAD_METHOD  = "daily_stats"  # "daily_stats" | "hourly"

ERA5_DATASET          = "derived-era5-single-levels-daily-statistics"
ERA5_WIND_STATISTIC   = "daily_mean"
ERA5_PRECIP_STATISTIC = "daily_sum"
ERA5_FREQUENCY        = "1_hourly"     # source resolution used for aggregation
ERA5_TIMEZONE         = "UTC+00:00"
ERA5_RESOLUTION       = 0.25
ERA5_FORMAT           = "netcdf"

# Hourly method dataset (used when ERA5_DOWNLOAD_METHOD = "hourly")
ERA5_HOURLY_DATASET   = "reanalysis-era5-single-levels"

# Long-form variable names used in the CDS API request
ERA5_WIND_REQUEST_VARS   = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",    # added: monsoon onset/withdrawal drives Ekman pumping
]
ERA5_PRECIP_REQUEST_VARS = ["total_precipitation"]

# Short names as they appear in the downloaded NetCDF files
# NOTE: msl is downloaded separately (download_era5_msl) then merged at load time.
# Both requests stay <=2 variables to qualify as CDS "small" requests.
ERA5_WIND_VARIABLES    = ["u10", "v10"]    # downloaded by download_era5_wind
ERA5_MSL_VARIABLES     = ["msl"]           # downloaded by download_era5_msl
ERA5_WIND_ALL_VARS     = ["u10", "v10", "msl"]  # combined after merge in pipeline
ERA5_PRECIP_VARIABLES  = ["tp"]


# ---------------------------------------------------------------------------
# Copernicus Emergency Management Service (CEMS) — GloFAS river discharge
# Dataset: cems-glofas-historical
# Coverage: 1979 → 2026
# Res     : 0.05°, daily
# API tool: cdsapi  |  Endpoint: https://ewds.climate.copernicus.eu/api
#
# KEY CHANGE from previous version:
#   Old format: GRIB2 → required cfgrib + eccodes binary libraries
#   New format: NetCDF-4 (Experimental) → xarray loads directly, no binary deps
#
# New variables added:
#   ro   : runoff water equivalent — captures total land-surface water flux,
#          including smaller coastal rivers not in GloFAS routing
#   swvl : soil wetness index (root zone) — 1-4 week leading indicator of
#          discharge; gives the model advance warning of freshwater pulses
# ---------------------------------------------------------------------------
GLOFAS_DATASET        = "cems-glofas-historical"
GLOFAS_SYSTEM_VERSION = "version_4_0"
GLOFAS_MODEL          = "lisflood"
GLOFAS_PRODUCT_TYPE   = "consolidated"
GLOFAS_FORMAT         = "netcdf4"      # was "grib2" — cfgrib/eccodes no longer needed
GLOFAS_RESOLUTION     = 0.05

# Long-form CDS request variable names
# NOTE: soil_wetness_index_root_zone is only in the "intermediate" product type,
# not "consolidated". Excluded to avoid download errors.
GLOFAS_REQUEST_VARS = [
    "river_discharge_in_the_last_24_hours",
    "runoff_water_equivalent",
]

# Short names in downloaded NetCDF-4 files
# GloFAS NetCDF-4 uses "rowe" for runoff water equivalent (not "ro")
DISCHARGE_VARIABLES = ["dis24", "rowe"]

# Combined forcing variable list (all NetCDF short names)
FORCING_VARIABLES = ERA5_WIND_VARIABLES + ERA5_PRECIP_VARIABLES + DISCHARGE_VARIABLES


# ---------------------------------------------------------------------------
# Spatial domain presets
# ---------------------------------------------------------------------------
DOMAINS: Dict[str, Dict] = {
    "global": {
        "lon_min": -180, "lon_max": 180,
        "lat_min":  -90, "lat_max":  90,
    },
    "north_atlantic": {
        "lon_min": -80,  "lon_max":  20,
        "lat_min":  20,  "lat_max":  70,
    },
    "gulf_of_mexico": {
        "lon_min": -98,  "lon_max": -80,
        "lat_min":  18,  "lat_max":  31,
    },
    "chesapeake_bay": {
        "lon_min": -77.5, "lon_max": -75.5,
        "lat_min":  36.8, "lat_max":  39.6,
    },
    "baltic_sea": {
        "lon_min":   9.0, "lon_max":  30.0,
        "lat_min":  53.0, "lat_max":  66.0,
    },
    # Bounds follow the IHO official definition (IHO Publication S-23).
    # Boundary points extracted directly from the IHO text:
    #
    #   West  : Point Calimere, India        10°18'N  79°53'E  → lon_min = 79.5
    #   East  : Pulau Breueh, Sumatra         5°45'N  95°02'E  → lon_max = 95.5
    #           (Cape Negrais, Myanmar        16°03'N  94°12'E  also inside)
    #   South : Pulau Breueh                  5°45'N  95°02'E
    #           Dondra Head, Sri Lanka        5°55'N  80°35'E  → lat_min = 5.5
    #   North : northern Bangladesh coast                       → lat_max = 22.5
    #
    # The eastern boundary runs along the WESTERN coasts of Andaman and Nicobar
    # Islands so all narrow waters between the islands lie east of the line and
    # are excluded (Andaman Sea).  The rectangular bounding box necessarily
    # includes some Andaman Sea pixels — the aligner's land/ocean masking and
    # the masker's land_mask will correctly exclude these at patch level.
    "bay_of_bengal": {
        "lon_min":  79.5, "lon_max":  95.5,
        "lat_min":   5.5, "lat_max":  22.5,
    },
}

ACTIVE_DOMAIN = "bay_of_bengal"


# ---------------------------------------------------------------------------
# Temporal settings — 5 years, Bay of Bengal
# ---------------------------------------------------------------------------
DATE_START   = "2021-01-01"
DATE_END     = "2025-12-31"
TEMPORAL_RES = "1D"


# ---------------------------------------------------------------------------
# Spatial resolution & grid
# ---------------------------------------------------------------------------
TARGET_RESOLUTION = 0.25     # degrees — matches BGC product native resolution
REGRID_METHOD     = "bilinear"


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------
PATCH_SIZE       = 64
PATCH_STRIDE     = 32
TIME_WINDOW      = 10        # input steps T
FORECAST_HORIZON = 5         # forecast steps H


# ---------------------------------------------------------------------------
# Chlorophyll-a processing
# ---------------------------------------------------------------------------
CHL_LOG_OFFSET   = 1e-4
CHL_VALID_MIN    = 0.001     # mg/m³
CHL_VALID_MAX    = 100.0     # mg/m³
CHL_BLOOM_THRESH = 10.0      # mg/m³


# ---------------------------------------------------------------------------
# Per-variable normalization statistics
# Region: Bay of Bengal (80-100°E, 5-25°N)
# Period: training split of 2020-01-01 to 2024-12-31
#
# Values marked PLACEHOLDER must be recomputed after the first pipeline run.
# Copy updated values from: data/stats/norm_stats_bay_of_bengal.json
#
# Pre-transforms applied BEFORE z-scoring are defined in normalizer.PRETRANSFORMS.
# ---------------------------------------------------------------------------
# Computed from Bay of Bengal training split (2021-01-01 to 2024-04-09).
# Source: data/stats/norm_stats_bay_of_bengal.json — generated by pipeline.py
# Pre-transforms applied before z-scoring are defined in normalizer.PRETRANSFORMS.
NORM_STATS: Dict[str, Dict[str, float]] = {

    # --- BGC target (log1p-transformed) ---
    "chl": {
        "mean":  0.1450,
        "std":   0.2077,
    },

    # --- BGC auxiliary (all log1p-transformed) ---
    "o2": {             # dissolved oxygen (mmol/m³)
        "mean":  5.3173,
        "std":   0.0393,
    },
    "no3": {            # nitrate (mmol/m³)
        "mean":  0.3802,
        "std":   0.7295,
    },
    "po4": {            # phosphate (mmol/m³)
        "mean":  0.0171,
        "std":   0.0371,
    },
    "si": {             # silicate (mmol/m³)
        "mean":  1.6466,
        "std":   0.4374,
    },
    "nppv": {           # net primary production (mg/m³/day)
        "mean":  1.6954,
        "std":   0.7910,
    },

    # --- Physics ---
    "thetao": {         # SST (°C)
        "mean": 29.0735,
        "std":   1.2650,
    },
    "uo": {             # zonal current (m/s)
        "mean": -0.0000301,
        "std":   0.2496,
    },
    "vo": {             # meridional current (m/s)
        "mean":  0.0078,
        "std":   0.2292,
    },
    "mlotst": {         # MLD (m), log1p-transformed
        "mean":  2.8626,
        "std":   0.4374,
    },
    "zos": {            # sea surface height (m)
        "mean":  0.6049,
        "std":   0.1042,
    },
    "so": {             # salinity (PSU)
        "mean": 32.0322,
        "std":   2.8256,
    },

    # --- ERA5 wind / atmospheric (daily mean) ---
    "u10": {            # zonal 10m wind (m/s)
        "mean":  0.6337,
        "std":   3.9110,
    },
    "v10": {            # meridional 10m wind (m/s)
        "mean":  0.9823,
        "std":   3.5697,
    },
    "msl": {            # mean sea level pressure (Pa)
        "mean": 100930.27,
        "std":    370.02,
    },

    # --- ERA5 precipitation (daily sum, log1p-transformed) ---
    "tp": {             # total precipitation (mm/day)
        "mean":  1.0281,
        "std":   1.1255,
    },

    # --- GloFAS freshwater forcing (all log1p-transformed) ---
    "dis24": {          # river discharge (m³/s)
        "mean":  1.3041,
        "std":   1.6450,
    },
    "rowe": {           # runoff water equivalent (m/day)
        "mean":  0.6349,
        "std":   0.7261,
    },

    # Aliases kept for compatibility
    "uas": {"mean":  0.6337, "std":  3.9110},
    "vas": {"mean":  0.9823, "std":  3.5697},
}
# Static variables (min-max scaled, not z-scored)
STATIC_VARIABLES = ["bathymetry", "distance_to_coast"]


# ---------------------------------------------------------------------------
# Mask settings
# ---------------------------------------------------------------------------
MCAR_SPATIAL_CORR_THRESHOLD = 0.1
MNAR_CHL_CORR_THRESHOLD     = 0.3
MIN_VALID_FRACTION          = 0.10


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
RAW_DATA_DIR        = "data/raw"
INTERIM_DATA_DIR    = "data/interim"
PROCESSED_DATA_DIR  = "data/processed"
PATCHES_DIR         = "data/patches"
STATS_DIR           = "data/stats"

OUTPUT_FORMAT = "zarr"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42