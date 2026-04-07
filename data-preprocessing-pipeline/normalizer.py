"""
normalizer.py
-------------
Variable-specific normalization for all MM-MARAS input channels.

Chl-a:      log1p transform → z-score standardization
BGC aux:    log1p → z-score  (o2, no3, po4, si, nppv — all right-skewed)
Physics:    z-score (thetao, uo, vo, zos, so)
            log1p → z-score  (mlotst — right-skewed)
Wind/atm:   z-score (u10, v10, msl)
Precip:     log1p → z-score  (tp — zero-inflated, heavy tail)
Discharge:  log1p → z-score  (dis24, rowe — heavy-tailed)
            Note: swvl removed — not present in consolidated GloFAS product
Static:     min-max scaling  (bathymetry, distance-to-coast)

All statistics are computed from training data only and saved to disk
so the same stats can be applied at inference without data leakage.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-variable transform definitions
# ---------------------------------------------------------------------------

PRETRANSFORMS = {
    # BGC target
    "chl":       "log1p",

    # BGC auxiliary — all right-skewed positive distributions
    "o2":        "log1p",
    "no3":       "log1p",
    "po4":       "log1p",
    "si":        "log1p",
    "nppv":      "log1p",

    # Physics
    "thetao":    "none",
    "uo":        "none",
    "vo":        "none",
    "mlotst":    "log1p",   # MLD is right-skewed
    "zos":       "none",
    "so":        "none",    # salinity — roughly normal

    # ERA5 atmospheric (daily mean)
    "u10":       "none",
    "v10":       "none",
    "msl":       "none",    # pressure — roughly Gaussian
    "uas":       "none",    # alias
    "vas":       "none",    # alias

    # ERA5 precipitation (daily sum)
    "tp":        "log1p",   # zero-inflated, heavy tail

    # GloFAS freshwater forcing
    "dis24":     "log1p",   # river discharge — heavy-tailed
    "rowe":      "log1p",   # runoff water equivalent — zero-inflated, heavy tail

    # Static context (handled separately — not z-scored)
    "bathymetry": "minmax",
    "dist_coast": "minmax",
    "discharge":  "log1p",  # generic alias
    "precip":     "log1p",  # generic alias
}


# ---------------------------------------------------------------------------
# Statistics computation (training set only)
# ---------------------------------------------------------------------------

def compute_stats(
    ds: xr.Dataset,
    variables: List[str],
    obs_mask: Optional[xr.DataArray] = None,
    land_mask: Optional[xr.DataArray] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-variable normalization statistics from a dataset.
    Only valid, ocean pixels are included in statistics.

    Parameters
    ----------
    ds        : dataset containing variables to normalize
    variables : list of variable names
    obs_mask  : (time, lat, lon) binary mask; only 1-pixels contribute to stats
    land_mask : (lat, lon) binary mask; only ocean pixels contribute

    Returns
    -------
    dict of {variable: {mean, std, min, max, p01, p99}}
    """
    stats = {}

    for var in variables:
        if var not in ds:
            logger.warning(f"Variable '{var}' not in dataset; skipping stats.")
            continue

        values = ds[var].values.astype(np.float64)

        pretransform = PRETRANSFORMS.get(var, "none")
        if pretransform == "log1p":
            values = np.log1p(np.clip(values, 0, None))
        elif pretransform == "minmax":
            finite_vals = values[np.isfinite(values)]
            stats[var] = {
                "min":  float(np.nanmin(finite_vals)),
                "max":  float(np.nanmax(finite_vals)),
                "mean": float(np.nanmean(finite_vals)),
                "std":  float(np.nanstd(finite_vals)),
                "p01":  float(np.nanpercentile(finite_vals, 1)),
                "p99":  float(np.nanpercentile(finite_vals, 99)),
            }
            continue

        valid = np.isfinite(values)
        if obs_mask is not None and values.ndim == obs_mask.values.ndim:
            valid &= (obs_mask.values == 1)
        if land_mask is not None:
            lm = land_mask.values
            if values.ndim == 3:
                lm = lm[np.newaxis, :, :]
            valid &= (lm == 1)

        flat = values[valid]
        flat = flat[np.isfinite(flat)]

        if len(flat) == 0:
            logger.warning(f"No valid values found for '{var}'; stats set to 0/1.")
            stats[var] = {"mean": 0.0, "std": 1.0, "min": 0.0,
                          "max": 1.0, "p01": 0.0, "p99": 1.0}
            continue

        stats[var] = {
            "mean": float(np.mean(flat)),
            "std":  max(float(np.std(flat)), 1e-8),
            "min":  float(np.min(flat)),
            "max":  float(np.max(flat)),
            "p01":  float(np.percentile(flat, 1)),
            "p99":  float(np.percentile(flat, 99)),
        }
        logger.debug(
            f"{var:>12s} | mean={stats[var]['mean']:+.3f} "
            f"std={stats[var]['std']:.3f} "
            f"[{stats[var]['p01']:.3f}, {stats[var]['p99']:.3f}]"
        )

    return stats


def save_stats(stats: Dict, path: Union[str, Path]) -> None:
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Normalization stats saved to {path}")


def load_stats(path: Union[str, Path]) -> Dict:
    with open(path) as f:
        stats = json.load(f)
    logger.info(f"Normalization stats loaded from {path}")
    return stats


# ---------------------------------------------------------------------------
# Normalization transforms
# ---------------------------------------------------------------------------

def normalize_variable(
    da: xr.DataArray,
    var_name: str,
    stats: Dict[str, Dict[str, float]],
    clip_outliers: bool = True,
) -> xr.DataArray:
    """
    Apply pre-transform + normalization to a single DataArray.

    Pre-transforms:
        log1p  : log(x + 1) for skewed positive variables
        minmax : (x - min) / (max - min) for static context
        none   : raw value → z-scoring only

    Then z-score: (x - mean) / std
    NaN pixels remain NaN after normalization.
    """
    if var_name not in stats:
        logger.warning(f"No stats found for '{var_name}'; returning unnormalized.")
        return da

    s = stats[var_name]
    values = da.values.astype(np.float32).copy()
    finite = np.isfinite(values)

    if clip_outliers and "p01" in s and "p99" in s:
        values[finite] = np.clip(values[finite], s["p01"], s["p99"])

    pretransform = PRETRANSFORMS.get(var_name, "none")

    if pretransform == "log1p":
        values[finite] = np.log1p(np.clip(values[finite], 0, None))
    elif pretransform == "minmax":
        denom = max(s["max"] - s["min"], 1e-8)
        values[finite] = (values[finite] - s["min"]) / denom
        return da.copy(data=values)

    values[finite] = (values[finite] - s["mean"]) / s["std"]
    return da.copy(data=values)


def denormalize_variable(
    da: xr.DataArray,
    var_name: str,
    stats: Dict[str, Dict[str, float]],
) -> xr.DataArray:
    """
    Inverse transform: normalized → physical units.
    Used for model outputs to recover Chl-a in mg/m³.
    """
    if var_name not in stats:
        logger.warning(f"No stats for '{var_name}'; returning as-is.")
        return da

    s = stats[var_name]
    values = da.values.astype(np.float32).copy()
    finite = np.isfinite(values)
    pretransform = PRETRANSFORMS.get(var_name, "none")

    if pretransform == "minmax":
        denom = max(s["max"] - s["min"], 1e-8)
        values[finite] = values[finite] * denom + s["min"]
        return da.copy(data=values)

    values[finite] = values[finite] * s["std"] + s["mean"]

    if pretransform == "log1p":
        values[finite] = np.expm1(values[finite])
        values[finite] = np.clip(values[finite], 0, None)

    return da.copy(data=values)


# ---------------------------------------------------------------------------
# Dataset-level normalization
# ---------------------------------------------------------------------------

def normalize_dataset(
    ds: xr.Dataset,
    stats: Dict[str, Dict[str, float]],
    variables: Optional[List[str]] = None,
    clip_outliers: bool = True,
) -> xr.Dataset:
    vars_to_norm = variables or list(ds.data_vars)
    normalized_vars = {}

    for var in vars_to_norm:
        if var not in ds:
            continue
        normalized_vars[var] = normalize_variable(ds[var], var, stats, clip_outliers)
        logger.debug(f"Normalized: {var}")

    unchanged = {v: ds[v] for v in ds.data_vars if v not in vars_to_norm}
    normalized_vars.update(unchanged)
    return xr.Dataset(normalized_vars, coords=ds.coords, attrs=ds.attrs)


def denormalize_dataset(
    ds: xr.Dataset,
    stats: Dict[str, Dict[str, float]],
    variables: Optional[List[str]] = None,
) -> xr.Dataset:
    vars_to_denorm = variables or list(ds.data_vars)
    result = {}

    for var in vars_to_denorm:
        if var not in ds:
            continue
        result[var] = denormalize_variable(ds[var], var, stats)

    unchanged = {v: ds[v] for v in ds.data_vars if v not in vars_to_denorm}
    result.update(unchanged)
    return xr.Dataset(result, coords=ds.coords, attrs=ds.attrs)


# ---------------------------------------------------------------------------
# Wind: u/v → speed + direction (optional preprocessing step)
# ---------------------------------------------------------------------------

def compute_wind_speed_direction(
    ds: xr.Dataset,
    u_var: str = "u10",
    v_var: str = "v10",
    drop_components: bool = False,
) -> xr.Dataset:
    """
    Compute wind speed and meteorological direction from u/v components.

    Wind speed     = sqrt(u² + v²)
    Wind direction = atan2(-u, -v) in degrees (met convention: direction FROM)
    """
    if u_var not in ds or v_var not in ds:
        logger.warning(f"Wind components {u_var}/{v_var} not found; skipping.")
        return ds

    u = ds[u_var].values
    v = ds[v_var].values

    speed     = np.sqrt(u**2 + v**2).astype(np.float32)
    direction = (np.degrees(np.arctan2(-u, -v)) % 360).astype(np.float32)

    ds = ds.assign({
        "wind_speed": xr.DataArray(speed, coords=ds[u_var].coords, dims=ds[u_var].dims,
                                   attrs={"units": "m/s"}),
        "wind_dir":   xr.DataArray(direction, coords=ds[u_var].coords, dims=ds[u_var].dims,
                                   attrs={"units": "degrees"}),
    })

    if drop_components:
        ds = ds.drop_vars([u_var, v_var])

    return ds