"""
loader.py
---------
Handles loading data from all formats produced by Copernicus Marine Service:
    - NetCDF  (.nc)
    - HDF5    (.h5 / .he5)
    - GeoTIFF (.tif / .tiff)
    - Zarr    (directory store)

All loaders return a standardized xarray.Dataset so every downstream
module works the same way regardless of source format.
"""

import os
import glob
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(path: Union[str, Path]) -> str:
    """
    Infer file format from extension or directory structure.
    Returns one of: 'netcdf' | 'hdf5' | 'geotiff' | 'zarr' | 'unknown'
    """
    path = Path(path)
    if path.is_dir():
        if (path / ".zgroup").exists() or (path / ".zarray").exists():
            return "zarr"
        return "unknown"
    suffix = path.suffix.lower()
    if suffix in [".nc", ".nc4"]:
        return "netcdf"
    if suffix in [".h5", ".he5", ".hdf5", ".hdf"]:
        return "hdf5"
    if suffix in [".tif", ".tiff"]:
        return "geotiff"
    return "unknown"


# ---------------------------------------------------------------------------
# Individual format loaders
# ---------------------------------------------------------------------------

def load_netcdf(
    path: Union[str, Path],
    variables: Optional[List[str]] = None,
    chunks: Optional[dict] = None,
) -> xr.Dataset:
    logger.info(f"Loading NetCDF: {path}")
    kwargs = {"engine": "netcdf4"}
    if chunks:
        kwargs["chunks"] = chunks
    ds = xr.open_dataset(path, **kwargs)
    if variables:
        available = [v for v in variables if v in ds]
        missing   = [v for v in variables if v not in ds]
        if missing:
            logger.warning(f"Variables not found in {path}: {missing}")
        ds = ds[available]
    return ds


def load_hdf5(
    path: Union[str, Path],
    variables: Optional[List[str]] = None,
    group: Optional[str] = None,
) -> xr.Dataset:
    logger.info(f"Loading HDF5: {path}")
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for HDF5 loading. pip install h5py")

    data_vars = {}
    coords    = {}
    with h5py.File(path, "r") as f:
        root = f[group] if group else f
        for coord_name in ["lat", "latitude", "Latitude", "lon", "longitude",
                           "Longitude", "time", "Time"]:
            if coord_name in root:
                arr = root[coord_name][:]
                canonical = (
                    "lat" if "lat" in coord_name.lower()
                    else "lon" if "lon" in coord_name.lower()
                    else "time"
                )
                coords[canonical] = arr
        target_vars = variables or list(root.keys())
        for var in target_vars:
            if var in root and var not in coords:
                arr = root[var][:]
                dims = _infer_dims(arr.shape, coords)
                attrs = dict(root[var].attrs) if hasattr(root[var], "attrs") else {}
                data_vars[var] = xr.Variable(dims, arr, attrs=attrs)
    return xr.Dataset(data_vars, coords=coords)


def load_geotiff(
    path: Union[str, Path],
    variables: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Load a GeoTIFF into an xarray Dataset using rioxarray.
    Requires: pip install rioxarray
    """
    logger.info(f"Loading GeoTIFF: {path}")
    try:
        import rioxarray as rxr
    except ImportError:
        raise ImportError("rioxarray is required for GeoTIFF loading. pip install rioxarray")

    da = rxr.open_rasterio(path)
    da = da.rename({"x": "lon", "y": "lat"})
    n_bands = da.sizes["band"]
    names   = variables if variables and len(variables) == n_bands \
              else [f"band_{i+1}" for i in range(n_bands)]
    data_vars = {names[i]: da.isel(band=i).drop_vars("band") for i in range(n_bands)}
    return xr.Dataset(data_vars)


def load_zarr(
    path: Union[str, Path],
    variables: Optional[List[str]] = None,
) -> xr.Dataset:
    logger.info(f"Loading Zarr: {path}")
    ds = xr.open_zarr(str(path), consolidated=True)
    if variables:
        available = [v for v in variables if v in ds]
        ds = ds[available]
    return ds


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def load(
    path: Union[str, Path],
    variables: Optional[List[str]] = None,
    chunks: Optional[dict] = None,
    hdf5_group: Optional[str] = None,
) -> xr.Dataset:
    fmt = detect_format(path)
    logger.info(f"Detected format: {fmt} for {path}")
    if fmt == "netcdf":
        return load_netcdf(path, variables, chunks)
    elif fmt == "hdf5":
        return load_hdf5(path, variables, hdf5_group)
    elif fmt == "geotiff":
        return load_geotiff(path, variables)
    elif fmt == "zarr":
        return load_zarr(path, variables)
    else:
        raise ValueError(
            f"Unsupported or unrecognized format for: {path}\n"
            f"Supported: .nc, .nc4, .h5, .hdf5, .tif, .tiff, zarr directory"
        )


# ---------------------------------------------------------------------------
# Multi-file loader
# ---------------------------------------------------------------------------

def load_time_series(
    directory: Union[str, Path],
    pattern: str = "*.nc",
    variables: Optional[List[str]] = None,
    chunks: Optional[dict] = None,
    date_parser: Optional[callable] = None,
) -> xr.Dataset:
    files = sorted(glob.glob(str(Path(directory) / pattern)))
    if not files:
        raise FileNotFoundError(f"No files matched pattern '{pattern}' in {directory}")
    logger.info(f"Found {len(files)} files in {directory}")

    datasets = []
    for fpath in files:
        try:
            ds = load(fpath, variables=variables, chunks=chunks)
            if "time" not in ds.coords and date_parser:
                t = date_parser(Path(fpath).name)
                ds = ds.expand_dims("time").assign_coords(time=("time", [t]))
            datasets.append(ds)
        except Exception as e:
            logger.warning(f"Skipping {fpath}: {e}")

    if not datasets:
        raise RuntimeError("All files failed to load. Check your data directory.")

    combined = xr.concat(datasets, dim="time").sortby("time")
    logger.info(f"Combined time range: {combined.time.values[0]} to {combined.time.values[-1]}")
    return combined


# ---------------------------------------------------------------------------
# CMEMS downloader — time-varying products
# ---------------------------------------------------------------------------

def download_copernicus(
    dataset_id: str,
    variables: List[str],
    date_start: str,
    date_end: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    depth_min: float = 0.49,
    depth_max: float = 0.49,
    output_dir: str = "data/raw",
    username: Optional[str] = None,
    password: Optional[str] = None,
    skip_depth: bool = False,
) -> Path:
    """
    Download a subset from Copernicus Marine Service.

    Parameters
    ----------
    depth_min / depth_max : use BGC_DEPTH_MIN/MAX (0.51) for BGC products and
                            PHY_DEPTH_MIN/MAX (0.49) for physics products.
                            These differ — do not share a single constant.
    skip_depth            : set False for all products; pass the correct
                            depth_min/max per product instead.
    """
    try:
        import copernicusmarine as cm
    except ImportError:
        raise ImportError(
            "copernicusmarine is required.\n"
            "Install: pip install copernicusmarine\n"
            "Auth:    copernicusmarine login"
        )

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{dataset_id}_{date_start}_{date_end}.nc"
    output_path = Path(output_dir) / output_filename

    if output_path.exists():
        logger.info(f"File already exists, skipping download: {output_path}")
        return output_path

    logger.info(f"Downloading {dataset_id} | {date_start} to {date_end}")

    subset_kwargs = dict(
        dataset_id        = dataset_id,
        variables         = variables,
        start_datetime    = date_start,
        end_datetime      = date_end,
        minimum_longitude = lon_min,
        maximum_longitude = lon_max,
        minimum_latitude  = lat_min,
        maximum_latitude  = lat_max,
        output_filename   = str(output_path),
        username          = username,
        password          = password,
    )

    if not skip_depth:
        subset_kwargs["minimum_depth"] = depth_min
        subset_kwargs["maximum_depth"] = depth_max

    cm.subset(**subset_kwargs)
    logger.info(f"Downloaded to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CMEMS downloader — static products (no time, no depth)
# Used for: cmems_mod_glo_phy_my_0.083deg_static (bathymetry)
# ---------------------------------------------------------------------------

def download_copernicus_static(
    dataset_id: str,
    variables: List[str],
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    output_dir: str = "data/raw",
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Path:
    """
    Download a static (no time, no depth dimension) CMEMS sub-dataset.
    Used for the physics bathymetry product.

    The downloaded file contains a single 2D field (lat × lon).
    """
    try:
        import copernicusmarine as cm
    except ImportError:
        raise ImportError("pip install copernicusmarine  &&  copernicusmarine login")

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{dataset_id}_static.nc"
    output_path = Path(output_dir) / output_filename

    if output_path.exists():
        logger.info(f"Static file already exists, skipping: {output_path}")
        return output_path

    logger.info(f"Downloading static dataset: {dataset_id}")

    cm.subset(
        dataset_id        = dataset_id,
        variables         = variables,
        minimum_longitude = lon_min,
        maximum_longitude = lon_max,
        minimum_latitude  = lat_min,
        maximum_latitude  = lat_max,
        output_filename   = str(output_path),
        username          = username,
        password          = password,
    )

    logger.info(f"Static dataset downloaded to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# ERA5 daily statistics downloaders — wind/pressure and precipitation
#
# Dataset: derived-era5-single-levels-daily-statistics
# Replaces: reanalysis-era5-single-levels (hourly) + accumulate_era5_precip_to_daily
#
# Wind+pressure (daily_mean) and precipitation (daily_sum) cannot be combined
# in a single request — different daily_statistic values are required.
# Each function downloads one calendar month per CDS request to stay within
# size limits, then merges the monthly files into a single output NetCDF.
# ---------------------------------------------------------------------------

def download_era5_wind(
    date_start: str,
    date_end: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    output_dir: str = "data/raw",
) -> Path:
    """
    Download ERA5 daily-mean 10m wind components (u10, v10).

    Split from msl deliberately — keeping requests to <=2 variables puts them
    in the CDS "small request" category (<=2 vars, 1 month), which has a
    lighter queue and avoids the "cost limits exceeded" 403 error that occurs
    when 3 variables are requested over a multi-month period.

    Variables: u10, v10
    Aggregation: daily_mean
    Requires: pip install cdsapi  |  credentials in ~/.cdsapirc
    """
    return _download_era5_daily(
        date_start        = date_start,
        date_end          = date_end,
        lon_min           = lon_min,
        lon_max           = lon_max,
        lat_min           = lat_min,
        lat_max           = lat_max,
        output_dir        = output_dir,
        request_variables = [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        daily_statistic   = "daily_mean",
        label             = "wind",
    )


def download_era5_msl(
    date_start: str,
    date_end: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    output_dir: str = "data/raw",
) -> Path:
    """
    Download ERA5 daily-mean mean sea level pressure (msl).

    Kept as a separate 1-variable request so that both wind and msl
    stay in the CDS small-request category.

    Variable  : msl
    Aggregation: daily_mean
    Requires: pip install cdsapi  |  credentials in ~/.cdsapirc
    """
    return _download_era5_daily(
        date_start        = date_start,
        date_end          = date_end,
        lon_min           = lon_min,
        lon_max           = lon_max,
        lat_min           = lat_min,
        lat_max           = lat_max,
        output_dir        = output_dir,
        request_variables = ["mean_sea_level_pressure"],
        daily_statistic   = "daily_mean",
        label             = "msl",
    )


def download_era5_precip(
    date_start: str,
    date_end: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    output_dir: str = "data/raw",
) -> Path:
    """
    Download ERA5 daily-sum total precipitation (tp) from the
    derived-era5-single-levels-daily-statistics dataset.

    Variable: tp
    Aggregation: daily_sum  (accumulation done server-side — no post-processing needed)
    Requires: pip install cdsapi  |  credentials in ~/.cdsapirc
    """
    return _download_era5_daily(
        date_start        = date_start,
        date_end          = date_end,
        lon_min           = lon_min,
        lon_max           = lon_max,
        lat_min           = lat_min,
        lat_max           = lat_max,
        output_dir        = output_dir,
        request_variables = ["total_precipitation"],
        daily_statistic   = "daily_sum",
        label             = "precip",
    )


def download_era5_wind_hourly(
    date_start: str,
    date_end: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    output_dir: str = "data/raw",
) -> Path:
    """
    Download ERA5 wind (u10, v10), mean sea level pressure (msl) and
    precipitation (tp) together from the raw hourly reanalysis-era5-single-levels
    dataset.

    Used when ERA5_DOWNLOAD_METHOD = "hourly" in config.py.

    Key differences from "daily_stats" method:
      - All 4 variables fetched in ONE request per month (simpler, fewer queue slots)
      - Data format: netcdf4 (not netcdf — specific to this dataset)
      - 24x larger downloads (raw hourly values, not pre-aggregated)
      - Precipitation accumulated to daily totals via accumulate_era5_precip_to_daily()
      - Wind and msl are daily-mean computed from 24 hourly steps after download
      - msl IS available in this dataset (unlike previous implementation)

    Returns the path to the merged NetCDF (contains u10, v10, msl, tp together).
    """
    try:
        import cdsapi
        import pandas as pd
        import zipfile
    except ImportError:
        raise ImportError("pip install cdsapi pandas")

    os.makedirs(output_dir, exist_ok=True)
    merged_path = Path(output_dir) / f"era5_hourly_{date_start}_{date_end}.nc"

    if merged_path.exists():
        logger.info(f"ERA5 hourly file already exists, skipping: {merged_path}")
        return merged_path

    full_range  = pd.date_range(date_start, date_end, freq="MS")
    year_months = [(ts.year, ts.month) for ts in full_range]
    end_ts = pd.Timestamp(date_end)
    if (end_ts.year, end_ts.month) not in year_months:
        year_months.append((end_ts.year, end_ts.month))

    client = cdsapi.Client()
    chunk_paths = []

    for year, month in year_months:
        chunk_path = Path(output_dir) / f"era5_hourly_{year}_{month:02d}.nc"

        if chunk_path.exists():
            try:
                ds_test = xr.open_dataset(str(chunk_path), engine="netcdf4")
                ds_test.close()
                logger.info(f"ERA5 hourly {year}-{month:02d} already exists, skipping.")
                chunk_paths.append(chunk_path)
                continue
            except Exception:
                logger.warning(f"Corrupt ERA5 hourly {year}-{month:02d}; re-downloading.")
                chunk_path.unlink()

        month_start = max(pd.Timestamp(f"{year}-{month:02d}-01"), pd.Timestamp(date_start))
        month_end   = min(
            pd.Timestamp(f"{year}-{month:02d}-01") + pd.offsets.MonthEnd(1),
            pd.Timestamp(date_end),
        )
        days = sorted(set(f"{d.day:02d}" for d in pd.date_range(month_start, month_end, freq="D")))

        logger.info(f"Downloading ERA5 hourly | {year}-{month:02d} (u10, v10, msl, tp)")

        tmp_path = chunk_path.with_suffix(".tmp")
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type":    ["reanalysis"],
                "variable":        [
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "mean_sea_level_pressure",   # available in hourly — no placeholder needed
                    "total_precipitation",
                ],
                "year":            [str(year)],
                "month":           [f"{month:02d}"],
                "day":             days,
                "time":            [f"{h:02d}:00" for h in range(24)],
                "area":            [lat_max, lon_min, lat_min, lon_max],
                "data_format":     "netcdf4",    # hourly dataset uses "netcdf4" not "netcdf"
                "download_format": "unarchived",
                "grid":            ["0.25", "0.25"],
            },
            str(tmp_path),
        )

        if zipfile.is_zipfile(str(tmp_path)):
            logger.info(f"ERA5 hourly {year}-{month:02d}: CDS returned zip; extracting.")
            try:
                _extract_nc_from_zip(tmp_path, chunk_path)
            finally:
                tmp_path.unlink(missing_ok=True)
        else:
            tmp_path.rename(chunk_path)

        try:
            ds_test = xr.open_dataset(str(chunk_path), engine="netcdf4")
            ds_test.close()
        except Exception as e:
            chunk_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"ERA5 hourly {year}-{month:02d} failed validation: {e}"
            )

        logger.info(f"ERA5 hourly {year}-{month:02d} saved: {chunk_path}")
        chunk_paths.append(chunk_path)

    logger.info(f"Merging {len(chunk_paths)} ERA5 hourly files -> {merged_path}")
    _merge_era5_chunks(chunk_paths, merged_path)

    for p in chunk_paths:
        try:
            p.unlink()
        except OSError:
            pass

    logger.info(f"ERA5 hourly merged file saved: {merged_path}")
    return merged_path


def accumulate_era5_precip_to_daily(ds: xr.Dataset, var: str = "tp") -> xr.Dataset:
    """
    Sum ERA5 hourly precipitation accumulations to daily totals and convert
    from metres to mm/day (* 1000).

    Used ONLY when ERA5_DOWNLOAD_METHOD = "hourly" in config.py.
    The daily-statistics method (default) already provides daily sums server-side.

    ERA5 tp = metres of water accumulated per 1-hour step.
    Daily total (mm) = sum of 24 hourly values * 1000.
    """
    if var not in ds:
        logger.warning(f"Variable '{var}' not found in dataset; skipping accumulation.")
        return ds

    daily = ds[var].resample(time="1D").sum(skipna=True) * 1000.0
    daily.attrs.update({
        "units":     "mm/day",
        "long_name": "Total precipitation (daily sum)",
    })
    logger.info("ERA5 precipitation accumulated to daily mm/day")
    return ds.assign({var: daily})


def _download_era5_daily(
    date_start: str,
    date_end: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    output_dir: str,
    request_variables: List[str],
    daily_statistic: str,
    label: str,
) -> Path:
    """
    Internal helper: download YEARLY chunks from derived-era5-single-levels-daily-statistics
    and merge into a single NetCDF file.

    Chunking strategy: ONE REQUEST PER YEAR (not per month).
    -------------------------------------------------------
    The CDS queue for this dataset has two tiers:
      - Large (> 2 variables OR > 1 month): limited to 40 concurrent, heavily queued
      - Small (<= 2 variables AND 1 month): limited to 60 concurrent, lighter queue

    Monthly chunking produced 60 wind + 60 precip = 120 requests, each sitting
    behind 1800+ queued jobs. Yearly chunking reduces this to 5 + 5 = 10 total
    requests. Each is still "large" but 10 queue slots vs 120 is a 12x improvement.

    Already-downloaded yearly files are skipped automatically (resumable).
    """
    try:
        import cdsapi
        import pandas as pd
        import zipfile
    except ImportError:
        raise ImportError("pip install cdsapi pandas")

    os.makedirs(output_dir, exist_ok=True)
    merged_path = Path(output_dir) / f"era5_{label}_{date_start}_{date_end}.nc"

    if merged_path.exists():
        logger.info(f"ERA5 {label} file already exists, skipping: {merged_path}")
        return merged_path

    # Monthly chunking: 1 month per request, <=2 variables per request.
    # CDS classifies requests as "small" when: <=2 variables AND 1 month.
    # Small requests use a lighter queue (60 concurrent vs 40 for large).
    # With u10/v10 split from msl, each request is <=2 variables.
    # Total requests: 60 wind + 60 msl + 60 precip = 180, all small-category.
    full_range  = pd.date_range(date_start, date_end, freq="MS")
    year_months = [(ts.year, ts.month) for ts in full_range]
    end_ts = pd.Timestamp(date_end)
    if (end_ts.year, end_ts.month) not in year_months:
        year_months.append((end_ts.year, end_ts.month))

    client = cdsapi.Client()
    chunk_paths = []

    for year, month in year_months:
        chunk_path = Path(output_dir) / f"era5_{label}_{year}_{month:02d}.nc"

        if chunk_path.exists():
            try:
                ds_test = xr.open_dataset(str(chunk_path), engine="netcdf4")
                ds_test.close()
                logger.info(f"ERA5 {label} {year}-{month:02d} already exists, skipping.")
                chunk_paths.append(chunk_path)
                continue
            except Exception:
                logger.warning(f"Corrupt ERA5 {label} {year}-{month:02d}; re-downloading.")
                chunk_path.unlink()

        month_start = max(pd.Timestamp(f"{year}-{month:02d}-01"), pd.Timestamp(date_start))
        month_end   = min(
            pd.Timestamp(f"{year}-{month:02d}-01") + pd.offsets.MonthEnd(1),
            pd.Timestamp(date_end),
        )
        days = sorted(set(f"{d.day:02d}" for d in pd.date_range(month_start, month_end, freq="D")))

        logger.info(f"Downloading ERA5 {label} | {year}-{month:02d} ({daily_statistic})")

        tmp_path = chunk_path.with_suffix(".tmp")
        client.retrieve(
            "derived-era5-single-levels-daily-statistics",
            {
                "product_type":    "reanalysis",
                "variable":        request_variables,
                "year":            [str(year)],
                "month":           [f"{month:02d}"],
                "day":             days,
                "daily_statistic": daily_statistic,
                "time_zone":       "UTC+00:00",
                "frequency":       "1_hourly",
                "area":            [lat_max, lon_min, lat_min, lon_max],
                "data_format":     "netcdf",
                "download_format": "unarchived",
                "grid":            ["0.25", "0.25"],
            },
            str(tmp_path),
        )

        if zipfile.is_zipfile(str(tmp_path)):
            logger.info(f"ERA5 {label} {year}-{month:02d}: CDS returned zip; extracting.")
            try:
                _extract_nc_from_zip(tmp_path, chunk_path)
            finally:
                tmp_path.unlink(missing_ok=True)
        else:
            tmp_path.rename(chunk_path)

        try:
            ds_test = xr.open_dataset(str(chunk_path), engine="netcdf4")
            ds_test.close()
        except Exception as e:
            chunk_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded ERA5 {label} {year}-{month:02d} failed validation: {e}"
            )

        logger.info(f"ERA5 {label} {year}-{month:02d} saved: {chunk_path}")
        chunk_paths.append(chunk_path)

    # Merge monthly files into one output NetCDF
    logger.info(f"Merging {len(chunk_paths)} ERA5 {label} monthly files -> {merged_path}")
    _merge_era5_chunks(chunk_paths, merged_path)

    for p in chunk_paths:
        try:
            p.unlink()
        except OSError:
            pass

    logger.info(f"ERA5 {label} merged file saved: {merged_path}")
    return merged_path


def _merge_era5_chunks(chunk_paths: list, merged_path: Path) -> None:
    """
    Merge monthly ERA5 NetCDF chunks into a single file using netCDF4.
    Renames valid_time/latitude/longitude to time/lat/lon for pipeline consistency.
    """
    import netCDF4 as nc4

    SKIP      = {"expver", "number"}
    DIM_MAP   = {"valid_time": "time", "latitude": "lat", "longitude": "lon"}
    VAR_MAP   = {"valid_time": "time", "latitude": "lat", "longitude": "lon"}

    def _out_dim(n): return DIM_MAP.get(n, n)
    def _out_var(n): return VAR_MAP.get(n, n)

    with nc4.Dataset(str(merged_path), "w", format="NETCDF4") as out:
        first = True
        time_index = 0

        for p in chunk_paths:
            with nc4.Dataset(str(p), "r") as src:
                time_dim = "valid_time" if "valid_time" in src.dimensions else "time"
                n_steps  = len(src.dimensions[time_dim])

                if first:
                    for name, dim in src.dimensions.items():
                        if name in SKIP:
                            continue
                        out_name = _out_dim(name)
                        size = None if name == time_dim else len(dim)
                        out.createDimension(out_name, size)

                    for name, var in src.variables.items():
                        if name in SKIP:
                            continue
                        out_name = _out_var(name)
                        out_dims = tuple(_out_dim(d) for d in var.dimensions if d not in SKIP)
                        if not out_dims:
                            continue
                        out_var = out.createVariable(out_name, var.datatype, out_dims,
                                                     zlib=True, complevel=4)
                        out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})

                    out.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
                    first = False

                t_sl = slice(time_index, time_index + n_steps)
                for name, var in src.variables.items():
                    if name in SKIP:
                        continue
                    out_name = _out_var(name)
                    if out_name not in out.variables:
                        continue
                    out_dims = [_out_dim(d) for d in var.dimensions if d not in SKIP]
                    if "time" in out_dims:
                        out.variables[out_name][t_sl] = var[:]
                    elif time_index == 0:
                        out.variables[out_name][:] = var[:]

                time_index += n_steps


# ---------------------------------------------------------------------------
# GloFAS river discharge downloader
# ---------------------------------------------------------------------------

def _merge_glofas_chunks(chunk_paths: list, merged_path: Path) -> None:
    """
    Merge monthly GloFAS NetCDF chunks into a single file using xarray.

    Root cause of previous failures:
      1. xr.concat OOM: loading all 60 files with .load() before concat
         allocated 2+ GB. Fixed by using lazy open_dataset (no .load()).
      2. valid_time alignment error: each monthly file has a different-length
         valid_time coordinate. Fixed by dropping it before concat.
      3. netCDF4 writer shape mismatch: valid_time IS the actual time
         dimension for data variables in GloFAS — skipping it caused data
         arrays to lose their time dim and be treated as static. Fixed by
         using xarray concat which handles dim renaming cleanly.

    Strategy:
      - Open each file lazily (no .load())
      - Drop valid_time coordinate (keep the "time" index dimension)
      - Rename latitude/longitude → lat/lon
      - Concat lazily along time with join="override"
      - Write to disk in one streaming to_netcdf call
    """
    datasets = []
    for p in chunk_paths:
        ds = xr.open_dataset(str(p), engine="netcdf4")

        # GloFAS files use "valid_time" as BOTH the dimension name and a
        # datetime coordinate. Each monthly file has a different-length
        # valid_time dimension (28/29/30/31 days), so xr.concat fails to
        # align them. Fix: rename the dimension itself to "time" first.
        rename = {}
        if "valid_time" in ds.dims:
            rename["valid_time"] = "time"
        if "latitude"  in ds.dims or "latitude"  in ds.coords:
            rename["latitude"]  = "lat"
        if "longitude" in ds.dims or "longitude" in ds.coords:
            rename["longitude"] = "lon"
        if rename:
            ds = ds.rename(rename)

        # Drop the valid_time coordinate if it still exists as a variable
        # after the rename (some versions keep it as an alias)
        if "valid_time" in ds.coords:
            ds = ds.drop_vars("valid_time")

        datasets.append(ds)

    # Concat along the renamed "time" dimension.
    # Each chunk has a unique time range so no alignment needed — use
    # join="override" and compat="override" to skip conflict checks.
    ds_merged = xr.concat(datasets, dim="time", coords="minimal", data_vars="minimal", join="override")

    # Write lazily — xarray streams without loading all data into RAM
    ds_merged.to_netcdf(str(merged_path))

    for ds in datasets:
        ds.close()
    ds_merged.close()


def download_glofas(
    date_start: str,
    date_end: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    output_dir: str = "data/raw",
    system_version: str = "version_4_0",
) -> Path:
    """
    Download GloFAS v4.0 river discharge reanalysis via cdsapi.

    Chunking: ONE REQUEST PER MONTH to stay within CEMS cost limits.
    Yearly requests hit the "cost limits exceeded" 403 error for our
    3-variable, multi-region request. Monthly chunks (60 total for 5 years)
    each stay within limits and are resumable individually.

    Variables: dis24, ro (runoff water equivalent), swvl (soil wetness index)
    Format   : NetCDF-4 (cfgrib/eccodes no longer required)

    Requires
    --------
    pip install cdsapi pandas
    CDS credentials in ~/.cdsapirc
    CEMS endpoint: https://ewds.climate.copernicus.eu/api
    """
    try:
        import cdsapi
        import pandas as pd
    except ImportError:
        raise ImportError("pip install cdsapi pandas")

    os.makedirs(output_dir, exist_ok=True)
    merged_path = Path(output_dir) / f"glofas_{date_start}_{date_end}.nc"

    if merged_path.exists():
        logger.info(f"GloFAS file already exists, skipping: {merged_path}")
        return merged_path

    full_range  = pd.date_range(date_start, date_end, freq="MS")
    year_months = [(ts.year, ts.month) for ts in full_range]
    end_ts = pd.Timestamp(date_end)
    if (end_ts.year, end_ts.month) not in year_months:
        year_months.append((end_ts.year, end_ts.month))

    # CEMS uses a different API endpoint from standard CDS
    client      = cdsapi.Client(url="https://ewds.climate.copernicus.eu/api")
    chunk_paths = []

    for year, month in year_months:
        chunk_path = Path(output_dir) / f"glofas_{year}_{month:02d}.nc"

        if chunk_path.exists():
            try:
                ds_test = xr.open_dataset(str(chunk_path), engine="netcdf4")
                ds_test.close()
                logger.info(f"GloFAS {year}-{month:02d} already exists, skipping.")
                chunk_paths.append(chunk_path)
                continue
            except Exception:
                logger.warning(f"Corrupt GloFAS {year}-{month:02d}; re-downloading.")
                chunk_path.unlink()

        month_start = max(pd.Timestamp(f"{year}-{month:02d}-01"), pd.Timestamp(date_start))
        month_end   = min(
            pd.Timestamp(f"{year}-{month:02d}-01") + pd.offsets.MonthEnd(1),
            pd.Timestamp(date_end),
        )
        month_dates = pd.date_range(month_start, month_end, freq="D")
        days        = sorted(set(f"{d.day:02d}" for d in month_dates))

        logger.info(f"Downloading GloFAS | {year}-{month:02d}")

        client.retrieve(
            "cems-glofas-historical",
            {
                "system_version":     [system_version],
                "hydrological_model": ["lisflood"],
                "product_type":       ["consolidated"],
                "variable": [
                    "river_discharge_in_the_last_24_hours",
                    "runoff_water_equivalent",
                    # soil_wetness_index_root_zone excluded: only in "intermediate"
                    # product type, not available in "consolidated"
                ],
                "hyear":              [str(year)],
                "hmonth":             [f"{month:02d}"],
                "hday":               days,
                "data_format":        "netcdf4",
                "download_format":    "unarchived",
                "area":               [lat_max, lon_min, lat_min, lon_max],
            },
        ).download(str(chunk_path))

        # Validate
        try:
            ds_test = xr.open_dataset(str(chunk_path), engine="netcdf4")
            ds_test.close()
        except Exception as e:
            chunk_path.unlink(missing_ok=True)
            raise RuntimeError(f"GloFAS {year}-{month:02d} failed validation: {e}")

        logger.info(f"GloFAS {year}-{month:02d} saved to: {chunk_path}")
        chunk_paths.append(chunk_path)

    # Merge monthly files using netCDF4 direct write — streams one month at a
    # time so peak memory is one monthly chunk, not all 60 simultaneously.
    # Also avoids the xr.concat valid_time alignment error that occurs when
    # monthly files have different-length valid_time coordinate arrays.
    logger.info(f"Merging {len(chunk_paths)} monthly GloFAS files -> {merged_path}")
    _merge_glofas_chunks(chunk_paths, merged_path)

    for p in chunk_paths:
        try:
            p.unlink()
        except OSError:
            pass

    logger.info(f"GloFAS merged file saved to: {merged_path}")
    return merged_path


def load_glofas(path: Union[str, Path]) -> xr.Dataset:
    """
    Load a GloFAS NetCDF file into an xarray Dataset.
    Renames latitude/longitude to lat/lon for pipeline consistency.
    Squeezes the "surface" dimension (size=1) that GloFAS adds to runoff
    and soil wetness variables — it has no physical meaning at the surface.

    Note: GRIB2 support removed — pipeline now downloads NetCDF-4 directly.
    """
    path = Path(path)
    logger.info(f"Loading GloFAS: {path}")
    ds = xr.open_dataset(str(path), engine="netcdf4")

    rename = {}
    if "latitude"  in ds.coords: rename["latitude"]  = "lat"
    if "longitude" in ds.coords: rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)

    # GloFAS NetCDF-4 files include a spurious "surface" dimension of size 1
    # on some variables. Squeeze it out so all variables are (time, lat, lon).
    if "surface" in ds.dims and ds.sizes["surface"] == 1:
        ds = ds.squeeze("surface", drop=True)

    return ds


# ---------------------------------------------------------------------------
# Zip extraction helper (CDS occasionally returns zip despite unarchived flag)
# ---------------------------------------------------------------------------

def _extract_nc_from_zip(zip_path: Path, dest_path: Path) -> None:
    import zipfile

    with zipfile.ZipFile(str(zip_path), "r") as zf:
        nc_members = [m for m in zf.namelist() if m.endswith(".nc")]
        if not nc_members:
            raise RuntimeError(
                f"No .nc file found in zip: {zip_path}\nContents: {zf.namelist()}"
            )
        if len(nc_members) == 1:
            with zf.open(nc_members[0]) as src, open(str(dest_path), "wb") as dst:
                dst.write(src.read())
            return

        # Multiple files: extract to temp dir, merge, save
        tmp_dir = zip_path.parent / "_zip_tmp"
        tmp_dir.mkdir(exist_ok=True)
        try:
            tmp_paths = []
            for member in nc_members:
                tmp_path = tmp_dir / Path(member).name
                with zf.open(member) as src, open(str(tmp_path), "wb") as dst:
                    dst.write(src.read())
                tmp_paths.append(tmp_path)
            datasets = []
            for p in tmp_paths:
                ds = xr.open_dataset(str(p), engine="netcdf4")
                datasets.append(ds.load())
                ds.close()
            ds_merged = xr.merge(datasets)
            ds_merged.to_netcdf(str(dest_path))
            ds_merged.close()
        finally:
            for p in tmp_paths:
                p.unlink(missing_ok=True)
            try:
                tmp_dir.rmdir()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_dims(shape: tuple, coords: dict) -> tuple:
    """
    Guess dimension names from array shape by matching coordinate lengths.
    Uses a length-bucket approach to avoid collision when lat and lon share
    the same size (common in square regional grids).
    """
    len_to_names: dict = defaultdict(list)
    for name, v in coords.items():
        len_to_names[len(v)].append(name)
    len_usage: dict = defaultdict(int)
    dims = []
    for s in shape:
        bucket = len_to_names.get(s)
        if bucket:
            idx = len_usage[s]
            dims.append(bucket[idx] if idx < len(bucket) else f"dim_{s}")
            len_usage[s] += 1
        else:
            dims.append(f"dim_{s}")
    return tuple(dims)


def print_dataset_summary(ds: xr.Dataset, label: str = "") -> None:
    """Print a compact summary of a dataset for quick inspection."""
    print(f"\n{'='*60}")
    if label:
        print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Variables : {list(ds.data_vars)}")
    print(f"  Coords    : {list(ds.coords)}")
    if "time" in ds.coords:
        print(f"  Time range: {ds.time.values[0]}  →  {ds.time.values[-1]}")
    if "lat" in ds.coords:
        print(f"  Lat range : {float(ds.lat.min()):.2f}  →  {float(ds.lat.max()):.2f}")
    if "lon" in ds.coords:
        print(f"  Lon range : {float(ds.lon.min()):.2f}  →  {float(ds.lon.max()):.2f}")
    print(f"  Size      : {ds.nbytes / 1e6:.1f} MB (in memory)")
    print()