"""
patcher.py
----------
Extracts spatiotemporal patches from aligned, normalized datasets
to create training samples for MM-MARAS.

Each sample is a dict of arrays:
    {
        "chl_obs"    : (T, H, W)            normalized log-Chl-a (NaN where missing)
        "obs_mask"   : (T, H, W)            binary validity mask
        "mcar_mask"  : (T, H, W)            MCAR classification
        "mnar_mask"  : (T, H, W)            MNAR classification
        "physics"    : (T, C_phy=6, H, W)   thetao, uo, vo, mlotst, zos, so
        "wind"       : (T, C_wind=4, H, W)  u10, v10, msl, tp
        "discharge"  : (T, C_dis=2, H, W)   dis24, rowe
        "bgc_aux"    : (T, C_bgc=5, H, W)   o2, no3, po4, si, nppv
        "static"     : (C_st=2, H, W)       bathymetry, distance-to-coast
        "bloom_mask" : (T, H, W)            bloom event labels
        "target_chl" : (H_fcast=5, H, W)    future Chl-a (forecast target)
        "center_lat" : float
        "center_lon" : float
        "time_start" : str
    }
"""

import logging
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core patch extractor
# ---------------------------------------------------------------------------

class PatchExtractor:
    """
    Slides a (T, H, W) window over aligned datasets to yield training samples.

    Parameters
    ----------
    patch_size       : spatial size H=W in pixels
    stride           : spatial stride (< patch_size for overlap)
    time_window      : number of input time steps T
    forecast_horizon : number of future steps to include as targets
    min_valid_frac   : minimum fraction of valid obs pixels per patch
    land_mask        : (H_full, W_full) array, 1=ocean; mostly-land patches skipped
    min_ocean_frac   : minimum fraction of patch pixels that must be ocean
    """

    def __init__(
        self,
        patch_size: int = 64,
        stride: int = 32,
        time_window: int = 10,
        forecast_horizon: int = 5,
        min_valid_frac: float = 0.10,
        land_mask: Optional[np.ndarray] = None,
        min_ocean_frac: float = 0.30,
    ):
        self.patch_size       = patch_size
        self.stride           = stride
        self.time_window      = time_window
        self.forecast_horizon = forecast_horizon
        self.min_valid_frac   = min_valid_frac
        self.land_mask        = land_mask
        self.min_ocean_frac   = min_ocean_frac
        self.total_time       = time_window + forecast_horizon

    def extract(
        self,
        chl_norm:    np.ndarray,    # (T_full, H_full, W_full)
        obs_mask:    np.ndarray,    # (T_full, H_full, W_full)
        mcar_mask:   np.ndarray,    # (T_full, H_full, W_full)
        mnar_mask:   np.ndarray,    # (T_full, H_full, W_full)
        physics:     np.ndarray,    # (T_full, 6, H_full, W_full)
        wind:        np.ndarray,    # (T_full, 4, H_full, W_full)
        discharge:   np.ndarray,    # (T_full, 3, H_full, W_full)
        bgc_aux:     np.ndarray,    # (T_full, 5, H_full, W_full)
        static:      np.ndarray,    # (2, H_full, W_full)
        bloom_mask:  np.ndarray,    # (T_full, H_full, W_full)
        lats:        np.ndarray,    # (H_full,)
        lons:        np.ndarray,    # (W_full,)
        times:       np.ndarray,    # (T_full,) datetime64
    ) -> Iterator[Dict[str, np.ndarray]]:
        """
        Generator that yields one patch dict per valid spatiotemporal window.
        """
        T_full, H_full, W_full = chl_norm.shape

        if T_full < self.total_time:
            raise ValueError(
                f"Dataset has only {T_full} timesteps but "
                f"time_window + forecast_horizon = {self.total_time} required."
            )

        n_patches = 0
        n_skipped = 0

        for t_start in range(0, T_full - self.total_time + 1):
            t_in_end  = t_start + self.time_window
            t_out_end = t_start + self.total_time

            for row in range(0, H_full - self.patch_size + 1, self.stride):
                for col in range(0, W_full - self.patch_size + 1, self.stride):

                    r_sl = slice(row, row + self.patch_size)
                    c_sl = slice(col, col + self.patch_size)

                    # Ocean coverage check
                    patch_land = (
                        self.land_mask[r_sl, c_sl]
                        if self.land_mask is not None
                        else np.ones((self.patch_size, self.patch_size))
                    )
                    if patch_land.mean() < self.min_ocean_frac:
                        n_skipped += 1
                        continue

                    # Valid data coverage check
                    patch_obs   = obs_mask[t_start:t_in_end, r_sl, c_sl]
                    ocean_pixels = patch_land.sum()
                    valid_frac = (
                        patch_obs * patch_land[np.newaxis]
                    ).sum() / (ocean_pixels * self.time_window + 1e-8)

                    if valid_frac < self.min_valid_frac:
                        n_skipped += 1
                        continue

                    sample = {
                        "chl_obs": chl_norm[t_start:t_in_end, r_sl, c_sl]
                                   .astype(np.float32),

                        "obs_mask":  obs_mask[t_start:t_in_end,  r_sl, c_sl]
                                     .astype(np.float32),
                        "mcar_mask": mcar_mask[t_start:t_in_end, r_sl, c_sl]
                                     .astype(np.float32),
                        "mnar_mask": mnar_mask[t_start:t_in_end, r_sl, c_sl]
                                     .astype(np.float32),

                        "physics":   physics[t_start:t_in_end, :, r_sl, c_sl]
                                     .astype(np.float32),
                        "wind":      wind[t_start:t_in_end, :, r_sl, c_sl]
                                     .astype(np.float32),
                        "discharge": discharge[t_start:t_in_end, :, r_sl, c_sl]
                                     .astype(np.float32),
                        "bgc_aux":   bgc_aux[t_start:t_in_end, :, r_sl, c_sl]
                                     .astype(np.float32),
                        "static":    static[:, r_sl, c_sl]
                                     .astype(np.float32),

                        "bloom_mask": bloom_mask[t_start:t_in_end, r_sl, c_sl]
                                      .astype(np.float32),

                        "target_chl": chl_norm[t_in_end:t_out_end, r_sl, c_sl]
                                      .astype(np.float32),

                        "center_lat": float(lats[row + self.patch_size // 2]),
                        "center_lon": float(lons[col + self.patch_size // 2]),
                        "time_start": str(times[t_start]),
                    }

                    n_patches += 1
                    yield sample

        logger.info(
            f"Patch extraction complete | "
            f"yielded: {n_patches} | skipped: {n_skipped} | "
            f"skip rate: {100*n_skipped/(n_patches+n_skipped+1e-8):.1f}%"
        )


# ---------------------------------------------------------------------------
# Save patches to disk
# ---------------------------------------------------------------------------

def save_patches(
    extractor:   PatchExtractor,
    chl_norm:    np.ndarray,
    obs_mask:    np.ndarray,
    mcar_mask:   np.ndarray,
    mnar_mask:   np.ndarray,
    physics:     np.ndarray,
    wind:        np.ndarray,
    discharge:   np.ndarray,
    bgc_aux:     np.ndarray,
    static:      np.ndarray,
    bloom_mask:  np.ndarray,
    lats:        np.ndarray,
    lons:        np.ndarray,
    times:       np.ndarray,
    output_dir:  str,
    split:       str = "train",
    max_patches: Optional[int] = None,
) -> int:
    """
    Extract patches and save each to a separate .npz file.

    Files are named: {split}_{index:06d}.npz
    An index JSON listing all files and their coordinates is written alongside.

    Returns the number of patches saved.
    """
    import json

    out_dir = Path(output_dir) / split
    os.makedirs(out_dir, exist_ok=True)

    index = []
    count = 0

    for sample in extractor.extract(
        chl_norm  = chl_norm,
        obs_mask  = obs_mask,
        mcar_mask = mcar_mask,
        mnar_mask = mnar_mask,
        physics   = physics,
        wind      = wind,
        discharge = discharge,
        bgc_aux   = bgc_aux,
        static    = static,
        bloom_mask= bloom_mask,
        lats      = lats,
        lons      = lons,
        times     = times,
    ):
        if max_patches and count >= max_patches:
            break

        fname = f"{split}_{count:06d}.npz"
        fpath = out_dir / fname

        meta = {
            "center_lat": sample.pop("center_lat"),
            "center_lon": sample.pop("center_lon"),
            "time_start": sample.pop("time_start"),
        }

        np.savez_compressed(fpath, **sample)

        meta["file"] = fname
        index.append(meta)
        count += 1

        if count % 500 == 0:
            logger.info(f"Saved {count} patches...")

    index_path = Path(output_dir) / f"{split}_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    logger.info(f"Saved {count} patches to {out_dir}")
    logger.info(f"Index written to {index_path}")
    return count


# ---------------------------------------------------------------------------
# Train / validation / test split (temporal block)
# ---------------------------------------------------------------------------

def temporal_split(
    total_timesteps: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[slice, slice, slice]:
    """
    Compute non-overlapping temporal slices for train/val/test splits.
    Uses block holdout (not random) to prevent data leakage.
    """
    test_frac = 1.0 - train_frac - val_frac
    assert test_frac > 0, "train_frac + val_frac must be < 1.0"

    train_end = int(total_timesteps * train_frac)
    val_end   = int(total_timesteps * (train_frac + val_frac))

    train_slice = slice(0, train_end)
    val_slice   = slice(train_end, val_end)
    test_slice  = slice(val_end, total_timesteps)

    logger.info(
        f"Temporal split | "
        f"train: [0,{train_end}) | "
        f"val: [{train_end},{val_end}) | "
        f"test: [{val_end},{total_timesteps})"
    )
    return train_slice, val_slice, test_slice