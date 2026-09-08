# -*- coding: utf-8 -*-
"""
PBL version 02 (Report-style output; robust to missing profiles / missing MPL PBL)

✅ New behavior requested by user:
1) Keep timeline intact (typically 48 profiles: 00:05, 00:35, ... 23:35)
   - If an NRB profile column is missing/incomplete (all NaN or too few valid bins)
     -> ALL outputs for that time become NaN (profile skipped).
2) If "PBL from MPL (m)" is missing / NaN / 0
   -> profile is NOT computed, outputs NaN.
3) If rmin/rmax is missing / NaN / invalid
   -> profile is NOT computed, outputs NaN.

Inputs:
- One Excel file that contains:
  * NRB wide sheet (Range(m) + profile columns)
  * rmin-rmax sheet (Time + rmin + rmax + optional PBL from MPL (m))

Outputs:
- Excel workbook with multiple sheets (same style as v01).

Note:
- Time matching is done by **HH:MM slot** (not nearest), to avoid wrongly borrowing
  rmin/rmax from adjacent slots when some slots are missing.

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd


# -------------------------
# Defaults (keep from v01)
# -------------------------
PBL_VERSION = 3  # v3: ALT chooses the negative (falling-edge) HWCT peak = layer top
FC_CYCLES_PER_M = 0.015  # cutoff wavelength ~67 m; tuned vs MPL ALT (2026-03-26 sample, 48 profiles)
FFT_ORDER = 6
FFT_PAD_FRAC = 0.10
SMOOTH_WIN_BINS = 7
HWCT_TOL_M = 22.5  # ±22.5 m half-window

# New defaults for "incomplete NRB" detection
MIN_VALID_FRAC_DEFAULT = 0.50  # if < 50% finite bins -> treat as incomplete
MIN_VALID_BINS_DEFAULT = 50    # guard for very short / broken columns


# -----------------------------------------------------------------------------
# Track A — algorithm improvements for nrb_profile ALT detection
# -----------------------------------------------------------------------------
# All five helpers are independent and optional. They only activate when the
# corresponding CLI flag / GUI checkbox is set; default behaviour matches the
# legacy fixed-window logic.

def adaptive_window_for_hour(hour: int) -> tuple:
    """Diurnal-adaptive search window for the Chiang Mai / NARIT site.

    Tropical mid-latitude convective boundary layer (CBL) over Chiang Mai shows
    a clear diurnal cycle (Janta et al. 2020; Pani et al. 2019; Lertusee &
    Sherwood 2018; long-term NARIT BL observations). Typical heights:

        Night (SBL):         100 - 500 m    (cool, stable)
        Morning growth:      200 - 1500 m   (rapid CBL expansion)
        Afternoon peak:      1500 - 2800 m  (deepest CBL)
        Evening collapse:    500 - 2000 m   (CBL shrinking)

    The function returns (rmin, rmax) windows that bracket each phase with
    a safety margin so HWCT search avoids residual aerosol layers above.

    Note: burning season (Feb–Apr) often shows lower CBL due to subsidence
    inversion; user can override these defaults via UI.
    """
    h = int(hour) % 24
    if 8 <= h < 11:       # Morning growth
        return (200.0, 2200.0)
    if 11 <= h < 16:      # CBL peak
        return (400.0, 3500.0)
    if 16 <= h < 20:      # Evening collapse
        return (200.0, 2500.0)
    return (150.0, 1500.0)  # Night SBL (20:00 - 07:59)


def peak_anchored_window(
    r_m,
    y,
    rmin_base: float,
    rmax_base: float,
    below_offset_m: float = 300.0,
    above_offset_m: float = 2500.0,
    min_width_m: float = 1000.0,
):
    """Narrow the search window around the NRB peak in the profile.

    Logic: the BL top sits ABOVE the aerosol concentration peak, at a roughly
    *fixed distance* (a few hundred metres to a couple of km) — not at a fixed
    *ratio*. A multiplicative window (peak × factor) collapses when the peak is
    near the surface (e.g. a nocturnal BL whose aerosol peaks at ~100 m), so the
    BL top at ~400 m falls outside [50, 200]. Using additive offsets keeps the
    window wide enough regardless of how low the peak sits.

        rmin = peak - below_offset_m   (trim noise just below the peak)
        rmax = peak + above_offset_m   (reach the BL top well above the peak)

    Both are clipped to the base window; a minimum width is guaranteed.

    Returns (rmin, rmax).
    """
    r_arr = np.asarray(r_m, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = (r_arr >= float(rmin_base)) & (r_arr <= float(rmax_base)) & np.isfinite(y_arr)
    if not np.any(mask):
        return float(rmin_base), float(rmax_base)
    # Light smoothing for stability (boxcar 5 bins) to avoid noise spikes
    y_in = np.where(mask, y_arr, np.nan)
    win = 5
    kernel = np.ones(win, dtype=float) / win
    finite = np.isfinite(y_in)
    y_filled = np.where(finite, y_in, 0.0)
    norm = np.convolve(finite.astype(float), kernel, mode="same")
    y_smooth = np.where(
        norm > 0,
        np.convolve(y_filled, kernel, mode="same") / np.maximum(norm, 1e-9),
        np.nan,
    )
    idx_peak = int(np.nanargmax(np.where(mask, y_smooth, -np.inf)))
    r_peak = float(r_arr[idx_peak])

    rmin = max(float(rmin_base), r_peak - float(below_offset_m))
    rmax = min(float(rmax_base), r_peak + float(above_offset_m))
    # Guarantee a usable width — extend rmax if the window collapsed
    if rmax - rmin < float(min_width_m):
        rmax = min(float(rmax_base), rmin + float(min_width_m))
    if rmax <= rmin:
        # Last-resort fallback — keep the base window
        return float(rmin_base), float(rmax_base)
    return rmin, rmax


def detect_cloud_in_window(
    r_m,
    y,
    rmin: float,
    rmax: float,
    threshold: float,
    surface_bl_max_m: float = 800.0,
) -> bool:
    """Return True only when an ELEVATED cloud is present in [rmin, rmax].

    A naive "max NRB > threshold" test wrongly flags a strong surface boundary
    layer — a nocturnal BL legitimately has very high NRB right at the surface.
    That would make the engine skip exactly the profiles where ALT is easiest
    to find.

    This version flags a cloud only when an above-threshold NRB region exists
    ABOVE the surface BL zone (`surface_bl_max_m`). Surface-attached dense
    aerosol (all high-NRB bins below that height) is treated as the BL, not a
    cloud, and is NOT screened out.
    """
    if threshold is None or float(threshold) <= 0:
        return False
    r_arr = np.asarray(r_m, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = (r_arr >= float(rmin)) & (r_arr <= float(rmax)) & np.isfinite(y_arr)
    if not np.any(mask):
        return False
    high = mask & (y_arr > float(threshold))
    if not np.any(high):
        return False
    # If every above-threshold bin sits within the surface BL zone, it's the
    # boundary layer itself — not a cloud.
    r_high = r_arr[high]
    if float(np.max(r_high)) <= float(surface_bl_max_m):
        return False
    return True


def nrb_cloud_cap_base(
    r_m,
    nrb,
    rmin: float,
    rmax: float,
    threshold: float,
    surface_gap_m: float = 150.0,
) -> Optional[float]:
    """Base range (m) of the lowest ELEVATED cloud in [rmin, rmax], else None.

    NRB-magnitude cloud guard for the ALT search. A cloud (water OR ice) is a
    contiguous region where normalized NRB exceeds `threshold` whose base is
    NOT surface-attached. Unlike a fixed `surface_bl_max_m`, "surface-attached"
    is defined relative to the search start (base within `surface_gap_m` of
    rmin), so a DEEP boundary layer — high NRB rising straight from the surface,
    however thick — is treated as the BL and NOT capped. Only a separated,
    elevated high-NRB layer (the classic cloud signature, including low-δ water
    clouds that the δ ice-screen misses) returns a cap height.

    Caller caps the ALT search rmax below this base so ALT = aerosol top below
    the cloud (instead of skipping the profile or grabbing the cloud edge).
    """
    if threshold is None or float(threshold) <= 0:
        return None
    r = np.asarray(r_m, dtype=float)
    y = np.asarray(nrb, dtype=float)
    mask = (r >= float(rmin)) & (r <= float(rmax)) & np.isfinite(y)
    if not np.any(mask):
        return None

    # Boxcar-3 smooth (mirror detect_cloud_layers) to suppress single-bin spikes.
    y_in = np.where(mask, y, np.nan)
    kernel = np.ones(3, dtype=float) / 3.0
    finite = np.isfinite(y_in)
    y_filled = np.where(finite, y_in, 0.0)
    norm = np.convolve(finite.astype(float), kernel, mode="same")
    y_smooth = np.where(norm > 0,
                        np.convolve(y_filled, kernel, mode="same") / np.maximum(norm, 1e-9),
                        np.nan)

    above = (y_smooth >= float(threshold)) & mask
    if not np.any(above):
        return None
    breaks = np.where(np.diff(above.astype(int)) != 0)[0] + 1
    chunks = [c for c in np.split(np.arange(len(above)), breaks)
              if c.size and bool(above[c[0]])]
    # chunks are in ascending range order; return the first ELEVATED one.
    for chunk in chunks:
        base_m = float(r[int(chunk[0])])
        if base_m <= float(rmin) + float(surface_gap_m):
            continue  # surface-attached boundary layer, not a cloud
        return base_m
    return None


def select_lowest_stable_edge(
    W,
    r_use,
    rmin: float,
    rmax: float,
    threshold_factor: float = 0.30,
):
    """Pick the lowest-altitude HWCT negative peak that crosses a threshold.

    Default behaviour (legacy) is to pick the SINGLE most-negative HWCT peak in
    the window — that often lands on a residual layer instead of the surface
    BL top. This helper instead returns the FIRST (lowest-R) statistically
    significant negative peak: |W| >= threshold_factor * max(|W_negative|).

    Returns (r_alt, W_value, mode_str).
    """
    W_arr = np.asarray(W, dtype=float)
    r_arr = np.asarray(r_use, dtype=float)
    m = np.isfinite(W_arr) & (r_arr >= float(rmin)) & (r_arr <= float(rmax))
    if not np.any(m):
        return float("nan"), float("nan"), "no_valid_hwct"
    W_win = W_arr[m]
    r_win = r_arr[m]
    w_min = float(np.nanmin(W_win))
    if w_min >= 0:
        return float("nan"), float("nan"), "no_negative_edge"
    thresh = float(threshold_factor) * w_min  # also negative
    candidates = W_win < thresh
    if not np.any(candidates):
        # No bin meets threshold — fall back to strongest peak
        idx = int(np.nanargmin(W_win))
        return float(r_win[idx]), float(W_win[idx]), "strongest_fallback"
    # Group consecutive candidate bins into chunks; pick the lowest-R chunk
    indices = np.where(candidates)[0]
    breaks = np.where(np.diff(indices) > 1)[0]
    chunks = np.split(indices, breaks + 1) if breaks.size else [indices]
    first_chunk = chunks[0]
    local_min_in_chunk = first_chunk[int(np.nanargmin(W_win[first_chunk]))]
    return float(r_win[local_min_in_chunk]), float(W_win[local_min_in_chunk]), "lowest_stable"


def detect_cloud_layers(
    r_m,
    nrb,
    threshold: float = 0.3,
    min_thickness_m: float = 100.0,
    max_layers: int = 3,
    r_min_m: float = 100.0,
    r_max_m: float = 15000.0,
):
    """Detect cloud layers from NRB profile (no inversion required).

    A cloud layer is a contiguous range bin region where smoothed NRB exceeds
    `threshold` (in normalized 0–1 units). Sharp NRB peaks above ambient aerosol
    indicate condensed water / ice (Mie scattering with very high β).

    Parameters
    ----------
    r_m : ndarray
        Range axis (m).
    nrb : ndarray
        Normalized NRB profile (0–1).
    threshold : float
        Minimum NRB value to flag as cloud (typical 0.3 — clouds usually >0.5
        but allowing 0.3 catches thin cirrus and partial layers).
    min_thickness_m : float
        Reject thin spurious peaks; require at least this thickness (m).
    max_layers : int
        Return up to N strongest cloud layers (sorted by peak NRB).
    r_min_m, r_max_m : float
        Search window — exclude near-surface noise and below ALT zone.

    Returns
    -------
    layers : list of dict
        Each dict has: base_m, top_m, peak_R_m, peak_value, thickness_m,
        rough_OD (optical depth proxy from cloud attenuation).
    """
    r_arr = np.asarray(r_m, dtype=float)
    y = np.asarray(nrb, dtype=float)
    base_mask = (r_arr >= float(r_min_m)) & (r_arr <= float(r_max_m)) & np.isfinite(y)
    if not np.any(base_mask):
        return []

    # Light smoothing (boxcar 3) to suppress single-bin spikes
    y_in = np.where(base_mask, y, np.nan)
    kernel = np.ones(3, dtype=float) / 3.0
    finite = np.isfinite(y_in)
    y_filled = np.where(finite, y_in, 0.0)
    norm = np.convolve(finite.astype(float), kernel, mode="same")
    y_smooth = np.where(
        norm > 0,
        np.convolve(y_filled, kernel, mode="same") / np.maximum(norm, 1e-9),
        np.nan,
    )

    above_thresh = (y_smooth >= float(threshold)) & base_mask

    # Find contiguous chunks of above-threshold bins
    if not np.any(above_thresh):
        return []
    chunk_breaks = np.where(np.diff(above_thresh.astype(int)) != 0)[0] + 1
    chunks = np.split(np.arange(len(above_thresh)), chunk_breaks)
    cloud_chunks = [c for c in chunks if c.size and bool(above_thresh[c[0]])]

    layers = []
    for chunk in cloud_chunks:
        if chunk.size < 2:
            continue
        i_base, i_top = int(chunk[0]), int(chunk[-1])
        base_m = float(r_arr[i_base])
        top_m = float(r_arr[i_top])
        thickness = top_m - base_m
        if thickness < float(min_thickness_m):
            continue
        peak_idx_local = int(np.nanargmax(y_smooth[chunk]))
        peak_idx = int(chunk[peak_idx_local])
        peak_R = float(r_arr[peak_idx])
        peak_val = float(y_smooth[peak_idx])

        # Rough optical depth proxy: log-ratio NRB before/after cloud
        # OD ≈ 0.5 × ln(NRB_below / NRB_above)
        # Use median of 5 bins below base and 5 above top
        below_idx = np.arange(max(0, i_base - 10), i_base)
        above_idx = np.arange(i_top + 1, min(len(y_smooth), i_top + 11))
        below_idx = below_idx[base_mask[below_idx]] if below_idx.size else below_idx
        above_idx = above_idx[base_mask[above_idx]] if above_idx.size else above_idx
        if below_idx.size >= 2 and above_idx.size >= 2:
            nrb_below = float(np.nanmedian(y_smooth[below_idx]))
            nrb_above = float(np.nanmedian(y_smooth[above_idx]))
            if nrb_below > 0 and nrb_above > 0 and nrb_below > nrb_above:
                od_rough = 0.5 * np.log(nrb_below / nrb_above)
            else:
                od_rough = float("nan")
        else:
            od_rough = float("nan")

        layers.append({
            "base_m": base_m,
            "top_m": top_m,
            "peak_R_m": peak_R,
            "peak_value": peak_val,
            "thickness_m": thickness,
            "rough_OD": od_rough,
        })

    # Sort by peak value descending; keep top max_layers
    layers.sort(key=lambda d: d["peak_value"], reverse=True)
    return layers[:int(max_layers)]


def temporal_smooth_alt(alt_series, window: int = 3, outlier_threshold_m: float = 500.0):
    """Apply rolling-median smoothing with outlier rejection on the ALT time series.

    Steps:
        1. Rolling median over `window` (centred, edge-tolerant).
        2. Replace points whose distance from the median exceeds
           `outlier_threshold_m` with the median value.

    Returns a new Series; the original is not modified.
    """
    if window is None or int(window) <= 1:
        return alt_series.copy()
    s = pd.Series(alt_series).astype(float)
    med = s.rolling(window=int(window), center=True, min_periods=1).median()
    diff = (s - med).abs()
    outliers = diff > float(outlier_threshold_m)
    out = s.copy()
    out[outliers] = med[outliers]
    return out


# -------------------------
# Helpers
# -------------------------
def _odd(k: int) -> int:
    k = int(k)
    if k <= 1:
        return 1
    return k if k % 2 == 1 else (k + 1)


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    k = _odd(k)
    if k == 1:
        return np.asarray(x, float)
    w = np.ones(k, dtype=float) / k
    return np.convolve(np.asarray(x, float), w, mode="same")


def fft_lowpass_fixed_fc(
    x: np.ndarray,
    dr: float,
    fc: float = FC_CYCLES_PER_M,
    order: int = FFT_ORDER,
    pad_frac: float = FFT_PAD_FRAC,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """FFT-domain Butterworth-like low-pass with fixed cutoff in cycles/m."""
    x = np.asarray(x, float)
    n0 = x.size
    if n0 < 8 or not np.isfinite(dr) or dr <= 0:
        return x.copy()

    # If the entire signal is NaN, return it directly (avoid FFT NaN pollution cost)
    if not np.any(np.isfinite(x)):
        return np.full_like(x, np.nan, dtype=float)

    pad = int(max(1, round(float(pad_frac) * n0)))
    xpad = np.pad(x, (pad, pad), mode=pad_mode)

    n = xpad.size
    X = np.fft.rfft(xpad)
    f = np.fft.rfftfreq(n, d=dr)  # cycles/m

    H = 1.0 / np.sqrt(1.0 + (f / fc) ** (2 * order))
    H[0] = 1.0

    ypad = np.fft.irfft(X * H, n=n)
    y = ypad[pad: pad + n0]
    return np.asarray(y, float)


def hwct_haar_step_right_minus_left(x: np.ndarray, half_bins: int) -> np.ndarray:
    """Haar step (HWCT-like): W[i] = mean(right) - mean(left)."""
    x = np.asarray(x, float)
    n = x.size
    half = int(max(1, half_bins))

    W = np.full(n, np.nan, dtype=float)
    if n < 2 * half + 1:
        return W

    # If there are NaNs, cumsum will propagate; keep behavior consistent by refusing
    # to compute where window contains NaN (we handle mask with finite(W)).
    cs = np.cumsum(np.insert(x, 0, 0.0))
    for i in range(half, n - half):
        l0, l1 = i - half, i
        r0, r1 = i, i + half
        left_mean = (cs[l1] - cs[l0]) / (l1 - l0)
        right_mean = (cs[r1] - cs[r0]) / (r1 - r0)
        W[i] = right_mean - left_mean
    return W


def prepare_profile_for_fft(r_m: np.ndarray, y_raw: np.ndarray) -> Dict[str, np.ndarray | float | int | bool | str]:
    """Prepare a single NRB profile for FFT/HWCT.

    Strategy:
    - sort by range,
    - keep only the contiguous analysis span from the first finite point to the last finite point,
    - fill internal NaN gaps by 1-D interpolation inside that span only,
    - leave leading/trailing bins outside the usable span excluded from analysis.
    """
    r_m = np.asarray(r_m, float)
    y_raw = np.asarray(y_raw, float)

    sidx = np.argsort(r_m)
    r_sorted = r_m[sidx]
    y_sorted = y_raw[sidx]

    valid = np.isfinite(r_sorted) & np.isfinite(y_sorted)
    if not np.any(valid):
        return {
            "status": "all_nan",
            "r_sorted": r_sorted,
            "y_sorted": y_sorted,
            "r_use": np.asarray([], float),
            "y_use": np.asarray([], float),
            "first_idx": None,
            "last_idx": None,
            "trim_leading_bins": int(len(y_sorted)),
            "trim_trailing_bins": 0,
            "internal_nan_filled": 0,
            "analysis_bins_used": 0,
            "analysis_range_max_m": np.nan,
        }

    valid_idx = np.where(valid)[0]
    first_idx = int(valid_idx[0])
    last_idx = int(valid_idx[-1])

    r_use = r_sorted[first_idx:last_idx + 1].astype(float, copy=True)
    y_use = y_sorted[first_idx:last_idx + 1].astype(float, copy=True)

    finite_use = np.isfinite(y_use)
    internal_nan_filled = int((~finite_use).sum())

    if not np.any(finite_use):
        return {
            "status": "all_nan",
            "r_sorted": r_sorted,
            "y_sorted": y_sorted,
            "r_use": np.asarray([], float),
            "y_use": np.asarray([], float),
            "first_idx": first_idx,
            "last_idx": last_idx,
            "trim_leading_bins": first_idx,
            "trim_trailing_bins": int(len(y_sorted) - 1 - last_idx),
            "internal_nan_filled": internal_nan_filled,
            "analysis_bins_used": 0,
            "analysis_range_max_m": np.nan,
        }

    if internal_nan_filled > 0:
        x_good = r_use[finite_use]
        y_good = y_use[finite_use]
        if x_good.size < 2:
            return {
                "status": "too_few_points",
                "r_sorted": r_sorted,
                "y_sorted": y_sorted,
                "r_use": np.asarray([], float),
                "y_use": np.asarray([], float),
                "first_idx": first_idx,
                "last_idx": last_idx,
                "trim_leading_bins": first_idx,
                "trim_trailing_bins": int(len(y_sorted) - 1 - last_idx),
                "internal_nan_filled": internal_nan_filled,
                "analysis_bins_used": int(x_good.size),
                "analysis_range_max_m": float(np.nanmax(x_good)) if x_good.size else np.nan,
            }
        y_use = np.interp(r_use, x_good, y_good).astype(float)

    return {
        "status": "ok",
        "r_sorted": r_sorted,
        "y_sorted": y_sorted,
        "r_use": r_use,
        "y_use": y_use,
        "first_idx": first_idx,
        "last_idx": last_idx,
        "trim_leading_bins": first_idx,
        "trim_trailing_bins": int(len(y_sorted) - 1 - last_idx),
        "internal_nan_filled": internal_nan_filled,
        "analysis_bins_used": int(len(r_use)),
        "analysis_range_max_m": float(np.nanmax(r_use)) if len(r_use) else np.nan,
    }


def denoise_profile_preserve_shape(
    r_m: np.ndarray,
    y_raw: np.ndarray,
    fc: float,
    order: int,
    pad_frac: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray | float | int | bool | str]]:
    """Denoise one profile while preserving original length/order.

    The denoised signal is only written back into the valid analysis span. Bins outside
    that span remain NaN.
    """
    prep = prepare_profile_for_fft(r_m, y_raw)
    y_out = np.full_like(np.asarray(y_raw, float), np.nan, dtype=float)

    if prep["status"] != "ok" or int(prep["analysis_bins_used"]) < 8:
        return y_out, prep

    r_use = np.asarray(prep["r_use"], float)
    y_use = np.asarray(prep["y_use"], float)
    dr = float(np.nanmedian(np.diff(r_use)))
    if not np.isfinite(dr) or dr <= 0:
        return y_out, prep

    y_dn = fft_lowpass_fixed_fc(y_use, dr=dr, fc=fc, order=order, pad_frac=pad_frac)

    first_idx = int(prep["first_idx"])
    last_idx = int(prep["last_idx"])
    sidx = np.argsort(np.asarray(r_m, float))
    y_sorted_out = np.full_like(np.asarray(y_raw, float), np.nan, dtype=float)
    y_sorted_out[first_idx:last_idx + 1] = y_dn

    inv = np.empty_like(sidx)
    inv[sidx] = np.arange(len(sidx))
    y_out = y_sorted_out[inv]
    return y_out, prep


def read_nrb_wide_excel(path: Path, sheet: str) -> Tuple[np.ndarray, List[pd.Timestamp], np.ndarray, List[str]]:
    """Reads a wide NRB table: col0=Range(m), col1..=profiles (datetime-like headers)."""
    df = pd.read_excel(path, sheet_name=sheet)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty or df.shape[1] < 2:
        raise ValueError(f"NRB sheet '{sheet}' is empty or not wide-format (need Range + profiles).")

    range_col = df.columns[0]
    r_m = pd.to_numeric(df[range_col], errors="coerce").to_numpy(float)
    keep = np.isfinite(r_m)
    df = df.loc[keep].copy()
    r_m = r_m[keep].astype(float)

    prof_cols = list(df.columns[1:])
    raw_colnames = [str(c) for c in prof_cols]

    times: List[pd.Timestamp] = []
    for c in prof_cols:
        t = pd.to_datetime(c, errors="coerce")
        if pd.isna(t):
            t = pd.to_datetime(str(c).strip(), errors="coerce")
        if pd.isna(t):
            raise ValueError(f"Cannot parse profile column header as datetime: {c!r}")
        times.append(pd.Timestamp(t))

    mat = df[prof_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return r_m, times, mat, raw_colnames


def read_rmin_rmax_excel(path: Path, sheet: str = "rmin-rmax") -> pd.DataFrame:
    """
    Reads rmin-rmax table.

    Required columns (case-insensitive): Time, rmin, rmax
    Optional: "PBL from MPL (m)" (any column containing both 'pbl' and 'mpl')

    IMPORTANT (v02): keep rows even if rmin/rmax/MPL-PBL are NaN.
    Only Time is required.
    """
    df = pd.read_excel(path, sheet_name=sheet)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty:
        raise ValueError(f"Sheet '{sheet}' is empty.")

    cols = {str(c).strip().lower(): c for c in df.columns}

    def pick(*cands: str) -> Optional[str]:
        for k in cands:
            if k in cols:
                return cols[k]
        return None

    time_col = pick("time", "datetime", "date time", "timestamp")
    rmin_col = pick("rmin", "rmin_m", "rmin (m)", "r min", "rmin(m)")
    rmax_col = pick("rmax", "rmax_m", "rmax (m)", "r max", "rmax(m)")

    mpl_col = None
    for key, orig in cols.items():
        # Accept both old PBL naming and new ALT naming.
        if (("pbl" in key or "alt" in key) and "mpl" in key) or key in {"alt_mpl_m", "pbl_mpl_m"}:
            mpl_col = orig
            break

    if time_col is None:
        raise ValueError("rmin-rmax sheet must have a 'Time' column (case-insensitive).")

    # Build output with missing columns allowed (filled as NaN)
    use_cols = [time_col]
    if rmin_col:
        use_cols.append(rmin_col)
    if rmax_col:
        use_cols.append(rmax_col)
    if mpl_col:
        use_cols.append(mpl_col)

    out = df[use_cols].copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    if rmin_col:
        out[rmin_col] = pd.to_numeric(out[rmin_col], errors="coerce")
    else:
        out["__rmin__"] = np.nan
        rmin_col = "__rmin__"
    if rmax_col:
        out[rmax_col] = pd.to_numeric(out[rmax_col], errors="coerce")
    else:
        out["__rmax__"] = np.nan
        rmax_col = "__rmax__"

    if mpl_col:
        out[mpl_col] = pd.to_numeric(out[mpl_col], errors="coerce")
    else:
        out["__mpl__"] = np.nan
        mpl_col = "__mpl__"

    out = out.dropna(subset=[time_col]).sort_values(time_col)
    out = out.rename(columns={time_col: "Time", rmin_col: "rmin", rmax_col: "rmax", mpl_col: "PBL from MPL (m)"})

    # slot for robust matching
    out["Slot"] = out["Time"].dt.strftime("%H:%M")

    # keep last if duplicate slot
    out = out.dropna(subset=["Slot"]).drop_duplicates(subset=["Slot"], keep="last")

    # ensure order
    out = out[["Time", "Slot", "PBL from MPL (m)", "rmin", "rmax"]]
    return out


def compute_pbl_for_profile(
    r_m: np.ndarray,
    y_raw: np.ndarray,
    rmin: float,
    rmax: float,
    fc: float,
    order: int,
    pad_frac: float,
    smooth_win: int,
    tol_m: float,
) -> Dict[str, float]:
    """Compute HWCT peaks inside [rmin,rmax] and choose per v01 rule."""
    r_m = np.asarray(r_m, float)
    y_raw = np.asarray(y_raw, float)

    all_nan = not np.any(np.isfinite(y_raw))
    prep = prepare_profile_for_fft(r_m, y_raw)

    if prep["status"] != "ok" or int(prep["analysis_bins_used"]) < 8:
        return {
            "dr": np.nan,
            "NRB_all_nan": all_nan,
            "PBL_pos_m": np.nan, "HWCT_peak_pos": np.nan,
            "PBL_neg_m": np.nan, "HWCT_peak_neg": np.nan,
            "Chosen_mode": "none",
            "PBL_TR40_m": np.nan, "HWCT_peak_chosen": np.nan,
            "Analysis_status": str(prep["status"]),
            "Analysis_bins_used": int(prep["analysis_bins_used"]),
            "Analysis_range_max_m": float(prep["analysis_range_max_m"]) if np.isfinite(prep["analysis_range_max_m"]) else np.nan,
            "Trim_leading_bins": int(prep["trim_leading_bins"]),
            "Trim_trailing_bins": int(prep["trim_trailing_bins"]),
            "Internal_nan_filled": int(prep["internal_nan_filled"]),
        }

    r_use = np.asarray(prep["r_use"], float)
    y_use = np.asarray(prep["y_use"], float)

    dr = float(np.nanmedian(np.diff(r_use)))
    if not np.isfinite(dr) or dr <= 0:
        return {
            "dr": np.nan,
            "NRB_all_nan": all_nan,
            "PBL_pos_m": np.nan, "HWCT_peak_pos": np.nan,
            "PBL_neg_m": np.nan, "HWCT_peak_neg": np.nan,
            "Chosen_mode": "none",
            "PBL_TR40_m": np.nan, "HWCT_peak_chosen": np.nan,
            "Analysis_status": "invalid_dr",
            "Analysis_bins_used": int(prep["analysis_bins_used"]),
            "Analysis_range_max_m": float(prep["analysis_range_max_m"]) if np.isfinite(prep["analysis_range_max_m"]) else np.nan,
            "Trim_leading_bins": int(prep["trim_leading_bins"]),
            "Trim_trailing_bins": int(prep["trim_trailing_bins"]),
            "Internal_nan_filled": int(prep["internal_nan_filled"]),
        }

    # Per Brooks (2003) and standard WCT/HWCT methodology, apply a single
    # low-pass step (FFT) before the wavelet transform; an additional moving
    # average would over-smooth and shift real BL edge features.
    y_dn = fft_lowpass_fixed_fc(y_use, dr=dr, fc=fc, order=order, pad_frac=pad_frac)

    half_bins = int(max(1, round(float(tol_m) / dr)))
    W = hwct_haar_step_right_minus_left(y_dn, half_bins)

    mask = np.isfinite(W) & (r_use >= float(rmin)) & (r_use <= float(rmax))
    if not np.any(mask):
        return {
            "dr": dr,
            "NRB_all_nan": all_nan,
            "PBL_pos_m": np.nan, "HWCT_peak_pos": np.nan,
            "PBL_neg_m": np.nan, "HWCT_peak_neg": np.nan,
            "Chosen_mode": "none",
            "PBL_TR40_m": np.nan, "HWCT_peak_chosen": np.nan,
            "Analysis_status": "no_valid_hwct_in_window",
            "Analysis_bins_used": int(prep["analysis_bins_used"]),
            "Analysis_range_max_m": float(prep["analysis_range_max_m"]) if np.isfinite(prep["analysis_range_max_m"]) else np.nan,
            "Trim_leading_bins": int(prep["trim_leading_bins"]),
            "Trim_trailing_bins": int(prep["trim_trailing_bins"]),
            "Internal_nan_filled": int(prep["internal_nan_filled"]),
        }

    r_win = r_use[mask]
    W_win = W[mask]

    pos_mask = W_win > 0
    if np.any(pos_mask):
        jpos = int(np.nanargmax(W_win))
        pbl_pos = float(r_win[jpos])
        peak_pos = float(W_win[jpos])
    else:
        pbl_pos = np.nan
        peak_pos = np.nan

    jneg = int(np.nanargmin(W_win))
    pbl_neg = float(r_win[jneg])
    peak_neg = float(W_win[jneg])

    # ALT (aerosol layer top) is where backscatter drops with height. With
    # W = mean(upper) - mean(lower), that falling edge is the most NEGATIVE
    # peak of W. Prefer it; fall back to the positive (rising) edge only when
    # the window contains no negative W at all.
    if np.any(W_win < 0):
        chosen_mode = "negative"
        pbl = pbl_neg
        peak = peak_neg
    else:
        chosen_mode = "positive"
        pbl = pbl_pos
        peak = peak_pos

    return {
        "dr": dr,
        "NRB_all_nan": all_nan,
        "PBL_pos_m": pbl_pos, "HWCT_peak_pos": peak_pos,
        "PBL_neg_m": pbl_neg, "HWCT_peak_neg": peak_neg,
        "Chosen_mode": chosen_mode,
        "PBL_TR40_m": pbl, "HWCT_peak_chosen": peak,
        "Analysis_status": "ok",
        "Analysis_bins_used": int(prep["analysis_bins_used"]),
        "Analysis_range_max_m": float(prep["analysis_range_max_m"]) if np.isfinite(prep["analysis_range_max_m"]) else np.nan,
        "Trim_leading_bins": int(prep["trim_leading_bins"]),
        "Trim_trailing_bins": int(prep["trim_trailing_bins"]),
        "Internal_nan_filled": int(prep["internal_nan_filled"]),
    }


def _profile_validity(y: np.ndarray, min_valid_frac: float, min_valid_bins: int) -> Tuple[bool, float, int]:
    y = np.asarray(y, float)
    n = int(y.size)
    k = int(np.isfinite(y).sum())
    frac = (k / n) if n > 0 else 0.0
    ok = (k >= int(min_valid_bins)) and (frac >= float(min_valid_frac))
    return ok, float(frac), int(k)



def _empty_profile_result(reason: str, r_m: np.ndarray, y_raw: np.ndarray, dr_global: float) -> Dict[str, float]:
    """Return a result dict with the same keys as compute_pbl_for_profile(), filled with NaN."""
    prep = prepare_profile_for_fft(r_m, y_raw)
    return {
        "dr": dr_global,
        "NRB_all_nan": not np.any(np.isfinite(y_raw)),
        "PBL_pos_m": np.nan,
        "HWCT_peak_pos": np.nan,
        "PBL_neg_m": np.nan,
        "HWCT_peak_neg": np.nan,
        "Chosen_mode": "skipped",
        "PBL_TR40_m": np.nan,
        "HWCT_peak_chosen": np.nan,
        "Analysis_status": f"skipped:{reason}",
        "Analysis_bins_used": int(prep["analysis_bins_used"]),
        "Analysis_range_max_m": float(prep["analysis_range_max_m"]) if np.isfinite(prep["analysis_range_max_m"]) else np.nan,
        "Trim_leading_bins": int(prep["trim_leading_bins"]),
        "Trim_trailing_bins": int(prep["trim_trailing_bins"]),
        "Internal_nan_filled": int(prep["internal_nan_filled"]),
    }


def _valid_window(rmin: float, rmax: float) -> bool:
    return np.isfinite(rmin) and np.isfinite(rmax) and float(rmax) > float(rmin)


# -----------------------------------------------------------------------------
# Depolarization support (Track 2) — optional aid to ALT detection
# -----------------------------------------------------------------------------
def load_depol_profiles(path: Path, sheet: str = "Depol_delta_v"):
    """Read a depol workbook -> (range_grid, {slot_HHMM: delta_v array})."""
    df = pd.read_excel(path, sheet_name=sheet)
    r = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(float)
    out: Dict[str, np.ndarray] = {}
    for c in df.columns[1:]:
        ts = pd.to_datetime(c, errors="coerce")
        slot = ts.strftime("%H:%M") if pd.notna(ts) else str(c)
        out[slot] = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
    return r, out


def load_snr_profiles(path: Path, sheet: str = "SNR"):
    """Read an SNR sheet (Range(m) + timestamp columns) from the NRB/depol
    workbook -> (range_grid, {slot_HHMM: snr array}). Same wide layout as the
    NRB profile, so the columns line up slot-for-slot."""
    df = pd.read_excel(path, sheet_name=sheet)
    r = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(float)
    out: Dict[str, np.ndarray] = {}
    for c in df.columns[1:]:
        ts = pd.to_datetime(c, errors="coerce")
        slot = ts.strftime("%H:%M") if pd.notna(ts) else str(c)
        out[slot] = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
    return r, out


def snr_trusted_top_m(r_m, snr, snr_min=3.0, smooth_bins=15):
    """Contiguous SNR-trusted-range top [m] from a SMOOTHED SNR profile (rolling
    median): walk UP from the peak-SNR bin until the smoothed SNR drops below
    snr_min. Mirrors nrb_engine.snr_trusted_top so the ALT cap matches the gate.
    Smoothing ignores the ~25% of daytime noise bins that randomly exceed 3σ."""
    r = np.asarray(r_m, float)
    s = np.asarray(snr, float)
    valid = np.isfinite(s) & np.isfinite(r)
    if not valid.any():
        return np.nan
    sm = pd.Series(np.where(valid, s, np.nan)).rolling(
        max(1, int(smooth_bins)), center=True,
        min_periods=max(3, int(smooth_bins) // 2)).median().to_numpy()
    ok = np.isfinite(sm) & (sm >= float(snr_min))
    if not ok.any():
        return np.nan
    core = int(np.nanargmax(np.where(np.isfinite(sm), sm, -np.inf)))
    if not ok[core]:
        return np.nan
    top = float(r[core])
    for i in range(core, len(r)):
        if not ok[i]:
            break
        top = float(r[i])
    return top


def depol_ice_cloud_base(r_m, delta, rmin, rmax, ice_thr, surface_bl_max_m=800.0):
    """Lowest range in (rmin,rmax] above the surface BL where delta_v exceeds
    `ice_thr` (ice cloud / strongly non-spherical). Returns that range or None.
    Used to cap the ALT search window below an overlying cloud."""
    r = np.asarray(r_m, float)
    d = np.asarray(delta, float)
    m = (r >= float(rmin)) & (r <= float(rmax)) & (r > float(surface_bl_max_m)) \
        & np.isfinite(d) & (d > float(ice_thr))
    if not np.any(m):
        return None
    return float(np.min(r[m]))


def compute_depol_alt(r_m, delta, rmin, rmax, fc, order, pad_frac, tol_m):
    """Independent ALT estimate from the depol profile: the aerosol top is the
    most-negative HWCT peak of delta_v (delta falls from aerosol value in the
    BL to molecular above). Returns range [m] or NaN."""
    prep = prepare_profile_for_fft(r_m, delta)
    if prep["status"] != "ok" or int(prep["analysis_bins_used"]) < 8:
        return np.nan
    r_use = np.asarray(prep["r_use"], float)
    y_use = np.asarray(prep["y_use"], float)
    dr = float(np.nanmedian(np.diff(r_use)))
    if not np.isfinite(dr) or dr <= 0:
        return np.nan
    y_dn = fft_lowpass_fixed_fc(y_use, dr=dr, fc=fc, order=order, pad_frac=pad_frac)
    half = int(max(1, round(float(tol_m) / dr)))
    W = hwct_haar_step_right_minus_left(y_dn, half)
    mask = np.isfinite(W) & (r_use >= float(rmin)) & (r_use <= float(rmax))
    if not np.any(mask):
        return np.nan
    W_win = W[mask]
    r_win = r_use[mask]
    if not np.any(W_win < 0):
        return np.nan
    return float(r_win[int(np.nanargmin(W_win))])


def main():
    ap = argparse.ArgumentParser(description="ALT report-style (Excel inputs + multi-sheet outputs).")
    ap.add_argument("--nrb", required=True, help="NRB Excel file path")
    ap.add_argument("--sheet", required=True, help="NRB profile sheet name (wide format)")
    ap.add_argument("--rminrmax_sheet", default="rmin-rmax", help="rmin-rmax sheet name (default: rmin-rmax)")
    ap.add_argument("--out", default="ALT_output.xlsx", help="Output Excel file name")

    ap.add_argument("--fc", type=float, default=FC_CYCLES_PER_M)
    ap.add_argument("--order", type=int, default=FFT_ORDER)
    ap.add_argument("--pad_frac", type=float, default=FFT_PAD_FRAC)
    ap.add_argument("--smooth_win", type=int, default=SMOOTH_WIN_BINS)
    ap.add_argument("--tol_m", type=float, default=HWCT_TOL_M)

    ap.add_argument("--min_valid_frac", type=float, default=MIN_VALID_FRAC_DEFAULT,
                    help="If finite bins / total bins < this => treat NRB profile as incomplete and skip")
    ap.add_argument("--min_valid_bins", type=int, default=MIN_VALID_BINS_DEFAULT,
                    help="Minimum number of finite bins required to compute")

    ap.add_argument(
        "--detection_mode",
        choices=["mpl_guided", "nrb_profile", "dual"],
        default="dual",
        help=(
            "mpl_guided = search only inside MPL-derived rmin/rmax; "
            "nrb_profile = search inside fixed NRB profile range; "
            "dual = calculate both and use MPL-guided as selected result if available"
        ),
    )
    ap.add_argument("--profile_rmin", type=float, default=0.0,
                    help="Lower range limit for independent NRB-profile ALT mode")
    ap.add_argument("--profile_rmax", type=float, default=4000.0,
                    help="Upper range limit for independent NRB-profile ALT mode")

    # ── Track A improvements (nrb_profile mode) ─────────────────────────────
    ap.add_argument("--adaptive_window", action="store_true",
                    help="Use diurnal-adaptive rmin/rmax for nrb_profile mode (Chiang Mai/NARIT tuned)")
    ap.add_argument("--peak_anchored", action="store_true",
                    help="Narrow nrb_profile window around NRB peak before HWCT")
    ap.add_argument("--cloud_screen_threshold", type=float, default=0.0,
                    help="If max NRB inside window > this (in normalized units), mark as cloudy and skip "
                         "(0 = disabled, typical 0.5-0.7)")
    ap.add_argument("--lowest_edge", action="store_true",
                    help="Multi-layer selection: prefer lowest stable HWCT negative peak instead of strongest")
    ap.add_argument("--lowest_edge_thr_factor", type=float, default=0.30,
                    help="Threshold factor for lowest-edge selection (fraction of strongest |W|)")
    ap.add_argument("--temporal_smooth_window", type=int, default=0,
                    help="Median-smooth ALT time series over N slots after detection (0 = disabled)")
    ap.add_argument("--temporal_smooth_outlier_m", type=float, default=500.0,
                    help="Outlier threshold for temporal smoothing (m); points beyond this are replaced")

    # ── Cloud detection (Track 3) ───────────────────────────────────────────
    ap.add_argument("--cloud_detect", action="store_true",
                    help="Run cloud layer detection on each profile (outputs Cloud_results sheet)")
    ap.add_argument("--cloud_threshold", type=float, default=0.3,
                    help="NRB threshold (normalized 0-1) for cloud detection (default 0.3)")
    ap.add_argument("--cloud_min_thickness_m", type=float, default=100.0,
                    help="Minimum cloud layer thickness in meters (default 100 m)")
    ap.add_argument("--cloud_max_layers", type=int, default=3,
                    help="Max number of cloud layers to report per profile (default 3)")
    ap.add_argument("--cloud_search_rmin", type=float, default=100.0,
                    help="Lower range bound for cloud search (m, default 100)")
    ap.add_argument("--cloud_search_rmax", type=float, default=15000.0,
                    help="Upper range bound for cloud search (m, default 15000)")
    ap.add_argument("--nrb_cloud_cap", action="store_true",
                    help="Cap the ALT search below the lowest ELEVATED NRB cloud "
                         "(catches water + ice clouds; primary cloud guard so ALT "
                         "is not mistaken for a cloud edge). Uses --cloud_threshold.")
    ap.add_argument("--nrb_cloud_surface_gap_m", type=float, default=150.0,
                    help="A high-NRB layer with base within this gap of the search "
                         "start is the surface BL (any depth), not a cloud (default 150 m)")

    # ── Depolarization aid (Track 2, optional) ──────────────────────────────
    ap.add_argument("--depol_file", default="",
                    help="Path to depol workbook (Step 6 output) to aid ALT detection")
    ap.add_argument("--depol_sheet", default="Depol_delta_v",
                    help="Sheet with delta_v (Range + per-time columns)")
    ap.add_argument("--depol_cloud_screen", action="store_true",
                    help="Cap ALT search below an ice cloud detected from delta_v")
    ap.add_argument("--depol_ice_thr", type=float, default=0.35,
                    help="delta_v above this = ice cloud / strongly non-spherical (default 0.35)")
    ap.add_argument("--depol_confirm", action="store_true",
                    help="Add an independent delta_v-based ALT and agree/disagree flag")
    ap.add_argument("--depol_confirm_tol_m", type=float, default=300.0,
                    help="NRB vs depol ALT agreement tolerance (m, default 300)")

    # ── SNR-trusted-range cap (Track: signal quality) ─────────────────────────
    ap.add_argument("--snr_cap", action="store_true",
                    help="Cap the ALT search at the SNR-trusted range top (from the "
                         "SNR sheet) so the search never enters the noise floor")
    ap.add_argument("--snr_sheet", default="SNR",
                    help="SNR sheet name in the NRB workbook (default: SNR)")
    ap.add_argument("--snr_cap_min", type=float, default=3.0,
                    help="SNR threshold for the trusted-range top (default 3.0)")

    args = ap.parse_args()

    in_path = Path(args.nrb)
    if not in_path.exists():
        raise FileNotFoundError(f"NRB file not found: {in_path}")

    # read inputs
    r_m, profile_times, mat_raw, _raw_cols = read_nrb_wide_excel(in_path, sheet=args.sheet)
    try:
        df_rr = read_rmin_rmax_excel(in_path, sheet=args.rminrmax_sheet)
    except Exception:
        # Independent mode can run without a real rmin-rmax table.
        # The GUI normally creates an empty aligned table for this case.
        df_rr = pd.DataFrame({"Time": profile_times})
        df_rr["Time"] = pd.to_datetime(df_rr["Time"], errors="coerce")
        df_rr["Slot"] = df_rr["Time"].dt.strftime("%H:%M")
        df_rr["PBL from MPL (m)"] = np.nan
        df_rr["rmin"] = np.nan
        df_rr["rmax"] = np.nan

    # Prepare mapping by slot (HH:MM)
    rr_map = df_rr.set_index("Slot") if "Slot" in df_rr.columns else pd.DataFrame().set_index(pd.Index([]))

    # Optional depolarization profiles (Track 2) keyed by HH:MM slot
    r_delta = None
    depol_map: Dict[str, np.ndarray] = {}
    if str(args.depol_file).strip():
        try:
            r_delta, depol_map = load_depol_profiles(Path(args.depol_file), args.depol_sheet)
            print(f"[depol] loaded {len(depol_map)} delta_v profile(s) from {Path(args.depol_file).name}")
        except Exception as e:
            print(f"[WARN] could not load depol file: {e}")

    # Optional SNR profiles (signal-quality cap) — read from the SAME NRB workbook,
    # which carries an "SNR" sheet on the same range grid / slots as the NRB.
    r_snr = None
    snr_map: Dict[str, np.ndarray] = {}
    if args.snr_cap:
        try:
            r_snr, snr_map = load_snr_profiles(Path(args.nrb), args.snr_sheet)
            print(f"[snr] loaded {len(snr_map)} SNR profile(s) from sheet '{args.snr_sheet}'")
        except Exception as e:
            print(f"[WARN] could not load SNR sheet '{args.snr_sheet}': {e} — snr_cap disabled")
            snr_map = {}

    # 1) Input_NRB_raw
    df_raw = pd.DataFrame(mat_raw, columns=profile_times)
    df_raw.insert(0, "Range(m)", r_m)

    # 2) Denoised (compute column-wise; robust to trailing/internal NaN)
    dr_global = float(np.nanmedian(np.diff(np.sort(r_m))))
    den = np.full_like(mat_raw, np.nan, dtype=float)
    for j in range(mat_raw.shape[1]):
        den[:, j], _prep_den = denoise_profile_preserve_shape(
            r_m=r_m,
            y_raw=mat_raw[:, j],
            fc=args.fc,
            order=args.order,
            pad_frac=args.pad_frac,
        )
    df_den = pd.DataFrame(den, columns=profile_times)
    df_den.insert(0, "Range(m)", r_m)

    results_rows = []
    cloud_rows = []  # per-cloud-layer rows for Cloud_results sheet
    for j, t_profile in enumerate(profile_times):
        t_profile = pd.Timestamp(t_profile)
        slot = t_profile.strftime("%H:%M")

        ycol = mat_raw[:, j]
        nrb_ok, valid_frac, valid_bins = _profile_validity(ycol, args.min_valid_frac, args.min_valid_bins)

        time_mapping = "slot_exact"
        rr_time = pd.NaT
        alt_mpl = np.nan
        rmin = np.nan
        rmax = np.nan

        if slot in rr_map.index:
            rr_row = rr_map.loc[slot]
            rr_time = rr_row.get("Time", pd.NaT)
            alt_mpl = float(rr_row.get("PBL from MPL (m)", np.nan)) if pd.notna(rr_row.get("PBL from MPL (m)", np.nan)) else np.nan
            rmin = float(rr_row.get("rmin", np.nan)) if pd.notna(rr_row.get("rmin", np.nan)) else np.nan
            rmax = float(rr_row.get("rmax", np.nan)) if pd.notna(rr_row.get("rmax", np.nan)) else np.nan
        else:
            time_mapping = "missing_rminrmax_slot"

        if not nrb_ok:
            guided_res = _empty_profile_result("NRB_incomplete", r_m, ycol, dr_global)
            profile_res = _empty_profile_result("NRB_incomplete", r_m, ycol, dr_global)
        else:
            if _valid_window(rmin, rmax):
                guided_res = compute_pbl_for_profile(
                    r_m=r_m,
                    y_raw=ycol,
                    rmin=rmin,
                    rmax=rmax,
                    fc=args.fc,
                    order=args.order,
                    pad_frac=args.pad_frac,
                    smooth_win=args.smooth_win,
                    tol_m=args.tol_m,
                )
            else:
                reason = "rminrmax_missing" if slot not in rr_map.index else "rminrmax_invalid"
                guided_res = _empty_profile_result(reason, r_m, ycol, dr_global)

            # ── Track A: derive effective nrb_profile window ──────────────────
            # Start with the user-supplied window then apply (in order):
            #   1. Adaptive diurnal window (overrides base if enabled)
            #   2. Peak-anchored narrowing (refines around NRB peak)
            #   3. Cloud screening (marks profile as cloudy and skips HWCT)
            prof_rmin = float(args.profile_rmin)
            prof_rmax = float(args.profile_rmax)
            window_source = "fixed_user"

            if args.adaptive_window:
                ad_rmin, ad_rmax = adaptive_window_for_hour(int(t_profile.hour))
                prof_rmin, prof_rmax = ad_rmin, ad_rmax
                window_source = f"adaptive_h{int(t_profile.hour):02d}"

            if args.peak_anchored and _valid_window(prof_rmin, prof_rmax):
                pa_rmin, pa_rmax = peak_anchored_window(r_m, ycol, prof_rmin, prof_rmax)
                prof_rmin, prof_rmax = pa_rmin, pa_rmax
                window_source = window_source + "+peak_anchor"

            # ── NRB-magnitude cloud cap (primary cloud guard) — cap search below
            # ── Signal-quality: cap the search at the SNR-trusted range top ────
            #    Above it the NRB is noise x R^2, so any "layer" there is spurious.
            #    Runs FIRST (most fundamental — no valid data above the trusted top).
            snr_trusted = np.nan
            if snr_map and r_snr is not None and slot in snr_map:
                snr_prof = np.interp(r_m, r_snr, snr_map[slot], left=np.nan, right=np.nan)
                snr_trusted = snr_trusted_top_m(r_m, snr_prof, float(args.snr_cap_min))
                if (np.isfinite(snr_trusted) and _valid_window(prof_rmin, prof_rmax)
                        and snr_trusted > prof_rmin):
                    prof_rmax = min(prof_rmax, snr_trusted)
                    window_source = window_source + "+snr_cap"

            #    the lowest ELEVATED cloud (water OR ice). Runs before the depol
            #    ice-screen because rainy-season clouds are low-δ water clouds the
            #    δ screen misses; NRB magnitude catches both.
            if args.nrb_cloud_cap and _valid_window(prof_rmin, prof_rmax):
                cbase = nrb_cloud_cap_base(
                    r_m, ycol, prof_rmin, prof_rmax,
                    float(args.cloud_threshold),
                    surface_gap_m=float(args.nrb_cloud_surface_gap_m))
                if cbase is not None and cbase > prof_rmin:
                    prof_rmax = min(prof_rmax, cbase)
                    window_source = window_source + "+nrb_cloudcap"

            # ── Track 2: depol-based cloud screen — cap search below ice cloud ─
            delta_prof = None
            if depol_map and r_delta is not None and slot in depol_map:
                delta_prof = np.interp(r_m, r_delta, depol_map[slot],
                                       left=np.nan, right=np.nan)
                if args.depol_cloud_screen and _valid_window(prof_rmin, prof_rmax):
                    base = depol_ice_cloud_base(
                        r_m, delta_prof, prof_rmin, prof_rmax, args.depol_ice_thr)
                    if base is not None and base > prof_rmin:
                        prof_rmax = min(prof_rmax, base)
                        window_source = window_source + "+depol_cloudcap"

            cloud_flag = detect_cloud_in_window(
                r_m, ycol, prof_rmin, prof_rmax, args.cloud_screen_threshold,
            )

            if cloud_flag:
                profile_res = _empty_profile_result("cloud_screened", r_m, ycol, dr_global)
                profile_res["Analysis_status"] = "cloud_screened"
            elif _valid_window(prof_rmin, prof_rmax):
                profile_res = compute_pbl_for_profile(
                    r_m=r_m,
                    y_raw=ycol,
                    rmin=prof_rmin,
                    rmax=prof_rmax,
                    fc=args.fc,
                    order=args.order,
                    pad_frac=args.pad_frac,
                    smooth_win=args.smooth_win,
                    tol_m=args.tol_m,
                )

                # ── Track A: lowest-stable edge re-selection (optional) ──
                if args.lowest_edge and profile_res.get("Analysis_status") == "ok":
                    try:
                        # Re-run HWCT to get W on the analysis grid (mirrors
                        # compute_pbl_for_profile so result is consistent)
                        prep = prepare_profile_for_fft(r_m, ycol)
                        if prep["status"] == "ok" and int(prep["analysis_bins_used"]) >= 8:
                            r_use = np.asarray(prep["r_use"], float)
                            y_use = np.asarray(prep["y_use"], float)
                            dr_use = float(np.nanmedian(np.diff(r_use)))
                            y_dn = fft_lowpass_fixed_fc(
                                y_use, dr=dr_use, fc=args.fc,
                                order=args.order, pad_frac=args.pad_frac,
                            )
                            half_bins = int(max(1, round(float(args.tol_m) / dr_use)))
                            W = hwct_haar_step_right_minus_left(y_dn, half_bins)
                            r_alt_new, w_new, mode_new = select_lowest_stable_edge(
                                W, r_use, prof_rmin, prof_rmax,
                                threshold_factor=float(args.lowest_edge_thr_factor),
                            )
                            if np.isfinite(r_alt_new):
                                profile_res["PBL_TR40_m"] = r_alt_new
                                profile_res["HWCT_peak_chosen"] = w_new
                                profile_res["Chosen_mode"] = mode_new
                    except Exception:
                        pass  # keep original result if anything goes wrong
            else:
                profile_res = _empty_profile_result("profile_window_invalid", r_m, ycol, dr_global)

            # Annotate which window was used for this profile (for QC sheet)
            profile_res["window_source"] = window_source
            profile_res["profile_rmin_used_m"] = float(prof_rmin)
            profile_res["profile_rmax_used_m"] = float(prof_rmax)
            profile_res["snr_trusted_range_m"] = float(snr_trusted)
            profile_res["cloud_screened"] = bool(cloud_flag)

            # ── Track 2: independent depol ALT + agreement flag ──────────────
            if (args.depol_confirm and delta_prof is not None
                    and _valid_window(prof_rmin, prof_rmax)):
                try:
                    a_dep = compute_depol_alt(
                        r_m, delta_prof, prof_rmin, prof_rmax,
                        args.fc, args.order, args.pad_frac, args.tol_m)
                    profile_res["ALT_depol_m"] = a_dep
                    nrb_alt = profile_res.get("PBL_TR40_m", np.nan)
                    if np.isfinite(a_dep) and np.isfinite(nrb_alt):
                        profile_res["depol_confirm"] = (
                            "agree" if abs(a_dep - nrb_alt) <= args.depol_confirm_tol_m
                            else "disagree")
                    else:
                        profile_res["depol_confirm"] = "no_depol_edge"
                except Exception:
                    profile_res["ALT_depol_m"] = np.nan
                    profile_res["depol_confirm"] = "error"

        guided_alt = guided_res["PBL_TR40_m"]
        profile_alt = profile_res["PBL_TR40_m"]

        if args.detection_mode == "mpl_guided":
            selected_label = "mpl_guided"
            selected_res = guided_res
        elif args.detection_mode == "nrb_profile":
            selected_label = "nrb_profile"
            selected_res = profile_res
        else:
            if np.isfinite(guided_alt):
                selected_label = "dual_selected_mpl_guided"
                selected_res = guided_res
            else:
                selected_label = "dual_selected_nrb_profile"
                selected_res = profile_res

        alt_tr40 = selected_res["PBL_TR40_m"]

        # ── δ-at-ALT QC flag: classify the SELECTED (reported) ALT edge as
        #    cloud / aerosol so a (low-δ) water cloud or (high-δ) ice cloud
        #    mistaken for the layer top is visible. Computed at alt_tr40 (the
        #    value actually reported) so the flag always matches ALT_TR40_m.
        #    Needs a depol file; otherwise left blank.
        if delta_prof is not None and np.isfinite(alt_tr40):
            d_at = float(np.interp(alt_tr40, r_m, delta_prof,
                                   left=np.nan, right=np.nan))
            profile_res["delta_at_ALT"] = d_at
            if not np.isfinite(d_at):
                profile_res["ALT_feature"] = "unknown"
            elif d_at >= 0.35:
                profile_res["ALT_feature"] = "cloud_ice"
            elif d_at >= 0.10:
                profile_res["ALT_feature"] = "dust_smoke"
            else:
                profile_res["ALT_feature"] = "aerosol"
        status = "ok" if np.isfinite(alt_tr40) else selected_res["Analysis_status"].replace("skipped:", "skip_")
        delta_selected = (alt_tr40 - alt_mpl) if (np.isfinite(alt_tr40) and np.isfinite(alt_mpl)) else np.nan
        delta_guided = (guided_alt - alt_mpl) if (np.isfinite(guided_alt) and np.isfinite(alt_mpl)) else np.nan
        delta_profile = (profile_alt - alt_mpl) if (np.isfinite(profile_alt) and np.isfinite(alt_mpl)) else np.nan

        results_rows.append(
            {
                "Time": t_profile,
                "Slot": slot,
                "Time_mapping": time_mapping,
                "Detection_mode": args.detection_mode,
                "ALT_selected_source": selected_label,
                "Status": status,
                "NRB_valid_frac": valid_frac,
                "NRB_valid_bins": valid_bins,

                # New ALT terminology
                "ALT_MPL_m": alt_mpl,
                "ALT_TR40_m": alt_tr40,
                "ALT_guided_m": guided_alt,
                "ALT_guided_pos_m": guided_res["PBL_pos_m"],
                "ALT_guided_neg_m": guided_res["PBL_neg_m"],
                "ALT_guided_chosen_mode": guided_res["Chosen_mode"],
                "ALT_guided_status": guided_res["Analysis_status"],
                "ALT_profile_m": profile_alt,
                "ALT_profile_pos_m": profile_res["PBL_pos_m"],
                "ALT_profile_neg_m": profile_res["PBL_neg_m"],
                "ALT_profile_chosen_mode": profile_res["Chosen_mode"],
                "ALT_profile_status": profile_res["Analysis_status"],
                "ALT_depol_m": profile_res.get("ALT_depol_m", np.nan),
                "depol_confirm": profile_res.get("depol_confirm", "off"),
                "delta_at_ALT": profile_res.get("delta_at_ALT", np.nan),
                "ALT_feature": profile_res.get("ALT_feature", ""),
                "Delta_ALT_selected_minus_MPL_m": delta_selected,
                "Delta_ALT_guided_minus_MPL_m": delta_guided,
                "Delta_ALT_profile_minus_MPL_m": delta_profile,

                # Search windows
                "rmin_m": rmin,
                "rmax_m": rmax,
                "profile_rmin_m": float(args.profile_rmin),
                "profile_rmax_m": float(args.profile_rmax),
                "rminrmax_Time": rr_time,

                # Backward-compatible aliases for older plotting code / old reports
                "PBL_MPL_m": alt_mpl,
                "PBL_TR40_m": alt_tr40,
                "Chosen_mode": selected_res["Chosen_mode"],
                "HWCT_peak_chosen": selected_res["HWCT_peak_chosen"],
                "PBL_pos_m": selected_res["PBL_pos_m"],
                "HWCT_peak_pos": selected_res["HWCT_peak_pos"],
                "PBL_neg_m": selected_res["PBL_neg_m"],
                "HWCT_peak_neg": selected_res["HWCT_peak_neg"],
                "Delta_TR40_minus_MPL_m": delta_selected,

                "NRB_all_nan": bool(selected_res["NRB_all_nan"]),
                "Analysis_status": selected_res["Analysis_status"],
                "Analysis_bins_used": selected_res["Analysis_bins_used"],
                "Analysis_range_max_m": selected_res["Analysis_range_max_m"],
                "Trim_leading_bins": selected_res["Trim_leading_bins"],
                "Trim_trailing_bins": selected_res["Trim_trailing_bins"],
                "Internal_nan_filled": selected_res["Internal_nan_filled"],

                # ── Track A annotations ───────────────────────────────────
                "window_source": profile_res.get("window_source", "fixed_user"),
                "profile_rmin_used_m": profile_res.get("profile_rmin_used_m", float(args.profile_rmin)),
                "profile_rmax_used_m": profile_res.get("profile_rmax_used_m", float(args.profile_rmax)),
                "cloud_screened": bool(profile_res.get("cloud_screened", False)),
            }
        )

        # ── Track 3: Cloud detection per profile ───────────────────────────
        if args.cloud_detect and nrb_ok:
            try:
                layers = detect_cloud_layers(
                    r_m, ycol,
                    threshold=float(args.cloud_threshold),
                    min_thickness_m=float(args.cloud_min_thickness_m),
                    max_layers=int(args.cloud_max_layers),
                    r_min_m=float(args.cloud_search_rmin),
                    r_max_m=float(args.cloud_search_rmax),
                )
            except Exception:
                layers = []
            # Annotate ALT row with cloud summary
            results_rows[-1]["cloud_count"] = int(len(layers))
            for li, layer in enumerate(layers):
                k = li + 1
                results_rows[-1][f"cloud_{k}_base_m"] = layer["base_m"]
                results_rows[-1][f"cloud_{k}_top_m"] = layer["top_m"]
                results_rows[-1][f"cloud_{k}_peak_m"] = layer["peak_R_m"]
                results_rows[-1][f"cloud_{k}_peak_nrb"] = layer["peak_value"]
                # Detailed row for Cloud_results sheet
                cloud_rows.append({
                    "Time": t_profile,
                    "Slot": slot,
                    "layer_index": k,
                    "base_m": layer["base_m"],
                    "top_m": layer["top_m"],
                    "peak_R_m": layer["peak_R_m"],
                    "peak_NRB": layer["peak_value"],
                    "thickness_m": layer["thickness_m"],
                    "rough_OD": layer["rough_OD"],
                })

    df_alt = pd.DataFrame(results_rows)
    df_clouds = pd.DataFrame(cloud_rows) if cloud_rows else pd.DataFrame(
        columns=["Time", "Slot", "layer_index", "base_m", "top_m",
                 "peak_R_m", "peak_NRB", "thickness_m", "rough_OD"]
    )

    # ── Track A: temporal smoothing on the SELECTED ALT time series ────────
    if int(args.temporal_smooth_window) > 1 and not df_alt.empty:
        df_alt = df_alt.sort_values("Time").reset_index(drop=True)
        smoothed = temporal_smooth_alt(
            df_alt["ALT_TR40_m"],
            window=int(args.temporal_smooth_window),
            outlier_threshold_m=float(args.temporal_smooth_outlier_m),
        )
        df_alt["ALT_TR40_m_raw"] = df_alt["ALT_TR40_m"]
        df_alt["ALT_TR40_m"] = smoothed
        # Recompute delta vs MPL
        df_alt["Delta_ALT_selected_minus_MPL_m"] = (
            pd.to_numeric(df_alt["ALT_TR40_m"], errors="coerce")
            - pd.to_numeric(df_alt["ALT_MPL_m"], errors="coerce")
        )
        df_alt["Delta_TR40_minus_MPL_m"] = df_alt["Delta_ALT_selected_minus_MPL_m"]
        df_alt["PBL_TR40_m"] = df_alt["ALT_TR40_m"]

    # Summary
    summary = []
    for mode, g in df_alt.groupby("ALT_selected_source"):
        summary.append(
            {
                "ALT_selected_source": mode,
                "count": int(len(g)),
                "ALT_TR40_mean": float(pd.to_numeric(g["ALT_TR40_m"], errors="coerce").mean()),
                "ALT_TR40_std": float(pd.to_numeric(g["ALT_TR40_m"], errors="coerce").std(ddof=1)),
                "Delta_selected_mean": float(pd.to_numeric(g["Delta_ALT_selected_minus_MPL_m"], errors="coerce").mean()),
                "Delta_selected_std": float(pd.to_numeric(g["Delta_ALT_selected_minus_MPL_m"], errors="coerce").std(ddof=1)),
                "ALT_guided_mean": float(pd.to_numeric(g["ALT_guided_m"], errors="coerce").mean()),
                "ALT_profile_mean": float(pd.to_numeric(g["ALT_profile_m"], errors="coerce").mean()),
            }
        )
    df_summary = pd.DataFrame(summary)

    df_status = df_alt["Status"].value_counts(dropna=False).reset_index()
    df_status.columns = ["Status", "count"]

    # Parameters
    params = pd.DataFrame(
        [{
            "Internal_engine_version": PBL_VERSION,
            "Terminology": "ALT = Aerosol Layer Top; old PBL columns are kept only as backward-compatible aliases",
            "profiles": int(mat_raw.shape[1]),
            "dr_m_per_bin": float(dr_global),
            "Detection_mode": args.detection_mode,
            "MPL_guided_window": "Uses rmin/rmax from rmin-rmax sheet",
            "NRB_profile_window_m": f"{float(args.profile_rmin)} to {float(args.profile_rmax)}",
            "FFT_fc_cycles_per_m": float(args.fc),
            "FFT_fc_equiv_wavelength_m": float(1.0 / float(args.fc)) if args.fc != 0 else np.nan,
            "FFT_order": int(args.order),
            "FFT_pad_frac": float(args.pad_frac),
            "HWCT_smooth_win_bins": "disabled (FFT-only smoothing, per Brooks 2003)",
            "HWCT_tol_m": float(args.tol_m),
            "MIN_VALID_FRAC": float(args.min_valid_frac),
            "MIN_VALID_BINS": int(args.min_valid_bins),
            "Time_join": "Slot(HH:MM) exact in engine; GUI pre-aligns nearest time when requested",

            # Track A flags (algorithm improvements for nrb_profile mode)
            "TrackA_adaptive_window": bool(args.adaptive_window),
            "TrackA_peak_anchored": bool(args.peak_anchored),
            "TrackA_cloud_screen_threshold": float(args.cloud_screen_threshold),
            "TrackA_lowest_edge": bool(args.lowest_edge),
            "TrackA_lowest_edge_thr_factor": float(args.lowest_edge_thr_factor),
            "TrackA_temporal_smooth_window": int(args.temporal_smooth_window),
            "TrackA_temporal_smooth_outlier_m": float(args.temporal_smooth_outlier_m),
            "TrackA_adaptive_reference": (
                "Chiang Mai / NARIT — Janta 2020, Pani 2019, Lertusee 2018 typical CBL heights"
            ),

            # Track 3 — Cloud detection
            "Cloud_detect_enabled": bool(args.cloud_detect),
            "Cloud_threshold_NRB": float(args.cloud_threshold),
            "Cloud_min_thickness_m": float(args.cloud_min_thickness_m),
            "Cloud_max_layers": int(args.cloud_max_layers),
            "Cloud_search_rmin_m": float(args.cloud_search_rmin),
            "Cloud_search_rmax_m": float(args.cloud_search_rmax),
        }]
    )

    # Write workbook.
    # Keep PBL_results for compatibility with the existing visualizer, and add ALT_results as the correct name.
    out_path = Path(args.out)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        df_rr.to_excel(xw, index=False, sheet_name="Input_rmin-rmax")
        df_raw.to_excel(xw, index=False, sheet_name="Input_NRB_raw")
        df_den.to_excel(xw, index=False, sheet_name="NRB_Denoised_FFT")
        df_alt.to_excel(xw, index=False, sheet_name="ALT_results")
        df_alt.to_excel(xw, index=False, sheet_name="PBL_results")
        df_summary.to_excel(xw, index=False, sheet_name="Summary")
        df_status.to_excel(xw, index=False, sheet_name="Status_counts")
        if args.cloud_detect:
            df_clouds.to_excel(xw, index=False, sheet_name="Cloud_results")
        params.to_excel(xw, index=False, sheet_name="Parameters")

    print(f"[OK] Saved: {out_path.resolve()}")
    print(df_status.to_string(index=False))


if __name__ == "__main__":
    main()
