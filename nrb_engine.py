from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------

def cosine_taper_weight(r: np.ndarray, r1: float, r2: float) -> np.ndarray:
    r = np.asarray(r, float)
    w = np.zeros_like(r, dtype=float)
    w[r >= r2] = 1.0
    mid = (r > r1) & (r < r2)
    if r2 <= r1:
        w[r >= r1] = 1.0
        return w
    w[mid] = 0.5 * (1.0 - np.cos(np.pi * (r[mid] - r1) / (r2 - r1)))
    return w


def dead_time_correct_mhz(rate_mhz: np.ndarray, dead_time_ns: float) -> np.ndarray:
    rate_mhz = np.asarray(rate_mhz, float)
    denom = 1.0 - rate_mhz * float(dead_time_ns) * 1e-3
    denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
    return rate_mhz / denom


def linear_regression_stats(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float, int]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    if n < 2:
        raise ValueError("Not enough points for linear regression.")
    A = np.vstack([x, np.ones_like(x)]).T
    slope, offset = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = slope * x + offset
    ss_res = float(np.nansum((y - yhat) ** 2))
    ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
    if ss_tot <= 0:
        r2 = 1.0 if ss_res <= 1e-12 else 0.0
    else:
        r2 = max(0.0, 1.0 - ss_res / ss_tot)
    rmse = float(np.sqrt(ss_res / max(n, 1)))
    return float(slope), float(offset), float(r2), float(rmse), n


def bin_width_ns_from_dr(dr_m: float) -> float:
    c = 299792458.0
    ns = (2.0 * float(dr_m) / c) * 1e9
    if abs(ns - 25.0) < 0.25:
        return 25.0
    return float(ns)


# -----------------------------------------------------------------------------
# Rayleigh-fit background (aerosol-free molecular fit)
# -----------------------------------------------------------------------------
# In a clean molecular region the measured signal is S(r) = a*M(r) + b, where
# M(r) = beta_mol(r)*T^2(r)/r^2 is the molecular signal shape and b the constant
# background pedestal. A linear fit S vs M gives the calibration (slope a) AND
# the background (intercept b), separating the range-dependent signal from the
# flat pedestal — more robust than a plain window mean when the far window still
# holds a little signal or the pedestal is uncertain (daytime). A significance
# gate (|b|/sigma_b > sigma_gate) avoids over-subtracting a clean profile.
# Reference: RMlicelMatlab rayleigh_fit.m (Barbosa); molecular model from
# fernald_engine (US Standard Atmosphere, Rayleigh 532 nm).

def molecular_signal_shape(
    r_m: np.ndarray,
    T_K: Optional[np.ndarray] = None,
    P_Pa: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Molecular (Rayleigh) lidar signal SHAPE M(r) = beta_mol(r)*T^2(r)/r^2.
    Absolute scale is arbitrary (absorbed by the fit slope); only the r-shape
    matters. Uses the US Standard Atmosphere unless T_K/P_Pa are supplied."""
    from fernald_engine import molecular_backscatter_532, molecular_extinction

    r = np.asarray(r_m, float)
    beta = molecular_backscatter_532(r, T_K, P_Pa)
    alpha = molecular_extinction(beta)
    tau = np.zeros_like(r)
    if r.size >= 2:
        dr = np.diff(r)
        incr = 0.5 * (alpha[1:] + alpha[:-1]) * dr
        tau[1:] = np.cumsum(incr)
    T2 = np.exp(-2.0 * tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        M = np.where(r > 0, beta * T2 / (r ** 2), np.nan)
    return M


def rayleigh_fit_background(
    r_m: np.ndarray,
    signal: np.ndarray,
    *,
    fit_rmin_m: float,
    fit_rmax_m: float,
    sigma_gate: float = 3.0,
    T_K: Optional[np.ndarray] = None,
    P_Pa: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Estimate the constant background as the intercept b of a linear fit
    ``signal = a*M(r) + b`` over an aerosol-free window [fit_rmin, fit_rmax].

    A significance gate returns background = 0 when b is statistically
    compatible with zero (|b|/sigma_b <= sigma_gate), so a clean profile is not
    over-subtracted (matches RMlicelMatlab). Returns a dict with keys:
    background, slope, b_raw, b_sigma, r2, n, significant."""
    r = np.asarray(r_m, float)
    y = np.asarray(signal, float)
    M = molecular_signal_shape(r, T_K, P_Pa)
    win = ((r >= float(fit_rmin_m)) & (r <= float(fit_rmax_m))
           & np.isfinite(y) & np.isfinite(M))
    n = int(np.sum(win))
    if n < 3:
        raise ValueError(
            f"rayleigh_fit: only {n} valid bins in [{fit_rmin_m:.0f}, "
            f"{fit_rmax_m:.0f}] m (need >= 3). Widen the fit window."
        )
    x = M[win]
    yy = y[win]
    X = np.vstack([x, np.ones_like(x)]).T
    coef = np.linalg.lstsq(X, yy, rcond=None)[0]
    a, b = float(coef[0]), float(coef[1])
    resid = yy - (a * x + b)
    dof = max(1, n - 2)
    s2 = float(np.sum(resid ** 2) / dof)
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        b_sigma = float(np.sqrt(max(s2 * XtX_inv[1, 1], 0.0)))
    except np.linalg.LinAlgError:
        b_sigma = float("nan")
    ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
    r2 = float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    significant = bool(np.isfinite(b_sigma) and b_sigma > 0 and abs(b) / b_sigma > float(sigma_gate))
    background = b if significant else 0.0
    return {"background": float(background), "slope": a, "b_raw": b,
            "b_sigma": b_sigma, "r2": r2, "n": n, "significant": significant}


# -----------------------------------------------------------------------------
# .dat reading helpers
# -----------------------------------------------------------------------------

def find_dat_data_start(lines: List[str]) -> int:
    for i, ln in enumerate(lines):
        if ("Analog" in ln) and ("Photon" in ln or "Photon Counting" in ln):
            return i + 1
    return 9


def _looks_like_licel_binary(path: Path) -> bool:
    """Heuristic format detector: a Licel raw binary file has a non-text binary
    body (NUL bytes / many non-printable bytes), whereas an Advanced-Viewer ASCII
    ``.dat`` export is entirely text. Both share the same ASCII header lines, so
    we sniff the file body rather than the header."""
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(65536)
    except OSError:
        return False
    if not chunk:
        return False
    if b"\x00" in chunk:            # raw uint32 counts contain NUL high-bytes; text never does
        return True
    text = sum(1 for b in chunk if b in (9, 10, 13) or 32 <= b <= 126)
    return (text / len(chunk)) < 0.90


# Licel raw filename: <prefix><yy><Mhex><dd><HH>.<MM><SS><ms>, e.g. a2682711.150166
# = 2026-08(hex 8)-27 11:15:01. Month is a single hex digit (Oct-Dec = A/B/C).
_LICEL_RAW_NAME_RE = re.compile(r"^[A-Za-z]{1,2}(\d{2})([0-9A-Ca-c])(\d{2})(\d{2})\.(\d{2})(\d{2})(\d{2,})$")


def parse_licel_raw_timestamp(name: str) -> "Optional[pd.Timestamp]":
    """Acquisition timestamp encoded in a Licel raw filename, or None if the name
    is not in that format (e.g. an ASCII ``.dat`` export)."""
    m = _LICEL_RAW_NAME_RE.match(name)
    if not m:
        return None
    yy, mhex, dd, hh, mm, ss, _ms = m.groups()
    try:
        return pd.Timestamp(year=2000 + int(yy), month=int(mhex, 16), day=int(dd),
                            hour=int(hh), minute=int(mm), second=int(ss))
    except ValueError:
        return None


def glob_lidar_files(folder: Path, pattern: str, *, recursive: bool = False) -> List[Path]:
    """Files matching ``pattern``; when the pattern matches nothing (e.g. the
    default ``*.dat`` against a folder of extensionless raw files) fall back to
    every file that sniffs as a Licel raw binary. ``.dat`` folders are unchanged."""
    folder = Path(folder)
    matched = sorted(folder.rglob(pattern) if recursive else folder.glob(pattern))
    if matched:
        return matched
    it = folder.rglob("*") if recursive else folder.iterdir()
    return [f for f in sorted(it) if f.is_file() and _looks_like_licel_binary(f)]


def _read_licel_binary_array(path: Path) -> np.ndarray:
    """Decode a Licel raw binary file to the same column layout an ASCII export
    would yield: cols 0-3 parallel, 4-7 the perpendicular channel, col 8 the
    per-bin overflow flag (binary files carry overflow; ASCII exports drop it)."""
    import warnings
    import licel_binary_reader as _lbr  # local import: no circular dependency

    rf = _lbr.parse_raw_file(path)
    if rf.shots == 0:
        warnings.warn(
            f"{Path(path).name}: raw file reports shots=0 (empty/aborted "
            f"acquisition); decoded values are not physically meaningful.",
            RuntimeWarning, stacklevel=3,
        )
    return _lbr.raw_file_to_array(rf)


def _parse_ascii_dat_array(path: Path, *, start_mode: str = "auto") -> np.ndarray:
    lines = path.read_text(errors="replace").splitlines()
    start = find_dat_data_start(lines) if start_mode == "auto" else 9

    rows = []
    ncol = None
    for ln in lines[start:]:
        ln = ln.strip()
        if not ln:
            continue
        parts = [p for p in re.split(r"[\t ]+", ln) if p != ""]
        if len(parts) < 4:
            continue
        try:
            vals = [float(p) for p in parts]
        except Exception:
            continue
        if ncol is None:
            # Parallel is columns 0-3 (analog, analog_stderr, photon, photon_stderr).
            # A 2-PMT file adds the perpendicular channel at columns 4-7 (+ an
            # overflow column); a single-PMT file has only 4 (+ optional overflow).
            ncol = 8 if len(vals) >= 8 else 4
        if len(vals) < ncol:
            continue
        rows.append(vals[:ncol])

    if not rows:
        raise ValueError(f"No numeric data rows found in {path.name}.")
    return np.asarray(rows, dtype=float)


def _read_tr40_dat_ascii_array(
    path: Path,
    *,
    start_mode: str = "auto",
    trim_trailing_zeros: bool = True,
) -> np.ndarray:
    """Read a TR40 profile into a numeric array, auto-detecting the file format.

    Accepts both a **Licel raw binary file** (the native file the Advanced Viewer
    opens — decoded directly, no export step) and a legacy **ASCII ``.dat``
    export**. Columns: 0-3 parallel (analog, analog_stderr, photon,
    photon_stderr), 4-7 the perpendicular channel (2-PMT files); binary files add
    col 8 = per-bin overflow flag.
    """
    path = Path(path)
    if _looks_like_licel_binary(path):
        arr = _read_licel_binary_array(path)
    else:
        arr = _parse_ascii_dat_array(path, start_mode=start_mode)

    if trim_trailing_zeros:
        nonzero_row = np.any(np.abs(arr[:, :4]) > 0, axis=1)
        if not np.any(nonzero_row):
            raise ValueError(f"All data rows are zero in {path.name}")
        arr = arr[: int(np.where(nonzero_row)[0][-1]) + 1, :]
    return arr


def _channel_col_offset(arr: np.ndarray, channel: str, name: str) -> int:
    """Column offset of the requested polarization channel in a .dat data array.
    Parallel = columns 0-3, perpendicular = columns 4-7 (2-PMT files only).
    A single-PMT file has only 4 signal columns → only 'parallel' is valid."""
    ch = str(channel).strip().lower()
    n_cols = int(arr.shape[1])
    if ch in ("perpendicular", "perp", "cross", "s", "l"):
        if n_cols < 8:
            raise ValueError(
                f"{name} has no perpendicular channel (a 2-PMT file has 8 signal "
                f"columns; found {n_cols}). Use channel='parallel'.")
        return 4
    if ch in ("parallel", "par", "co", "p", ""):
        return 0
    raise ValueError(f"Unknown channel {channel!r}; use 'parallel' or 'perpendicular'.")


def read_tr40_dat_ascii(
    path: Path,
    dr_m: float = 3.75,
    start_mode: str = "auto",
    trim_trailing_zeros: bool = True,
    pretrigger_bins: int = 0,
    first_signal_bin: Optional[int] = None,
    first_signal_range_m: float = 3.75,
    channel: str = "parallel",
) -> pd.DataFrame:
    arr = _read_tr40_dat_ascii_array(path, start_mode=start_mode, trim_trailing_zeros=trim_trailing_zeros)
    c0 = _channel_col_offset(arr, channel, path.name)
    total_bins = int(arr.shape[0])
    pretrigger_bins = int(pretrigger_bins)
    if pretrigger_bins < 0:
        raise ValueError("pretrigger_bins must be >= 0")
    if pretrigger_bins > total_bins:
        raise ValueError(
            f"pretrigger_bins={pretrigger_bins} is too large for {path.name} (available bins={total_bins})."
        )

    if first_signal_bin is None:
        first_signal_bin = pretrigger_bins + 1
    first_signal_bin = int(first_signal_bin)
    if first_signal_bin < 1:
        raise ValueError("first_signal_bin must be >= 1")
    if first_signal_bin > total_bins:
        raise ValueError(
            f"first_signal_bin={first_signal_bin} is too large for {path.name} (available bins={total_bins})."
        )
    if first_signal_bin <= pretrigger_bins:
        raise ValueError("first_signal_bin must be greater than pretrigger_bins.")

    arr_sig = arr[first_signal_bin - 1 :, :]
    n = arr_sig.shape[0]
    source_bin_index = np.arange(first_signal_bin, first_signal_bin + n)
    bin_index = np.arange(1, n + 1)
    range_m = float(first_signal_range_m) + np.arange(n, dtype=float) * float(dr_m)

    df = pd.DataFrame(
        {
            "bin_index": bin_index,
            "source_bin_index": source_bin_index,
            "range_m": range_m,
            "analog_mV": arr_sig[:, c0 + 0],
            "analog_stderr_mV": arr_sig[:, c0 + 1],
            "photon_MHz": arr_sig[:, c0 + 2],
            "photon_stderr_MHz": arr_sig[:, c0 + 3],
        }
    )
    # Per-bin overflow/saturation flag (col 8) is present for raw binary files
    # (all channels share it); ASCII exports drop it, so default to 0.
    df["overflow"] = arr_sig[:, 8] if arr_sig.shape[1] > 8 else 0.0
    return df


def extract_pretrigger_background(
    path: Path,
    *,
    pretrigger_bins: int,
    start_mode: str = "auto",
    trim_trailing_zeros: bool = True,
    channel: str = "parallel",
) -> Tuple[np.ndarray, np.ndarray]:
    arr = _read_tr40_dat_ascii_array(path, start_mode=start_mode, trim_trailing_zeros=trim_trailing_zeros)
    c0 = _channel_col_offset(arr, channel, path.name)
    pretrigger_bins = int(pretrigger_bins)
    if pretrigger_bins <= 0:
        raise ValueError("pretrigger_bins must be > 0 for pretrigger background mode.")
    if pretrigger_bins > arr.shape[0]:
        raise ValueError(
            f"pretrigger_bins={pretrigger_bins} is too large for {path.name} (available bins={arr.shape[0]})."
        )
    return arr[:pretrigger_bins, c0 + 0].astype(float), arr[:pretrigger_bins, c0 + 2].astype(float)


# -----------------------------------------------------------------------------
# Reference-aligned glue helpers
# -----------------------------------------------------------------------------

def _interpolate_threshold_crossing_descending(
    r_m: np.ndarray,
    y: np.ndarray,
    threshold: float,
    start_idx: int,
) -> Tuple[float, Optional[int]]:
    r = np.asarray(r_m, float)
    y = np.asarray(y, float)
    n = int(min(r.size, y.size))
    if n < 2:
        return np.nan, None
    i0 = max(0, int(start_idx))
    for i in range(i0, n - 1):
        y0 = y[i]
        y1 = y[i + 1]
        if not (np.isfinite(y0) and np.isfinite(y1) and np.isfinite(r[i]) and np.isfinite(r[i + 1])):
            continue
        crosses = ((y0 >= threshold) and (y1 <= threshold)) or ((y0 > threshold) and (y1 < threshold))
        if not crosses:
            continue
        if abs(y1 - y0) <= 1e-12:
            return float(r[i + 1]), i
        t = (float(threshold) - float(y0)) / (float(y1) - float(y0))
        t = min(max(t, 0.0), 1.0)
        rc = float(r[i] + t * (r[i + 1] - r[i]))
        return rc, i
    return np.nan, None


def auto_find_blend_zone_from_threshold_crossings(
    r_m: np.ndarray,
    photon_dt_mhz: np.ndarray,
    *,
    sig_start_m: float,
    sig_end_m: float,
    min_toggle_rate: float,
    max_toggle_rate: float,
) -> Dict[str, float]:
    r = np.asarray(r_m, float)
    y = np.asarray(photon_dt_mhz, float)
    mask = (
        np.isfinite(r)
        & np.isfinite(y)
        & (r >= float(sig_start_m))
        & (r <= float(sig_end_m))
    )
    idx = np.where(mask)[0]
    if idx.size < 3:
        raise ValueError("Not enough valid points to find blend zone from threshold crossings.")

    ysig = y[idx]
    i_peak_local = int(np.nanargmax(ysig))
    i_peak = int(idx[i_peak_local])
    peak_rate = float(y[i_peak])
    peak_range = float(r[i_peak])

    r1, i1 = _interpolate_threshold_crossing_descending(r, y, float(max_toggle_rate), i_peak)
    if i1 is None:
        raise ValueError("No descending crossing found for max_toggle_rate.")
    r2, i2 = _interpolate_threshold_crossing_descending(r, y, float(min_toggle_rate), i1)
    if i2 is None:
        raise ValueError("No descending crossing found for min_toggle_rate.")
    if not np.isfinite(r1) or not np.isfinite(r2) or r2 <= r1:
        raise ValueError("Invalid threshold-based blend zone.")

    return {
        "blend_r1_m": float(r1),
        "blend_r2_m": float(r2),
        "peak_rate_mhz": float(peak_rate),
        "peak_range_m": float(peak_range),
        "cross_max_toggle_m": float(r1),
        "cross_min_toggle_m": float(r2),
        "cross_max_toggle_idx": float(i1),
        "cross_min_toggle_idx": float(i2),
    }


def trim_pretrigger_tail(arr: np.ndarray, trim_bins: int) -> np.ndarray:
    arr = np.asarray(arr, float)
    trim_bins = max(0, int(trim_bins))
    if trim_bins <= 0 or arr.size <= trim_bins:
        return arr.copy()
    return arr[:-trim_bins].copy()


def select_toggle_rates(
    bg_pretrigger_photon_dt_mhz: float,
    *,
    auto_toggle_selector: bool,
    night_min_toggle_rate: float,
    night_max_toggle_rate: float,
    day_min_toggle_rate: float,
    day_max_toggle_rate: float,
    bg_switch_threshold_mhz: float,
) -> Dict[str, float | str]:
    if not bool(auto_toggle_selector):
        return {
            "min_toggle_rate_selected": float(night_min_toggle_rate),
            "max_toggle_rate_selected": float(night_max_toggle_rate),
            "toggle_mode_text": "manual_night_values",
        }
    if np.isfinite(bg_pretrigger_photon_dt_mhz) and bg_pretrigger_photon_dt_mhz > float(bg_switch_threshold_mhz):
        return {
            "min_toggle_rate_selected": float(day_min_toggle_rate),
            "max_toggle_rate_selected": float(day_max_toggle_rate),
            "toggle_mode_text": "auto_daylight_high_background",
        }
    return {
        "min_toggle_rate_selected": float(night_min_toggle_rate),
        "max_toggle_rate_selected": float(night_max_toggle_rate),
        "toggle_mode_text": "auto_night_default",
    }


# -----------------------------------------------------------------------------
# Glue / NRB
# -----------------------------------------------------------------------------

def compute_nrb_reference_glue(
    r_m: np.ndarray,
    analog_raw: np.ndarray,
    photon_rate_mhz: np.ndarray,
    *,
    dead_time_ns: float,
    sig_start_m: float,
    sig_end_m: float,
    min_toggle_rate: float,
    max_toggle_rate: float,
    energy_mj: float,
    bg_pretrigger_photon_mhz: float,
    auto_blend: bool = True,
    blend_r1_m: float = 1200.0,
    blend_r2_m: float = 1800.0,
    blend_search_start_m: Optional[float] = None,
    blend_search_end_m: Optional[float] = None,
    blend_window_m: float = 300.0,
    blend_step_bins: int = 5,
    blend_r2_threshold: float = 0.985,
    blend_min_points: int = 25,
    blend_norm_rmse_max: float = 0.12,
    blend_cluster_min_m: Optional[float] = None,
    gluing_mode: str = "auto",
    day_glue_min_r2: float = 0.85,
    overlap_O_R: Optional[np.ndarray] = None,
    overlap_O_min: float = 0.1,
    afterpulse_A_R: Optional[np.ndarray] = None,
    saturation_mask: Optional[np.ndarray] = None,
    exclude_saturated: bool = False,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, np.ndarray]]:
    r = np.asarray(r_m, float)
    analog_raw = np.asarray(analog_raw, float)
    photon_rate_mhz = np.asarray(photon_rate_mhz, float)

    # Per-bin saturation flag from the raw file's overflow dataset (OF0): bins
    # where the analog channel clipped the ADC. Excluding them from the
    # analog<->photon slope fit keeps a saturated near-field bin from biasing the
    # glue scale. Only the FIT is affected; the blended profile is untouched.
    if exclude_saturated and saturation_mask is not None:
        sat = np.asarray(saturation_mask, bool)
        if sat.shape != r.shape:
            raise ValueError(
                f"saturation_mask shape {sat.shape} does not match range shape {r.shape}."
            )
    else:
        sat = np.zeros_like(r, dtype=bool)
    n_saturated_excluded = int(np.sum(sat))

    gluing_mode_l = str(gluing_mode).strip().lower()
    if gluing_mode_l not in ("auto", "photon_only", "auto_bgsub", "auto_bgsub_guarded"):
        raise ValueError(
            f"Unknown gluing_mode: {gluing_mode!r}. Use 'auto', 'photon_only', "
            "'auto_bgsub' or 'auto_bgsub_guarded'."
        )
    # BG-subtracted glue (daytime): a high solar background lifts the whole photon
    # profile above the toggle window so the raw-rate window finds no fit bins.
    # Selecting the fit region / blend zone on the SIGNAL above background
    # (photon_dt − BG) restores the glue where the analog is still linear (dawn),
    # matching Licel §9.7.5 ("background above min toggle → use scaled analog").
    _bgsub = gluing_mode_l in ("auto_bgsub", "auto_bgsub_guarded")

    # Shift intentionally disabled in this workflow.
    photon_shifted = photon_rate_mhz.copy()
    photon_dt_mhz = dead_time_correct_mhz(photon_shifted, dead_time_ns)
    # Toggle thresholds are compared against the DEAD-TIME-CORRECTED rate (matches
    # the lab's LabVIEW processing — verified). For the BG-subtracted daytime modes
    # the thresholds are applied to the signal above background instead.
    photon_toggle = (photon_dt_mhz - float(bg_pretrigger_photon_mhz)) if _bgsub else photon_dt_mhz

    sig_mask = (r >= float(sig_start_m)) & (r <= float(sig_end_m))

    analog_scaled = np.full_like(analog_raw, np.nan, dtype=float)
    slope = np.nan
    offset = np.nan
    fit_quality_r2 = np.nan
    fit_rmse = np.nan
    n_fit = 0
    fit_mode = "toggle_window"
    blend_r1_used = float(blend_r1_m)
    blend_r2_used = float(blend_r2_m)
    auto_ok = 0.0
    auto_out: Dict[str, float] = {
        "peak_rate_mhz": np.nan,
        "peak_range_m": np.nan,
        "cross_max_toggle_m": np.nan,
        "cross_min_toggle_m": np.nan,
        "cross_max_toggle_idx": np.nan,
        "cross_min_toggle_idx": np.nan,
    }

    if gluing_mode_l == "photon_only":
        # Skip toggle fit and blend zone search; use dead-time-corrected photon only.
        fit_mode = "photon_only_skipped"
    else:
        fit_mask = (
            sig_mask
            & np.isfinite(analog_raw)
            & np.isfinite(photon_dt_mhz)
            & np.isfinite(photon_toggle)
            & (photon_toggle >= float(min_toggle_rate))   # dead-time-corrected (LabVIEW-matched)
            & (photon_toggle <= float(max_toggle_rate))
            & (~sat)                                       # drop saturated analog bins
        )
        n_fit = int(np.sum(fit_mask))

        if n_fit >= 2:
            slope, offset, fit_quality_r2, fit_rmse, _ = linear_regression_stats(
                analog_raw[fit_mask], photon_dt_mhz[fit_mask]
            )
            analog_scaled = slope * analog_raw + offset
        else:
            fallback_mask = sig_mask & np.isfinite(analog_raw) & np.isfinite(photon_dt_mhz) & (~sat)
            if int(np.sum(fallback_mask)) >= 2:
                slope, offset, fit_quality_r2, fit_rmse, n_fit = linear_regression_stats(
                    analog_raw[fallback_mask], photon_dt_mhz[fallback_mask]
                )
                analog_scaled = slope * analog_raw + offset
                fit_mode = "fallback_all_finite"
            else:
                fit_mode = "photon_only_no_fit"

        if bool(auto_blend):
            try:
                auto_out = auto_find_blend_zone_from_threshold_crossings(
                    r,
                    photon_toggle,   # dead-time-corrected rate crossing the toggles
                    sig_start_m=sig_start_m,
                    sig_end_m=sig_end_m,
                    min_toggle_rate=min_toggle_rate,
                    max_toggle_rate=max_toggle_rate,
                )
                blend_r1_used = float(auto_out["blend_r1_m"])
                blend_r2_used = float(auto_out["blend_r2_m"])
                auto_ok = 1.0
            except Exception:
                auto_ok = 0.0

    # Quality guard (daytime BG-subtracted glue only): if the analog↔photon fit is
    # poor — the analog channel degraded by daytime clipping / PMT protection — do
    # NOT trust the glue; fall back to photon-only for this profile. Verified on
    # 04-21 dawn: BG≤40 gives r²≥0.88 (glue kept), BG≥80 gives r²≤0.74 (rejected).
    guard_failed = (
        gluing_mode_l == "auto_bgsub_guarded"
        and not (np.isfinite(fit_quality_r2) and fit_quality_r2 >= float(day_glue_min_r2)
                 and np.isfinite(slope) and slope > 0.0)
    )

    if np.any(np.isfinite(analog_scaled)) and not guard_failed:
        w = cosine_taper_weight(r, blend_r1_used, blend_r2_used)
        hi_mask = sig_mask & np.isfinite(photon_toggle) & (photon_toggle > float(max_toggle_rate))
        lo_mask = sig_mask & np.isfinite(photon_toggle) & (photon_toggle < float(min_toggle_rate))
        w[hi_mask] = 0.0  # above upper toggle: scaled analog
        w[lo_mask] = 1.0  # below lower toggle: photon counting
        glued_profile = (1.0 - w) * analog_scaled + w * photon_dt_mhz
    else:
        w = np.ones_like(r, dtype=float)
        glued_profile = photon_dt_mhz.copy()

    bg_glued = float(bg_pretrigger_photon_mhz)
    glued_bgsub = glued_profile - bg_glued

    # ── Afterpulse correction (right after BG subtraction) ──────────────────
    # NRB = [(Raw·DT) − Afterpulse − BG] / [Overlap · Energy]  (SigmaMPL form)
    # Afterpulse is a detector-side artifact: photo-electrons trapped in PMT
    # dynodes are released later in time, producing a range-dependent
    # "ghost signal" that must be subtracted before geometric corrections.
    afterpulse_applied = False
    glued_ap = glued_bgsub
    if afterpulse_A_R is not None:
        A_arr = np.asarray(afterpulse_A_R, dtype=float)
        if A_arr.shape != glued_bgsub.shape:
            raise ValueError(
                f"afterpulse_A_R shape {A_arr.shape} does not match signal shape "
                f"{glued_bgsub.shape}; supply A(R) on the same range grid."
            )
        glued_ap = glued_bgsub - A_arr
        afterpulse_applied = True

    # ── Overlap correction (after afterpulse, before R² range correction) ──
    # P_corrected(R) = (P(R) - BG - A(R)) / O(R) ;  NaN where O(R) < O_min so
    # that near-range noise isn't amplified to meaningless values.
    overlap_applied = False
    overlap_first_trust_m = np.nan
    glued_oc = glued_ap
    if overlap_O_R is not None:
        O_arr = np.asarray(overlap_O_R, dtype=float)
        if O_arr.shape != glued_ap.shape:
            raise ValueError(
                f"overlap_O_R shape {O_arr.shape} does not match signal shape "
                f"{glued_ap.shape}; supply O(R) on the same range grid."
            )
        trust_mask = (O_arr >= float(overlap_O_min)) & np.isfinite(O_arr)
        glued_oc = np.where(
            trust_mask,
            glued_ap / np.maximum(O_arr, 1e-12),
            np.nan,
        )
        overlap_applied = True
        # Record the smallest R where O first crosses O_min (informational)
        if np.any(trust_mask):
            overlap_first_trust_m = float(r[trust_mask].min())

    nrb_1st = glued_oc * (r ** 2)
    energy_j = float(energy_mj) * 1e-3
    if not np.isfinite(energy_j) or energy_j <= 0:
        raise ValueError("energy_mj must be > 0")
    nrb_final = nrb_1st / energy_j

    max_nrb = float(np.nanmax(nrb_final)) if np.any(np.isfinite(nrb_final)) else np.nan
    if np.isfinite(max_nrb) and max_nrb > 0:
        nrb_norm = nrb_final / max_nrb
    else:
        nrb_norm = np.full_like(nrb_final, np.nan, dtype=float)

    qc: Dict[str, float] = {
        "bin_shift_bins_software": 0.0,
        "bin_shift_bins_python": 0.0,
        "slope": float(slope) if np.isfinite(slope) else np.nan,
        "offset": float(offset) if np.isfinite(offset) else np.nan,
        "fit_quality_r2": float(fit_quality_r2) if np.isfinite(fit_quality_r2) else np.nan,
        "fit_rmse": float(fit_rmse) if np.isfinite(fit_rmse) else np.nan,
        "n_toggle_points": float(n_fit),
        "bg_glued": float(bg_glued),
        "energy_mj": float(energy_mj),
        "sig_start_m": float(sig_start_m),
        "sig_end_m": float(sig_end_m),
        "min_toggle_rate": float(min_toggle_rate),
        "max_toggle_rate": float(max_toggle_rate),
        "blend_r1_used_m": float(blend_r1_used),
        "blend_r2_used_m": float(blend_r2_used),
        "auto_blend_enabled": 1.0 if bool(auto_blend) else 0.0,
        "auto_blend_ok": float(auto_ok),
        "blend_r1_m_auto": float(blend_r1_used) if auto_ok else np.nan,
        "blend_r2_m_auto": float(blend_r2_used) if auto_ok else np.nan,
        "peak_rate_mhz": float(auto_out.get("peak_rate_mhz", np.nan)),
        "peak_range_m": float(auto_out.get("peak_range_m", np.nan)),
        "cross_max_toggle_m": float(auto_out.get("cross_max_toggle_m", np.nan)),
        "cross_min_toggle_m": float(auto_out.get("cross_min_toggle_m", np.nan)),
        "cross_max_toggle_idx": float(auto_out.get("cross_max_toggle_idx", np.nan)),
        "cross_min_toggle_idx": float(auto_out.get("cross_min_toggle_idx", np.nan)),
        "overlap_applied": 1.0 if overlap_applied else 0.0,
        "overlap_o_min": float(overlap_O_min) if overlap_applied else np.nan,
        "overlap_first_trust_range_m": float(overlap_first_trust_m) if overlap_applied else np.nan,
        "afterpulse_applied": 1.0 if afterpulse_applied else 0.0,
        "exclude_saturated": 1.0 if (exclude_saturated and saturation_mask is not None) else 0.0,
        "saturated_bins_excluded": float(n_saturated_excluded),
    }
    inter = {
        "analog_raw_mV": analog_raw,
        "photon_raw_mhz": photon_rate_mhz,
        "photon_shifted_mhz": photon_shifted,
        "photon_dt_mhz": photon_dt_mhz,
        "analog_scaled_mhz": analog_scaled,
        "blend_weight": w,
        "glued_profile_mhz": glued_profile,
        "glued_bgsub_mhz": glued_bgsub,
        "glued_afterpulse_corr_mhz": glued_ap,
        "afterpulse_A_R": (np.asarray(afterpulse_A_R, dtype=float)
                           if afterpulse_A_R is not None else np.full_like(r, np.nan, dtype=float)),
        "glued_overlap_corr_mhz": glued_oc,
        "overlap_O_R": (np.asarray(overlap_O_R, dtype=float)
                        if overlap_O_R is not None else np.full_like(r, np.nan, dtype=float)),
        "nrb_1st": nrb_1st,
        "nrb_final": nrb_final,
        "nrb_norm": nrb_norm,
    }
    if guard_failed:
        glue_mode_text = "day_glue_guard_failed_photon_only"
    elif gluing_mode_l == "photon_only":
        glue_mode_text = "photon_only_no_glue"
    elif auto_ok:
        glue_mode_text = "threshold_crossing_cosine_bgsub" if _bgsub else "threshold_crossing_cosine"
    else:
        glue_mode_text = "manual_cosine_fallback_bgsub" if _bgsub else "manual_cosine_fallback"
    qc_text: Dict[str, float | str] = {
        "shift_mode": "disabled_zero",
        "fit_mode_text": fit_mode,
        "glue_mode_text": glue_mode_text,
        "gluing_mode": gluing_mode_l,
    }
    return nrb_norm, {**qc, **qc_text}, inter


# -----------------------------------------------------------------------------
# Profile-level API
# -----------------------------------------------------------------------------

def build_single_profile(
    path: Path,
    *,
    dr_m: float = 3.75,
    dead_time_ns: float = 3.06,
    bg_mode: str = "pretrigger",
    bg_start_m: float = 0.0,
    bg_end_m: float = 3750.0,
    ray_fit_rmin_m: float = 4000.0,
    ray_fit_rmax_m: float = 6000.0,
    ray_fit_sigma_gate: float = 3.0,
    pretrigger_bins: int = 1024,
    first_signal_bin: Optional[int] = None,
    first_signal_range_m: float = 3.75,
    blend_r1_m: float = 1200.0,
    blend_r2_m: float = 1800.0,
    shift_mode: str = "manual",
    bin_shift_bins: int = 0,
    onset_start_m: float = 3500.0,
    onset_end_m: float = 4000.0,
    onset_smooth_bins: int = 5,
    sig_start_m: float = 0.0,
    sig_end_m: float = 15000.0,
    min_toggle_rate: float = 0.5,
    max_toggle_rate: float = 10.0,
    auto_toggle_selector: bool = True,
    day_min_toggle_rate: float = 75.0,
    day_max_toggle_rate: float = 130.0,
    toggle_bg_switch_threshold_mhz: float = 10.0,
    pretrigger_trim_bins: int = 24,
    energy_mj: float = 25.0,
    auto_blend: bool = True,
    blend_search_start_m: Optional[float] = None,
    blend_search_end_m: Optional[float] = None,
    blend_window_m: float = 300.0,
    blend_step_bins: int = 5,
    blend_r2_threshold: float = 0.985,
    blend_min_points: int = 25,
    blend_norm_rmse_max: float = 0.12,
    blend_cluster_min_m: Optional[float] = None,
    gluing_mode: str = "auto",
    overlap_O_R: Optional[np.ndarray] = None,
    overlap_O_min: float = 0.1,
    afterpulse_A_R: Optional[np.ndarray] = None,
    channel: str = "parallel",
    exclude_saturated: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    raw_df = read_tr40_dat_ascii(
        path,
        dr_m=dr_m,
        trim_trailing_zeros=True,
        pretrigger_bins=pretrigger_bins,
        first_signal_bin=first_signal_bin,
        first_signal_range_m=first_signal_range_m,
        channel=channel,
    )
    r = raw_df["range_m"].to_numpy(float)
    analog = raw_df["analog_mV"].to_numpy(float)
    photon = raw_df["photon_MHz"].to_numpy(float)
    photon_stderr = raw_df["photon_stderr_MHz"].to_numpy(float)
    # Per-bin saturation flag (overflow>0) from the raw file; ASCII exports lack
    # it, so the column is all-zeros there and the mask is a no-op.
    if "overflow" in raw_df.columns:
        saturation_mask = raw_df["overflow"].to_numpy(float) > 0.0
    else:
        saturation_mask = np.zeros_like(r, dtype=bool)
    dr_eff = float(np.nanmedian(np.diff(r))) if len(r) >= 2 else float(dr_m)
    bin_width_ns = bin_width_ns_from_dr(dr_eff)

    bg_mode_l = str(bg_mode).strip().lower()
    if bg_mode_l not in ("pretrigger", "fixed", "far_range", "rayleigh_fit"):
        raise ValueError(
            f"Unknown bg_mode: {bg_mode!r}. Use 'pretrigger', 'fixed', "
            "'far_range', or 'rayleigh_fit'."
        )

    ray_fit_info: Dict[str, float] = {}
    if bg_mode_l == "pretrigger":
        if int(pretrigger_bins) <= 0:
            raise ValueError("bg_mode='pretrigger' requires pretrigger_bins > 0.")
        _bg_a_pre, bg_p_pre_mhz = extract_pretrigger_background(
            path, pretrigger_bins=pretrigger_bins, channel=channel)
        bg_p_pre_trim_mhz = trim_pretrigger_tail(bg_p_pre_mhz, pretrigger_trim_bins)
        bg_pre_photon_dt = float(np.nanmean(dead_time_correct_mhz(bg_p_pre_trim_mhz, dead_time_ns)))
        bg_window_start_m_used = 0.0
        bg_window_end_m_used = float(pretrigger_bins) * float(dr_m)
    elif bg_mode_l == "rayleigh_fit":
        # Fit the (dead-time-corrected) photon signal against the molecular signal
        # shape in an aerosol-free window; the intercept is the background. A 3-sigma
        # gate keeps a clean profile from being over-subtracted (daytime robustness).
        photon_dt_all = dead_time_correct_mhz(photon, dead_time_ns)
        ray_fit_info = rayleigh_fit_background(
            r, photon_dt_all,
            fit_rmin_m=ray_fit_rmin_m, fit_rmax_m=ray_fit_rmax_m,
            sigma_gate=ray_fit_sigma_gate,
        )
        bg_pre_photon_dt = float(ray_fit_info["background"])
        bg_window_start_m_used = float(ray_fit_rmin_m)
        bg_window_end_m_used = float(ray_fit_rmax_m)
    else:
        if bg_mode_l == "fixed":
            if not (np.isfinite(bg_start_m) and np.isfinite(bg_end_m) and float(bg_end_m) > float(bg_start_m)):
                raise ValueError(
                    "bg_mode='fixed' requires finite bg_start_m and bg_end_m with bg_end_m > bg_start_m (meters)."
                )
            bg_lo = float(bg_start_m)
            bg_hi = float(bg_end_m)
        else:  # far_range
            rmax_signal = float(np.nanmax(r)) if np.any(np.isfinite(r)) else np.nan
            if not np.isfinite(rmax_signal) or rmax_signal <= 0:
                raise ValueError("bg_mode='far_range' could not determine rmax from signal range.")
            bg_lo = 0.88 * rmax_signal
            bg_hi = 0.98 * rmax_signal
        bg_mask = (r >= bg_lo) & (r <= bg_hi) & np.isfinite(photon)
        if int(np.sum(bg_mask)) < 1:
            raise ValueError(
                f"bg_mode={bg_mode_l!r}: no valid bins in BG window [{bg_lo:.1f}, {bg_hi:.1f}] m."
            )
        bg_pre_photon_dt = float(np.nanmean(dead_time_correct_mhz(photon[bg_mask], dead_time_ns)))
        bg_window_start_m_used = bg_lo
        bg_window_end_m_used = bg_hi
    toggle_sel = select_toggle_rates(
        bg_pre_photon_dt,
        auto_toggle_selector=auto_toggle_selector,
        night_min_toggle_rate=min_toggle_rate,
        night_max_toggle_rate=max_toggle_rate,
        day_min_toggle_rate=day_min_toggle_rate,
        day_max_toggle_rate=day_max_toggle_rate,
        bg_switch_threshold_mhz=toggle_bg_switch_threshold_mhz,
    )
    min_toggle_used = float(toggle_sel["min_toggle_rate_selected"])
    max_toggle_used = float(toggle_sel["max_toggle_rate_selected"])

    # gluing_mode "auto_day_night": pick the glue per PROFILE from the pretrigger
    # background. In daylight the solar pedestal alone (170-260 MHz here) already
    # pushes the photon channel past the dead-time model's validity (~1/tau_d =
    # 326 MHz), so no valid analog<->photon overlap window exists — the fit then
    # falls back to all-bins and produces a wrong scale. Photon-only is the honest
    # choice by day; at night the near field IS photon-saturated, so glue is needed.
    gluing_mode_eff = str(gluing_mode).strip().lower()
    day_night_pick = ""
    if gluing_mode_eff in ("auto_day_night", "day_night"):
        is_day = (np.isfinite(bg_pre_photon_dt)
                  and bg_pre_photon_dt > float(toggle_bg_switch_threshold_mhz))
        # Daytime: attempt a BG-subtracted glue (keeps the still-linear analog at
        # dawn/dusk); the r² guard inside compute_nrb_reference_glue drops back to
        # photon-only when the analog has degraded (full daylight). Night: normal glue.
        gluing_mode_eff = "auto_bgsub_guarded" if is_day else "auto"
        day_night_pick = ("day_glue_bgsub" if is_day else "night_glue")
    gluing_mode = gluing_mode_eff

    nrb_norm, qc, inter = compute_nrb_reference_glue(
        r,
        analog,
        photon,
        dead_time_ns=dead_time_ns,
        sig_start_m=sig_start_m,
        sig_end_m=sig_end_m,
        min_toggle_rate=min_toggle_used,
        max_toggle_rate=max_toggle_used,
        energy_mj=energy_mj,
        bg_pretrigger_photon_mhz=bg_pre_photon_dt,
        auto_blend=auto_blend,
        blend_r1_m=blend_r1_m,
        blend_r2_m=blend_r2_m,
        blend_search_start_m=blend_search_start_m,
        blend_search_end_m=blend_search_end_m,
        blend_window_m=blend_window_m,
        blend_step_bins=blend_step_bins,
        blend_r2_threshold=blend_r2_threshold,
        blend_min_points=blend_min_points,
        blend_norm_rmse_max=blend_norm_rmse_max,
        blend_cluster_min_m=blend_cluster_min_m,
        gluing_mode=gluing_mode,
        overlap_O_R=overlap_O_R,
        overlap_O_min=overlap_O_min,
        afterpulse_A_R=afterpulse_A_R,
        saturation_mask=saturation_mask,
        exclude_saturated=exclude_saturated,
    )

    out = raw_df[["range_m", "source_bin_index", "analog_mV", "photon_MHz"]].copy()
    out["photon_shifted_MHz"] = inter["photon_shifted_mhz"]
    out["photon_deadtime_corr_MHz"] = inter["photon_dt_mhz"]
    out["analog_scaled_MHz"] = inter["analog_scaled_mhz"]
    out["blend_weight"] = inter["blend_weight"]
    out["glued_profile_MHz"] = inter["glued_profile_mhz"]
    out["glued_bgsub_MHz"] = inter["glued_bgsub_mhz"]
    out["glued_afterpulse_corr_MHz"] = inter["glued_afterpulse_corr_mhz"]
    out["afterpulse_A_R"] = inter["afterpulse_A_R"]
    out["glued_overlap_corr_MHz"] = inter["glued_overlap_corr_mhz"]
    out["overlap_O_R"] = inter["overlap_O_R"]
    out["nrb_final"] = inter["nrb_final"]
    out["nrb"] = inter["nrb_norm"]

    # ── Signal-to-noise ratio (photon channel) ──────────────────────────────
    # SNR = (dead-time-corrected photon − BG) / dead-time-corrected stderr.
    # Per the Licel manual §9.3 the file's stderr column is the standard error of
    # the mean σ_µ = s/√N of the RAW photon rate. The dead-time correction
    # N_true = N/(1 − N·τ) is non-linear, so its error propagates as
    #   σ_corr = σ_raw · |dN_true/dN| = σ_raw / (1 − N·τ)² .
    # Numerator and denominator must refer to the SAME (corrected) quantity;
    # correcting only the numerator overstated SNR in the near field.
    # Net effect: SNR_corr = (1 − N·τ)·SNR_raw — negligible far-field (N·τ→0),
    # a small honest reduction near field where correction amplifies noise.
    bg_for_snr = float(qc.get("bg_glued", np.nan))
    with np.errstate(divide="ignore", invalid="ignore"):
        dt_denom = 1.0 - photon * float(dead_time_ns) * 1e-3
        dt_denom = np.where(np.abs(dt_denom) < 1e-12, np.nan, dt_denom)
        photon_stderr_dt = photon_stderr / (dt_denom ** 2)
        sig_bgsub = inter["photon_dt_mhz"] - bg_for_snr
        snr = np.where(
            (photon_stderr_dt > 0) & np.isfinite(photon_stderr_dt),
            sig_bgsub / photon_stderr_dt, np.nan,
        )
    out["photon_stderr_MHz"] = photon_stderr           # raw σ_µ from the file (unchanged)
    out["photon_stderr_dt_MHz"] = photon_stderr_dt      # dead-time-propagated σ_µ
    out["snr"] = snr

    meta: Dict[str, float] = {
        "bin_spacing_m": dr_eff,
        "bin_width_ns": float(bin_width_ns),
        "bg_mode_used": bg_mode_l,
        "bg_start_m_used": float(bg_window_start_m_used),
        "bg_end_m_used": float(bg_window_end_m_used),
        "pretrigger_bins": int(pretrigger_bins),
        "first_signal_bin": int(first_signal_bin if first_signal_bin is not None else pretrigger_bins + 1),
        "first_signal_range_m": float(first_signal_range_m),
        "shift_mode": "disabled_zero",
        "toggle_mode": str(toggle_sel.get("toggle_mode_text", "")),
        "fit_mode": str(qc.get("fit_mode_text", "")),
        "glue_mode": str(qc.get("glue_mode_text", "")),
        "gluing_mode": str(qc.get("gluing_mode", "auto")),
        "day_night_glue_pick": day_night_pick,
        "bin_shift_bins_software": 0.0,
        "bin_shift_bins_python": 0.0,
        "sig_start_m": float(sig_start_m),
        "sig_end_m": float(sig_end_m),
        "min_toggle_rate_input": float(min_toggle_rate),
        "max_toggle_rate_input": float(max_toggle_rate),
        "min_toggle_rate": float(min_toggle_used),
        "max_toggle_rate": float(max_toggle_used),
        "day_min_toggle_rate": float(day_min_toggle_rate),
        "day_max_toggle_rate": float(day_max_toggle_rate),
        "toggle_bg_switch_threshold_mhz": float(toggle_bg_switch_threshold_mhz),
        "pretrigger_trim_bins": int(pretrigger_trim_bins),
        "bg_pretrigger_photon_dt_trimmed_mhz": float(bg_pre_photon_dt),
        "ray_fit_background_mhz": float(ray_fit_info.get("background", np.nan)),
        "ray_fit_b_raw_mhz": float(ray_fit_info.get("b_raw", np.nan)),
        "ray_fit_b_sigma_mhz": float(ray_fit_info.get("b_sigma", np.nan)),
        "ray_fit_r2": float(ray_fit_info.get("r2", np.nan)),
        "ray_fit_significant": 1.0 if ray_fit_info.get("significant") else 0.0,
        "blend_r1_m_manual": float(blend_r1_m),
        "blend_r2_m_manual": float(blend_r2_m),
        "blend_r1_used_m": float(qc["blend_r1_used_m"]),
        "blend_r2_used_m": float(qc["blend_r2_used_m"]),
        "slope": float(qc["slope"]),
        "offset": float(qc["offset"]),
        "fit_quality_r2": float(qc["fit_quality_r2"]),
        "fit_rmse": float(qc["fit_rmse"]),
        "n_toggle_points": int(round(qc["n_toggle_points"])),
        "bg_glued": float(qc["bg_glued"]),
        "energy_mj": float(energy_mj),
        "auto_blend": bool(auto_blend),
        "auto_blend_ok": float(qc["auto_blend_ok"]),
        "blend_r1_m_auto": float(qc["blend_r1_m_auto"]),
        "blend_r2_m_auto": float(qc["blend_r2_m_auto"]),
        "peak_rate_mhz": float(qc["peak_rate_mhz"]),
        "peak_range_m": float(qc["peak_range_m"]),
        "cross_max_toggle_m": float(qc["cross_max_toggle_m"]),
        "cross_min_toggle_m": float(qc["cross_min_toggle_m"]),
        "cross_max_toggle_idx": float(qc["cross_max_toggle_idx"]),
        "cross_min_toggle_idx": float(qc["cross_min_toggle_idx"]),
        "overlap_applied": float(qc["overlap_applied"]),
        "overlap_o_min": float(qc["overlap_o_min"]),
        "overlap_first_trust_range_m": float(qc["overlap_first_trust_range_m"]),
        "afterpulse_applied": float(qc["afterpulse_applied"]),
        "exclude_saturated": float(qc["exclude_saturated"]),
        "saturated_bins_excluded": float(qc["saturated_bins_excluded"]),
    }
    # Trusted range = farthest bin with SNR >= 3 (informational QC metric).
    m_snr3 = np.isfinite(snr) & (snr >= 3.0) & (r > 0)
    meta["trusted_range_snr3_m"] = float(np.nanmax(r[m_snr3])) if np.any(m_snr3) else np.nan
    meta["snr_max"] = float(np.nanmax(snr)) if np.any(np.isfinite(snr)) else np.nan
    return out, meta


# -----------------------------------------------------------------------------
# Signal-quality (SNR) gating
# -----------------------------------------------------------------------------
# "Signal-quality-first": above the SNR-trusted range the NRB is pure noise x R^2
# (no real signal), so a normalise-by-global-max lands the reference on a far-range
# noise spike and squashes the real boundary layer. Gating masks the low-SNR bins
# (blank, not shown as signal) and normalises within the trusted region so the BL
# peak sits near 1.0. Default OFF everywhere → existing results unchanged.

SNR_SMOOTH_BINS = 15   # rolling window for the trusted-range walk (~50 m at 3.75 m bins)


def _rolling_median(a: np.ndarray, win: int) -> np.ndarray:
    """Centered rolling median, NaN-aware, same length as `a`."""
    win = max(1, int(win))
    if win <= 1:
        return np.asarray(a, float)
    s = pd.Series(np.asarray(a, float))
    return s.rolling(win, center=True, min_periods=max(3, win // 2)).median().to_numpy()


def snr_trusted_top(r, snr, snr_min: float = 3.0, smooth_bins: int = SNR_SMOOTH_BINS) -> float:
    """Contiguous trusted-range top from a SMOOTHED SNR profile.

    Walks UP from the peak of the smoothed SNR (the signal core) until the
    smoothed SNR first drops below snr_min. Smoothing (rolling median) is the
    key: daytime noise has ~25% of bins randomly exceeding 3 sigma, so a raw
    per-bin criterion keeps a spray of isolated noise bins far aloft; the median
    ignores those single-bin spikes, so the top lands at the real boundary-layer
    (or sustained cloud) top. Returns metres, or NaN if nothing is trusted."""
    r = np.asarray(r, float)
    snr = np.asarray(snr, float)
    valid = np.isfinite(snr) & np.isfinite(r)
    if not valid.any():
        return float("nan")
    sm = _rolling_median(np.where(valid, snr, np.nan), smooth_bins)
    ok = np.isfinite(sm) & (sm >= float(snr_min))
    if not ok.any():
        return float("nan")
    core = int(np.nanargmax(np.where(np.isfinite(sm), sm, -np.inf)))
    if not ok[core]:
        return float("nan")
    top = float(r[core])
    for i in range(core, len(r)):
        if not ok[i]:
            break
        top = float(r[i])
    return top


SNR_NORM_CAP_M = 6000.0   # lower-troposphere cap for the normalisation reference


def snr_gate_nrb(r, nrb_final, snr, snr_min: float = 3.0,
                 *, normalize_within_trusted: bool = True,
                 norm_cap_m: float = SNR_NORM_CAP_M,
                 mask_above_trusted: bool = True):
    """Return a 0-1 NRB profile whose NORMALISATION is anchored to the boundary
    layer (not a far-range R² noise spike), keyed to the CONTIGUOUS smoothed
    SNR-trusted top (see snr_trusted_top).

    • NORMALISE — by the max within the trusted top AND the lower troposphere
      (r <= min(top, norm_cap_m)); the cap guards against a far detector artifact
      when the trusted top runs high at night. Anchors the boundary layer near 1.0.
    • MASK (``mask_above_trusted``) — when True (default) everything ABOVE the
      trusted top is set NaN, a clean cutoff for a compact daytime view. When
      **False** the FULL profile is kept (only the normalisation reference is
      confined to the trusted/lower-troposphere window) so the whole curve stays
      visible for a full-range overlay (e.g. Prototype vs Mini-MPL in Step 5) —
      the boundary layer sits near 1.0 and any far-range noise simply runs off the
      top of a 0-1 axis instead of the line being cut short.

    Returns (nrb_norm, trusted_top_m)."""
    r = np.asarray(r, float)
    nrb = np.asarray(nrb_final, float)
    snr = np.asarray(snr, float)
    top = snr_trusted_top(r, snr, snr_min)

    # Normalisation reference window (max → 1.0):
    #  • masked view  : within the trusted top AND the lower-troposphere cap, so
    #    the visible (below-top) profile peaks at 1.0.
    #  • unmasked view: the lower-troposphere cap alone (r <= norm_cap_m), giving a
    #    predictable boundary-layer peak of 1.0 regardless of a low trusted top —
    #    the far-range R² noise then simply runs off the top of a 0-1 axis.
    if normalize_within_trusted:
        if mask_above_trusted and np.isfinite(top):
            ref = (r <= top) & (r <= float(norm_cap_m)) & np.isfinite(nrb)
        else:
            ref = (r <= float(norm_cap_m)) & np.isfinite(nrb)
        if not ref.any():
            ref = np.isfinite(nrb)
        mx = float(np.nanmax(nrb[ref])) if ref.any() else float("nan")
    else:
        mx = float(np.nanmax(nrb)) if np.any(np.isfinite(nrb)) else float("nan")

    if mask_above_trusted:
        if not np.isfinite(top):
            return np.full_like(nrb, np.nan), float("nan")
        out = np.where(r <= top, nrb, np.nan)
    else:
        out = nrb.copy()

    if np.isfinite(mx) and mx > 0:
        nrb_norm = out / mx
    else:
        nrb_norm = np.full_like(nrb, np.nan)
    return nrb_norm, top


# -----------------------------------------------------------------------------
# Timestamp helpers / daily builder
# -----------------------------------------------------------------------------

def parse_time_from_filename(name: str) -> Optional[Tuple[int, int]]:
    patterns = [
        r"(?<!\d)(\d{2})[.:_-](\d{2})(?!\d)",
        r"T(\d{2})(\d{2})(?!\d)",
    ]
    for pat in patterns:
        matches = list(re.finditer(pat, name))
        if not matches:
            continue
        m = matches[-1]
        hh, mm = int(m.group(1)), int(m.group(2))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return hh, mm
    return None


def parse_timestamp_from_filename(name: str, date_str: Optional[str] = None) -> Optional[pd.Timestamp]:
    hits12 = re.findall(r"(\d{12})", name)
    for token in reversed(hits12):
        ts = pd.to_datetime(token, format="%Y%m%d%H%M", errors="coerce")
        if pd.notna(ts):
            return pd.Timestamp(ts)

    if date_str:
        t = parse_time_from_filename(name)
        if t is not None:
            return make_timestamp(date_str, t[0], t[1])
    return None


def make_timestamp(date_str: str, hh: int, mm: int) -> pd.Timestamp:
    base = pd.to_datetime(date_str).normalize()
    return base + pd.Timedelta(hours=hh, minutes=mm)


def _parse_start_time_to_minutes(start_time: Optional[str]) -> Optional[int]:
    if start_time is None:
        return None
    s = str(start_time).strip()
    if not s:
        return None
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", s)
    if not m:
        raise ValueError("Start time must be in HH:MM format.")
    hh = int(m.group(1))
    mm = int(m.group(2))
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError("Start time must be in HH:MM format.")
    return hh * 60 + mm


def collect_actual_timestamps(
    folder: Path,
    *,
    pattern: str,
    date_str: Optional[str] = None,
    start_time: Optional[str] = None,
    recursive: bool = False,
    logger: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Tuple[pd.Timestamp, Path]], List[Dict[str, object]]]:
    files = glob_lidar_files(folder, pattern, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No files found in {folder} with pattern: {pattern}")

    date_filter = pd.to_datetime(date_str).normalize() if str(date_str).strip() else None
    start_min_total = _parse_start_time_to_minutes(start_time)

    file_map: Dict[pd.Timestamp, Path] = {}
    qc_rows: List[Dict[str, object]] = []

    for f in files:
        ts = parse_timestamp_from_filename(f.name, date_str=date_str)
        if ts is None:
            ts = parse_licel_raw_timestamp(f.name)   # extensionless Licel raw files
        if ts is None:
            qc_rows.append({"file": f.name, "time": pd.NaT, "status": "skip_no_timestamp"})
            if logger:
                logger(f"Skip (no timestamp): {f.name}")
            continue

        if date_filter is not None and ts.normalize() != date_filter:
            qc_rows.append({"file": f.name, "time": ts, "status": "skip_other_date"})
            continue

        if start_min_total is not None:
            ts_min_total = int(ts.hour) * 60 + int(ts.minute)
            if ts_min_total < start_min_total:
                qc_rows.append({"file": f.name, "time": ts, "status": "skip_before_start_time"})
                continue

        if ts in file_map:
            qc_rows.append({"file": f.name, "time": ts, "status": "duplicate_time_ignored"})
            continue
        file_map[ts] = f

    ordered = sorted(file_map.items(), key=lambda kv: kv[0])
    return ordered, qc_rows


def build_daily_profile_from_folder(
    folder: Path,
    *,
    date_str: str,
    pattern: str,
    out_path: Path,
    start_time: str = "",
    recursive: bool = False,
    dr_m: float = 3.75,
    dead_time_ns: float = 3.06,
    bg_mode: str = "pretrigger",
    bg_start_m: float = 0.0,
    bg_end_m: float = 3750.0,
    ray_fit_rmin_m: float = 4000.0,
    ray_fit_rmax_m: float = 6000.0,
    ray_fit_sigma_gate: float = 3.0,
    pretrigger_bins: int = 1024,
    first_signal_bin: Optional[int] = None,
    first_signal_range_m: float = 3.75,
    blend_r1_m: float = 1200.0,
    blend_r2_m: float = 1800.0,
    shift_mode: str = "manual",
    bin_shift_bins: int = 0,
    onset_start_m: float = 3500.0,
    onset_end_m: float = 4000.0,
    onset_smooth_bins: int = 5,
    sig_start_m: float = 0.0,
    sig_end_m: float = 15000.0,
    min_toggle_rate: float = 0.5,
    max_toggle_rate: float = 10.0,
    auto_toggle_selector: bool = True,
    day_min_toggle_rate: float = 75.0,
    day_max_toggle_rate: float = 130.0,
    toggle_bg_switch_threshold_mhz: float = 10.0,
    pretrigger_trim_bins: int = 24,
    energy_mj: float = 25.0,
    auto_blend: bool = True,
    blend_search_start_m: Optional[float] = None,
    blend_search_end_m: Optional[float] = None,
    blend_window_m: float = 300.0,
    blend_step_bins: int = 5,
    blend_r2_threshold: float = 0.985,
    blend_min_points: int = 25,
    blend_norm_rmse_max: float = 0.12,
    blend_cluster_min_m: Optional[float] = None,
    gluing_mode: str = "auto",
    overlap_O_R: Optional[np.ndarray] = None,
    overlap_O_min: float = 0.1,
    afterpulse_A_R: Optional[np.ndarray] = None,
    snr_gate: bool = True,
    snr_min: float = 3.0,
    channel: str = "parallel",
    exclude_saturated: bool = False,
    strict: bool = True,
    logger: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered_files, qc_rows = collect_actual_timestamps(
        folder,
        pattern=pattern,
        date_str=date_str,
        start_time=start_time,
        recursive=recursive,
        logger=logger,
    )

    if not ordered_files:
        raise ValueError("No valid .dat files found for the selected date/start time.")

    ref_r: Optional[np.ndarray] = None
    ts_list: List[pd.Timestamp] = []
    nrb_cols: List[np.ndarray] = []
    snr_cols: List[np.ndarray] = []
    total = len(ordered_files)

    for idx, (ts, f) in enumerate(ordered_files, start=1):
        if logger:
            logger(f"[{idx}/{total}] {f.name} -> {ts.strftime('%H:%M')}")
        try:
            profile_one, meta = build_single_profile(
                f,
                dr_m=dr_m,
                dead_time_ns=dead_time_ns,
                bg_mode=bg_mode,
                bg_start_m=bg_start_m,
                bg_end_m=bg_end_m,
                ray_fit_rmin_m=ray_fit_rmin_m,
                ray_fit_rmax_m=ray_fit_rmax_m,
                ray_fit_sigma_gate=ray_fit_sigma_gate,
                pretrigger_bins=pretrigger_bins,
                first_signal_bin=first_signal_bin,
                first_signal_range_m=first_signal_range_m,
                blend_r1_m=blend_r1_m,
                blend_r2_m=blend_r2_m,
                shift_mode=shift_mode,
                bin_shift_bins=bin_shift_bins,
                onset_start_m=onset_start_m,
                onset_end_m=onset_end_m,
                onset_smooth_bins=onset_smooth_bins,
                sig_start_m=sig_start_m,
                sig_end_m=sig_end_m,
                min_toggle_rate=min_toggle_rate,
                max_toggle_rate=max_toggle_rate,
                auto_toggle_selector=auto_toggle_selector,
                day_min_toggle_rate=day_min_toggle_rate,
                day_max_toggle_rate=day_max_toggle_rate,
                toggle_bg_switch_threshold_mhz=toggle_bg_switch_threshold_mhz,
                pretrigger_trim_bins=pretrigger_trim_bins,
                energy_mj=energy_mj,
                auto_blend=auto_blend,
                blend_search_start_m=blend_search_start_m,
                blend_search_end_m=blend_search_end_m,
                blend_window_m=blend_window_m,
                blend_step_bins=blend_step_bins,
                blend_r2_threshold=blend_r2_threshold,
                blend_min_points=blend_min_points,
                blend_norm_rmse_max=blend_norm_rmse_max,
                blend_cluster_min_m=blend_cluster_min_m,
                gluing_mode=gluing_mode,
                overlap_O_R=overlap_O_R,
                overlap_O_min=overlap_O_min,
                afterpulse_A_R=afterpulse_A_R,
                channel=channel,
                exclude_saturated=exclude_saturated,
            )
            r = profile_one["range_m"].to_numpy(float)
            snr = profile_one["snr"].to_numpy(float)
            if snr_gate:
                # Normalise within the SNR-trusted lower troposphere (instead of
                # the global-max normalise in "nrb"). mask_above_trusted=False
                # keeps the full profile visible for a full-range overlay.
                nrb, trusted_top = snr_gate_nrb(
                    r, profile_one["nrb_final"].to_numpy(float), snr, snr_min,
                    mask_above_trusted=False)
                meta["snr_gate"] = 1.0
                meta["snr_gate_min"] = float(snr_min)
                meta["snr_gate_trusted_range_m"] = float(trusted_top)
            else:
                # snr_gate off: still normalise by the max within the lower
                # troposphere (r <= SNR_NORM_CAP_M), NOT the global max — the
                # daytime global max is a far-range R^2-amplified noise spike that
                # squashes the real boundary layer. Falls back to the pre-computed
                # global-max "nrb" only if that window is empty.
                nrb_final_arr = profile_one["nrb_final"].to_numpy(float)
                cap = np.isfinite(nrb_final_arr) & (r <= SNR_NORM_CAP_M)
                if cap.any() and np.isfinite(nrb_final_arr[cap]).any():
                    mx = float(np.nanmax(nrb_final_arr[cap]))
                    nrb = (nrb_final_arr / mx if (np.isfinite(mx) and mx > 0)
                           else profile_one["nrb"].to_numpy(float))
                else:
                    nrb = profile_one["nrb"].to_numpy(float)
        except Exception as e:
            if strict:
                raise
            qc_rows.append({"file": f.name, "time": ts, "status": f"error: {e}"})
            if logger:
                logger(f"error: {e}")
            if progress_cb:
                progress_cb(100.0 * idx / total)
            continue

        if ref_r is None:
            ref_r = r.copy()
        elif len(r) != len(ref_r) or np.nanmax(np.abs(r - ref_r)) > 1e-6:
            nrb = np.interp(ref_r, r, nrb, left=np.nan, right=np.nan)
            snr = np.interp(ref_r, r, snr, left=np.nan, right=np.nan)

        ts_list.append(ts)
        nrb_cols.append(nrb.astype(float))
        snr_cols.append(snr.astype(float))
        qc_rows.append({"file": f.name, "time": ts, "status": "ok", **meta})
        if progress_cb:
            progress_cb(100.0 * idx / total)

    if ref_r is None or not ts_list:
        raise ValueError("No valid .dat files processed.")

    mat = np.column_stack(nrb_cols)
    df_profile = pd.DataFrame(mat, columns=ts_list)
    df_profile.insert(0, "Range(m)", ref_r)
    df_snr = pd.DataFrame(np.column_stack(snr_cols), columns=ts_list)
    df_snr.insert(0, "Range(m)", ref_r)
    df_qc = pd.DataFrame(qc_rows)
    if not df_qc.empty and "time" in df_qc.columns and "status" in df_qc.columns:
        df_qc = df_qc.sort_values(["time", "status"], na_position="last").reset_index(drop=True)

    params = pd.DataFrame([
        {
            "version": "v9_2_reference_glue",
            "date": date_str,
            "start_time": str(start_time).strip(),
            "pattern": pattern,
            "recursive": bool(recursive),
            "timestamp_mode": "actual_files",
            "n_profiles_actual": int(len(ts_list)),
            "dr_m": float(dr_m),
            "dead_time_ns": float(dead_time_ns),
            "bg_mode": bg_mode,
            "bg_start_m": float(bg_start_m),
            "bg_end_m": float(bg_end_m),
            "pretrigger_bins": int(pretrigger_bins),
            "first_signal_bin": int(first_signal_bin if first_signal_bin is not None else pretrigger_bins + 1),
            "first_signal_range_m": float(first_signal_range_m),
            "shift_mode": "disabled_zero",
            "bin_shift_bins_software_manual": 0,
            "sig_start_m": float(sig_start_m),
            "sig_end_m": float(sig_end_m),
            "min_toggle_rate_night_default": float(min_toggle_rate),
            "max_toggle_rate_night_default": float(max_toggle_rate),
            "auto_toggle_selector": bool(auto_toggle_selector),
            "day_min_toggle_rate": float(day_min_toggle_rate),
            "day_max_toggle_rate": float(day_max_toggle_rate),
            "toggle_bg_switch_threshold_mhz": float(toggle_bg_switch_threshold_mhz),
            "pretrigger_trim_bins": int(pretrigger_trim_bins),
            "blend_r1_manual_m": float(blend_r1_m),
            "blend_r2_manual_m": float(blend_r2_m),
            "auto_blend_threshold_crossing": bool(auto_blend),
            "gluing_mode": str(gluing_mode),
            "energy_mj": float(energy_mj),
            "strict": bool(strict),
            "overlap_correction": "applied" if overlap_O_R is not None else "not_applied",
            "overlap_o_min": float(overlap_O_min) if overlap_O_R is not None else float("nan"),
            "afterpulse_correction": "applied" if afterpulse_A_R is not None else "not_applied",
            "snr_gate": "applied" if snr_gate else "not_applied",
            "snr_gate_min": float(snr_min) if snr_gate else float("nan"),
        }
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        df_profile.to_excel(xw, index=False, sheet_name="NRB profile")
        df_snr.to_excel(xw, index=False, sheet_name="SNR")
        df_qc.to_excel(xw, index=False, sheet_name="QC_params")
        params.to_excel(xw, index=False, sheet_name="Parameters")

    return df_profile, df_qc, params
