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
# .dat reading helpers
# -----------------------------------------------------------------------------

def find_dat_data_start(lines: List[str]) -> int:
    for i, ln in enumerate(lines):
        if ("Analog" in ln) and ("Photon" in ln or "Photon Counting" in ln):
            return i + 1
    return 9


def _read_tr40_dat_ascii_array(
    path: Path,
    *,
    start_mode: str = "auto",
    trim_trailing_zeros: bool = True,
) -> np.ndarray:
    lines = path.read_text(errors="replace").splitlines()
    start = find_dat_data_start(lines) if start_mode == "auto" else 9

    rows = []
    for ln in lines[start:]:
        ln = ln.strip()
        if not ln:
            continue
        parts = re.split(r"[\t ]+", ln)
        if len(parts) < 4:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))
        except Exception:
            continue

    if not rows:
        raise ValueError(f"No numeric data rows found in {path.name}.")

    arr = np.asarray(rows, dtype=float)
    if trim_trailing_zeros:
        nonzero_row = np.any(np.abs(arr[:, :4]) > 0, axis=1)
        if not np.any(nonzero_row):
            raise ValueError(f"All data rows are zero in {path.name}")
        arr = arr[: int(np.where(nonzero_row)[0][-1]) + 1, :]
    return arr


def read_tr40_dat_ascii(
    path: Path,
    dr_m: float = 3.75,
    start_mode: str = "auto",
    trim_trailing_zeros: bool = True,
    pretrigger_bins: int = 0,
    first_signal_bin: Optional[int] = None,
    first_signal_range_m: float = 3.75,
) -> pd.DataFrame:
    arr = _read_tr40_dat_ascii_array(path, start_mode=start_mode, trim_trailing_zeros=trim_trailing_zeros)
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

    return pd.DataFrame(
        {
            "bin_index": bin_index,
            "source_bin_index": source_bin_index,
            "range_m": range_m,
            "analog_mV": arr_sig[:, 0],
            "analog_stderr_mV": arr_sig[:, 1],
            "photon_MHz": arr_sig[:, 2],
            "photon_stderr_MHz": arr_sig[:, 3],
        }
    )


def extract_pretrigger_background(
    path: Path,
    *,
    pretrigger_bins: int,
    start_mode: str = "auto",
    trim_trailing_zeros: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    arr = _read_tr40_dat_ascii_array(path, start_mode=start_mode, trim_trailing_zeros=trim_trailing_zeros)
    pretrigger_bins = int(pretrigger_bins)
    if pretrigger_bins <= 0:
        raise ValueError("pretrigger_bins must be > 0 for pretrigger background mode.")
    if pretrigger_bins > arr.shape[0]:
        raise ValueError(
            f"pretrigger_bins={pretrigger_bins} is too large for {path.name} (available bins={arr.shape[0]})."
        )
    return arr[:pretrigger_bins, 0].astype(float), arr[:pretrigger_bins, 2].astype(float)


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
    overlap_O_R: Optional[np.ndarray] = None,
    overlap_O_min: float = 0.1,
    afterpulse_A_R: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, np.ndarray]]:
    r = np.asarray(r_m, float)
    analog_raw = np.asarray(analog_raw, float)
    photon_rate_mhz = np.asarray(photon_rate_mhz, float)

    gluing_mode_l = str(gluing_mode).strip().lower()
    if gluing_mode_l not in ("auto", "photon_only"):
        raise ValueError(
            f"Unknown gluing_mode: {gluing_mode!r}. Use 'auto' or 'photon_only'."
        )

    # Shift intentionally disabled in this workflow.
    photon_shifted = photon_rate_mhz.copy()
    photon_dt_mhz = dead_time_correct_mhz(photon_shifted, dead_time_ns)

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
            & (photon_dt_mhz >= float(min_toggle_rate))
            & (photon_dt_mhz <= float(max_toggle_rate))
        )
        n_fit = int(np.sum(fit_mask))

        if n_fit >= 2:
            slope, offset, fit_quality_r2, fit_rmse, _ = linear_regression_stats(
                analog_raw[fit_mask], photon_dt_mhz[fit_mask]
            )
            analog_scaled = slope * analog_raw + offset
        else:
            fallback_mask = sig_mask & np.isfinite(analog_raw) & np.isfinite(photon_dt_mhz)
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
                    photon_dt_mhz,
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

    if np.any(np.isfinite(analog_scaled)):
        w = cosine_taper_weight(r, blend_r1_used, blend_r2_used)
        hi_mask = sig_mask & np.isfinite(photon_dt_mhz) & (photon_dt_mhz > float(max_toggle_rate))
        lo_mask = sig_mask & np.isfinite(photon_dt_mhz) & (photon_dt_mhz < float(min_toggle_rate))
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
    if gluing_mode_l == "photon_only":
        glue_mode_text = "photon_only_no_glue"
    elif auto_ok:
        glue_mode_text = "threshold_crossing_cosine"
    else:
        glue_mode_text = "manual_cosine_fallback"
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
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    raw_df = read_tr40_dat_ascii(
        path,
        dr_m=dr_m,
        trim_trailing_zeros=True,
        pretrigger_bins=pretrigger_bins,
        first_signal_bin=first_signal_bin,
        first_signal_range_m=first_signal_range_m,
    )
    r = raw_df["range_m"].to_numpy(float)
    analog = raw_df["analog_mV"].to_numpy(float)
    photon = raw_df["photon_MHz"].to_numpy(float)
    dr_eff = float(np.nanmedian(np.diff(r))) if len(r) >= 2 else float(dr_m)
    bin_width_ns = bin_width_ns_from_dr(dr_eff)

    bg_mode_l = str(bg_mode).strip().lower()
    if bg_mode_l not in ("pretrigger", "fixed", "far_range"):
        raise ValueError(
            f"Unknown bg_mode: {bg_mode!r}. Use 'pretrigger', 'fixed', or 'far_range'."
        )

    if bg_mode_l == "pretrigger":
        if int(pretrigger_bins) <= 0:
            raise ValueError("bg_mode='pretrigger' requires pretrigger_bins > 0.")
        _bg_a_pre, bg_p_pre_mhz = extract_pretrigger_background(path, pretrigger_bins=pretrigger_bins)
        bg_p_pre_trim_mhz = trim_pretrigger_tail(bg_p_pre_mhz, pretrigger_trim_bins)
        bg_pre_photon_dt = float(np.nanmean(dead_time_correct_mhz(bg_p_pre_trim_mhz, dead_time_ns)))
        bg_window_start_m_used = 0.0
        bg_window_end_m_used = float(pretrigger_bins) * float(dr_m)
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
    }
    return out, meta


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
    files = sorted(folder.rglob(pattern) if recursive else folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {folder} with pattern: {pattern}")

    date_filter = pd.to_datetime(date_str).normalize() if str(date_str).strip() else None
    start_min_total = _parse_start_time_to_minutes(start_time)

    file_map: Dict[pd.Timestamp, Path] = {}
    qc_rows: List[Dict[str, object]] = []

    for f in files:
        ts = parse_timestamp_from_filename(f.name, date_str=date_str)
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
            )
            r = profile_one["range_m"].to_numpy(float)
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

        ts_list.append(ts)
        nrb_cols.append(nrb.astype(float))
        qc_rows.append({"file": f.name, "time": ts, "status": "ok", **meta})
        if progress_cb:
            progress_cb(100.0 * idx / total)

    if ref_r is None or not ts_list:
        raise ValueError("No valid .dat files processed.")

    mat = np.column_stack(nrb_cols)
    df_profile = pd.DataFrame(mat, columns=ts_list)
    df_profile.insert(0, "Range(m)", ref_r)
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
        }
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        df_profile.to_excel(xw, index=False, sheet_name="NRB profile")
        df_qc.to_excel(xw, index=False, sheet_name="QC_params")
        params.to_excel(xw, index=False, sheet_name="Parameters")

    return df_profile, df_qc, params
