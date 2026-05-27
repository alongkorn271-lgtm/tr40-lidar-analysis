"""Afterpulse correction module for the TR40 LiDAR.

Two ways to obtain the afterpulse profile A(R):

1. Raw .dat file (recommended workflow)
   - Calibration data taken with telescope covered + laser firing.
   - This module reads the .dat through the standard TR40 reader, applies
     dead-time correction, then subtracts the pretrigger (or far-range) noise
     floor. The residual time-resolved counts are the afterpulse profile.

2. Pre-computed CSV / Excel
   - 2 columns: range_m, afterpulse_MHz
   - Useful if you have processed A(R) from earlier campaigns or external tools.

The result is interpolated onto the NRB target range grid and intended to be
passed to `nrb_engine.compute_nrb_reference_glue()` via the new
`afterpulse_A_R` parameter.

References
----------
- Welton & Campbell (2002), J. Atmos. Oceanic Technol. 19, 2089-2094
- Campbell et al. (2002), J. Atmos. Oceanic Technol. 19, 431-442
- SigmaMPL User's Manual Section 4.6 (NRB formula includes afterpulse term)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Extract A(R) from a raw .dat calibration file
# -----------------------------------------------------------------------------
def compute_afterpulse_from_dat(
    dat_path: Union[str, Path],
    *,
    dr_m: float = 3.75,
    dead_time_ns: float = 3.06,
    pretrigger_bins: int = 1024,
    first_signal_bin: Optional[int] = None,
    first_signal_range_m: float = 3.75,
    pretrigger_trim_bins: int = 24,
    bg_mode: str = "pretrigger",
) -> Tuple[np.ndarray, np.ndarray]:
    """Process a raw afterpulse .dat file and return (range_m, A_R).

    Workflow (mirrors part of compute_nrb_reference_glue):
        Read .dat
          -> photon_dt = dead_time_correct_mhz(photon, dead_time_ns)
          -> BG (mean of pretrigger OR mean of far-range bins, dead-time corrected)
          -> A(R) = photon_dt - BG

    Parameters
    ----------
    dat_path : path-like
        TR40 .dat file recorded with telescope covered + laser firing.
    dr_m, dead_time_ns, pretrigger_bins, first_signal_bin,
    first_signal_range_m, pretrigger_trim_bins :
        Same meaning as in nrb_engine.compute_nrb_reference_glue().
    bg_mode : str
        'pretrigger' (default) or 'far_range'.

    Returns
    -------
    r_m, A_R : ndarray
        Range axis (m) and afterpulse profile (MHz) on that grid.
    """
    # Lazy import — avoid circular dependency at module load time
    from nrb_engine import (
        read_tr40_dat_ascii,
        dead_time_correct_mhz,
        extract_pretrigger_background,
        trim_pretrigger_tail,
    )

    dat_path = Path(dat_path)
    if not dat_path.exists():
        raise FileNotFoundError(f"Afterpulse .dat not found: {dat_path}")

    df = read_tr40_dat_ascii(
        dat_path,
        dr_m=dr_m,
        pretrigger_bins=pretrigger_bins,
        first_signal_bin=first_signal_bin,
        first_signal_range_m=first_signal_range_m,
    )
    r = df["range_m"].to_numpy(float)
    photon = df["photon_MHz"].to_numpy(float)
    photon_dt = dead_time_correct_mhz(photon, dead_time_ns)

    bg_mode_l = str(bg_mode).strip().lower()
    if bg_mode_l == "pretrigger":
        _, bg_p_pre = extract_pretrigger_background(dat_path, pretrigger_bins=pretrigger_bins)
        bg_p_trim = trim_pretrigger_tail(bg_p_pre, pretrigger_trim_bins)
        bg = float(np.nanmean(dead_time_correct_mhz(bg_p_trim, dead_time_ns)))
    elif bg_mode_l == "far_range":
        far_lo, far_hi = 0.88 * r[-1], 0.98 * r[-1]
        m = (r >= far_lo) & (r <= far_hi) & np.isfinite(photon_dt)
        if int(np.sum(m)) < 1:
            raise ValueError("No valid bins in far-range BG window")
        bg = float(np.nanmean(photon_dt[m]))
    else:
        raise ValueError(f"Unsupported bg_mode for afterpulse: {bg_mode!r}")

    A_R = photon_dt - bg
    return r, np.asarray(A_R, dtype=float)


# -----------------------------------------------------------------------------
# Load A(R) from pre-computed CSV / Excel
# -----------------------------------------------------------------------------
def load_empirical_afterpulse(
    file_path: Union[str, Path],
    r_m_target: np.ndarray,
) -> np.ndarray:
    """Load A(R) from CSV/Excel and interpolate onto r_m_target.

    File format:
        - >= 2 columns; first = range (m), second = afterpulse (MHz)
        - Optional header row (auto-skipped by pandas)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Afterpulse file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in (".csv", ".txt", ".tsv"):
        try:
            df = pd.read_csv(file_path)
        except Exception:
            df = pd.read_csv(file_path, sep=r"\s+", engine="python")
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(
            f"Unsupported afterpulse file extension: {suffix} (use .csv, .xlsx, or .dat)"
        )

    if df.shape[1] < 2:
        raise ValueError(
            f"Afterpulse file must have >= 2 columns (range_m, afterpulse_MHz); got {df.shape[1]}"
        )

    r_src = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(float)
    A_src = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(float)
    mask = np.isfinite(r_src) & np.isfinite(A_src)
    r_src = r_src[mask]
    A_src = A_src[mask]
    if r_src.size < 2:
        raise ValueError("Afterpulse file has fewer than 2 valid (range, A) rows.")

    order = np.argsort(r_src)
    r_src = r_src[order]
    A_src = A_src[order]

    r_target = np.asarray(r_m_target, dtype=float)
    # Use 0 for out-of-range — afterpulse should be ~0 far away
    return np.interp(r_target, r_src, A_src, left=0.0, right=0.0).astype(float)


# -----------------------------------------------------------------------------
# Top-level dispatcher
# -----------------------------------------------------------------------------
def get_afterpulse(
    mode: str,
    r_m_target: np.ndarray,
    file_path: Optional[Union[str, Path]] = None,
    *,
    dr_m: float = 3.75,
    dead_time_ns: float = 3.06,
    pretrigger_bins: int = 1024,
    first_signal_bin: Optional[int] = None,
    first_signal_range_m: float = 3.75,
    pretrigger_trim_bins: int = 24,
    bg_mode: str = "pretrigger",
) -> Optional[np.ndarray]:
    """Top-level entry point used by the GUI / pipeline.

    mode: 'disabled' | 'load' (or 'file').
    When loading, the file extension determines the path:
        .dat            -> compute_afterpulse_from_dat()
        .csv / .xlsx    -> load_empirical_afterpulse()
    """
    m = (mode or "").strip().lower()
    if m in ("", "disabled", "off", "none"):
        return None
    if m.startswith("load") or m == "file":
        if not file_path:
            raise ValueError("File path required when afterpulse mode is 'load from file'.")
        file_path = Path(file_path)
        if file_path.suffix.lower() == ".dat":
            r_src, A_src = compute_afterpulse_from_dat(
                file_path,
                dr_m=dr_m, dead_time_ns=dead_time_ns,
                pretrigger_bins=pretrigger_bins,
                first_signal_bin=first_signal_bin,
                first_signal_range_m=first_signal_range_m,
                pretrigger_trim_bins=pretrigger_trim_bins,
                bg_mode=bg_mode,
            )
            r_target = np.asarray(r_m_target, dtype=float)
            return np.interp(r_target, r_src, A_src, left=0.0, right=0.0).astype(float)
        else:
            return load_empirical_afterpulse(file_path, r_m_target)
    raise ValueError(
        f"Unknown afterpulse mode: {mode!r}. Use 'disabled' or 'load from file'."
    )


# -----------------------------------------------------------------------------
# Self test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        dat = sys.argv[1]
    else:
        dat = r"C:\Users\Alongkorn\Desktop\afterpulse.dat"
    print(f"Processing: {dat}")
    r, A = compute_afterpulse_from_dat(dat)
    print(f"  n_bins:   {len(r)}")
    print(f"  range:    {r[0]:.2f} -> {r[-1]:.1f} m")
    print(f"  A(R=10m): {A[2]:.4e} MHz")
    print(f"  A(R=500m): {np.interp(500, r, A):.4e} MHz")
    print(f"  A(R=5km): {np.interp(5000, r, A):.4e} MHz")
    print(f"  mean(near<500m): {np.nanmean(A[r<500]):.4e}")
    print(f"  mean(far>8km):   {np.nanmean(A[r>8000]):.4e}  (~0 if BG correct)")
