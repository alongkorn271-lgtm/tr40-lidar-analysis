"""Overlap function O(R) module for the TR40 LiDAR.

Two ways to obtain O(R):
1. Analytical (Gaussian beam + Rice CDF — Kumar & Rocadenbosch 2013, J. Appl.
   Remote Sens. 7, 073591) using telescope/laser/detector hardware spec.
2. Empirical — load O(R) from a CSV/Excel file with (range_m, O) columns;
   the array is interpolated onto the target NRB range grid.

The output of `get_overlap(...)` is intended to be passed directly to
nrb_engine.compute_nrb_reference_glue() via the `overlap_O_R` parameter.

References
----------
- Kumar & Rocadenbosch (2013), J. Appl. Remote Sens. 7, 073591
- Halldorsson & Langerholc (1978), Applied Optics 17(2), 240-244
- Stelmaszczyk et al. (2005), Applied Optics 44(7), 1323-1331
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass
class OverlapHardware:
    """Hardware spec for analytical O(R) computation.

    Defaults match the NARIT TR40 setup:
        Telescope:  Celestron EdgeHD 800 (D=203 mm, f=2032 mm, F/10)
        Laser:      Quantel VIRON @ 532 nm, D4sigma=3.8 mm, divergence 1.5 mrad
        Detector:   Licel PM-HV / R9880U, 8 mm cathode
        Geometry:   Biaxial, d_perp = 148.12 mm
    """
    # Telescope
    f_mm: float = 2032.0
    primary_diameter_mm: float = 203.0
    secondary_obstruction_mm: float = 68.6
    # Laser
    laser_d4sigma_mm: float = 3.8
    laser_div_full_mrad: float = 1.5
    beam_expander: float = 1.0
    # Receiver
    field_stop_diameter_mm: float = 8.0
    # Biaxial geometry
    d_perp_mm: float = 148.12
    laser_tilt_rad: float = 0.0

    @property
    def sigma_0_mm(self) -> float:
        """1-sigma beam radius at exit (after any beam expander)."""
        return self.laser_d4sigma_mm * self.beam_expander / 4

    @property
    def sigma_theta_rad(self) -> float:
        """1-sigma half-divergence (after any beam expander)."""
        return (self.laser_div_full_mrad / self.beam_expander) / 4 * 1e-3


def compute_analytical_overlap(
    r_m: np.ndarray,
    hw: Optional[OverlapHardware] = None,
) -> np.ndarray:
    """Compute O(R) using the Gaussian beam + Rice CDF model.

    Parameters
    ----------
    r_m : ndarray
        Target range grid in meters.
    hw : OverlapHardware
        Hardware spec; uses NARIT TR40 defaults if None.

    Returns
    -------
    O : ndarray
        Overlap fraction in [0, 1], same shape as r_m.
    """
    try:
        from scipy.stats import rice
    except ImportError as exc:
        raise ImportError(
            "scipy is required for analytical overlap. Install via `pip install scipy`."
        ) from exc

    if hw is None:
        hw = OverlapHardware()

    r_arr = np.atleast_1d(np.asarray(r_m, dtype=float))
    r_mm = r_arr * 1000.0
    safe_r = np.where(r_mm > 0, r_mm, 1.0)

    f = hw.f_mm
    r_fs = hw.field_stop_diameter_mm / 2
    sigma_0 = hw.sigma_0_mm
    sigma_theta = hw.sigma_theta_rad
    d_perp = hw.d_perp_mm
    tilt = hw.laser_tilt_rad

    sigma_beam = np.sqrt(sigma_0 ** 2 + (sigma_theta * r_mm) ** 2)
    delta = np.abs(f * (d_perp - tilt * r_mm) / safe_r)
    sigma_img = (f / safe_r) * sigma_beam

    b = delta / sigma_img
    x = r_fs / sigma_img
    O = rice.cdf(x, b)
    O = np.where(r_mm < 1.0, 0.0, O)
    O = np.clip(O, 0.0, 1.0)
    return O.reshape(r_arr.shape).astype(float)


def load_empirical_overlap(
    file_path: Union[str, Path],
    r_m_target: np.ndarray,
) -> np.ndarray:
    """Load O(R) from a CSV/Excel file and interpolate onto a target range grid.

    File format requirements:
        - At least 2 columns: first = range (m), second = O(R) in [0, 1]
        - Optional header row (non-numeric is auto-skipped by pandas)
        - Additional columns are ignored

    Out-of-bounds behavior:
        - Below the source min range  → 0.0
        - Above the source max range  → 1.0  (assumes full overlap far away)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Overlap file not found: {file_path}")

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
            f"Unsupported overlap file extension: {suffix} (use .csv or .xlsx)"
        )

    if df.shape[1] < 2:
        raise ValueError(
            f"Overlap file must have at least 2 columns (range_m, O); got {df.shape[1]}"
        )

    r_src = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(float)
    O_src = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(float)
    mask = np.isfinite(r_src) & np.isfinite(O_src)
    r_src = r_src[mask]
    O_src = O_src[mask]
    if r_src.size < 2:
        raise ValueError("Overlap file has fewer than 2 valid (range, O) rows.")

    order = np.argsort(r_src)
    r_src = r_src[order]
    O_src = O_src[order]

    r_target = np.asarray(r_m_target, dtype=float)
    O_target = np.interp(r_target, r_src, O_src, left=0.0, right=1.0)
    return np.clip(O_target, 0.0, 1.0).astype(float)


def get_overlap(
    mode: str,
    r_m_target: np.ndarray,
    file_path: Optional[Union[str, Path]] = None,
    hardware: Optional[OverlapHardware] = None,
) -> Optional[np.ndarray]:
    """Top-level entry point used by the GUI / pipeline.

    Parameters
    ----------
    mode : str
        One of: 'disabled' / 'analytical' (or starts with 'analytical') /
        'file' (or starts with 'load').
    r_m_target : ndarray
        Target range grid in meters (the NRB profile's range axis).
    file_path : str or Path, optional
        Path to CSV/Excel; required when mode is 'file' / 'load from file'.
    hardware : OverlapHardware, optional
        Hardware spec for analytical mode (defaults to NARIT TR40).

    Returns
    -------
    O : ndarray or None
        O(R) array same shape as r_m_target. None when mode is 'disabled'.
    """
    m = (mode or "").strip().lower()
    if m in ("", "disabled", "off", "none"):
        return None
    if m.startswith("analytical"):
        return compute_analytical_overlap(r_m_target, hardware)
    if m.startswith("load") or m == "file":
        if not file_path:
            raise ValueError("File path required when overlap mode is 'load from file'.")
        return load_empirical_overlap(file_path, r_m_target)
    raise ValueError(
        f"Unknown overlap mode: {mode!r}. "
        "Use 'disabled', 'analytical (from hardware)', or 'load from file'."
    )


# -----------------------------------------------------------------------------
# Self test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    r = np.linspace(1.0, 15000.0, 200)
    O = compute_analytical_overlap(r)
    print(f"Analytical overlap with NARIT defaults — {len(r)} points:")
    for r_check in (50, 75, 100, 125, 150, 200, 500, 1000):
        i = int(np.argmin(np.abs(r - r_check)))
        print(f"  R = {r[i]:7.1f} m   O(R) = {O[i]:.4f}")
