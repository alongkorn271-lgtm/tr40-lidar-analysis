"""Mini-MPL CSV reader helpers used by Step 1 (rmin-rmax builder).

Extracted from main.py so the Streamlit web app does not have to pull in
main.py's Tkinter imports — Streamlit Cloud's Linux Python has no system
Tcl/Tk and the import would crash at module load.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Column / row locations of the "Pbls" field inside Mini-MPL CSVs.
# (Sigma Mini-MPL writes PBL at fixed cell DI2 ≡ column 112, row 2.)
PBL_COL_DI_0 = 112   # 0-based index
PBL_ROW_2_0 = 1      # 0-based index

# Sigma Mini-MPL particle_type numeric codes (column "particle_type") mapped to
# the labels listed in the file's "particle_type_mapping" column.
MPL_PARTICLE_TYPE_MAP = {
    0: "WaterCloud", 1: "MixedCloud", 2: "Ice/dust/ash", 3: "Rain/Dust",
    4: "Molecular", 5: "CleanAerosol", 6: "PollutedAerosol", 7: "Undetected",
}

# Per-range MPL product columns (validation references for the TR40 products).
_MPL_PROFILE_COLS = (
    "copol_nrb", "crosspol_nrb", "depolarization_ratio",
    "extinction_coefficient", "mass_concentration", "particle_type",
)
# Per-profile scalar MPL columns.
_MPL_SCALAR_COLS = ("pbls", "lidar_ratio", "aod")

# Mini-MPL NetCDF (.nc) files store timestamps (filename + internal date/time
# fields) in UTC. The NARIT site is UTC+7, and the whole TR40 pipeline works in
# local time, so .nc timestamps are shifted by this many hours on read.
NC_TZ_SHIFT_HOURS = 7


def is_nc_path(path) -> bool:
    """True if `path` is a NetCDF (.nc) Mini-MPL file (vs. the CSV export)."""
    return str(path).lower().endswith(".nc")


# ---------------------------------------------------------------------------
# Small parse helpers (no tkinter)
# ---------------------------------------------------------------------------

def parse_hhmm_text(text: str, *, field_name: str = "Start time") -> Optional[Tuple[int, int]]:
    text = str(text or "").strip()
    if not text:
        return None
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    if not m:
        raise ValueError(f"{field_name} must be HH:MM, for example 09:00")
    hh = int(m.group(1)); mm = int(m.group(2))
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError(f"{field_name} must be HH:MM, for example 09:00")
    return hh, mm


def minutes_of_day(ts: pd.Timestamp) -> int:
    return int(ts.hour) * 60 + int(ts.minute)


# ---------------------------------------------------------------------------
# Step 1 helpers
# ---------------------------------------------------------------------------

def parse_ts_from_filename(name: str) -> Optional[pd.Timestamp]:
    """Extract a YYYYMMDDHHMM stamp from an MPL filename."""
    hits = re.findall(r"(\d{12})", name)
    if not hits:
        return None
    dt = pd.to_datetime(hits[-1], format="%Y%m%d%H%M", errors="coerce")
    return None if pd.isna(dt) else pd.Timestamp(dt)


def read_pbl_km(path: Path) -> float:
    """Read the 'Pbls' PBL height (km) from a Mini-MPL CSV."""
    try:
        dfh = pd.read_csv(path, sep=None, engine="python")
        for c in dfh.columns:
            if str(c).strip().lower() == "pbls":
                for r in range(min(5, len(dfh))):
                    v = pd.to_numeric(dfh.iloc[r][c], errors="coerce")
                    if pd.notna(v):
                        return float(v)
    except Exception:
        pass
    try:
        import csv
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            sample = f.read(8192); f.seek(0)
            dia = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
            rd = csv.reader(f, dia); next(rd, None); row2 = next(rd, None)
        if row2 and len(row2) > PBL_COL_DI_0:
            v = pd.to_numeric(row2[PBL_COL_DI_0], errors="coerce")
            if pd.notna(v):
                return float(v)
    except Exception:
        pass
    try:
        df = pd.read_csv(path, sep=None, engine="python", header=None)
        if df.shape[0] > PBL_ROW_2_0 and df.shape[1] > PBL_COL_DI_0:
            return float(pd.to_numeric(df.iat[PBL_ROW_2_0, PBL_COL_DI_0], errors="coerce"))
    except Exception:
        pass
    return float("nan")


def read_copol(path: Path, row_start: int = 0, row_end: int = 498):
    """Return (range_m, copol_nrb_normalised) read from a Mini-MPL CSV."""
    try:
        dfh = pd.read_csv(path, sep=None, engine="python")
        if "range_nrb" in dfh.columns and "copol_nrb" in dfh.columns:
            rng = pd.to_numeric(dfh["range_nrb"], errors="coerce").to_numpy(float)
            cop = pd.to_numeric(dfh["copol_nrb"], errors="coerce").to_numpy(float)
        else:
            raise KeyError
    except Exception:
        df0 = pd.read_csv(path, header=None, sep=None, engine="python")
        rng = pd.to_numeric(df0.iloc[:, 89], errors="coerce").to_numpy(float)
        cop = pd.to_numeric(df0.iloc[:, 90], errors="coerce").to_numpy(float)
    rng = rng[row_start:row_end] * 1000.0
    cop = cop[row_start:row_end]
    mx = np.nanmax(cop) if cop.size else float("nan")
    cop_n = cop / mx if (np.isfinite(mx) and mx != 0) else np.full_like(cop, float("nan"))
    return rng, cop_n


def collect_actual_files(
    folder: Path,
    date_text: str,
    start_time_text: str,
) -> Tuple[pd.Timestamp, List[Tuple[pd.Timestamp, Path]]]:
    """Collect MPL CSVs from `folder`, filter by date / start-time, return ordered list."""
    mapping: Dict[pd.Timestamp, Path] = {}
    for f in sorted(folder.glob("*")):
        ts = parse_ts_from_filename(f.name)
        if ts is None:
            continue
        if ts not in mapping:
            mapping[ts] = f
    if not mapping:
        raise ValueError("No MPL CSV files with YYYYMMDDHHMM timestamp found.")

    if str(date_text).strip():
        target_date = pd.to_datetime(date_text, errors="raise").normalize()
    else:
        unique_dates = sorted({pd.Timestamp(ts).normalize() for ts in mapping})
        if len(unique_dates) == 1:
            target_date = unique_dates[0]
        else:
            dates_txt = ", ".join(pd.Timestamp(d).strftime("%Y-%m-%d") for d in unique_dates[:5])
            raise ValueError(f"Multiple dates found in filenames ({dates_txt}). Please specify Date.")

    start_pair = parse_hhmm_text(start_time_text, field_name="Start time")
    start_min_total = None if start_pair is None else start_pair[0] * 60 + start_pair[1]

    ordered = []
    for ts, f in sorted(mapping.items(), key=lambda kv: kv[0]):
        if pd.Timestamp(ts).normalize() != target_date:
            continue
        if start_min_total is not None and minutes_of_day(pd.Timestamp(ts)) < start_min_total:
            continue
        ordered.append((pd.Timestamp(ts), f))

    if not ordered:
        raise ValueError("No MPL CSV files found for the selected date/start time.")
    return target_date, ordered


# ---------------------------------------------------------------------------
# Full MPL product extraction (validation references for TR40)
# ---------------------------------------------------------------------------

def read_mpl_products(path: Path, row_start: int = 0, row_end: int = 498) -> Dict[str, object]:
    """
    Read ALL Sigma Mini-MPL products from one CSV (one profile / time).

    Handles BOTH header layouts automatically:
      • raw export  — single header row 0 (135 cols incl. weather/snr/aeronet)
      • grouped/edited export — row 0 = "Group 1/2/3" labels, row 1 = names
    Values are RAW (not normalised) so they can be used directly as validation
    ground truth for the TR40 products.

    Returns dict:
      range_m : range axis [m] (range_nrb × 1000)
      copol_nrb, crosspol_nrb, depolarization_ratio,
      extinction_coefficient, mass_concentration, particle_type : 1-D arrays
      pbls, lidar_ratio, aod, aeronet_aod, laser_energy : per-profile scalars
    """
    if is_nc_path(path):
        return read_mpl_products_nc(path, row_start, row_end)
    path = Path(path)
    first = path.read_text(errors="replace").splitlines()[0] if path.exists() else ""
    skip = 1 if first.lstrip().lower().startswith("group") else 0
    df = pd.read_csv(path, skiprows=skip)

    # Range grid: the column named range_nrb (duplicates auto-suffixed by pandas).
    rcol = "range_nrb" if "range_nrb" in df.columns else df.columns[min(6, df.shape[1]-1)]
    r_km = pd.to_numeric(df[rcol], errors="coerce").to_numpy(float)

    def _grab(name: str) -> np.ndarray:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").to_numpy(float)
        return np.full(r_km.shape, np.nan)

    sl = slice(row_start, row_end)
    out: Dict[str, object] = {"range_m": (r_km[sl] * 1000.0)}
    for c in _MPL_PROFILE_COLS:
        out[c] = _grab(c)[sl]

    # Scalars: first finite value in the column (raw export carries extras).
    for c in (*_MPL_SCALAR_COLS, "aeronet_aod", "laser_energy"):
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            out[c] = float(s.iloc[0]) if len(s) else float("nan")
        else:
            out[c] = float("nan")
    return out


def read_mpl_raw(path: Path, row_start: int = 0, row_end: int = 498) -> Dict[str, object]:
    """Read the Mini-MPL RAW co/cross counts (Group 1) on the range_raw grid.

    Kept separate from read_mpl_products because the raw signals live on the
    `range_raw` grid (starts ~30 m), NOT the `range_nrb` grid (~120 m) used by
    the NRB products. Reads .nc directly when given one. Returns range_m (from range_raw ×1000), copol_raw, crosspol_raw.
    """
    if is_nc_path(path):
        return read_mpl_raw_nc(path, row_start, row_end)
    path = Path(path)
    first = path.read_text(errors="replace").splitlines()[0] if path.exists() else ""
    skip = 1 if first.lstrip().lower().startswith("group") else 0
    df = pd.read_csv(path, skiprows=skip)
    rcol = "range_raw" if "range_raw" in df.columns else df.columns[0]
    r_km = pd.to_numeric(df[rcol], errors="coerce").to_numpy(float)

    def _grab(name: str) -> np.ndarray:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").to_numpy(float)
        return np.full(r_km.shape, np.nan)

    sl = slice(row_start, row_end)
    return {
        "range_m": r_km[sl] * 1000.0,
        "copol_raw": _grab("copol_raw")[sl],
        "crosspol_raw": _grab("crosspol_raw")[sl],
    }


# ---------------------------------------------------------------------------
# NetCDF (.nc) native readers  —  Mini-MPL raw .nc export
# ---------------------------------------------------------------------------
# The .nc export holds every product the CSV export does, under the SAME variable
# names, but split across three range grids:
#   range_raw (~30 m step, ~1000 bins) : copol_raw, crosspol_raw
#   range_nrb (~120 m start, ~664 bins): copol_nrb, crosspol_nrb,
#                                         depolarization_ratio, particle_type
#   range_vbp (shares range_nrb's start, ~197 bins): extinction_coefficient,
#                                         mass_concentration, VBP
# range_vbp is just a truncated range_nrb, so the vbp products are interpolated
# back onto range_nrb — this keeps Step 1's single-grid assumption intact.

def _nc_1d(ds, name: str, n_fallback: int) -> np.ndarray:
    """A data variable flattened to 1-D float (NaN-filled if absent)."""
    if name in ds:
        return np.asarray(ds[name].values, float).ravel()
    return np.full(n_fallback, np.nan)


def _nc_scalar(ds, name: str) -> float:
    """First finite value of a per-time variable (NaN if absent/empty)."""
    if name in ds:
        v = np.asarray(ds[name].values, float).ravel()
        v = v[np.isfinite(v)]
        if v.size:
            return float(v[0])
    return float("nan")


def read_pbl_km_nc(path, *_a, **_kw) -> float:
    """PBL height (km) from the .nc `pbls` variable (first finite = primary PBL)."""
    import xarray as xr
    with xr.open_dataset(path) as ds:
        return _nc_scalar(ds, "pbls")


def read_copol_nc(path, row_start: int = 0, row_end: int = 498):
    """Return (range_m, copol_nrb_normalised) from a Mini-MPL .nc — mirrors read_copol."""
    import xarray as xr
    with xr.open_dataset(path) as ds:
        rng = _nc_1d(ds, "range_nrb", 0)
        cop = _nc_1d(ds, "copol_nrb", rng.size)
    rng = rng[row_start:row_end] * 1000.0
    cop = cop[row_start:row_end]
    mx = np.nanmax(cop) if cop.size else float("nan")
    cop_n = cop / mx if (np.isfinite(mx) and mx != 0) else np.full_like(cop, float("nan"))
    return rng, cop_n


def read_mpl_products_nc(path, row_start: int = 0, row_end: int = 498) -> Dict[str, object]:
    """Read all Mini-MPL products from a .nc file — mirrors read_mpl_products.

    extinction_coefficient / mass_concentration live on range_vbp and are
    interpolated onto range_nrb so every product shares one range axis."""
    import xarray as xr
    with xr.open_dataset(path) as ds:
        r_nrb = _nc_1d(ds, "range_nrb", 0)            # km
        n = r_nrb.size
        r_vbp = _nc_1d(ds, "range_vbp", n)            # km (subset of r_nrb)

        def on_nrb(name):   # variable already on the range_nrb grid
            return _nc_1d(ds, name, n)

        def vbp_on_nrb(name):   # range_vbp variable → interp to range_nrb
            a = _nc_1d(ds, name, r_vbp.size)
            if not np.isfinite(r_vbp).any() or not np.isfinite(a).any():
                return np.full(n, np.nan)
            return np.interp(r_nrb, r_vbp, a, left=np.nan, right=np.nan)

        sl = slice(row_start, row_end)
        out: Dict[str, object] = {"range_m": (r_nrb * 1000.0)[sl]}
        out["copol_nrb"]            = on_nrb("copol_nrb")[sl]
        out["crosspol_nrb"]         = on_nrb("crosspol_nrb")[sl]
        out["depolarization_ratio"] = on_nrb("depolarization_ratio")[sl]
        out["particle_type"]        = on_nrb("particle_type")[sl]
        out["extinction_coefficient"] = vbp_on_nrb("extinction_coefficient")[sl]
        out["mass_concentration"]     = vbp_on_nrb("mass_concentration")[sl]
        for c in (*_MPL_SCALAR_COLS, "aeronet_aod", "laser_energy"):
            out[c] = _nc_scalar(ds, c)
    return out


def read_mpl_raw_nc(path, row_start: int = 0, row_end: int = 498) -> Dict[str, object]:
    """Read raw co/cross counts from a .nc file — mirrors read_mpl_raw."""
    import xarray as xr
    with xr.open_dataset(path) as ds:
        r_raw = _nc_1d(ds, "range_raw", 0)            # km
        n = r_raw.size
        sl = slice(row_start, row_end)
        return {
            "range_m": (r_raw * 1000.0)[sl],
            "copol_raw": _nc_1d(ds, "copol_raw", n)[sl],
            "crosspol_raw": _nc_1d(ds, "crosspol_raw", n)[sl],
        }
