"""Mini-MPL CSV reader helpers used by Step 1 (rmin-rmax builder).

Extracted from main.py so the Streamlit web app does not have to pull in
main.py's Tkinter imports — Streamlit Cloud's Linux Python has no system
Tcl/Tk and the import would crash at module load.
"""
from __future__ import annotations

import re
from contextlib import contextmanager
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
# copol_snr / crosspol_snr are native to the range_raw grid and are resampled
# onto range_nrb here so every profile column shares one range axis — they are
# what lets a comparison be restricted to the range MPL itself can see.
_MPL_PROFILE_COLS = (
    "copol_nrb", "crosspol_nrb", "depolarization_ratio",
    "extinction_coefficient", "mass_concentration", "particle_type",
    "copol_snr", "crosspol_snr",
)
# Per-profile scalar MPL columns.
_MPL_SCALAR_COLS = ("pbls", "lidar_ratio", "aod")

# Ambient weather. NOTE: when no weather station is attached the Mini-MPL writes
# the sentinel -32768 into every field except barometric pressure — measured on
# the 2026-09-08/09 batches, outside temperature / humidity / wind / dew point
# are ALL sentinel, only pressure is real. Sentinels are converted to NaN on
# read, so check for finite values before relying on any of these.
_MPL_WEATHER_COLS = (
    "weather_outside_temperature", "weather_outside_humidity",
    "weather_barometric_pressure", "weather_wind_speed",
    "weather_wind_direction", "weather_dew_point", "weather_rain_rate",
)
# Instrument housekeeping: background level (MPL's own estimate, comparable to
# our pre-trigger floor), energy monitor, temperatures, pointing.
_MPL_HOUSEKEEPING_COLS = (
    "copol_background", "crosspol_background", "laser_energy",
    "telescope_temperature", "detector_temperature", "laser_temperature",
    "elevation_angle", "azimuth_angle",
)

# Sigma's "no reading" fill values (weather fields, occasionally others). Both
# signs of the 16-bit rail appear: -32768 for temperature/humidity/wind speed,
# +32767 for wind direction.
_MPL_MISSING_SENTINELS = (-32768.0, -32767.0, 32767.0, 32768.0, -9999.0, -999.0)


def _clean_missing(value: float) -> float:
    """Sentinel fill value -> NaN (a real reading passes through untouched)."""
    v = float(value)
    if not np.isfinite(v):
        return float("nan")
    return float("nan") if any(abs(v - s) < 1e-6 for s in _MPL_MISSING_SENTINELS) else v

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
    # NOTE (CSV path only): the flat export lays every variable out side by side,
    # so columns native to another grid (extinction/mass on range_vbp, the SNRs on
    # range_raw) are taken row-for-row and are only approximately aligned to
    # range_nrb. The .nc path resamples them properly — prefer .nc for these.
    for c in _MPL_PROFILE_COLS:
        out[c] = _grab(c)[sl]

    # Scalars: first finite value in the column (raw export carries extras).
    for c in (*_MPL_SCALAR_COLS, "aeronet_aod",
              *_MPL_WEATHER_COLS, *_MPL_HOUSEKEEPING_COLS):
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            out[c] = _clean_missing(s.iloc[0]) if len(s) else float("nan")
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

class _NcVars:
    """Read-only ``name -> float ndarray`` view of an open NetCDF file.

    Masked/fill values come back as NaN so callers never see sentinel numbers,
    and the interface is the small subset (``in`` and ``[]``) the readers below
    need — which is what lets the netCDF4 and xarray backends be interchangeable.
    """

    __slots__ = ("_get", "_names")

    def __init__(self, getter, names):
        self._get = getter
        self._names = frozenset(names)

    def __contains__(self, name: str) -> bool:
        return name in self._names

    def __getitem__(self, name: str) -> np.ndarray:
        return self._get(name)


@contextmanager
def _open_nc(path):
    """Open a Mini-MPL .nc and yield a :class:`_NcVars` view.

    Uses **netCDF4 directly**: on this dataset ``xarray.open_dataset`` measures
    ~9.7 s per file against ~9 ms for netCDF4 — a ~1000x difference that turns a
    day of 5-minute profiles into a ~40 minute read. xarray is kept as a fallback
    for environments where netCDF4 is not installed.
    """
    try:
        import netCDF4  # noqa: F401
    except ImportError:
        import xarray as xr
        with xr.open_dataset(path) as ds:
            yield _NcVars(lambda n: _as_float_array(ds[n].values), set(ds.variables))
        return

    import netCDF4
    ds = netCDF4.Dataset(path)
    try:
        yield _NcVars(lambda n: _as_float_array(ds.variables[n][:]), set(ds.variables))
    finally:
        ds.close()


def _as_float_array(values) -> np.ndarray:
    """Any NetCDF variable payload -> float ndarray with fills as NaN."""
    arr = np.ma.asarray(values).astype(float)
    return np.ma.filled(arr, np.nan)


def _nc_1d(ds, name: str, n_fallback: int) -> np.ndarray:
    """A data variable flattened to 1-D float (NaN-filled if absent)."""
    if name in ds:
        return ds[name].ravel()
    return np.full(n_fallback, np.nan)


def _nc_scalar(ds, name: str) -> float:
    """First finite value of a per-time variable (NaN if absent/empty)."""
    if name in ds:
        v = ds[name].ravel()
        v = v[np.isfinite(v)]
        if v.size:
            return float(v[0])
    return float("nan")


def read_pbl_km_nc(path, *_a, **_kw) -> float:
    """PBL height (km) from the .nc `pbls` variable (first finite = primary PBL)."""
    with _open_nc(path) as ds:
        return _nc_scalar(ds, "pbls")


def read_copol_nc(path, row_start: int = 0, row_end: int = 498):
    """Return (range_m, copol_nrb_normalised) from a Mini-MPL .nc — mirrors read_copol."""
    with _open_nc(path) as ds:
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
    with _open_nc(path) as ds:
        r_nrb = _nc_1d(ds, "range_nrb", 0)            # km
        n = r_nrb.size
        r_vbp = _nc_1d(ds, "range_vbp", n)            # km (subset of r_nrb)
        r_raw = _nc_1d(ds, "range_raw", n)            # km (SNR lives here)

        def on_nrb(name):   # variable already on the range_nrb grid
            return _nc_1d(ds, name, n)

        def _regrid(name, r_src):   # variable on another grid → interp to range_nrb
            a = _nc_1d(ds, name, r_src.size)
            if (not np.isfinite(r_src).any() or not np.isfinite(a).any()
                    or a.size != r_src.size):
                return np.full(n, np.nan)
            return np.interp(r_nrb, r_src, a, left=np.nan, right=np.nan)

        def vbp_on_nrb(name):
            return _regrid(name, r_vbp)

        def raw_on_nrb(name):
            return _regrid(name, r_raw)

        sl = slice(row_start, row_end)
        out: Dict[str, object] = {"range_m": (r_nrb * 1000.0)[sl]}
        out["copol_nrb"]            = on_nrb("copol_nrb")[sl]
        out["crosspol_nrb"]         = on_nrb("crosspol_nrb")[sl]
        out["depolarization_ratio"] = on_nrb("depolarization_ratio")[sl]
        out["particle_type"]        = on_nrb("particle_type")[sl]
        out["extinction_coefficient"] = vbp_on_nrb("extinction_coefficient")[sl]
        out["mass_concentration"]     = vbp_on_nrb("mass_concentration")[sl]
        # SNR is reported on range_raw; resample so it indexes the same rows as
        # the NRB products and can be used directly as a validity mask.
        out["copol_snr"]            = raw_on_nrb("copol_snr")[sl]
        out["crosspol_snr"]         = raw_on_nrb("crosspol_snr")[sl]
        for c in (*_MPL_SCALAR_COLS, "aeronet_aod",
                  *_MPL_WEATHER_COLS, *_MPL_HOUSEKEEPING_COLS):
            out[c] = _clean_missing(_nc_scalar(ds, c))
        # Layer geometry from the same open handle — opening a .nc is the
        # expensive part of Step 1, so this saves a whole extra pass per file.
        out.update(_layers_from_ds(ds))
    return out


# ---------------------------------------------------------------------------
# Layer geometry: every PBL candidate + every detected cloud
# ---------------------------------------------------------------------------
# Step 1 historically kept only the primary PBL (`pbls[0]`). The file actually
# carries up to 10 PBL candidates and up to 10 cloud structures, each described
# by an "outline" of boundary heights (km): the first is the base, the last
# finite one is the top. Observed shapes: [base, top, nan] for a simple cloud
# and [base, mid, top] for a structured one. These are the reference the
# depolarization cloud screen (delta >= ice threshold) is checked against, and
# the layer heights the ALT result is compared to.

_EMPTY_LAYERS = {"pbl_km": np.array([]), "cloud_base_km": np.array([]),
                 "cloud_top_km": np.array([]), "n_clouds": 0}


def _layers_from_ds(ds) -> Dict[str, object]:
    """Layer geometry from an already-open dataset view (see read_mpl_layers)."""
    pbl = ds["pbls"].ravel() if "pbls" in ds else np.array([])
    if "clouds" not in ds:
        return {**_EMPTY_LAYERS, "pbl_km": pbl}
    cl = ds["clouds"]
    # (time, cloud, outline) -> (cloud, outline); one profile per file.
    cl = cl.reshape(-1, cl.shape[-1]) if cl.ndim >= 2 else cl.reshape(1, -1)
    bases, tops = [], []
    for row in cl:
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            continue
        bases.append(float(finite[0]))
        tops.append(float(finite[-1]))
    return {
        "pbl_km": pbl,
        "cloud_base_km": np.asarray(bases, float),
        "cloud_top_km": np.asarray(tops, float),
        "n_clouds": len(bases),
    }


def read_mpl_layers(path) -> Dict[str, object]:
    """PBL candidates and cloud boundaries from one Mini-MPL profile.

    Returns
      pbl_km        : 1-D array of PBL candidate heights [km]
      cloud_base_km : 1-D array of cloud base heights [km], one per cloud
      cloud_top_km  : 1-D array of cloud top heights [km], one per cloud
      n_clouds      : number of clouds with a finite base
    Arrays are ordered as the file stores them (lowest first). Non-.nc inputs
    return empty arrays — the CSV export does not carry the outlines.

    ``read_mpl_products`` already returns these keys for a .nc input; prefer
    that when you need the products too, so the file is opened only once.
    """
    if not is_nc_path(path):
        return dict(_EMPTY_LAYERS)
    with _open_nc(path) as ds:
        return _layers_from_ds(ds)


def read_mpl_raw_nc(path, row_start: int = 0, row_end: int = 498) -> Dict[str, object]:
    """Read raw co/cross counts from a .nc file — mirrors read_mpl_raw."""
    with _open_nc(path) as ds:
        r_raw = _nc_1d(ds, "range_raw", 0)            # km
        n = r_raw.size
        sl = slice(row_start, row_end)
        return {
            "range_m": (r_raw * 1000.0)[sl],
            "copol_raw": _nc_1d(ds, "copol_raw", n)[sl],
            "crosspol_raw": _nc_1d(ds, "crosspol_raw", n)[sl],
        }
