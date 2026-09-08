"""AERONET (Aerosol Robotic Network) AOD file loader and 532 nm interpolation.

Supports Version 3 Level 1.0 / 1.5 / 2.0 ".lev10 / .lev15 / .lev20" files
downloaded from https://aeronet.gsfc.nasa.gov/

For TR40 LiDAR (532 nm) validation, we need AOD at 532 nm. Many AERONET
instruments have a 532 nm channel directly; for those that do not, we
interpolate from neighbouring channels (typically 500 nm and 440 nm) using
the Ångström exponent.

References
----------
Holben B.N. et al. (1998), Remote Sens. Environ. 66(1).
Eck T.F. et al. (1999), JGR Atmospheres 104(D24).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

AERONET_MISSING = -999.0   # AERONET fill value for missing data


def _read_text(src) -> str:
    """Accept a path or a file-like object, return decoded text."""
    if hasattr(src, "read"):
        raw = src.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return raw
    return Path(src).read_text(encoding="utf-8", errors="replace")


def _find_header_index(lines) -> Optional[int]:
    """Locate the row that contains the AERONET column names."""
    for i, ln in enumerate(lines):
        low = ln.lower()
        # The header is the line that contains the date column descriptor;
        # also accept lines that *start* with AERONET_Site (V3 layout).
        if "date(dd:mm:yyyy)" in low or low.startswith("aeronet_site,date"):
            return i
    return None


def load_aeronet_lev(
    src: Union[str, Path, io.IOBase],
) -> pd.DataFrame:
    """
    Load an AERONET .lev10 / .lev15 / .lev20 AOD file.

    Returns a DataFrame with columns:
        Time            : pandas Timestamp (UTC)
        AOD_532nm       : direct measurement if available, otherwise
                          Ångström-interpolated from 500 nm and 440 nm
        AOD_500nm,
        AOD_440nm,
        AOD_675nm       : raw channels kept for reference (if present)
        Angstrom_440_675: Ångström exponent (returned by AERONET as
                          "440-675_Angstrom_Exponent" or computed locally)
        AOD_532_source  : "direct" or "angstrom_500_440" or "nan"
    """
    text = _read_text(src)
    lines = text.splitlines()
    hdr = _find_header_index(lines)
    if hdr is None:
        raise ValueError("AERONET column header (with 'Date(dd:mm:yyyy)') not found.")

    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), engine="python")
    df = df.replace(AERONET_MISSING, np.nan)

    # --- timestamp -----------------------------------------------------------
    date_col = next((c for c in df.columns if c.lower().startswith("date(")), None)
    time_col = next((c for c in df.columns if c.lower().startswith("time(")), None)
    if date_col is None or time_col is None:
        raise ValueError("AERONET file missing Date/Time columns.")
    df["Time"] = pd.to_datetime(
        df[date_col].astype(str) + " " + df[time_col].astype(str),
        format="%d:%m:%Y %H:%M:%S", errors="coerce",
    )
    df = df.dropna(subset=["Time"]).reset_index(drop=True)

    # --- pick AOD columns ----------------------------------------------------
    out = pd.DataFrame({"Time": df["Time"]})
    for wl in ("AOD_532nm", "AOD_500nm", "AOD_440nm", "AOD_675nm"):
        if wl in df.columns:
            out[wl] = pd.to_numeric(df[wl], errors="coerce")

    # Ångström exponent (440-675) if AERONET provides it
    ang_col = next((c for c in df.columns if "440-675_angstrom" in c.lower()), None)
    if ang_col:
        out["Angstrom_440_675"] = pd.to_numeric(df[ang_col], errors="coerce")

    # --- ensure AOD_532nm exists, interpolating if needed --------------------
    if "AOD_532nm" not in out.columns:
        out["AOD_532nm"] = np.nan
    src_label = np.where(out["AOD_532nm"].notna(), "direct", "nan")

    if "AOD_500nm" in out.columns and "AOD_440nm" in out.columns:
        a500 = out["AOD_500nm"].to_numpy(float)
        a440 = out["AOD_440nm"].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            angst = -np.log(a500 / a440) / np.log(500.0 / 440.0)
            a532  = a500 * (532.0 / 500.0) ** (-angst)
        cur = out["AOD_532nm"].to_numpy(float)
        need = ~np.isfinite(cur) & np.isfinite(a532)
        cur[need] = a532[need]
        out["AOD_532nm"] = cur
        src_label = np.where(need, "angstrom_500_440", src_label)
        if "Angstrom_440_500" not in out.columns:
            out["Angstrom_440_500"] = angst

    out["AOD_532_source"] = src_label
    return out


# ---------------------------------------------------------------------------
# Match AERONET AOD to Fernald AOD by time
# ---------------------------------------------------------------------------

def match_aeronet_to_fernald(
    aeronet_df: pd.DataFrame,
    fernald_aod_df: pd.DataFrame,
    tol_min: float = 15.0,
) -> pd.DataFrame:
    """
    Pair each Fernald timestamp with the nearest AERONET measurement.

    Parameters
    ----------
    aeronet_df     : output of load_aeronet_lev() (must have Time, AOD_532nm).
    fernald_aod_df : the AOD sheet from compute_fernald_from_nrb_df() — a
                     1-row DataFrame whose columns include metadata
                     {R_ref_m, lidar_ratio_aer_sr, method} plus one
                     timestamp column per profile.
    tol_min        : maximum allowed time-difference (minutes).

    Returns
    -------
    DataFrame with one row per Fernald profile:
        Time, AOD_Fernald_532, AOD_AERONET_532, dt_min, delta, ratio
    """
    if "AOD_532nm" not in aeronet_df.columns:
        raise ValueError("AERONET DataFrame is missing 'AOD_532nm'.")

    meta = {"R_ref_m", "lidar_ratio_aer_sr", "method"}
    ts_cols = [c for c in fernald_aod_df.columns if c not in meta]

    a_times = pd.to_datetime(aeronet_df["Time"]).reset_index(drop=True)
    a_aod   = pd.to_numeric(aeronet_df["AOD_532nm"], errors="coerce").reset_index(drop=True)

    rows = []
    for col in ts_cols:
        t_f = pd.to_datetime(col, errors="coerce")
        aod_f = pd.to_numeric(fernald_aod_df[col].iloc[0], errors="coerce")
        if pd.isna(t_f):
            continue
        dt_min = (a_times - t_f).dt.total_seconds().abs() / 60.0
        if dt_min.notna().sum() == 0:
            rows.append((t_f, aod_f, np.nan, np.nan)); continue
        j = int(dt_min.idxmin())
        if float(dt_min.iloc[j]) > float(tol_min):
            rows.append((t_f, aod_f, np.nan, float(dt_min.iloc[j])))
            continue
        rows.append((t_f, aod_f, float(a_aod.iloc[j]), float(dt_min.iloc[j])))

    df_cmp = pd.DataFrame(rows, columns=["Time", "AOD_Fernald_532", "AOD_AERONET_532", "dt_min"])
    df_cmp["delta"] = df_cmp["AOD_Fernald_532"] - df_cmp["AOD_AERONET_532"]
    df_cmp["ratio"] = df_cmp["AOD_Fernald_532"] / df_cmp["AOD_AERONET_532"]
    return df_cmp


def summary_stats(df_cmp: pd.DataFrame) -> dict:
    """Compute N / MAE / bias / RMSE / R² between Fernald and AERONET."""
    d = df_cmp["delta"].dropna().to_numpy(float)
    if d.size == 0:
        return {"n": 0, "MAE": np.nan, "bias": np.nan, "RMSE": np.nan, "R2": np.nan}
    f = df_cmp["AOD_Fernald_532"].to_numpy(float)
    a = df_cmp["AOD_AERONET_532"].to_numpy(float)
    m = np.isfinite(f) & np.isfinite(a)
    if m.sum() >= 2:
        ss_res = float(np.sum((f[m] - a[m]) ** 2))
        ss_tot = float(np.sum((a[m] - a[m].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        r2 = np.nan
    return {
        "n":   int(d.size),
        "MAE": float(np.mean(np.abs(d))),
        "bias": float(np.mean(d)),
        "RMSE": float(np.sqrt(np.mean(d ** 2))),
        "R2":  r2,
    }
