"""Validation engine — quantify how well a TR40 prototype profile reproduces the
Mini-MPL reference, using the metrics the lidar intercomparison literature uses.

Scope and intent
----------------
The Mini-MPL is a **temporary referee**, not a permanent input: the goal is to
establish that the prototype can stand on its own, after which MPL drops out of
the workflow entirely. So every metric here answers "is the prototype right?",
never "what does MPL say the answer is".

Comparison window
-----------------
Work is done on **real range with the pre-trigger already removed** — the
``Range(m)`` axis the NRB/depol workbooks carry (3.75 m .. or 30 m ..), NOT bin
indices, which differ 8x between a 3.75 m and a 30 m acquisition for the same
distance. The default window is 300 m (above the incomplete-overlap near field)
to 5000 m (above that is no longer boundary layer, and the Mini-MPL's own SNR
rarely reaches past ~4 km anyway).

Metrics (Matthias et al. 2004; Boeckmann et al. 2004; Wandinger et al. 2016)
----------------------------------------------------------------------------
1. Relative difference profile  d(z) = (proto/k - MPL)/MPL * 100 %
2. Ratio constancy              ratio(z) = proto/MPL, reported as CV
3. Regression                   slope / intercept / r^2 of proto vs MPL
4. Layer height                 ALT vs PBL bias and RMSE
5. Depolarization               our delta_v vs the MPL ratio, definitions aligned

On the calibration constant
---------------------------
Prototype NRB and Mini-MPL NRB are in different units (their ratio is ~4e11), so
an absolute difference is meaningless. Each profile therefore gets ONE scalar
k = median(proto/MPL) over the window, and the metrics compare **shape** after
dividing it out. That is the intercomparison convention: a constant ratio means
the two instruments see the same atmosphere and differ only by calibration.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Default comparison window, in real range (pre-trigger already removed).
COMPARE_RMIN_M = 300.0
COMPARE_RMAX_M = 5000.0
# Minimum SNR for a bin to take part, applied to BOTH instruments when each has
# an SNR profile available.
COMPARE_SNR_MIN = 3.0
# Nearest-time matching tolerance between a prototype and an MPL profile.
MATCH_TOLERANCE_MIN = 5.0
# Fallback day/night split by local hour, used only when a workbook carries no
# QC sheet to classify from the signal itself.
NIGHT_START_HOUR = 18
NIGHT_END_HOUR = 6
# Profiles accumulated from fewer shots are aborted/partial acquisitions (the
# short files at the start of a run) and are dropped before comparison.
MIN_SHOTS = 1500
# Signal-based day/night: a profile is DAY when its background is high, or when
# the analog/photon glue could not be fitted (both are what sunlight does).
# 10 MHz is the same threshold Step 2 uses to switch to its daytime glue.
DAY_BG_THRESHOLD_MHZ = 10.0
DAY_GLUE_R2_MIN = 0.75
# Tolerance bands drawn on the relative-difference figure.
TOLERANCE_BANDS_PCT = (10.0, 20.0)


# ---------------------------------------------------------------------------
# Sheet loading
# ---------------------------------------------------------------------------

def _to_timestamp(col) -> Optional[pd.Timestamp]:
    """A sheet column header -> Timestamp. Handles real datetimes (prototype
    workbooks) and bare 'HH:MM' strings (MPL workbooks, date-less)."""
    if isinstance(col, pd.Timestamp):
        return col
    txt = str(col).strip()
    if not txt or txt.lower().startswith("range"):
        return None
    ts = pd.to_datetime(txt, errors="coerce")
    return None if pd.isna(ts) else pd.Timestamp(ts)


def _read_range_sheet(xl: pd.ExcelFile, sheet: str) -> Optional[Dict]:
    """Read a Range(m) x time sheet -> {'range_m': arr, 'cols': {ts: arr}}."""
    if sheet not in xl.sheet_names:
        return None
    df = xl.parse(sheet)
    if df.empty or len(df.columns) < 2:
        return None
    r = pd.to_numeric(df[df.columns[0]], errors="coerce").to_numpy(float)
    cols: Dict[pd.Timestamp, np.ndarray] = {}
    for c in df.columns[1:]:
        ts = _to_timestamp(c)
        if ts is not None:
            cols[ts] = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
    return {"range_m": r, "cols": cols} if cols else None


def _first_sheet(xl: pd.ExcelFile, names: Tuple[str, ...]) -> Optional[Dict]:
    for n in names:
        got = _read_range_sheet(xl, n)
        if got is not None:
            return got
    return None


def _read_qc(xl: pd.ExcelFile) -> Dict:
    """Per-profile QC from Step 2's ``QC_calibration`` sheet, keyed by time.

    Returns {Timestamp: {'shots', 'bg_mhz', 'glue_r2'}}; values are NaN when a
    column is absent (``shots`` only exists in workbooks written after it was
    added to Step 2).
    """
    if "QC_calibration" not in xl.sheet_names:
        return {}
    q = xl.parse("QC_calibration")
    if "time" not in q.columns:
        return {}

    def col(name):
        return (pd.to_numeric(q[name], errors="coerce") if name in q.columns
                else pd.Series(np.nan, index=q.index))

    shots, bg, r2 = col("shots"), col("bg_par_mhz"), col("par_fit_quality_r2")
    # A profile glued with the night's gain keeps the r2 of that forced line
    # against its own (cloud-contaminated) bins, which says nothing about day or
    # night -- ignore it there, as for any skipped row.
    mode = (q["par_glue_fit_mode"].astype(str) if "par_glue_fit_mode" in q.columns
            else pd.Series("", index=q.index))
    status = q["status"].astype(str) if "status" in q.columns else pd.Series("ok", index=q.index)
    out = {}
    for i, t in enumerate(pd.to_datetime(q["time"], errors="coerce")):
        if pd.notna(t):
            r2_i = float(r2.iloc[i])
            if mode.iloc[i] == "night_gain_override":
                r2_i = float("nan")
            row = {"shots": float(shots.iloc[i]), "bg_mhz": float(bg.iloc[i]), "glue_r2": r2_i}
            # Files Step 2 skipped (too few shots) have no profile column; their
            # rows sit seconds from the real profile and must never shadow it.
            if status.iloc[i].startswith("skipped"):
                continue
            out[pd.Timestamp(t)] = row
    return out


def _qc_for(qc: Dict, ts, tolerance_s: float = 90.0) -> Optional[Dict]:
    """QC row for a profile time: exact match, else the nearest within tolerance."""
    if not qc:
        return None
    t = pd.Timestamp(ts)
    if t in qc:
        return qc[t]
    best = min(qc, key=lambda k: abs((k - t).total_seconds()))
    return qc[best] if abs((best - t).total_seconds()) <= tolerance_s else None


def classify_day_night(ts, qc_row: Optional[Dict],
                       bg_threshold_mhz: float = DAY_BG_THRESHOLD_MHZ,
                       glue_r2_min: float = DAY_GLUE_R2_MIN) -> Tuple[str, str]:
    """('day' | 'night', reason) for one profile.

    Decided from the signal when QC is available: day if the background is at
    or above ``bg_threshold_mhz``, or if a glue fit exists and its r^2 is below
    ``glue_r2_min``. The background carries most of the weight — it separates
    night (~0.1 MHz) from day (60-250 MHz) by orders of magnitude, while a good
    glue fit is still possible in the afternoon and a photon-only run has no
    fit at all. Without QC the clock is used.
    """
    bg = qc_row.get("bg_mhz", np.nan) if qc_row else np.nan
    r2 = qc_row.get("glue_r2", np.nan) if qc_row else np.nan
    if np.isfinite(bg):
        if bg >= bg_threshold_mhz:
            return "day", f"BG {bg:.1f} MHz >= {bg_threshold_mhz:g}"
        if np.isfinite(r2) and r2 < glue_r2_min:
            return "day", f"glue r2 {r2:.2f} < {glue_r2_min:g}"
        r2_txt = f"{r2:.2f}" if np.isfinite(r2) else "n/a"
        return "night", f"BG {bg:.2f} MHz, glue r2 {r2_txt}"
    return ("night" if is_night(ts) else "day"), "clock (no QC)"


def load_prototype(path) -> Dict:
    """Load a prototype workbook (Step 2 NRB or the depolarization workbook).

    Prefers the physical co-polar NRB (``NRB_co``) over the normalised
    ``NRB profile`` so the ratio against MPL carries a real calibration
    constant rather than a per-profile rescaling.
    """
    xl = pd.ExcelFile(path)
    nrb = _first_sheet(xl, ("NRB_co", "NRB profile"))
    if nrb is None:
        raise ValueError(
            f"{Path(path).name}: no 'NRB_co' or 'NRB profile' sheet — is this a "
            f"prototype NRB/depolarization workbook?")
    depol = _read_range_sheet(xl, "Depol_delta_v") or {}
    return {
        "label": Path(path).stem,
        "range_m": nrb["range_m"],
        "nrb": nrb["cols"],
        "snr": (_first_sheet(xl, ("SNR_co", "SNR")) or {}).get("cols", {}),
        "depol": depol.get("cols", {}),
        "depol_range_m": depol.get("range_m", nrb["range_m"]),
        "qc": _read_qc(xl),
    }


def load_mpl(path) -> Dict:
    """Load an MPL reference workbook written by Step 1.

    ``MPL_copol_snr`` and ``MPL_clouds`` are only present in workbooks built by
    the extended Step 1 ingest; older workbooks simply compare without an
    MPL-side SNR mask (the window cap then does the work on its own).
    """
    xl = pd.ExcelFile(path)
    nrb = _first_sheet(xl, ("MPL_copol_nrb", "copol_nrb_norm"))
    if nrb is None:
        raise ValueError(
            f"{Path(path).name}: no 'MPL_copol_nrb' sheet — is this a Step 1 "
            f"MPL workbook?")
    pbl: Dict[pd.Timestamp, float] = {}
    if "rmin-rmax" in xl.sheet_names:
        rr = xl.parse("rmin-rmax")
        tcol = next((c for c in rr.columns if str(c).strip().lower() == "time"), None)
        pcol = next((c for c in rr.columns
                     if "pbl" in str(c).lower() and "mpl" in str(c).lower()), None)
        if tcol is not None and pcol is not None:
            for t, v in zip(pd.to_datetime(rr[tcol], errors="coerce"),
                            pd.to_numeric(rr[pcol], errors="coerce")):
                if pd.notna(t):
                    pbl[pd.Timestamp(t)] = float(v)
    return {
        "label": Path(path).stem,
        "range_m": nrb["range_m"],
        "nrb": nrb["cols"],
        "snr": (_read_range_sheet(xl, "MPL_copol_snr") or {}).get("cols", {}),
        "depol": (_read_range_sheet(xl, "MPL_depol") or {}).get("cols", {}),
        "pbl_m": pbl,
        "clouds": xl.parse("MPL_clouds") if "MPL_clouds" in xl.sheet_names else None,
    }


# ---------------------------------------------------------------------------
# Grid alignment and profile matching
# ---------------------------------------------------------------------------

def resample_to_grid(r_src: np.ndarray, y_src: np.ndarray,
                     r_dst: np.ndarray) -> np.ndarray:
    """Average ``y_src`` into the bins of the (coarser) ``r_dst`` grid.

    Block averaging, not interpolation: a 3.75 m profile resampled onto the
    Mini-MPL's 30 m grid averages 8 source bins per destination bin, which
    *improves* its noise by ~sqrt(8) instead of discarding 7 of every 8 samples.
    Destination bins with no source sample come back NaN.
    """
    r_src = np.asarray(r_src, float)
    y_src = np.asarray(y_src, float)
    r_dst = np.asarray(r_dst, float)
    if r_dst.size < 2:
        return np.full(r_dst.shape, np.nan)

    step = float(np.median(np.diff(r_dst)))
    edges = np.concatenate(([r_dst[0] - step / 2.0], r_dst + step / 2.0))
    idx = np.digitize(r_src, edges) - 1
    ok = (idx >= 0) & (idx < r_dst.size) & np.isfinite(y_src)
    if not ok.any():
        return np.full(r_dst.shape, np.nan)

    total = np.bincount(idx[ok], weights=y_src[ok], minlength=r_dst.size)
    count = np.bincount(idx[ok], minlength=r_dst.size)
    out = np.full(r_dst.size, np.nan)
    nz = count > 0
    out[nz] = total[nz] / count[nz]
    return out


def is_night(ts) -> bool:
    h = int(pd.Timestamp(ts).hour)
    return h >= NIGHT_START_HOUR or h < NIGHT_END_HOUR


def match_profiles(proto_ts: List, mpl_ts: List,
                   tolerance_min: float = MATCH_TOLERANCE_MIN,
                   ) -> List[Tuple]:
    """Pair each prototype profile with the nearest MPL profile in time.

    MPL columns often carry only HH:MM (no date), so matching is done on
    time-of-day; a pair more than ``tolerance_min`` apart is dropped.
    """
    if not proto_ts or not mpl_ts:
        return []
    mpl_minutes = np.array([pd.Timestamp(t).hour * 60 + pd.Timestamp(t).minute
                            + pd.Timestamp(t).second / 60.0 for t in mpl_ts])
    out = []
    for pt in proto_ts:
        p = pd.Timestamp(pt)
        pm = p.hour * 60 + p.minute + p.second / 60.0
        # Compare across midnight as well as within the day.
        d = np.abs(mpl_minutes - pm)
        d = np.minimum(d, 1440.0 - d)
        j = int(np.argmin(d))
        if d[j] <= tolerance_min:
            out.append((pt, mpl_ts[j], float(d[j])))
    return out


# ---------------------------------------------------------------------------
# Per-profile metrics
# ---------------------------------------------------------------------------

def _safe_cv_pct(values: np.ndarray) -> float:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan")
    m = float(np.nanmedian(v))
    return float("nan") if m == 0 else abs(float(np.nanstd(v)) / m) * 100.0


def compare_one(r_grid: np.ndarray, proto: np.ndarray, mpl: np.ndarray,
                mask: np.ndarray) -> Dict[str, object]:
    """All shape metrics for one matched profile pair, on a shared grid.

    ``proto`` is divided by a single calibration constant k = median(proto/MPL)
    before the difference is taken, so what is measured is shape agreement.
    """
    nan_result = {
        "n_bins": 0, "k_calibration": np.nan, "ratio_cv_pct": np.nan,
        "reldiff_median_pct": np.nan, "reldiff_rms_pct": np.nan,
        "within_10pct": np.nan, "within_20pct": np.nan,
        "slope": np.nan, "intercept": np.nan, "r2": np.nan,
        "ratio_profile": np.full(r_grid.shape, np.nan),
        "reldiff_profile": np.full(r_grid.shape, np.nan),
        "proto_scaled_profile": np.full(r_grid.shape, np.nan),
        "mpl_profile": np.full(r_grid.shape, np.nan),
    }
    use = mask & np.isfinite(proto) & np.isfinite(mpl) & (np.abs(mpl) > 0)
    if use.sum() < 3:
        return nan_result

    ratio_full = np.full(r_grid.shape, np.nan)
    ratio_full[use] = proto[use] / mpl[use]
    k = float(np.nanmedian(ratio_full[use]))
    if not np.isfinite(k) or k == 0:
        return nan_result

    scaled = proto / k
    reldiff = np.full(r_grid.shape, np.nan)
    reldiff[use] = (scaled[use] - mpl[use]) / mpl[use] * 100.0
    d = reldiff[use]

    # Regression of the calibration-removed prototype against MPL.
    x, y = mpl[use], scaled[use]
    slope = intercept = r2 = float("nan")
    if x.size >= 3 and np.nanstd(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = float(np.nansum((y - pred) ** 2))
        ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "n_bins": int(use.sum()),
        "k_calibration": k,
        "ratio_cv_pct": _safe_cv_pct(ratio_full[use]),
        "reldiff_median_pct": float(np.nanmedian(d)),
        "reldiff_rms_pct": float(np.sqrt(np.nanmean(d ** 2))),
        "within_10pct": float(np.mean(np.abs(d) <= 10.0) * 100.0),
        "within_20pct": float(np.mean(np.abs(d) <= 20.0) * 100.0),
        "slope": float(slope), "intercept": float(intercept), "r2": float(r2),
        "ratio_profile": ratio_full / k,   # normalised: 1.0 = perfectly flat
        "reldiff_profile": reldiff,
        # The two curves that were actually compared (bins outside the mask NaN),
        # prototype already divided by k so both sit on the MPL scale.
        "proto_scaled_profile": np.where(use, scaled, np.nan),
        "mpl_profile": np.where(use, mpl, np.nan),
    }


def mpl_depol_to_delta(d) -> np.ndarray:
    """Mini-MPL depolarization -> our volume depolarization ratio.

    SigmaMPL reports  d = cross/(cross+co);  the project uses the standard
    volume ratio  delta = cross/co = d/(1-d). Values outside [0,1) are dropped.
    """
    a = np.asarray(d, float)
    out = np.full(a.shape, np.nan)
    ok = np.isfinite(a) & (a >= 0) & (a < 1)
    out[ok] = a[ok] / (1.0 - a[ok])
    return out


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

def run_validation(
    proto_path,
    mpl_path,
    *,
    case_label: str = "",
    rmin_m: float = COMPARE_RMIN_M,
    rmax_m: float = COMPARE_RMAX_M,
    snr_min: float = COMPARE_SNR_MIN,
    time_filter: str = "night",           # 'night' | 'day' | 'all'
    match_tolerance_min: float = MATCH_TOLERANCE_MIN,
    min_shots: float = MIN_SHOTS,
    day_bg_threshold_mhz: float = DAY_BG_THRESHOLD_MHZ,
    day_glue_r2_min: float = DAY_GLUE_R2_MIN,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """Validate one prototype case against the Mini-MPL reference.

    Returns a dict of DataFrames ready to be written to a workbook / plotted:
      summary        one row, the case verdict
      per_profile    one row per matched profile pair
      reldiff        Range(m) x time, relative difference [%]
      ratio          Range(m) x time, ratio normalised to its own median
      reldiff_stats  Range(m) + median / mean / std across profiles
      depol          per-profile depolarization agreement
    """
    def log(msg: str) -> None:
        if logger:
            logger(msg)

    proto = load_prototype(proto_path)
    mpl = load_mpl(mpl_path)
    label = case_label or proto["label"]

    # The MPL grid is the coarser of the two -> compare on it.
    r_grid = np.asarray(mpl["range_m"], float)
    in_window = np.isfinite(r_grid) & (r_grid >= rmin_m) & (r_grid <= rmax_m)
    if not in_window.any():
        raise ValueError(
            f"No MPL range bins between {rmin_m:.0f} and {rmax_m:.0f} m "
            f"(grid is {np.nanmin(r_grid):.0f}..{np.nanmax(r_grid):.0f} m).")

    tf = str(time_filter).strip().lower()
    qc = proto.get("qc", {})
    all_ts = sorted(proto["nrb"])

    # 1) Drop aborted / partial acquisitions by shot count.
    have_shots = any(np.isfinite(r.get("shots", np.nan)) for r in qc.values())
    dropped_shots: List[str] = []
    kept: List = []
    for t in all_ts:
        row = _qc_for(qc, t)
        n = row.get("shots", np.nan) if row else np.nan
        if have_shots and np.isfinite(n) and n < min_shots:
            dropped_shots.append(f"{pd.Timestamp(t).strftime('%H:%M:%S')} ({int(n)} shots)")
        else:
            kept.append(t)
    if not have_shots:
        log(f"[{label}] shot filter unavailable — this workbook has no 'shots' "
            f"column. Re-run Step 2 to enable it.")
    elif dropped_shots:
        log(f"[{label}] dropped {len(dropped_shots)} profile(s) with < "
            f"{min_shots:g} shots: " + ", ".join(dropped_shots))
    else:
        log(f"[{label}] shot filter: every profile has >= {min_shots:g} shots")

    # 2) Day / night from the signal (background + glue fit), clock as fallback.
    day_night: Dict = {}
    for t in kept:
        day_night[t] = classify_day_night(t, _qc_for(qc, t),
                                          day_bg_threshold_mhz, day_glue_r2_min)
    basis = "signal (BG / glue r2)" if any(r != "clock (no QC)"
                                          for _, r in day_night.values()) else "clock"
    n_day = sum(1 for d, _ in day_night.values() if d == "day")
    log(f"[{label}] day/night by {basis}: {len(kept) - n_day} night, {n_day} day")
    for t in kept:
        d, why = day_night[t]
        log(f"    {pd.Timestamp(t).strftime('%H:%M:%S')}  {d:<5}  {why}")

    proto_ts = [t for t in kept if tf == "all" or day_night[t][0] == tf]
    pairs = match_profiles(proto_ts, sorted(mpl["nrb"]), match_tolerance_min)
    log(f"[{label}] {len(all_ts)} prototype profile(s), {len(kept)} after the shot "
        f"filter, {len(proto_ts)} '{tf}', {len(pairs)} matched to MPL")
    if not pairs:
        raise ValueError(
            f"No prototype profile matched an MPL profile within "
            f"{match_tolerance_min:g} min under the '{tf}' filter.")

    rows: List[dict] = []
    depol_rows: List[dict] = []
    reldiff_cols: Dict[str, np.ndarray] = {}
    ratio_cols: Dict[str, np.ndarray] = {}
    proto_cols: Dict[str, np.ndarray] = {}
    mplnrb_cols: Dict[str, np.ndarray] = {}
    dep_proto_cols: Dict[str, np.ndarray] = {}
    dep_mpl_cols: Dict[str, np.ndarray] = {}

    for p_ts, m_ts, dt_min in pairs:
        p_nrb = resample_to_grid(proto["range_m"], proto["nrb"][p_ts], r_grid)
        m_nrb = np.asarray(mpl["nrb"][m_ts], float)

        mask = in_window.copy()
        if proto["snr"] and p_ts in proto["snr"]:
            p_snr = resample_to_grid(proto["range_m"], proto["snr"][p_ts], r_grid)
            mask &= np.isfinite(p_snr) & (p_snr >= snr_min)
        if mpl["snr"] and m_ts in mpl["snr"]:
            m_snr = np.asarray(mpl["snr"][m_ts], float)
            mask &= np.isfinite(m_snr) & (m_snr >= snr_min)

        res = compare_one(r_grid, p_nrb, m_nrb, mask)
        name = pd.Timestamp(p_ts).strftime("%H:%M")
        reldiff_cols[name] = res.pop("reldiff_profile")
        ratio_cols[name] = res.pop("ratio_profile")
        proto_cols[name] = res.pop("proto_scaled_profile")
        mplnrb_cols[name] = res.pop("mpl_profile")
        qrow = _qc_for(qc, p_ts) or {}
        rows.append({"Time": pd.Timestamp(p_ts), "MPL time": pd.Timestamp(m_ts),
                     "dt (min)": round(dt_min, 1),
                     "shots": qrow.get("shots", np.nan),
                     "bg_par_mhz": qrow.get("bg_mhz", np.nan),
                     "glue_r2": qrow.get("glue_r2", np.nan),
                     "day_night": day_night[p_ts][0],
                     "day_night_reason": day_night[p_ts][1],
                     **res})

        # Depolarization: ours vs MPL, definitions aligned.
        if proto["depol"] and mpl["depol"] and p_ts in proto["depol"] and m_ts in mpl["depol"]:
            p_dep = resample_to_grid(proto["depol_range_m"], proto["depol"][p_ts], r_grid)
            m_dep = mpl_depol_to_delta(mpl["depol"][m_ts])
            dm = in_window & np.isfinite(p_dep) & np.isfinite(m_dep) & (m_dep > 0)
            if dm.sum() >= 3:
                dep_proto_cols[name] = np.where(dm, p_dep, np.nan)
                dep_mpl_cols[name] = np.where(dm, m_dep, np.nan)
                diff = (p_dep[dm] - m_dep[dm]) / m_dep[dm] * 100.0
                depol_rows.append({
                    "Time": pd.Timestamp(p_ts), "n_bins": int(dm.sum()),
                    "delta_proto_median": float(np.nanmedian(p_dep[dm])),
                    "delta_mpl_median": float(np.nanmedian(m_dep[dm])),
                    "reldiff_median_pct": float(np.nanmedian(diff)),
                    "within_20pct": float(np.mean(np.abs(diff) <= 20.0) * 100.0),
                })

    per_profile = pd.DataFrame(rows)
    reldiff_df = pd.DataFrame({"Range(m)": r_grid, **reldiff_cols})
    ratio_df = pd.DataFrame({"Range(m)": r_grid, **ratio_cols})

    # Range bins outside every profile's mask are all-NaN rows; numpy warns on
    # those, but an empty bin is the expected answer, not a problem.
    block = reldiff_df.drop(columns=["Range(m)"]).to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reldiff_stats = pd.DataFrame({
            "Range(m)": r_grid,
            "median_pct": np.nanmedian(block, axis=1),
            "mean_pct": np.nanmean(block, axis=1),
            "std_pct": np.nanstd(block, axis=1),
            "n_profiles": np.isfinite(block).sum(axis=1),
        })

    depol_df = pd.DataFrame(depol_rows)
    valid = per_profile[per_profile["n_bins"] >= 3] if len(per_profile) else per_profile

    def _med(col: str) -> float:
        if not len(valid):
            return float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(np.nanmedian(valid[col]))

    # A case that loses most of its profiles to the SNR mask is almost always a
    # workbook whose SNR came from a corrupt hardware StErr (30 m acquisitions
    # processed before the Poisson repair) — not a genuinely bad measurement.
    if len(per_profile) and len(valid) < 0.5 * len(per_profile):
        log(f"[{label}] WARNING: only {len(valid)}/{len(per_profile)} profiles "
            f"kept enough bins. If this is a 30 m case, re-run it with the "
            f"photon-StErr repair enabled — a corrupt StErr drives SNR to ~0 and "
            f"the mask then discards everything. See docs/licel_stderr_limitation.md.")

    summary = pd.DataFrame([{
        "Case": label,
        "Prototype file": Path(proto_path).name,
        "MPL file": Path(mpl_path).name,
        "Window (m)": f"{rmin_m:.0f}-{rmax_m:.0f}",
        "Time filter": tf,
        "SNR min": snr_min,
        "Min shots": min_shots,
        "Profiles dropped (shots)": len(dropped_shots),
        "Day/night basis": basis,
        "Profiles matched": len(per_profile),
        "Profiles valid": len(valid),
        "Bins per profile (median)": _med("n_bins"),
        "Ratio CV % (median)": _med("ratio_cv_pct"),
        "Rel diff % (median)": _med("reldiff_median_pct"),
        "Rel diff RMS % (median)": _med("reldiff_rms_pct"),
        "Within 10% (median)": _med("within_10pct"),
        "Within 20% (median)": _med("within_20pct"),
        "Regression slope (median)": _med("slope"),
        "Regression r2 (median)": _med("r2"),
        "Calibration k (median)": _med("k_calibration"),
        "Depol rel diff % (median)": (float(np.nanmedian(depol_df["reldiff_median_pct"]))
                                      if len(depol_df) else float("nan")),
    }])

    log(f"[{label}] valid {len(valid)}/{len(per_profile)} · "
        f"ratio CV {_med('ratio_cv_pct'):.1f}% · "
        f"rel diff {_med('reldiff_median_pct'):+.1f}% · "
        f"within 20% on {_med('within_20pct'):.0f}% of bins")

    return {
        "label": label,
        "range_m": r_grid,
        "summary": summary,
        "per_profile": per_profile,
        "reldiff": reldiff_df,
        "ratio": ratio_df,
        "reldiff_stats": reldiff_stats,
        "depol": depol_df,
        # Range x time curves, for drawing every view against range.
        "proto_scaled": pd.DataFrame({"Range(m)": r_grid, **proto_cols}),
        "mpl_nrb": pd.DataFrame({"Range(m)": r_grid, **mplnrb_cols}),
        "depol_proto": pd.DataFrame({"Range(m)": r_grid, **dep_proto_cols}),
        "depol_mpl": pd.DataFrame({"Range(m)": r_grid, **dep_mpl_cols}),
    }


def compare_cases(results: List[Dict[str, object]]) -> pd.DataFrame:
    """Stack the one-row summaries of several cases into a comparison table."""
    frames = [r["summary"] for r in results if r and "summary" in r]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_validation_workbook(results: List[Dict[str, object]], out_path) -> Path:
    """Write every case's sheets, plus the cross-case comparison, to one file."""
    out_path = Path(out_path)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        compare_cases(results).to_excel(xw, index=False, sheet_name="Case_comparison")
        for res in results:
            tag = str(res.get("label", "case"))[:18]
            res["per_profile"].to_excel(xw, index=False, sheet_name=f"{tag}_profiles"[:31])
            res["reldiff_stats"].to_excel(xw, index=False, sheet_name=f"{tag}_reldiff_st"[:31])
            res["reldiff"].to_excel(xw, index=False, sheet_name=f"{tag}_reldiff"[:31])
            res["ratio"].to_excel(xw, index=False, sheet_name=f"{tag}_ratio"[:31])
            if len(res.get("depol", [])):
                res["depol"].to_excel(xw, index=False, sheet_name=f"{tag}_depol"[:31])
    return out_path
