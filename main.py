#!/usr/bin/env python3
# main.py  –  LiDAR Analysis Suite
"""
Unified 4-step LiDAR workflow in one tabbed window:
  Step 1  MPL → rmin-rmax Builder
  Step 2  NRB Daily Profile Builder
  Step 3  ALT Calculator
  Step 4  RTI Visualizer

Place this file in the SAME folder as:
  modern_theme.py, gui_theme.py, nrb_engine.py, pbl_engine.py
"""
from __future__ import annotations

import os, re, sys, threading, subprocess, tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ── path setup ────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from modern_theme import (
    apply_theme, make_card, make_header, make_status_bar, make_log,
    append_log,
    BG, CARD, DARK, PRIMARY, BORDER, TEXT, SUBTEXT,
    SUCCESS, WARNING, ERROR, F_H1, F_H2, F_BODY, F_SMALL,
    PLIGHT, LBLUE,
)

# Optional NRB backend
try:
    from nrb_engine import (
        _read_tr40_dat_ascii_array,
        build_single_profile,
        build_daily_profile_from_folder as nrb_build_daily_profile_from_folder,
        collect_actual_timestamps as nrb_collect_actual_timestamps,
    )
    _HAS_NRB = True
except ImportError:
    _HAS_NRB = False

# Matplotlib for Step 2 profile plot and Step 4 RTI
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import LogLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import colors as mcolors

# ═════════════════════════════════════════════════════════════════════════════
# Colour / font extras
# ═════════════════════════════════════════════════════════════════════════════
STEP_COLORS = {
    "pending": ("#94A3B8", "#F8FAFC"),
    "running": (WARNING, "#FFFBEB"),
    "done":    (SUCCESS, "#ECFDF5"),
    "error":   (ERROR,   "#FEF2F2"),
}

# ═════════════════════════════════════════════════════════════════════════════
# Shared App State
# ═════════════════════════════════════════════════════════════════════════════
class AppState:
    """File paths shared across all 4 steps."""
    def __init__(self):
        self.step1_output: Optional[str] = None
        self.step2_output: Optional[str] = None
        self.step3_output: Optional[str] = None
        self._cbs: List[Callable] = []

    def register_cb(self, fn: Callable): self._cbs.append(fn)
    def _notify(self):
        for fn in self._cbs:
            try: fn()
            except Exception: pass

    def set_step1(self, p: str): self.step1_output = p; self._notify()
    def set_step2(self, p: str): self.step2_output = p; self._notify()
    def set_step3(self, p: str): self.step3_output = p; self._notify()


# ═════════════════════════════════════════════════════════════════════════════
# ScrollableFrame (used by RTI tab)
# ═════════════════════════════════════════════════════════════════════════════
class ScrollableFrame(tk.Frame):
    def __init__(self, master, bg=BG, width=None, **kw):
        super().__init__(master, bg=bg, **kw)
        kw2 = {"width": width} if width else {}
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, **kw2)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")
        self.inner.bind("<Configure>", lambda _: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self._win, width=e.width))
        self.canvas.bind_all("<MouseWheel>", self._wheel)

    def _wheel(self, e):
        w = self.winfo_containing(e.x_root, e.y_root)
        if w and str(w).startswith(str(self.canvas)):
            self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")


# ═════════════════════════════════════════════════════════════════════════════
# Step 1 + Step 2 Backend Helpers  (actual-files mode)
# ═════════════════════════════════════════════════════════════════════════════
PBL_COL_DI_0 = 112;  PBL_ROW_2_0 = 1


def _infer_interval_minutes(timestamps: List[pd.Timestamp]) -> Optional[float]:
    if len(timestamps) < 2:
        return None
    diffs = []
    for a, b in zip(timestamps[:-1], timestamps[1:]):
        dt_min = (b - a).total_seconds() / 60.0
        if dt_min > 0:
            diffs.append(dt_min)
    if not diffs:
        return None
    s = pd.Series(diffs)
    mode_vals = s.mode(dropna=True)
    if not mode_vals.empty:
        return float(mode_vals.iloc[0])
    return float(s.median())


def _parse_hhmm_text(text: str, *, field_name: str = "Start time") -> Optional[Tuple[int, int]]:
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


def _minutes_of_day(ts: pd.Timestamp) -> int:
    return int(ts.hour) * 60 + int(ts.minute)


def _s1_parse_ts(name: str) -> Optional[pd.Timestamp]:
    hits = re.findall(r"(\d{12})", name)
    if not hits:
        return None
    dt = pd.to_datetime(hits[-1], format="%Y%m%d%H%M", errors="coerce")
    return None if pd.isna(dt) else pd.Timestamp(dt)


def _s1_read_pbl_km(path: Path) -> float:
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
            dia = csv.Sniffer().sniff(sample, delimiters=[",", ";", "	"])
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


def _s1_read_copol(path: Path, row_start=0, row_end=498):
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
    rng = rng[row_start:row_end] * 1000.0; cop = cop[row_start:row_end]
    mx = np.nanmax(cop) if cop.size else float("nan")
    cop_n = cop / mx if (np.isfinite(mx) and mx != 0) else np.full_like(cop, float("nan"))
    return rng, cop_n


def _s1_collect_actual_files(folder: Path, date_text: str, start_time_text: str) -> Tuple[pd.Timestamp, List[Tuple[pd.Timestamp, Path]]]:
    mapping: Dict[pd.Timestamp, Path] = {}
    for f in sorted(folder.glob("*")):
        ts = _s1_parse_ts(f.name)
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

    start_pair = _parse_hhmm_text(start_time_text, field_name="Start time")
    start_min_total = None if start_pair is None else start_pair[0] * 60 + start_pair[1]

    ordered = []
    for ts, f in sorted(mapping.items(), key=lambda kv: kv[0]):
        if pd.Timestamp(ts).normalize() != target_date:
            continue
        if start_min_total is not None and _minutes_of_day(pd.Timestamp(ts)) < start_min_total:
            continue
        ordered.append((pd.Timestamp(ts), f))

    if not ordered:
        raise ValueError("No MPL CSV files found for the selected date/start time.")
    return target_date, ordered


# Step 2 actual-files helpers

def _s2_parse_time_from_filename(name: str) -> Optional[Tuple[int, int]]:
    patterns = [
        r"(?<!\d)(\d{1,2})[.:_-](\d{2})(?!\d)",
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


def _s2_make_timestamp(date_str: str, hh: int, mm: int) -> pd.Timestamp:
    base = pd.to_datetime(date_str).normalize()
    return base + pd.Timedelta(hours=hh, minutes=mm)


def _s2_parse_timestamp_from_filename(name: str, date_str: Optional[str] = None) -> Optional[pd.Timestamp]:
    hits12 = re.findall(r"(\d{12})", name)
    for token in reversed(hits12):
        ts = pd.to_datetime(token, format="%Y%m%d%H%M", errors="coerce")
        if pd.notna(ts):
            return pd.Timestamp(ts)
    if date_str:
        t = _s2_parse_time_from_filename(name)
        if t is not None:
            return _s2_make_timestamp(date_str, t[0], t[1])
    return None


def _s2_collect_actual_timestamps(
    folder: Path,
    *,
    pattern: str,
    date_str: str,
    start_time: str = "",
    logger: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Tuple[pd.Timestamp, Path]], List[Dict[str, object]]]:
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {folder} with pattern: {pattern}")

    target_date = pd.to_datetime(date_str).normalize()
    start_pair = _parse_hhmm_text(start_time, field_name="Start time")
    start_min_total = None if start_pair is None else start_pair[0] * 60 + start_pair[1]

    file_map: Dict[pd.Timestamp, Path] = {}
    qc_rows: List[Dict[str, object]] = []

    for f in files:
        ts = _s2_parse_timestamp_from_filename(f.name, date_str=date_str)
        if ts is None:
            qc_rows.append({"file": f.name, "time": pd.NaT, "status": "skip_no_timestamp"})
            if logger:
                logger(f"Skip (no timestamp): {f.name}")
            continue
        if ts.normalize() != target_date:
            qc_rows.append({"file": f.name, "time": ts, "status": "skip_other_date"})
            continue
        if start_min_total is not None and _minutes_of_day(ts) < start_min_total:
            qc_rows.append({"file": f.name, "time": ts, "status": "skip_before_start_time"})
            continue
        if ts in file_map:
            qc_rows.append({"file": f.name, "time": ts, "status": "duplicate_time_ignored"})
            continue
        file_map[ts] = f

    ordered = sorted(file_map.items(), key=lambda kv: kv[0])
    return ordered, qc_rows


def _s2_build_daily_profile_from_folder(
    folder: Path,
    *,
    date_str: str,
    pattern: str,
    out_path: Path,
    start_time: str = "",
    dr_m: float = 3.75,
    dead_time_ns: float = 3.06,
    bg_mode: str = "pretrigger",
    bg_start_m: float = 6500.0,
    bg_end_m: float = 7500.0,
    pretrigger_bins: int = 1024,
    first_signal_bin: Optional[int] = None,
    first_signal_range_m: float = 3.75,
    blend_r1_m: float = 1200.0,
    blend_r2_m: float = 1800.0,
    overlap_r1_m: float = 0.0,
    overlap_r2_m: float = 1000.0,
    overlap_min_r_m: float = 200.0,
    bin_shift_bins: int = 16,
    strict: bool = True,
    logger: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered_files, qc_rows = _s2_collect_actual_timestamps(
        folder,
        pattern=pattern,
        date_str=date_str,
        start_time=start_time,
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
                overlap_r1_m=overlap_r1_m,
                overlap_r2_m=overlap_r2_m,
                overlap_min_r_m=overlap_min_r_m,
                bin_shift_bins=bin_shift_bins,
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
            "date": date_str,
            "start_time": str(start_time).strip(),
            "pattern": pattern,
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
            "blend_r1_m": float(blend_r1_m),
            "blend_r2_m": float(blend_r2_m),
            "overlap_r1_m": float(overlap_r1_m),
            "overlap_r2_m": float(overlap_r2_m),
            "overlap_min_r_m": float(overlap_min_r_m),
            "bin_shift_bins": int(bin_shift_bins),
            "strict": bool(strict),
        }
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        df_profile.to_excel(xw, index=False, sheet_name="NRB profile")
        df_qc.to_excel(xw, index=False, sheet_name="QC_params")
        params.to_excel(xw, index=False, sheet_name="Parameters")

    return df_profile, df_qc, params


# ═════════════════════════════════════════════════════════════════════════════
# Step 3 Backend Helpers  (from pbl_gui_v5)
# ═════════════════════════════════════════════════════════════════════════════
RE_YMD  = re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})")
RE_DMY  = re.compile(r"(\d{2})[-_](\d{2})[-_](\d{4})")
RE_HHMM = re.compile(r"(\d{2}):(\d{2})(?::(\d{2}))?$")
PREF_NRB_SHEET  = "NRB profile";  PREF_RMIN_SHEET = "rmin-rmax"
ENGINE_CANDIDATES = ["pbl_engine.py", "pbl_v02_report.py", "pbl_v01_report.py"]

def _s3_guess_date(path: str):
    name = os.path.basename(path)
    m = RE_YMD.search(name)
    if m: y,mo,d = map(int, m.groups()); return pd.Timestamp(year=y, month=mo, day=d)
    m = RE_DMY.search(name)
    if m: d,mo,y = map(int, m.groups()); return pd.Timestamp(year=y, month=mo, day=d)
    return None

def _s3_parse_times(colnames, base_date):
    times = []
    for c in colnames:
        t = pd.to_datetime(c, errors="coerce")
        if pd.notna(t):
            t = pd.Timestamp(t)
            if t.year < 1975:
                t = pd.Timestamp(base_date.date()) + pd.Timedelta(hours=t.hour, minutes=t.minute)
            times.append(t); continue
        m = RE_HHMM.search(str(c).strip())
        if m:
            hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
            times.append(pd.Timestamp(base_date.date()) + pd.Timedelta(hours=hh, minutes=mm, seconds=ss)); continue
        raise ValueError(f"Cannot parse: {c!r}")
    return times

def _s3_pick_rmin_cols(df):
    cols = {str(c).strip().lower(): c for c in df.columns}
    def pick(*cands):
        for k in cands:
            if k in cols: return cols[k]
        return None
    tc = pick("time","datetime","timestamp"); rmc = pick("rmin","rmin_m","rmin (m)","rmin(m)")
    rxc = pick("rmax","rmax_m","rmax (m)","rmax(m)"); mc = None
    for k, orig in cols.items():
        if (("pbl" in k or "alt" in k) and "mpl" in k) or k in ("alt_mpl_m", "pbl_mpl_m"): mc = orig; break
    if tc is None: raise ValueError("rmin-rmax sheet must have a 'Time' column.")
    return tc, rmc, rxc, mc

def _s3_map_slots(df_rmin, slot_times, tol_min):
    tc, rmc, rxc, mc = _s3_pick_rmin_cols(df_rmin)
    work = df_rmin.copy().dropna(how="all").dropna(axis=1, how="all")
    work[tc] = pd.to_datetime(work[tc], errors="coerce"); work = work.dropna(subset=[tc]).sort_values(tc)
    if rmc: work[rmc] = pd.to_numeric(work[rmc], errors="coerce")
    if rxc: work[rxc] = pd.to_numeric(work[rxc], errors="coerce")
    if mc:  work[mc]  = pd.to_numeric(work[mc],  errors="coerce")
    times_mpl = work[tc].to_list(); rows = []
    for ts in slot_times:
        best_idx = best_abs = None
        for i, mt in enumerate(times_mpl):
            ad = abs((pd.Timestamp(mt) - pd.Timestamp(ts)).total_seconds() / 60.0)
            if best_abs is None or ad < best_abs: best_abs, best_idx = ad, i
        if best_abs is not None and best_abs <= tol_min:
            row = work.iloc[best_idx]
            rows.append({"Time": pd.Timestamp(ts), "Slot": pd.Timestamp(ts).strftime("%H:%M"),
                         "PBL from MPL (m)": float(row[mc]) if mc else np.nan,
                         "rmin": float(row[rmc]) if rmc else np.nan,
                         "rmax": float(row[rxc]) if rxc else np.nan,
                         "MPL_Time": pd.Timestamp(row[tc]), "dt_min": 0.0})
        else:
            rows.append({"Time": pd.Timestamp(ts), "Slot": pd.Timestamp(ts).strftime("%H:%M"),
                         "PBL from MPL (m)": np.nan, "rmin": np.nan, "rmax": np.nan,
                         "MPL_Time": pd.NaT, "dt_min": np.nan})
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# Step 4 RTI Data Helpers  (from rti_plot_gui_v10)
# ═════════════════════════════════════════════════════════════════════════════
PROTO_SHEET = "NRB profile"; MPL_SHEET = "copol_nrb_norm"
ALT_SHEET = "ALT_results"; LEGACY_ALT_SHEET = "PBL_results"; DENOISED_SHEET = "NRB_Denoised_FFT"
ALT_COL_MPL_CHOICES = ["ALT_MPL_m", "PBL_MPL_m"]
ALT_COL_PROTO_CHOICES = ["ALT_TR40_m", "ALT_guided_m", "ALT_profile_m", "ALT_MPL_m"]
CMAP_CHOICES = ["jet","turbo","viridis","plasma","inferno"]
INTERP_CHOICES = ["nearest","none","bilinear"]
HHMM_PAT = re.compile(r"^\d{2}:\d{2}(:\d{2})?$")

def _r4_float(s):
    s = (s or "").strip(); return None if s == "" else (float(s) if s else None)
def _r4_int(s):
    s = (s or "").strip(); return None if s == "" else int(float(s))
def _r4_first_ts(cols):
    try:
        arr = pd.to_datetime(list(cols), errors="coerce")
        valid = arr[~pd.isna(arr)]
        return pd.Timestamp(valid[0]) if len(valid) > 0 else None
    except Exception: return None

def _r4_parse_tcols(cols, base_date):
    cols = list(cols); sc = [str(c).strip() for c in cols]
    hc = sum(1 for s in sc if HHMM_PAT.match(s))
    if hc >= max(1, int(0.8 * len(sc))):
        out = []
        for s in sc:
            if not HHMM_PAT.match(s): out.append(pd.NaT); continue
            fmt = "%H:%M:%S" if len(s) == 8 else "%H:%M"
            tod = pd.to_datetime(s, format=fmt, errors="coerce")
            if pd.isna(tod): out.append(pd.NaT); continue
            if base_date is None: out.append(pd.Timestamp(tod))
            else:
                out.append(pd.Timestamp(base_date.date()) + pd.Timedelta(hours=tod.hour, minutes=tod.minute, seconds=tod.second))
        return out
    t = pd.to_datetime(cols, errors="coerce"); out = [pd.Timestamp(x) if not pd.isna(x) else pd.NaT for x in t]
    if base_date is not None:
        fixed = []
        for x in out:
            if pd.isna(x): fixed.append(pd.NaT); continue
            x = pd.Timestamp(x)
            if x.year < 1975: x = pd.Timestamp(base_date.date()) + pd.Timedelta(hours=x.hour, minutes=x.minute)
            fixed.append(x)
        out = fixed
    return out

def _r4_nearest(proto_times, mpl_times, max_min=3):
    valid = [(j, pd.Timestamp(mt)) for j, mt in enumerate(mpl_times) if not pd.isna(mt)]
    out_idx, out_dt = [], []
    for pt in proto_times:
        if pd.isna(pt) or not valid: out_idx.append(None); out_dt.append(None); continue
        pt = pd.Timestamp(pt); best = None
        for j, mt in valid:
            dt = (mt - pt).total_seconds() / 60.0; ad = abs(dt)
            if best is None or ad < best[0]: best = (ad, dt, j)
        if best and best[0] <= max_min: out_idx.append(best[2]); out_dt.append(best[1])
        else: out_idx.append(None); out_dt.append(best[1] if best else None)
    return out_idx, out_dt

def _r4_build_x_labels(timestamps):
    """Auto-format x-axis labels: HH:MM for single day, MM-DD HH:MM for multi-day."""
    valid = [pd.Timestamp(t) for t in timestamps if not pd.isna(t)]
    if not valid:
        return ["NaT"] * len(timestamps)
    multi_day = len({t.date() for t in valid}) > 1
    fmt = "%m-%d %H:%M" if multi_day else "%H:%M"
    return [pd.Timestamp(t).strftime(fmt) if not pd.isna(t) else "NaT" for t in timestamps]


def _r4_detect_interval_min(timestamps):
    """Detect typical sampling interval (minutes), snapped to standard slot."""
    ts = sorted(pd.Timestamp(t) for t in timestamps if not pd.isna(t))
    if len(ts) < 2:
        return 30
    diffs = np.array([(ts[i + 1] - ts[i]).total_seconds() / 60.0 for i in range(len(ts) - 1)])
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 30
    med = float(np.median(diffs))
    standards = [1, 5, 10, 15, 20, 30, 60]
    return min(standards, key=lambda s: abs(s - med))


def _r4_interp(x_src, y_src, x_dst):
    x_src = np.asarray(x_src, float); y_src = np.asarray(y_src, float); x_dst = np.asarray(x_dst, float)
    m = np.isfinite(x_src) & np.isfinite(y_src)
    if m.sum() < 2: return np.full_like(x_dst, np.nan, dtype=float)
    xs = x_src[m]; ys = y_src[m]; o = np.argsort(xs); xs = xs[o]; ys = ys[o]
    y_out = np.interp(x_dst, xs, ys, left=ys[0], right=ys[-1]).astype(float)
    y_out[(x_dst < xs[0]) | (x_dst > xs[-1])] = np.nan
    return y_out


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 FRAME  –  MPL → rmin-rmax
# ═════════════════════════════════════════════════════════════════════════════
class Step1Frame(tk.Frame):
    def __init__(self, parent, app_state: AppState):
        super().__init__(parent, bg=BG)
        self.app_state = app_state
        self.folder = tk.StringVar()
        self.date = tk.StringVar()
        self.start_time = tk.StringVar(value="")
        self.delta_m = tk.IntVar(value=300)
        self.out_path = tk.StringVar()
        self.progress = tk.DoubleVar(value=0.0)
        self._status_set = None
        self._build_ui()

    def _build_ui(self):
        make_header(
            self,
            "Step 1  ·  MPL → rmin-rmax Builder",
            "Actual-files mode · uses real timestamps from filenames",
        ).pack(fill="x")
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=16, pady=14)
        main.columnconfigure(0, weight=3); main.columnconfigure(1, weight=2); main.rowconfigure(0, weight=1)

        left = tk.Frame(main, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        c_in, f_in = make_card(left, "📁  Input Folder"); c_in.pack(fill="x", pady=(0, 10))
        tk.Label(f_in, text="Folder containing MPL CSV files:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(anchor="w")
        rf = tk.Frame(f_in, bg=CARD); rf.pack(fill="x", pady=(4, 0))
        ttk.Entry(rf, textvariable=self.folder).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(rf, text="Browse…", style="S.TButton", command=self.pick_folder).pack(side="right")
        tk.Label(f_in, text="Filenames must contain YYYYMMDDHHMM (12 digits)", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(anchor="w", pady=(4,0))

        c_p, f_p = make_card(left, "⚙️  Filter & Parameters"); c_p.pack(fill="x", pady=(0, 10))
        r1 = tk.Frame(f_p, bg=CARD); r1.pack(fill="x", pady=(0, 6))
        for lbl, var, w in [("Date (YYYY-MM-DD):", self.date, 12), ("Start time (HH:MM):", self.start_time, 8), ("± delta (m):", self.delta_m, 8)]:
            tk.Label(r1, text=lbl, bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left", padx=(0,4))
            ttk.Entry(r1, textvariable=var, width=w).pack(side="left", padx=(0,14))
        tk.Label(
            f_p,
            text="Mode A: use actual files only. No fixed 48 slots. Files earlier than Start time are excluded.",
            bg=CARD, fg=SUBTEXT, font=F_SMALL, justify="left"
        ).pack(anchor="w")

        c_o, f_o = make_card(left, "💾  Output Excel"); c_o.pack(fill="x", pady=(0, 10))
        ro = tk.Frame(f_o, bg=CARD); ro.pack(fill="x")
        ttk.Entry(ro, textvariable=self.out_path).pack(side="left", fill="x", expand=True, padx=(0,8))
        ttk.Button(ro, text="Save As…", style="S.TButton", command=self.pick_output).pack(side="right")

        c_r, f_r = make_card(left); c_r.pack(fill="x", pady=(0,6))
        rr = tk.Frame(f_r, bg=CARD); rr.pack(fill="x")
        self.run_btn = ttk.Button(rr, text="▶  Run Build rmin-rmax", style="P.TButton", command=self.run)
        self.run_btn.pack(side="left")
        self.pb = ttk.Progressbar(rr, variable=self.progress, maximum=100, length=260)
        self.pb.pack(side="left", padx=(16,0))

        right = tk.Frame(main, bg=BG); right.grid(row=0, column=1, sticky="nsew")
        lh = tk.Frame(right, bg=BG); lh.pack(fill="x", pady=(0,6))
        tk.Label(lh, text="Console Log", bg=BG, fg=TEXT, font=F_H2).pack(side="left")
        ttk.Button(lh, text="Clear", style="Sm.TButton", command=self._clear_log).pack(side="right")
        lo, self.log_txt = make_log(right, height=26); lo.pack(fill="both", expand=True)
        sb, self.status_var, self._status_set = make_status_bar(self); sb.pack(fill="x", side="bottom")

    def _log(self, msg): append_log(self.log_txt, msg)
    def _safe(self, fn, *a, **kw): self.after(0, lambda: fn(*a, **kw))
    def _clear_log(self):
        self.log_txt.configure(state="normal"); self.log_txt.delete("1.0","end"); self.log_txt.configure(state="disabled")

    def pick_folder(self):
        p = filedialog.askdirectory(title="Select MPL CSV folder")
        if not p: return
        self.folder.set(p); self._log(f"Folder: {p}")
        self.out_path.set(str(Path(p) / "rmin-rmax.xlsx")); self._log(f"Output: {self.out_path.get()}")

    def pick_output(self):
        out = filedialog.asksaveasfilename(title="Save rmin-rmax as", defaultextension=".xlsx", filetypes=[("Excel files","*.xlsx")])
        if out: self.out_path.set(out); self._log(f"Output: {out}")

    def run(self):
        folder = self.folder.get().strip(); out_path = self.out_path.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error","Please choose a valid folder."); return
        if not out_path:
            messagebox.showerror("Error","Please choose an output path."); return
        try:
            delta = float(self.delta_m.get())
            _parse_hhmm_text(self.start_time.get(), field_name="Start time")
        except Exception as e:
            messagebox.showerror("Error", str(e)); return

        self.run_btn.configure(state="disabled"); self.progress.set(0.0)
        self._status_set("Running…","running"); self._log("=== START ===")

        def worker():
            try:
                target_date, ordered = _s1_collect_actual_files(Path(folder), self.date.get().strip(), self.start_time.get().strip())
                stamps = [ts for ts, _ in ordered]
                interval_min = _infer_interval_minutes(stamps)
                self._safe(self._log, f"Using date: {target_date.strftime('%Y-%m-%d')}")
                if self.start_time.get().strip():
                    self._safe(self._log, f"Start time filter: {self.start_time.get().strip()}")
                self._safe(self._log, f"Profiles used: {len(ordered)}")
                if interval_min is not None:
                    self._safe(self._log, f"Inferred interval: {interval_min:g} min")

                rows = []
                copol_cols: Dict[pd.Timestamp, np.ndarray] = {}
                range_m_master = None
                copol_n = 498
                total = len(ordered)

                for i, (ts, f) in enumerate(ordered, 1):
                    src = f.name
                    try:
                        rng_m, cop = _s1_read_copol(f, 0, copol_n)
                        if range_m_master is None and np.isfinite(rng_m).any():
                            range_m_master = rng_m
                        copol_cols[ts] = cop
                    except Exception as e:
                        self._safe(self._log, f"   [WARN] copol: {e}")
                        copol_cols[ts] = np.full((copol_n,), np.nan)

                    pbl_km = _s1_read_pbl_km(f)
                    if np.isfinite(pbl_km):
                        pbl_m = float(pbl_km) * 1000.0
                        rmin = max(0.0, pbl_m - delta)
                        rmax = pbl_m + delta
                        status = "ok"
                        self._safe(self._log, f"[{i:02d}/{total}] {ts.strftime('%H:%M')} ← {src}  PBL={pbl_m:.1f} m")
                    else:
                        pbl_m = rmin = rmax = np.nan
                        status = "read_fail"
                        self._safe(self._log, f"[{i:02d}/{total}] {ts.strftime('%H:%M')} ← {src}  PBL=NaN")

                    rows.append({
                        "Time": ts,
                        "PBL from MPL (m)": pbl_m,
                        "rmin": rmin,
                        "rmax": rmax,
                        "Status": status,
                        "Source file": src,
                    })
                    self._safe(self.progress.set, 100.0 * i / total)

                df = pd.DataFrame(rows)
                if range_m_master is None:
                    range_m_master = np.full((copol_n,), np.nan)
                copol_df = pd.DataFrame({"Range(m)": range_m_master})
                for ts in stamps:
                    copol_df[ts.strftime("%H:%M")] = copol_cols.get(ts, np.full((copol_n,), np.nan))

                outp = Path(out_path)
                with pd.ExcelWriter(outp, engine="openpyxl") as xw:
                    df.to_excel(xw, index=False, sheet_name="rmin-rmax")
                    copol_df.to_excel(xw, index=False, sheet_name="copol_nrb_norm")

                self._safe(self._log, f"[OK] Saved: {outp.resolve()}")
                self._safe(self._status_set, "Done ✅", "ok")
                self._safe(self.progress.set, 100.0)
                self._safe(self.app_state.set_step1, str(outp.resolve()))
                self._safe(messagebox.showinfo, "Step 1 Complete", f"Saved:\n{outp}")
            except Exception as e:
                self._safe(self._log, f"[FAILED] {e}")
                self._safe(self._status_set, "Failed ❌","error")
                self._safe(messagebox.showerror, "Failed", str(e))
            finally:
                self._safe(self.run_btn.configure, state="normal")
                self._safe(self._log, "=== END ===")
        threading.Thread(target=worker, daemon=True).start()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 FRAME  –  NRB Daily Profile Builder
# ═════════════════════════════════════════════════════════════════════════════
class Step2Frame(tk.Frame):
    def __init__(self, parent, app_state: AppState):
        super().__init__(parent, bg=BG)
        self.app_state = app_state
        self.folder = tk.StringVar(); self.date = tk.StringVar(value="2026-02-13")
        self.start_time = tk.StringVar(value="")
        self.pattern = tk.StringVar(value="*.dat"); self.out_path = tk.StringVar()
        self.bin_spacing_m = tk.DoubleVar(value=3.75); self.dead_time_ns = tk.DoubleVar(value=3.06)
        self.bg_mode = tk.StringVar(value="pretrigger"); self.bg_start_m = tk.DoubleVar(value=0.0)
        self.bg_end_m = tk.DoubleVar(value=3750.0); self.pretrigger_bins = tk.IntVar(value=1024)
        self.first_signal_bin = tk.IntVar(value=1025); self.first_signal_range_m = tk.DoubleVar(value=3.75)
        self.sig_start_m = tk.DoubleVar(value=0.0); self.sig_end_m = tk.DoubleVar(value=15000.0)
        self.min_toggle_rate = tk.DoubleVar(value=0.5); self.max_toggle_rate = tk.DoubleVar(value=10.0)
        self.auto_toggle_selector = tk.BooleanVar(value=True)
        self.day_min_toggle_rate = tk.DoubleVar(value=75.0); self.day_max_toggle_rate = tk.DoubleVar(value=130.0)
        self.toggle_bg_switch_threshold_mhz = tk.DoubleVar(value=10.0); self.pretrigger_trim_bins = tk.IntVar(value=24)
        self.blend_r1_m = tk.DoubleVar(value=1200.0); self.blend_r2_m = tk.DoubleVar(value=1800.0)
        self.auto_blend = tk.BooleanVar(value=True)
        self.photon_only = tk.BooleanVar(value=False)
        self.bin_shift_bins = tk.IntVar(value=0)
        self.energy_mj = tk.DoubleVar(value=25.0)
        self.strict = tk.BooleanVar(value=True); self.preset_name = tk.StringVar(value="Custom")
        self.bg_mode_help_var = tk.StringVar(value="")
        self.pretrigger_role_var = tk.StringVar(value="")
        self.data_source = tk.StringVar(value="Processed"); self.plot_mode = tk.StringVar(value="NRB")
        self.xmin_var = tk.StringVar(); self.xmax_var = tk.StringVar()
        self.ymin_var = tk.StringVar(); self.ymax_var = tk.StringVar()
        self.chart_title_var = tk.StringVar(); self.x_title_var = tk.StringVar(); self.y_title_var = tk.StringVar()
        self.progress = tk.DoubleVar(value=0.0)
        self._status_set = None; self._times_map: Dict[str, Path] = {}
        self._latest_profile_df = None; self._plot_cache: Dict[tuple, pd.DataFrame] = {}
        if not _HAS_NRB:
            self._show_missing_backend(); return
        self._build_ui(); self._update_bg_mode_ui(); self._update_plot_mode_options()
        self.date.trace_add("write", lambda *_: self._refresh_time_list())
        self.start_time.trace_add("write", lambda *_: self._refresh_time_list())
        self.pattern.trace_add("write", lambda *_: self._refresh_time_list())

    def _show_missing_backend(self):
        f = tk.Frame(self, bg=BG); f.pack(expand=True)
        tk.Label(f, text="⚠  nrb_engine.py not found in same folder", bg=BG, fg=WARNING, font=F_H2).pack(pady=40)

    def _build_ui(self):
        make_header(self, "Step 2  ·  NRB Daily Profile Builder", "v11.3 · actual-files mode · latest NRB engine · auto day/night toggle selector").pack(fill="x")
        main = tk.Frame(self, bg=BG); main.pack(fill="both", expand=True, padx=16, pady=10)
        main.grid_columnconfigure(0, weight=5); main.grid_columnconfigure(1, weight=6); main.grid_rowconfigure(0, weight=1)

        left = tk.Frame(main, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0,12))
        left.grid_columnconfigure(0, weight=1); left.grid_rowconfigure(3, weight=1)

        right = tk.Frame(main, bg=BG)
        right.grid(row=0, column=1, sticky="nsew"); right.grid_columnconfigure(0, weight=1); right.grid_rowconfigure(0, weight=1)

        c_in, f_in = make_card(left, "📁 Input / Output"); c_in.grid(row=0, column=0, sticky="ew", pady=(0,10))
        r = tk.Frame(f_in, bg=CARD); r.pack(fill="x")
        ttk.Entry(r, textvariable=self.folder).pack(side="left", fill="x", expand=True, padx=(0,8))
        ttk.Button(r, text="Browse…", command=self.pick_folder).pack(side="left")
        r2 = tk.Frame(f_in, bg=CARD); r2.pack(fill="x", pady=(8,0))
        tk.Label(r2, text="Date:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(r2, textvariable=self.date, width=12).pack(side="left", padx=(4,12))
        tk.Label(r2, text="Start time:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(r2, textvariable=self.start_time, width=8).pack(side="left", padx=(4,12))
        tk.Label(r2, text="Pattern:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(r2, textvariable=self.pattern, width=10).pack(side="left", padx=(4,12))
        ttk.Checkbutton(r2, text="Strict mode", variable=self.strict).pack(side="left")
        r3 = tk.Frame(f_in, bg=CARD); r3.pack(fill="x", pady=(8,0))
        ttk.Entry(r3, textvariable=self.out_path).pack(side="left", fill="x", expand=True, padx=(0,8))
        ttk.Button(r3, text="Save As…", command=self.pick_output).pack(side="left")
        tk.Label(f_in, text="Actual-files mode: no fixed 48 slots. Files earlier than Start time are excluded. Latest engine: shift fixed to zero, pretrigger BG trim, auto day/night toggle.", bg=CARD, fg=SUBTEXT, font=F_SMALL, justify="left").pack(anchor="w", pady=(8,0))

        c_p, f_p = make_card(left, "⚙ Parameters"); c_p.grid(row=1, column=0, sticky="ew", pady=(0,10))
        p0 = tk.Frame(f_p, bg=CARD); p0.pack(fill="x", pady=(0,6))
        ttk.Button(p0, text="Licel pretrigger 1024", command=lambda: self.apply_preset("Licel pretrigger 1024")).pack(side="left", padx=(0,6))
        ttk.Button(p0, text="Legacy no pretrigger", command=lambda: self.apply_preset("Legacy no pretrigger")).pack(side="left", padx=(0,6))
        ttk.Button(p0, text="Auto detect", command=self.auto_detect_bins).pack(side="left")
        ttk.Label(p0, textvariable=self.preset_name).pack(side="right")

        note = tk.Label(f_p, text="Grouped by latest pipeline: Raw layout → Dead time → Auto day/night toggle select → Toggle fit → Threshold-based glue → Pretrigger background → Normalization", bg=CARD, fg=SUBTEXT, font=F_SMALL, justify="left", wraplength=760)
        note.pack(anchor="w", pady=(0,6))

        param_rows = []
        for _ in range(8):
            r = tk.Frame(f_p, bg=CARD); r.pack(fill="x", pady=(0,5)); param_rows.append(r)

        for lbl, var, w in [("bin_spacing_m:", self.bin_spacing_m, 8), ("dead_time_ns:", self.dead_time_ns, 8), ("BG mode:", None, 0)]:
            tk.Label(param_rows[0], text=lbl, bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
            if var:
                ttk.Entry(param_rows[0], textvariable=var, width=w).pack(side="left", padx=(4,12))
        self.bg_mode_box = ttk.Combobox(param_rows[0], textvariable=self.bg_mode, state="readonly", width=12, values=["pretrigger","fixed","far_range"])
        self.bg_mode_box.pack(side="left", padx=(4,0))
        self.bg_mode.trace_add("write", lambda *_: self._update_bg_mode_ui())

        tk.Label(param_rows[1], text="layout_pretrigger_bins:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        self.pretrigger_entry = ttk.Entry(param_rows[1], textvariable=self.pretrigger_bins, width=8)
        self.pretrigger_entry.pack(side="left", padx=(4,12))
        tk.Label(param_rows[1], text="first_signal_bin:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        self.first_signal_entry = ttk.Entry(param_rows[1], textvariable=self.first_signal_bin, width=8)
        self.first_signal_entry.pack(side="left", padx=(4,12))
        self.sync_first_signal_btn = ttk.Button(param_rows[1], text="Sync = pre+1", command=self.sync_first_signal_bin)
        self.sync_first_signal_btn.pack(side="left")
        tk.Label(param_rows[1], textvariable=self.pretrigger_role_var, bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left", padx=(10,0))

        tk.Label(param_rows[2], text="first_signal_range_m:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(param_rows[2], textvariable=self.first_signal_range_m, width=8).pack(side="left", padx=(4,12))
        tk.Label(param_rows[2], text="bg_start_m:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        self.bg_start_entry = ttk.Entry(param_rows[2], textvariable=self.bg_start_m, width=8)
        self.bg_start_entry.pack(side="left", padx=(4,12))
        tk.Label(param_rows[2], text="bg_end_m:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        self.bg_end_entry = ttk.Entry(param_rows[2], textvariable=self.bg_end_m, width=8)
        self.bg_end_entry.pack(side="left", padx=(4,12))

        tk.Label(param_rows[3], textvariable=self.bg_mode_help_var, bg=CARD, fg=PRIMARY, font=F_SMALL).pack(anchor="w")

        for lbl, var in [("min_toggle_rate:", self.min_toggle_rate), ("max_toggle_rate:", self.max_toggle_rate), ("blend_r1_m:", self.blend_r1_m), ("blend_r2_m:", self.blend_r2_m)]:
            tk.Label(param_rows[4], text=lbl, bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
            ttk.Entry(param_rows[4], textvariable=var, width=8).pack(side="left", padx=(4,12))
        ttk.Checkbutton(param_rows[4], text="Auto day/night toggle", variable=self.auto_toggle_selector).pack(side="left")

        tk.Label(param_rows[5], text="day_min_toggle:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(param_rows[5], textvariable=self.day_min_toggle_rate, width=8).pack(side="left", padx=(4,12))
        tk.Label(param_rows[5], text="day_max_toggle:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(param_rows[5], textvariable=self.day_max_toggle_rate, width=8).pack(side="left", padx=(4,12))
        tk.Label(param_rows[5], text="BG switch threshold:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(param_rows[5], textvariable=self.toggle_bg_switch_threshold_mhz, width=8).pack(side="left", padx=(4,12))
        tk.Label(param_rows[5], text="pretrigger trim bins:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(param_rows[5], textvariable=self.pretrigger_trim_bins, width=8).pack(side="left", padx=(4,12))

        tk.Label(param_rows[6], text="sig_start_m:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(param_rows[6], textvariable=self.sig_start_m, width=8).pack(side="left", padx=(4,12))
        tk.Label(param_rows[6], text="sig_end_m:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(param_rows[6], textvariable=self.sig_end_m, width=8).pack(side="left", padx=(4,12))
        tk.Label(param_rows[6], text="energy_mj:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(param_rows[6], textvariable=self.energy_mj, width=8).pack(side="left", padx=(4,12))
        ttk.Checkbutton(param_rows[6], text="Auto blend r1/r2", variable=self.auto_blend).pack(side="left")
        ttk.Checkbutton(param_rows[6], text="Skip glue (photon only)", variable=self.photon_only).pack(side="left", padx=(12,0))

        tk.Label(param_rows[7], text="shift fixed = 0 in latest engine · auto blend uses photon_dt threshold crossings · BG mode: pretrigger / fixed (meters) / far_range", bg=CARD, fg="#92400E", font=F_SMALL).pack(anchor="w")

        c_run, f_run = make_card(left, "▶ Run / Console"); c_run.grid(row=3, column=0, sticky="nsew")
        top = tk.Frame(f_run, bg=CARD); top.pack(fill="x")
        self.run_btn = ttk.Button(top, text="Run Build NRB Profile", style="P.TButton", command=self.run)
        self.run_btn.pack(side="left")
        self.pb = ttk.Progressbar(top, variable=self.progress, maximum=100, length=200)
        self.pb.pack(side="left", padx=(12,0))
        ttk.Button(top, text="Clear", command=self._clear_log, style="Sm.TButton").pack(side="right")
        self.log_txt = tk.Text(f_run, height=14, bg="#0B1220", fg="#E2E8F0", insertbackground="#E2E8F0", font=("Consolas",9))
        self.log_txt.pack(fill="both", expand=True, pady=(8,0)); self.log_txt.configure(state="disabled")

        c_plot, f_plot = make_card(right, "📈 Profile Plot"); c_plot.grid(row=0, column=0, sticky="nsew")
        ctrl1 = tk.Frame(f_plot, bg=CARD); ctrl1.pack(fill="x", pady=(0,6))
        tk.Label(ctrl1, text="Source:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        self.source_box = ttk.Combobox(ctrl1, textvariable=self.data_source, state="readonly", width=12, values=["Processed","Raw .dat"])
        self.source_box.pack(side="left", padx=(4,12)); self.source_box.bind("<<ComboboxSelected>>", lambda e: self._update_plot_mode_options())
        tk.Label(ctrl1, text="Plot:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        self.plot_mode_box = ttk.Combobox(ctrl1, textvariable=self.plot_mode, state="readonly", width=14)
        self.plot_mode_box.pack(side="left", padx=(4,12))
        ttk.Button(ctrl1, text="Refresh Plot", command=self.refresh_plot).pack(side="left")
        ttk.Button(ctrl1, text="Auto Scale", command=self.auto_scale).pack(side="left", padx=(6,0))
        ttk.Button(ctrl1, text="Save PNG", command=self.save_png).pack(side="left", padx=(18,0))
        ttk.Button(ctrl1, text="Export CSV", command=self.save_csv).pack(side="left", padx=(6,0))
        ctrl2 = tk.Frame(f_plot, bg=CARD); ctrl2.pack(fill="x", pady=(0,8))
        for lbl, var in [("x min:", self.xmin_var), ("x max:", self.xmax_var), ("y min:", self.ymin_var), ("y max:", self.ymax_var)]:
            tk.Label(ctrl2, text=lbl, bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
            ttk.Entry(ctrl2, textvariable=var, width=8).pack(side="left", padx=(4,8))
        ttk.Button(ctrl2, text="Apply Axis", command=self.apply_axes).pack(side="left")
        ctrl3 = tk.Frame(f_plot, bg=CARD); ctrl3.pack(fill="x", pady=(0,8))
        for lbl, var, w in [("Chart title:", self.chart_title_var, 22), ("X title:", self.x_title_var, 16), ("Y title:", self.y_title_var, 16)]:
            tk.Label(ctrl3, text=lbl, bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
            ttk.Entry(ctrl3, textvariable=var, width=w).pack(side="left", padx=(4,12))
        sel_row = tk.Frame(f_plot, bg=CARD); sel_row.pack(fill="x", expand=False)
        tk.Label(sel_row, text="Times (multi-select):", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(anchor="w")
        lb_o = tk.Frame(sel_row, bg=CARD); lb_o.pack(fill="x", pady=(4,8))
        self.time_list = tk.Listbox(lb_o, selectmode="extended", exportselection=False, height=7)
        self.time_list.pack(side="left", fill="x", expand=True)
        ysb = ttk.Scrollbar(lb_o, orient="vertical", command=self.time_list.yview)
        ysb.pack(side="left", fill="y"); self.time_list.config(yscrollcommand=ysb.set)
        btns = tk.Frame(lb_o, bg=CARD); btns.pack(side="left", padx=(8,0), anchor="n")
        ttk.Button(btns, text="Select All", command=lambda: self.time_list.selection_set(0,"end")).pack(fill="x", pady=(0,4))
        ttk.Button(btns, text="Clear", command=lambda: self.time_list.selection_clear(0,"end")).pack(fill="x")
        self.fig2 = plt.Figure(figsize=(7,5.5), dpi=100); self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_title("No data"); self.ax2.grid(True, alpha=0.3)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=f_plot); self.canvas2.get_tk_widget().pack(fill="both", expand=True)
        sb, self.status_var, self._status_set = make_status_bar(self); sb.pack(fill="x", side="bottom")

    def _log(self, msg): append_log(self.log_txt, msg)
    def _safe(self, fn, *a, **kw): self.after(0, lambda: fn(*a, **kw))
    def _safe_log(self, msg): self._safe(self._log, msg)
    def _clear_log(self):
        self.log_txt.configure(state="normal"); self.log_txt.delete("1.0","end"); self.log_txt.configure(state="disabled")

    def sync_first_signal_bin(self):
        try:
            self.first_signal_bin.set(int(self.pretrigger_bins.get()) + 1)
            self.preset_name.set("Custom")
        except Exception:
            pass

    def _update_bg_mode_ui(self):
        mode = self.bg_mode.get().strip().lower()
        self.pretrigger_entry.configure(state="normal")
        if hasattr(self, "first_signal_entry"):
            self.first_signal_entry.configure(state="normal")
        if hasattr(self, "sync_first_signal_btn"):
            self.sync_first_signal_btn.configure(state="normal")
        fixed_state = "normal" if mode == "fixed" else "disabled"
        if hasattr(self, "bg_start_entry"):
            self.bg_start_entry.configure(state=fixed_state)
        if hasattr(self, "bg_end_entry"):
            self.bg_end_entry.configure(state=fixed_state)
        if mode == "pretrigger":
            self.bg_mode_help_var.set("BG source: mean of pretrigger bins after dead-time correction.")
            self.pretrigger_role_var.set("used for BG + layout/cut")
        elif mode == "far_range":
            self.bg_mode_help_var.set("BG source: auto far-range window 0.88*rmax to 0.98*rmax (meters). bg_start_m/bg_end_m ignored.")
            self.pretrigger_role_var.set("used for layout/cut only (set 0 if file has no pretrigger)")
        elif mode == "fixed":
            self.bg_mode_help_var.set("BG source: user-defined range bg_start_m to bg_end_m (meters).")
            self.pretrigger_role_var.set("used for layout/cut only (set 0 if file has no pretrigger)")
        else:
            self.bg_mode_help_var.set("")
            self.pretrigger_role_var.set("")

    def _update_plot_mode_options(self):
        vals = ["Analog Raw","Photon Raw"] if self.data_source.get() == "Raw .dat" else ["Photon DT","Analog Scaled","Glue","Glue Overlay","NRB"]
        self.plot_mode_box.configure(values=vals)
        if self.plot_mode.get() not in vals: self.plot_mode.set(vals[0])

    def apply_preset(self, name):
        if name == "Licel pretrigger 1024":
            self.bg_mode.set("pretrigger"); self.pretrigger_bins.set(1024); self.first_signal_bin.set(1025)
            self.first_signal_range_m.set(3.75); self.blend_r1_m.set(1200.0); self.blend_r2_m.set(1800.0)
            self.min_toggle_rate.set(0.5); self.max_toggle_rate.set(10.0)
            self.day_min_toggle_rate.set(75.0); self.day_max_toggle_rate.set(130.0)
            self.toggle_bg_switch_threshold_mhz.set(10.0); self.pretrigger_trim_bins.set(24)
            self.sig_start_m.set(0.0); self.sig_end_m.set(15000.0); self.energy_mj.set(25.0)
            self.bin_shift_bins.set(0)
        elif name == "Legacy no pretrigger":
            self.bg_mode.set("fixed"); self.pretrigger_bins.set(0); self.first_signal_bin.set(1)
            self.first_signal_range_m.set(3.75); self.bin_shift_bins.set(0)
            self.bg_start_m.set(13000.0); self.bg_end_m.set(14500.0)
            self.blend_r1_m.set(1200.0); self.blend_r2_m.set(1800.0)
            self.min_toggle_rate.set(0.5); self.max_toggle_rate.set(10.0)
            self.day_min_toggle_rate.set(75.0); self.day_max_toggle_rate.set(130.0)
            self.toggle_bg_switch_threshold_mhz.set(10.0); self.pretrigger_trim_bins.set(0)
            self.sig_start_m.set(0.0); self.sig_end_m.set(15000.0); self.energy_mj.set(25.0)
        self.preset_name.set(name); self._update_bg_mode_ui()

    def auto_detect_bins(self):
        folder = self.folder.get().strip()
        if not folder or not os.path.isdir(folder): messagebox.showerror("Error","Choose a folder first."); return
        files = sorted(Path(folder).glob(self.pattern.get().strip() or "*.dat"))
        if not files: messagebox.showerror("Error","No .dat files found."); return
        try:
            arr = _read_tr40_dat_ascii_array(files[0]); bins = int(arr.shape[0])
            self._log(f"Auto detect: {files[0].name} → {bins} bins")
            self.apply_preset("Licel pretrigger 1024" if bins >= 5024 else "Legacy no pretrigger")
        except Exception as e: messagebox.showerror("Auto detect failed", str(e))

    def pick_folder(self):
        p = filedialog.askdirectory(title="Select .dat folder")
        if not p: return
        self.folder.set(p); self._log(f"Folder: {p}")
        self.out_path.set(str(Path(p) / f"NRB-{self.date.get()}.xlsx")); self._refresh_time_list()

    def pick_output(self):
        out = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel","*.xlsx")])
        if out: self.out_path.set(out)

    def _refresh_time_list(self):
        if not hasattr(self, "time_list"):
            return
        self.time_list.delete(0,"end"); self._times_map.clear()
        folder = self.folder.get().strip()
        if not folder or not os.path.isdir(folder):
            return
        try:
            ordered, _ = nrb_collect_actual_timestamps(Path(folder), pattern=self.pattern.get().strip() or "*.dat", date_str=self.date.get().strip(), start_time=self.start_time.get().strip(), logger=None)
        except Exception:
            return
        for ts, f in ordered:
            hhmm = ts.strftime("%H:%M")
            if hhmm not in self._times_map:
                self._times_map[hhmm] = f
                self.time_list.insert("end", hhmm)

    def _selected_times(self): return [self.time_list.get(i) for i in self.time_list.curselection()]

    def _gluing_mode_str(self) -> str:
        return "photon_only" if bool(self.photon_only.get()) else "auto"

    def _build_profile_for_plot(self, hhmm):
        """Run build_single_profile for the selected time and return (prof_df, meta)."""
        path = self._times_map.get(hhmm)
        if path is None: raise ValueError(f"No file for {hhmm}")
        return build_single_profile(
            path,
            dr_m=float(self.bin_spacing_m.get()), dead_time_ns=float(self.dead_time_ns.get()),
            bg_mode=self.bg_mode.get().strip().lower(), bg_start_m=float(self.bg_start_m.get()), bg_end_m=float(self.bg_end_m.get()),
            pretrigger_bins=int(self.pretrigger_bins.get()), first_signal_bin=int(self.first_signal_bin.get()),
            first_signal_range_m=float(self.first_signal_range_m.get()),
            blend_r1_m=float(self.blend_r1_m.get()), blend_r2_m=float(self.blend_r2_m.get()),
            shift_mode="manual", bin_shift_bins=0,
            sig_start_m=float(self.sig_start_m.get()), sig_end_m=float(self.sig_end_m.get()),
            min_toggle_rate=float(self.min_toggle_rate.get()), max_toggle_rate=float(self.max_toggle_rate.get()),
            auto_toggle_selector=bool(self.auto_toggle_selector.get()),
            day_min_toggle_rate=float(self.day_min_toggle_rate.get()), day_max_toggle_rate=float(self.day_max_toggle_rate.get()),
            toggle_bg_switch_threshold_mhz=float(self.toggle_bg_switch_threshold_mhz.get()),
            pretrigger_trim_bins=int(self.pretrigger_trim_bins.get()),
            energy_mj=float(self.energy_mj.get()), auto_blend=bool(self.auto_blend.get()),
            gluing_mode=self._gluing_mode_str(),
        )

    def _build_plot_df(self, hhmm):
        prof, _ = self._build_profile_for_plot(hhmm)
        src = self.data_source.get(); mode = self.plot_mode.get()
        out = pd.DataFrame({"Range (m)": prof["range_m"].to_numpy(float)})
        if src == "Raw .dat":
            out[hhmm] = prof["analog_mV" if mode == "Analog Raw" else "photon_MHz"].to_numpy(float)
        else:
            col = {"Photon DT": "photon_deadtime_corr_MHz", "Analog Scaled": "analog_scaled_MHz", "Glue": "glued_profile_MHz", "NRB": "nrb"}[mode]
            out[hhmm] = prof[col].to_numpy(float)
        return out

    def _plot_glue_overlay(self, hhmm):
        """Plot Photon DT + Scaled Analog + Glued profile with r1/r2 and toggle threshold lines."""
        prof, meta = self._build_profile_for_plot(hhmm)
        r = prof["range_m"].to_numpy(float)
        ph_dt = prof["photon_deadtime_corr_MHz"].to_numpy(float)
        an_sc = prof["analog_scaled_MHz"].to_numpy(float)
        glued = prof["glued_profile_MHz"].to_numpy(float)

        def _plot_finite(x, y, **kw):
            m = np.isfinite(x) & np.isfinite(y)
            if m.any(): self.ax2.plot(x[m], y[m], **kw)

        _plot_finite(r, ph_dt, color="#1f77b4", linewidth=1.0, alpha=0.85, label="Photon DT")
        _plot_finite(r, an_sc, color="#2ca02c", linewidth=1.0, linestyle="--", alpha=0.85, label="Scaled Analog")
        _plot_finite(r, glued, color="#d62728", linewidth=1.6, label="Glued")

        min_tog = float(meta.get("min_toggle_rate", np.nan))
        max_tog = float(meta.get("max_toggle_rate", np.nan))
        r1 = float(meta.get("blend_r1_used_m", np.nan))
        r2 = float(meta.get("blend_r2_used_m", np.nan))
        if np.isfinite(min_tog):
            self.ax2.axhline(min_tog, color="gray", linestyle=":", linewidth=0.9, alpha=0.7, label=f"min toggle = {min_tog:g}")
        if np.isfinite(max_tog):
            self.ax2.axhline(max_tog, color="gray", linestyle=":", linewidth=0.9, alpha=0.7, label=f"max toggle = {max_tog:g}")
        if np.isfinite(r1):
            self.ax2.axvline(r1, color="orange", linestyle="--", linewidth=0.9, alpha=0.85, label=f"r1 = {r1:.0f} m")
        if np.isfinite(r2):
            self.ax2.axvline(r2, color="purple", linestyle="--", linewidth=0.9, alpha=0.85, label=f"r2 = {r2:.0f} m")

        glue_mode_txt = str(meta.get("glue_mode", "")) or str(meta.get("gluing_mode", ""))
        self.ax2.set_title(self.chart_title_var.get() or f"Glue Overlay · {hhmm} · {glue_mode_txt}")

    def refresh_plot(self):
        times = self._selected_times()
        if not times: return
        self.ax2.clear()
        mode = self.plot_mode.get()
        if mode == "Glue Overlay":
            try:
                self._plot_glue_overlay(times[0])
            except Exception as e:
                self.ax2.set_title(f"Error: {e}")
        else:
            for hhmm in times:
                try:
                    df = self._build_plot_df(hhmm)
                    rx = df["Range (m)"].to_numpy(float); y = df[hhmm].to_numpy(float)
                    m = np.isfinite(rx) & np.isfinite(y)
                    if m.any(): self.ax2.plot(rx[m], y[m], linewidth=1.2, label=hhmm)
                except Exception as e:
                    self.ax2.set_title(str(e))
            self.ax2.set_title(self.chart_title_var.get() or f"{self.data_source.get()} · {mode}")
        self.ax2.set_xlabel(self.x_title_var.get() or "Range (m)"); self.ax2.set_ylabel(self.y_title_var.get() or "Signal")
        self.ax2.grid(True, alpha=0.3)
        if mode == "Glue Overlay" or len(times) > 1: self.ax2.legend(fontsize=8, loc="best")
        self.apply_axes(redraw=False); self.fig2.tight_layout(); self.canvas2.draw_idle()

    def _parse_lim(self, s):
        s = s.strip(); return None if s == "" else float(s)
    def apply_axes(self, redraw=True):
        try:
            xmin = self._parse_lim(self.xmin_var.get()); xmax = self._parse_lim(self.xmax_var.get())
            ymin = self._parse_lim(self.ymin_var.get()); ymax = self._parse_lim(self.ymax_var.get())
            if xmin is not None or xmax is not None: self.ax2.set_xlim(left=xmin, right=xmax)
            if ymin is not None or ymax is not None: self.ax2.set_ylim(bottom=ymin, top=ymax)
            if redraw: self.canvas2.draw_idle()
        except Exception: pass
    def auto_scale(self):
        for v in [self.xmin_var, self.xmax_var, self.ymin_var, self.ymax_var]: v.set("")
        self.ax2.relim(); self.ax2.autoscale_view(); self.canvas2.draw_idle()
    def save_png(self):
        out = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png")])
        if out: self.fig2.savefig(out, dpi=150, bbox_inches="tight"); self._log(f"Saved PNG: {out}")
    def save_csv(self):
        times = self._selected_times()
        if not times: messagebox.showinfo("No selection","Select a time first."); return
        out = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if not out: return
        try: df = self._build_plot_df(times[0]); df.to_csv(out, index=False); self._log(f"Saved CSV: {out}")
        except Exception as e: messagebox.showerror("Error", str(e))

    def run(self):
        if not _HAS_NRB: messagebox.showerror("Missing","nrb_engine.py not found."); return
        folder = self.folder.get().strip(); out_path = self.out_path.get().strip()
        if not folder or not os.path.isdir(folder): messagebox.showerror("Error","Choose a valid folder."); return
        if not out_path: messagebox.showerror("Error","Choose output path."); return
        try:
            pd.to_datetime(self.date.get().strip())
            _parse_hhmm_text(self.start_time.get().strip(), field_name="Start time")
        except Exception as e:
            messagebox.showerror("Error", str(e)); return
        self.run_btn.configure(state="disabled"); self.progress.set(0.0)
        self._status_set("Running…","running") if self._status_set else None
        self._log("=== START ==="); self._plot_cache.clear(); self._refresh_time_list()
        def worker():
            try:
                profile_df, qc_df, params_df = nrb_build_daily_profile_from_folder(
                    Path(folder), date_str=self.date.get().strip(), pattern=self.pattern.get().strip() or "*.dat",
                    out_path=Path(out_path), start_time=self.start_time.get().strip(), dr_m=float(self.bin_spacing_m.get()),
                    dead_time_ns=float(self.dead_time_ns.get()), bg_mode=self.bg_mode.get().strip().lower(),
                    bg_start_m=float(self.bg_start_m.get()), bg_end_m=float(self.bg_end_m.get()),
                    pretrigger_bins=int(self.pretrigger_bins.get()), first_signal_bin=int(self.first_signal_bin.get()),
                    first_signal_range_m=float(self.first_signal_range_m.get()), blend_r1_m=float(self.blend_r1_m.get()),
                    blend_r2_m=float(self.blend_r2_m.get()), shift_mode="manual", bin_shift_bins=0,
                    sig_start_m=float(self.sig_start_m.get()), sig_end_m=float(self.sig_end_m.get()),
                    min_toggle_rate=float(self.min_toggle_rate.get()), max_toggle_rate=float(self.max_toggle_rate.get()),
                    auto_toggle_selector=bool(self.auto_toggle_selector.get()),
                    day_min_toggle_rate=float(self.day_min_toggle_rate.get()), day_max_toggle_rate=float(self.day_max_toggle_rate.get()),
                    toggle_bg_switch_threshold_mhz=float(self.toggle_bg_switch_threshold_mhz.get()),
                    pretrigger_trim_bins=int(self.pretrigger_trim_bins.get()),
                    energy_mj=float(self.energy_mj.get()), auto_blend=bool(self.auto_blend.get()),
                    gluing_mode=self._gluing_mode_str(),
                    strict=bool(self.strict.get()), logger=self._safe_log, progress_cb=lambda v: self._safe(self.progress.set, v))
                self._latest_profile_df = profile_df
                ts_cols = [pd.Timestamp(c) for c in profile_df.columns[1:] if pd.notna(pd.to_datetime(c, errors="coerce"))]
                interval_min = _infer_interval_minutes(ts_cols)
                self._safe(self._log, f"Profiles used: {len(ts_cols)}")
                if interval_min is not None:
                    self._safe(self._log, f"Inferred interval: {interval_min:g} min")
                self._safe(self._log, f"[OK] Saved: {Path(out_path).resolve()}")
                if self._status_set: self._safe(self._status_set, "Done ✅","ok")
                self._safe(self.progress.set, 100.0)
                self._safe(self.app_state.set_step2, str(Path(out_path).resolve()))
                self._safe(self._refresh_time_list)
                if self.time_list.size() > 0: self._safe(lambda: (self.time_list.selection_set(0), self.refresh_plot()))
                self._safe(messagebox.showinfo, "Step 2 Complete", f"Saved:\n{out_path}")
            except Exception as e:
                self._safe(self._log, f"[FAILED] {e}")
                if self._status_set: self._safe(self._status_set, "Failed ❌","error")
                self._safe(messagebox.showerror, "Failed", str(e))
            finally:
                self._safe(self.run_btn.configure, state="normal"); self._safe(self._log, "=== END ===")
        threading.Thread(target=worker, daemon=True).start()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 FRAME  –  ALT Calculator
# ═════════════════════════════════════════════════════════════════════════════
class Step3Frame(tk.Frame):
    def __init__(self, parent, app_state: AppState):
        super().__init__(parent, bg=BG)
        self.app_state = app_state
        self.engine_path = tk.StringVar(value=self._auto_engine())
        self.nrb_path = tk.StringVar(); self.nrb_sheet = tk.StringVar()
        self.rmin_path = tk.StringVar(); self.rmin_sheet = tk.StringVar()
        self.out_path = tk.StringVar(); self.tol_min = tk.StringVar(value="3")
        self.min_valid_frac = tk.StringVar(value="0.50"); self.min_valid_bins = tk.StringVar(value="50")
        self.detection_mode = tk.StringVar(value="dual")
        self.profile_rmin = tk.StringVar(value="0"); self.profile_rmax = tk.StringVar(value="4000")
        self._status_set = None
        self._build_ui()
        app_state.register_cb(self._on_state_change)

    def _auto_engine(self):
        for name in ENGINE_CANDIDATES:
            p = os.path.join(_DIR, name)
            if os.path.exists(p): return p
        return ""

    def _on_state_change(self):
        """Auto-populate paths when upstream steps complete."""
        if self.app_state.step2_output and not self.nrb_path.get():
            self.nrb_path.set(self.app_state.step2_output); self.load_nrb_sheets()
        if self.app_state.step1_output and not self.rmin_path.get():
            self.rmin_path.set(self.app_state.step1_output); self.load_rmin_sheets()

    def _build_ui(self):
        make_header(self, "Step 3  ·  ALT Calculator", "NRB profile → ALT, with optional MPL-guided rmin-rmax").pack(fill="x")
        main = tk.Frame(self, bg=BG); main.pack(fill="both", expand=True, padx=16, pady=14)
        main.columnconfigure(0, weight=3); main.columnconfigure(1, weight=2); main.rowconfigure(0, weight=1)

        left = tk.Frame(main, bg=BG); left.grid(row=0, column=0, sticky="nsew", padx=(0,10))

        c_e, f_e = make_card(left, "🔧  ALT Engine Script"); c_e.pack(fill="x", pady=(0,10))
        tk.Label(f_e, text="Python engine file (pbl_engine.py):", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(anchor="w")
        re_ = tk.Frame(f_e, bg=CARD); re_.pack(fill="x", pady=(4,0))
        ttk.Entry(re_, textvariable=self.engine_path).pack(side="left", fill="x", expand=True, padx=(0,8))
        ttk.Button(re_, text="Browse…", style="S.TButton", command=self.pick_engine).pack(side="right")
        if not self.engine_path.get():
            tk.Label(f_e, text="⚠  Engine not found. Browse to pbl_engine.py", bg=CARD, fg=WARNING, font=F_SMALL).pack(anchor="w", pady=(4,0))

        c_n, f_n = make_card(left, "📊  NRB Excel File"); c_n.pack(fill="x", pady=(0,10))
        rn = tk.Frame(f_n, bg=CARD); rn.pack(fill="x")
        ttk.Entry(rn, textvariable=self.nrb_path).pack(side="left", fill="x", expand=True, padx=(0,8))
        ttk.Button(rn, text="Browse…", style="S.TButton", command=self.pick_nrb).pack(side="right")
        rns = tk.Frame(f_n, bg=CARD); rns.pack(fill="x", pady=(8,0))
        tk.Label(rns, text="Sheet:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left", padx=(0,8))
        self.nrb_combo = ttk.Combobox(rns, textvariable=self.nrb_sheet, values=[], state="readonly", width=28)
        self.nrb_combo.pack(side="left")
        ttk.Button(rns, text="↻ Reload", style="Sm.TButton", command=self.load_nrb_sheets).pack(side="left", padx=(8,0))

        c_r, f_r = make_card(left, "📐  MPL rmin-rmax Excel File (optional for NRB-profile mode)"); c_r.pack(fill="x", pady=(0,10))
        rm = tk.Frame(f_r, bg=CARD); rm.pack(fill="x")
        ttk.Entry(rm, textvariable=self.rmin_path).pack(side="left", fill="x", expand=True, padx=(0,8))
        ttk.Button(rm, text="Browse…", style="S.TButton", command=self.pick_rmin).pack(side="right")
        rs = tk.Frame(f_r, bg=CARD); rs.pack(fill="x", pady=(8,0))
        tk.Label(rs, text="Sheet:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left", padx=(0,8))
        self.rmin_combo = ttk.Combobox(rs, textvariable=self.rmin_sheet, values=[], state="readonly", width=28)
        self.rmin_combo.pack(side="left")
        ttk.Button(rs, text="↻ Reload", style="Sm.TButton", command=self.load_rmin_sheets).pack(side="left", padx=(8,0))

        c_o, f_o = make_card(left, "💾  Output"); c_o.pack(fill="x", pady=(0,10))
        ro = tk.Frame(f_o, bg=CARD); ro.pack(fill="x")
        ttk.Entry(ro, textvariable=self.out_path).pack(side="left", fill="x", expand=True, padx=(0,8))
        ttk.Button(ro, text="Save As…", style="S.TButton", command=self.pick_output).pack(side="right")

        c_opt, f_opt = make_card(left, "⚙️  Options"); c_opt.pack(fill="x", pady=(0,10))

        mode_row = tk.Frame(f_opt, bg=CARD); mode_row.pack(fill="x", pady=(0,6))
        tk.Label(mode_row, text="ALT detection mode:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left", padx=(0,4))
        ttk.OptionMenu(mode_row, self.detection_mode, self.detection_mode.get(),
                       "dual", "mpl_guided", "nrb_profile").pack(side="left", padx=(0,10))
        tk.Label(mode_row, text="NRB range:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left", padx=(0,4))
        ttk.Entry(mode_row, textvariable=self.profile_rmin, width=7).pack(side="left", padx=(0,3))
        tk.Label(mode_row, text="to", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left", padx=(0,3))
        ttk.Entry(mode_row, textvariable=self.profile_rmax, width=7).pack(side="left", padx=(0,8))

        opt = tk.Frame(f_opt, bg=CARD); opt.pack(fill="x")
        for lbl, var, w in [("Tol (min):", self.tol_min, 6), ("min_valid_frac:", self.min_valid_frac, 6), ("min_valid_bins:", self.min_valid_bins, 6)]:
            tk.Label(opt, text=lbl, bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left", padx=(0,4))
            ttk.Entry(opt, textvariable=var, width=w).pack(side="left", padx=(0,16))

        c_run, f_run = make_card(left); c_run.pack(fill="x")
        self.run_btn = ttk.Button(f_run, text="▶  Calculate ALT", style="P.TButton", command=self.run_pbl)
        self.run_btn.pack(anchor="w")

        right = tk.Frame(main, bg=BG); right.grid(row=0, column=1, sticky="nsew")
        lh = tk.Frame(right, bg=BG); lh.pack(fill="x", pady=(0,6))
        tk.Label(lh, text="Console Log", bg=BG, fg=TEXT, font=F_H2).pack(side="left")
        ttk.Button(lh, text="Clear", style="Sm.TButton", command=self._clear_log).pack(side="right")
        lo, self.log_txt = make_log(right, height=32); lo.pack(fill="both", expand=True)
        sb, self.status_var, self._status_set = make_status_bar(self); sb.pack(fill="x", side="bottom")

    def _log(self, msg): append_log(self.log_txt, msg)
    def _safe(self, fn, *a, **kw): self.after(0, lambda: fn(*a, **kw))
    def _clear_log(self):
        self.log_txt.configure(state="normal"); self.log_txt.delete("1.0","end"); self.log_txt.configure(state="disabled")

    def pick_engine(self):
        p = filedialog.askopenfilename(title="Select engine", filetypes=[("Python","*.py")])
        if p: self.engine_path.set(p)
    def pick_nrb(self):
        p = filedialog.askopenfilename(title="NRB Excel", filetypes=[("Excel","*.xlsx *.xls")])
        if not p: return
        self.nrb_path.set(p); self.load_nrb_sheets()
        if not self.out_path.get():
            self.out_path.set(os.path.join(os.path.dirname(p), f"ALT_{os.path.splitext(os.path.basename(p))[0]}.xlsx"))
    def pick_rmin(self):
        p = filedialog.askopenfilename(title="rmin-rmax Excel", filetypes=[("Excel","*.xlsx *.xls")])
        if not p: return
        self.rmin_path.set(p); self.load_rmin_sheets()
    def pick_output(self):
        p = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel","*.xlsx")])
        if p: self.out_path.set(p)

    def load_nrb_sheets(self):
        p = self.nrb_path.get().strip()
        if not p: return
        try:
            sheets = list(pd.ExcelFile(p).sheet_names); self.nrb_combo["values"] = sheets
            self.nrb_sheet.set(PREF_NRB_SHEET if PREF_NRB_SHEET in sheets else (sheets[0] if sheets else ""))
            self._log(f"NRB sheets: {sheets}")
        except Exception as e: messagebox.showerror("Error", str(e))

    def load_rmin_sheets(self):
        p = self.rmin_path.get().strip()
        if not p: return
        try:
            sheets = list(pd.ExcelFile(p).sheet_names); self.rmin_combo["values"] = sheets
            self.rmin_sheet.set(PREF_RMIN_SHEET if PREF_RMIN_SHEET in sheets else (sheets[0] if sheets else ""))
            self._log(f"rmin sheets: {sheets}")
        except Exception as e: messagebox.showerror("Error", str(e))

    def run_pbl(self):
        engine = self.engine_path.get().strip(); nrb = self.nrb_path.get().strip()
        nrb_sheet = self.nrb_sheet.get().strip(); rmin = self.rmin_path.get().strip()
        rmin_sheet = self.rmin_sheet.get().strip(); out = self.out_path.get().strip()
        det_mode = self.detection_mode.get().strip() or "dual"
        need_rmin = det_mode in ("mpl_guided", "dual")
        for msg, cond in [("Select engine.", not engine or not os.path.exists(engine)),
                          ("NRB file not found.", not nrb or not os.path.exists(nrb)),
                          ("rmin-rmax file not found for MPL-guided / dual mode.", need_rmin and (not rmin or not os.path.exists(rmin))),
                          ("Select NRB sheet.", not nrb_sheet), ("Select rmin sheet for MPL-guided / dual mode.", need_rmin and not rmin_sheet),
                          ("Specify output path.", not out)]:
            if cond: messagebox.showerror("Input error", msg); return
        try:
            tol = float(self.tol_min.get()); mvf = float(self.min_valid_frac.get()); mvb = int(float(self.min_valid_bins.get()))
            prmin = float(self.profile_rmin.get()); prmax = float(self.profile_rmax.get())
        except Exception:
            tol, mvf, mvb, prmin, prmax = 3.0, 0.50, 50, 0.0, 4000.0
        self.run_btn.configure(state="disabled"); self._status_set("Running…","running")
        self._log("=== RUN START ==="); self._log(f"Engine: {engine}")
        def worker():
            tmp_path = None
            try:
                df_nrb = pd.read_excel(nrb, sheet_name=nrb_sheet).dropna(how="all").dropna(axis=1,how="all")
                if df_nrb.empty or df_nrb.shape[1] < 2: raise ValueError("NRB sheet empty.")
                prof_cols = list(df_nrb.columns[1:])
                base_dates = pd.to_datetime(prof_cols, errors="coerce")
                base_dates = base_dates[~pd.isna(base_dates)]
                base_date = (pd.Timestamp(base_dates[0]) if len(base_dates) > 0 else
                             (_s3_guess_date(nrb) or _s3_guess_date(rmin) or pd.Timestamp.today()))
                slot_times = _s3_parse_times(prof_cols, pd.Timestamp(base_date))
                new_cols = [df_nrb.columns[0]] + [pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S") for t in slot_times]
                df_nrb2 = df_nrb.copy(); df_nrb2.columns = new_cols
                if rmin and os.path.exists(rmin):
                    df_rmin = pd.read_excel(rmin, sheet_name=rmin_sheet).dropna(how="all").dropna(axis=1,how="all")
                    if df_rmin.empty: raise ValueError("rmin-rmax sheet empty.")
                    df_map = _s3_map_slots(df_rmin, slot_times, tol_min=tol)
                    matched = int(df_map["rmin"].notna().sum())
                else:
                    df_map = pd.DataFrame({
                        "Time": slot_times,
                        "Slot": [pd.Timestamp(t).strftime("%H:%M") for t in slot_times],
                        "PBL from MPL (m)": np.nan,
                        "rmin": np.nan,
                        "rmax": np.nan,
                        "MPL_Time": pd.NaT,
                        "dt_min": np.nan,
                    })
                    matched = 0
                self._safe(self._log, f"Matched MPL/rmin-rmax: {matched}/{len(slot_times)} slots")
                tmp_fd, tmp_path = tempfile.mkstemp(prefix="pbl_", suffix=".xlsx"); os.close(tmp_fd)
                with pd.ExcelWriter(tmp_path, engine="openpyxl") as w:
                    df_nrb2.to_excel(w, sheet_name=PREF_NRB_SHEET, index=False)
                    df_map.to_excel(w, sheet_name=PREF_RMIN_SHEET, index=False)
                cmd = [sys.executable, engine, "--nrb", tmp_path, "--sheet", PREF_NRB_SHEET,
                       "--rminrmax_sheet", PREF_RMIN_SHEET, "--out", out,
                       "--min_valid_frac", str(mvf), "--min_valid_bins", str(mvb),
                       "--detection_mode", det_mode,
                       "--profile_rmin", str(prmin), "--profile_rmax", str(prmax)]
                self._safe(self._log, "CMD: " + " ".join(cmd))
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.stdout: self._safe(self._log, proc.stdout.strip())
                if proc.stderr: self._safe(self._log, proc.stderr.strip())
                if proc.returncode == 0:
                    self._safe(self._log, "=== RUN OK ==="); self._safe(self._status_set, "Done ✅","ok")
                    self._safe(self.app_state.set_step3, str(Path(out).resolve()))
                    self._safe(messagebox.showinfo, "Step 3 Complete", f"Saved:\n{out}")
                else:
                    self._safe(self._log, "=== RUN FAILED ==="); self._safe(self._status_set, "Failed ❌","error")
                    self._safe(messagebox.showerror, "Failed", "See log for details.")
            except Exception as e:
                self._safe(self._log, f"[FAILED] {e}"); self._safe(self._status_set, "Failed ❌","error")
                self._safe(messagebox.showerror, "Error", str(e))
            finally:
                try:
                    if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)
                except Exception: pass
                self._safe(self.run_btn.configure, state="normal")
        threading.Thread(target=worker, daemon=True).start()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 FRAME  –  RTI Visualizer
# ═════════════════════════════════════════════════════════════════════════════
class Step4Frame(tk.Frame):
    def __init__(self, parent, app_state: AppState):
        super().__init__(parent, bg=BG)
        self.app_state = app_state
        self._init_data()
        self._init_vars()
        self._build_ui()
        self._init_blank_plot()

    def _init_data(self):
        self.proto_path = self.mpl_path = None; self.proto_date = None
        self.r_proto = self.t_proto = self.t_labels_hhmm = self.t_list_labels = None
        self.Z_proto = self.r_mpl = self.t_mpl = self.Z_mpl = None
        self.match_idx = self.dt_min = self.Z_mpl_on_proto = None
        self.alt_df = self.alt_proto_by_slot = self.alt_mpl_by_slot = None
        self.r_proto_den_src = self.t_proto_den_src = self.Z_proto_den_src = None
        self.Z_proto_den = None

    def _init_vars(self):
        self.date_str = tk.StringVar(value="")
        self.show_proto_var = tk.BooleanVar(value=True); self.show_mpl_var = tk.BooleanVar(value=True)
        self.show_profile_var = tk.BooleanVar(value=True); self.profile_proto_mode_var = tk.StringVar(value="raw")
        self.show_alt_overlay_var = tk.BooleanVar(value=False); self.show_alt_compare_var = tk.BooleanVar(value=False)
        self.max_dt_min_var = tk.StringVar(value="3"); self.alt_path_var = tk.StringVar(value="")
        self.alt_proto_col = tk.StringVar(value=ALT_COL_PROTO_CHOICES[0])
        self.cmap_var = tk.StringVar(value="jet"); self.interp_var = tk.StringVar(value="nearest")
        self.discrete_var = tk.BooleanVar(value=False); self.step_var = tk.StringVar(value="0.05")
        self.use_log_color_var = tk.BooleanVar(value=False)
        self.vmin_var = tk.DoubleVar(value=0.0); self.vmax_var = tk.DoubleVar(value=1.0)
        self.scale_mode_var = tk.StringVar(value="preset")
        self.rti_tickN_var = tk.StringVar(value="4"); self.rti_ymin_var = tk.StringVar(value="0")
        self.rti_ymax_var = tk.StringVar(value="15000"); self.rti_ytick_var = tk.StringVar(value="3000")
        self.prof_xmin_var = tk.StringVar(value="0"); self.prof_xmax_var = tk.StringVar(value="15000")
        self.prof_ymin_var = tk.StringVar(value="0"); self.prof_ymax_var = tk.StringVar(value="1")
        self.prof_ytick_var = tk.StringVar(value="0.1")
        self.alt_tickN_var = tk.StringVar(value="4"); self.alt_ymin_var = tk.StringVar(value="0")
        self.alt_ymax_var = tk.StringVar(value="4000"); self.alt_ytick_var = tk.StringVar(value="500")
        self.log_nrb_axis_var = tk.BooleanVar(value=False); self.log_base_var = tk.StringVar(value="10")
        self.title_rti_proto_var = tk.StringVar(value="LiDAR Prototype"); self.title_rti_mpl_var = tk.StringVar(value="Mini MPL")
        self.title_prof_var = tk.StringVar(value="Prototype NRB vs MPL copol NRB")
        self.title_altcmp_var = tk.StringVar(value="MPL ALT vs Prototype ALT")
        self.rti_xlabel_var = tk.StringVar(value="Time (HH:MM)"); self.rti_ylabel_var = tk.StringVar(value="Range (m)")
        self.prof_xlabel_var = tk.StringVar(value="Range (m)"); self.prof_ylabel_var = tk.StringVar(value="NRB")
        self.alt_xlabel_var = tk.StringVar(value="Time (HH:MM)"); self.alt_ylabel_var = tk.StringVar(value="Height (m)")
        self.cbar_label_var = tk.StringVar(value="NRB (0–1)"); self.status_var = tk.StringVar(value="Ready.")

    def _build_ui(self):
        # Top header
        top = tk.Frame(self, bg=DARK); top.pack(fill="x")
        tk.Label(top, text="Step 4  ·  RTI Visualizer  —  Prototype vs Mini MPL",
                 bg=DARK, fg="white", font=F_H2, padx=18, pady=10).pack(side="left")
        tk.Label(top, textvariable=self.date_str, bg=DARK, fg="#BFDBFE", font=F_SMALL, padx=8).pack(side="left")
        tk.Label(top, textvariable=self.status_var, bg=DARK, fg="#E0F2FE", font=F_SMALL, padx=18).pack(side="right")

        main = ttk.Panedwindow(self, orient="horizontal"); main.pack(fill="both", expand=True)
        lw = tk.Frame(main, bg=BG, width=400); lw.pack_propagate(False)
        lsf = ScrollableFrame(lw, bg=BG); lsf.pack(fill="both", expand=True)
        left = lsf.inner; left.columnconfigure(0, weight=1)
        right = tk.Frame(main, bg=BG, padx=6, pady=6)
        main.add(lw, weight=0); main.add(right, weight=1)

        # Plot area
        pw = tk.Frame(right, bg=BG); pw.pack(fill="both", expand=True)
        tw = tk.Frame(pw, bg=BG); tw.pack(side="bottom", fill="x")
        self.fig = plt.Figure(figsize=(12, 8.5)); self._setup_axes()
        self.cbar1 = self.cbar2 = None; self.im1 = self.im2 = None; self.sel_lines = []; self._mpl_cid = None
        self.canvas = FigureCanvasTkAgg(self.fig, master=pw); self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, tw, pack_toolbar=False)
        self.toolbar.update(); self.toolbar.pack(side="left", fill="x")

        PAD = dict(padx=8, pady=4, sticky="ew")

        # Load files
        wi, ci = make_card(left, title="📂  Load Files"); wi.grid(row=0, column=0, **PAD)
        br = tk.Frame(ci, bg=CARD); br.pack(fill="x", pady=(0,6))
        ttk.Button(br, text="Load Prototype", style="P.TButton", command=self.load_prototype).pack(side="left", padx=(0,6))
        ttk.Button(br, text="Load Folder", style="S.TButton", command=self.load_prototype_folder).pack(side="left", padx=(0,8))
        ttk.Button(br, text="Load MPL", style="S.TButton", command=self.load_mpl).pack(side="left")
        ar = tk.Frame(ci, bg=CARD); ar.pack(fill="x")
        tk.Label(ar, text="Match ±", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(ar, textvariable=self.max_dt_min_var, width=5).pack(side="left", padx=(2,2))
        tk.Label(ar, text="min", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left", padx=(0,8))
        ttk.Button(ar, text="↺ Re-align", style="S.TButton", command=self.align_and_refresh).pack(side="left")

        # ALT
        wa, ca = make_card(left, title="📊  ALT Results"); wa.grid(row=1, column=0, **PAD)
        aef = tk.Frame(ca, bg=CARD); aef.pack(fill="x", pady=(0,4)); aef.columnconfigure(0, weight=1)
        ttk.Entry(aef, textvariable=self.alt_path_var).grid(row=0, column=0, sticky="ew", padx=(0,6))
        ttk.Button(aef, text="Load ALT", style="S.TButton", command=self.load_alt).grid(row=0, column=1)
        acr = tk.Frame(ca, bg=CARD); acr.pack(anchor="w")
        tk.Label(acr, text="Proto ALT mode/column:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left", padx=(0,6))
        self.alt_proto_combo = ttk.Combobox(acr, textvariable=self.alt_proto_col, values=ALT_COL_PROTO_CHOICES, state="readonly", width=22)
        self.alt_proto_combo.pack(side="left"); self.alt_proto_combo.bind("<<ComboboxSelected>>", lambda _: self.apply_alt_settings())

        # Show/Hide
        ws, cs = make_card(left, title="👁  Show / Hide"); ws.grid(row=2, column=0, **PAD)
        for txt, var in [("Prototype RTI", self.show_proto_var), ("Mini MPL RTI", self.show_mpl_var),
                         ("Profile view", self.show_profile_var), ("ALT overlay (red line)", self.show_alt_overlay_var),
                         ("ALT Compare (time series)", self.show_alt_compare_var)]:
            ttk.Checkbutton(cs, text=txt, variable=var, command=self.refresh_all).pack(anchor="w", pady=1)

        # Time selector
        wt, ct = make_card(left, title="🕐  Profile Time"); wt.grid(row=3, column=0, **PAD)
        lf = tk.Frame(ct, bg=CARD); lf.pack(fill="both", expand=True); lf.columnconfigure(0, weight=1)
        self.time_list = tk.Listbox(lf, height=8, selectmode=tk.EXTENDED, exportselection=False,
                                    bg=CARD, fg=TEXT, font=F_SMALL, selectbackground=PRIMARY, selectforeground="white",
                                    activestyle="none", relief="flat", bd=0)
        sb_l = ttk.Scrollbar(lf, orient="vertical", command=self.time_list.yview)
        self.time_list.configure(yscrollcommand=sb_l.set)
        self.time_list.grid(row=0, column=0, sticky="nsew"); sb_l.grid(row=0, column=1, sticky="ns")
        self.time_list.bind("<<ListboxSelect>>", lambda _: self.update_profile_plot(True))
        mr = tk.Frame(ct, bg=CARD); mr.pack(fill="x", pady=(6,0))
        tk.Label(mr, text="Proto profile:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Radiobutton(mr, text="Raw", value="raw", variable=self.profile_proto_mode_var, command=self.refresh_all).pack(side="left", padx=(8,0))
        ttk.Radiobutton(mr, text="Denoised", value="denoised", variable=self.profile_proto_mode_var, command=self.refresh_all).pack(side="left", padx=(8,0))

        # Save
        wv, cv = make_card(left, title="💾  Save"); wv.grid(row=4, column=0, **PAD)
        bf = tk.Frame(cv, bg=CARD); bf.pack(fill="x")
        ttk.Button(bf, text="Save Figure PNG", style="P.TButton", command=self.save_png).pack(side="left", padx=(0,8))
        ttk.Button(bf, text="Save Profile PNG", style="S.TButton", command=self.save_profile_png).pack(side="left")

        # Display
        wd, cd = make_card(left, title="🎨  Display"); wd.grid(row=5, column=0, **PAD)
        d0 = tk.Frame(cd, bg=CARD); d0.pack(fill="x", pady=(0,4))
        tk.Label(d0, text="CMap", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.OptionMenu(d0, self.cmap_var, self.cmap_var.get(), *CMAP_CHOICES).pack(side="left", padx=(4,14))
        tk.Label(d0, text="Interp", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.OptionMenu(d0, self.interp_var, self.interp_var.get(), *INTERP_CHOICES).pack(side="left", padx=(4,0))
        d1 = tk.Frame(cd, bg=CARD); d1.pack(fill="x", pady=(0,4))
        ttk.Checkbutton(d1, text="Discrete", variable=self.discrete_var).pack(side="left")
        tk.Label(d1, text="Step", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left", padx=(8,4))
        ttk.Entry(d1, textvariable=self.step_var, width=7).pack(side="left")
        ttk.Checkbutton(d1, text="Log colour", variable=self.use_log_color_var).pack(side="left", padx=(10,0))
        d2 = tk.Frame(cd, bg=CARD); d2.pack(fill="x", pady=(0,4))
        for lbl, var in [("vmin", self.vmin_var), ("vmax", self.vmax_var)]:
            tk.Label(d2, text=lbl, bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
            tk.Scale(d2, from_=-1, to=1, resolution=0.01, orient="horizontal", variable=var,
                     command=lambda _=None: self._on_manual_scale(), bg=CARD, length=160, showvalue=True,
                     highlightthickness=0).pack(side="left", padx=(4,10))
        d3 = tk.Frame(cd, bg=CARD); d3.pack(fill="x", pady=(0,6))
        tk.Label(d3, text="Scale:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        for val, txt in [("preset","0–1"),("auto","Auto"),("manual","Manual")]:
            ttk.Radiobutton(d3, text=txt, value=val, variable=self.scale_mode_var).pack(side="left", padx=(6,0))
        ttk.Button(cd, text="Apply Display", style="P.TButton", command=self.apply_display).pack(anchor="w")

        # Axis
        wax, cax_f = make_card(left, title="📐  Axis Settings"); wax.grid(row=6, column=0, **PAD)
        def arow(parent, pairs):
            f = tk.Frame(parent, bg=CARD); f.pack(fill="x", pady=2)
            for lbl, var, w in pairs:
                tk.Label(f, text=lbl, bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left", padx=(0,3))
                ttk.Entry(f, textvariable=var, width=w).pack(side="left", padx=(0,10))
        arow(cax_f, [("RTI tickN",self.rti_tickN_var,5),("RTI ytick",self.rti_ytick_var,7)])
        arow(cax_f, [("RTI ymin",self.rti_ymin_var,7),("RTI ymax",self.rti_ymax_var,7)])
        arow(cax_f, [("Prof xmin",self.prof_xmin_var,7),("Prof xmax",self.prof_xmax_var,7)])
        arow(cax_f, [("Prof ymin",self.prof_ymin_var,7),("Prof ymax",self.prof_ymax_var,7)])
        arow(cax_f, [("Prof ytick",self.prof_ytick_var,6),("Log base",self.log_base_var,6)])
        ttk.Checkbutton(cax_f, text="Log NRB (Profile Y)", variable=self.log_nrb_axis_var).pack(anchor="w", pady=(2,4))
        arow(cax_f, [("ALT tickN",self.alt_tickN_var,5),("ALT ytick",self.alt_ytick_var,7)])
        arow(cax_f, [("ALT ymin",self.alt_ymin_var,7),("ALT ymax",self.alt_ymax_var,7)])
        ttk.Button(cax_f, text="Apply Axis", style="P.TButton", command=self.apply_axis).pack(anchor="w", pady=(6,0))

        # Titles & labels
        wtl, ctl = make_card(left, title="✏️  Titles & Labels"); wtl.grid(row=7, column=0, **PAD)
        def trow(parent, pairs):
            f = tk.Frame(parent, bg=CARD); f.pack(fill="x", pady=2)
            for lbl, var in pairs:
                tk.Label(f, text=lbl, bg=CARD, fg=SUBTEXT, font=F_SMALL, width=12, anchor="w").pack(side="left")
                ttk.Entry(f, textvariable=var, width=22).pack(side="left", padx=(0,12))
        trow(ctl, [("RTI Prototype", self.title_rti_proto_var), ("RTI MPL", self.title_rti_mpl_var)])
        trow(ctl, [("Profile", self.title_prof_var), ("ALT Compare", self.title_altcmp_var)])
        trow(ctl, [("RTI X label", self.rti_xlabel_var), ("RTI Y label", self.rti_ylabel_var)])
        trow(ctl, [("Prof X label", self.prof_xlabel_var), ("Prof Y label", self.prof_ylabel_var)])
        trow(ctl, [("ALT X label", self.alt_xlabel_var), ("ALT Y label", self.alt_ylabel_var)])
        trow(ctl, [("CBar label", self.cbar_label_var)])
        ttk.Button(ctl, text="Apply Titles", style="P.TButton", command=self.apply_titles).pack(anchor="w", pady=(6,0))

        tk.Frame(left, bg=BG, height=20).grid(row=8, column=0)

    def _setup_axes(self):
        self.fig.clear()
        self.ax_rti1 = self.fig.add_subplot(411); self.ax_rti2 = self.fig.add_subplot(412)
        self.ax_prof  = self.fig.add_subplot(413); self.ax_altcmp = self.fig.add_subplot(414)
        self.cax1 = self.fig.add_axes([0.935, 0.70, 0.015, 0.17])
        self.cax2 = self.fig.add_axes([0.935, 0.50, 0.015, 0.17])

    def _read_wide(self, path, sheet, base_date=None):
        df = pd.read_excel(path, sheet_name=sheet); rc = df.columns[0]; tc = df.columns[1:]
        r = pd.to_numeric(df[rc], errors="coerce").to_numpy(float)
        Z = df[tc].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        base = base_date or (_r4_first_ts(tc) or pd.Timestamp.today())
        t = _r4_parse_tcols(tc, base); return r, t, Z

    def _merge_nrb_files(self, file_list, sheet):
        """Read one or more NRB Excel files and produce a unified (r, t_list, Z) tuple.

        - Single file: preserve the file's original time columns exactly (NaN columns from
          the source Excel still produce black bands as before).
        - Multiple files: build a continuous time grid anchored on the first valid record
          (so the data's natural phase like HH:05 / HH:35 is preserved) with NaN columns
          in any expected slot that has no data.
        """
        if not file_list:
            raise ValueError("No files provided.")
        parts = [self._read_wide(fp, sheet) for fp in file_list]
        r_common = parts[0][0]
        nR = int(len(r_common))

        # Single-file fast path: keep original layout verbatim (matches pre-merge behavior)
        if len(file_list) == 1:
            r, t, Z = parts[0]
            same_r = (len(r) == nR) and bool(np.allclose(r, r_common, equal_nan=True))
            if same_r:
                return r_common, list(t), np.asarray(Z, float)
            nT = int(Z.shape[1])
            Z_out = np.full((nR, nT), np.nan, dtype=float)
            for j in range(nT):
                Z_out[:, j] = _r4_interp(r, Z[:, j], r_common)
            return r_common, list(t), Z_out

        # Multi-file: collect valid (timestamp, column) records on the common range axis
        records = []
        for r, t, Z in parts:
            same_r = (len(r) == nR) and bool(np.allclose(r, r_common, equal_nan=True))
            for j, ts in enumerate(t):
                if pd.isna(ts):
                    continue
                col = Z[:, j] if same_r else _r4_interp(r, Z[:, j], r_common)
                records.append((pd.Timestamp(ts), col))
        if not records:
            raise ValueError("No valid timestamped columns found in selected files.")
        records.sort(key=lambda x: x[0])

        interval_min = _r4_detect_interval_min([rec[0] for rec in records])
        # Anchor the grid on the FIRST valid record so we keep the natural phase
        # (e.g. data at 00:05, 00:35, ... stays on those exact slots).
        t_anchor = records[0][0]
        t_last = records[-1][0]
        n_slots = int(np.ceil((t_last - t_anchor).total_seconds() / 60.0 / float(interval_min))) + 1
        expected = [t_anchor + pd.Timedelta(minutes=interval_min * k) for k in range(max(1, n_slots))]
        Z_out = np.full((nR, len(expected)), np.nan, dtype=float)

        expected_ns = np.array([e.value for e in expected], dtype=np.int64)
        tol_ns = int(interval_min * 30 * 1e9)  # half slot, in ns
        for ts, col in records:
            idx = int(np.argmin(np.abs(expected_ns - ts.value)))
            if abs(expected_ns[idx] - ts.value) <= tol_ns:
                Z_out[:, idx] = col

        return r_common, [pd.Timestamp(e) for e in expected], Z_out

    def _load_prototype_files(self, file_list):
        try:
            file_list = list(file_list)
            r, t, Z = self._merge_nrb_files(file_list, PROTO_SHEET)
            base = next((x for x in t if not pd.isna(x)), pd.Timestamp.today())
            self.proto_date = pd.Timestamp(base); self.date_str.set(self.proto_date.strftime("%Y-%m-%d"))
            lhhmm = _r4_build_x_labels(t)
            llist = [pd.Timestamp(x).strftime("%Y-%m-%d %H:%M") if not pd.isna(x) else "NaT" for x in t]
            self.proto_path = file_list[0] if len(file_list) == 1 else f"{len(file_list)} files merged"
            self.r_proto = r; self.Z_proto = Z; self.t_proto = t
            self.t_labels_hhmm = lhhmm; self.t_list_labels = llist
            self.time_list.delete(0, tk.END)
            for lab in llist: self.time_list.insert(tk.END, lab)
            if llist: self.time_list.selection_set(0)
            self._rebuild_denoised(); self._align_alt()
            if len(file_list) == 1:
                self.status_var.set(f"Prototype: {os.path.basename(file_list[0])}")
            else:
                span = f"{lhhmm[0]} → {lhhmm[-1]}" if lhhmm else ""
                self.status_var.set(f"Prototype: {len(file_list)} files merged ({span})")
            self.align_and_refresh()
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def load_prototype(self):
        fps = filedialog.askopenfilenames(title="Prototype NRB Excel (Ctrl+click for multiple days)", filetypes=[("Excel","*.xlsx *.xls")])
        if not fps: return
        self._load_prototype_files(fps)

    def load_prototype_folder(self):
        folder = filedialog.askdirectory(title="Folder containing NRB-*.xlsx files")
        if not folder: return
        files = sorted(Path(folder).glob("NRB-*.xlsx"))
        if not files:
            files = sorted(Path(folder).glob("*.xlsx"))
        if not files:
            messagebox.showerror("No files", f"No .xlsx files found in:\n{folder}")
            return
        self._load_prototype_files([str(f) for f in files])

    def load_mpl(self):
        fp = filedialog.askopenfilename(title="MPL Excel", filetypes=[("Excel","*.xlsx *.xls")])
        if not fp: return
        try:
            df = pd.read_excel(fp, sheet_name=MPL_SHEET); rc = df.columns[0]; tc = df.columns[1:]
            r = pd.to_numeric(df[rc], errors="coerce").to_numpy(float)
            Z = df[tc].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            base = self.proto_date or (_r4_first_ts(tc) or pd.Timestamp.today())
            t = _r4_parse_tcols(tc, base)
            self.mpl_path = fp; self.r_mpl = r; self.Z_mpl = Z; self.t_mpl = t
            self.status_var.set(f"MPL: {os.path.basename(fp)}"); self.align_and_refresh()
        except Exception as e: messagebox.showerror("Load error", str(e))

    def load_alt(self):
        fp = filedialog.askopenfilename(title="ALT Excel", filetypes=[("Excel","*.xlsx *.xls")])
        if not fp: return
        try:
            xl = pd.ExcelFile(fp)
            sheet = ALT_SHEET if ALT_SHEET in xl.sheet_names else (LEGACY_ALT_SHEET if LEGACY_ALT_SHEET in xl.sheet_names else xl.sheet_names[0])
            df = pd.read_excel(fp, sheet_name=sheet)
            if "Time" not in df.columns: raise ValueError("ALT sheet must have 'Time' column.")
            df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
            df = df.dropna(subset=["Time"]).reset_index(drop=True)
            self.alt_df = df; self.alt_path_var.set(fp)
            combo_values = [c for c in ALT_COL_PROTO_CHOICES if c in df.columns]
            if combo_values:
                self.alt_proto_combo["values"] = combo_values
                if self.alt_proto_col.get() not in combo_values:
                    self.alt_proto_col.set(combo_values[0])
            self.r_proto_den_src = self.t_proto_den_src = self.Z_proto_den_src = None; self.Z_proto_den = None
            den_ok = False
            try:
                r_d, t_d, Z_d = self._read_wide(fp, DENOISED_SHEET, self.proto_date)
                self.r_proto_den_src = r_d; self.t_proto_den_src = t_d; self.Z_proto_den_src = Z_d
                self._rebuild_denoised(); den_ok = self.Z_proto_den is not None
            except Exception: pass
            self._align_alt()
            msg = f"ALT: {os.path.basename(fp)}"
            if den_ok: msg += f" + {DENOISED_SHEET}"
            self.status_var.set(msg); self.refresh_all()
        except Exception as e: messagebox.showerror("Load error", str(e))

    def _rebuild_denoised(self):
        self.Z_proto_den = None
        if self.r_proto is None or self.Z_proto_den_src is None: return
        mx = _r4_int(self.max_dt_min_var.get()) or 3
        mi, _ = _r4_nearest(self.t_proto, self.t_proto_den_src, max_min=mx)
        nR = self.r_proto.size; nT = len(self.t_proto); Zd = np.full((nR, nT), np.nan)
        for i in range(nT):
            j = mi[i]
            if j is not None: Zd[:, i] = _r4_interp(self.r_proto_den_src, self.Z_proto_den_src[:, j], self.r_proto)
        self.Z_proto_den = Zd

    def _align_alt(self):
        self.alt_proto_by_slot = self.alt_mpl_by_slot = None
        if self.alt_df is None or self.t_proto is None: return
        try: tol = float(self.max_dt_min_var.get() or "3")
        except Exception: tol = 3.0
        df = self.alt_df; times = df["Time"]; cp = self.alt_proto_col.get().strip()
        if cp not in df.columns:
            for c in ALT_COL_PROTO_CHOICES:
                if c in df.columns: cp = c; self.alt_proto_col.set(c); break
        cm = next((c for c in ALT_COL_MPL_CHOICES if c in df.columns), None)
        nT = len(self.t_proto); ap = np.full(nT, np.nan); am = np.full(nT, np.nan)
        if times is None or len(times) == 0 or times.isna().all():
            self.alt_proto_by_slot = ap; self.alt_mpl_by_slot = am
            return
        for i, ts in enumerate(self.t_proto):
            if pd.isna(ts): continue
            try:
                ts = pd.Timestamp(ts)
                dt = (times - ts).dt.total_seconds().abs() / 60.0
                if dt.isna().all():
                    continue
                j = int(dt.idxmin())
            except Exception:
                continue
            if float(dt.loc[j]) <= tol:
                if cp in df.columns: ap[i] = pd.to_numeric(df.loc[j, cp], errors="coerce")
                if cm: am[i] = pd.to_numeric(df.loc[j, cm], errors="coerce")
        self.alt_proto_by_slot = ap; self.alt_mpl_by_slot = am

    def align_and_refresh(self):
        if self.Z_proto is None or self.Z_mpl is None: self.refresh_all(); return
        mx = _r4_int(self.max_dt_min_var.get()) or 3
        self.match_idx, self.dt_min = _r4_nearest(self.t_proto, self.t_mpl, max_min=mx)
        nR = self.r_proto.size; nT = len(self.t_proto); Zm = np.full((nR, nT), np.nan)
        for i in range(nT):
            j = self.match_idx[i]
            if j is not None: Zm[:, i] = _r4_interp(self.r_mpl, self.Z_mpl[:, j], self.r_proto)
        self.Z_mpl_on_proto = Zm; self._rebuild_denoised(); self._align_alt()
        n_ok = sum(x is not None for x in self.match_idx)
        self.status_var.set(f"Aligned: {n_ok}/{nT} matched (±{mx} min)"); self.refresh_all()

    def apply_display(self): self._apply_scale_mode(); self.refresh_all()
    def apply_axis(self): self.refresh_all()
    def apply_titles(self): self.refresh_all()
    def apply_alt_settings(self): self._align_alt(); self.refresh_all()
    def _on_manual_scale(self):
        if self.scale_mode_var.get() != "manual": self.scale_mode_var.set("manual")
        self.refresh_all()

    def _apply_scale_mode(self):
        mode = self.scale_mode_var.get()
        if mode == "preset": self.vmin_var.set(0.0); self.vmax_var.set(1.0)
        elif mode == "auto":
            vals = []
            for Z in (self.Z_proto, self.Z_mpl_on_proto):
                if Z is not None:
                    v = Z[np.isfinite(Z)]
                    if v.size: vals.append(v)
            if vals:
                allv = np.concatenate(vals); p1, p99 = np.percentile(allv, [1, 99])
                self.vmin_var.set(float(p1)); self.vmax_var.set(float(p99))

    def _get_cmap(self):
        name = self.cmap_var.get().strip()
        cmap = plt.get_cmap(name if name in CMAP_CHOICES else "jet").copy(); cmap.set_bad("black"); return cmap

    def _get_norm(self, vmin, vmax, cmap):
        if self.use_log_color_var.get():
            return mcolors.LogNorm(vmin=max(vmin, 1e-8), vmax=max(vmax, max(vmin, 1e-8) * 10))
        if self.discrete_var.get():
            step = _r4_float(self.step_var.get()) or 0.05
            if step <= 0: step = 0.05
            levels = np.arange(vmin, vmax + step, step)
            if levels.size < 2: levels = np.array([vmin, vmax])
            return mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    def _mask(self, Z):
        Z = np.asarray(Z, float)
        if self.use_log_color_var.get(): return np.ma.masked_where(~np.isfinite(Z) | (Z <= 0), Z)
        return np.ma.masked_invalid(Z)

    def _init_blank_plot(self):
        for ax in (self.ax_rti1, self.ax_rti2, self.ax_prof, self.ax_altcmp): ax.clear()
        self._reset_cbars(); self._apply_layout(); self.canvas.draw_idle()

    def _reset_cbars(self):
        self.cax1.cla(); self.cax2.cla(); self.cbar1 = self.cbar2 = None
        self.cax1.set_visible(False); self.cax2.set_visible(False)

    def _apply_layout(self):
        vis = []
        if self.show_proto_var.get(): vis.append(("rti1", self.ax_rti1))
        if self.show_mpl_var.get(): vis.append(("rti2", self.ax_rti2))
        if self.show_profile_var.get(): vis.append(("prof", self.ax_prof))
        if self.show_alt_compare_var.get(): vis.append(("altcmp", self.ax_altcmp))
        for ax in (self.ax_rti1, self.ax_rti2, self.ax_prof, self.ax_altcmp):
            ax.set_visible(False)
        if not vis:
            return

        left_ = 0.07
        right_ = 0.885
        top_ = 0.965
        bottom_ = 0.075
        weights = [1.10 if n.startswith("rti") else 0.72 for n, _ in vis]

        gaps = []
        for i in range(len(vis) - 1):
            cur_name = vis[i][0]
            next_name = vis[i + 1][0]
            if cur_name.startswith("rti") and next_name.startswith("rti"):
                gaps.append(0.135)
            elif cur_name.startswith("rti") or next_name.startswith("rti"):
                gaps.append(0.125)
            else:
                gaps.append(0.105)
        total_gap = sum(gaps)
        avail = (top_ - bottom_) - total_gap
        if avail <= 0.2:
            gaps = [0.09] * max(0, len(vis) - 1)
            total_gap = sum(gaps)
            avail = (top_ - bottom_) - total_gap
        sumw = float(sum(weights))
        heights = [avail * (w / sumw) for w in weights]
        y = top_
        for i, ((nm, ax), h) in enumerate(zip(vis, heights)):
            ax.set_position([left_, y - h, right_ - left_, h])
            ax.set_visible(True)
            if i < len(gaps):
                y = y - h - gaps[i]
            else:
                y = y - h
        self._place_cbars()

    def _place_cbars(self):
        pad, w = 0.012, 0.016
        self.cax1.set_visible(False)
        self.cax2.set_visible(False)
        rti_axes = []
        if self.show_proto_var.get() and self.ax_rti1.get_visible():
            rti_axes.append(self.ax_rti1)
        if self.show_mpl_var.get() and self.ax_rti2.get_visible():
            rti_axes.append(self.ax_rti2)
        caxes = [self.cax1, self.cax2]
        for ax, cax in zip(rti_axes, caxes):
            p = ax.get_position()
            cax.set_position([p.x1 + pad, p.y0, w, p.height])
            cax.set_visible(True)

    def _rti_style(self, ax, *, show_xlabel: bool = True, show_xticks: bool = True):
        ax.set_xlabel(self.rti_xlabel_var.get() if show_xlabel else "", labelpad=8)
        ax.set_ylabel(self.rti_ylabel_var.get())
        ax.tick_params(axis="y", labelsize=8)
        if self.t_labels_hhmm:
            stepN = max(1, _r4_int(self.rti_tickN_var.get()) or 4)
            xt = np.arange(0, len(self.t_labels_hhmm), stepN)
            ax.set_xticks(xt)
            if show_xticks:
                ax.set_xticklabels([self.t_labels_hhmm[i] for i in xt], rotation=30, ha="right")
                ax.tick_params(axis="x", labelsize=7, labelbottom=True, bottom=True, pad=1)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", labelbottom=False, bottom=False)
        ymin = _r4_float(self.rti_ymin_var.get()); ymax = _r4_float(self.rti_ymax_var.get())
        ytick = _r4_float(self.rti_ytick_var.get())
        if ymin is not None and ymax is not None and ymax > ymin:
            ax.set_ylim(ymin, ymax)
        if ytick and ytick > 0:
            ax.yaxis.set_major_locator(mticker.MultipleLocator(ytick))

    def _update_cbars(self):
        label = self.cbar_label_var.get()
        self.cbar1 = None
        self.cbar2 = None
        visible_images = []
        if self.show_proto_var.get() and self.ax_rti1.get_visible() and self.im1 is not None:
            visible_images.append((self.cax1 if not visible_images else self.cax2, self.im1))
        if self.show_mpl_var.get() and self.ax_rti2.get_visible() and self.im2 is not None:
            visible_images.append((self.cax1 if not visible_images else self.cax2, self.im2))
        for cax in (self.cax1, self.cax2):
            if cax.get_visible():
                cax.cla()
        if len(visible_images) >= 1 and visible_images[0][0].get_visible():
            self.cbar1 = self.fig.colorbar(visible_images[0][1], cax=visible_images[0][0])
            self.cbar1.set_label(label)
        if len(visible_images) >= 2 and visible_images[1][0].get_visible():
            self.cbar2 = self.fig.colorbar(visible_images[1][1], cax=visible_images[1][0])
            self.cbar2.set_label(label)

    def _style_altcmp(self):
        self.ax_altcmp.set_title(self.title_altcmp_var.get(), pad=8)
        self.ax_altcmp.set_xlabel(self.alt_xlabel_var.get())
        self.ax_altcmp.set_ylabel(self.alt_ylabel_var.get())
        if self.t_labels_hhmm:
            stepN = max(1, _r4_int(self.alt_tickN_var.get()) or 4)
            xt = np.arange(0, len(self.t_labels_hhmm), stepN)
            self.ax_altcmp.set_xticks(xt)
            self.ax_altcmp.set_xticklabels([self.t_labels_hhmm[i] for i in xt], rotation=30, ha="right")
            self.ax_altcmp.tick_params(axis="x", labelsize=8, pad=2)
        ymin = _r4_float(self.alt_ymin_var.get())
        ymax = _r4_float(self.alt_ymax_var.get())
        ytick = _r4_float(self.alt_ytick_var.get())
        if ymin is not None and ymax is not None and ymax > ymin:
            self.ax_altcmp.set_ylim(ymin, ymax)
        if ytick and ytick > 0:
            self.ax_altcmp.yaxis.set_major_locator(mticker.MultipleLocator(ytick))
        self.ax_altcmp.tick_params(axis="y", labelsize=8)
        self.ax_altcmp.grid(True, alpha=0.25)

    def refresh_all(self):
        for ax in (self.ax_rti1, self.ax_rti2, self.ax_prof, self.ax_altcmp): ax.clear()
        self._reset_cbars(); self._apply_layout()
        cmap = self._get_cmap(); interp = self.interp_var.get().strip()
        if interp not in INTERP_CHOICES: interp = "nearest"
        vmin = float(self.vmin_var.get()); vmax = float(self.vmax_var.get())
        if vmax <= vmin: vmax = vmin + 1e-6
        norm = self._get_norm(vmin, vmax, cmap)
        self.im1 = None
        if self.show_proto_var.get() and self.ax_rti1.get_visible() and self.Z_proto is not None and self.r_proto is not None:
            Zm = self._mask(np.clip(self.Z_proto, vmin, vmax) if self.scale_mode_var.get() == "preset" else self.Z_proto)
            self.im1 = self.ax_rti1.imshow(Zm, aspect="auto", origin="lower",
                extent=[0, Zm.shape[1]-1, float(np.nanmin(self.r_proto)), float(np.nanmax(self.r_proto))],
                cmap=cmap, norm=norm, interpolation=interp, resample=False)
        rti_visible = [ax for ax in (self.ax_rti1 if self.show_proto_var.get() else None, self.ax_rti2 if self.show_mpl_var.get() else None) if ax is not None and ax.get_visible()]
        bottom_rti_ax = rti_visible[-1] if rti_visible else None
        if self.ax_rti1.get_visible():
            self.ax_rti1.set_title(self.title_rti_proto_var.get(), pad=8)
            self._rti_style(self.ax_rti1, show_xlabel=True, show_xticks=True)
        self.im2 = None
        if self.show_mpl_var.get() and self.ax_rti2.get_visible() and self.Z_mpl_on_proto is not None and self.r_proto is not None:
            Zm = self._mask(np.clip(self.Z_mpl_on_proto, vmin, vmax) if self.scale_mode_var.get() == "preset" else self.Z_mpl_on_proto)
            self.im2 = self.ax_rti2.imshow(Zm, aspect="auto", origin="lower",
                extent=[0, Zm.shape[1]-1, float(np.nanmin(self.r_proto)), float(np.nanmax(self.r_proto))],
                cmap=cmap, norm=norm, interpolation=interp, resample=False)
        if self.ax_rti2.get_visible():
            self.ax_rti2.set_title(self.title_rti_mpl_var.get(), pad=8)
            self._rti_style(self.ax_rti2, show_xlabel=True, show_xticks=True)
        if self.show_alt_overlay_var.get() and self.t_labels_hhmm:
            x = np.arange(len(self.t_labels_hhmm))
            if self.ax_rti1.get_visible() and self.alt_proto_by_slot is not None:
                self.ax_rti1.plot(x, self.alt_proto_by_slot.astype(float), color="red", lw=1.6, alpha=0.95)
            if self.ax_rti2.get_visible() and self.alt_mpl_by_slot is not None:
                self.ax_rti2.plot(x, self.alt_mpl_by_slot.astype(float), color="red", lw=1.6, alpha=0.95)
        if self.show_profile_var.get() and self.ax_prof.get_visible():
            self.update_profile_plot(True)
        else:
            self._update_markers()
        if self.show_alt_compare_var.get() and self.ax_altcmp.get_visible():
            self._style_altcmp()
            if self.t_labels_hhmm and self.alt_proto_by_slot is not None:
                x = np.arange(len(self.t_labels_hhmm))
                self.ax_altcmp.plot(x, self.alt_mpl_by_slot.astype(float), "--", lw=1.6, label="MPL ALT")
                self.ax_altcmp.plot(x, self.alt_proto_by_slot.astype(float), lw=1.8, label=f"Prototype ALT ({self.alt_proto_col.get()})")
                self.ax_altcmp.legend(fontsize=8)
        self._update_cbars()
        if self._mpl_cid is None: self._mpl_cid = self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.draw_idle()

    def _get_sel(self):
        sel = list(self.time_list.curselection())
        if not sel and self.t_list_labels: sel = [0]; self.time_list.selection_set(0)
        return sel

    def _get_proto_mat(self):
        mode = self.profile_proto_mode_var.get().strip().lower()
        if mode == "denoised":
            if self.Z_proto_den is not None: return self.Z_proto_den, "Denoised", None
            return None, "Denoised", f"Load {DENOISED_SHEET} from ALT file first"
        return self.Z_proto, "Raw", None

    def update_profile_plot(self, redraw_markers=False):
        self.ax_prof.clear()
        self.ax_prof.set_title(self.title_prof_var.get(), pad=10)
        self.ax_prof.set_xlabel(self.prof_xlabel_var.get()); self.ax_prof.set_ylabel(self.prof_ylabel_var.get())
        if self.r_proto is None or self.Z_proto is None:
            self.ax_prof.text(0.5, 0.5, "Load Prototype first", ha="center", va="center", transform=self.ax_prof.transAxes); return
        proto_mat, lbl, note = self._get_proto_mat(); sel = self._get_sel()
        lc = 0
        for idx in sel[:10]:
            if proto_mat is not None:
                p = proto_mat[:, idx]; mp = np.isfinite(p)
                if mp.any(): self.ax_prof.plot(self.r_proto[mp], p[mp], lw=1.2, label=f"Proto {lbl} {self.t_labels_hhmm[idx]}"); lc += 1
            if self.Z_mpl_on_proto is not None:
                m = self.Z_mpl_on_proto[:, idx]; mm = np.isfinite(m)
                if mm.any(): self.ax_prof.plot(self.r_proto[mm], m[mm], "--", lw=1.2, label=f"MPL {self.t_labels_hhmm[idx]}"); lc += 1
        if note:
            self.ax_prof.text(0.02, 0.98, note, ha="left", va="top", transform=self.ax_prof.transAxes, fontsize=8,
                              bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF3C7", edgecolor="#F59E0B"))
        self.ax_prof.grid(True, alpha=0.25)
        if lc > 0: self.ax_prof.legend(fontsize=8)
        try:
            xmin = _r4_float(self.prof_xmin_var.get()); xmax = _r4_float(self.prof_xmax_var.get())
            ymin = _r4_float(self.prof_ymin_var.get()); ymax = _r4_float(self.prof_ymax_var.get())
            if xmin is not None and xmax is not None and xmax > xmin: self.ax_prof.set_xlim(xmin, xmax)
            if ymin is not None and ymax is not None and ymax > ymin: self.ax_prof.set_ylim(ymin, ymax)
        except Exception: pass
        self._update_markers()

    def _update_markers(self):
        for ln in self.sel_lines:
            try: ln.remove()
            except Exception: pass
        self.sel_lines = []
        for idx in self._get_sel():
            for ax in (self.ax_rti1, self.ax_rti2):
                if ax.get_visible():
                    self.sel_lines.append(ax.axvline(idx, color="white", lw=1.0, alpha=0.85))

    def _on_click(self, event):
        if self.t_list_labels is None: return
        if event.inaxes not in (self.ax_rti1, self.ax_rti2): return
        if event.xdata is None: return
        idx = max(0, min(len(self.t_list_labels)-1, int(round(event.xdata))))
        self.time_list.selection_clear(0, tk.END); self.time_list.selection_set(idx); self.time_list.see(idx)
        self.update_profile_plot(True); self.canvas.draw_idle()

    def save_png(self):
        fp = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png")])
        if not fp: return
        try: self.fig.savefig(fp, dpi=300, bbox_inches="tight"); messagebox.showinfo("Saved", f"Saved:\n{fp}")
        except Exception as e: messagebox.showerror("Error", str(e))

    def save_profile_png(self):
        if self.r_proto is None: messagebox.showinfo("No data","Load Prototype first."); return
        fp = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png")])
        if not fp: return
        sel = self._get_sel()
        if not sel: messagebox.showinfo("No selection","Select at least one time."); return
        fig = plt.Figure(figsize=(10, 3.2)); ax = fig.add_subplot(111)
        ax.set_title(self.title_prof_var.get(), pad=10)
        ax.set_xlabel(self.prof_xlabel_var.get()); ax.set_ylabel(self.prof_ylabel_var.get())
        proto_mat, lbl, _ = self._get_proto_mat()
        for idx in sel:
            if proto_mat is not None:
                p = proto_mat[:, idx]; mp = np.isfinite(p)
                if mp.any(): ax.plot(self.r_proto[mp], p[mp], label=f"Proto {lbl} {self.t_labels_hhmm[idx]}")
            if self.Z_mpl_on_proto is not None:
                m = self.Z_mpl_on_proto[:, idx]; mm = np.isfinite(m)
                if mm.any(): ax.plot(self.r_proto[mm], m[mm], "--", label=f"MPL {self.t_labels_hhmm[idx]}")
        ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
        try: fig.savefig(fp, dpi=300, bbox_inches="tight"); messagebox.showinfo("Saved", f"Saved:\n{fp}")
        except Exception as e: messagebox.showerror("Error", str(e))


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE OVERVIEW TAB
# ═════════════════════════════════════════════════════════════════════════════
class PipelineTab(tk.Frame):
    STEPS = [
        ("1", "MPL → rmin-rmax", "CSV folder  →  Excel with rmin/rmax/copol"),
        ("2", "NRB Daily Profile", ".dat folder  →  NRB profile Excel"),
        ("3", "ALT Calculator",   "NRB profile / MPL-guided rmin-rmax  →  ALT results Excel"),
        ("4", "RTI Visualizer",   "All outputs  →  RTI + profile plots"),
    ]

    def __init__(self, parent, app_state: AppState, notebook: ttk.Notebook):
        super().__init__(parent, bg=BG)
        self.app_state = app_state; self.notebook = notebook
        self._card_frames: List[tk.Frame] = []
        self._status_labels: List[tk.Label] = []
        self._file_labels: List[tk.Label] = []
        self._build_ui()
        app_state.register_cb(self._refresh_status)

    def _build_ui(self):
        make_header(self, "LiDAR Analysis Suite  v1.1",
                    "4-step workflow with actual-files mode for Step 1 and Step 2").pack(fill="x")

        # pipeline cards container
        outer = tk.Frame(self, bg=BG)
        outer.pack(fill="both", expand=True, padx=30, pady=30)

        # step cards row
        cards_row = tk.Frame(outer, bg=BG)
        cards_row.pack(fill="x")

        for i, (num, name, desc) in enumerate(self.STEPS):
            col = tk.Frame(cards_row, bg=BG); col.pack(side="left", fill="both", expand=True, padx=(0 if i == 0 else 0, 0))
            # Arrow between cards
            if i > 0:
                arr = tk.Frame(col, bg=BG, width=30)
                arr.pack(side="left", fill="y")
                tk.Label(arr, text="→", bg=BG, fg=BORDER, font=("Segoe UI", 20, "bold")).pack(expand=True)

            # Card
            card_wrap = tk.Frame(col, bg=BORDER, padx=1, pady=1)
            card_wrap.pack(side="left", fill="both", expand=True)
            card = tk.Frame(card_wrap, bg=CARD, padx=18, pady=16)
            card.pack(fill="both", expand=True)
            self._card_frames.append(card)

            # Step number badge
            badge_f = tk.Frame(card, bg=PRIMARY, padx=10, pady=3)
            badge_f.pack(anchor="w", pady=(0, 10))
            tk.Label(badge_f, text=f"STEP {num}", bg=PRIMARY, fg="white", font=("Segoe UI", 8, "bold")).pack()

            tk.Label(card, text=name, bg=CARD, fg=TEXT, font=("Segoe UI", 11, "bold")).pack(anchor="w")
            tk.Label(card, text=desc, bg=CARD, fg=SUBTEXT, font=F_SMALL, wraplength=160).pack(anchor="w", pady=(4, 12))

            # Status indicator
            status_lbl = tk.Label(card, text="⬜ Pending", bg=CARD, fg=SUBTEXT, font=("Segoe UI", 9))
            status_lbl.pack(anchor="w")
            self._status_labels.append(status_lbl)

            # File label
            file_lbl = tk.Label(card, text="", bg=CARD, fg=SUBTEXT, font=("Segoe UI", 8), wraplength=160)
            file_lbl.pack(anchor="w", pady=(4, 8))
            self._file_labels.append(file_lbl)

            # Go button
            idx = i + 1
            ttk.Button(card, text=f"Open Step {num} →", style="S.TButton",
                       command=lambda n=idx: self.notebook.select(n)).pack(anchor="w")

        # Quick info section
        info_frame = tk.Frame(outer, bg=CARD, padx=20, pady=16, highlightbackground=BORDER, highlightthickness=1)
        info_frame.pack(fill="x", pady=(30, 0))
        tk.Label(info_frame, text="🔗  How to use", bg=CARD, fg=PRIMARY, font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))
        steps_info = [
            "1️⃣  Go to Step 1 → select MPL CSV folder → choose Date / optional Start time → click Run",
            "2️⃣  Go to Step 2 → select .dat folder → choose Date / optional Start time → click Run",
            "3️⃣  Go to Step 3 → files auto-populated from Steps 1 & 2 → click Calculate ALT",
            "4️⃣  Go to Step 4 → load Prototype (Step 2 output) + MPL (Step 1 output) + ALT (Step 3 output) → visualize",
        ]
        for txt in steps_info:
            tk.Label(info_frame, text=txt, bg=CARD, fg=TEXT, font=F_SMALL, anchor="w").pack(anchor="w", pady=2)

    def _refresh_status(self):
        outputs = [self.app_state.step1_output, self.app_state.step2_output, self.app_state.step3_output, None]
        for i, (lbl, file_lbl, out) in enumerate(zip(self._status_labels, self._file_labels, outputs)):
            if out:
                lbl.configure(text="✅ Done", fg=SUCCESS)
                fname = os.path.basename(out)
                file_lbl.configure(text=f"📄 {fname}")
            else:
                lbl.configure(text="⬜ Pending", fg=SUBTEXT)
                file_lbl.configure(text="")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═════════════════════════════════════════════════════════════════════════════
class LiDARSuite(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LiDAR Analysis Suite  v1.1")
        sw = max(1280, int(self.winfo_screenwidth()))
        sh = max(800,  int(self.winfo_screenheight()))
        ww = min(1600, max(1280, int(sw * 0.92)))
        wh = min(960,  max(800,  int(sh * 0.90)))
        x = max(0, (sw - ww) // 2); y = max(0, (sh - wh) // 2)
        self.geometry(f"{ww}x{wh}+{x}+{y}")
        self.minsize(1100, 720)
        apply_theme(self)
        self._style_notebook()

        self.app_state = AppState()
        self.notebook = ttk.Notebook(self, style="Suite.TNotebook")
        self.notebook.pack(fill="both", expand=True)

        # Create all tabs
        self.tab_pipeline = PipelineTab(self.notebook, self.app_state, self.notebook)
        self.tab_step1    = Step1Frame(self.notebook, self.app_state)
        self.tab_step2    = Step2Frame(self.notebook, self.app_state)
        self.tab_step3    = Step3Frame(self.notebook, self.app_state)
        self.tab_step4    = Step4Frame(self.notebook, self.app_state)

        self.notebook.add(self.tab_pipeline, text="  🏠  Pipeline  ")
        self.notebook.add(self.tab_step1,    text="  📁  Step 1: rmin-rmax  ")
        self.notebook.add(self.tab_step2,    text="  📊  Step 2: NRB Profile  ")
        self.notebook.add(self.tab_step3,    text="  🔬  Step 3: ALT  ")
        self.notebook.add(self.tab_step4,    text="  📈  Step 4: RTI  ")

    def _style_notebook(self):
        s = ttk.Style(self)
        s.configure("Suite.TNotebook", background=DARK, tabmargins=[0, 0, 0, 0], borderwidth=0)
        s.configure("Suite.TNotebook.Tab", background="#253A5E", foreground="#94A3B8",
                    font=("Segoe UI", 10), padding=[14, 10], borderwidth=0)
        s.map("Suite.TNotebook.Tab",
              background=[("selected", CARD), ("active", "#1E3A5F")],
              foreground=[("selected", PRIMARY), ("active", "white")])


def main():
    app = LiDARSuite()
    app.mainloop()


if __name__ == "__main__":
    main()
