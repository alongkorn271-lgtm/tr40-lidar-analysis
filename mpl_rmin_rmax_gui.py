from __future__ import annotations

import os
import re
import sys
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ── modern theme (same folder) ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from modern_theme import (  # type: ignore
    apply_theme, make_card, make_header, make_status_bar, make_log,
    append_log, BG, CARD, TEXT, SUBTEXT, F_H2, F_SMALL
)

# ── Backend constants ─────────────────────────────────────────────────────────
PBL_COL_DI_0 = 112
PBL_ROW_2_0 = 1
COPOL_N = 498


# ── Backend helpers ───────────────────────────────────────────────────────────
def parse_ts_from_filename(name: str) -> Optional[pd.Timestamp]:
    hits = re.findall(r"(\d{12})", name)
    if not hits:
        return None
    dt = pd.to_datetime(hits[-1], format="%Y%m%d%H%M", errors="coerce")
    return None if pd.isna(dt) else pd.Timestamp(dt)


def parse_hhmm(text: str) -> Optional[Tuple[int, int]]:
    text = text.strip()
    if not text:
        return None
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    if not m:
        raise ValueError("Start time must be HH:MM, for example 09:00")
    hh = int(m.group(1))
    mm = int(m.group(2))
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError("Start time must be HH:MM, for example 09:00")
    return hh, mm


def infer_interval_minutes(timestamps: List[pd.Timestamp]) -> Optional[float]:
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


def read_pbl_km_from_csv(path: Path) -> float:
    try:
        dfh = pd.read_csv(path, sep=None, engine="python")
        col_candidates = [c for c in dfh.columns if str(c).strip().lower() == "pbls"]
        if col_candidates:
            col = col_candidates[0]
            for ridx in range(min(5, len(dfh.index))):
                v = pd.to_numeric(dfh.iloc[ridx][col], errors="coerce")
                if pd.notna(v):
                    return float(v)
    except Exception:
        pass
    try:
        import csv
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            sample = f.read(8192)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
            reader = csv.reader(f, dialect)
            next(reader, None)
            row2 = next(reader, None)
        if row2 is not None and len(row2) > PBL_COL_DI_0:
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


def read_range_copol_norm_from_csv(path: Path, row_start: int = 0, row_end: int = COPOL_N):
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
    cop_norm = cop / mx if (np.isfinite(mx) and mx != 0) else np.full_like(cop, float("nan"), dtype=float)
    return rng, cop_norm


class MPLRminRmaxGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MPL → rmin-rmax Builder  v9")
        self.geometry("1080x700")
        self.minsize(920, 580)
        apply_theme(self)

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
            "MPL → rmin-rmax Builder",
            "v9 · Actual-files mode · uses real timestamps from filenames"
        ).pack(fill="x")

        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=16, pady=14)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        left = tk.Frame(main, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        c_in, f_in = make_card(left, "📁  Input Folder")
        c_in.pack(fill="x", pady=(0, 10))
        tk.Label(f_in, text="Folder containing MPL CSV files:", bg=CARD,
                 fg=SUBTEXT, font=F_SMALL).pack(anchor="w")
        row_f = tk.Frame(f_in, bg=CARD)
        row_f.pack(fill="x", pady=(4, 0))
        ttk.Entry(row_f, textvariable=self.folder).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row_f, text="Browse…", style="S.TButton", command=self.pick_folder).pack(side="right")
        tk.Label(
            f_in,
            text="Accepts filenames containing YYYYMMDDHHMM (12 digits)",
            bg=CARD, fg=SUBTEXT, font=F_SMALL
        ).pack(anchor="w", pady=(4, 0))

        c_p, f_p = make_card(left, "⚙️  Filter & Parameters")
        c_p.pack(fill="x", pady=(0, 10))

        r1 = tk.Frame(f_p, bg=CARD)
        r1.pack(fill="x", pady=(0, 6))
        for label, var, w in [
            ("Date (YYYY-MM-DD):", self.date, 12),
            ("Start time (HH:MM):", self.start_time, 8),
            ("± delta (m):", self.delta_m, 8),
        ]:
            tk.Label(r1, text=label, bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left", padx=(0, 4))
            ttk.Entry(r1, textvariable=var, width=w).pack(side="left", padx=(0, 14))

        tk.Label(
            f_p,
            text=(
                "Mode A: use actual files only. No fixed 48 slots. "
                "If Start time is set, files earlier than that time are excluded."
            ),
            bg=CARD, fg=SUBTEXT, font=F_SMALL, justify="left"
        ).pack(anchor="w")

        c_o, f_o = make_card(left, "💾  Output Excel")
        c_o.pack(fill="x", pady=(0, 10))
        row_o = tk.Frame(f_o, bg=CARD)
        row_o.pack(fill="x")
        ttk.Entry(row_o, textvariable=self.out_path).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row_o, text="Save As…", style="S.TButton", command=self.pick_output).pack(side="right")

        c_r, f_r = make_card(left)
        c_r.pack(fill="x", pady=(0, 6))
        run_row = tk.Frame(f_r, bg=CARD)
        run_row.pack(fill="x")
        self.run_btn = ttk.Button(run_row, text="▶  Run Build rmin-rmax", style="P.TButton", command=self.run)
        self.run_btn.pack(side="left")
        self.pb = ttk.Progressbar(run_row, variable=self.progress, maximum=100, length=280)
        self.pb.pack(side="left", padx=(16, 0))

        right = tk.Frame(main, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")

        log_hdr = tk.Frame(right, bg=BG)
        log_hdr.pack(fill="x", pady=(0, 6))
        tk.Label(log_hdr, text="Console Log", bg=BG, fg=TEXT, font=F_H2).pack(side="left")
        ttk.Button(log_hdr, text="Clear", style="Sm.TButton", command=self._clear_log).pack(side="right")

        log_outer, self.log_txt = make_log(right, height=28)
        log_outer.pack(fill="both", expand=True)

        sb, self.status_var, self._status_set = make_status_bar(self)
        sb.pack(fill="x", side="bottom")

    def _log(self, msg: str):
        append_log(self.log_txt, msg)

    def _safe(self, fn, *a, **kw):
        self.after(0, lambda: fn(*a, **kw))

    def _clear_log(self):
        self.log_txt.configure(state="normal")
        self.log_txt.delete("1.0", "end")
        self.log_txt.configure(state="disabled")

    def pick_folder(self):
        p = filedialog.askdirectory(title="Select folder containing MPL CSV files")
        if not p:
            return
        self.folder.set(p)
        self._log(f"Folder: {p}")
        self.out_path.set(str(Path(p) / "rmin-rmax.xlsx"))
        self._log(f"Default output: {self.out_path.get()}")

    def pick_output(self):
        out = filedialog.asksaveasfilename(
            title="Save rmin-rmax as",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if out:
            self.out_path.set(out)
            self._log(f"Output: {out}")

    def run(self):
        folder = self.folder.get().strip()
        out_path = self.out_path.get().strip()
        date_text = self.date.get().strip()
        start_time_text = self.start_time.get().strip()

        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Please choose a valid folder.")
            return
        if not out_path:
            messagebox.showerror("Error", "Please choose an output path.")
            return
        try:
            delta = float(self.delta_m.get())
            if delta < 0:
                raise ValueError
            start_hm = parse_hhmm(start_time_text)
        except Exception as e:
            messagebox.showerror("Error", str(e) if str(e) else "Please enter valid parameters.")
            return

        self.run_btn.configure(state="disabled")
        self.progress.set(0.0)
        self._status_set("Running…", "running")
        self._log("=== START ===")

        def worker():
            try:
                fldr = Path(folder)
                files_all: List[Tuple[pd.Timestamp, Path]] = []
                for f in sorted(fldr.glob("*")):
                    ts = parse_ts_from_filename(f.name)
                    if ts is not None:
                        files_all.append((ts, f))

                if not files_all:
                    raise ValueError("No filenames contained a valid YYYYMMDDHHMM timestamp.")

                files_all.sort(key=lambda x: x[0])
                dates_found = sorted({ts.normalize() for ts, _ in files_all})

                if date_text:
                    selected_date = pd.to_datetime(date_text, errors="raise").normalize()
                elif len(dates_found) == 1:
                    selected_date = dates_found[0]
                    self._safe(self._log, f"Inferred date: {selected_date.date()}")
                else:
                    found = ", ".join(d.strftime("%Y-%m-%d") for d in dates_found[:6])
                    more = " ..." if len(dates_found) > 6 else ""
                    raise ValueError(
                        "Folder contains multiple dates. Please specify Date (YYYY-MM-DD). "
                        f"Found: {found}{more}"
                    )

                day_files = [(ts, p) for ts, p in files_all if ts.normalize() == selected_date]
                if not day_files:
                    raise ValueError(f"No files found for date {selected_date.strftime('%Y-%m-%d')}.")

                if start_hm is not None:
                    hh, mm = start_hm
                    start_ts = selected_date + pd.Timedelta(hours=hh, minutes=mm)
                    selected_files = [(ts, p) for ts, p in day_files if ts >= start_ts]
                    self._safe(self._log, f"Start filter: keep files from {start_ts.strftime('%Y-%m-%d %H:%M')} onward")
                else:
                    selected_files = day_files
                    self._safe(self._log, "Start filter: none (use all files on selected date)")

                if not selected_files:
                    raise ValueError("No files remain after applying the selected start time.")

                timestamps = [ts for ts, _ in selected_files]
                inferred_interval = infer_interval_minutes(timestamps)
                if inferred_interval is None:
                    self._safe(self._log, f"Selected files: {len(selected_files)} file")
                else:
                    self._safe(
                        self._log,
                        f"Selected files: {len(selected_files)} files | inferred interval ≈ {inferred_interval:g} min"
                    )

                rows = []
                copol_cols = {}
                range_m_master = None
                total = len(selected_files)

                for i, (ts, f) in enumerate(selected_files, start=1):
                    src_file = f.name
                    status = "exact"
                    try:
                        rng_m, cop_norm = read_range_copol_norm_from_csv(f, row_start=0, row_end=COPOL_N)
                        if range_m_master is None and np.isfinite(rng_m).any():
                            range_m_master = rng_m
                        copol_cols[ts] = cop_norm
                    except Exception as e_copol:
                        self._safe(self._log, f"   [WARN] copol extract failed for {src_file}: {e_copol}")
                        copol_cols[ts] = np.full((COPOL_N,), float('nan'), dtype=float)

                    pbl_km = read_pbl_km_from_csv(f)
                    if np.isfinite(pbl_km):
                        pbl_m = float(pbl_km) * 1000.0
                        rmin = max(0.0, pbl_m - delta)
                        rmax = pbl_m + delta
                        self._safe(
                            self._log,
                            f"[{i:03d}/{total:03d}] {ts.strftime('%H:%M')} ← {src_file}  PBL={pbl_m:.1f} m"
                        )
                    else:
                        pbl_m = rmin = rmax = float("nan")
                        status = "read_fail"
                        self._safe(
                            self._log,
                            f"[{i:03d}/{total:03d}] {ts.strftime('%H:%M')} ← {src_file}  PBL=NaN (read_fail)"
                        )

                    rows.append({
                        "Time": ts,
                        "PBL from MPL (m)": pbl_m,
                        "rmin": rmin,
                        "rmax": rmax,
                        "Status": status,
                        "Source file": src_file,
                        "Date": ts.strftime("%Y-%m-%d"),
                        "HH:MM": ts.strftime("%H:%M"),
                    })
                    self._safe(self.progress.set, 100.0 * i / total)

                df = pd.DataFrame(rows)
                if range_m_master is None:
                    range_m_master = np.full((COPOL_N,), float("nan"), dtype=float)

                copol_df = pd.DataFrame({"Range(m)": range_m_master})
                for ts, _ in selected_files:
                    copol_df[ts.strftime("%H:%M")] = copol_cols.get(
                        ts, np.full((COPOL_N,), float("nan"), dtype=float)
                    )

                outp = Path(out_path)
                with pd.ExcelWriter(outp, engine="openpyxl") as xw:
                    df.to_excel(xw, index=False, sheet_name="rmin-rmax")
                    copol_df.to_excel(xw, index=False, sheet_name="copol_nrb_norm")

                self._safe(self._log, f"[OK] Saved: {outp.resolve()}")
                self._safe(self._status_set, "Done ✅", "ok")
                self._safe(self.progress.set, 100.0)
                self._safe(messagebox.showinfo, "Success", f"Saved:\n{outp}")
            except Exception as e:
                self._safe(self._log, f"[FAILED] {e}")
                self._safe(self._status_set, "Failed ❌", "error")
                self._safe(messagebox.showerror, "Failed", str(e))
            finally:
                self._safe(self.run_btn.configure, state="normal")
                self._safe(self._log, "=== END ===")

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    app = MPLRminRmaxGUI()
    app.mainloop()
