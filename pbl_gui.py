# pbl_gui.py  –  modern UI (requires modern_theme.py in same folder)
from __future__ import annotations

import os, re, sys, subprocess, tempfile, threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from modern_theme import (
    apply_theme, make_card, make_header, make_status_bar, make_log,
    append_log,
    BG, CARD, DARK, PRIMARY, BORDER, TEXT, SUBTEXT,
    SUCCESS, WARNING, ERROR, F_H1, F_H2, F_BODY, F_SMALL
)

PREF_NRB_SHEET  = "NRB profile"
PREF_RMIN_SHEET = "rmin-rmax"
ENGINE_CANDIDATES = ["pbl_engine.py", "pbl_v02_report.py", "pbl_v01_report.py"]

RE_YMD  = re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})")
RE_DMY  = re.compile(r"(\d{2})[-_](\d{2})[-_](\d{4})")
RE_HHMM = re.compile(r"(\d{2}):(\d{2})(?::(\d{2}))?$")


# ── Backend helpers (unchanged) ───────────────────────────────────────────────
def guess_date_from_filename(path: str):
    name = os.path.basename(path)
    m = RE_YMD.search(name)
    if m:
        y, mo, d = map(int, m.groups())
        return pd.Timestamp(year=y, month=mo, day=d)
    m = RE_DMY.search(name)
    if m:
        d, mo, y = map(int, m.groups())
        return pd.Timestamp(year=y, month=mo, day=d)
    return None


def parse_profile_times(colnames, base_date):
    times = []
    for c in colnames:
        t = pd.to_datetime(c, errors="coerce")
        if pd.notna(t):
            t = pd.Timestamp(t)
            if t.year < 1975:
                t = pd.Timestamp(base_date.date()) + pd.Timedelta(
                    hours=t.hour, minutes=t.minute, seconds=t.second)
            times.append(t); continue
        m = RE_HHMM.search(str(c).strip())
        if m:
            hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
            times.append(pd.Timestamp(base_date.date()) +
                         pd.Timedelta(hours=hh, minutes=mm, seconds=ss))
            continue
        raise ValueError(f"Cannot parse NRB column header: {c!r}")
    return times


def pick_rmin_columns(df):
    cols = {str(c).strip().lower(): c for c in df.columns}
    def pick(*cands):
        for k in cands:
            if k in cols: return cols[k]
        return None
    tc  = pick("time", "datetime", "timestamp")
    rmc = pick("rmin", "rmin_m", "rmin (m)", "r min", "rmin(m)")
    rxc = pick("rmax", "rmax_m", "rmax (m)", "r max", "rmax(m)")
    mc  = None
    for k, orig in cols.items():
        if "pbl" in k and "mpl" in k: mc = orig; break
    if tc is None:
        raise ValueError("rmin-rmax sheet must have a 'Time' column.")
    return tc, rmc, rxc, mc


def map_rminrmax_to_slots(df_rmin, slot_times, tol_min):
    tc, rmc, rxc, mc = pick_rmin_columns(df_rmin)
    work = df_rmin.copy().dropna(how="all").dropna(axis=1, how="all")
    work[tc] = pd.to_datetime(work[tc], errors="coerce")
    work = work.dropna(subset=[tc]).sort_values(tc)
    if rmc: work[rmc] = pd.to_numeric(work[rmc], errors="coerce")
    if rxc: work[rxc] = pd.to_numeric(work[rxc], errors="coerce")
    if mc:  work[mc]  = pd.to_numeric(work[mc],  errors="coerce")
    times_mpl = work[tc].to_list()
    rows = []
    for ts in slot_times:
        best_idx = best_abs = best_dt = None
        for i, mt in enumerate(times_mpl):
            dt = (pd.Timestamp(mt) - pd.Timestamp(ts)).total_seconds() / 60.0
            ad = abs(dt)
            if best_abs is None or ad < best_abs:
                best_abs, best_idx, best_dt = ad, i, dt
        if best_abs is not None and best_abs <= tol_min:
            row = work.iloc[best_idx]
            rows.append({
                "Time": pd.Timestamp(ts),
                "Slot": pd.Timestamp(ts).strftime("%H:%M"),
                "PBL from MPL (m)": float(row[mc])  if mc  else np.nan,
                "rmin":             float(row[rmc]) if rmc else np.nan,
                "rmax":             float(row[rxc]) if rxc else np.nan,
                "MPL_Time": pd.Timestamp(row[tc]),
                "dt_min":   float(best_dt),
            })
        else:
            rows.append({
                "Time": pd.Timestamp(ts),
                "Slot": pd.Timestamp(ts).strftime("%H:%M"),
                "PBL from MPL (m)": np.nan, "rmin": np.nan, "rmax": np.nan,
                "MPL_Time": pd.NaT, "dt_min": np.nan,
            })
    return pd.DataFrame(rows)


# ── GUI ────────────────────────────────────────────────────────────────────────
class PBLGuiV5(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ALT Calculator")
        self.geometry("1060x700")
        self.minsize(880, 580)
        apply_theme(self)

        self.engine_path  = tk.StringVar(value=self._auto_find_engine())
        self.nrb_path     = tk.StringVar()
        self.nrb_sheet    = tk.StringVar()
        self.rmin_path    = tk.StringVar()
        self.rmin_sheet   = tk.StringVar()
        self.out_path     = tk.StringVar()
        self.tol_min      = tk.StringVar(value="3")
        self.min_valid_frac = tk.StringVar(value="0.50")
        self.min_valid_bins = tk.StringVar(value="50")
        self.detection_mode = tk.StringVar(value="dual")
        self.profile_rmin = tk.StringVar(value="0")
        self.profile_rmax = tk.StringVar(value="4000")
        self._status_set  = None

        self._build_ui()

    def _auto_find_engine(self):
        base = os.path.dirname(__file__)
        for name in ENGINE_CANDIDATES:
            p = os.path.join(base, name)
            if os.path.exists(p): return p
        return ""

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build_ui(self):
        make_header(self, "ALT Calculator",
                    "NRB profile → ALT, with optional MPL-guided rmin-rmax").pack(fill="x")

        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=16, pady=14)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        # ── LEFT ─────────────────────────────────────────────────────────────
        left = tk.Frame(main, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # Engine card
        c_e, f_e = make_card(left, "🔧  ALT Engine Script")
        c_e.pack(fill="x", pady=(0, 10))
        tk.Label(f_e, text="Python engine file (pbl_engine.py):",
                 bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(anchor="w")
        row_e = tk.Frame(f_e, bg=CARD); row_e.pack(fill="x", pady=(4, 0))
        ttk.Entry(row_e, textvariable=self.engine_path).pack(
            side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row_e, text="Browse…", style="S.TButton",
                   command=self.pick_engine).pack(side="right")
        if not self.engine_path.get().strip():
            tk.Label(f_e, text="⚠  Engine not found. Browse to select pbl_engine.py",
                     bg=CARD, fg=WARNING, font=F_SMALL).pack(anchor="w", pady=(4, 0))

        # NRB file card
        c_n, f_n = make_card(left, "📊  NRB Excel File")
        c_n.pack(fill="x", pady=(0, 10))
        row_n = tk.Frame(f_n, bg=CARD); row_n.pack(fill="x")
        ttk.Entry(row_n, textvariable=self.nrb_path).pack(
            side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row_n, text="Browse…", style="S.TButton",
                   command=self.pick_nrb).pack(side="right")
        row_ns = tk.Frame(f_n, bg=CARD); row_ns.pack(fill="x", pady=(8, 0))
        tk.Label(row_ns, text="Sheet:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(
            side="left", padx=(0, 8))
        self.nrb_combo = ttk.Combobox(row_ns, textvariable=self.nrb_sheet,
                                      values=[], state="readonly", width=30)
        self.nrb_combo.pack(side="left")
        ttk.Button(row_ns, text="↻ Reload", style="Sm.TButton",
                   command=self.load_nrb_sheets).pack(side="left", padx=(8, 0))

        # rmin-rmax card
        c_r, f_r = make_card(left, "📐  MPL rmin-rmax Excel File (optional for NRB-profile mode)")
        c_r.pack(fill="x", pady=(0, 10))
        row_rm = tk.Frame(f_r, bg=CARD); row_rm.pack(fill="x")
        ttk.Entry(row_rm, textvariable=self.rmin_path).pack(
            side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row_rm, text="Browse…", style="S.TButton",
                   command=self.pick_rmin).pack(side="right")
        row_rs = tk.Frame(f_r, bg=CARD); row_rs.pack(fill="x", pady=(8, 0))
        tk.Label(row_rs, text="Sheet:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(
            side="left", padx=(0, 8))
        self.rmin_combo = ttk.Combobox(row_rs, textvariable=self.rmin_sheet,
                                       values=[], state="readonly", width=30)
        self.rmin_combo.pack(side="left")
        ttk.Button(row_rs, text="↻ Reload", style="Sm.TButton",
                   command=self.load_rmin_sheets).pack(side="left", padx=(8, 0))

        # Output card
        c_o, f_o = make_card(left, "💾  Output")
        c_o.pack(fill="x", pady=(0, 10))
        row_o = tk.Frame(f_o, bg=CARD); row_o.pack(fill="x")
        ttk.Entry(row_o, textvariable=self.out_path).pack(
            side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row_o, text="Save As…", style="S.TButton",
                   command=self.pick_output).pack(side="right")

        # Options card
        c_opt, f_opt = make_card(left, "⚙️  Options")
        c_opt.pack(fill="x", pady=(0, 10))

        mode_row = tk.Frame(f_opt, bg=CARD); mode_row.pack(fill="x", pady=(0, 6))
        tk.Label(mode_row, text="ALT detection mode:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left", padx=(0, 6))
        ttk.OptionMenu(mode_row, self.detection_mode, self.detection_mode.get(),
                       "dual", "mpl_guided", "nrb_profile").pack(side="left", padx=(0, 12))
        tk.Label(mode_row, text="NRB profile range (m):", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left", padx=(0, 4))
        ttk.Entry(mode_row, textvariable=self.profile_rmin, width=7).pack(side="left", padx=(0, 4))
        tk.Label(mode_row, text="to", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left", padx=(0, 4))
        ttk.Entry(mode_row, textvariable=self.profile_rmax, width=7).pack(side="left", padx=(0, 12))

        opt_row = tk.Frame(f_opt, bg=CARD); opt_row.pack(fill="x")
        for label, var, w in [
            ("Nearest match tol (min):", self.tol_min, 6),
            ("min_valid_frac (0–1):",    self.min_valid_frac, 6),
            ("min_valid_bins:",          self.min_valid_bins, 6),
        ]:
            tk.Label(opt_row, text=label, bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(
                side="left", padx=(0, 4))
            ttk.Entry(opt_row, textvariable=var, width=w).pack(
                side="left", padx=(0, 16))

        # Run
        c_run, f_run = make_card(left)
        c_run.pack(fill="x")
        self.run_btn = ttk.Button(f_run, text="▶  Calculate ALT",
                                  style="P.TButton", command=self.run_pbl)
        self.run_btn.pack(anchor="w")

        # ── RIGHT ────────────────────────────────────────────────────────────
        right = tk.Frame(main, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")

        log_hdr = tk.Frame(right, bg=BG); log_hdr.pack(fill="x", pady=(0, 6))
        tk.Label(log_hdr, text="Console Log", bg=BG, fg=TEXT,
                 font=F_H2).pack(side="left")
        ttk.Button(log_hdr, text="Clear", style="Sm.TButton",
                   command=self._clear_log).pack(side="right")
        log_outer, self.log_txt = make_log(right, height=32)
        log_outer.pack(fill="both", expand=True)

        sb, self.status_var, self._status_set = make_status_bar(self)
        sb.pack(fill="x", side="bottom")

    # ── helpers ──────────────────────────────────────────────────────────────
    def _log(self, msg: str):
        append_log(self.log_txt, msg)

    def _safe(self, fn, *a, **kw):
        self.after(0, lambda: fn(*a, **kw))

    def _clear_log(self):
        self.log_txt.configure(state="normal")
        self.log_txt.delete("1.0", "end")
        self.log_txt.configure(state="disabled")

    def pick_engine(self):
        path = filedialog.askopenfilename(
            title="Select ALT engine script",
            filetypes=[("Python files", "*.py")])
        if path:
            self.engine_path.set(path)
            self._log(f"Engine: {path}")

    def pick_nrb(self):
        path = filedialog.askopenfilename(
            title="Select NRB Excel file",
            filetypes=[("Excel files", "*.xlsx *.xls")])
        if not path: return
        self.nrb_path.set(path)
        self._log(f"NRB: {path}")
        self.load_nrb_sheets()
        folder = os.path.dirname(path)
        base   = os.path.splitext(os.path.basename(path))[0]
        if not self.out_path.get().strip():
            self.out_path.set(os.path.join(folder, f"ALT_{base}.xlsx"))

    def pick_rmin(self):
        path = filedialog.askopenfilename(
            title="Select rmin-rmax Excel file",
            filetypes=[("Excel files", "*.xlsx *.xls")])
        if not path: return
        self.rmin_path.set(path)
        self._log(f"rmin-rmax: {path}")
        self.load_rmin_sheets()

    def pick_output(self):
        path = filedialog.asksaveasfilename(
            title="Save output as", defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")])
        if path:
            self.out_path.set(path)
            self._log(f"Output: {path}")

    def load_nrb_sheets(self):
        path = self.nrb_path.get().strip()
        if not path: return
        try:
            sheets = list(pd.ExcelFile(path).sheet_names)
            self.nrb_combo["values"] = sheets
            self.nrb_sheet.set(PREF_NRB_SHEET if PREF_NRB_SHEET in sheets
                               else (sheets[0] if sheets else ""))
            self._log(f"NRB sheets: {sheets}")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot read NRB sheets:\n{e}")

    def load_rmin_sheets(self):
        path = self.rmin_path.get().strip()
        if not path: return
        try:
            sheets = list(pd.ExcelFile(path).sheet_names)
            self.rmin_combo["values"] = sheets
            self.rmin_sheet.set(PREF_RMIN_SHEET if PREF_RMIN_SHEET in sheets
                                else (sheets[0] if sheets else ""))
            self._log(f"rmin-rmax sheets: {sheets}")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot read rmin-rmax sheets:\n{e}")

    # ── Run ──────────────────────────────────────────────────────────────────
    def run_pbl(self):
        engine     = self.engine_path.get().strip()
        nrb_path   = self.nrb_path.get().strip()
        nrb_sheet  = self.nrb_sheet.get().strip()
        rmin_path  = self.rmin_path.get().strip()
        rmin_sheet = self.rmin_sheet.get().strip()
        out_path   = self.out_path.get().strip()
        det_mode   = self.detection_mode.get().strip() or "dual"

        need_rmin = det_mode in ("mpl_guided", "dual")
        for msg, cond in [
            ("Select engine script.", not engine or not os.path.exists(engine)),
            ("NRB file not found.",   not nrb_path or not os.path.exists(nrb_path)),
            ("rmin-rmax file not found for MPL-guided / dual mode.", need_rmin and (not rmin_path or not os.path.exists(rmin_path))),
            ("Select NRB sheet.",     not nrb_sheet),
            ("Select rmin-rmax sheet for MPL-guided / dual mode.", need_rmin and not rmin_sheet),
            ("Specify output path.",  not out_path),
        ]:
            if cond:
                messagebox.showerror("Input error", msg); return

        try:
            tol  = float(self.tol_min.get().strip())
            mvf  = float(self.min_valid_frac.get().strip())
            mvb  = int(float(self.min_valid_bins.get().strip()))
            prmin = float(self.profile_rmin.get().strip())
            prmax = float(self.profile_rmax.get().strip())
        except Exception:
            tol, mvf, mvb, prmin, prmax = 3.0, 0.50, 50, 0.0, 4000.0

        self.run_btn.configure(state="disabled")
        self._status_set("Running…", "running")
        self._log("=== RUN START ===")
        self._log(f"Engine: {engine}")

        def worker():
            tmp_path = None
            try:
                df_nrb = pd.read_excel(nrb_path, sheet_name=nrb_sheet)
                df_nrb = df_nrb.dropna(how="all").dropna(axis=1, how="all")
                if df_nrb.empty or df_nrb.shape[1] < 2:
                    raise ValueError("NRB sheet is empty.")

                prof_cols = list(df_nrb.columns[1:])
                base_dates = pd.to_datetime(prof_cols, errors="coerce")
                base_dates = base_dates[~pd.isna(base_dates)]
                base_date  = (pd.Timestamp(base_dates[0]) if len(base_dates) > 0
                              else (guess_date_from_filename(nrb_path) or
                                    guess_date_from_filename(rmin_path) or
                                    pd.Timestamp.today()))

                slot_times = parse_profile_times(prof_cols, pd.Timestamp(base_date))
                new_cols   = [df_nrb.columns[0]] + [
                    pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S") for t in slot_times]
                df_nrb2 = df_nrb.copy(); df_nrb2.columns = new_cols

                if rmin_path and os.path.exists(rmin_path):
                    df_rmin = pd.read_excel(rmin_path, sheet_name=rmin_sheet)
                    df_rmin = df_rmin.dropna(how="all").dropna(axis=1, how="all")
                    if df_rmin.empty:
                        raise ValueError("rmin-rmax sheet is empty.")
                    df_map = map_rminrmax_to_slots(df_rmin, slot_times, tol_min=tol)
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
                self._safe(self._log, f"Matched MPL/rmin-rmax slots: {matched}/{len(slot_times)}")

                tmp_fd, tmp_path = tempfile.mkstemp(prefix="pbl_merge_", suffix=".xlsx")
                os.close(tmp_fd)
                with pd.ExcelWriter(tmp_path, engine="openpyxl") as w:
                    df_nrb2.to_excel(w, sheet_name=PREF_NRB_SHEET, index=False)
                    df_map.to_excel(w, sheet_name=PREF_RMIN_SHEET, index=False)

                cmd = [
                    sys.executable, engine,
                    "--nrb",            tmp_path,
                    "--sheet",          PREF_NRB_SHEET,
                    "--rminrmax_sheet", PREF_RMIN_SHEET,
                    "--out",            out_path,
                    "--min_valid_frac", str(mvf),
                    "--min_valid_bins", str(mvb),
                    "--detection_mode", det_mode,
                    "--profile_rmin", str(prmin),
                    "--profile_rmax", str(prmax),
                ]
                self._safe(self._log, "Command: " + " ".join(
                    [f'"{c}"' if " " in c else c for c in cmd]))

                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.stdout: self._safe(self._log, proc.stdout.strip())
                if proc.stderr: self._safe(self._log, proc.stderr.strip())

                if proc.returncode == 0:
                    self._safe(self._log, "=== RUN OK ===")
                    self._safe(self._status_set, "Done ✅", "ok")
                    self._safe(messagebox.showinfo, "Success", f"Saved:\n{out_path}")
                else:
                    self._safe(self._log, "=== RUN FAILED ===")
                    self._safe(self._status_set, "Failed ❌", "error")
                    self._safe(messagebox.showerror, "Failed", "Process failed. See log.")

            except Exception as e:
                self._safe(self._log, f"[FAILED] {e}")
                self._safe(self._status_set, "Failed ❌", "error")
                self._safe(messagebox.showerror, "Error", str(e))
            finally:
                try:
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
                self._safe(self.run_btn.configure, state="normal")

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    app = PBLGuiV5()
    app.mainloop()
