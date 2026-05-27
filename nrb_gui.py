from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from nrb_engine import (
    _read_tr40_dat_ascii_array,
    build_daily_profile_from_folder,
    build_single_profile,
    collect_actual_timestamps,
)

BG = "#EEF2F7"
CARD = "#FFFFFF"
TEXT = "#0F172A"
SUBTEXT = "#475569"
ACCENT = "#2563EB"
F_H1 = ("Segoe UI", 15, "bold")
F_H2 = ("Segoe UI", 11, "bold")
F_SMALL = ("Segoe UI", 9)


def apply_theme(root: tk.Tk):
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("TFrame", background=BG)
    style.configure("TLabelframe", background=CARD)
    style.configure("TLabelframe.Label", background=CARD, foreground=TEXT)
    style.configure("TButton", padding=6)
    style.configure("TEntry", padding=4)
    style.configure("TCombobox", padding=4)
    root.configure(bg=BG)


def make_header(parent: tk.Widget, title: str, subtitle: str) -> tk.Frame:
    f = tk.Frame(parent, bg=BG)
    tk.Label(f, text=title, bg=BG, fg=TEXT, font=F_H1).pack(anchor="w")
    tk.Label(f, text=subtitle, bg=BG, fg=SUBTEXT, font=F_SMALL).pack(anchor="w")
    return f


def make_card(parent: tk.Widget, title: str):
    outer = tk.Frame(parent, bg=BG)
    card = tk.Frame(outer, bg=CARD, highlightbackground="#D7DFEA", highlightthickness=1)
    card.pack(fill="both", expand=True)
    tk.Label(card, text=title, bg=CARD, fg=ACCENT, font=F_H2).pack(anchor="w", padx=12, pady=(10, 6))
    body = tk.Frame(card, bg=CARD)
    body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
    return outer, body


def make_status_bar(parent: tk.Widget):
    var = tk.StringVar(value="Ready")
    f = tk.Frame(parent, bg="#E2E8F0")
    lbl = tk.Label(f, textvariable=var, bg="#E2E8F0", fg=TEXT, anchor="w", font=F_SMALL)
    lbl.pack(fill="x", padx=8, pady=4)

    def set_status(msg: str, tag: str = "info"):
        colors = {"info": TEXT, "running": "#92400E", "ok": "#166534", "error": "#B91C1C"}
        lbl.configure(fg=colors.get(tag, TEXT))
        var.set(msg)

    return f, var, set_status


def append_log(widget: tk.Text, msg: str):
    widget.configure(state="normal")
    widget.insert("end", msg.rstrip() + "\n")
    widget.see("end")
    widget.configure(state="disabled")


class NRBDatDailyGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NRB Daily Builder v11.3 Day/Night Toggle")
        self.geometry("1360x900")
        self.minsize(980, 640)
        apply_theme(self)

        self.folder = tk.StringVar()
        self.date = tk.StringVar(value="2026-02-13")
        self.start_time = tk.StringVar(value="")
        self.pattern = tk.StringVar(value="*.dat")
        self.out_path = tk.StringVar()
        self.bin_spacing_m = tk.DoubleVar(value=3.75)
        self.dead_time_ns = tk.DoubleVar(value=3.06)
        self.bg_mode = tk.StringVar(value="pretrigger")
        self.bg_start_m = tk.DoubleVar(value=0.0)
        self.bg_end_m = tk.DoubleVar(value=3750.0)
        self.pretrigger_bins = tk.IntVar(value=1024)
        self.first_signal_bin = tk.IntVar(value=1025)
        self.first_signal_range_m = tk.DoubleVar(value=3.75)

        self.shift_mode = tk.StringVar(value="manual")
        self.bin_shift_bins = tk.IntVar(value=0)
        self.onset_start_m = tk.DoubleVar(value=3500.0)
        self.onset_end_m = tk.DoubleVar(value=4000.0)
        self.onset_smooth_bins = tk.IntVar(value=5)

        self.sig_start_m = tk.DoubleVar(value=0.0)
        self.sig_end_m = tk.DoubleVar(value=15000.0)
        self.min_toggle_rate = tk.DoubleVar(value=0.5)
        self.max_toggle_rate = tk.DoubleVar(value=10.0)
        self.auto_toggle_selector = tk.BooleanVar(value=True)
        self.day_min_toggle_rate = tk.DoubleVar(value=75.0)
        self.day_max_toggle_rate = tk.DoubleVar(value=130.0)
        self.toggle_bg_switch_threshold_mhz = tk.DoubleVar(value=10.0)
        self.pretrigger_trim_bins = tk.IntVar(value=24)
        self.blend_r1_m = tk.DoubleVar(value=1200.0)
        self.blend_r2_m = tk.DoubleVar(value=1800.0)
        self.energy_mj = tk.DoubleVar(value=25.0)

        self.auto_blend = tk.BooleanVar(value=True)
        self.photon_only = tk.BooleanVar(value=False)
        self.blend_search_start_m = tk.DoubleVar(value=500.0)  # unused in v11.2
        self.blend_search_end_m = tk.DoubleVar(value=3000.0)  # unused in v11.2
        self.blend_window_m = tk.DoubleVar(value=300.0)
        self.blend_step_bins = tk.IntVar(value=5)
        self.blend_r2_threshold = tk.DoubleVar(value=0.985)
        self.blend_min_points = tk.IntVar(value=25)
        self.blend_norm_rmse_max = tk.DoubleVar(value=0.12)
        self.blend_cluster_min_m = tk.DoubleVar(value=300.0)

        self.strict = tk.BooleanVar(value=True)
        self.preset_name = tk.StringVar(value="Custom")

        self.data_source = tk.StringVar(value="Processed")
        self.plot_mode = tk.StringVar(value="NRB")
        self.xmin_var = tk.StringVar(value="")
        self.xmax_var = tk.StringVar(value="")
        self.ymin_var = tk.StringVar(value="")
        self.ymax_var = tk.StringVar(value="")
        self.chart_title_var = tk.StringVar(value="")
        self.x_title_var = tk.StringVar(value="")
        self.y_title_var = tk.StringVar(value="")
        self.progress = tk.DoubleVar(value=0.0)

        self._status_set = None
        self._times_map: Dict[str, Path] = {}
        self._latest_profile_df: Optional[pd.DataFrame] = None
        self._plot_cache: Dict[tuple, pd.DataFrame] = {}
        self._last_prof_meta = None

        self._build_ui()
        self._update_bg_mode_ui()
        self._update_shift_mode_ui()
        self._update_plot_mode_options()
        for var in (self.date, self.pattern, self.start_time):
            var.trace_add("write", lambda *_: self._refresh_time_list())
        self.bg_mode.trace_add("write", lambda *_: self._update_bg_mode_ui())

    def _build_ui(self):
        make_header(
            self,
            "NRB Daily Builder",
            "v11.3 · DAT input · shift fixed to zero · pretrigger BG trim + auto day/night toggle selector",
        ).pack(fill="x", padx=16, pady=(12, 4))

        main = ttk.Panedwindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=16, pady=10)

        left_host = tk.Frame(main, bg=BG)
        right = tk.Frame(main, bg=BG)
        main.add(left_host, weight=5)
        main.add(right, weight=6)

        # Scrollable left column so Run/Console remains reachable on smaller screens.
        self.left_canvas = tk.Canvas(left_host, bg=BG, highlightthickness=0)
        left_scroll = ttk.Scrollbar(left_host, orient="vertical", command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=left_scroll.set)
        self.left_canvas.pack(side="left", fill="both", expand=True)
        left_scroll.pack(side="right", fill="y")

        left = tk.Frame(self.left_canvas, bg=BG)
        self._left_canvas_window = self.left_canvas.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>", self._on_left_inner_configure)
        self.left_canvas.bind("<Configure>", self._on_left_canvas_configure)
        self.left_canvas.bind("<Enter>", self._bind_left_mousewheel)
        self.left_canvas.bind("<Leave>", self._unbind_left_mousewheel)

        c_in, f_in = make_card(left, "📁 Input / Output")
        c_in.pack(fill="x", pady=(0, 10))
        row = tk.Frame(f_in, bg=CARD)
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self.folder).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row, text="Browse…", command=self.pick_folder).pack(side="left")
        row2 = tk.Frame(f_in, bg=CARD)
        row2.pack(fill="x", pady=(8, 0))
        tk.Label(row2, text="Date:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(row2, textvariable=self.date, width=12).pack(side="left", padx=(4, 12))
        tk.Label(row2, text="Start time (HH:MM):", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(row2, textvariable=self.start_time, width=8).pack(side="left", padx=(4, 12))
        tk.Label(row2, text="Pattern:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(row2, textvariable=self.pattern, width=10).pack(side="left", padx=(4, 12))
        ttk.Checkbutton(row2, text="Strict mode", variable=self.strict).pack(side="left")
        row3 = tk.Frame(f_in, bg=CARD)
        row3.pack(fill="x", pady=(8, 0))
        ttk.Entry(row3, textvariable=self.out_path).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row3, text="Save As…", command=self.pick_output).pack(side="left")

        c_p, f_p = make_card(left, "⚙ Parameters")
        c_p.pack(fill="x", pady=(0, 10))
        p0 = tk.Frame(f_p, bg=CARD)
        p0.pack(fill="x", pady=(0, 6))
        ttk.Button(p0, text="Licel pretrigger 1024", command=lambda: self.apply_preset("Licel pretrigger 1024")).pack(side="left", padx=(0, 6))
        ttk.Button(p0, text="Legacy no pretrigger", command=lambda: self.apply_preset("Legacy no pretrigger")).pack(side="left", padx=(0, 6))
        ttk.Button(p0, text="Auto detect", command=self.auto_detect_bins).pack(side="left", padx=(0, 6))
        ttk.Label(p0, textvariable=self.preset_name).pack(side="right")

        tk.Label(
            f_p,
            text="Grouped by pipeline order: Raw layout → Dead time → Auto day/night toggle select → Toggle fit → Threshold-based glue → Pretrigger background → Normalization",
            bg=CARD,
            fg=SUBTEXT,
            font=F_SMALL,
        ).pack(anchor="w", pady=(0, 8))

        g0 = ttk.LabelFrame(f_p, text="Raw layout / geometry")
        g0.pack(fill="x", pady=(0, 8))
        g0r1 = tk.Frame(g0, bg=CARD)
        g0r1.pack(fill="x", padx=8, pady=(8, 4))
        tk.Label(g0r1, text="bin_spacing_m:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(g0r1, textvariable=self.bin_spacing_m, width=8).pack(side="left", padx=(4, 12))
        tk.Label(g0r1, text="pretrigger_bins:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        self.pretrigger_entry = ttk.Entry(g0r1, textvariable=self.pretrigger_bins, width=8)
        self.pretrigger_entry.pack(side="left", padx=(4, 12))
        tk.Label(g0r1, text="first_signal_bin:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(g0r1, textvariable=self.first_signal_bin, width=8).pack(side="left", padx=(4, 12))
        ttk.Button(g0r1, text="Sync = pre+1", command=self.sync_first_signal_bin).pack(side="left")
        g0r2 = tk.Frame(g0, bg=CARD)
        g0r2.pack(fill="x", padx=8, pady=(0, 8))
        tk.Label(g0r2, text="first_signal_range_m:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(g0r2, textvariable=self.first_signal_range_m, width=8).pack(side="left", padx=(4, 0))

        g1 = ttk.LabelFrame(f_p, text="Step 1 · Shift policy")
        g1.pack(fill="x", pady=(0, 8))
        g1r1 = tk.Frame(g1, bg=CARD)
        g1r1.pack(fill="x", padx=8, pady=(8, 4))
        tk.Label(g1r1, text="shift_mode:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        self.shift_mode_box = ttk.Combobox(g1r1, textvariable=self.shift_mode, state="readonly", width=12, values=["manual"])
        self.shift_mode_box.pack(side="left", padx=(4, 12))
        tk.Label(g1r1, text="manual bin_shift:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        self.bin_shift_entry = ttk.Entry(g1r1, textvariable=self.bin_shift_bins, width=8)
        self.bin_shift_entry.pack(side="left", padx=(4, 12))
        tk.Label(g1r1, text="Shift disabled in this version (fixed = 0)", bg=CARD, fg=TEXT, font=F_SMALL).pack(side="left")
        g1r2 = tk.Frame(g1, bg=CARD)
        g1r2.pack(fill="x", padx=8, pady=(0, 8))
        tk.Label(g1r2, text="No onset alignment / no bin shift in v11.2", bg=CARD, fg="#92400E", font=F_SMALL).pack(side="left")

        g2 = ttk.LabelFrame(f_p, text="Step 2 · Dead time + toggle fit / threshold glue")
        g2.pack(fill="x", pady=(0, 8))
        g2r1 = tk.Frame(g2, bg=CARD)
        g2r1.pack(fill="x", padx=8, pady=(8, 4))
        tk.Label(g2r1, text="dead_time_ns:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(g2r1, textvariable=self.dead_time_ns, width=8).pack(side="left", padx=(4, 12))
        tk.Label(g2r1, text="sig_start_m:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(g2r1, textvariable=self.sig_start_m, width=8).pack(side="left", padx=(4, 12))
        tk.Label(g2r1, text="sig_end_m:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(g2r1, textvariable=self.sig_end_m, width=8).pack(side="left", padx=(4, 12))
        g2r2 = tk.Frame(g2, bg=CARD)
        g2r2.pack(fill="x", padx=8, pady=(0, 4))
        tk.Label(g2r2, text="min_toggle_rate:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(g2r2, textvariable=self.min_toggle_rate, width=8).pack(side="left", padx=(4, 12))
        tk.Label(g2r2, text="max_toggle_rate:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(g2r2, textvariable=self.max_toggle_rate, width=8).pack(side="left", padx=(4, 12))
        tk.Label(g2r2, text="blend_r1_m:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(g2r2, textvariable=self.blend_r1_m, width=8).pack(side="left", padx=(4, 12))
        tk.Label(g2r2, text="blend_r2_m:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(g2r2, textvariable=self.blend_r2_m, width=8).pack(side="left", padx=(4, 12))
        ttk.Checkbutton(g2r2, text="Auto day/night toggle", variable=self.auto_toggle_selector).pack(side="left")

        g2r3 = tk.Frame(g2, bg=CARD)
        g2r3.pack(fill="x", padx=8, pady=(0, 4))
        tk.Label(g2r3, text="day_min_toggle:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(g2r3, textvariable=self.day_min_toggle_rate, width=8).pack(side="left", padx=(4, 12))
        tk.Label(g2r3, text="day_max_toggle:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(g2r3, textvariable=self.day_max_toggle_rate, width=8).pack(side="left", padx=(4, 12))
        tk.Label(g2r3, text="BG switch threshold:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(g2r3, textvariable=self.toggle_bg_switch_threshold_mhz, width=8).pack(side="left", padx=(4, 12))
        tk.Label(g2r3, text="pretrigger trim bins:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(g2r3, textvariable=self.pretrigger_trim_bins, width=8).pack(side="left", padx=(4, 0))

        g2r4 = tk.Frame(g2, bg=CARD)
        g2r4.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Checkbutton(g2r4, text="Auto blend r1/r2 from photon_dt threshold crossings", variable=self.auto_blend).pack(side="left")
        ttk.Checkbutton(g2r4, text="Skip glue (photon only)", variable=self.photon_only).pack(side="left", padx=(12, 0))
        tk.Label(g2r4, text="manual fallback r1/r2 used if threshold crossings fail", bg=CARD, fg="#92400E", font=F_SMALL).pack(side="left", padx=(8, 0))

        g2r5 = tk.Frame(g2, bg=CARD)
        g2r5.pack(fill="x", padx=8, pady=(0, 8))
        tk.Label(
            g2r5,
            text="Auto toggle: if trimmed pretrigger photon_dt BG > switch threshold, use day min/max; otherwise use night min/max.",
            bg=CARD,
            fg="#92400E",
            font=F_SMALL,
        ).pack(side="left")

        g3 = ttk.LabelFrame(f_p, text="Step 3 · Background (pretrigger only)")
        g3.pack(fill="x", pady=(0, 8))
        g3r1 = tk.Frame(g3, bg=CARD)
        g3r1.pack(fill="x", padx=8, pady=(8, 4))
        tk.Label(g3r1, text="BG mode:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        self.bg_mode_box = ttk.Combobox(g3r1, textvariable=self.bg_mode, state="readonly", width=12, values=["pretrigger", "fixed", "far_range"])
        self.bg_mode_box.pack(side="left", padx=(4, 12))
        tk.Label(g3r1, text="bg_start_m:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        self.bg_start_entry = ttk.Entry(g3r1, textvariable=self.bg_start_m, width=8)
        self.bg_start_entry.pack(side="left", padx=(4, 12))
        tk.Label(g3r1, text="bg_end_m:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        self.bg_end_entry = ttk.Entry(g3r1, textvariable=self.bg_end_m, width=8)
        self.bg_end_entry.pack(side="left", padx=(4, 0))

        g4 = ttk.LabelFrame(f_p, text="Step 4 · Normalization / output scale")
        g4.pack(fill="x", pady=(0, 6))
        g4r1 = tk.Frame(g4, bg=CARD)
        g4r1.pack(fill="x", padx=8, pady=(8, 4))
        tk.Label(g4r1, text="energy_mJ:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(g4r1, textvariable=self.energy_mj, width=8).pack(side="left", padx=(4, 12))
        g4r2 = tk.Frame(g4, bg=CARD)
        g4r2.pack(fill="x", padx=8, pady=(0, 8))
        tk.Label(g4r2, text="Processed pipeline:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        tk.Label(g4r2, text="shift=0 → dead time → toggle fit → auto/manual blend → pretrigger BG → r² → /energy → normalize", bg=CARD, fg=TEXT, font=F_SMALL).pack(side="left", padx=(6, 0))

        c_run, f_run = make_card(left, "▶ Run / Console")
        c_run.pack(fill="both", expand=True)
        top = tk.Frame(f_run, bg=CARD)
        top.pack(fill="x")
        self.run_btn = ttk.Button(top, text="Run Build NRB Profile", command=self.run)
        self.run_btn.pack(side="left")
        self.pb = ttk.Progressbar(top, variable=self.progress, maximum=100, length=220)
        self.pb.pack(side="left", padx=(12, 0))
        ttk.Button(top, text="Clear", command=self._clear_log).pack(side="right")
        self.log_txt = tk.Text(f_run, height=16, bg="#0B1220", fg="#E2E8F0", insertbackground="#E2E8F0")
        self.log_txt.pack(fill="both", expand=True, pady=(8, 0))
        self.log_txt.configure(state="disabled")

        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)
        c_plot, f_plot = make_card(right, "📈 Profile Plot")
        c_plot.grid(row=0, column=0, sticky="nsew")

        ctrl1 = tk.Frame(f_plot, bg=CARD)
        ctrl1.pack(fill="x", pady=(0, 6))
        tk.Label(ctrl1, text="Source:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        self.source_box = ttk.Combobox(ctrl1, textvariable=self.data_source, state="readonly", width=12, values=["Processed", "Raw .dat"])
        self.source_box.pack(side="left", padx=(4, 12))
        self.source_box.bind("<<ComboboxSelected>>", lambda e: self._update_plot_mode_options())
        tk.Label(ctrl1, text="Plot:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        self.plot_mode_box = ttk.Combobox(ctrl1, textvariable=self.plot_mode, state="readonly", width=16)
        self.plot_mode_box.pack(side="left", padx=(4, 12))
        ttk.Button(ctrl1, text="Refresh Plot", command=self.refresh_plot).pack(side="left")
        ttk.Button(ctrl1, text="Auto Scale", command=self.auto_scale).pack(side="left", padx=(6, 0))
        ttk.Button(ctrl1, text="Save PNG", command=self.save_png).pack(side="left", padx=(18, 0))
        ttk.Button(ctrl1, text="Export CSV", command=self.save_csv).pack(side="left", padx=(6, 0))

        ctrl2 = tk.Frame(f_plot, bg=CARD)
        ctrl2.pack(fill="x", pady=(0, 8))
        tk.Label(ctrl2, text="x min:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(ctrl2, textvariable=self.xmin_var, width=8).pack(side="left", padx=(4, 8))
        tk.Label(ctrl2, text="x max:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(ctrl2, textvariable=self.xmax_var, width=8).pack(side="left", padx=(4, 12))
        tk.Label(ctrl2, text="y min:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(ctrl2, textvariable=self.ymin_var, width=8).pack(side="left", padx=(4, 8))
        tk.Label(ctrl2, text="y max:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(ctrl2, textvariable=self.ymax_var, width=8).pack(side="left", padx=(4, 12))
        ttk.Button(ctrl2, text="Apply Axis", command=self.apply_axes).pack(side="left")

        ctrl3 = tk.Frame(f_plot, bg=CARD)
        ctrl3.pack(fill="x", pady=(0, 8))
        tk.Label(ctrl3, text="Chart title:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(ctrl3, textvariable=self.chart_title_var, width=24).pack(side="left", padx=(4, 12))
        tk.Label(ctrl3, text="X title:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(ctrl3, textvariable=self.x_title_var, width=18).pack(side="left", padx=(4, 12))
        tk.Label(ctrl3, text="Y title:", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(side="left")
        ttk.Entry(ctrl3, textvariable=self.y_title_var, width=18).pack(side="left", padx=(4, 12))
        ttk.Button(ctrl3, text="Apply Title", command=self.apply_axes).pack(side="left")

        select_row = tk.Frame(f_plot, bg=CARD)
        select_row.pack(fill="both", expand=False)
        tk.Label(select_row, text="Times (multi-select):", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack(anchor="w")
        lb_outer = tk.Frame(select_row, bg=CARD)
        lb_outer.pack(fill="x", pady=(4, 8))
        self.time_list = tk.Listbox(lb_outer, selectmode="extended", exportselection=False, height=8)
        self.time_list.pack(side="left", fill="x", expand=True)
        ysb = ttk.Scrollbar(lb_outer, orient="vertical", command=self.time_list.yview)
        ysb.pack(side="left", fill="y")
        self.time_list.config(yscrollcommand=ysb.set)
        btns = tk.Frame(lb_outer, bg=CARD)
        btns.pack(side="left", padx=(8, 0), anchor="n")
        ttk.Button(btns, text="Select All", command=self.select_all_times).pack(fill="x", pady=(0, 4))
        ttk.Button(btns, text="Clear", command=lambda: self.time_list.selection_clear(0, "end")).pack(fill="x")

        self.fig = Figure(figsize=(7.2, 6.4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("No data")
        self.ax.set_xlabel("Range (m)")
        self.ax.set_ylabel("Signal")
        self.ax.grid(True, alpha=0.3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=f_plot)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        sb, self.status_var, self._status_set = make_status_bar(self)
        sb.pack(fill="x", side="bottom")

    def _on_left_inner_configure(self, _event=None):
        if hasattr(self, 'left_canvas'):
            self.left_canvas.configure(scrollregion=self.left_canvas.bbox('all'))

    def _on_left_canvas_configure(self, event):
        if hasattr(self, '_left_canvas_window'):
            self.left_canvas.itemconfigure(self._left_canvas_window, width=event.width)

    def _bind_left_mousewheel(self, _event=None):
        self.left_canvas.bind_all('<MouseWheel>', self._on_left_mousewheel)
        self.left_canvas.bind_all('<Button-4>', self._on_left_mousewheel)
        self.left_canvas.bind_all('<Button-5>', self._on_left_mousewheel)

    def _unbind_left_mousewheel(self, _event=None):
        self.left_canvas.unbind_all('<MouseWheel>')
        self.left_canvas.unbind_all('<Button-4>')
        self.left_canvas.unbind_all('<Button-5>')

    def _on_left_mousewheel(self, event):
        if getattr(event, 'num', None) == 4:
            self.left_canvas.yview_scroll(-1, 'units')
        elif getattr(event, 'num', None) == 5:
            self.left_canvas.yview_scroll(1, 'units')
        else:
            delta = getattr(event, 'delta', 0)
            if delta:
                self.left_canvas.yview_scroll(int(-delta / 120), 'units')

    def _safe(self, fn, *a, **kw):
        self.after(0, lambda: fn(*a, **kw))

    def _log(self, msg: str):
        append_log(self.log_txt, msg)

    def _safe_log(self, msg: str):
        self._safe(self._log, msg)

    def _clear_log(self):
        self.log_txt.configure(state="normal")
        self.log_txt.delete("1.0", "end")
        self.log_txt.configure(state="disabled")

    def _set_status(self, msg: str, tag: str = "info"):
        if self._status_set is not None:
            self._status_set(msg, tag)

    def _update_bg_mode_ui(self):
        mode = self.bg_mode.get().strip().lower()
        self.pretrigger_entry.configure(state="normal")
        if mode == "pretrigger":
            try:
                self.bg_start_m.set(0.0)
                self.bg_end_m.set(float(self.pretrigger_bins.get()) * float(self.bin_spacing_m.get()))
            except Exception:
                pass
            self.bg_start_entry.configure(state="disabled")
            self.bg_end_entry.configure(state="disabled")
        elif mode == "fixed":
            self.bg_start_entry.configure(state="normal")
            self.bg_end_entry.configure(state="normal")
        else:  # far_range
            self.bg_start_entry.configure(state="disabled")
            self.bg_end_entry.configure(state="disabled")

    def _update_shift_mode_ui(self):
        self.shift_mode.set("manual")
        self.bin_shift_bins.set(0)
        if hasattr(self, "bin_shift_entry"):
            self.bin_shift_entry.configure(state="disabled")
        # v11 has no onset widgets; keep guards so init can call this safely.
        for name in ("onset_start_entry", "onset_end_entry", "onset_smooth_entry"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.configure(state="disabled")

    def _update_plot_mode_options(self):
        vals = ["Analog Raw", "Photon Raw"] if self.data_source.get() == "Raw .dat" else ["Photon DT", "Analog Scaled", "Glue", "Glue Overlay", "NRB"]
        self.plot_mode_box.configure(values=vals)
        if self.plot_mode.get() not in vals:
            self.plot_mode.set(vals[0])

    def sync_first_signal_bin(self):
        self.first_signal_bin.set(int(self.pretrigger_bins.get()) + 1)
        self.preset_name.set("Custom")

    def apply_preset(self, name: str):
        if name == "Licel pretrigger 1024":
            self.bg_mode.set("pretrigger")
            self.pretrigger_bins.set(1024)
            self.first_signal_bin.set(1025)
            self.first_signal_range_m.set(3.75)
            self.shift_mode.set("manual")
            self.bin_shift_bins.set(0)
            self.onset_start_m.set(3500.0)
            self.onset_end_m.set(4000.0)
            self.onset_smooth_bins.set(5)
            self.sig_start_m.set(0.0)
            self.sig_end_m.set(15000.0)
            self.min_toggle_rate.set(0.5)
            self.max_toggle_rate.set(10.0)
            self.day_min_toggle_rate.set(75.0)
            self.day_max_toggle_rate.set(130.0)
            self.toggle_bg_switch_threshold_mhz.set(10.0)
            self.pretrigger_trim_bins.set(24)
            self.blend_r1_m.set(1200.0)
            self.blend_r2_m.set(1800.0)
            self.auto_blend.set(True)
            self.blend_search_start_m.set(500.0)
            self.blend_search_end_m.set(3000.0)
            self.blend_window_m.set(300.0)
            self.blend_step_bins.set(5)
            self.blend_r2_threshold.set(0.985)
            self.blend_min_points.set(25)
            self.blend_norm_rmse_max.set(0.12)
            self.blend_cluster_min_m.set(300.0)
            self.energy_mj.set(25.0)
        elif name == "Legacy no pretrigger":
            self.bg_mode.set("fixed")
            self.pretrigger_bins.set(0)
            self.first_signal_bin.set(1)
            self.first_signal_range_m.set(3.75)
            self.bg_start_m.set(13000.0)
            self.bg_end_m.set(14500.0)
            self.shift_mode.set("manual")
            self.bin_shift_bins.set(0)
            self.sig_start_m.set(0.0)
            self.sig_end_m.set(15000.0)
            self.min_toggle_rate.set(0.5)
            self.max_toggle_rate.set(10.0)
            self.day_min_toggle_rate.set(75.0)
            self.day_max_toggle_rate.set(130.0)
            self.toggle_bg_switch_threshold_mhz.set(10.0)
            self.pretrigger_trim_bins.set(0)
            self.blend_r1_m.set(1200.0)
            self.blend_r2_m.set(1800.0)
            self.auto_blend.set(True)
            self.blend_search_start_m.set(500.0)
            self.blend_search_end_m.set(3000.0)
            self.blend_window_m.set(300.0)
            self.blend_step_bins.set(5)
            self.blend_r2_threshold.set(0.985)
            self.blend_min_points.set(25)
            self.blend_norm_rmse_max.set(0.12)
            self.blend_cluster_min_m.set(300.0)
            self.energy_mj.set(25.0)
        self.preset_name.set(name)
        self._update_bg_mode_ui()
        self._update_shift_mode_ui()

    def auto_detect_bins(self):
        folder = self.folder.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Please choose a valid folder first.")
            return
        files = sorted(Path(folder).glob(self.pattern.get().strip() or "*.dat"))
        if not files:
            messagebox.showerror("Error", "No matching .dat files found.")
            return
        try:
            arr = _read_tr40_dat_ascii_array(files[0])
            bins = int(arr.shape[0])
            self._log(f"Auto detect: {files[0].name} has {bins} numeric bins")
            if bins >= 5024:
                self.apply_preset("Licel pretrigger 1024")
            else:
                self.apply_preset("Legacy no pretrigger")
        except Exception as e:
            messagebox.showerror("Auto detect failed", str(e))

    def pick_folder(self):
        p = filedialog.askdirectory(title="Select folder containing .dat files")
        if not p:
            return
        self.folder.set(p)
        self._log(f"Folder: {p}")
        self.out_path.set(str(Path(p).parent / f"NRB-reference-glue-{self.date.get().strip()}.xlsx"))
        self._refresh_time_list()

    def pick_output(self):
        out = filedialog.asksaveasfilename(title="Save NRB profile as", defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if out:
            self.out_path.set(out)
            self._log(f"Output: {out}")

    def _validate_inputs(self):
        folder = self.folder.get().strip()
        if not folder or not os.path.isdir(folder):
            raise ValueError("Please choose a valid folder.")
        pd.to_datetime(self.date.get().strip())
        if self.start_time.get().strip():
            pd.to_datetime(self.start_time.get().strip(), format="%H:%M", errors="raise")
        if not self.out_path.get().strip():
            raise ValueError("Please choose output path.")
        if int(self.first_signal_bin.get()) <= int(self.pretrigger_bins.get()):
            raise ValueError("first_signal_bin must be greater than pretrigger_bins.")
        if float(self.first_signal_range_m.get()) <= 0:
            raise ValueError("first_signal_range_m must be > 0.")
        if float(self.energy_mj.get()) <= 0:
            raise ValueError("energy_mj must be > 0.")
        if float(self.sig_end_m.get()) <= float(self.sig_start_m.get()):
            raise ValueError("sig_end_m must be greater than sig_start_m.")
        if float(self.max_toggle_rate.get()) <= float(self.min_toggle_rate.get()):
            raise ValueError("night max_toggle_rate must be greater than night min_toggle_rate.")
        if float(self.day_max_toggle_rate.get()) <= float(self.day_min_toggle_rate.get()):
            raise ValueError("day max_toggle_rate must be greater than day min_toggle_rate.")
        if int(self.pretrigger_trim_bins.get()) < 0:
            raise ValueError("pretrigger_trim_bins must be >= 0.")
        if int(self.pretrigger_trim_bins.get()) >= int(self.pretrigger_bins.get()):
            raise ValueError("pretrigger_trim_bins must be smaller than pretrigger_bins.")
        if not bool(self.auto_blend.get()) and float(self.blend_r2_m.get()) <= float(self.blend_r1_m.get()):
            raise ValueError("blend_r2_m must be greater than blend_r1_m when auto blend is off.")

    def _refresh_time_list(self):
        self.time_list.delete(0, "end")
        self._times_map.clear()
        folder = self.folder.get().strip()
        if not folder or not os.path.isdir(folder):
            return
        try:
            ordered, _qc = collect_actual_timestamps(
                Path(folder),
                pattern=self.pattern.get().strip() or "*.dat",
                date_str=self.date.get().strip(),
                start_time=self.start_time.get().strip(),
            )
        except Exception:
            return
        for ts, f in ordered:
            hhmm = ts.strftime("%H:%M")
            if hhmm not in self._times_map:
                self._times_map[hhmm] = f
        for hhmm in sorted(self._times_map.keys()):
            self.time_list.insert("end", hhmm)

    def select_all_times(self):
        self.time_list.selection_set(0, "end")

    def _selected_times(self) -> List[str]:
        return [self.time_list.get(i) for i in self.time_list.curselection()]

    def _build_plot_df(self, hhmm: str) -> pd.DataFrame:
        key = (
            self.data_source.get(), self.plot_mode.get(), hhmm,
            self.bg_mode.get(), int(self.pretrigger_bins.get()), int(self.first_signal_bin.get()),
            float(self.first_signal_range_m.get()), float(self.bin_spacing_m.get()),
            float(self.dead_time_ns.get()), float(self.bg_start_m.get()), float(self.bg_end_m.get()),
            self.shift_mode.get(), int(self.bin_shift_bins.get()), float(self.onset_start_m.get()), float(self.onset_end_m.get()),
            int(self.onset_smooth_bins.get()), float(self.sig_start_m.get()), float(self.sig_end_m.get()),
            float(self.min_toggle_rate.get()), float(self.max_toggle_rate.get()), bool(self.auto_toggle_selector.get()),
            float(self.day_min_toggle_rate.get()), float(self.day_max_toggle_rate.get()),
            float(self.toggle_bg_switch_threshold_mhz.get()), int(self.pretrigger_trim_bins.get()),
            float(self.blend_r1_m.get()), float(self.blend_r2_m.get()), bool(self.auto_blend.get()), float(self.blend_search_start_m.get()),
            float(self.blend_search_end_m.get()), float(self.blend_window_m.get()), int(self.blend_step_bins.get()),
            float(self.blend_r2_threshold.get()), int(self.blend_min_points.get()), float(self.blend_norm_rmse_max.get()),
            float(self.blend_cluster_min_m.get()), float(self.energy_mj.get()),
            bool(self.photon_only.get()),
        )
        if key in self._plot_cache:
            return self._plot_cache[key]
        path = self._times_map.get(hhmm)
        if path is None:
            raise ValueError(f"No file found for {hhmm}")
        prof, meta = build_single_profile(
            path,
            dr_m=float(self.bin_spacing_m.get()),
            dead_time_ns=float(self.dead_time_ns.get()),
            bg_mode=self.bg_mode.get().strip().lower(),
            bg_start_m=float(self.bg_start_m.get()),
            bg_end_m=float(self.bg_end_m.get()),
            pretrigger_bins=int(self.pretrigger_bins.get()),
            first_signal_bin=int(self.first_signal_bin.get()),
            first_signal_range_m=float(self.first_signal_range_m.get()),
            shift_mode="manual",
            bin_shift_bins=0,
            onset_start_m=float(self.onset_start_m.get()),
            onset_end_m=float(self.onset_end_m.get()),
            onset_smooth_bins=int(self.onset_smooth_bins.get()),
            sig_start_m=float(self.sig_start_m.get()),
            sig_end_m=float(self.sig_end_m.get()),
            min_toggle_rate=float(self.min_toggle_rate.get()),
            max_toggle_rate=float(self.max_toggle_rate.get()),
            auto_toggle_selector=bool(self.auto_toggle_selector.get()),
            day_min_toggle_rate=float(self.day_min_toggle_rate.get()),
            day_max_toggle_rate=float(self.day_max_toggle_rate.get()),
            toggle_bg_switch_threshold_mhz=float(self.toggle_bg_switch_threshold_mhz.get()),
            pretrigger_trim_bins=int(self.pretrigger_trim_bins.get()),
            blend_r1_m=float(self.blend_r1_m.get()),
            blend_r2_m=float(self.blend_r2_m.get()),
            auto_blend=bool(self.auto_blend.get()),
            blend_search_start_m=float(self.blend_search_start_m.get()),
            blend_search_end_m=float(self.blend_search_end_m.get()),
            blend_window_m=float(self.blend_window_m.get()),
            blend_step_bins=int(self.blend_step_bins.get()),
            blend_r2_threshold=float(self.blend_r2_threshold.get()),
            blend_min_points=int(self.blend_min_points.get()),
            blend_norm_rmse_max=float(self.blend_norm_rmse_max.get()),
            blend_cluster_min_m=float(self.blend_cluster_min_m.get()),
            energy_mj=float(self.energy_mj.get()),
            gluing_mode=self._gluing_mode_str(),
        )
        self._last_prof_meta = (hhmm, prof, meta)
        src = self.data_source.get()
        mode = self.plot_mode.get()
        out = pd.DataFrame({"Range (m)": prof["range_m"].to_numpy(float)})
        if src == "Raw .dat":
            if mode == "Analog Raw":
                out[hhmm] = prof["analog_mV"].to_numpy(float)
            else:
                out[hhmm] = prof["photon_MHz"].to_numpy(float)
        else:
            if mode == "Photon DT":
                out[hhmm] = prof["photon_deadtime_corr_MHz"].to_numpy(float)
            elif mode == "Analog Scaled":
                out[hhmm] = prof["analog_scaled_MHz"].to_numpy(float)
            elif mode == "Glue":
                out[hhmm] = prof["glued_profile_MHz"].to_numpy(float)
            elif mode == "Glue Overlay":
                # Overlay mode: stash 3 traces as columns
                out[f"{hhmm} · Photon DT"] = prof["photon_deadtime_corr_MHz"].to_numpy(float)
                out[f"{hhmm} · Scaled Analog"] = prof["analog_scaled_MHz"].to_numpy(float)
                out[f"{hhmm} · Glued"] = prof["glued_profile_MHz"].to_numpy(float)
            else:
                out[hhmm] = prof["nrb"].to_numpy(float)
        self._plot_cache[key] = out
        return out

    def _gluing_mode_str(self) -> str:
        return "photon_only" if bool(self.photon_only.get()) else "auto"

    def _merge_selected_plot_df(self) -> pd.DataFrame:
        times = self._selected_times()
        if not times:
            raise ValueError("Please select at least one time.")
        merged: Optional[pd.DataFrame] = None
        for hhmm in times:
            one = self._build_plot_df(hhmm)
            if merged is None:
                merged = one.copy()
            else:
                merged = merged.merge(one, on="Range (m)", how="outer")
        assert merged is not None
        return merged.sort_values("Range (m)")

    def _default_axis_titles(self):
        mode = self.plot_mode.get()
        src = self.data_source.get()
        if src == "Raw .dat":
            ytitle = "Analog (mV)" if mode == "Analog Raw" else "Photon Counting (MHz)"
        else:
            ytitle = {
                "Photon DT": "Photon dead-time corrected (MHz)",
                "Analog Scaled": "Analog scaled (MHz)",
                "Glue": "Glued profile (MHz)",
                "Glue Overlay": "Count rate (MHz)",
                "NRB": "NRB norm",
            }[mode]
        return f"{src} · {mode}", "Range (m)", ytitle

    def refresh_plot(self):
        try:
            df = self._merge_selected_plot_df()
        except Exception as e:
            self.ax.clear()
            self.ax.set_title(str(e))
            self.ax.grid(True, alpha=0.3)
            self.canvas.draw_idle()
            return

        self.ax.clear()
        x = df["Range (m)"].to_numpy(float)
        mode = self.plot_mode.get()
        for col in df.columns[1:]:
            y = df[col].to_numpy(float)
            mask = np.isfinite(x) & np.isfinite(y)
            if np.any(mask):
                style = {}
                if mode == "Glue Overlay":
                    if "Photon DT" in col:
                        style = dict(color="#1f77b4", linewidth=1.0, alpha=0.85)
                    elif "Scaled Analog" in col:
                        style = dict(color="#2ca02c", linewidth=1.0, linestyle="--", alpha=0.85)
                    elif "Glued" in col:
                        style = dict(color="#d62728", linewidth=1.6)
                self.ax.plot(x[mask], y[mask], label=col, **(style or dict(linewidth=1.2)))

        # Glue Overlay extras: threshold + blend zone lines from the most recent profile
        if mode == "Glue Overlay" and getattr(self, "_last_prof_meta", None) is not None:
            _hhmm, _prof, meta = self._last_prof_meta
            min_tog = float(meta.get("min_toggle_rate", np.nan))
            max_tog = float(meta.get("max_toggle_rate", np.nan))
            r1 = float(meta.get("blend_r1_used_m", np.nan))
            r2 = float(meta.get("blend_r2_used_m", np.nan))
            if np.isfinite(min_tog):
                self.ax.axhline(min_tog, color="gray", linestyle=":", linewidth=0.9, alpha=0.7, label=f"min toggle = {min_tog:g}")
            if np.isfinite(max_tog):
                self.ax.axhline(max_tog, color="gray", linestyle=":", linewidth=0.9, alpha=0.7, label=f"max toggle = {max_tog:g}")
            if np.isfinite(r1):
                self.ax.axvline(r1, color="orange", linestyle="--", linewidth=0.9, alpha=0.85, label=f"r1 = {r1:.0f} m")
            if np.isfinite(r2):
                self.ax.axvline(r2, color="purple", linestyle="--", linewidth=0.9, alpha=0.85, label=f"r2 = {r2:.0f} m")

        default_title, default_xtitle, default_ytitle = self._default_axis_titles()
        self.ax.set_xlabel(self.x_title_var.get().strip() or default_xtitle)
        self.ax.set_ylabel(self.y_title_var.get().strip() or default_ytitle)
        self.ax.set_title(self.chart_title_var.get().strip() or default_title)
        self.ax.grid(True, alpha=0.3)
        if mode == "Glue Overlay" or len(df.columns) > 2:
            self.ax.legend(fontsize=8, loc="best")
        self.apply_axes(redraw=False)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _parse_lim(self, s: str):
        s = s.strip()
        return None if s == "" else float(s)

    def apply_axes(self, redraw: bool = True):
        try:
            xmin = self._parse_lim(self.xmin_var.get())
            xmax = self._parse_lim(self.xmax_var.get())
            ymin = self._parse_lim(self.ymin_var.get())
            ymax = self._parse_lim(self.ymax_var.get())
            if xmin is not None or xmax is not None:
                self.ax.set_xlim(left=xmin, right=xmax)
            if ymin is not None or ymax is not None:
                self.ax.set_ylim(bottom=ymin, top=ymax)
            default_title, default_xtitle, default_ytitle = self._default_axis_titles()
            self.ax.set_title(self.chart_title_var.get().strip() or default_title)
            self.ax.set_xlabel(self.x_title_var.get().strip() or default_xtitle)
            self.ax.set_ylabel(self.y_title_var.get().strip() or default_ytitle)
            if redraw:
                self.canvas.draw_idle()
        except Exception as e:
            messagebox.showerror("Axis/title error", str(e))

    def auto_scale(self):
        self.xmin_var.set("")
        self.xmax_var.set("")
        self.ymin_var.set("")
        self.ymax_var.set("")
        self.ax.relim()
        self.ax.autoscale_view()
        self.apply_axes(redraw=True)

    def save_png(self):
        out = filedialog.asksaveasfilename(title="Save PNG", defaultextension=".png", filetypes=[("PNG image", "*.png")])
        if not out:
            return
        self.fig.savefig(out, dpi=150, bbox_inches="tight")
        self._log(f"Saved PNG: {out}")

    def save_csv(self):
        try:
            df = self._merge_selected_plot_df()
        except Exception as e:
            messagebox.showerror("CSV export", str(e))
            return
        try:
            xmin = self._parse_lim(self.xmin_var.get())
            xmax = self._parse_lim(self.xmax_var.get())
            if xmin is not None:
                df = df[df["Range (m)"] >= xmin]
            if xmax is not None:
                df = df[df["Range (m)"] <= xmax]
        except Exception:
            pass
        out = filedialog.asksaveasfilename(title="Save CSV", defaultextension=".csv", filetypes=[("CSV file", "*.csv")])
        if not out:
            return
        df.to_csv(out, index=False)
        self._log(f"Saved CSV: {out}")

    def run(self):
        try:
            self._validate_inputs()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.run_btn.configure(state="disabled")
        self.progress.set(0.0)
        self._set_status("Running…", "running")
        self._log("=== START v11.3 Day/Night Toggle Selector ===")
        self._plot_cache.clear()
        self._refresh_time_list()

        folder = Path(self.folder.get().strip())
        date_str = self.date.get().strip()
        start_time = self.start_time.get().strip()
        pattern = self.pattern.get().strip() or "*.dat"
        out_path = Path(self.out_path.get().strip())

        def worker():
            try:
                profile_df, qc_df, params_df = build_daily_profile_from_folder(
                    folder,
                    date_str=date_str,
                    start_time=start_time,
                    pattern=pattern,
                    out_path=out_path,
                    dr_m=float(self.bin_spacing_m.get()),
                    dead_time_ns=float(self.dead_time_ns.get()),
                    bg_mode=self.bg_mode.get().strip().lower(),
                    bg_start_m=float(self.bg_start_m.get()),
                    bg_end_m=float(self.bg_end_m.get()),
                    pretrigger_bins=int(self.pretrigger_bins.get()),
                    first_signal_bin=int(self.first_signal_bin.get()),
                    first_signal_range_m=float(self.first_signal_range_m.get()),
                    shift_mode="manual",
                    bin_shift_bins=0,
                    onset_start_m=float(self.onset_start_m.get()),
                    onset_end_m=float(self.onset_end_m.get()),
                    onset_smooth_bins=int(self.onset_smooth_bins.get()),
                    sig_start_m=float(self.sig_start_m.get()),
                    sig_end_m=float(self.sig_end_m.get()),
                    min_toggle_rate=float(self.min_toggle_rate.get()),
                    max_toggle_rate=float(self.max_toggle_rate.get()),
                    auto_toggle_selector=bool(self.auto_toggle_selector.get()),
                    day_min_toggle_rate=float(self.day_min_toggle_rate.get()),
                    day_max_toggle_rate=float(self.day_max_toggle_rate.get()),
                    toggle_bg_switch_threshold_mhz=float(self.toggle_bg_switch_threshold_mhz.get()),
                    pretrigger_trim_bins=int(self.pretrigger_trim_bins.get()),
                    blend_r1_m=float(self.blend_r1_m.get()),
                    blend_r2_m=float(self.blend_r2_m.get()),
                    auto_blend=bool(self.auto_blend.get()),
                    blend_search_start_m=float(self.blend_search_start_m.get()),
                    blend_search_end_m=float(self.blend_search_end_m.get()),
                    blend_window_m=float(self.blend_window_m.get()),
                    blend_step_bins=int(self.blend_step_bins.get()),
                    blend_r2_threshold=float(self.blend_r2_threshold.get()),
                    blend_min_points=int(self.blend_min_points.get()),
                    blend_norm_rmse_max=float(self.blend_norm_rmse_max.get()),
                    blend_cluster_min_m=float(self.blend_cluster_min_m.get()),
                    energy_mj=float(self.energy_mj.get()),
                    gluing_mode=self._gluing_mode_str(),
                    strict=bool(self.strict.get()),
                    logger=self._safe_log,
                    progress_cb=lambda v: self._safe(self.progress.set, v),
                )
                self._latest_profile_df = profile_df
                self._safe(self._after_run_success, out_path)
            except Exception as e:
                self._safe_log(f"[FAILED] {e}")
                self._safe(self._set_status, "Failed ❌", "error")
                self._safe(messagebox.showerror, "Failed", str(e))
            finally:
                self._safe(self.run_btn.configure, state="normal")
                self._safe_log("=== END ===")

        threading.Thread(target=worker, daemon=True).start()

    def _after_run_success(self, out_path: Path):
        self._log(f"[OK] Saved: {out_path.resolve()}")
        self._set_status("Done ✅", "ok")
        self.progress.set(100.0)
        self._refresh_time_list()
        if self.time_list.size() > 0:
            self.time_list.selection_set(0)
            self.refresh_plot()


if __name__ == "__main__":
    app = NRBDatDailyGUI()
    app.mainloop()
