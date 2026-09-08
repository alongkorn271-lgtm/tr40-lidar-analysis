"""LiDAR Analysis Suite — Bao Solutions inspired UI (CustomTkinter).

Phase 1: foundation (app shell + sidebar)
Phase 2: Step 1 migrated
Phase 3: Step 2 migrated
Phase 4: Step 3 migrated
Phase 5: Step 4 migrated + configurable save figsize (this commit)
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

import bao_theme as theme

# Reuse the well-tested data-processing helpers from the legacy main.py.
from main import (
    _s1_collect_actual_files,
    _s1_read_copol,
    _s1_read_pbl_km,
    _infer_interval_minutes,
    _parse_hhmm_text,
    _s3_guess_date,
    _s3_parse_times,
    _s3_map_slots,
    PREF_NRB_SHEET,
    PREF_RMIN_SHEET,
    ENGINE_CANDIDATES,
    # Step 4 helpers
    _r4_first_ts,
    _r4_parse_tcols,
    _r4_nearest,
    _r4_int,
    _r4_float,
    _r4_interp,
    _r4_build_x_labels,
    _r4_detect_interval_min,
    # Step 4 constants
    PROTO_SHEET, MPL_SHEET, ALT_SHEET, LEGACY_ALT_SHEET, DENOISED_SHEET,
    ALT_COL_MPL_CHOICES, ALT_COL_PROTO_CHOICES,
    CMAP_CHOICES, INTERP_CHOICES,
)

# Full MPL product extraction (Step 1 — validation references for TR40).
# mpl_reader has no tkinter dependency, safe to import alongside main.
try:
    from mpl_reader import (read_mpl_products, read_mpl_raw,
                            MPL_PARTICLE_TYPE_MAP, _MPL_PROFILE_COLS)
    _HAS_MPL_PRODUCTS = True
except ImportError:
    _HAS_MPL_PRODUCTS = False

# NRB engine (Step 2 backend)
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

# Overlap module (Step 2 — overlap correction)
try:
    from overlap import get_overlap as _get_overlap_array, OverlapHardware
    _HAS_OVERLAP = True
except ImportError:
    _HAS_OVERLAP = False

# Afterpulse module (Step 2 — afterpulse correction)
try:
    from afterpulse import get_afterpulse as _get_afterpulse_array
    _HAS_AFTERPULSE = True
except ImportError:
    _HAS_AFTERPULSE = False

# Fernald/Klett inversion engine (Step 5 — post-NRB aerosol retrieval)
try:
    from fernald_engine import (
        compute_fernald_from_nrb_df as _fernald_run,
        SIGMA_EXT_PRESETS as _SIGMA_PRESETS,
    )
    _HAS_FERNALD = True
except ImportError:
    _HAS_FERNALD = False
    _SIGMA_PRESETS = {}

# AERONET AOD comparison (Step 5 — Track 1 validation)
try:
    from aeronet_loader import (
        load_aeronet_lev as _aeronet_load,
        match_aeronet_to_fernald as _aeronet_match,
        summary_stats as _aeronet_stats,
    )
    _HAS_AERONET = True
except ImportError:
    _HAS_AERONET = False

# Depolarization engine (Step 6 — Track 2 cross-pol)
try:
    from depol_engine import (
        build_daily_depol_from_folders as _depol_run,
        pair_co_cross_files as _depol_pair,
        resolve_co_cross_pairs as _depol_resolve,
        classify_aerosol as _depol_classify,
        _glob_lidar_files as _depol_glob,
        DELTA_MOL_532_NARROW, DELTA_MOL_532_BROAD,
        AEROSOL_TYPE_BANDS,
    )
    _HAS_DEPOL = True
except ImportError:
    _HAS_DEPOL = False
    AEROSOL_TYPE_BANDS = []

# ALT denoiser (Step 5 Display — compute "Denoised" on-the-fly from any loaded NRB,
# using the SAME FFT low-pass ALT uses, so Denoised works without an ALT file).
try:
    from pbl_engine import (
        denoise_profile_preserve_shape as _pbl_denoise,
        FC_CYCLES_PER_M as _PBL_FC, FFT_ORDER as _PBL_ORDER, FFT_PAD_FRAC as _PBL_PAD,
    )
    _HAS_PBL_DENOISE = True
except Exception:
    _HAS_PBL_DENOISE = False

# Matplotlib (Step 2 + Step 4 plots)
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import colors as mcolors, patheffects as mpe
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Match the Tk root to the Bao light theme up-front.
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")  # we override colors per-widget anyway


# ═════════════════════════════════════════════════════════════════════════════
# Shared app state (carry over from main.py)
# ═════════════════════════════════════════════════════════════════════════════
class AppState:
    def __init__(self):
        self.step1_output: Optional[str] = None
        self.step2_output: Optional[str] = None
        self.step3_output: Optional[str] = None
        self.depol_output: Optional[str] = None   # Step 6 (Depolarization) workbook
        self._cbs: List[Callable] = []

    def register_cb(self, fn: Callable):
        self._cbs.append(fn)

    def _notify(self):
        for fn in self._cbs:
            try:
                fn()
            except Exception:
                pass

    def set_step1(self, p: str):
        self.step1_output = p
        self._notify()

    def set_step2(self, p: str):
        self.step2_output = p
        self._notify()

    def set_step3(self, p: str):
        self.step3_output = p
        self._notify()

    def set_depol(self, p: str):
        self.depol_output = p
        self._notify()


# ═════════════════════════════════════════════════════════════════════════════
# Reusable Bao-style components
# ═════════════════════════════════════════════════════════════════════════════
class Card(ctk.CTkFrame):
    """White rounded card with optional title row-+-++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++----------."""

    def __init__(self, master, title: str = "", icon: str = ""):
        super().__init__(master, **theme.card_style())
        self.grid_columnconfigure(0, weight=1)
        self._row = 0
        if title:
            header = ctk.CTkFrame(self, fg_color="transparent")
            header.grid(row=0, column=0, sticky="ew", padx=theme.CARD_PAD_X, pady=(theme.CARD_PAD_Y, 10))
            header.grid_columnconfigure(1, weight=1)
            if icon:
                ctk.CTkLabel(header, text=icon, font=(theme.FONT_FAMILY, 16)).grid(row=0, column=0, padx=(0, 8))
            ctk.CTkLabel(
                header, text=title, font=theme.F_H2, text_color=theme.TEXT_PRIMARY, anchor="w"
            ).grid(row=0, column=1, sticky="w")
            self._row = 1

        # Body container the caller adds widgets to.
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.grid(
            row=self._row, column=0, sticky="nsew",
            padx=theme.CARD_PAD_X,
            pady=(0 if title else theme.CARD_PAD_Y, theme.CARD_PAD_Y),
        )
        self.body.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(self._row, weight=1)


class PageHeader(ctk.CTkFrame):
    def __init__(self, master, title: str, subtitle: str = "", badge: Optional[Tuple[str, str]] = None):
        """badge: (text, tone) where tone in {orange, green, red, navy, default}."""
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(0, weight=1)

        row0 = ctk.CTkFrame(self, fg_color="transparent")
        row0.grid(row=0, column=0, sticky="ew")
        row0.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            row0,
            text=title,
            font=(theme.FONT_FAMILY, 26, "bold"),
            text_color=theme.TEXT_PRIMARY,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        if badge:
            txt, tone = badge
            ctk.CTkLabel(
                row0,
                text=f"  {txt}  ",
                **theme.chip_style(tone),
            ).grid(row=0, column=1, sticky="e", padx=(8, 0))

        if subtitle:
            ctk.CTkLabel(
                self,
                text=subtitle,
                font=theme.F_SMALL,
                text_color=theme.TEXT_SECONDARY,
                anchor="w",
            ).grid(row=1, column=0, sticky="w", pady=(4, 0))


class FieldRow(ctk.CTkFrame):
    """Reusable label+entry row that lays children inline.

    Label auto-wraps to follow the actual row width so long names never get clipped.
    """

    def __init__(self, master, label: str, var: tk.Variable, width: int = 140, hint: str = ""):
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(0, weight=1)
        self._label_text = label
        self.label_widget = ctk.CTkLabel(
            self, text=label, font=theme.F_SMALL, text_color=theme.TEXT_SECONDARY,
            anchor="w", justify="left", wraplength=120,
        )
        self.label_widget.grid(row=0, column=0, sticky="ew")
        self.entry = ctk.CTkEntry(self, textvariable=var, width=width, **theme.input_style())
        self.entry.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        if hint:
            ctk.CTkLabel(
                self, text=hint, font=(theme.FONT_FAMILY, 10), text_color=theme.TEXT_MUTED,
                anchor="w", wraplength=200,
            ).grid(row=2, column=0, sticky="w", pady=(2, 0))

        # Track actual row width and update wraplength reactively
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        # Use almost full row width; never less than 80 px so very short labels stay 1 line.
        w = max(80, int(event.width) - 4)
        try:
            if self.label_widget.cget("wraplength") != w:
                self.label_widget.configure(wraplength=w)
        except Exception:
            pass


class ConsoleLog(ctk.CTkFrame):
    """Dark-tinted console log card with a clear button."""

    def __init__(self, master, title: str = "Console Log"):
        super().__init__(master, **theme.card_style())
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=theme.CARD_PAD_X, pady=(theme.CARD_PAD_Y, 8))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header, text=title, font=theme.F_H2, text_color=theme.TEXT_PRIMARY, anchor="w"
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkButton(
            header,
            text="Clear",
            command=self.clear,
            **theme.ghost_button_style(),
            width=70,
        ).grid(row=0, column=1, padx=(8, 0))

        # The text widget itself: tk.Text inside a CTk container for control
        text_wrap = ctk.CTkFrame(self, fg_color=theme.CHARCOAL, corner_radius=theme.RADIUS_INPUT)
        text_wrap.grid(row=1, column=0, sticky="nsew", padx=theme.CARD_PAD_X, pady=(0, theme.CARD_PAD_Y))
        text_wrap.grid_columnconfigure(0, weight=1)
        text_wrap.grid_rowconfigure(0, weight=1)

        self.text = tk.Text(
            text_wrap,
            bg=theme.CHARCOAL,
            fg="#E8E5E0",
            insertbackground="#E8E5E0",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            font=("Consolas", 10),
            wrap="word",
            state="disabled",
            padx=12,
            pady=10,
        )
        self.text.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

    def log(self, msg: str):
        self.text.configure(state="normal")
        self.text.insert("end", msg + "\n")
        self.text.see("end")
        self.text.configure(state="disabled")

    def clear(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")


class StatusBar(ctk.CTkFrame):
    """Inline status indicator: 🟢/🟡/🔴 + message."""

    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.dot = ctk.CTkLabel(self, text="●", font=(theme.FONT_FAMILY, 12), text_color=theme.TEXT_MUTED)
        self.dot.pack(side="left", padx=(0, 6))
        self.label = ctk.CTkLabel(self, text="Ready", font=theme.F_SMALL, text_color=theme.TEXT_SECONDARY)
        self.label.pack(side="left")

    def set(self, msg: str, tone: str = "info"):
        colors = {
            "info": (theme.TEXT_MUTED, theme.TEXT_SECONDARY),
            "running": (theme.ORANGE, theme.ORANGE),
            "ok": (theme.GREEN, theme.GREEN),
            "error": (theme.RED, theme.RED),
        }
        dot, lbl = colors.get(tone, colors["info"])
        self.dot.configure(text_color=dot)
        self.label.configure(text=msg, text_color=lbl)


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════════
# Display order follows the data-dependency flow:
#   MPL → NRB → Depolarization → ALT → Fernald → Display/Visualize
# (sidebar "Step N" numbers come from list position; page IDs keep their
#  original names internally — only the displayed order/number changes.)
NAV_ITEMS = [
    # Step 2 (NRB Profile) was removed from the workflow — Depolarization now does
    # the same single-channel NRB processing (and publishes step2_output for ALT/
    # Fernald), so it is redundant. The Step2Page class + page id "step2" are kept
    # (unreachable) to avoid touching internal references; only the sidebar order
    # and the displayed "Step N" numbers changed.
    ("step1", "MPL rmin-rmax",       "🌅"),  # 1
    ("step6", "Depolarization",      "🧭"),  # 2
    ("step3", "ALT",                 "📊"),  # 3
    ("step5", "Fernald",             "🔬"),  # 4
    ("step4", "Display / Visualize", "🌈"),  # 5
]


class Sidebar(ctk.CTkFrame):
    EXPANDED_WIDTH = 240
    COLLAPSED_WIDTH = 72

    def __init__(self, master, on_navigate: Callable[[str], None]):
        super().__init__(
            master,
            width=self.EXPANDED_WIDTH,
            corner_radius=0,
            fg_color=theme.SIDEBAR_BG,
            border_width=1,
            border_color=theme.BORDER,
        )
        self.on_navigate = on_navigate
        self._nav_buttons: Dict[str, ctk.CTkButton] = {}
        self._active: Optional[str] = None
        self.collapsed = False

        # Build full and collapsed text for each nav item
        self._full_text = {
            sid: f"  {icon}   Step {i+1} · {label}"
            for i, (sid, label, icon) in enumerate(NAV_ITEMS)
        }
        self._icon_text = {sid: f" {icon}" for sid, _, icon in NAV_ITEMS}

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(99, weight=1)
        self.grid_propagate(False)  # respect explicit width

        # ── Top bar: logo + collapse toggle ─────────────────────────────────
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=12, pady=(20, 8))
        top.grid_columnconfigure(0, weight=1)

        self.logo_frame = ctk.CTkFrame(top, fg_color="transparent")
        self.logo_frame.grid(row=0, column=0, sticky="w", padx=(8, 0))
        self.logo_main = ctk.CTkLabel(
            self.logo_frame, text="LiDAR", font=(theme.FONT_FAMILY, 22, "bold"),
            text_color=theme.ORANGE, anchor="w",
        )
        self.logo_main.grid(row=0, column=0, sticky="w")
        self.logo_sub = ctk.CTkLabel(
            self.logo_frame, text="Analysis Suite", font=(theme.FONT_FAMILY, 11),
            text_color=theme.TEXT_SECONDARY, anchor="w",
        )
        self.logo_sub.grid(row=1, column=0, sticky="w")

        # Collapse toggle button (top-right of sidebar)
        self.toggle_btn = ctk.CTkButton(
            top, text="◀", command=self.toggle_collapse,
            width=28, height=28, corner_radius=8,
            fg_color=theme.CREAM_SOFT, hover_color=theme.CREAM,
            text_color=theme.TEXT_SECONDARY, font=(theme.FONT_FAMILY, 12),
        )
        self.toggle_btn.grid(row=0, column=1, sticky="e")

        # ── Section label ─────────────────────────────────────────────────
        self.section_label = ctk.CTkLabel(
            self, text="WORKFLOW", font=(theme.FONT_FAMILY, 10, "bold"),
            text_color=theme.TEXT_MUTED, anchor="w",
        )
        self.section_label.grid(row=1, column=0, sticky="ew", padx=24, pady=(20, 8))

        # ── Nav items ─────────────────────────────────────────────────────
        for i, (step_id, label, icon) in enumerate(NAV_ITEMS):
            btn = ctk.CTkButton(
                self,
                text=self._full_text[step_id],
                command=lambda sid=step_id: self.on_navigate(sid),
                **theme.nav_item_style(active=False),
            )
            btn.grid(row=2 + i, column=0, sticky="ew", padx=12, pady=2)
            self._nav_buttons[step_id] = btn

        # ── User profile (bottom) ─────────────────────────────────────────
        self.profile = ctk.CTkFrame(self, fg_color=theme.CREAM_SOFT,
                                    corner_radius=theme.RADIUS_BUTTON, height=56)
        self.profile.grid(row=100, column=0, sticky="ew", padx=12, pady=(8, 16))
        self.profile.grid_columnconfigure(1, weight=1)
        self.profile.grid_propagate(False)

        self.profile_avatar = ctk.CTkLabel(
            self.profile, text="👤", font=(theme.FONT_FAMILY, 18), text_color=theme.ORANGE
        )
        self.profile_avatar.grid(row=0, column=0, padx=(12, 8), pady=12, rowspan=2)
        self.profile_name = ctk.CTkLabel(
            self.profile, text="Researcher", font=theme.F_BODY_B,
            text_color=theme.TEXT_PRIMARY, anchor="w",
        )
        self.profile_name.grid(row=0, column=1, sticky="sw", pady=(8, 0))
        self.profile_sub = ctk.CTkLabel(
            self.profile, text="TR40 LiDAR", font=(theme.FONT_FAMILY, 10),
            text_color=theme.TEXT_SECONDARY, anchor="w",
        )
        self.profile_sub.grid(row=1, column=1, sticky="nw", pady=(0, 8))

    def set_active(self, step_id: str):
        if self._active == step_id:
            return
        if self._active and self._active in self._nav_buttons:
            self._nav_buttons[self._active].configure(**theme.nav_item_style(active=False))
            self._refresh_nav_text(self._active)
        if step_id in self._nav_buttons:
            self._nav_buttons[step_id].configure(**theme.nav_item_style(active=True))
            self._refresh_nav_text(step_id)
        self._active = step_id

    def _refresh_nav_text(self, sid: str):
        text = self._icon_text[sid] if self.collapsed else self._full_text[sid]
        self._nav_buttons[sid].configure(text=text)

    def toggle_collapse(self):
        self.collapsed = not self.collapsed
        new_width = self.COLLAPSED_WIDTH if self.collapsed else self.EXPANDED_WIDTH
        self.configure(width=new_width)
        self.toggle_btn.configure(text="▶" if self.collapsed else "◀")

        # Update nav labels
        for sid in self._nav_buttons:
            self._refresh_nav_text(sid)

        # Hide/show logo subtitle, section label, profile labels
        if self.collapsed:
            self.logo_sub.grid_remove()
            self.section_label.grid_remove()
            self.profile_name.grid_remove()
            self.profile_sub.grid_remove()
            self.logo_main.configure(text="L")
        else:
            self.logo_sub.grid()
            self.section_label.grid()
            self.profile_name.grid()
            self.profile_sub.grid()
            self.logo_main.configure(text="LiDAR")


# ═════════════════════════════════════════════════════════════════════════════
# Step 1 Page  –  MPL → rmin-rmax Builder
# ═════════════════════════════════════════════════════════════════════════════
class Step1Page(ctk.CTkFrame):
    def __init__(self, master, app_state: AppState):
        super().__init__(master, fg_color="transparent")
        self.app_state = app_state

        # State variables
        self.folder = tk.StringVar()
        self.date = tk.StringVar()
        self.start_time = tk.StringVar(value="")
        self.delta_m = tk.IntVar(value=300)
        self.out_path = tk.StringVar()
        self.progress = tk.DoubleVar(value=0.0)
        self.rti_channel = tk.StringVar(value="NRB co (parallel)")
        self._copol_df: Optional[pd.DataFrame] = None
        self._crosspol_norm_df: Optional[pd.DataFrame] = None
        self._rti_cbar = None

        # Layout
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(1, weight=1)

        # Header
        header = PageHeader(
            self,
            title="Step 1 · MPL → rmin-rmax Builder",
            subtitle="Actual-files mode · uses real timestamps from filenames",
            badge=("Workflow start", "orange"),
        )
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 16))

        # ── Left column (form) ─────────────────────────────────────────────
        left = ctk.CTkFrame(self, fg_color="transparent")
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        left.grid_columnconfigure(0, weight=1)

        self._build_input_card(left).grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self._build_param_card(left).grid(row=1, column=0, sticky="ew", pady=(0, 12))
        self._build_output_card(left).grid(row=2, column=0, sticky="ew", pady=(0, 12))
        self._build_run_card(left).grid(row=3, column=0, sticky="ew")

        # ── Right column (console) ─────────────────────────────────────────
        right = ctk.CTkFrame(self, fg_color="transparent")
        right.grid(row=1, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=3)   # RTI plot
        right.grid_rowconfigure(1, weight=1)   # console

        self._build_rti_card(right).grid(row=0, column=0, sticky="nsew", pady=(0, 8))

        self.console = ConsoleLog(right, title="Console Log")
        self.console.grid(row=1, column=0, sticky="nsew", pady=(0, 8))

        self.status = StatusBar(right)
        self.status.grid(row=2, column=0, sticky="ew", pady=(4, 0))

    # ── Card builders ──────────────────────────────────────────────────────
    def _build_input_card(self, parent) -> Card:
        card = Card(parent, title="Input Folder", icon="📁")
        body = card.body

        ctk.CTkLabel(
            body, text="Folder containing MPL CSV files",
            font=theme.F_SMALL, text_color=theme.TEXT_SECONDARY, anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))

        ctk.CTkEntry(body, textvariable=self.folder, **theme.input_style()).grid(
            row=1, column=0, sticky="ew", padx=(0, 8)
        )
        ctk.CTkButton(
            body, text="Browse…", command=self.pick_folder, width=100,
            **theme.secondary_button_style(),
        ).grid(row=1, column=1)
        body.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            body, text="Filenames must contain YYYYMMDDHHMM (12 digits)",
            font=(theme.FONT_FAMILY, 10), text_color=theme.TEXT_MUTED, anchor="w",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        return card

    def _build_param_card(self, parent) -> Card:
        card = Card(parent, title="Filter & Parameters", icon="⚙️")
        body = card.body
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_columnconfigure(2, weight=1)

        FieldRow(body, "Date (YYYY-MM-DD)", self.date).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        FieldRow(body, "Start time (HH:MM)", self.start_time).grid(row=0, column=1, sticky="ew", padx=4)
        FieldRow(body, "± Delta (m)", self.delta_m).grid(row=0, column=2, sticky="ew", padx=(8, 0))

        ctk.CTkLabel(
            body,
            text="Mode A: actual files only · no fixed 48 slots · files before Start time are excluded.",
            font=(theme.FONT_FAMILY, 10), text_color=theme.TEXT_MUTED, anchor="w", justify="left",
            wraplength=420,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(10, 0))

        return card

    def _build_output_card(self, parent) -> Card:
        card = Card(parent, title="Output Excel", icon="💾")
        body = card.body
        body.grid_columnconfigure(0, weight=1)

        ctk.CTkEntry(body, textvariable=self.out_path, **theme.input_style()).grid(
            row=0, column=0, sticky="ew", padx=(0, 8)
        )
        ctk.CTkButton(
            body, text="Save As…", command=self.pick_output, width=100,
            **theme.secondary_button_style(),
        ).grid(row=0, column=1)
        return card

    def _build_run_card(self, parent) -> Card:
        card = Card(parent)
        body = card.body
        body.grid_columnconfigure(1, weight=1)

        self.run_btn = ctk.CTkButton(
            body,
            text="▶  Run Build rmin-rmax",
            command=self.run,
            **theme.primary_button_style(),
            width=220,
        )
        self.run_btn.grid(row=0, column=0, sticky="w")

        self.pb = ctk.CTkProgressBar(
            body, variable=self.progress, progress_color=theme.ORANGE,
            fg_color=theme.LIGHT_GRAY, height=8, corner_radius=4,
        )
        self.pb.set(0)
        self.pb.grid(row=0, column=1, sticky="ew", padx=(16, 0))

        return card

    def _build_rti_card(self, parent) -> Card:
        card = Card(parent, title="MPL NRB · RTI Plot", icon="🗺️")
        body = card.body
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(1, weight=1)

        ctrl = ctk.CTkFrame(body, fg_color="transparent")
        ctrl.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ctk.CTkLabel(ctrl, text="Channel:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).pack(side="left")
        self._rti_menu = ctk.CTkOptionMenu(
            ctrl, variable=self.rti_channel,
            values=["NRB co (parallel)", "NRB cross (perpendicular)"],
            command=lambda _v: self.refresh_rti(),
            fg_color=theme.CARD_BG, button_color=theme.CARD_BG, button_hover_color=theme.CREAM_SOFT,
            text_color=theme.TEXT_PRIMARY, dropdown_fg_color=theme.CARD_BG,
            dropdown_text_color=theme.TEXT_PRIMARY, dropdown_hover_color=theme.CREAM,
            corner_radius=theme.RADIUS_INPUT, height=32, font=theme.F_BODY, width=220,
        )
        self._rti_menu.pack(side="left", padx=(6, 12))
        ctk.CTkButton(ctrl, text="Refresh", command=self.refresh_rti,
                      **theme.primary_button_style(width=90, height=32)).pack(side="left", padx=(0, 6))
        ctk.CTkButton(ctrl, text="PNG", command=self.save_rti_png,
                      **theme.ghost_button_style(width=60, height=32)).pack(side="left")

        canvas_wrap = ctk.CTkFrame(body, fg_color=theme.CARD_BG, corner_radius=theme.RADIUS_INPUT,
                                   border_width=1, border_color=theme.BORDER)
        canvas_wrap.grid(row=1, column=0, sticky="nsew")
        canvas_wrap.grid_columnconfigure(0, weight=1)
        canvas_wrap.grid_rowconfigure(0, weight=1)
        self.fig1 = plt.Figure(figsize=(6.5, 4.2), dpi=100, facecolor=theme.CARD_BG)
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_title("Run Step 1 to see the MPL NRB RTI")
        self.ax1.grid(True, alpha=0.3)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=canvas_wrap)
        self.canvas1.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        return card

    def refresh_rti(self):
        ch = self.rti_channel.get()
        df = self._copol_df if ch.startswith("NRB co") else self._crosspol_norm_df
        # Rebuild the whole figure each refresh. Re-using self.ax1 + adding a fresh
        # colorbar every time made the axes shrink on each refresh: colorbar(ax=ax1)
        # steals width from ax1, and .remove() does NOT restore that width, so the
        # plot got progressively smaller. Clearing the figure resets the layout.
        self.fig1.clear()
        self.ax1 = self.fig1.add_subplot(111)
        self._rti_cbar = None
        if df is None or df.shape[1] < 2:
            self.ax1.set_title("No data — run Step 1 first")
            self.ax1.grid(True, alpha=0.3)
            self.fig1.tight_layout(); self.canvas1.draw_idle(); return
        r = df["Range(m)"].to_numpy(float)
        cols = [c for c in df.columns if c != "Range(m)"]
        Z = df[cols].to_numpy(float)          # (nR, nT)
        m = np.isfinite(r)
        x = np.arange(len(cols))
        try:
            mesh = self.ax1.pcolormesh(x, r[m], Z[m, :], cmap="jet",
                                       vmin=0, vmax=1, shading="nearest")
        except Exception as e:
            self.ax1.set_title(f"Plot error: {e}"); self.canvas1.draw_idle(); return
        self._rti_cbar = self.fig1.colorbar(mesh, ax=self.ax1, pad=0.02)
        self._rti_cbar.set_label("Normalized NRB (a.u.)")
        step = max(1, len(cols) // 8)
        self.ax1.set_xticks(x[::step])
        self.ax1.set_xticklabels([cols[i] for i in range(0, len(cols), step)])
        self.ax1.set_xlabel("Time (Local)")
        self.ax1.set_ylabel("Height (m)")
        top = float(np.nanmax(r[m])) if m.any() else 8000.0
        self.ax1.set_ylim(0, min(top, 8000.0))
        self.ax1.set_title(f"MPL {ch} · RTI")
        self.fig1.tight_layout()
        self.canvas1.draw_idle()

    def save_rti_png(self):
        out = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if out:
            self.fig1.savefig(out, dpi=150, bbox_inches="tight")
            self._log(f"Saved PNG: {out}")

    # ── Actions ────────────────────────────────────────────────────────────
    def _safe(self, fn, *a, **kw):
        self.after(0, lambda: fn(*a, **kw))

    def _log(self, msg: str):
        self.console.log(msg)

    def pick_folder(self):
        p = filedialog.askdirectory(title="Select MPL CSV folder")
        if not p:
            return
        self.folder.set(p)
        self._log(f"Folder: {p}")
        self.out_path.set(str(Path(p) / "rmin-rmax.xlsx"))
        self._log(f"Output: {self.out_path.get()}")

    def pick_output(self):
        out = filedialog.asksaveasfilename(
            title="Save rmin-rmax as", defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if out:
            self.out_path.set(out)
            self._log(f"Output: {out}")

    def run(self):
        folder = self.folder.get().strip()
        out_path = self.out_path.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Please choose a valid folder.")
            return
        if not out_path:
            messagebox.showerror("Error", "Please choose an output path.")
            return
        try:
            float(self.delta_m.get())
            _parse_hhmm_text(self.start_time.get(), field_name="Start time")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.run_btn.configure(state="disabled")
        self.progress.set(0.0)
        self.pb.set(0.0)
        self.status.set("Running…", "running")
        self._log("=== START ===")

        delta = float(self.delta_m.get())

        def worker():
            try:
                target_date, ordered = _s1_collect_actual_files(
                    Path(folder), self.date.get().strip(), self.start_time.get().strip(),
                )
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
                # MPL reference products (raw) collected per time for validation
                mpl_cols: Dict[str, Dict[pd.Timestamp, np.ndarray]] = (
                    {c: {} for c in _MPL_PROFILE_COLS} if _HAS_MPL_PRODUCTS else {})
                mpl_range_master = None
                # RAW co/cross counts live on the range_raw grid (separate)
                mpl_raw_par: Dict[pd.Timestamp, np.ndarray] = {}
                mpl_raw_perp: Dict[pd.Timestamp, np.ndarray] = {}
                mpl_raw_range = None
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

                    # Pull all MPL reference products (cross NRB, depol, ext, mass, type)
                    mpl_aod = mpl_lr = np.nan
                    if _HAS_MPL_PRODUCTS:
                        try:
                            prod = read_mpl_products(f, 0, copol_n)
                            if mpl_range_master is None and np.isfinite(prod["range_m"]).any():
                                mpl_range_master = prod["range_m"]
                            for c in _MPL_PROFILE_COLS:
                                mpl_cols[c][ts] = np.asarray(prod[c], float)
                            mpl_aod = float(prod.get("aod", np.nan))
                            mpl_lr = float(prod.get("lidar_ratio", np.nan))
                        except Exception as e:
                            self._safe(self._log, f"   [WARN] MPL products: {e}")
                        try:
                            raw = read_mpl_raw(f, 0, copol_n)
                            if mpl_raw_range is None and np.isfinite(raw["range_m"]).any():
                                mpl_raw_range = np.asarray(raw["range_m"], float)
                            mpl_raw_par[ts] = np.asarray(raw["copol_raw"], float)
                            mpl_raw_perp[ts] = np.asarray(raw["crosspol_raw"], float)
                        except Exception as e:
                            self._safe(self._log, f"   [WARN] MPL raw: {e}")

                    pbl_km = _s1_read_pbl_km(f)
                    if np.isfinite(pbl_km):
                        pbl_m = float(pbl_km) * 1000.0
                        rmin = max(0.0, pbl_m - delta)
                        rmax = pbl_m + delta
                        status = "ok"
                        self._safe(
                            self._log,
                            f"[{i:02d}/{total}] {ts.strftime('%H:%M')} <- {src}  PBL={pbl_m:.1f} m",
                        )
                    else:
                        pbl_m = rmin = rmax = np.nan
                        status = "read_fail"
                        self._safe(
                            self._log,
                            f"[{i:02d}/{total}] {ts.strftime('%H:%M')} <- {src}  PBL=NaN",
                        )

                    row = {
                        "Time": ts,
                        "PBL from MPL (m)": pbl_m,
                        "rmin": rmin,
                        "rmax": rmax,
                        "Status": status,
                        "Source file": src,
                    }
                    if _HAS_MPL_PRODUCTS:
                        row["MPL_aod"] = mpl_aod
                        row["MPL_lidar_ratio"] = mpl_lr
                    rows.append(row)
                    self._safe(self.progress.set, 100.0 * i / total)
                    self._safe(self.pb.set, i / total)

                df = pd.DataFrame(rows)
                if range_m_master is None:
                    range_m_master = np.full((copol_n,), np.nan)
                copol_df = pd.DataFrame({"Range(m)": range_m_master})
                for ts in stamps:
                    copol_df[ts.strftime("%H:%M")] = copol_cols.get(ts, np.full((copol_n,), np.nan))

                # MPL reference product sheets (raw values, range × time)
                ref_r = (mpl_range_master if mpl_range_master is not None
                         else range_m_master)
                sheet_name_map = {
                    "copol_nrb":              "MPL_copol_nrb",
                    "crosspol_nrb":           "MPL_crosspol_nrb",
                    "depolarization_ratio":   "MPL_depol",
                    "extinction_coefficient": "MPL_extinction",
                    "mass_concentration":     "MPL_mass",
                    "particle_type":          "MPL_particle_type",
                }
                mpl_dfs: Dict[str, pd.DataFrame] = {}
                if _HAS_MPL_PRODUCTS and mpl_range_master is not None:
                    for col, sheet in sheet_name_map.items():
                        d = pd.DataFrame({"Range(m)": ref_r})
                        for ts in stamps:
                            d[ts.strftime("%H:%M")] = mpl_cols.get(col, {}).get(
                                ts, np.full(len(ref_r), np.nan))
                        mpl_dfs[sheet] = d

                # Normalised cross-pol NRB (each column ÷ its own max, 0-1) — the
                # symmetric partner of copol_nrb_norm so the workbook has both.
                crosspol_norm_df = None
                if _HAS_MPL_PRODUCTS and mpl_range_master is not None and "crosspol_nrb" in mpl_cols:
                    crosspol_norm_df = pd.DataFrame({"Range(m)": ref_r})
                    for ts in stamps:
                        col = np.asarray(mpl_cols.get("crosspol_nrb", {}).get(
                            ts, np.full(len(ref_r), np.nan)), float)
                        mx = np.nanmax(col) if np.isfinite(col).any() else np.nan
                        crosspol_norm_df[ts.strftime("%H:%M")] = (
                            col / mx if (np.isfinite(mx) and mx != 0)
                            else np.full(len(ref_r), np.nan))

                # RAW co/cross counts, resampled from the range_raw grid onto ref_r
                # so all MPL sheets share one grid (for Step 6 profile figures).
                raw_par_df = raw_perp_df = None
                if _HAS_MPL_PRODUCTS and mpl_raw_range is not None:
                    def _raw_df(store):
                        d = pd.DataFrame({"Range(m)": ref_r})
                        for ts in stamps:
                            col = store.get(ts)
                            d[ts.strftime("%H:%M")] = (
                                np.interp(ref_r, mpl_raw_range, np.asarray(col, float),
                                          left=np.nan, right=np.nan)
                                if col is not None else np.full(len(ref_r), np.nan))
                        return d
                    raw_par_df = _raw_df(mpl_raw_par)
                    raw_perp_df = _raw_df(mpl_raw_perp)

                outp = Path(out_path)
                with pd.ExcelWriter(outp, engine="openpyxl") as xw:
                    df.to_excel(xw, index=False, sheet_name="rmin-rmax")
                    copol_df.to_excel(xw, index=False, sheet_name="copol_nrb_norm")
                    if crosspol_norm_df is not None:
                        crosspol_norm_df.to_excel(xw, index=False, sheet_name="crosspol_nrb_norm")
                    if raw_par_df is not None:
                        raw_par_df.to_excel(xw, index=False, sheet_name="MPL_copol_raw")
                        raw_perp_df.to_excel(xw, index=False, sheet_name="MPL_crosspol_raw")
                    for sheet, d in mpl_dfs.items():
                        d.to_excel(xw, index=False, sheet_name=sheet)

                if mpl_dfs:
                    self._safe(self._log,
                               f"MPL reference sheets written: {', '.join(mpl_dfs.keys())}")
                self._safe(self._log, f"[OK] Saved: {outp.resolve()}")
                self._copol_df = copol_df
                self._crosspol_norm_df = crosspol_norm_df
                self._safe(self.refresh_rti)
                self._safe(self.status.set, "Done", "ok")
                self._safe(self.progress.set, 100.0)
                self._safe(self.pb.set, 1.0)
                self._safe(self.app_state.set_step1, str(outp.resolve()))
                self._safe(messagebox.showinfo, "Step 1 Complete", f"Saved:\n{outp}")
            except Exception as e:
                self._safe(self._log, f"[FAILED] {e}")
                self._safe(self.status.set, "Failed", "error")
                self._safe(messagebox.showerror, "Failed", str(e))
            finally:
                self._safe(self.run_btn.configure, state="normal")
                self._safe(self._log, "=== END ===")

        threading.Thread(target=worker, daemon=True).start()


# ═════════════════════════════════════════════════════════════════════════════
# Step 2 Page  –  NRB Daily Profile Builder
# ═════════════════════════════════════════════════════════════════════════════
PROCESSED_PLOT_MODES = ["Photon DT", "Analog Scaled", "Glue", "Glue Overlay", "NRB"]
RAW_PLOT_MODES = ["Analog Raw", "Photon Raw"]


class Step2Page(ctk.CTkFrame):
    def __init__(self, master, app_state: AppState):
        super().__init__(master, fg_color="transparent")
        self.app_state = app_state

        # ── State variables (mirror main.py Step2Frame) ─────────────────────
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
        self.sig_start_m = tk.DoubleVar(value=0.0)
        self.sig_end_m = tk.DoubleVar(value=15000.0)
        # min 2.0: must sit ABOVE the photon far-range floor (~1.1 MHz) or the
        # auto-blend descending crossing is never found and it silently falls back
        # to the manual blend_r1/r2. max 10.0 keeps dead-time correction at ~3%.
        # Kept identical to Step 3 so both steps give the same result by default.
        self.min_toggle_rate = tk.DoubleVar(value=2.0)
        self.max_toggle_rate = tk.DoubleVar(value=10.0)
        self.auto_toggle_selector = tk.BooleanVar(value=True)
        self.day_min_toggle_rate = tk.DoubleVar(value=75.0)
        self.day_max_toggle_rate = tk.DoubleVar(value=130.0)
        self.toggle_bg_switch_threshold_mhz = tk.DoubleVar(value=10.0)
        self.pretrigger_trim_bins = tk.IntVar(value=24)
        self.blend_r1_m = tk.DoubleVar(value=1200.0)
        self.blend_r2_m = tk.DoubleVar(value=1800.0)
        self.auto_blend = tk.BooleanVar(value=True)
        self.photon_only = tk.BooleanVar(value=False)
        # Per-profile glue choice from the pretrigger BG: daylight → photon only
        # (no valid analog/photon overlap window), night → glue. Overrides the
        # Skip-glue checkbox when ticked. Default ON: safe in every case (day →
        # photon only where no valid glue window exists, night → glue) and the
        # user's folders mix day+night.
        self.glue_auto_daynight = tk.BooleanVar(value=True)
        self.bin_shift_bins = tk.IntVar(value=0)
        self.energy_mj = tk.DoubleVar(value=25.0)
        # Signal-quality (SNR) gating — mask low-SNR bins + normalise within the
        # SNR-trusted range so daytime noise x R^2 isn't shown/normalised as signal.
        # Default ON: essential by day, ~no-op at night (trusted range extends high).
        self.snr_gate = tk.BooleanVar(value=True)
        self.snr_min = tk.DoubleVar(value=3.0)
        # Exclude ADC-saturated (overflow>0) analog bins from the glue fit.
        # Raw binary files carry the per-bin overflow flag; default OFF.
        self.exclude_saturated = tk.BooleanVar(value=False)
        # Overlap correction — ON by default (analytical NARIT geometry). O_min=0.99
        # keeps only the near field where overlap is essentially complete (R≳150 m);
        # below that O(R)<0.99 → NaN rather than dividing up amplified noise.
        self.overlap_mode = tk.StringVar(value="Analytical (from hardware)")
        self.overlap_file = tk.StringVar(value="")
        self.overlap_o_min = tk.DoubleVar(value=0.99)
        # Afterpulse correction
        self.afterpulse_mode = tk.StringVar(value="Disabled")
        self.afterpulse_file = tk.StringVar(value="")
        # Afterpulse .dat recorded WITHOUT a pretrigger block (laser firing,
        # telescope covered). Then the pretrigger BG does not exist → read with
        # pretrigger_bins=0 and take the BG from the far range instead.
        self.ap_no_pretrigger = tk.BooleanVar(value=False)
        self.strict = tk.BooleanVar(value=True)
        self.preset_name = tk.StringVar(value="Custom")
        self.bg_mode_help_var = tk.StringVar(value="")
        self.pretrigger_role_var = tk.StringVar(value="")

        # Plot controls
        self.data_source = tk.StringVar(value="Processed")
        self.plot_mode = tk.StringVar(value="NRB")
        self.xmin_var = tk.StringVar(); self.xmax_var = tk.StringVar()
        self.ymin_var = tk.StringVar(); self.ymax_var = tk.StringVar()
        self.chart_title_var = tk.StringVar()
        self.x_title_var = tk.StringVar()
        self.y_title_var = tk.StringVar()

        self.progress = tk.DoubleVar(value=0.0)
        self._times_map: Dict[str, Path] = {}
        self._latest_profile_df = None

        # Bail early if NRB backend missing
        if not _HAS_NRB:
            self._build_missing_backend_view()
            return

        self._build_ui()
        self._update_bg_mode_ui()
        self._update_plot_mode_options()
        self._update_overlap_ui()
        self._update_afterpulse_ui()

        # Refresh time list when folder/date/pattern changes
        self.date.trace_add("write", lambda *_: self._refresh_time_list())
        self.start_time.trace_add("write", lambda *_: self._refresh_time_list())
        self.pattern.trace_add("write", lambda *_: self._refresh_time_list())
        self.bg_mode.trace_add("write", lambda *_: self._update_bg_mode_ui())

    # ── Backend missing fallback ───────────────────────────────────────────
    def _build_missing_backend_view(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        card = Card(self, title="Backend missing", icon="⚠️")
        card.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(
            card.body,
            text="nrb_engine.py was not found alongside this script.\n"
                 "Step 2 needs that module to run.",
            font=theme.F_BODY, text_color=theme.TEXT_SECONDARY, justify="left",
        ).pack(pady=20)

    # ── Layout ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = PageHeader(
            self,
            title="Step 2 · NRB Daily Profile Builder",
            subtitle="Actual-files mode · latest NRB engine · auto day/night toggle selector · drag the divider to resize panels",
            badge=("v11.3", "navy"),
        )
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))

        # Resizable split: drag the sash to widen either side.
        self.paned = tk.PanedWindow(
            self,
            orient="horizontal",
            bg=theme.APP_BG,
            sashrelief="flat",
            sashwidth=8,
            sashpad=0,
            bd=0,
            showhandle=False,
        )
        self.paned.grid(row=1, column=0, sticky="nsew")

        # Left pane wrapper (plain tk.Frame so PanedWindow can manage it directly)
        left_pane = tk.Frame(self.paned, bg=theme.APP_BG, bd=0, highlightthickness=0)
        left_pane.grid_columnconfigure(0, weight=1)
        left_pane.grid_rowconfigure(0, weight=1)
        self.paned.add(left_pane, minsize=380, width=640, stretch="always")

        # Scrollable content inside the left pane
        left = ctk.CTkScrollableFrame(left_pane, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_columnconfigure(0, weight=1)
        self._build_io_card(left).grid(row=0, column=0, sticky="ew", pady=(0, 12), padx=(0, 10))
        self._build_params_card(left).grid(row=1, column=0, sticky="ew", pady=(0, 12), padx=(0, 10))
        self._build_run_card(left).grid(row=2, column=0, sticky="ew", pady=(0, 12), padx=(0, 10))
        self._build_console_card(left).grid(row=3, column=0, sticky="nsew", pady=(0, 0), padx=(0, 10))

        # Right pane wrapper
        right_pane = tk.Frame(self.paned, bg=theme.APP_BG, bd=0, highlightthickness=0)
        right_pane.grid_columnconfigure(0, weight=1)
        right_pane.grid_rowconfigure(0, weight=1)
        self.paned.add(right_pane, minsize=380, stretch="always")

        # Plot inside the right pane
        right = ctk.CTkFrame(right_pane, fg_color="transparent")
        right.grid(row=0, column=0, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)
        self._build_plot_card(right).grid(row=0, column=0, sticky="nsew")

    # ── Card: I/O ──────────────────────────────────────────────────────────
    def _build_io_card(self, parent) -> "Card":
        card = Card(parent, title="Input / Output", icon="📁")
        body = card.body
        body.grid_columnconfigure(0, weight=1)

        # Folder
        row1 = ctk.CTkFrame(body, fg_color="transparent")
        row1.grid(row=0, column=0, sticky="ew")
        row1.grid_columnconfigure(0, weight=1)
        ctk.CTkEntry(row1, textvariable=self.folder, **theme.input_style()).grid(
            row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(row1, text="Browse…", command=self.pick_folder, width=100,
                      **theme.secondary_button_style()).grid(row=0, column=1)

        # Date / Start / Pattern (3 equal columns)
        row2 = ctk.CTkFrame(body, fg_color="transparent")
        row2.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        for i in range(3):
            row2.grid_columnconfigure(i, weight=1)
        FieldRow(row2, "Date", self.date).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        FieldRow(row2, "Start time (HH:MM)", self.start_time).grid(row=0, column=1, sticky="ew", padx=3)
        FieldRow(row2, "Pattern", self.pattern).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        # Strict mode on its own row (doesn't compete with field labels)
        row2b = ctk.CTkFrame(body, fg_color="transparent")
        row2b.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ctk.CTkCheckBox(
            row2b, text="Strict mode (stop on first error)", variable=self.strict,
            font=theme.F_SMALL, text_color=theme.TEXT_PRIMARY,
            fg_color=theme.ORANGE, hover_color=theme.ORANGE_HOVER,
            border_color=theme.BORDER, checkmark_color="#fff",
            corner_radius=4, border_width=2,
        ).pack(side="left")

        # Output
        row3 = ctk.CTkFrame(body, fg_color="transparent")
        row3.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        row3.grid_columnconfigure(0, weight=1)
        ctk.CTkEntry(row3, textvariable=self.out_path, **theme.input_style()).grid(
            row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(row3, text="Save As…", command=self.pick_output, width=100,
                      **theme.secondary_button_style()).grid(row=0, column=1)

        # Hint
        ctk.CTkLabel(
            body,
            text="Actual-files mode · no fixed 48 slots · latest engine: shift fixed = 0",
            font=(theme.FONT_FAMILY, 10), text_color=theme.TEXT_MUTED, anchor="w",
            wraplength=520, justify="left",
        ).grid(row=4, column=0, sticky="w", pady=(8, 0))

        return card

    # ── Card: Parameters (with subsections) ────────────────────────────────
    def _build_params_card(self, parent) -> "Card":
        card = Card(parent, title="Parameters", icon="⚙️")
        body = card.body
        body.grid_columnconfigure(0, weight=1)

        # Presets row (responsive grid — buttons share width equally)
        presets = ctk.CTkFrame(body, fg_color="transparent")
        presets.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        for i in range(3):
            presets.grid_columnconfigure(i, weight=1)
        preset_labels = [
            ("Pre-trigger",   "Licel pretrigger 1024"),
            ("No-pretrigger", "Legacy no pretrigger"),
            ("Auto-detect",   "Auto detect"),
        ]
        for i, (short, full) in enumerate(preset_labels):
            cmd = (self.auto_detect_bins if full == "Auto detect"
                   else (lambda n=full: self.apply_preset(n)))
            ctk.CTkButton(
                presets, text=short, command=cmd,
                fg_color=theme.CREAM, hover_color=theme.ORANGE_SOFT,
                text_color=theme.TEXT_PRIMARY,
                corner_radius=theme.RADIUS_BUTTON,
                height=32, font=theme.F_SMALL,
            ).grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 4, 0))

        # Active preset name on its OWN row (below presets, not overlapping)
        preset_status = ctk.CTkFrame(body, fg_color="transparent")
        preset_status.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        preset_status.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            preset_status, text="Active preset:",
            font=(theme.FONT_FAMILY, 10), text_color=theme.TEXT_MUTED, anchor="e",
        ).grid(row=0, column=0, sticky="e", padx=(0, 6))
        ctk.CTkLabel(
            preset_status, textvariable=self.preset_name,
            font=theme.F_SMALL, text_color=theme.ORANGE, anchor="e",
        ).grid(row=0, column=1, sticky="e")

        # Subsection: Raw layout
        self._subsection_label(body, "Raw layout · Dead time").grid(row=2, column=0, sticky="ew")
        row1 = ctk.CTkFrame(body, fg_color="transparent")
        row1.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        for i in range(4):
            row1.grid_columnconfigure(i, weight=1)
        FieldRow(row1, "bin_spacing_m", self.bin_spacing_m).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        FieldRow(row1, "dead_time_ns", self.dead_time_ns).grid(row=0, column=1, sticky="ew", padx=3)
        # BG mode dropdown
        bg_wrap = ctk.CTkFrame(row1, fg_color="transparent")
        bg_wrap.grid(row=0, column=2, columnspan=2, sticky="ew", padx=(6, 0))
        bg_wrap.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(bg_wrap, text="BG mode", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY, anchor="w").grid(row=0, column=0, sticky="w")
        self.bg_mode_box = ctk.CTkOptionMenu(
            bg_wrap, variable=self.bg_mode, values=["pretrigger", "fixed", "far_range", "rayleigh_fit"],
            fg_color=theme.CARD_BG, button_color=theme.CARD_BG, button_hover_color=theme.CREAM_SOFT,
            text_color=theme.TEXT_PRIMARY, dropdown_fg_color=theme.CARD_BG,
            dropdown_text_color=theme.TEXT_PRIMARY, dropdown_hover_color=theme.CREAM,
            corner_radius=theme.RADIUS_INPUT, height=36, font=theme.F_BODY,
        )
        self.bg_mode_box.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        # Hack: add border via outer frame because CTkOptionMenu doesn't expose border
        bg_wrap.configure(border_width=0)

        row2 = ctk.CTkFrame(body, fg_color="transparent")
        row2.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        for i in range(4):
            row2.grid_columnconfigure(i, weight=1)
        FieldRow(row2, "layout pretrigger bins", self.pretrigger_bins).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        # First signal entry with sync button
        fs_wrap = ctk.CTkFrame(row2, fg_color="transparent")
        fs_wrap.grid(row=0, column=1, columnspan=2, sticky="ew", padx=3)
        fs_wrap.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(fs_wrap, text="first_signal_bin", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY, anchor="w").grid(row=0, column=0, columnspan=2, sticky="w")
        self.first_signal_entry = ctk.CTkEntry(fs_wrap, textvariable=self.first_signal_bin, **theme.input_style())
        self.first_signal_entry.grid(row=1, column=0, sticky="ew", padx=(0, 6))
        self.sync_first_signal_btn = ctk.CTkButton(
            fs_wrap, text="Sync = pre+1", command=self.sync_first_signal_bin,
            fg_color=theme.NAVY_SOFT, hover_color=theme.CREAM,
            text_color=theme.TEXT_PRIMARY,
            corner_radius=theme.RADIUS_BUTTON, width=110, height=36, font=theme.F_SMALL,
        )
        self.sync_first_signal_btn.grid(row=1, column=1)
        FieldRow(row2, "first_signal_range_m", self.first_signal_range_m).grid(row=0, column=3, sticky="ew", padx=(6, 0))

        # Hint for pretrigger role
        self.pretrigger_role_label = ctk.CTkLabel(
            body, textvariable=self.pretrigger_role_var, font=theme.F_TINY,
            text_color=theme.TEXT_MUTED, anchor="w",
        )
        self.pretrigger_role_label.grid(row=5, column=0, sticky="w", pady=(0, 6))

        # Subsection: Background
        self._subsection_label(body, "Background window (fixed mode only)").grid(row=6, column=0, sticky="ew")
        row3 = ctk.CTkFrame(body, fg_color="transparent")
        row3.grid(row=7, column=0, sticky="ew", pady=(0, 4))
        for i in range(2):
            row3.grid_columnconfigure(i, weight=1)
        bg_start_row = FieldRow(row3, "bg_start_m", self.bg_start_m)
        bg_start_row.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.bg_start_entry = bg_start_row.entry
        bg_end_row = FieldRow(row3, "bg_end_m", self.bg_end_m)
        bg_end_row.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.bg_end_entry = bg_end_row.entry

        # Help text under BG params
        self.bg_help_label = ctk.CTkLabel(
            body, textvariable=self.bg_mode_help_var, font=theme.F_TINY,
            text_color=theme.ORANGE, anchor="w", wraplength=520, justify="left",
        )
        self.bg_help_label.grid(row=8, column=0, sticky="w", pady=(0, 8))

        # Subsection: Toggle rates
        self._subsection_label(body, "Toggle window & day/night").grid(row=9, column=0, sticky="ew")
        row4 = ctk.CTkFrame(body, fg_color="transparent")
        row4.grid(row=10, column=0, sticky="ew", pady=(0, 4))
        for i in range(4):
            row4.grid_columnconfigure(i, weight=1)
        FieldRow(row4, "min_toggle_rate", self.min_toggle_rate).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        FieldRow(row4, "max_toggle_rate", self.max_toggle_rate).grid(row=0, column=1, sticky="ew", padx=3)
        FieldRow(row4, "day_min_toggle", self.day_min_toggle_rate).grid(row=0, column=2, sticky="ew", padx=3)
        FieldRow(row4, "day_max_toggle", self.day_max_toggle_rate).grid(row=0, column=3, sticky="ew", padx=(6, 0))

        row5 = ctk.CTkFrame(body, fg_color="transparent")
        row5.grid(row=11, column=0, sticky="ew", pady=(4, 8))
        for i in range(3):
            row5.grid_columnconfigure(i, weight=1)
        FieldRow(row5, "BG switch threshold (MHz)", self.toggle_bg_switch_threshold_mhz).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        FieldRow(row5, "pretrigger trim bins", self.pretrigger_trim_bins).grid(row=0, column=1, sticky="ew", padx=3)
        ctk.CTkCheckBox(
            row5, text="Auto day/night toggle", variable=self.auto_toggle_selector,
            font=theme.F_SMALL, text_color=theme.TEXT_PRIMARY,
            fg_color=theme.ORANGE, hover_color=theme.ORANGE_HOVER,
            border_color=theme.BORDER, checkmark_color="#fff",
            corner_radius=4, border_width=2,
        ).grid(row=0, column=2, sticky="sw", padx=(6, 0), pady=(18, 0))

        # Subsection: Glue
        self._subsection_label(body, "Blend zone & glue").grid(row=12, column=0, sticky="ew")
        row6 = ctk.CTkFrame(body, fg_color="transparent")
        row6.grid(row=13, column=0, sticky="ew", pady=(0, 4))
        for i in range(4):
            row6.grid_columnconfigure(i, weight=1)
        FieldRow(row6, "blend_r1_m", self.blend_r1_m).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        FieldRow(row6, "blend_r2_m", self.blend_r2_m).grid(row=0, column=1, sticky="ew", padx=3)
        ctk.CTkCheckBox(
            row6, text="Auto blend r1/r2", variable=self.auto_blend,
            font=theme.F_SMALL, text_color=theme.TEXT_PRIMARY,
            fg_color=theme.ORANGE, hover_color=theme.ORANGE_HOVER,
            border_color=theme.BORDER, checkmark_color="#fff",
            corner_radius=4, border_width=2,
        ).grid(row=0, column=2, sticky="sw", padx=3, pady=(18, 0))
        ctk.CTkCheckBox(
            row6, text="Skip glue (photon only)", variable=self.photon_only,
            font=theme.F_SMALL, text_color=theme.TEXT_PRIMARY,
            fg_color=theme.ORANGE, hover_color=theme.ORANGE_HOVER,
            border_color=theme.BORDER, checkmark_color="#fff",
            corner_radius=4, border_width=2,
        ).grid(row=0, column=3, sticky="sw", padx=(6, 0), pady=(18, 0))
        ctk.CTkCheckBox(
            row6, text="Auto day/night glue: day → photon only, night → glue (overrides Skip glue)",
            variable=self.glue_auto_daynight,
            font=theme.F_SMALL, text_color=theme.TEXT_PRIMARY,
            fg_color=theme.ORANGE, hover_color=theme.ORANGE_HOVER,
            border_color=theme.BORDER, checkmark_color="#fff",
            corner_radius=4, border_width=2,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))

        # Subsection: Afterpulse correction
        self._subsection_label(body, "Afterpulse correction").grid(row=14, column=0, sticky="ew")
        ap_row = ctk.CTkFrame(body, fg_color="transparent")
        ap_row.grid(row=15, column=0, sticky="ew", pady=(0, 4))
        ap_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(ap_row, text="Mode:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 8))
        self.afterpulse_mode_box = ctk.CTkOptionMenu(
            ap_row, variable=self.afterpulse_mode,
            values=["Disabled", "Load from file"],
            command=lambda _v: self._update_afterpulse_ui(),
            fg_color=theme.CARD_BG, button_color=theme.CARD_BG, button_hover_color=theme.CREAM_SOFT,
            text_color=theme.TEXT_PRIMARY, dropdown_fg_color=theme.CARD_BG,
            dropdown_text_color=theme.TEXT_PRIMARY, dropdown_hover_color=theme.CREAM,
            corner_radius=theme.RADIUS_INPUT, height=32, font=theme.F_BODY, width=240,
        )
        self.afterpulse_mode_box.grid(row=0, column=1, sticky="w")

        ap_row2 = ctk.CTkFrame(body, fg_color="transparent")
        ap_row2.grid(row=16, column=0, sticky="ew", pady=(0, 4))
        ap_row2.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(ap_row2, text="File:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 8))
        self.afterpulse_file_entry = ctk.CTkEntry(
            ap_row2, textvariable=self.afterpulse_file, **theme.input_style())
        self.afterpulse_file_entry.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self.afterpulse_file_btn = ctk.CTkButton(
            ap_row2, text="Browse…", command=self._pick_afterpulse_file, width=90,
            **theme.secondary_button_style(height=34, font=theme.F_SMALL),
        )
        self.afterpulse_file_btn.grid(row=0, column=2)
        ctk.CTkCheckBox(
            ap_row2, text="Calibration .dat has NO pretrigger (BG from far range)",
            variable=self.ap_no_pretrigger,
            font=theme.F_SMALL, text_color=theme.TEXT_PRIMARY,
            fg_color=theme.ORANGE, hover_color=theme.ORANGE_HOVER,
            border_color=theme.BORDER, checkmark_color="#fff",
            corner_radius=4, border_width=2,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))

        self.afterpulse_help_label = ctk.CTkLabel(
            body,
            text="Detector ghost-signal correction. Accept .dat (auto-processed) "
                 "or CSV/Excel with (range_m, afterpulse_MHz). "
                 "Applied right after BG subtraction.",
            font=theme.F_TINY, text_color=theme.TEXT_MUTED, anchor="w",
            wraplength=520, justify="left",
        )
        self.afterpulse_help_label.grid(row=17, column=0, sticky="w", pady=(0, 8))

        # Subsection: Overlap correction
        self._subsection_label(body, "Overlap correction").grid(row=18, column=0, sticky="ew")
        ov_row = ctk.CTkFrame(body, fg_color="transparent")
        ov_row.grid(row=19, column=0, sticky="ew", pady=(0, 4))
        ov_row.grid_columnconfigure(1, weight=1)

        # Mode dropdown
        ctk.CTkLabel(ov_row, text="Mode:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 8))
        self.overlap_mode_box = ctk.CTkOptionMenu(
            ov_row, variable=self.overlap_mode,
            values=["Disabled", "Analytical (from hardware)", "Load from file"],
            command=lambda _v: self._update_overlap_ui(),
            fg_color=theme.CARD_BG, button_color=theme.CARD_BG, button_hover_color=theme.CREAM_SOFT,
            text_color=theme.TEXT_PRIMARY, dropdown_fg_color=theme.CARD_BG,
            dropdown_text_color=theme.TEXT_PRIMARY, dropdown_hover_color=theme.CREAM,
            corner_radius=theme.RADIUS_INPUT, height=32, font=theme.F_BODY, width=240,
        )
        self.overlap_mode_box.grid(row=0, column=1, sticky="w")

        # O_min row
        ov_row2 = ctk.CTkFrame(body, fg_color="transparent")
        ov_row2.grid(row=20, column=0, sticky="ew", pady=(0, 4))
        ov_row2.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(ov_row2, text="O_min:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 8))
        self.overlap_omin_entry = ctk.CTkEntry(
            ov_row2, textvariable=self.overlap_o_min, width=80, **theme.input_style())
        self.overlap_omin_entry.grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(ov_row2,
                     text="(bins where O(R) < O_min become NaN — typical 0.1 to 0.5)",
                     font=theme.F_TINY, text_color=theme.TEXT_MUTED, anchor="w",
                     wraplength=380).grid(row=0, column=2, sticky="w", padx=(8, 0))

        # File picker row (only relevant for "Load from file")
        ov_row3 = ctk.CTkFrame(body, fg_color="transparent")
        ov_row3.grid(row=21, column=0, sticky="ew", pady=(0, 4))
        ov_row3.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(ov_row3, text="File:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 8))
        self.overlap_file_entry = ctk.CTkEntry(
            ov_row3, textvariable=self.overlap_file, **theme.input_style())
        self.overlap_file_entry.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self.overlap_file_btn = ctk.CTkButton(
            ov_row3, text="Browse…", command=self._pick_overlap_file, width=90,
            **theme.secondary_button_style(height=34, font=theme.F_SMALL),
        )
        self.overlap_file_btn.grid(row=0, column=2)

        # Help text under overlap section
        self.overlap_help_label = ctk.CTkLabel(
            body,
            text="Inserted between BG subtraction and R² range correction "
                 "(per Weitkamp 2005 / Kumar 2013).",
            font=theme.F_TINY, text_color=theme.TEXT_MUTED, anchor="w",
            wraplength=520, justify="left",
        )
        self.overlap_help_label.grid(row=22, column=0, sticky="w", pady=(0, 8))

        # Subsection: Signal & energy
        self._subsection_label(body, "Signal window & energy").grid(row=23, column=0, sticky="ew")
        row7 = ctk.CTkFrame(body, fg_color="transparent")
        row7.grid(row=24, column=0, sticky="ew", pady=(0, 0))
        for i in range(3):
            row7.grid_columnconfigure(i, weight=1)
        FieldRow(row7, "sig_start_m", self.sig_start_m).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        FieldRow(row7, "sig_end_m", self.sig_end_m).grid(row=0, column=1, sticky="ew", padx=3)
        FieldRow(row7, "energy_mj", self.energy_mj).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        # Subsection: Signal quality (SNR gating)
        self._subsection_label(body, "Signal quality (SNR)").grid(row=25, column=0, sticky="ew")
        sq_row = ctk.CTkFrame(body, fg_color="transparent")
        sq_row.grid(row=26, column=0, sticky="ew", pady=(0, 4))
        for i in range(2):
            sq_row.grid_columnconfigure(i, weight=1)
        ctk.CTkCheckBox(
            sq_row, text="Normalise NRB in the lower troposphere (boundary layer → 1.0, full curve)",
            variable=self.snr_gate,
            font=theme.F_SMALL, text_color=theme.TEXT_PRIMARY,
            fg_color=theme.ORANGE, hover_color=theme.ORANGE_HOVER,
            border_color=theme.BORDER, checkmark_color="#fff",
            corner_radius=4, border_width=2,
        ).grid(row=0, column=0, sticky="w", pady=(6, 0))
        ctk.CTkCheckBox(
            sq_row, text="Exclude ADC-saturated bins from glue fit (raw binary overflow flag)",
            variable=self.exclude_saturated,
            font=theme.F_SMALL, text_color=theme.TEXT_PRIMARY,
            fg_color=theme.ORANGE, hover_color=theme.ORANGE_HOVER,
            border_color=theme.BORDER, checkmark_color="#fff",
            corner_radius=4, border_width=2,
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        FieldRow(sq_row, "min SNR", self.snr_min).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        ctk.CTkLabel(
            body,
            text="÷max is taken within the lower troposphere (≤6 km / SNR-trusted "
                 "range) so the boundary layer peaks near 1.0 instead of a far-range "
                 "noise×R² spike. The FULL curve is kept (not cut) for a full-range "
                 "Mini-MPL overlay. Default OFF = global-max normalise.",
            font=theme.F_TINY, text_color=theme.TEXT_MUTED, anchor="w",
            wraplength=520, justify="left",
        ).grid(row=27, column=0, sticky="w", pady=(0, 8))

        return card

    @staticmethod
    def _subsection_label(parent, text: str) -> ctk.CTkLabel:
        return ctk.CTkLabel(
            parent, text=text.upper(),
            font=(theme.FONT_FAMILY, 10, "bold"),
            text_color=theme.TEXT_MUTED, anchor="w",
        )

    # ── Card: Run + Console (combined for compactness) ─────────────────────
    def _build_run_card(self, parent) -> "Card":
        card = Card(parent)
        body = card.body
        body.grid_columnconfigure(1, weight=1)

        self.run_btn = ctk.CTkButton(
            body, text="▶  Run Build NRB Profile", command=self.run,
            **theme.primary_button_style(width=240),
        )
        self.run_btn.grid(row=0, column=0, sticky="w")

        self.pb = ctk.CTkProgressBar(
            body, variable=self.progress, progress_color=theme.ORANGE,
            fg_color=theme.LIGHT_GRAY, height=8, corner_radius=4,
        )
        self.pb.set(0)
        self.pb.grid(row=0, column=1, sticky="ew", padx=(16, 0))

        self.status = StatusBar(body)
        self.status.grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))

        return card

    def _build_console_card(self, parent) -> "ConsoleLog":
        self.console = ConsoleLog(parent, title="Console Log")
        return self.console

    # ── Card: Plot ─────────────────────────────────────────────────────────
    def _build_plot_card(self, parent) -> "Card":
        card = Card(parent, title="Profile Plot", icon="📈")
        body = card.body
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(4, weight=1)

        # Controls row 1: source + mode + actions
        ctrl1 = ctk.CTkFrame(body, fg_color="transparent")
        ctrl1.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ctk.CTkLabel(ctrl1, text="Source:", font=theme.F_SMALL, text_color=theme.TEXT_SECONDARY).pack(side="left")
        self.source_box = ctk.CTkOptionMenu(
            ctrl1, variable=self.data_source, values=["Processed", "Raw .dat"],
            command=lambda _v: self._update_plot_mode_options(),
            fg_color=theme.CARD_BG, button_color=theme.CARD_BG, button_hover_color=theme.CREAM_SOFT,
            text_color=theme.TEXT_PRIMARY, dropdown_fg_color=theme.CARD_BG,
            dropdown_text_color=theme.TEXT_PRIMARY, dropdown_hover_color=theme.CREAM,
            corner_radius=theme.RADIUS_INPUT, height=32, font=theme.F_BODY, width=130,
        )
        self.source_box.pack(side="left", padx=(6, 12))
        ctk.CTkLabel(ctrl1, text="Plot:", font=theme.F_SMALL, text_color=theme.TEXT_SECONDARY).pack(side="left")
        self.plot_mode_box = ctk.CTkOptionMenu(
            ctrl1, variable=self.plot_mode, values=PROCESSED_PLOT_MODES,
            fg_color=theme.CARD_BG, button_color=theme.CARD_BG, button_hover_color=theme.CREAM_SOFT,
            text_color=theme.TEXT_PRIMARY, dropdown_fg_color=theme.CARD_BG,
            dropdown_text_color=theme.TEXT_PRIMARY, dropdown_hover_color=theme.CREAM,
            corner_radius=theme.RADIUS_INPUT, height=32, font=theme.F_BODY, width=140,
        )
        self.plot_mode_box.pack(side="left", padx=(6, 12))

        ctk.CTkButton(ctrl1, text="Refresh", command=self.refresh_plot,
                      **theme.primary_button_style(width=90, height=32)).pack(side="left", padx=(0, 6))
        ctk.CTkButton(ctrl1, text="Auto Scale", command=self.auto_scale,
                      **theme.secondary_button_style(width=90, height=32)).pack(side="left", padx=(0, 6))
        ctk.CTkButton(ctrl1, text="PNG", command=self.save_png,
                      **theme.ghost_button_style(width=60, height=32)).pack(side="left", padx=(8, 0))
        ctk.CTkButton(ctrl1, text="CSV", command=self.save_csv,
                      **theme.ghost_button_style(width=60, height=32)).pack(side="left")

        # Controls row 2: axis limits
        ctrl2 = ctk.CTkFrame(body, fg_color="transparent")
        ctrl2.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        for lbl, var in [("x min", self.xmin_var), ("x max", self.xmax_var),
                         ("y min", self.ymin_var), ("y max", self.ymax_var)]:
            ctk.CTkLabel(ctrl2, text=lbl, font=theme.F_SMALL, text_color=theme.TEXT_SECONDARY).pack(side="left")
            ctk.CTkEntry(ctrl2, textvariable=var, width=80,
                         **theme.input_style()).pack(side="left", padx=(4, 8))
        ctk.CTkButton(ctrl2, text="Apply Axis", command=self.apply_axes,
                      **theme.secondary_button_style(width=100, height=32)).pack(side="left")

        # Controls row 3: titles
        ctrl3 = ctk.CTkFrame(body, fg_color="transparent")
        ctrl3.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        for lbl, var, w in [("Chart title", self.chart_title_var, 200),
                            ("X title", self.x_title_var, 140),
                            ("Y title", self.y_title_var, 140)]:
            ctk.CTkLabel(ctrl3, text=lbl, font=theme.F_SMALL, text_color=theme.TEXT_SECONDARY).pack(side="left")
            ctk.CTkEntry(ctrl3, textvariable=var, width=w,
                         **theme.input_style()).pack(side="left", padx=(4, 10))

        # Times list
        sel_row = ctk.CTkFrame(body, fg_color="transparent")
        sel_row.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        sel_row.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(sel_row, text="Times (multi-select)", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY, anchor="w").grid(row=0, column=0, sticky="w")
        list_wrap = ctk.CTkFrame(sel_row, fg_color=theme.CREAM_SOFT,
                                 corner_radius=theme.RADIUS_INPUT, border_width=1,
                                 border_color=theme.BORDER, height=120)
        list_wrap.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        list_wrap.grid_columnconfigure(0, weight=1)
        list_wrap.grid_rowconfigure(0, weight=1)
        list_wrap.grid_propagate(False)

        self.time_list = tk.Listbox(
            list_wrap, selectmode="extended", exportselection=False, height=6,
            bg=theme.CREAM_SOFT, fg=theme.TEXT_PRIMARY,
            selectbackground=theme.ORANGE_SOFT, selectforeground=theme.ORANGE,
            relief="flat", borderwidth=0, highlightthickness=0, font=(theme.FONT_FAMILY, 11),
        )
        self.time_list.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        btns = ctk.CTkFrame(list_wrap, fg_color="transparent")
        btns.grid(row=0, column=1, sticky="ns", padx=(0, 8), pady=8)
        ctk.CTkButton(btns, text="Select All",
                      command=lambda: self.time_list.selection_set(0, "end"),
                      **theme.ghost_button_style(width=90, height=28)).pack(pady=(0, 4))
        ctk.CTkButton(btns, text="Clear",
                      command=lambda: self.time_list.selection_clear(0, "end"),
                      **theme.ghost_button_style(width=90, height=28)).pack()

        # Matplotlib canvas
        canvas_wrap = ctk.CTkFrame(body, fg_color=theme.CARD_BG, corner_radius=theme.RADIUS_INPUT,
                                   border_width=1, border_color=theme.BORDER)
        canvas_wrap.grid(row=4, column=0, sticky="nsew", pady=(6, 0))
        canvas_wrap.grid_columnconfigure(0, weight=1)
        canvas_wrap.grid_rowconfigure(0, weight=1)

        self.fig2 = plt.Figure(figsize=(7, 5.0), dpi=100, facecolor=theme.CARD_BG)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_title("No data")
        self.ax2.grid(True, alpha=0.3)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=canvas_wrap)
        self.canvas2.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        return card

    # ── Logic methods (mirror Step2Frame) ──────────────────────────────────
    def _log(self, msg): self.console.log(msg)

    def _safe(self, fn, *a, **kw):
        self.after(0, lambda: fn(*a, **kw))

    def _safe_log(self, msg): self._safe(self._log, msg)

    def sync_first_signal_bin(self):
        try:
            self.first_signal_bin.set(int(self.pretrigger_bins.get()) + 1)
            self.preset_name.set("Custom")
        except Exception:
            pass

    def _update_overlap_ui(self):
        """Enable/disable overlap inputs based on selected mode."""
        if not hasattr(self, "overlap_file_entry"):
            return
        mode = self.overlap_mode.get()
        if mode == "Disabled":
            self.overlap_file_entry.configure(state="disabled")
            self.overlap_file_btn.configure(state="disabled")
            self.overlap_omin_entry.configure(state="disabled")
            self.overlap_help_label.configure(
                text="Overlap correction disabled. NRB pipeline runs BG → R² (current default).",
                text_color=theme.TEXT_MUTED,
            )
        elif mode == "Analytical (from hardware)":
            self.overlap_file_entry.configure(state="disabled")
            self.overlap_file_btn.configure(state="disabled")
            self.overlap_omin_entry.configure(state="normal")
            self.overlap_help_label.configure(
                text="O(R) computed from telescope/laser/detector spec (Rice CDF model). "
                     "Inserted between BG sub and R² range correction.",
                text_color=theme.TEXT_SECONDARY,
            )
        else:  # "Load from file"
            self.overlap_file_entry.configure(state="normal")
            self.overlap_file_btn.configure(state="normal")
            self.overlap_omin_entry.configure(state="normal")
            self.overlap_help_label.configure(
                text="Load O(R) from CSV/Excel (2 columns: range_m, O). "
                     "Will be interpolated onto the NRB range grid.",
                text_color=theme.TEXT_SECONDARY,
            )

    def _pick_overlap_file(self):
        p = filedialog.askopenfilename(
            title="Overlap function O(R)",
            filetypes=[("Tabular", "*.csv *.xlsx *.xls"), ("CSV", "*.csv"), ("Excel", "*.xlsx *.xls")],
        )
        if p:
            self.overlap_file.set(p)

    def _update_afterpulse_ui(self):
        if not hasattr(self, "afterpulse_file_entry"):
            return
        mode = self.afterpulse_mode.get()
        if mode == "Disabled":
            self.afterpulse_file_entry.configure(state="disabled")
            self.afterpulse_file_btn.configure(state="disabled")
            self.afterpulse_help_label.configure(
                text="Afterpulse correction disabled. Pipeline skips the A(R) subtraction step.",
                text_color=theme.TEXT_MUTED,
            )
        else:  # Load from file
            self.afterpulse_file_entry.configure(state="normal")
            self.afterpulse_file_btn.configure(state="normal")
            self.afterpulse_help_label.configure(
                text="Detector ghost-signal correction. Accept .dat (auto-processed) "
                     "or CSV/Excel with (range_m, afterpulse_MHz). "
                     "Applied right after BG subtraction.",
                text_color=theme.TEXT_SECONDARY,
            )

    def _pick_afterpulse_file(self):
        p = filedialog.askopenfilename(
            title="Afterpulse calibration file",
            filetypes=[
                ("All supported", "*.dat *.csv *.xlsx *.xls"),
                ("TR40 .dat", "*.dat"),
                ("CSV", "*.csv"),
                ("Excel", "*.xlsx *.xls"),
            ],
        )
        if p:
            self.afterpulse_file.set(p)

    def _update_bg_mode_ui(self):
        if not hasattr(self, "bg_start_entry"):
            return
        mode = self.bg_mode.get().strip().lower()
        fixed_state = "normal" if mode == "fixed" else "disabled"
        try:
            self.bg_start_entry.configure(state=fixed_state)
            self.bg_end_entry.configure(state=fixed_state)
        except Exception:
            pass
        if mode == "pretrigger":
            self.bg_mode_help_var.set("BG source: mean of pretrigger bins after dead-time correction.")
            self.pretrigger_role_var.set("used for BG + layout/cut")
        elif mode == "far_range":
            self.bg_mode_help_var.set("BG source: auto far-range window 0.88*rmax to 0.98*rmax (meters). bg_start_m/bg_end_m ignored.")
            self.pretrigger_role_var.set("used for layout/cut only (set 0 if file has no pretrigger)")
        elif mode == "fixed":
            self.bg_mode_help_var.set("BG source: user-defined range bg_start_m to bg_end_m (meters).")
            self.pretrigger_role_var.set("used for layout/cut only (set 0 if file has no pretrigger)")
        elif mode == "rayleigh_fit":
            self.bg_mode_help_var.set("BG source: intercept of a signal-vs-molecular fit in an aerosol-free "
                                      "window (4-6 km); 3-sigma gate avoids over-subtraction. Best for daytime.")
            self.pretrigger_role_var.set("used for layout/cut only (set 0 if file has no pretrigger)")
        else:
            self.bg_mode_help_var.set("")
            self.pretrigger_role_var.set("")

    def _update_plot_mode_options(self):
        if not hasattr(self, "plot_mode_box"):
            return
        vals = RAW_PLOT_MODES if self.data_source.get() == "Raw .dat" else PROCESSED_PLOT_MODES
        self.plot_mode_box.configure(values=vals)
        if self.plot_mode.get() not in vals:
            self.plot_mode.set(vals[0])

    def apply_preset(self, name: str):
        if name == "Licel pretrigger 1024":
            self.bg_mode.set("pretrigger"); self.pretrigger_bins.set(1024); self.first_signal_bin.set(1025)
            self.first_signal_range_m.set(3.75); self.blend_r1_m.set(1200.0); self.blend_r2_m.set(1800.0)
            self.min_toggle_rate.set(2.0); self.max_toggle_rate.set(10.0)
            self.day_min_toggle_rate.set(75.0); self.day_max_toggle_rate.set(130.0)
            self.toggle_bg_switch_threshold_mhz.set(10.0); self.pretrigger_trim_bins.set(24)
            self.sig_start_m.set(0.0); self.sig_end_m.set(15000.0); self.energy_mj.set(25.0)
            self.bin_shift_bins.set(0)
        elif name == "Legacy no pretrigger":
            self.bg_mode.set("fixed"); self.pretrigger_bins.set(0); self.first_signal_bin.set(1)
            self.first_signal_range_m.set(3.75); self.bin_shift_bins.set(0)
            self.bg_start_m.set(13000.0); self.bg_end_m.set(14500.0)
            self.blend_r1_m.set(1200.0); self.blend_r2_m.set(1800.0)
            self.min_toggle_rate.set(2.0); self.max_toggle_rate.set(10.0)
            self.day_min_toggle_rate.set(75.0); self.day_max_toggle_rate.set(130.0)
            self.toggle_bg_switch_threshold_mhz.set(10.0); self.pretrigger_trim_bins.set(0)
            self.sig_start_m.set(0.0); self.sig_end_m.set(15000.0); self.energy_mj.set(25.0)
        self.preset_name.set(name)
        self._update_bg_mode_ui()

    def auto_detect_bins(self):
        folder = self.folder.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Choose a folder first.")
            return
        files = sorted(Path(folder).glob(self.pattern.get().strip() or "*.dat"))
        if not files:
            messagebox.showerror("Error", "No .dat files found.")
            return
        try:
            arr = _read_tr40_dat_ascii_array(files[0])
            bins = int(arr.shape[0])
            self._log(f"Auto detect: {files[0].name} → {bins} bins")
            self.apply_preset("Licel pretrigger 1024" if bins >= 5024 else "Legacy no pretrigger")
        except Exception as e:
            messagebox.showerror("Auto detect failed", str(e))

    def pick_folder(self):
        p = filedialog.askdirectory(title="Select .dat folder")
        if not p:
            return
        self.folder.set(p)
        self._log(f"Folder: {p}")
        self.out_path.set(str(Path(p) / f"NRB-{self.date.get()}.xlsx"))
        self._refresh_time_list()

    def pick_output(self):
        out = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if out:
            self.out_path.set(out)

    def _refresh_time_list(self):
        if not hasattr(self, "time_list"):
            return
        self.time_list.delete(0, "end")
        self._times_map.clear()
        folder = self.folder.get().strip()
        if not folder or not os.path.isdir(folder):
            return
        try:
            ordered, _ = nrb_collect_actual_timestamps(
                Path(folder),
                pattern=self.pattern.get().strip() or "*.dat",
                date_str=self.date.get().strip(),
                start_time=self.start_time.get().strip(),
                logger=None,
            )
        except Exception:
            return
        for ts, f in ordered:
            hhmm = ts.strftime("%H:%M")
            if hhmm not in self._times_map:
                self._times_map[hhmm] = f
                self.time_list.insert("end", hhmm)

    def _selected_times(self):
        return [self.time_list.get(i) for i in self.time_list.curselection()]

    def _gluing_mode_str(self) -> str:
        # Auto day/night wins: the engine then picks per profile from the BG.
        if bool(self.glue_auto_daynight.get()):
            return "auto_day_night"
        return "photon_only" if bool(self.photon_only.get()) else "auto"

    def _overlap_mode_key(self) -> str:
        """Map GUI dropdown label to overlap.get_overlap() mode string."""
        m = self.overlap_mode.get().strip().lower()
        if m.startswith("disabled"):
            return "disabled"
        if m.startswith("analytical"):
            return "analytical"
        if m.startswith("load"):
            return "load"
        return "disabled"

    def _compute_range_axis_for_file(self, path: Path) -> np.ndarray:
        """Pre-read a .dat file to determine the actual range axis (after trimming
        trailing zeros + first_signal_bin cut). Needed to build O(R) before calling
        the engine."""
        arr = _read_tr40_dat_ascii_array(path)
        total_bins = int(arr.shape[0])
        pretrigger_bins = int(self.pretrigger_bins.get())
        first_signal_bin = int(self.first_signal_bin.get())
        if first_signal_bin > total_bins:
            raise ValueError(
                f"first_signal_bin={first_signal_bin} exceeds total bins ({total_bins}) in {path.name}"
            )
        n = total_bins - (first_signal_bin - 1)
        first_signal_range_m = float(self.first_signal_range_m.get())
        dr_m = float(self.bin_spacing_m.get())
        return first_signal_range_m + np.arange(n, dtype=float) * dr_m

    def _build_overlap_for_path(self, path: Path) -> Optional[np.ndarray]:
        """Build O(R) for a single .dat file based on current UI overlap settings.
        Returns None if overlap is disabled or module unavailable."""
        mode_key = self._overlap_mode_key()
        if mode_key == "disabled":
            return None
        if not _HAS_OVERLAP:
            self._safe(messagebox.showwarning, "Overlap module missing",
                       "overlap.py not found — running without overlap correction.")
            return None
        try:
            r_axis = self._compute_range_axis_for_file(path)
            return _get_overlap_array(
                mode=mode_key,
                r_m_target=r_axis,
                file_path=self.overlap_file.get().strip() or None,
                hardware=None,  # uses NARIT defaults
            )
        except Exception as e:
            self._safe(messagebox.showerror, "Overlap error",
                       f"Could not build O(R):\n{e}")
            return None

    def _afterpulse_mode_key(self) -> str:
        m = self.afterpulse_mode.get().strip().lower()
        if m.startswith("disabled"):
            return "disabled"
        if m.startswith("load"):
            return "load"
        return "disabled"

    def _build_afterpulse_for_path(self, path: Path) -> Optional[np.ndarray]:
        """Build A(R) for the current measurement based on UI settings.

        The afterpulse calibration file is shared across all profiles (same
        hardware) — but is interpolated onto the per-file range grid.
        Returns None if disabled or module unavailable."""
        mode_key = self._afterpulse_mode_key()
        if mode_key == "disabled":
            return None
        if not _HAS_AFTERPULSE:
            self._safe(messagebox.showwarning, "Afterpulse module missing",
                       "afterpulse.py not found — running without afterpulse correction.")
            return None
        file_path = self.afterpulse_file.get().strip()
        if not file_path:
            self._safe(messagebox.showerror, "Afterpulse",
                       "Please choose an afterpulse calibration file (.dat / .csv / .xlsx).")
            return None
        try:
            r_axis = self._compute_range_axis_for_file(path)
            # The calibration .dat may have a different layout from the measurement
            # files: if it carries no pretrigger block, read it with 0 pretrigger
            # bins and take the BG from the far range instead.
            no_pre = bool(self.ap_no_pretrigger.get())
            return _get_afterpulse_array(
                mode="load",
                r_m_target=r_axis,
                file_path=file_path,
                # Use same hardware-related params so .dat files process consistently
                dr_m=float(self.bin_spacing_m.get()),
                dead_time_ns=float(self.dead_time_ns.get()),
                pretrigger_bins=0 if no_pre else int(self.pretrigger_bins.get()),
                first_signal_bin=1 if no_pre else int(self.first_signal_bin.get()),
                first_signal_range_m=float(self.first_signal_range_m.get()),
                pretrigger_trim_bins=0 if no_pre else int(self.pretrigger_trim_bins.get()),
                bg_mode="far_range" if no_pre else "pretrigger",
            )
        except Exception as e:
            self._safe(messagebox.showerror, "Afterpulse error",
                       f"Could not build A(R):\n{e}")
            return None

    def _build_profile_for_plot(self, hhmm):
        path = self._times_map.get(hhmm)
        if path is None:
            raise ValueError(f"No file for {hhmm}")
        overlap_O_R = self._build_overlap_for_path(path)
        afterpulse_A_R = self._build_afterpulse_for_path(path)
        return build_single_profile(
            path,
            dr_m=float(self.bin_spacing_m.get()), dead_time_ns=float(self.dead_time_ns.get()),
            bg_mode=self.bg_mode.get().strip().lower(),
            bg_start_m=float(self.bg_start_m.get()), bg_end_m=float(self.bg_end_m.get()),
            pretrigger_bins=int(self.pretrigger_bins.get()),
            first_signal_bin=int(self.first_signal_bin.get()),
            first_signal_range_m=float(self.first_signal_range_m.get()),
            blend_r1_m=float(self.blend_r1_m.get()), blend_r2_m=float(self.blend_r2_m.get()),
            shift_mode="manual", bin_shift_bins=0,
            sig_start_m=float(self.sig_start_m.get()), sig_end_m=float(self.sig_end_m.get()),
            min_toggle_rate=float(self.min_toggle_rate.get()),
            max_toggle_rate=float(self.max_toggle_rate.get()),
            auto_toggle_selector=bool(self.auto_toggle_selector.get()),
            day_min_toggle_rate=float(self.day_min_toggle_rate.get()),
            day_max_toggle_rate=float(self.day_max_toggle_rate.get()),
            toggle_bg_switch_threshold_mhz=float(self.toggle_bg_switch_threshold_mhz.get()),
            pretrigger_trim_bins=int(self.pretrigger_trim_bins.get()),
            energy_mj=float(self.energy_mj.get()),
            auto_blend=bool(self.auto_blend.get()),
            gluing_mode=self._gluing_mode_str(),
            overlap_O_R=overlap_O_R,
            overlap_O_min=float(self.overlap_o_min.get()),
            afterpulse_A_R=afterpulse_A_R,
        )

    def _build_plot_df(self, hhmm):
        prof, _ = self._build_profile_for_plot(hhmm)
        src = self.data_source.get()
        mode = self.plot_mode.get()
        out = pd.DataFrame({"Range (m)": prof["range_m"].to_numpy(float)})
        if src == "Raw .dat":
            out[hhmm] = prof["analog_mV" if mode == "Analog Raw" else "photon_MHz"].to_numpy(float)
        else:
            col = {"Photon DT": "photon_deadtime_corr_MHz",
                   "Analog Scaled": "analog_scaled_MHz",
                   "Glue": "glued_profile_MHz",
                   "NRB": "nrb"}[mode]
            out[hhmm] = prof[col].to_numpy(float)
        return out

    def _plot_glue_overlay(self, hhmm):
        prof, meta = self._build_profile_for_plot(hhmm)
        r = prof["range_m"].to_numpy(float)
        ph_dt = prof["photon_deadtime_corr_MHz"].to_numpy(float)
        an_sc = prof["analog_scaled_MHz"].to_numpy(float)
        glued = prof["glued_profile_MHz"].to_numpy(float)

        def _plot_finite(x, y, **kw):
            m = np.isfinite(x) & np.isfinite(y)
            if m.any():
                self.ax2.plot(x[m], y[m], **kw)

        _plot_finite(r, ph_dt, color=theme.NAVY, linewidth=1.0, alpha=0.85, label="Photon DT")
        _plot_finite(r, an_sc, color=theme.GREEN, linewidth=1.0, linestyle="--", alpha=0.85, label="Scaled Analog")
        _plot_finite(r, glued, color=theme.ORANGE, linewidth=1.8, label="Glued")

        min_tog = float(meta.get("min_toggle_rate", np.nan))
        max_tog = float(meta.get("max_toggle_rate", np.nan))
        r1 = float(meta.get("blend_r1_used_m", np.nan))
        r2 = float(meta.get("blend_r2_used_m", np.nan))
        if np.isfinite(min_tog):
            self.ax2.axhline(min_tog, color=theme.TEXT_MUTED, linestyle=":", linewidth=0.9,
                             alpha=0.7, label=f"min toggle = {min_tog:g}")
        if np.isfinite(max_tog):
            self.ax2.axhline(max_tog, color=theme.TEXT_MUTED, linestyle=":", linewidth=0.9,
                             alpha=0.7, label=f"max toggle = {max_tog:g}")
        if np.isfinite(r1):
            self.ax2.axvline(r1, color=theme.BROWN, linestyle="--", linewidth=0.9,
                             alpha=0.85, label=f"r1 = {r1:.0f} m")
        if np.isfinite(r2):
            self.ax2.axvline(r2, color=theme.RED, linestyle="--", linewidth=0.9,
                             alpha=0.85, label=f"r2 = {r2:.0f} m")

        glue_mode_txt = str(meta.get("glue_mode", "")) or str(meta.get("gluing_mode", ""))
        self.ax2.set_title(self.chart_title_var.get() or f"Glue Overlay · {hhmm} · {glue_mode_txt}")

    def refresh_plot(self):
        times = self._selected_times()
        if not times:
            return
        self.ax2.clear()
        mode = self.plot_mode.get()
        if mode == "Glue Overlay":
            try:
                self._plot_glue_overlay(times[0])
            except Exception as e:
                self.ax2.set_title(f"Error: {e}")
        else:
            gate_nrb = bool(self.snr_gate.get()) and mode == "NRB"
            trusted_tops = []
            for hhmm in times:
                try:
                    df = self._build_plot_df(hhmm)
                    rx = df["Range (m)"].to_numpy(float)
                    y = df[hhmm].to_numpy(float)
                    m = np.isfinite(rx) & np.isfinite(y)
                    if m.any():
                        self.ax2.plot(rx[m], y[m], linewidth=1.3, label=hhmm)
                        if gate_nrb:
                            # NRB is masked above the SNR-trusted range → the last
                            # finite bin marks that top; annotate it.
                            trusted_tops.append(float(np.nanmax(rx[m])))
                except Exception as e:
                    self.ax2.set_title(str(e))
            for i, xt in enumerate(trusted_tops):
                if np.isfinite(xt):
                    self.ax2.axvline(xt, color=theme.TEXT_MUTED, ls="--", lw=0.9, alpha=0.7,
                                     label="SNR-trusted top" if i == 0 else None)
            self.ax2.set_title(self.chart_title_var.get()
                               or f"{self.data_source.get()} · {mode}")
        self.ax2.set_xlabel(self.x_title_var.get() or "Range (m)")
        self.ax2.set_ylabel(self.y_title_var.get() or "Signal")
        self.ax2.grid(True, alpha=0.3)
        if mode == "Glue Overlay" or len(times) > 1:
            self.ax2.legend(fontsize=8, loc="best")
        self.apply_axes(redraw=False)
        self.fig2.tight_layout()
        self.canvas2.draw_idle()

    @staticmethod
    def _parse_lim(s):
        s = s.strip()
        return None if s == "" else float(s)

    def apply_axes(self, redraw: bool = True):
        try:
            xmin = self._parse_lim(self.xmin_var.get())
            xmax = self._parse_lim(self.xmax_var.get())
            ymin = self._parse_lim(self.ymin_var.get())
            ymax = self._parse_lim(self.ymax_var.get())
            if xmin is not None or xmax is not None:
                self.ax2.set_xlim(left=xmin, right=xmax)
            if ymin is not None or ymax is not None:
                self.ax2.set_ylim(bottom=ymin, top=ymax)
            if redraw:
                self.canvas2.draw_idle()
        except Exception:
            pass

    def auto_scale(self):
        for v in [self.xmin_var, self.xmax_var, self.ymin_var, self.ymax_var]:
            v.set("")
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.canvas2.draw_idle()

    def save_png(self):
        out = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if out:
            self.fig2.savefig(out, dpi=150, bbox_inches="tight")
            self._log(f"Saved PNG: {out}")

    def save_csv(self):
        times = self._selected_times()
        if not times:
            messagebox.showinfo("No selection", "Select a time first.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not out:
            return
        try:
            df = self._build_plot_df(times[0])
            df.to_csv(out, index=False)
            self._log(f"Saved CSV: {out}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run(self):
        if not _HAS_NRB:
            messagebox.showerror("Missing", "nrb_engine.py not found.")
            return
        folder = self.folder.get().strip()
        out_path = self.out_path.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Choose a valid folder.")
            return
        if not out_path:
            messagebox.showerror("Error", "Choose output path.")
            return
        try:
            pd.to_datetime(self.date.get().strip())
            _parse_hhmm_text(self.start_time.get().strip(), field_name="Start time")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.run_btn.configure(state="disabled")
        self.progress.set(0.0); self.pb.set(0.0)
        self.status.set("Running…", "running")
        self._log("=== START ===")
        self._refresh_time_list()

        def progress_cb(v):
            self._safe(self.progress.set, v)
            self._safe(self.pb.set, max(0.0, min(1.0, v / 100.0)))

        def worker():
            try:
                # Build O(R) and A(R) once using the first .dat file's range axis
                # (all files share the same hardware → same correction functions).
                overlap_O_R = None
                afterpulse_A_R = None
                try:
                    first_file = next(Path(folder).glob(
                        self.pattern.get().strip() or "*.dat"))
                except StopIteration:
                    first_file = None

                if first_file is not None and self._overlap_mode_key() != "disabled":
                    try:
                        overlap_O_R = self._build_overlap_for_path(first_file)
                        if overlap_O_R is not None:
                            n_trust = int(np.sum(overlap_O_R >= float(self.overlap_o_min.get())))
                            self._safe(self._log,
                                       f"Overlap correction enabled: "
                                       f"{self._overlap_mode_key()} mode · "
                                       f"{n_trust}/{len(overlap_O_R)} bins trusted "
                                       f"(O >= {self.overlap_o_min.get():.2f})")
                    except Exception as e:
                        self._safe(self._log, f"[WARN] overlap setup: {e}")

                if first_file is not None and self._afterpulse_mode_key() != "disabled":
                    try:
                        afterpulse_A_R = self._build_afterpulse_for_path(first_file)
                        if afterpulse_A_R is not None:
                            self._safe(self._log,
                                       f"Afterpulse correction enabled: "
                                       f"{len(afterpulse_A_R)} bins · "
                                       f"A(R~10m) = {afterpulse_A_R[2]:.3e} MHz")
                    except Exception as e:
                        self._safe(self._log, f"[WARN] afterpulse setup: {e}")

                profile_df, qc_df, params_df = nrb_build_daily_profile_from_folder(
                    Path(folder),
                    date_str=self.date.get().strip(),
                    pattern=self.pattern.get().strip() or "*.dat",
                    out_path=Path(out_path),
                    start_time=self.start_time.get().strip(),
                    dr_m=float(self.bin_spacing_m.get()),
                    dead_time_ns=float(self.dead_time_ns.get()),
                    bg_mode=self.bg_mode.get().strip().lower(),
                    bg_start_m=float(self.bg_start_m.get()),
                    bg_end_m=float(self.bg_end_m.get()),
                    pretrigger_bins=int(self.pretrigger_bins.get()),
                    first_signal_bin=int(self.first_signal_bin.get()),
                    first_signal_range_m=float(self.first_signal_range_m.get()),
                    blend_r1_m=float(self.blend_r1_m.get()),
                    blend_r2_m=float(self.blend_r2_m.get()),
                    shift_mode="manual", bin_shift_bins=0,
                    sig_start_m=float(self.sig_start_m.get()),
                    sig_end_m=float(self.sig_end_m.get()),
                    min_toggle_rate=float(self.min_toggle_rate.get()),
                    max_toggle_rate=float(self.max_toggle_rate.get()),
                    auto_toggle_selector=bool(self.auto_toggle_selector.get()),
                    day_min_toggle_rate=float(self.day_min_toggle_rate.get()),
                    day_max_toggle_rate=float(self.day_max_toggle_rate.get()),
                    toggle_bg_switch_threshold_mhz=float(self.toggle_bg_switch_threshold_mhz.get()),
                    pretrigger_trim_bins=int(self.pretrigger_trim_bins.get()),
                    energy_mj=float(self.energy_mj.get()),
                    auto_blend=bool(self.auto_blend.get()),
                    gluing_mode=self._gluing_mode_str(),
                    overlap_O_R=overlap_O_R,
                    overlap_O_min=float(self.overlap_o_min.get()),
                    afterpulse_A_R=afterpulse_A_R,
                    snr_gate=bool(self.snr_gate.get()),
                    snr_min=float(self.snr_min.get()),
                    exclude_saturated=bool(self.exclude_saturated.get()),
                    strict=bool(self.strict.get()),
                    logger=self._safe_log,
                    progress_cb=progress_cb,
                )
                self._latest_profile_df = profile_df
                ts_cols = [pd.Timestamp(c) for c in profile_df.columns[1:]
                           if pd.notna(pd.to_datetime(c, errors="coerce"))]
                interval_min = _infer_interval_minutes(ts_cols)
                self._safe(self._log, f"Profiles used: {len(ts_cols)}")
                if interval_min is not None:
                    self._safe(self._log, f"Inferred interval: {interval_min:g} min")
                self._safe(self._log, f"[OK] Saved: {Path(out_path).resolve()}")
                self._safe(self.status.set, "Done", "ok")
                self._safe(self.progress.set, 100.0); self._safe(self.pb.set, 1.0)
                self._safe(self.app_state.set_step2, str(Path(out_path).resolve()))
                self._safe(self._refresh_time_list)
                self._safe(messagebox.showinfo, "Step 2 Complete", f"Saved:\n{out_path}")
            except Exception as e:
                self._safe(self._log, f"[FAILED] {e}")
                self._safe(self.status.set, "Failed", "error")
                self._safe(messagebox.showerror, "Failed", str(e))
            finally:
                self._safe(self.run_btn.configure, state="normal")
                self._safe(self._log, "=== END ===")

        threading.Thread(target=worker, daemon=True).start()


# ═════════════════════════════════════════════════════════════════════════════
# Step 3 Page  –  ALT Calculator
# ═════════════════════════════════════════════════════════════════════════════
DETECTION_MODES = ["dual", "mpl_guided", "nrb_profile"]


class Step3Page(ctk.CTkFrame):
    def __init__(self, master, app_state: AppState):
        super().__init__(master, fg_color="transparent")
        self.app_state = app_state

        self.engine_path = tk.StringVar(value=self._auto_engine())
        self.nrb_path = tk.StringVar()
        self.nrb_sheet = tk.StringVar()
        self.rmin_path = tk.StringVar()
        self.rmin_sheet = tk.StringVar()
        self.out_path = tk.StringVar()
        self.tol_min = tk.StringVar(value="3")
        self.min_valid_frac = tk.StringVar(value="0.50")
        self.min_valid_bins = tk.StringVar(value="50")
        self.detection_mode = tk.StringVar(value="dual")
        self.profile_rmin = tk.StringVar(value="0")
        self.profile_rmax = tk.StringVar(value="4000")
        self.tol_m = tk.StringVar(value="250")  # HWCT Haar half-window (m)
        # ── Track 3 — Cloud detection ───────────────────────────────────────
        self.cloud_detect_enable = tk.BooleanVar(value=False)
        # NRB-magnitude cloud cap = primary guard so ALT is not mistaken for a
        # (water OR ice) cloud edge. ON by default; the surface-BL guard makes it
        # a no-op on clean profiles. Uses the cloud_threshold below.
        self.nrb_cloud_cap = tk.BooleanVar(value=True)
        self.cloud_threshold = tk.StringVar(value="0.3")
        self.cloud_min_thickness = tk.StringVar(value="100")
        self.cloud_max_layers = tk.StringVar(value="3")
        # SNR-trusted-range cap: keep the ALT search out of the noise floor.
        # Reads the "SNR" sheet from the NRB workbook (Depol/Step-2 output).
        self.snr_cap = tk.BooleanVar(value=True)
        self.snr_cap_min = tk.StringVar(value="3.0")
        # ── Track 2 — Depolarization aid (from Step 6) ──────────────────────
        self.depol_file = tk.StringVar(value="")
        self.depol_cloud_screen = tk.BooleanVar(value=False)
        self.depol_ice_thr = tk.StringVar(value="0.35")
        self.depol_confirm = tk.BooleanVar(value=False)
        self.depol_confirm_tol = tk.StringVar(value="300")

        self._build_ui()
        app_state.register_cb(self._on_state_change)

    @staticmethod
    def _auto_engine() -> str:
        for name in ENGINE_CANDIDATES:
            p = os.path.join(_DIR, name)
            if os.path.exists(p):
                return p
        return ""

    def _on_state_change(self):
        """Auto-populate paths when upstream steps complete."""
        if self.app_state.step2_output and not self.nrb_path.get():
            self.nrb_path.set(self.app_state.step2_output)
            self.load_nrb_sheets()
        if self.app_state.step1_output and not self.rmin_path.get():
            self.rmin_path.set(self.app_state.step1_output)
            self.load_rmin_sheets()
        # Auto-chain: Depolarization (Step 3) output → ALT depol aid field
        if getattr(self.app_state, "depol_output", None) and not self.depol_file.get().strip():
            self.depol_file.set(self.app_state.depol_output)
            self.depol_confirm.set(True)   # turn the cross-check on by default once δ is available

    # ── Layout ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = PageHeader(
            self,
            title="Step 3 · ALT Calculator",
            subtitle="NRB profile → Aerosol Layer Top, with optional MPL-guided rmin/rmax",
            badge=("FFT + HWCT", "navy"),
        )
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))

        # Resizable split
        self.paned = tk.PanedWindow(
            self, orient="horizontal", bg=theme.APP_BG,
            sashrelief="flat", sashwidth=8, sashpad=0, bd=0, showhandle=False,
        )
        self.paned.grid(row=1, column=0, sticky="nsew")

        # ── Left pane: form ────────────────────────────────────────────────
        left_pane = tk.Frame(self.paned, bg=theme.APP_BG, bd=0, highlightthickness=0)
        left_pane.grid_columnconfigure(0, weight=1)
        left_pane.grid_rowconfigure(0, weight=1)
        self.paned.add(left_pane, minsize=400, width=620, stretch="always")

        left = ctk.CTkScrollableFrame(left_pane, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_columnconfigure(0, weight=1)

        self._build_engine_card(left).grid(row=0, column=0, sticky="ew", pady=(0, 12), padx=(0, 10))
        self._build_nrb_card(left).grid(row=1, column=0, sticky="ew", pady=(0, 12), padx=(0, 10))
        self._build_rmin_card(left).grid(row=2, column=0, sticky="ew", pady=(0, 12), padx=(0, 10))
        self._build_output_card(left).grid(row=3, column=0, sticky="ew", pady=(0, 12), padx=(0, 10))
        self._build_options_card(left).grid(row=4, column=0, sticky="ew", pady=(0, 12), padx=(0, 10))
        self._build_run_card(left).grid(row=5, column=0, sticky="ew", padx=(0, 10))

        # ── Right pane: console + status ───────────────────────────────────
        right_pane = tk.Frame(self.paned, bg=theme.APP_BG, bd=0, highlightthickness=0)
        right_pane.grid_columnconfigure(0, weight=1)
        right_pane.grid_rowconfigure(0, weight=1)
        self.paned.add(right_pane, minsize=380, stretch="always")

        right = ctk.CTkFrame(right_pane, fg_color="transparent")
        right.grid(row=0, column=0, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)

        self.console = ConsoleLog(right, title="Console Log")
        self.console.grid(row=0, column=0, sticky="nsew", pady=(0, 8))

        self.status = StatusBar(right)
        self.status.grid(row=1, column=0, sticky="ew", pady=(4, 0))

    # ── Card builders ──────────────────────────────────────────────────────
    def _build_engine_card(self, parent) -> "Card":
        card = Card(parent, title="ALT Engine Script", icon="🔧")
        body = card.body
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            body, text="Python engine file (pbl_engine.py)",
            font=theme.F_SMALL, text_color=theme.TEXT_SECONDARY, anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        ctk.CTkEntry(body, textvariable=self.engine_path, **theme.input_style()).grid(
            row=1, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(
            body, text="Browse…", command=self.pick_engine, width=100,
            **theme.secondary_button_style(),
        ).grid(row=1, column=1)
        if not self.engine_path.get():
            ctk.CTkLabel(
                body, text="⚠  Engine not found. Browse to pbl_engine.py",
                font=theme.F_TINY, text_color=theme.RED, anchor="w",
            ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))
        return card

    def _build_nrb_card(self, parent) -> "Card":
        card = Card(parent, title="NRB Excel File", icon="📊")
        body = card.body
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkEntry(body, textvariable=self.nrb_path, **theme.input_style()).grid(
            row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(
            body, text="Browse…", command=self.pick_nrb, width=100,
            **theme.secondary_button_style(),
        ).grid(row=0, column=1)

        row2 = ctk.CTkFrame(body, fg_color="transparent")
        row2.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        row2.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(row2, text="Sheet:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 8))
        self.nrb_combo = ctk.CTkOptionMenu(
            row2, variable=self.nrb_sheet, values=[""], width=300,
            fg_color=theme.CARD_BG, button_color=theme.CARD_BG, button_hover_color=theme.CREAM_SOFT,
            text_color=theme.TEXT_PRIMARY, dropdown_fg_color=theme.CARD_BG,
            dropdown_text_color=theme.TEXT_PRIMARY, dropdown_hover_color=theme.CREAM,
            corner_radius=theme.RADIUS_INPUT, height=32, font=theme.F_BODY,
        )
        self.nrb_combo.grid(row=0, column=1, sticky="w")
        ctk.CTkButton(
            row2, text="↻ Reload", command=self.load_nrb_sheets,
            **theme.ghost_button_style(width=90, height=32),
        ).grid(row=0, column=2, padx=(8, 0))
        return card

    def _build_rmin_card(self, parent) -> "Card":
        card = Card(parent, title="MPL rmin-rmax Excel (optional for nrb_profile mode)", icon="📐")
        body = card.body
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkEntry(body, textvariable=self.rmin_path, **theme.input_style()).grid(
            row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(
            body, text="Browse…", command=self.pick_rmin, width=100,
            **theme.secondary_button_style(),
        ).grid(row=0, column=1)

        row2 = ctk.CTkFrame(body, fg_color="transparent")
        row2.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        row2.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(row2, text="Sheet:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 8))
        self.rmin_combo = ctk.CTkOptionMenu(
            row2, variable=self.rmin_sheet, values=[""], width=300,
            fg_color=theme.CARD_BG, button_color=theme.CARD_BG, button_hover_color=theme.CREAM_SOFT,
            text_color=theme.TEXT_PRIMARY, dropdown_fg_color=theme.CARD_BG,
            dropdown_text_color=theme.TEXT_PRIMARY, dropdown_hover_color=theme.CREAM,
            corner_radius=theme.RADIUS_INPUT, height=32, font=theme.F_BODY,
        )
        self.rmin_combo.grid(row=0, column=1, sticky="w")
        ctk.CTkButton(
            row2, text="↻ Reload", command=self.load_rmin_sheets,
            **theme.ghost_button_style(width=90, height=32),
        ).grid(row=0, column=2, padx=(8, 0))
        return card

    def _build_output_card(self, parent) -> "Card":
        card = Card(parent, title="Output", icon="💾")
        body = card.body
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkEntry(body, textvariable=self.out_path, **theme.input_style()).grid(
            row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(
            body, text="Save As…", command=self.pick_output, width=100,
            **theme.secondary_button_style(),
        ).grid(row=0, column=1)
        return card

    def _build_options_card(self, parent) -> "Card":
        card = Card(parent, title="Options", icon="⚙️")
        body = card.body
        body.grid_columnconfigure(0, weight=1)

        # Detection mode + NRB range row
        row1 = ctk.CTkFrame(body, fg_color="transparent")
        row1.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        row1.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(row1, text="ALT detection mode:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 8))
        ctk.CTkOptionMenu(
            row1, variable=self.detection_mode, values=DETECTION_MODES, width=140,
            fg_color=theme.CARD_BG, button_color=theme.CARD_BG, button_hover_color=theme.CREAM_SOFT,
            text_color=theme.TEXT_PRIMARY, dropdown_fg_color=theme.CARD_BG,
            dropdown_text_color=theme.TEXT_PRIMARY, dropdown_hover_color=theme.CREAM,
            corner_radius=theme.RADIUS_INPUT, height=32, font=theme.F_BODY,
        ).grid(row=0, column=1, sticky="w")

        # NRB range row
        row2 = ctk.CTkFrame(body, fg_color="transparent")
        row2.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ctk.CTkLabel(row2, text="NRB profile range (m):", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 8))
        ctk.CTkEntry(row2, textvariable=self.profile_rmin, width=80,
                     **theme.input_style()).grid(row=0, column=1, padx=(0, 4))
        ctk.CTkLabel(row2, text="to", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=2, padx=4)
        ctk.CTkEntry(row2, textvariable=self.profile_rmax, width=80,
                     **theme.input_style()).grid(row=0, column=3, padx=(4, 0))
        ctk.CTkLabel(row2, text="HWCT tol_m (m):", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=4, padx=(20, 8))
        ctk.CTkEntry(row2, textvariable=self.tol_m, width=80,
                     **theme.input_style()).grid(row=0, column=5)
        ctk.CTkLabel(row2, text="Haar step half-window — broad BL top ~250, sharp edge ~25",
                     font=theme.F_TINY, text_color=theme.TEXT_MUTED).grid(
            row=1, column=0, columnspan=6, sticky="w", pady=(3, 0))

        # Numerical options row
        row3 = ctk.CTkFrame(body, fg_color="transparent")
        row3.grid(row=2, column=0, sticky="ew")
        for i in range(3):
            row3.grid_columnconfigure(i, weight=1)
        FieldRow(row3, "Tol (min)", self.tol_min).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        FieldRow(row3, "min_valid_frac", self.min_valid_frac).grid(row=0, column=1, sticky="ew", padx=3)
        FieldRow(row3, "min_valid_bins", self.min_valid_bins).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        # Checkbox helper + style (used by the cloud / depol sections below).
        check_kw = dict(
            font=theme.F_SMALL, text_color=theme.TEXT_PRIMARY,
            fg_color=theme.ORANGE, hover_color=theme.ORANGE_HOVER,
            border_color=theme.BORDER, checkmark_color="#fff",
            corner_radius=4, border_width=2,
            onvalue=True, offvalue=False,
        )

        def _add_cb(parent, text, var, **pack_kw):
            """Create CTkCheckBox and sync its visual state to the variable's
            initial value (workaround for CTkCheckBox not honouring BooleanVar
            initial value when created)."""
            cb = ctk.CTkCheckBox(parent, text=text, variable=var, **check_kw)
            cb.pack(**pack_kw)
            if bool(var.get()):
                cb.select()
            else:
                cb.deselect()
            return cb

        # ── Cloud detection (Track 3) ──────────────────────────────────────
        ctk.CTkLabel(
            body, text="CLOUD DETECTION (Track 3) — cap ALT below clouds + 'Cloud_results'",
            font=(theme.FONT_FAMILY, 10, "bold"),
            text_color=theme.TEXT_MUTED, anchor="w",
        ).grid(row=8, column=0, sticky="ew", pady=(12, 4))

        # NRB-magnitude cloud cap = primary guard (catches water + ice clouds)
        cd_row0 = ctk.CTkFrame(body, fg_color="transparent")
        cd_row0.grid(row=9, column=0, sticky="ew", pady=(0, 2))
        _add_cb(cd_row0, "Cap ALT search below NRB cloud (water+ice) — primary guard",
                self.nrb_cloud_cap, side="left")

        # Enable + threshold row (shared NRB threshold for cap + detection)
        cd_row1 = ctk.CTkFrame(body, fg_color="transparent")
        cd_row1.grid(row=10, column=0, sticky="ew", pady=(0, 4))
        _add_cb(cd_row1, "Also list layers (Cloud_results) — NRB threshold:",
                self.cloud_detect_enable, side="left")
        ctk.CTkEntry(cd_row1, textvariable=self.cloud_threshold, width=70,
                     **theme.input_style()).pack(side="left", padx=(8, 4))
        ctk.CTkLabel(cd_row1, text="(normalized, 0.2-0.7)",
                     font=theme.F_TINY, text_color=theme.TEXT_MUTED).pack(side="left")

        # Thickness + max layers row
        cd_row2 = ctk.CTkFrame(body, fg_color="transparent")
        cd_row2.grid(row=11, column=0, sticky="ew", pady=(4, 0))
        ctk.CTkLabel(cd_row2, text="Min thickness (m):",
                     font=theme.F_SMALL, text_color=theme.TEXT_SECONDARY).pack(side="left")
        ctk.CTkEntry(cd_row2, textvariable=self.cloud_min_thickness, width=70,
                     **theme.input_style()).pack(side="left", padx=(4, 12))
        ctk.CTkLabel(cd_row2, text="Max layers per profile:",
                     font=theme.F_SMALL, text_color=theme.TEXT_SECONDARY).pack(side="left")
        ctk.CTkEntry(cd_row2, textvariable=self.cloud_max_layers, width=70,
                     **theme.input_style()).pack(side="left", padx=(4, 0))

        # ── Depolarization aid (Track 2) ───────────────────────────────────
        ctk.CTkLabel(
            body, text="DEPOLARIZATION AID (Track 2) — use δ from Step 6 to help ALT",
            font=(theme.FONT_FAMILY, 10, "bold"),
            text_color=theme.TEXT_MUTED, anchor="w",
        ).grid(row=12, column=0, sticky="ew", pady=(12, 4))

        dp_row0 = ctk.CTkFrame(body, fg_color="transparent")
        dp_row0.grid(row=13, column=0, sticky="ew", pady=(0, 4))
        dp_row0.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(dp_row0, text="Depol workbook (Step 6 output, optional):",
                     font=theme.F_SMALL, text_color=theme.TEXT_SECONDARY).grid(
            row=0, column=0, sticky="w", columnspan=2)
        ctk.CTkEntry(dp_row0, textvariable=self.depol_file, **theme.input_style()).grid(
            row=1, column=0, sticky="ew", padx=(0, 8), pady=(2, 0))
        ctk.CTkButton(dp_row0, text="Browse…", command=self._pick_depol, width=90,
                      **theme.secondary_button_style(height=34, font=theme.F_SMALL)).grid(
            row=1, column=1, pady=(2, 0))

        dp_row1 = ctk.CTkFrame(body, fg_color="transparent")
        dp_row1.grid(row=14, column=0, sticky="ew", pady=(4, 0))
        _add_cb(dp_row1, "Cloud screen (cap below ice cloud) — δ ice thr:",
                self.depol_cloud_screen, side="left")
        ctk.CTkEntry(dp_row1, textvariable=self.depol_ice_thr, width=60,
                     **theme.input_style()).pack(side="left", padx=(8, 4))

        dp_row2 = ctk.CTkFrame(body, fg_color="transparent")
        dp_row2.grid(row=15, column=0, sticky="ew", pady=(4, 0))
        _add_cb(dp_row2, "Confirm edge (δ-based ALT cross-check) — tol (m):",
                self.depol_confirm, side="left")
        ctk.CTkEntry(dp_row2, textvariable=self.depol_confirm_tol, width=60,
                     **theme.input_style()).pack(side="left", padx=(8, 4))

        # ── Signal quality (SNR-trusted-range cap) ─────────────────────────
        ctk.CTkLabel(
            body, text="SIGNAL QUALITY — keep the ALT search out of the noise floor",
            font=(theme.FONT_FAMILY, 10, "bold"),
            text_color=theme.TEXT_MUTED, anchor="w",
        ).grid(row=16, column=0, sticky="ew", pady=(12, 4))
        sq_row = ctk.CTkFrame(body, fg_color="transparent")
        sq_row.grid(row=17, column=0, sticky="ew", pady=(0, 2))
        _add_cb(sq_row, "Cap ALT search at SNR-trusted range (from the NRB 'SNR' sheet) — min SNR:",
                self.snr_cap, side="left")
        ctk.CTkEntry(sq_row, textvariable=self.snr_cap_min, width=60,
                     **theme.input_style()).pack(side="left", padx=(8, 4))
        ctk.CTkLabel(
            body,
            text="Above the SNR-trusted top the profile is noise×R², so any layer there is "
                 "spurious. Reads the SNR sheet the NRB/Depol workbook already writes; no "
                 "extra file needed. Complements the cloud caps (this guards the noise floor).",
            font=theme.F_TINY, text_color=theme.TEXT_MUTED, anchor="w",
            wraplength=520, justify="left",
        ).grid(row=18, column=0, sticky="w", pady=(0, 4))

        return card

    def _pick_depol(self):
        p = filedialog.askopenfilename(
            title="Depol workbook (Step 6 output)",
            filetypes=[("Excel", "*.xlsx *.xls")])
        if p:
            self.depol_file.set(p)

    def _build_run_card(self, parent) -> "Card":
        card = Card(parent)
        body = card.body
        body.grid_columnconfigure(0, weight=1)
        self.run_btn = ctk.CTkButton(
            body, text="▶  Calculate ALT", command=self.run_pbl,
            **theme.primary_button_style(width=240),
        )
        self.run_btn.grid(row=0, column=0, sticky="w")
        return card

    # ── Helpers ────────────────────────────────────────────────────────────
    def _log(self, msg): self.console.log(msg)

    def _safe(self, fn, *a, **kw):
        self.after(0, lambda: fn(*a, **kw))

    # ── Pickers ────────────────────────────────────────────────────────────
    def pick_engine(self):
        p = filedialog.askopenfilename(title="Select engine", filetypes=[("Python", "*.py")])
        if p:
            self.engine_path.set(p)

    def pick_nrb(self):
        p = filedialog.askopenfilename(title="NRB Excel", filetypes=[("Excel", "*.xlsx *.xls")])
        if not p:
            return
        self.nrb_path.set(p)
        self.load_nrb_sheets()
        if not self.out_path.get():
            self.out_path.set(os.path.join(
                os.path.dirname(p),
                f"ALT_{os.path.splitext(os.path.basename(p))[0]}.xlsx",
            ))

    def pick_rmin(self):
        p = filedialog.askopenfilename(title="rmin-rmax Excel", filetypes=[("Excel", "*.xlsx *.xls")])
        if not p:
            return
        self.rmin_path.set(p)
        self.load_rmin_sheets()

    def pick_output(self):
        p = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if p:
            self.out_path.set(p)

    def load_nrb_sheets(self):
        p = self.nrb_path.get().strip()
        if not p:
            return
        try:
            sheets = list(pd.ExcelFile(p).sheet_names)
            self.nrb_combo.configure(values=sheets if sheets else [""])
            self.nrb_sheet.set(PREF_NRB_SHEET if PREF_NRB_SHEET in sheets else (sheets[0] if sheets else ""))
            self._log(f"NRB sheets: {sheets}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_rmin_sheets(self):
        p = self.rmin_path.get().strip()
        if not p:
            return
        try:
            sheets = list(pd.ExcelFile(p).sheet_names)
            self.rmin_combo.configure(values=sheets if sheets else [""])
            self.rmin_sheet.set(PREF_RMIN_SHEET if PREF_RMIN_SHEET in sheets else (sheets[0] if sheets else ""))
            self._log(f"rmin sheets: {sheets}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ── Run ────────────────────────────────────────────────────────────────
    def run_pbl(self):
        engine = self.engine_path.get().strip()
        nrb = self.nrb_path.get().strip()
        nrb_sheet = self.nrb_sheet.get().strip()
        rmin = self.rmin_path.get().strip()
        rmin_sheet = self.rmin_sheet.get().strip()
        out = self.out_path.get().strip()
        det_mode = self.detection_mode.get().strip() or "dual"
        need_rmin = det_mode in ("mpl_guided", "dual")

        for msg, cond in [
            ("Select engine.", not engine or not os.path.exists(engine)),
            ("NRB file not found.", not nrb or not os.path.exists(nrb)),
            ("rmin-rmax file not found for MPL-guided / dual mode.",
                need_rmin and (not rmin or not os.path.exists(rmin))),
            ("Select NRB sheet.", not nrb_sheet),
            ("Select rmin sheet for MPL-guided / dual mode.", need_rmin and not rmin_sheet),
            ("Specify output path.", not out),
        ]:
            if cond:
                messagebox.showerror("Input error", msg)
                return

        try:
            tol = float(self.tol_min.get())
            mvf = float(self.min_valid_frac.get())
            mvb = int(float(self.min_valid_bins.get()))
            prmin = float(self.profile_rmin.get())
            prmax = float(self.profile_rmax.get())
            tol_m_val = float(self.tol_m.get())
        except Exception:
            tol, mvf, mvb, prmin, prmax, tol_m_val = 3.0, 0.50, 50, 0.0, 4000.0, 250.0

        self.run_btn.configure(state="disabled")
        self.status.set("Running…", "running")
        self._log("=== RUN START ===")
        self._log(f"Engine: {engine}")

        def worker():
            tmp_path = None
            try:
                df_nrb = pd.read_excel(nrb, sheet_name=nrb_sheet).dropna(how="all").dropna(axis=1, how="all")
                if df_nrb.empty or df_nrb.shape[1] < 2:
                    raise ValueError("NRB sheet empty.")
                prof_cols = list(df_nrb.columns[1:])
                base_dates = pd.to_datetime(prof_cols, errors="coerce")
                base_dates = base_dates[~pd.isna(base_dates)]
                base_date = (
                    pd.Timestamp(base_dates[0]) if len(base_dates) > 0
                    else (_s3_guess_date(nrb) or _s3_guess_date(rmin) or pd.Timestamp.today())
                )
                slot_times = _s3_parse_times(prof_cols, pd.Timestamp(base_date))
                new_cols = [df_nrb.columns[0]] + [
                    pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S") for t in slot_times
                ]
                df_nrb2 = df_nrb.copy()
                df_nrb2.columns = new_cols

                if rmin and os.path.exists(rmin):
                    df_rmin = pd.read_excel(rmin, sheet_name=rmin_sheet).dropna(how="all").dropna(axis=1, how="all")
                    if df_rmin.empty:
                        raise ValueError("rmin-rmax sheet empty.")
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

                tmp_fd, tmp_path = tempfile.mkstemp(prefix="pbl_", suffix=".xlsx")
                os.close(tmp_fd)
                with pd.ExcelWriter(tmp_path, engine="openpyxl") as w:
                    df_nrb2.to_excel(w, sheet_name=PREF_NRB_SHEET, index=False)
                    df_map.to_excel(w, sheet_name=PREF_RMIN_SHEET, index=False)

                cmd = [
                    sys.executable, engine, "--nrb", tmp_path, "--sheet", PREF_NRB_SHEET,
                    "--rminrmax_sheet", PREF_RMIN_SHEET, "--out", out,
                    "--min_valid_frac", str(mvf), "--min_valid_bins", str(mvb),
                    "--detection_mode", det_mode,
                    "--profile_rmin", str(prmin), "--profile_rmax", str(prmax),
                    "--tol_m", str(tol_m_val),
                ]
                # Track 3 — Cloud detection flags
                try:
                    cd_thr = float(self.cloud_threshold.get())
                except Exception:
                    cd_thr = 0.3
                cap_on = bool(self.nrb_cloud_cap.get())
                if cap_on:
                    cmd.append("--nrb_cloud_cap")
                if bool(self.cloud_detect_enable.get()):
                    cmd.append("--cloud_detect")
                    try:
                        cd_thick = float(self.cloud_min_thickness.get())
                    except Exception:
                        cd_thick = 100.0
                    try:
                        cd_max = int(float(self.cloud_max_layers.get()))
                    except Exception:
                        cd_max = 3
                    cmd += [
                        "--cloud_threshold", str(cd_thr),
                        "--cloud_min_thickness_m", str(cd_thick),
                        "--cloud_max_layers", str(cd_max),
                    ]
                elif cap_on:
                    # cap needs the threshold even when full cloud-detect is off
                    cmd += ["--cloud_threshold", str(cd_thr)]
                # ── Track 2: depolarization aid (optional) ───────────────────
                depf = self.depol_file.get().strip()
                if depf and os.path.isfile(depf):
                    cmd += ["--depol_file", depf, "--depol_sheet", "Depol_delta_v"]
                    if bool(self.depol_cloud_screen.get()):
                        cmd += ["--depol_cloud_screen",
                                "--depol_ice_thr", str(self.depol_ice_thr.get())]
                    if bool(self.depol_confirm.get()):
                        cmd += ["--depol_confirm",
                                "--depol_confirm_tol_m", str(self.depol_confirm_tol.get())]
                # ── Signal quality: SNR-trusted-range cap (reads the NRB 'SNR' sheet) ──
                if bool(self.snr_cap.get()):
                    cmd += ["--snr_cap", "--snr_sheet", "SNR",
                            "--snr_cap_min", str(self.snr_cap_min.get())]
                self._safe(self._log, "CMD: " + " ".join(cmd))
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.stdout:
                    self._safe(self._log, proc.stdout.strip())
                if proc.stderr:
                    self._safe(self._log, proc.stderr.strip())
                if proc.returncode == 0:
                    self._safe(self._log, "=== RUN OK ===")
                    self._safe(self.status.set, "Done", "ok")
                    self._safe(self.app_state.set_step3, str(Path(out).resolve()))
                    self._safe(messagebox.showinfo, "Step 3 Complete", f"Saved:\n{out}")
                else:
                    self._safe(self._log, "=== RUN FAILED ===")
                    self._safe(self.status.set, "Failed", "error")
                    self._safe(messagebox.showerror, "Failed", "See log for details.")
            except Exception as e:
                self._safe(self._log, f"[FAILED] {e}")
                self._safe(self.status.set, "Failed", "error")
                self._safe(messagebox.showerror, "Error", str(e))
            finally:
                try:
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
                self._safe(self.run_btn.configure, state="normal")

        threading.Thread(target=worker, daemon=True).start()


# ═════════════════════════════════════════════════════════════════════════════
# Step 4 Page  –  RTI Visualizer
# ═════════════════════════════════════════════════════════════════════════════
# Preset figure sizes for "Save Figure PNG"
FIGSIZE_PRESETS = {
    "Auto (current canvas)": None,
    "Wide 14 × 10 (default)": (14.0, 10.0),
    "Square 12 × 12":         (12.0, 12.0),
    "Tall 12 × 14":           (12.0, 14.0),
    "Compact 10 × 8":         (10.0, 8.0),
    "A4 landscape 11.7 × 8.3": (11.69, 8.27),
    "A4 portrait 8.3 × 11.7": (8.27, 11.69),
    # Golden ratio φ=1.618 (landscape: width = short-side × φ), short side 6–12 in
    "Golden φ  9.71 × 6":    (9.71, 6.0),
    "Golden φ  11.33 × 7":   (11.33, 7.0),
    "Golden φ  12.94 × 8":   (12.94, 8.0),
    "Golden φ  14.56 × 9":   (14.56, 9.0),
    "Golden φ  16.18 × 10":  (16.18, 10.0),
    "Golden φ  17.80 × 11":  (17.80, 11.0),
    "Golden φ  19.42 × 12":  (19.42, 12.0),
    "Custom (W × H)":         "custom",
}


class FigsizeDialog(ctk.CTkToplevel):
    """Modal dialog to pick figure size before saving."""

    def __init__(self, master, default_choice: str = "Golden φ  9.71 × 6",
                 default_w: float = 9.71, default_h: float = 6.0,
                 default_dpi: int = 600):
        super().__init__(master)
        self.title("Save figure — size options")
        self.geometry("440x300")
        self.resizable(False, False)
        self.transient(master)
        self.configure(fg_color=theme.APP_BG)
        self.result: Optional[Tuple[float, float, int]] = None

        self.choice = tk.StringVar(value=default_choice)
        self.w_var = tk.StringVar(value=str(default_w))
        self.h_var = tk.StringVar(value=str(default_h))
        self.dpi_var = tk.StringVar(value=str(default_dpi))

        ctk.CTkLabel(
            self, text="Figure size", font=theme.F_H2,
            text_color=theme.TEXT_PRIMARY, anchor="w",
        ).pack(fill="x", padx=20, pady=(18, 4))

        ctk.CTkLabel(
            self,
            text="Pick a preset or set custom width × height (inches). "
                 "Higher DPI = sharper PNG but larger file.",
            font=theme.F_SMALL, text_color=theme.TEXT_SECONDARY,
            wraplength=400, justify="left",
        ).pack(fill="x", padx=20, pady=(0, 14))

        ctk.CTkOptionMenu(
            self, variable=self.choice, values=list(FIGSIZE_PRESETS.keys()),
            command=self._on_choice,
            fg_color=theme.CARD_BG, button_color=theme.CARD_BG,
            button_hover_color=theme.CREAM_SOFT,
            text_color=theme.TEXT_PRIMARY,
            dropdown_fg_color=theme.CARD_BG, dropdown_text_color=theme.TEXT_PRIMARY,
            dropdown_hover_color=theme.CREAM,
            corner_radius=theme.RADIUS_INPUT, height=36, width=400,
        ).pack(padx=20, pady=(0, 12))

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=(0, 14))
        for col in range(3):
            row.grid_columnconfigure(col, weight=1)
        for i, (lbl, var) in enumerate([("Width (in)", self.w_var),
                                        ("Height (in)", self.h_var),
                                        ("DPI", self.dpi_var)]):
            FieldRow(row, lbl, var).grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 6, 0))

        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.pack(fill="x", padx=20, pady=(0, 18))
        ctk.CTkButton(
            btns, text="Cancel", command=self._cancel,
            **theme.secondary_button_style(width=120, height=36),
        ).pack(side="right", padx=(8, 0))
        ctk.CTkButton(
            btns, text="Save…", command=self._confirm,
            **theme.primary_button_style(width=140, height=36),
        ).pack(side="right")

        self._on_choice(default_choice)
        self.grab_set()
        self.wait_window(self)

    def _on_choice(self, value: str):
        size = FIGSIZE_PRESETS.get(value)
        if isinstance(size, tuple):
            self.w_var.set(str(size[0]))
            self.h_var.set(str(size[1]))

    def _confirm(self):
        try:
            w = float(self.w_var.get())
            h = float(self.h_var.get())
            dpi = int(float(self.dpi_var.get()))
            if w <= 0 or h <= 0 or dpi <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid input", "Width, height, and DPI must be positive numbers.", parent=self)
            return
        self.result = (w, h, dpi)
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


class Step4Page(ctk.CTkFrame):
    def __init__(self, master, app_state: AppState):
        super().__init__(master, fg_color="transparent")
        self.app_state = app_state
        self._init_data()
        self._init_vars()
        self._build_ui()
        self._init_blank_plot()

    # ── State init ─────────────────────────────────────────────────────────
    def _init_data(self):
        self.proto_path = self.mpl_path = None
        self.proto_date = None
        self.r_proto = self.t_proto = self.t_labels_hhmm = self.t_list_labels = None
        self.Z_proto = self.r_mpl = self.t_mpl = self.Z_mpl = None
        # Channel matrices keyed by label (co / cross / δ), on their own r grid.
        # Populated when the workbook has NRB_cross / Depol_delta_v / MPL_crosspol.
        self._proto_mats = {}   # {"NRB parallel": Z, "NRB perpendicular": Z, "δ_v": Z}
        self._proto_sig = {}    # signal sheets: {"Analog_par": Z, "Glued_perp": Z, ...}
        self._mpl_mats = {}     # {"NRB parallel": Z, "NRB perpendicular": Z}
        self.match_idx = self.dt_min = self.Z_mpl_on_proto = None
        self.alt_df = self.alt_proto_by_slot = self.alt_mpl_by_slot = None
        self.alt_feature_by_slot = None   # ALT_feature per time slot (cloud/aerosol)
        self.r_proto_den_src = self.t_proto_den_src = self.Z_proto_den_src = None
        self.Z_proto_den = None
        self.Z_proto_snr = None   # prototype SNR curtain (from the "SNR" sheet)
        # Cloud detection (from Cloud_results sheet of ALT file)
        self.cloud_df = None
        self.cloud_base_by_slot = None  # dict {layer_index: ndarray[nT]}
        self.cloud_top_by_slot = None   # dict {layer_index: ndarray[nT]}
        self.cloud_max_layer = 0

    def _init_vars(self):
        self.date_str = tk.StringVar(value="")
        self.show_proto_var = tk.BooleanVar(value=True)
        self.show_mpl_var = tk.BooleanVar(value=True)
        self.show_profile_var = tk.BooleanVar(value=True)
        # Channel selectors (real values; co/cross/δ from the loaded workbook)
        self.proto_rti_source = tk.StringVar(value="NRB parallel")
        self.mpl_rti_source = tk.StringVar(value="NRB parallel")
        self.profile_source = tk.StringVar(value="Duo")
        self.profile_channels = tk.StringVar(value="Selected channel")
        self.profile_proto_mode_var = tk.StringVar(value="raw")
        self.show_alt_overlay_var = tk.BooleanVar(value=False)
        self.show_alt_compare_var = tk.BooleanVar(value=False)
        self.show_cloud_markers_var = tk.BooleanVar(value=False)
        self.show_cloud_panel_var = tk.BooleanVar(value=False)
        # RTI-as-lines: stacked vertical NRB profiles at each timestamp (the 5th
        # summary figure) shown live as its own panel. Default OFF.
        self.show_rti_lines_var = tk.BooleanVar(value=False)
        # SNR curtain: heatmap of the prototype photon SNR over range × time,
        # marking the trusted range across the day. Default OFF.
        self.show_snr_var = tk.BooleanVar(value=False)
        # X-axis date: ON = auto MM-DD when the span crosses days (default),
        # OFF = HH:MM only (some users prefer time-only labels for a 2-day plot).
        self.rti_show_date_var = tk.BooleanVar(value=True)
        # Graph title: ON = show (default). OFF = no title at all — for figures
        # going into a paper, where the caption lives below the figure instead.
        self.rti_show_title_var = tk.BooleanVar(value=True)
        self.max_dt_min_var = tk.StringVar(value="3")
        self.alt_path_var = tk.StringVar(value="")
        self.alt_proto_col = tk.StringVar(value=ALT_COL_PROTO_CHOICES[0])
        self.cmap_var = tk.StringVar(value="jet")
        self.interp_var = tk.StringVar(value="nearest")
        self.discrete_var = tk.BooleanVar(value=False)
        self.step_var = tk.StringVar(value="0.05")
        self.use_log_color_var = tk.BooleanVar(value=False)
        self.vmin_var = tk.DoubleVar(value=0.0)
        self.vmax_var = tk.DoubleVar(value=1.0)
        self.scale_mode_var = tk.StringVar(value="preset")
        self.rti_tickN_var = tk.StringVar(value="2")
        # RTI display starts at 120 m (real valid-NRB floor) — display crop only;
        # the workbook data is untouched. Applies to both Prototype and MPL RTI.
        self.rti_ymin_var = tk.StringVar(value="120")
        self.rti_ymax_var = tk.StringVar(value="10000")
        self.rti_ytick_var = tk.StringVar(value="1000")
        self.prof_xmin_var = tk.StringVar(value="0")
        self.prof_xmax_var = tk.StringVar(value="15000")
        self.prof_ymin_var = tk.StringVar(value="0")
        self.prof_ymax_var = tk.StringVar(value="1")
        self.prof_ytick_var = tk.StringVar(value="0.1")
        self.alt_tickN_var = tk.StringVar(value="2")
        self.alt_ymin_var = tk.StringVar(value="120")
        self.alt_ymax_var = tk.StringVar(value="5000")
        self.alt_ytick_var = tk.StringVar(value="500")
        self.log_nrb_axis_var = tk.BooleanVar(value=False)
        self.log_base_var = tk.StringVar(value="10")
        # Titles hold the BASE name only; the date/time range is appended
        # automatically from the loaded data (see _title_with_date).
        self.title_rti_proto_var = tk.StringVar(value="Prototype")
        self.title_rti_mpl_var = tk.StringVar(value="Mini MPL")
        self.title_prof_var = tk.StringVar(value="Prototype VS Mini MPL")
        self.title_altcmp_var = tk.StringVar(value="Prototype VS Mini MPL")
        self.rti_xlabel_var = tk.StringVar(value="Time (Local time)")
        self.rti_ylabel_var = tk.StringVar(value="Height (m)")
        self.prof_xlabel_var = tk.StringVar(value="Height (m)")
        self.prof_ylabel_var = tk.StringVar(value="Normalized Relative Backscatter (a.u.)")
        self.alt_xlabel_var = tk.StringVar(value="Time (Local time)")
        # Editable legend names for the profile view (blank = auto label)
        self.proto_legend_prefix = tk.StringVar(value="Prototype")
        self.mpl_legend_prefix = tk.StringVar(value="Mini MPL")
        # Editable legend names for the ALT-compare panel (blank = auto label)
        self.alt_proto_legend = tk.StringVar(value="Prototype Aerosol Layer Top (ALT)")
        self.alt_mpl_legend = tk.StringVar(value="Mini MPL Aerosol Layer Top (ALT)")
        # Line style (Excel "Format Data Series") for Profile + ALT-compare lines.
        # Color blank = auto (matplotlib cycle, useful for multi-time profiles).
        self.proto_line_width = tk.StringVar(value="2.0")
        self.proto_line_color = tk.StringVar(value="#FF5632")
        self.proto_line_dash  = tk.StringVar(value="solid")
        self.mpl_line_width = tk.StringVar(value="2.0")
        self.mpl_line_color = tk.StringVar(value="#35375B")
        self.mpl_line_dash  = tk.StringVar(value="dashed")
        # Font controls for ALL Display axes (like a Word/Excel font toolbar).
        self.font_family = tk.StringVar(value="Times New Roman")
        self.font_bold = tk.BooleanVar(value=True)
        self.font_title_size = tk.StringVar(value="17")
        self.font_label_size = tk.StringVar(value="15")
        self.font_tick_size = tk.StringVar(value="12")
        self.font_legend_size = tk.StringVar(value="12")
        # Export colour mode: CMYK (print, saved as TIFF) vs RGB PNG.
        self.cmyk_var = tk.BooleanVar(value=True)
        self.alt_ylabel_var = tk.StringVar(value="Height (m)")
        self.cbar_label_var = tk.StringVar(value="Normalized Relative Backscatter (a.u.)")

    # ── UI build ───────────────────────────────────────────────────────────
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = PageHeader(
            self,
            title="Step 5 · Display / Visualize",
            subtitle="Range-Time Intensity · Prototype vs Mini MPL · ALT overlay · drag the divider to resize panels",
            badge=("Final stage", "navy"),
        )
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))

        # Resizable split
        self.paned = tk.PanedWindow(
            self, orient="horizontal", bg=theme.APP_BG,
            sashrelief="flat", sashwidth=8, sashpad=0, bd=0, showhandle=False,
        )
        self.paned.grid(row=1, column=0, sticky="nsew")

        # ── Left pane: control cards (scrollable) ─────────────────────────
        left_pane = tk.Frame(self.paned, bg=theme.APP_BG, bd=0, highlightthickness=0)
        left_pane.grid_columnconfigure(0, weight=1)
        left_pane.grid_rowconfigure(0, weight=1)
        self.paned.add(left_pane, minsize=380, width=440, stretch="never")

        left = ctk.CTkScrollableFrame(left_pane, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_columnconfigure(0, weight=1)

        row = 0
        self._build_load_card(left).grid(row=row, column=0, sticky="ew", pady=(0, 10), padx=(0, 10)); row += 1
        self._build_alt_card(left).grid(row=row, column=0, sticky="ew", pady=(0, 10), padx=(0, 10)); row += 1
        self._build_show_hide_card(left).grid(row=row, column=0, sticky="ew", pady=(0, 10), padx=(0, 10)); row += 1
        self._build_time_card(left).grid(row=row, column=0, sticky="ew", pady=(0, 10), padx=(0, 10)); row += 1
        self._build_save_card(left).grid(row=row, column=0, sticky="ew", pady=(0, 10), padx=(0, 10)); row += 1
        self._build_display_card(left).grid(row=row, column=0, sticky="ew", pady=(0, 10), padx=(0, 10)); row += 1
        self._build_axis_card(left).grid(row=row, column=0, sticky="ew", pady=(0, 10), padx=(0, 10)); row += 1
        self._build_titles_card(left).grid(row=row, column=0, sticky="ew", pady=(0, 10), padx=(0, 10)); row += 1
        self._build_line_style_card(left).grid(row=row, column=0, sticky="ew", pady=(0, 10), padx=(0, 10)); row += 1
        self._build_font_card(left).grid(row=row, column=0, sticky="ew", pady=(0, 10), padx=(0, 10)); row += 1

        # ── Right pane: figure + toolbar ──────────────────────────────────
        right_pane = tk.Frame(self.paned, bg=theme.APP_BG, bd=0, highlightthickness=0)
        right_pane.grid_columnconfigure(0, weight=1)
        right_pane.grid_rowconfigure(0, weight=1)
        self.paned.add(right_pane, minsize=600, stretch="always")

        plot_card = Card(right_pane, title="Visualization", icon="🖼")
        plot_card.grid(row=0, column=0, sticky="nsew")
        plot_card.body.grid_columnconfigure(0, weight=1)
        plot_card.body.grid_rowconfigure(0, weight=1)

        canvas_wrap = ctk.CTkFrame(
            plot_card.body, fg_color=theme.CARD_BG,
            corner_radius=theme.RADIUS_INPUT, border_width=1, border_color=theme.BORDER,
        )
        canvas_wrap.grid(row=0, column=0, sticky="nsew")
        canvas_wrap.grid_columnconfigure(0, weight=1)
        canvas_wrap.grid_rowconfigure(0, weight=1)

        self.fig = plt.Figure(figsize=(12, 8.5), facecolor=theme.CARD_BG)
        self._setup_axes()
        self.cbar1 = self.cbar2 = None
        self.im1 = self.im2 = None
        self.sel_lines = []
        self._mpl_cid = None
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_wrap)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        # matplotlib toolbar (native style — outside the card for visual breathing room)
        toolbar_wrap = tk.Frame(plot_card.body, bg=theme.CARD_BG)
        toolbar_wrap.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_wrap, pack_toolbar=False)
        self.toolbar.configure(background=theme.CARD_BG)
        for child in self.toolbar.winfo_children():
            try:
                child.configure(background=theme.CARD_BG)
            except Exception:
                pass
        self.toolbar.update()
        self.toolbar.pack(side="left", fill="x")

        # Status row below toolbar
        self.status = StatusBar(plot_card.body)
        self.status.grid(row=2, column=0, sticky="ew", pady=(6, 0))

    # ── Cards ──────────────────────────────────────────────────────────────
    def _build_load_card(self, parent) -> "Card":
        card = Card(parent, title="Load Files", icon="📂")
        body = card.body
        body.grid_columnconfigure(0, weight=1)

        # Buttons row (3 wrap-friendly)
        br = ctk.CTkFrame(body, fg_color="transparent")
        br.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        for i in range(3):
            br.grid_columnconfigure(i, weight=1)
        ctk.CTkButton(
            br, text="Load Prototype", command=self.load_prototype,
            **theme.primary_button_style(height=34, font=theme.F_SMALL),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ctk.CTkButton(
            br, text="Load Folder", command=self.load_prototype_folder,
            **theme.secondary_button_style(height=34, font=theme.F_SMALL),
        ).grid(row=0, column=1, sticky="ew", padx=2)
        ctk.CTkButton(
            br, text="Load MPL", command=self.load_mpl,
            **theme.secondary_button_style(height=34, font=theme.F_SMALL),
        ).grid(row=0, column=2, sticky="ew", padx=(4, 0))

        # Match ± row
        ar = ctk.CTkFrame(body, fg_color="transparent")
        ar.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        ar.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(ar, text="Match ±", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 6))
        ctk.CTkEntry(ar, textvariable=self.max_dt_min_var, width=70,
                     **theme.input_style()).grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(ar, text="min", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=2, padx=(4, 8))
        ctk.CTkButton(
            ar, text="↺ Re-align", command=self.align_and_refresh,
            **theme.ghost_button_style(width=110, height=32),
        ).grid(row=0, column=3, sticky="e")

        return card

    def _build_alt_card(self, parent) -> "Card":
        card = Card(parent, title="ALT Results", icon="📊")
        body = card.body
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkEntry(body, textvariable=self.alt_path_var, **theme.input_style()).grid(
            row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(
            body, text="Load ALT", command=self.load_alt, width=100,
            **theme.secondary_button_style(),
        ).grid(row=0, column=1)

        row2 = ctk.CTkFrame(body, fg_color="transparent")
        row2.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ctk.CTkLabel(row2, text="Proto ALT column:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).pack(side="left", padx=(0, 6))
        self.alt_proto_combo = ctk.CTkOptionMenu(
            row2, variable=self.alt_proto_col, values=ALT_COL_PROTO_CHOICES,
            command=lambda _v: self.apply_alt_settings(),
            fg_color=theme.CARD_BG, button_color=theme.CARD_BG, button_hover_color=theme.CREAM_SOFT,
            text_color=theme.TEXT_PRIMARY, dropdown_fg_color=theme.CARD_BG,
            dropdown_text_color=theme.TEXT_PRIMARY, dropdown_hover_color=theme.CREAM,
            corner_radius=theme.RADIUS_INPUT, height=32, font=theme.F_BODY, width=200,
        )
        self.alt_proto_combo.pack(side="left")
        return card

    def _build_show_hide_card(self, parent) -> "Card":
        card = Card(parent, title="Show / Hide", icon="👁")
        body = card.body
        body.grid_columnconfigure(0, weight=1)
        items = [
            ("Prototype RTI", self.show_proto_var),
            ("Mini MPL RTI", self.show_mpl_var),
            ("RTI as lines (proto vs MPL)", self.show_rti_lines_var),
            ("SNR profile (proto, lines)", self.show_snr_var),
            ("Profile view", self.show_profile_var),
            ("ALT overlay (white line)", self.show_alt_overlay_var),
            ("ALT Compare (time series)", self.show_alt_compare_var),
            ("Cloud markers (on RTI)", self.show_cloud_markers_var),
            ("Cloud layers panel", self.show_cloud_panel_var),
            ("Show date (X-axis + title)", self.rti_show_date_var),
            ("Show graph title (off for paper)", self.rti_show_title_var),
        ]
        for i, (txt, var) in enumerate(items):
            cb = ctk.CTkCheckBox(
                body, text=txt, variable=var, command=self.refresh_all,
                font=theme.F_SMALL, text_color=theme.TEXT_PRIMARY,
                fg_color=theme.ORANGE, hover_color=theme.ORANGE_HOVER,
                border_color=theme.BORDER, checkmark_color="#fff",
                corner_radius=4, border_width=2,
                onvalue=True, offvalue=False,
            )
            cb.grid(row=i, column=0, sticky="w", pady=2)
            # Sync visual state to BooleanVar initial value
            cb.select() if bool(var.get()) else cb.deselect()

        # Channel selectors (co / cross / δ — populated from the loaded workbook)
        row = len(items)

        def _src_menu(label, var, cmd):
            nonlocal row
            ctk.CTkLabel(body, text=label, font=theme.F_SMALL,
                         text_color=theme.TEXT_SECONDARY, anchor="w").grid(
                row=row, column=0, sticky="w", pady=(8, 0)); row += 1
            m = ctk.CTkOptionMenu(
                body, variable=var, values=["NRB parallel"], command=lambda _v: cmd(),
                fg_color=theme.CARD_BG, button_color=theme.CARD_BG, button_hover_color=theme.CREAM_SOFT,
                text_color=theme.TEXT_PRIMARY, dropdown_fg_color=theme.CARD_BG,
                dropdown_text_color=theme.TEXT_PRIMARY, dropdown_hover_color=theme.CREAM,
                corner_radius=theme.RADIUS_INPUT, height=30, font=theme.F_SMALL, width=220)
            m.grid(row=row, column=0, sticky="w", pady=(0, 2)); row += 1
            return m

        self._proto_src_menu = _src_menu("Prototype channel (RTI + profile)",
                                         self.proto_rti_source, self._apply_proto_source)
        self._mpl_src_menu = _src_menu("Mini MPL channel (RTI + profile)",
                                       self.mpl_rti_source, self._apply_mpl_source)
        self._prof_src_menu = _src_menu("Profile source (dataset)",
                                        self.profile_source, self._apply_profile_source)
        self._prof_ch_menu = _src_menu("Profile channels",
                                       self.profile_channels, self._apply_profile_source)
        self._prof_ch_menu.configure(values=["Selected channel", "Parallel + Perpendicular"])
        self._refresh_source_menus()
        return card

    def _build_time_card(self, parent) -> "Card":
        card = Card(parent, title="Profile Time", icon="🕐")
        body = card.body
        body.grid_columnconfigure(0, weight=1)

        # Listbox in styled wrap
        list_wrap = ctk.CTkFrame(
            body, fg_color=theme.CREAM_SOFT, corner_radius=theme.RADIUS_INPUT,
            border_width=1, border_color=theme.BORDER, height=160,
        )
        list_wrap.grid(row=0, column=0, sticky="ew")
        list_wrap.grid_columnconfigure(0, weight=1)
        list_wrap.grid_rowconfigure(0, weight=1)
        list_wrap.grid_propagate(False)
        self.time_list = tk.Listbox(
            list_wrap, height=8, selectmode=tk.EXTENDED, exportselection=False,
            bg=theme.CREAM_SOFT, fg=theme.TEXT_PRIMARY,
            selectbackground=theme.ORANGE_SOFT, selectforeground=theme.ORANGE,
            activestyle="none", relief="flat", bd=0, highlightthickness=0,
            font=(theme.FONT_FAMILY, 11),
        )
        self.time_list.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.time_list.bind("<<ListboxSelect>>", lambda _: self.update_profile_plot(True))

        # Display button — redraw EVERYTHING for the current selection (RTI time
        # cursor + profile overlay + ALT markers), right here so there's no need
        # to scroll. Selecting a time already auto-updates the profile preview;
        # this forces the full render.
        ctk.CTkButton(
            body, text="🔄  Display selected", command=self.refresh_all,
            **theme.primary_button_style(height=36, font=theme.F_SMALL),
        ).grid(row=1, column=0, sticky="ew", pady=(8, 0))

        # Raw / Denoised radio
        rad = ctk.CTkFrame(body, fg_color="transparent")
        rad.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ctk.CTkLabel(rad, text="Proto profile:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).pack(side="left", padx=(0, 8))
        radio_kw = dict(
            font=theme.F_SMALL, text_color=theme.TEXT_PRIMARY,
            fg_color=theme.ORANGE, hover_color=theme.ORANGE_HOVER,
            border_color=theme.BORDER, command=self.refresh_all,
        )
        ctk.CTkRadioButton(rad, text="Raw", value="raw",
                           variable=self.profile_proto_mode_var, **radio_kw).pack(side="left", padx=(0, 8))
        ctk.CTkRadioButton(rad, text="Denoised", value="denoised",
                           variable=self.profile_proto_mode_var, **radio_kw).pack(side="left")
        return card

    def _build_save_card(self, parent) -> "Card":
        card = Card(parent, title="Save", icon="💾")
        body = card.body
        body.grid_columnconfigure(0, weight=1)

        # Two action buttons side-by-side
        actions = ctk.CTkFrame(body, fg_color="transparent")
        actions.grid(row=0, column=0, sticky="ew")
        for i in range(2):
            actions.grid_columnconfigure(i, weight=1)
        ctk.CTkButton(
            actions, text="Save Figure PNG", command=self.save_png,
            **theme.primary_button_style(height=36, font=theme.F_SMALL),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ctk.CTkButton(
            actions, text="Save Profile PNG", command=self.save_profile_png,
            **theme.secondary_button_style(height=36, font=theme.F_SMALL),
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ctk.CTkButton(
            body, text="Save all 4 (Proto RTI · MPL RTI · ALT compare · RTI pair)",
            command=self.save_all_figures,
            **theme.primary_button_style(height=36, font=theme.F_SMALL),
        ).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ctk.CTkButton(
            body, text="Save MPL set (co/cross raw + NRB profile, par & perp)",
            command=self.save_mpl_set,
            **theme.secondary_button_style(height=36, font=theme.F_SMALL),
        ).grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ctk.CTkButton(
            body, text="Save prototype set (Analog / Photon / Glue / NRB, par & perp)",
            command=self.save_proto_set,
            **theme.secondary_button_style(height=36, font=theme.F_SMALL),
        ).grid(row=3, column=0, sticky="ew", pady=(6, 0))
        ctk.CTkButton(
            body, text="Save NRB compare (Prototype vs MPL · par & perp · 2 per time)",
            command=self.save_nrb_compare_set,
            **theme.secondary_button_style(height=36, font=theme.F_SMALL),
        ).grid(row=4, column=0, sticky="ew", pady=(6, 0))
        ctk.CTkButton(
            body, text="⭐ Save ALL → picture-<date> folder (summary + all sets)",
            command=self.save_all_organized,
            **theme.primary_button_style(height=40, font=theme.F_SMALL),
        ).grid(row=5, column=0, sticky="ew", pady=(10, 0))
        ctk.CTkCheckBox(
            body, text="CMYK for print (saves .tif)  — off = RGB .png",
            variable=self.cmyk_var,
            font=theme.F_SMALL, text_color=theme.TEXT_PRIMARY,
            fg_color=theme.ORANGE, hover_color=theme.ORANGE_HOVER,
            border_color=theme.BORDER, checkmark_color="#fff",
            corner_radius=4, border_width=2,
        ).grid(row=6, column=0, sticky="w", pady=(8, 0))

        # Hint
        ctk.CTkLabel(
            body, text=("Save ALL builds picture-<date>/ with 'result summary' (4 whole-day "
                        "figures) + prototype/MPL/NRB sets for every profile time. "
                        "Pick size & DPI once. Golden-ratio presets in the size list."),
            font=theme.F_TINY, text_color=theme.TEXT_MUTED, anchor="w",
            wraplength=440, justify="left",
        ).grid(row=7, column=0, sticky="w", pady=(6, 0))
        return card

    def _build_display_card(self, parent) -> "Card":
        card = Card(parent, title="Display", icon="🎨")
        body = card.body
        body.grid_columnconfigure(0, weight=1)

        # CMap + Interp row
        d0 = ctk.CTkFrame(body, fg_color="transparent")
        d0.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        d0.grid_columnconfigure(1, weight=1)
        d0.grid_columnconfigure(3, weight=1)
        ctk.CTkLabel(d0, text="CMap", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 4))
        ctk.CTkOptionMenu(
            d0, variable=self.cmap_var, values=CMAP_CHOICES, width=110,
            fg_color=theme.CARD_BG, button_color=theme.CARD_BG, button_hover_color=theme.CREAM_SOFT,
            text_color=theme.TEXT_PRIMARY, dropdown_fg_color=theme.CARD_BG,
            dropdown_text_color=theme.TEXT_PRIMARY, dropdown_hover_color=theme.CREAM,
            corner_radius=theme.RADIUS_INPUT, height=30, font=theme.F_SMALL,
        ).grid(row=0, column=1, sticky="w", padx=(0, 14))
        ctk.CTkLabel(d0, text="Interp", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=2, padx=(0, 4))
        ctk.CTkOptionMenu(
            d0, variable=self.interp_var, values=INTERP_CHOICES, width=110,
            fg_color=theme.CARD_BG, button_color=theme.CARD_BG, button_hover_color=theme.CREAM_SOFT,
            text_color=theme.TEXT_PRIMARY, dropdown_fg_color=theme.CARD_BG,
            dropdown_text_color=theme.TEXT_PRIMARY, dropdown_hover_color=theme.CREAM,
            corner_radius=theme.RADIUS_INPUT, height=30, font=theme.F_SMALL,
        ).grid(row=0, column=3, sticky="w")

        # Discrete + step + log color
        d1 = ctk.CTkFrame(body, fg_color="transparent")
        d1.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        ctk.CTkCheckBox(
            d1, text="Discrete", variable=self.discrete_var,
            font=theme.F_SMALL, text_color=theme.TEXT_PRIMARY,
            fg_color=theme.ORANGE, hover_color=theme.ORANGE_HOVER,
            border_color=theme.BORDER, checkmark_color="#fff",
            corner_radius=4, border_width=2,
        ).pack(side="left")
        ctk.CTkLabel(d1, text="Step", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).pack(side="left", padx=(8, 4))
        ctk.CTkEntry(d1, textvariable=self.step_var, width=70,
                     **theme.input_style()).pack(side="left", padx=(0, 10))
        ctk.CTkCheckBox(
            d1, text="Log colour", variable=self.use_log_color_var,
            font=theme.F_SMALL, text_color=theme.TEXT_PRIMARY,
            fg_color=theme.ORANGE, hover_color=theme.ORANGE_HOVER,
            border_color=theme.BORDER, checkmark_color="#fff",
            corner_radius=4, border_width=2,
        ).pack(side="left")

        # vmin/vmax sliders
        d2 = ctk.CTkFrame(body, fg_color="transparent")
        d2.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        d2.grid_columnconfigure(1, weight=1)
        for i, (lbl, var) in enumerate([("vmin", self.vmin_var), ("vmax", self.vmax_var)]):
            ctk.CTkLabel(d2, text=lbl, font=theme.F_SMALL,
                         text_color=theme.TEXT_SECONDARY).grid(row=i, column=0, sticky="w", padx=(0, 6))
            ctk.CTkSlider(
                d2, variable=var, from_=-1, to=1, number_of_steps=200,
                command=lambda _v=None: self._on_manual_scale(),
                progress_color=theme.ORANGE, button_color=theme.ORANGE,
                button_hover_color=theme.ORANGE_HOVER, fg_color=theme.LIGHT_GRAY,
                height=14,
            ).grid(row=i, column=1, sticky="ew", padx=(0, 4))
            ctk.CTkLabel(d2, textvariable=var, font=theme.F_TINY,
                         text_color=theme.TEXT_MUTED, width=50,
                         anchor="e").grid(row=i, column=2, sticky="e")

        # Scale mode radio
        d3 = ctk.CTkFrame(body, fg_color="transparent")
        d3.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        ctk.CTkLabel(d3, text="Scale:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).pack(side="left", padx=(0, 6))
        radio_kw = dict(
            font=theme.F_SMALL, text_color=theme.TEXT_PRIMARY,
            fg_color=theme.ORANGE, hover_color=theme.ORANGE_HOVER,
            border_color=theme.BORDER,
        )
        for val, txt in [("preset", "0–1"), ("auto", "Auto"), ("manual", "Manual")]:
            ctk.CTkRadioButton(
                d3, text=txt, value=val, variable=self.scale_mode_var, **radio_kw,
            ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            body, text="Apply Display", command=self.apply_display,
            **theme.primary_button_style(height=34, font=theme.F_SMALL),
        ).grid(row=4, column=0, sticky="w", pady=(4, 0))
        return card

    def _build_axis_card(self, parent) -> "Card":
        card = Card(parent, title="Axis Settings", icon="📐")
        body = card.body
        body.grid_columnconfigure(0, weight=1)

        def row_pair(parent_, items, ridx):
            f = ctk.CTkFrame(parent_, fg_color="transparent")
            f.grid(row=ridx, column=0, sticky="ew", pady=2)
            for i in range(len(items)):
                f.grid_columnconfigure(i, weight=1)
            for ci, (lbl, var) in enumerate(items):
                FieldRow(f, lbl, var).grid(row=0, column=ci, sticky="ew",
                                           padx=(0 if ci == 0 else 4, 0))
            return f

        row_pair(body, [("RTI tickN", self.rti_tickN_var), ("RTI ytick", self.rti_ytick_var)], 0)
        row_pair(body, [("RTI ymin", self.rti_ymin_var), ("RTI ymax", self.rti_ymax_var)], 1)
        row_pair(body, [("Prof xmin", self.prof_xmin_var), ("Prof xmax", self.prof_xmax_var)], 2)
        row_pair(body, [("Prof ymin", self.prof_ymin_var), ("Prof ymax", self.prof_ymax_var)], 3)
        row_pair(body, [("Prof ytick", self.prof_ytick_var), ("Log base", self.log_base_var)], 4)

        ctk.CTkCheckBox(
            body, text="Log NRB (Profile Y axis)", variable=self.log_nrb_axis_var,
            font=theme.F_SMALL, text_color=theme.TEXT_PRIMARY,
            fg_color=theme.ORANGE, hover_color=theme.ORANGE_HOVER,
            border_color=theme.BORDER, checkmark_color="#fff",
            corner_radius=4, border_width=2,
        ).grid(row=5, column=0, sticky="w", pady=(4, 4))

        row_pair(body, [("ALT tickN", self.alt_tickN_var), ("ALT ytick", self.alt_ytick_var)], 6)
        row_pair(body, [("ALT ymin", self.alt_ymin_var), ("ALT ymax", self.alt_ymax_var)], 7)

        ctk.CTkButton(
            body, text="Apply Axis", command=self.apply_axis,
            **theme.primary_button_style(height=34, font=theme.F_SMALL),
        ).grid(row=8, column=0, sticky="w", pady=(6, 0))
        return card

    def _build_titles_card(self, parent) -> "Card":
        card = Card(parent, title="Titles & Labels", icon="✏️")
        body = card.body
        body.grid_columnconfigure(0, weight=1)

        def trow(items, ridx):
            f = ctk.CTkFrame(body, fg_color="transparent")
            f.grid(row=ridx, column=0, sticky="ew", pady=2)
            for i in range(len(items)):
                f.grid_columnconfigure(i, weight=1)
            for ci, (lbl, var) in enumerate(items):
                FieldRow(f, lbl, var).grid(row=0, column=ci, sticky="ew",
                                           padx=(0 if ci == 0 else 4, 0))

        trow([("RTI Prototype", self.title_rti_proto_var), ("RTI MPL", self.title_rti_mpl_var)], 0)
        trow([("Profile", self.title_prof_var), ("ALT Compare", self.title_altcmp_var)], 1)
        trow([("RTI X label", self.rti_xlabel_var), ("RTI Y label", self.rti_ylabel_var)], 2)
        trow([("Prof X label", self.prof_xlabel_var), ("Prof Y label", self.prof_ylabel_var)], 3)
        trow([("ALT X label", self.alt_xlabel_var), ("ALT Y label", self.alt_ylabel_var)], 4)
        trow([("CBar label", self.cbar_label_var)], 5)
        # Editable legend names for the Profile view (blank = auto). Time is kept.
        trow([("Profile legend (proto)", self.proto_legend_prefix),
              ("Profile legend (MPL)", self.mpl_legend_prefix)], 6)
        # Editable legend names for the ALT-compare panel (blank = auto).
        trow([("ALT legend (proto)", self.alt_proto_legend),
              ("ALT legend (MPL)", self.alt_mpl_legend)], 7)

        ctk.CTkButton(
            body, text="Apply Titles", command=self.apply_titles,
            **theme.primary_button_style(height=34, font=theme.F_SMALL),
        ).grid(row=8, column=0, sticky="w", pady=(6, 0))
        return card

    def _build_line_style_card(self, parent) -> "Card":
        """Excel-style 'Format Data Series' for the Profile + ALT-compare lines."""
        card = Card(parent, title="Line Style", icon="✏️")
        body = card.body
        body.grid_columnconfigure(0, weight=1)

        def _series_row(r, title, wvar, cvar, dvar):
            ctk.CTkLabel(body, text=title, font=(theme.FONT_FAMILY, 10, "bold"),
                         text_color=theme.TEXT_MUTED, anchor="w").grid(
                row=r, column=0, sticky="w", pady=(6 if r else 0, 2))
            rf = ctk.CTkFrame(body, fg_color="transparent")
            rf.grid(row=r + 1, column=0, sticky="ew")
            ctk.CTkLabel(rf, text="Width", font=theme.F_SMALL,
                         text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 2))
            ctk.CTkEntry(rf, textvariable=wvar, width=46, **theme.input_style()).grid(row=0, column=1, padx=(0, 8))
            ctk.CTkLabel(rf, text="Color", font=theme.F_SMALL,
                         text_color=theme.TEXT_SECONDARY).grid(row=0, column=2, padx=(0, 2))
            ctk.CTkEntry(rf, textvariable=cvar, width=76, **theme.input_style()).grid(row=0, column=3, padx=(0, 2))
            ctk.CTkButton(rf, text="🎨", width=30, command=lambda v=cvar: self._pick_color(v),
                          **theme.secondary_button_style(height=30, font=theme.F_SMALL)).grid(row=0, column=4, padx=(0, 8))
            ctk.CTkOptionMenu(
                rf, variable=dvar, values=self.LINE_DASH_CHOICES, width=96,
                command=lambda _v: self.refresh_all(),
                fg_color=theme.CARD_BG, button_color=theme.CARD_BG, button_hover_color=theme.CREAM_SOFT,
                text_color=theme.TEXT_PRIMARY, dropdown_fg_color=theme.CARD_BG,
                dropdown_text_color=theme.TEXT_PRIMARY, dropdown_hover_color=theme.CREAM,
                corner_radius=theme.RADIUS_INPUT, height=30, font=theme.F_SMALL,
            ).grid(row=0, column=5)

        _series_row(0, "Prototype", self.proto_line_width, self.proto_line_color, self.proto_line_dash)
        _series_row(2, "Mini MPL", self.mpl_line_width, self.mpl_line_color, self.mpl_line_dash)
        ctk.CTkLabel(body, text="Applies to Profile + ALT-compare. Color blank = auto per-time colours.",
                     font=theme.F_TINY, text_color=theme.TEXT_MUTED, anchor="w",
                     wraplength=440, justify="left").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ctk.CTkButton(body, text="Apply Line Style", command=self.refresh_all,
                      **theme.primary_button_style(height=34, font=theme.F_SMALL)).grid(
            row=5, column=0, sticky="w", pady=(6, 0))
        return card

    def _build_font_card(self, parent) -> "Card":
        """Font toolbar for ALL Display axes — family, per-element size, bold."""
        card = Card(parent, title="Fonts", icon="🔤")
        body = card.body
        body.grid_columnconfigure(0, weight=1)

        fr = ctk.CTkFrame(body, fg_color="transparent")
        fr.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ctk.CTkLabel(fr, text="Font", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 4))
        ctk.CTkOptionMenu(
            fr, variable=self.font_family, width=170,
            values=["Times New Roman", "Arial", "Calibri", "Tahoma",
                    "DejaVu Sans", "DejaVu Serif"],
            command=lambda _v: self.refresh_all(),
            fg_color=theme.CARD_BG, button_color=theme.CARD_BG, button_hover_color=theme.CREAM_SOFT,
            text_color=theme.TEXT_PRIMARY, dropdown_fg_color=theme.CARD_BG,
            dropdown_text_color=theme.TEXT_PRIMARY, dropdown_hover_color=theme.CREAM,
            corner_radius=theme.RADIUS_INPUT, height=30, font=theme.F_SMALL,
        ).grid(row=0, column=1, padx=(0, 8))
        ctk.CTkCheckBox(
            fr, text="Bold", variable=self.font_bold, command=self.refresh_all,
            font=theme.F_SMALL, text_color=theme.TEXT_PRIMARY,
            fg_color=theme.ORANGE, hover_color=theme.ORANGE_HOVER,
            border_color=theme.BORDER, checkmark_color="#fff",
            corner_radius=4, border_width=2,
        ).grid(row=0, column=2)

        sizes = ctk.CTkFrame(body, fg_color="transparent")
        sizes.grid(row=1, column=0, sticky="ew")
        for i in range(4):
            sizes.grid_columnconfigure(i, weight=1)
        for ci, (lbl, var) in enumerate([
                ("Title", self.font_title_size), ("Label", self.font_label_size),
                ("Ticks", self.font_tick_size), ("Legend", self.font_legend_size)]):
            cell = ctk.CTkFrame(sizes, fg_color="transparent")
            cell.grid(row=0, column=ci, sticky="ew", padx=(0 if ci == 0 else 4, 0))
            ctk.CTkLabel(cell, text=lbl, font=theme.F_TINY,
                         text_color=theme.TEXT_SECONDARY, anchor="w").grid(row=0, column=0, sticky="w")
            ctk.CTkEntry(cell, textvariable=var, **theme.input_style()).grid(row=1, column=0, sticky="ew")

        ctk.CTkButton(body, text="Apply Fonts", command=self.refresh_all,
                      **theme.primary_button_style(height=34, font=theme.F_SMALL)).grid(
            row=2, column=0, sticky="w", pady=(8, 0))
        return card

    def _pick_color(self, var):
        from tkinter import colorchooser
        try:
            c = colorchooser.askcolor(color=(var.get().strip() or "#FF5632"),
                                      title="Line colour")
        except Exception:
            return
        if c and c[1]:
            var.set(c[1])
            self.refresh_all()

    # ── Status helper (sidebar status removed; reuse plot status) ──────────
    @property
    def status_var(self):
        return self.status.label.cget("text")

    def _set_status(self, msg, tone="info"):
        self.status.set(msg, tone)

    # ────────────────────────────────────────────────────────────────────────
    # The rest of the logic mirrors main.py Step4Frame 1:1.
    # ────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_date_from_filename(fname: str) -> Optional[pd.Timestamp]:
        """Pull a date out of common filename patterns (YYYY-MM-DD or YYYYMMDD)."""
        import re
        name = os.path.basename(str(fname))
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", name)
        if m:
            try:
                return pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
            except Exception:
                pass
        m = re.search(r"(\d{8})", name)
        if m:
            try:
                return pd.Timestamp(m.group(1))
            except Exception:
                pass
        return None

    def _read_wide(self, path, sheet, base_date=None):
        df = pd.read_excel(path, sheet_name=sheet)
        rc = df.columns[0]; tc = df.columns[1:]
        r = pd.to_numeric(df[rc], errors="coerce").to_numpy(float)
        Z = df[tc].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        base = base_date or (_r4_first_ts(tc) or pd.Timestamp.today())
        t = _r4_parse_tcols(tc, base)
        return r, t, Z

    def _merge_nrb_files(self, file_list, sheet, fallback_base_date=None):
        if not file_list:
            raise ValueError("No files provided.")
        # When time columns lack date info (e.g. MPL HH:MM headers), infer a
        # base date per file from its filename so consecutive days don't collide.
        # If the filename has no parseable date, use `fallback_base_date`
        # (typically the prototype date) instead of today().
        parts = []
        for fp in file_list:
            base_date = self._parse_date_from_filename(fp) or fallback_base_date
            parts.append(self._read_wide(fp, sheet, base_date=base_date))
        r_common = parts[0][0]
        nR = int(len(r_common))
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
        t_anchor = records[0][0]
        t_last = records[-1][0]
        n_slots = int(np.ceil((t_last - t_anchor).total_seconds() / 60.0 / float(interval_min))) + 1
        expected = [t_anchor + pd.Timedelta(minutes=interval_min * k) for k in range(max(1, n_slots))]
        Z_out = np.full((nR, len(expected)), np.nan, dtype=float)
        expected_ns = np.array([e.value for e in expected], dtype=np.int64)
        tol_ns = int(interval_min * 30 * 1e9)
        # The timeline carries the ACTUAL timestamp wherever a profile lands.
        # A day whose cadence is phase-shifted from the grid (e.g. 04-21 at
        # :X5 vs a grid anchored at :X0) would otherwise be reported at the
        # grid time, throwing ALT/MPL time-alignment off by several minutes.
        t_axis = list(expected)
        for ts, col in records:
            idx = int(np.argmin(np.abs(expected_ns - ts.value)))
            if abs(expected_ns[idx] - ts.value) <= tol_ns:
                Z_out[:, idx] = col
                t_axis[idx] = ts
        return r_common, [pd.Timestamp(e) for e in t_axis], Z_out

    def _load_prototype_files(self, file_list):
        try:
            file_list = list(file_list)
            r, t, Z = self._merge_nrb_files(file_list, PROTO_SHEET)
            # Trim phantom all-NaN columns at the head/tail. A phase-mismatched
            # multi-day merge overshoots the fixed-interval grid by one slot
            # (e.g. an empty 06:40 slot after the last real 06:35 profile), which
            # renders as a black set_bad column. Interior all-NaN columns are kept
            # (genuine data gaps). The SAME [lo:hi] slice is applied to every
            # channel matrix below so all sheets stay column-aligned with t_proto.
            _nT0 = int(np.asarray(Z, float).shape[1]) if np.asarray(Z, float).ndim == 2 else len(t)
            _col_ok = np.isfinite(np.asarray(Z, float)).any(axis=0)
            if _col_ok.any():
                _lo = int(np.argmax(_col_ok))
                _hi = int(_col_ok.size - np.argmax(_col_ok[::-1]))
            else:
                _lo, _hi = 0, _nT0
            if (_lo, _hi) != (0, _nT0):
                t = list(t)[_lo:_hi]
                Z = np.asarray(Z, float)[:, _lo:_hi]
            base = next((x for x in t if not pd.isna(x)), pd.Timestamp.today())
            self.proto_date = pd.Timestamp(base)
            self.date_str.set(self.proto_date.strftime("%Y-%m-%d"))
            lhhmm = _r4_build_x_labels(t)
            llist = [pd.Timestamp(x).strftime("%Y-%m-%d %H:%M") if not pd.isna(x) else "NaT" for x in t]
            self.proto_path = file_list[0] if len(file_list) == 1 else f"{len(file_list)} files merged"
            self.r_proto = r; self.t_proto = t
            # Base co channel + any extra channels from a depol workbook.
            self._proto_mats = {"NRB parallel": Z}
            # co/cross each in normalised (÷ own max, 0-1) + real (physical) form;
            # δ_v = depolarization ratio.
            for label, sheet in [("NRB parallel (real)", "NRB_co"),
                                 ("NRB perpendicular", "NRB_cross_norm"),
                                 ("NRB perpendicular (real)", "NRB_cross"),
                                 ("δ_v", "Depol_delta_v")]:
                try:
                    if any(sheet in pd.ExcelFile(fp).sheet_names for fp in file_list):
                        _, _, Zx = self._merge_nrb_files(file_list, sheet)
                        Zx = np.asarray(Zx, float)
                        if Zx.ndim == 2 and Zx.shape[1] == _nT0 and (_lo, _hi) != (0, _nT0):
                            Zx = Zx[:, _lo:_hi]
                        self._proto_mats[label] = Zx
                except Exception:
                    pass
            # Per-channel signal sheets (analog/photon/glue) for the prototype
            # figure set — written by Step 3. Missing sheets are simply skipped.
            self._proto_sig = {}
            for sheet in ("Analog_par", "Analog_perp", "Photon_par", "Photon_perp",
                          "PhotonDT_par", "PhotonDT_perp", "AnalogScaled_par",
                          "AnalogScaled_perp", "Glued_par", "Glued_perp"):
                try:
                    if any(sheet in pd.ExcelFile(fp).sheet_names for fp in file_list):
                        _, _, Zx = self._merge_nrb_files(file_list, sheet)
                        Zx = np.asarray(Zx, float)
                        if Zx.ndim == 2 and Zx.shape[1] == _nT0 and (_lo, _hi) != (0, _nT0):
                            Zx = Zx[:, _lo:_hi]
                        self._proto_sig[sheet] = Zx
                except Exception:
                    pass
            # SNR curtain matrix (parallel/co photon SNR) — same [lo:hi] slice.
            self.Z_proto_snr = None
            try:
                if any("SNR" in pd.ExcelFile(fp).sheet_names for fp in file_list):
                    _, _, Zsnr = self._merge_nrb_files(file_list, "SNR")
                    Zsnr = np.asarray(Zsnr, float)
                    if Zsnr.ndim == 2 and Zsnr.shape[1] == _nT0 and (_lo, _hi) != (0, _nT0):
                        Zsnr = Zsnr[:, _lo:_hi]
                    self.Z_proto_snr = Zsnr
            except Exception:
                pass
            if self.proto_rti_source.get() not in self._proto_mats:
                self.proto_rti_source.set("NRB parallel")
            self.Z_proto = self._proto_mats.get(self.proto_rti_source.get(), Z)
            self._refresh_source_menus()
            self.t_labels_hhmm = lhhmm; self.t_list_labels = llist
            self.time_list.delete(0, tk.END)
            for lab in llist:
                self.time_list.insert(tk.END, lab)
            if llist:
                self.time_list.selection_set(0)
            self._rebuild_denoised(); self._align_alt()
            if len(file_list) == 1:
                self._set_status(f"Prototype: {os.path.basename(file_list[0])}", "ok")
            else:
                span = f"{lhhmm[0]} → {lhhmm[-1]}" if lhhmm else ""
                self._set_status(f"{len(file_list)} files merged ({span})", "ok")
            self.align_and_refresh()
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def load_prototype(self):
        fps = filedialog.askopenfilenames(
            title="Prototype NRB Excel (Ctrl+click for multiple days)",
            filetypes=[("Excel", "*.xlsx *.xls")],
        )
        if not fps:
            return
        self._load_prototype_files(fps)

    def load_prototype_folder(self):
        folder = filedialog.askdirectory(title="Folder containing NRB-*.xlsx files")
        if not folder:
            return
        files = sorted(Path(folder).glob("NRB-*.xlsx"))
        if not files:
            files = sorted(Path(folder).glob("*.xlsx"))
        if not files:
            messagebox.showerror("No files", f"No .xlsx files found in:\n{folder}")
            return
        self._load_prototype_files([str(f) for f in files])

    def load_mpl(self):
        fps = filedialog.askopenfilenames(
            title="MPL Excel (Ctrl+click for multiple days)",
            filetypes=[("Excel", "*.xlsx *.xls")],
        )
        if not fps:
            return
        self._load_mpl_files(list(fps))

    def _load_mpl_files(self, file_list):
        """Load 1+ MPL files. Multi-file inputs are merged by timeline.

        MPL `copol_nrb_norm` columns are HH:MM only (no date). The base date is
        taken from the filename when possible, otherwise falls back to the
        prototype's date so alignment with the prototype timeline still works.
        """
        try:
            file_list = list(file_list)
            r, t, Z = self._merge_nrb_files(
                file_list, MPL_SHEET, fallback_base_date=self.proto_date,
            )
            # Trim phantom all-NaN head/tail columns (same phase-mismatch grid
            # overshoot as the prototype loader); one [lo:hi] slice for every
            # MPL channel so all sheets stay column-aligned with t_mpl.
            _nT0 = int(np.asarray(Z, float).shape[1]) if np.asarray(Z, float).ndim == 2 else len(t)
            _col_ok = np.isfinite(np.asarray(Z, float)).any(axis=0)
            if _col_ok.any():
                _lo = int(np.argmax(_col_ok))
                _hi = int(_col_ok.size - np.argmax(_col_ok[::-1]))
            else:
                _lo, _hi = 0, _nT0
            if (_lo, _hi) != (0, _nT0):
                t = list(t)[_lo:_hi]
                Z = np.asarray(Z, float)[:, _lo:_hi]
            self.r_mpl = r; self.t_mpl = t
            self._mpl_mats = {"NRB parallel": Z}   # copol_nrb_norm (normalised)

            def _load_mpl_sheet(sheet):
                try:
                    if any(sheet in pd.ExcelFile(fp).sheet_names for fp in file_list):
                        _, _, Zx = self._merge_nrb_files(
                            file_list, sheet, fallback_base_date=self.proto_date)
                        Zx = np.asarray(Zx, float)
                        if Zx.ndim == 2 and Zx.shape[1] == _nT0 and (_lo, _hi) != (0, _nT0):
                            Zx = Zx[:, _lo:_hi]
                        return Zx
                except Exception:
                    pass
                return None

            Zco_real = _load_mpl_sheet("MPL_copol_nrb")
            if Zco_real is not None:
                self._mpl_mats["NRB parallel (real)"] = Zco_real
            Zcr = _load_mpl_sheet("MPL_crosspol_nrb")
            if Zcr is not None:
                mx = float(np.nanmax(Zcr)) if np.any(np.isfinite(Zcr)) else np.nan
                if np.isfinite(mx) and mx > 0:
                    self._mpl_mats["NRB perpendicular"] = Zcr / mx   # ÷ own max
                self._mpl_mats["NRB perpendicular (real)"] = Zcr     # physical
            # Raw co/cross counts (physical) — for the MPL figure set.
            Zrp = _load_mpl_sheet("MPL_copol_raw")
            if Zrp is not None:
                self._mpl_mats["raw parallel"] = Zrp
            Zrc = _load_mpl_sheet("MPL_crosspol_raw")
            if Zrc is not None:
                self._mpl_mats["raw perpendicular"] = Zrc
            if self.mpl_rti_source.get() not in self._mpl_mats:
                self.mpl_rti_source.set("NRB parallel")
            self.Z_mpl = self._mpl_mats.get(self.mpl_rti_source.get(), Z)
            self._refresh_source_menus()
            if len(file_list) == 1:
                self.mpl_path = file_list[0]
                self._set_status(f"MPL: {os.path.basename(file_list[0])}", "ok")
            else:
                self.mpl_path = f"{len(file_list)} files merged"
                self._set_status(f"MPL: {len(file_list)} files merged", "ok")
            self.align_and_refresh()
            # Warn if alignment produced zero matches (date mismatch symptom)
            if self.match_idx is not None:
                n_ok = sum(x is not None for x in self.match_idx)
                if n_ok == 0:
                    messagebox.showwarning(
                        "MPL alignment",
                        "MPL loaded but 0 profiles matched the prototype timeline.\n\n"
                        "Likely a date mismatch. Make sure:\n"
                        "  • You loaded the Step-1 output Excel "
                        "(with 'copol_nrb_norm' sheet), not the raw MPL CSV.\n"
                        "  • The MPL file date matches the prototype date.\n"
                        "  • The 'Match ±' tolerance is wide enough.",
                    )
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def load_alt(self):
        fps = filedialog.askopenfilenames(
            title="ALT Excel (Ctrl+click for multiple days)",
            filetypes=[("Excel", "*.xlsx *.xls")],
        )
        if not fps:
            return
        self._load_alt_files(list(fps))

    def _load_alt_files(self, file_list):
        """Load 1+ ALT files. The ALT_results sheets (long format with Time
        column) are concatenated and de-duplicated by Time. NRB_Denoised_FFT
        sheets are merged via the same wide-file logic used for Prototype."""
        try:
            file_list = list(file_list)
            dfs: List[pd.DataFrame] = []
            successful_files: List[str] = []
            for fp in file_list:
                try:
                    xl = pd.ExcelFile(fp)
                    sheet = (
                        ALT_SHEET if ALT_SHEET in xl.sheet_names
                        else (LEGACY_ALT_SHEET if LEGACY_ALT_SHEET in xl.sheet_names
                              else xl.sheet_names[0])
                    )
                    df = pd.read_excel(fp, sheet_name=sheet)
                    if "Time" not in df.columns:
                        continue
                    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
                    df = df.dropna(subset=["Time"]).reset_index(drop=True)
                    if df.empty:
                        continue
                    dfs.append(df)
                    successful_files.append(fp)
                except Exception:
                    continue
            if not dfs:
                raise ValueError("No valid ALT data found in selected files.")

            df_merged = (
                pd.concat(dfs, ignore_index=True)
                  .sort_values("Time")
                  .drop_duplicates(subset=["Time"], keep="last")
                  .reset_index(drop=True)
            )
            self.alt_df = df_merged

            if len(successful_files) == 1:
                self.alt_path_var.set(successful_files[0])
            else:
                self.alt_path_var.set(f"{len(successful_files)} ALT files merged")

            combo_values = [c for c in ALT_COL_PROTO_CHOICES if c in df_merged.columns]
            if combo_values:
                self.alt_proto_combo.configure(values=combo_values)
                if self.alt_proto_col.get() not in combo_values:
                    self.alt_proto_col.set(combo_values[0])

            # Reset denoised buffers
            self.r_proto_den_src = self.t_proto_den_src = self.Z_proto_den_src = None
            self.Z_proto_den = None
            den_ok = False
            try:
                # Re-use the wide-format merger across multiple ALT files
                files_with_denoised: List[str] = []
                for fp in successful_files:
                    try:
                        if DENOISED_SHEET in pd.ExcelFile(fp).sheet_names:
                            files_with_denoised.append(fp)
                    except Exception:
                        pass
                if files_with_denoised:
                    r_d, t_d, Z_d = self._merge_nrb_files(files_with_denoised, DENOISED_SHEET)
                    self.r_proto_den_src = r_d
                    self.t_proto_den_src = t_d
                    self.Z_proto_den_src = Z_d
                    self._rebuild_denoised()
                    den_ok = self.Z_proto_den is not None
            except Exception:
                pass

            self._align_alt()

            # ── Load Cloud_results sheet (Track 3 output) if present ──────────
            self.cloud_df = None
            cloud_ok = False
            try:
                cloud_parts = []
                for fp in successful_files:
                    try:
                        if "Cloud_results" in pd.ExcelFile(fp).sheet_names:
                            cdf = pd.read_excel(fp, sheet_name="Cloud_results")
                            if "Time" in cdf.columns and not cdf.empty:
                                cdf["Time"] = pd.to_datetime(cdf["Time"], errors="coerce")
                                cdf = cdf.dropna(subset=["Time"])
                                cloud_parts.append(cdf)
                    except Exception:
                        pass
                if cloud_parts:
                    self.cloud_df = (
                        pd.concat(cloud_parts, ignore_index=True)
                          .sort_values(["Time", "layer_index"])
                          .reset_index(drop=True)
                    )
                    cloud_ok = True
            except Exception:
                self.cloud_df = None
            self._align_clouds()

            if len(successful_files) == 1:
                msg = f"ALT: {os.path.basename(successful_files[0])}"
            else:
                msg = f"ALT: {len(successful_files)} files merged ({len(df_merged)} rows)"
            if den_ok:
                msg += f" + {DENOISED_SHEET}"
            if cloud_ok:
                msg += f" + Cloud_results ({len(self.cloud_df)} layers)"
            self._set_status(msg, "ok")
            self.refresh_all()
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def _align_clouds(self):
        """Map Cloud_results rows onto the prototype time grid (per layer)."""
        self.cloud_base_by_slot = None
        self.cloud_top_by_slot = None
        self.cloud_max_layer = 0
        if self.cloud_df is None or self.t_proto is None or len(self.t_proto) == 0:
            return
        try:
            tol = float(self.max_dt_min_var.get() or "3")
        except Exception:
            tol = 3.0
        df = self.cloud_df
        if "Time" not in df.columns or df.empty:
            return
        times = df["Time"]
        nT = len(self.t_proto)
        max_layer = int(pd.to_numeric(df.get("layer_index", pd.Series([0])),
                                      errors="coerce").max() or 0)
        if max_layer < 1:
            return
        base_by = {k: np.full(nT, np.nan) for k in range(1, max_layer + 1)}
        top_by = {k: np.full(nT, np.nan) for k in range(1, max_layer + 1)}
        for i, ts in enumerate(self.t_proto):
            if pd.isna(ts):
                continue
            try:
                ts = pd.Timestamp(ts)
                dt = (times - ts).dt.total_seconds().abs() / 60.0
                if dt.isna().all():
                    continue
                near = dt <= tol
            except Exception:
                continue
            if not bool(near.any()):
                continue
            sub = df[near]
            for _, row in sub.iterrows():
                k = int(row.get("layer_index", 0))
                if 1 <= k <= max_layer:
                    base_by[k][i] = pd.to_numeric(row.get("base_m"), errors="coerce")
                    top_by[k][i] = pd.to_numeric(row.get("top_m"), errors="coerce")
        self.cloud_base_by_slot = base_by
        self.cloud_top_by_slot = top_by
        self.cloud_max_layer = max_layer

    def _rebuild_denoised(self):
        self.Z_proto_den = None
        if self.r_proto is None:
            return
        # Preferred: an ALT file supplied a denoised source (NRB_Denoised_FFT) →
        # resample it onto the prototype grid (exact match to what ALT computed).
        if self.Z_proto_den_src is not None:
            mx = _r4_int(self.max_dt_min_var.get()) or 3
            mi, _ = _r4_nearest(self.t_proto, self.t_proto_den_src, max_min=mx)
            nR = self.r_proto.size; nT = len(self.t_proto)
            Zd = np.full((nR, nT), np.nan)
            for i in range(nT):
                j = mi[i]
                if j is not None:
                    Zd[:, i] = _r4_interp(self.r_proto_den_src, self.Z_proto_den_src[:, j], self.r_proto)
            self.Z_proto_den = Zd
            return
        # Fallback: no ALT file loaded → denoise the parallel (co) NRB on the fly
        # with the SAME FFT low-pass ALT uses, so "Denoised" works straight from a
        # Step-2 / Depolarization NRB without having to run ALT first.
        if not _HAS_PBL_DENOISE:
            return
        base = None
        if getattr(self, "_proto_mats", None):
            base = self._proto_mats.get("NRB parallel")
        if base is None:
            base = self.Z_proto
        if base is None:
            return
        r = np.asarray(self.r_proto, float)
        nR = r.size
        nT = min(base.shape[1], len(self.t_proto))
        Zd = np.full((nR, len(self.t_proto)), np.nan)
        for i in range(nT):
            try:
                y_dn, _ = _pbl_denoise(r, np.asarray(base[:, i], float),
                                       _PBL_FC, _PBL_ORDER, _PBL_PAD)
                Zd[:, i] = y_dn
            except Exception:
                pass
        self.Z_proto_den = Zd

    def _align_alt(self):
        self.alt_proto_by_slot = self.alt_mpl_by_slot = None
        self.alt_feature_by_slot = None
        if self.alt_df is None or self.t_proto is None:
            return
        try:
            tol = float(self.max_dt_min_var.get() or "3")
        except Exception:
            tol = 3.0
        df = self.alt_df; times = df["Time"]
        cp = self.alt_proto_col.get().strip()
        if cp not in df.columns:
            for c in ALT_COL_PROTO_CHOICES:
                if c in df.columns:
                    cp = c; self.alt_proto_col.set(c); break
        cm = next((c for c in ALT_COL_MPL_CHOICES if c in df.columns), None)
        has_feat = "ALT_feature" in df.columns
        nT = len(self.t_proto)
        ap = np.full(nT, np.nan); am = np.full(nT, np.nan)
        af = np.empty(nT, dtype=object); af[:] = ""
        times = pd.to_datetime(times, errors="coerce")
        if times is None or len(times) == 0 or times.isna().all():
            self.alt_proto_by_slot = ap; self.alt_mpl_by_slot = am
            self.alt_feature_by_slot = af if has_feat else None
            return
        # Bridge a constant DATE mismatch: .dat files without a date in their
        # filename default to 1900-01-01, so an ALT workbook's Time can be dated
        # differently from the (re-dated) prototype even though the times-of-day
        # match. Shift the ALT times by the whole-day offset so they align.
        try:
            proto_valid = pd.to_datetime(pd.Series(list(self.t_proto)),
                                         errors="coerce").dropna()
            if len(proto_valid):
                offset = proto_valid.min().normalize() - times.dropna().min().normalize()
                if abs(offset.days) >= 1:
                    times = times + offset
        except Exception:
            pass
        for i, ts in enumerate(self.t_proto):
            if pd.isna(ts):
                continue
            try:
                ts = pd.Timestamp(ts)
                dt = (times - ts).dt.total_seconds().abs() / 60.0
                if dt.isna().all():
                    continue
                j = int(dt.idxmin())
            except Exception:
                continue
            if float(dt.loc[j]) <= tol:
                if cp in df.columns:
                    ap[i] = pd.to_numeric(df.loc[j, cp], errors="coerce")
                if cm:
                    am[i] = pd.to_numeric(df.loc[j, cm], errors="coerce")
                if has_feat:
                    af[i] = str(df.loc[j, "ALT_feature"])
        self.alt_proto_by_slot = ap; self.alt_mpl_by_slot = am
        self.alt_feature_by_slot = af if has_feat else None

    # δ-classified ALT feature → marker colour (from Step-4 ALT 'ALT_feature').
    ALT_FEATURE_COLORS = {
        "aerosol":    "#00BE7C",  # green — real aerosol top (trustworthy)
        "dust_smoke": "#FF9E1F",  # amber — depolarizing aerosol
        "cloud_ice":  "#FF3257",  # red — ICE cloud (NOT a real ALT)
        "unknown":    "#9AA0B5",  # grey — δ unavailable (e.g. above ~3 km)
    }

    def _draw_alt_feature_markers(self, ax, y_slot, label=False, zorder=5):
        """Overlay ALT points coloured by δ-classified ALT_feature so aerosol
        tops vs cloud-flagged edges are visible. No-op without depol feature
        data. Returns True if anything was drawn."""
        feat = self.alt_feature_by_slot
        if feat is None or y_slot is None:
            return False
        y = np.asarray(y_slot, float)
        n = min(len(y), len(feat))
        x = np.arange(n)
        drew = False
        for key, color in self.ALT_FEATURE_COLORS.items():
            m = np.array([str(feat[i]) == key for i in range(n)]) & np.isfinite(y[:n])
            if m.any():
                ax.scatter(x[m], y[:n][m], s=24, c=color, edgecolors="black",
                           linewidths=0.5, zorder=zorder,
                           label=(f"ALT: {key}" if label else None))
                drew = True
        return drew

    def align_and_refresh(self):
        if self.Z_proto is None or self.Z_mpl is None:
            self.refresh_all(); return
        mx = _r4_int(self.max_dt_min_var.get()) or 3
        self.match_idx, self.dt_min = _r4_nearest(self.t_proto, self.t_mpl, max_min=mx)
        nR = self.r_proto.size; nT = len(self.t_proto)
        Zm = np.full((nR, nT), np.nan)
        for i in range(nT):
            j = self.match_idx[i]
            if j is not None:
                Zm[:, i] = _r4_interp(self.r_mpl, self.Z_mpl[:, j], self.r_proto)
        self.Z_mpl_on_proto = Zm
        self._rebuild_denoised(); self._align_alt(); self._align_clouds()
        n_ok = sum(x is not None for x in self.match_idx)
        self._set_status(f"Aligned: {n_ok}/{nT} matched (±{mx} min)", "ok")
        self.refresh_all()

    def apply_display(self):
        self._apply_scale_mode(); self.refresh_all()

    def apply_axis(self): self.refresh_all()
    def apply_titles(self): self.refresh_all()

    _DASH_MAP = {"solid": "-", "dashed": "--", "dotted": ":", "dash-dot": "-."}
    LINE_DASH_CHOICES = ["solid", "dashed", "dotted", "dash-dot"]

    def _line_style(self, which):
        """Excel-style line kwargs (width, colour, dash) for the proto/MPL series.
        Colour blank → matplotlib auto (per-time colours in the profile)."""
        if which == "proto":
            w = _r4_float(self.proto_line_width.get()) or 2.0
            c = self.proto_line_color.get().strip()
            ls = self._DASH_MAP.get(self.proto_line_dash.get(), "-")
        else:
            w = _r4_float(self.mpl_line_width.get()) or 2.0
            c = self.mpl_line_color.get().strip()
            ls = self._DASH_MAP.get(self.mpl_line_dash.get(), "--")
        kw = dict(lw=w, linestyle=ls)
        if c:
            kw["color"] = c
        return kw

    @staticmethod
    def _yticks_with_floor(ymin, ymax, ytick):
        """Tick list = the ymin floor (e.g. 120) + every `ytick` multiple above it,
        so the axis bottom value is always labelled."""
        ticks = [ymin]
        if ytick and ytick > 0:
            import math
            t = (math.floor(ymin / ytick) + 1) * ytick
            while t <= ymax + 1e-6:
                if t > ymin:
                    ticks.append(t)
                t += ytick
        return ticks

    def _title_date_suffix(self):
        """'YYYY-MM-DD (HH:MM-HH:MM)' single day, else 'start -- end', from the
        loaded prototype timeline. Empty when nothing is loaded. When the
        'Show date on X-axis' checkbox is OFF, the date is dropped and only the
        HH:MM range is kept (so the title matches the time-only axis)."""
        if not self.t_proto:
            return ""
        ts = [pd.Timestamp(t) for t in self.t_proto if not pd.isna(t)]
        if not ts:
            return ""
        d0, d1 = ts[0], ts[-1]
        show_date = self.rti_show_date_var.get()
        if d0.date() == d1.date():
            if show_date:
                return f"{d0:%Y-%m-%d} ({d0:%H:%M}-{d1:%H:%M})"
            return f"{d0:%H:%M}-{d1:%H:%M}"
        if show_date:
            return f"{d0:%Y-%m-%d %H:%M} -- {d1:%Y-%m-%d %H:%M}"
        return f"{d0:%H:%M} -- {d1:%H:%M}"

    def _title_with_date(self, base):
        """Append the auto date/time range to a base title (blank base → date only).
        Returns "" when 'Show graph title' is OFF, so every RTI/ALT/cloud plot and
        the exported figures come out title-free (for paper figures)."""
        if not self.rti_show_title_var.get():
            return ""
        suf = self._title_date_suffix()
        base = (base or "").strip()
        return f"{base} {suf}".strip() if suf else base

    def _log(self, msg):
        # Step4Page (Display) has no Console widget → print (ASCII-safe for the
        # Windows console, which can't render unicode).
        try:
            print(str(msg).encode("ascii", "replace").decode("ascii"))
        except Exception:
            pass

    def _safe(self, fn, *a, **kw):
        try:
            self.after(0, lambda: fn(*a, **kw))
        except Exception:
            pass

    def _date_tag_for_filename(self):
        """A filesystem-safe date tag from the loaded data, e.g. '2026-07-13' (single
        day) or '2026-07-13_to_2026-07-14' (multi-day). Blank if no timestamps."""
        if not self.t_proto:
            return ""
        ts = [pd.Timestamp(t) for t in self.t_proto if not pd.isna(t)]
        if not ts:
            return ""
        d0, d1 = ts[0], ts[-1]
        if d0.date() == d1.date():
            return f"{d0:%Y-%m-%d}"
        return f"{d0:%Y-%m-%d}_to_{d1:%Y-%m-%d}"
    def apply_alt_settings(self): self._align_alt(); self.refresh_all()

    # ── Channel selection (co / cross / δ) for the RTI panels + profile ────────
    def _refresh_source_menus(self):
        if getattr(self, "_proto_src_menu", None) is not None:
            self._proto_src_menu.configure(values=list(self._proto_mats.keys()) or ["NRB parallel"])
        if getattr(self, "_mpl_src_menu", None) is not None:
            self._mpl_src_menu.configure(values=list(self._mpl_mats.keys()) or ["NRB parallel"])
        if getattr(self, "_prof_src_menu", None) is not None:
            # Profile source = which DATASET to draw; the channel comes from the
            # Prototype/Mini MPL channel dropdowns above.
            opts = ["Prototype", "Mini MPL", "Duo"]
            self._prof_src_menu.configure(values=opts)
            if self.profile_source.get() not in opts:
                self.profile_source.set("Duo")

    def _maybe_auto_scale(self, src):
        # A physical "real" or δ channel does not fit the preset 0-1 colour scale
        # → switch to auto so it renders on its true value range.
        if (not self._source_is_normalized(src)
                and self.scale_mode_var.get() == "preset"):
            self.scale_mode_var.set("auto")

    def _apply_proto_source(self, *_):
        if self._proto_mats:
            self.Z_proto = self._proto_mats.get(self.proto_rti_source.get(), self.Z_proto)
        self._maybe_auto_scale(self.proto_rti_source.get())
        self.refresh_all()

    def _apply_mpl_source(self, *_):
        if self._mpl_mats:
            self.Z_mpl = self._mpl_mats.get(self.mpl_rti_source.get(), self.Z_mpl)
        self._maybe_auto_scale(self.mpl_rti_source.get())
        self.align_and_refresh()

    def _apply_profile_source(self, *_):
        self.refresh_all()

    def _mpl_channel_on_proto(self, ch):
        """Resample an MPL channel matrix onto the prototype (r, t) grid."""
        Zc = self._mpl_mats.get(ch)
        if (Zc is None or self.match_idx is None or self.r_proto is None
                or self.t_proto is None or self.r_mpl is None):
            return None
        nR = len(self.r_proto); nT = len(self.t_proto)
        out = np.full((nR, nT), np.nan)
        for i in range(nT):
            j = self.match_idx[i] if i < len(self.match_idx) else None
            if j is not None:
                out[:, i] = _r4_interp(self.r_mpl, Zc[:, j], self.r_proto)
        return out

    def _denoised_active(self) -> bool:
        """True when the Raw/Denoised radio is on Denoised AND a denoised matrix
        exists (the denoise only applies to the parallel / co channel)."""
        return (self.profile_proto_mode_var.get().strip().lower() == "denoised"
                and self.Z_proto_den is not None)

    def _proto_mat_for(self, ch):
        """Prototype matrix for channel `ch`, honouring the Raw/Denoised radio so
        SAVED figures match what is on screen. Denoise exists only for the
        parallel (co) channel; every other channel returns its raw matrix."""
        if ch == "NRB parallel" and self._denoised_active():
            return self.Z_proto_den
        return self._proto_mats.get(ch)

    def _proto_profile_channel(self):
        """(Z, label, is_normalised) for the profile, using the Prototype channel
        dropdown (the same one that drives the Prototype RTI)."""
        ch = self.proto_rti_source.get()          # "NRB parallel", "NRB perpendicular", "δ_v", ...
        is_norm = self._source_is_normalized(ch)
        label = "Prototype " + ("δ_v" if ch == "δ_v" else ch.replace("NRB ", ""))
        if ch == "NRB parallel":                  # honour Raw/Denoised for the co channel
            mode = self.profile_proto_mode_var.get().strip().lower()
            if mode == "denoised" and self.Z_proto_den is not None:
                return self.Z_proto_den, label + " (Denoised)", is_norm
        return self._proto_mats.get(ch), label, is_norm

    def _mpl_profile_channel(self):
        """(Z_on_proto, label, is_normalised) for the profile, using the Mini MPL
        channel dropdown (resampled onto the prototype range grid)."""
        ch = self.mpl_rti_source.get()
        is_norm = self._source_is_normalized(ch)
        label = "MPL " + ch.replace("NRB ", "")
        return self._mpl_channel_on_proto(ch), label, is_norm

    def _profile_series(self):
        """The list of curves to draw in the profile panel, from Profile source
        (dataset: Prototype / Mini MPL / Duo) x Profile channels (single channel
        from the dropdown, or both parallel + perpendicular). Each item is
        (Z, label, is_norm, dataset, channel)."""
        src = self.profile_source.get()
        both = self.profile_channels.get().startswith("Parallel")
        show_proto = src in ("Prototype", "Duo")
        show_mpl = src in ("Mini MPL", "Duo")
        out = []
        if show_proto:
            if both:
                for ch in ("NRB parallel", "NRB perpendicular"):
                    Z = self._proto_mats.get(ch)
                    if Z is not None:
                        out.append((Z, "Prototype " + ch.replace("NRB ", ""),
                                    self._source_is_normalized(ch), "proto", ch))
            else:
                Z, lbl, nm = self._proto_profile_channel()
                out.append((Z, lbl, nm, "proto", self.proto_rti_source.get()))
        if show_mpl:
            if both:
                for ch in ("NRB parallel", "NRB perpendicular"):
                    Z = self._mpl_channel_on_proto(ch)
                    if Z is not None:
                        out.append((Z, "MPL " + ch.replace("NRB ", ""),
                                    self._source_is_normalized(ch), "mpl", ch))
            else:
                Z, lbl, nm = self._mpl_profile_channel()
                out.append((Z, lbl, nm, "mpl", self.mpl_rti_source.get()))
        return out

    def _profile_curve_style(self, dataset, ch, both):
        """Line style for a profile curve. In both-channel mode, COLOUR encodes
        the channel (parallel=navy, perpendicular=orange) so the two channels are
        easy to tell apart even within a single dataset, and the DASH encodes the
        dataset (Prototype=solid, Mini MPL=dashed). In single-channel mode the
        user's Line-style settings are honoured."""
        if both:
            color = theme.ORANGE if "perpendicular" in str(ch) else theme.NAVY
            ls = "--" if dataset == "mpl" else "-"
            return dict(color=color, linestyle=ls, linewidth=1.5)
        return self._line_style(dataset)

    def _on_manual_scale(self):
        if self.scale_mode_var.get() != "manual":
            self.scale_mode_var.set("manual")
        self.refresh_all()

    def _apply_scale_mode(self):
        mode = self.scale_mode_var.get()
        if mode == "preset":
            self.vmin_var.set(0.0); self.vmax_var.set(1.0)
        elif mode == "auto":
            vals = []
            for Z in (self.Z_proto, self.Z_mpl_on_proto):
                if Z is not None:
                    v = Z[np.isfinite(Z)]
                    if v.size:
                        vals.append(v)
            if vals:
                allv = np.concatenate(vals)
                p1, p99 = np.percentile(allv, [1, 99])
                self.vmin_var.set(float(p1)); self.vmax_var.set(float(p99))

    def _get_cmap(self):
        name = self.cmap_var.get().strip()
        cmap = plt.get_cmap(name if name in CMAP_CHOICES else "jet").copy()
        cmap.set_bad("black")
        return cmap

    def _get_norm(self, vmin, vmax, cmap):
        if self.use_log_color_var.get():
            return mcolors.LogNorm(vmin=max(vmin, 1e-8), vmax=max(vmax, max(vmin, 1e-8) * 10))
        if self.discrete_var.get():
            step = _r4_float(self.step_var.get()) or 0.05
            if step <= 0:
                step = 0.05
            levels = np.arange(vmin, vmax + step, step)
            if levels.size < 2:
                levels = np.array([vmin, vmax])
            return mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    def _mask(self, Z):
        Z = np.asarray(Z, float)
        if self.use_log_color_var.get():
            return np.ma.masked_where(~np.isfinite(Z) | (Z <= 0), Z)
        return np.ma.masked_invalid(Z)

    def _setup_axes(self):
        self.fig.clear()
        self.ax_rti1 = self.fig.add_subplot(511)
        self.ax_rti2 = self.fig.add_subplot(512)
        self.ax_prof = self.fig.add_subplot(513)
        self.ax_altcmp = self.fig.add_subplot(514)
        self.ax_cloud = self.fig.add_subplot(515)
        self.ax_rti_lines = self.fig.add_subplot(616)  # repositioned by _apply_layout
        self.ax_rti_lines.set_visible(False)
        self.ax_snr = self.fig.add_subplot(717)         # SNR curtain, repositioned too
        self.ax_snr.set_visible(False)
        self.cax1 = self.fig.add_axes([0.935, 0.70, 0.015, 0.17])
        self.cax2 = self.fig.add_axes([0.935, 0.50, 0.015, 0.17])
        self.cax_snr = self.fig.add_axes([0.935, 0.30, 0.015, 0.17])
        self.cax_snr.set_visible(False)
        self.cbar_snr = None

    def _init_blank_plot(self):
        for ax in (self.ax_rti1, self.ax_rti2, self.ax_prof, self.ax_altcmp,
                   self.ax_cloud, self.ax_rti_lines, self.ax_snr):
            ax.clear()
        self._reset_cbars()
        self._apply_layout()
        self.canvas.draw_idle()

    def _reset_cbars(self):
        self.cax1.cla(); self.cax2.cla(); self.cbar1 = self.cbar2 = None
        self.cax1.set_visible(False); self.cax2.set_visible(False)
        self.cax_snr.cla(); self.cbar_snr = None; self.cax_snr.set_visible(False)

    def _apply_layout(self):
        vis = []
        if self.show_proto_var.get(): vis.append(("rti1", self.ax_rti1))
        if self.show_mpl_var.get(): vis.append(("rti2", self.ax_rti2))
        if self.show_rti_lines_var.get(): vis.append(("rtilines", self.ax_rti_lines))
        if self.show_snr_var.get(): vis.append(("snr", self.ax_snr))
        if self.show_profile_var.get(): vis.append(("prof", self.ax_prof))
        if self.show_alt_compare_var.get(): vis.append(("altcmp", self.ax_altcmp))
        if self.show_cloud_panel_var.get(): vis.append(("cloud", self.ax_cloud))
        for ax in (self.ax_rti1, self.ax_rti2, self.ax_prof, self.ax_altcmp,
                   self.ax_cloud, self.ax_rti_lines, self.ax_snr):
            ax.set_visible(False)
        if not vis:
            return
        left_ = 0.07; right_ = 0.885; top_ = 0.93; bottom_ = 0.075
        weights = [1.10 if n.startswith("rti") else 0.72 for n, _ in vis]
        gaps = []
        for i in range(len(vis) - 1):
            cur_name = vis[i][0]; next_name = vis[i + 1][0]
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
            total_gap = sum(gaps); avail = (top_ - bottom_) - total_gap
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
        self.cax1.set_visible(False); self.cax2.set_visible(False)
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
        self.cax_snr.set_visible(False)   # SNR panel is a line plot — no colorbar

    def _axis_tick_labels(self):
        """X-axis tick labels honoring the 'Show date on X-axis' checkbox. When
        OFF, strip a leading 'MM-DD ' so only HH:MM remains — even for a span
        that crosses days (where _r4_build_x_labels auto-adds the date)."""
        labs = self.t_labels_hhmm or []
        if self.rti_show_date_var.get():
            return list(labs)
        return [str(s).split()[-1] if str(s).strip() else s for s in labs]

    def _rti_style(self, ax, *, show_xlabel=True, show_xticks=True):
        ax.set_xlabel(self.rti_xlabel_var.get() if show_xlabel else "", labelpad=8)
        ax.set_ylabel(self.rti_ylabel_var.get())
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))  # 10,000
        if self.t_labels_hhmm:
            labs = self._axis_tick_labels()
            stepN = max(1, _r4_int(self.rti_tickN_var.get()) or 4)
            xt = np.arange(0, len(self.t_labels_hhmm), stepN)
            ax.set_xticks(xt)
            if show_xticks:
                ax.set_xticklabels([labs[i] for i in xt], rotation=30, ha="right")
                ax.tick_params(axis="x", labelbottom=True, bottom=True, pad=1)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", labelbottom=False, bottom=False)
        ymin = _r4_float(self.rti_ymin_var.get())
        ymax = _r4_float(self.rti_ymax_var.get())
        ytick = _r4_float(self.rti_ytick_var.get())
        if ymin is not None and ymax is not None and ymax > ymin:
            ax.set_ylim(ymin, ymax)
            # include the ymin floor (120 m) as a labelled tick
            ax.set_yticks(self._yticks_with_floor(ymin, ymax, ytick))

    def _update_cbars(self):
        label = self.cbar_label_var.get()
        self.cbar1 = None; self.cbar2 = None
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
        self.ax_altcmp.set_title(self._title_with_date(self.title_altcmp_var.get()), pad=8)
        self.ax_altcmp.set_xlabel(self.alt_xlabel_var.get())
        self.ax_altcmp.set_ylabel(self.alt_ylabel_var.get())
        if self.t_labels_hhmm:
            labs = self._axis_tick_labels()
            stepN = max(1, _r4_int(self.alt_tickN_var.get()) or 4)
            xt = np.arange(0, len(self.t_labels_hhmm), stepN)
            self.ax_altcmp.set_xticks(xt)
            self.ax_altcmp.set_xticklabels(
                [labs[i] for i in xt], rotation=30, ha="right")
            self.ax_altcmp.tick_params(axis="x", pad=2)
        ymin = _r4_float(self.alt_ymin_var.get())
        ymax = _r4_float(self.alt_ymax_var.get())
        ytick = _r4_float(self.alt_ytick_var.get())
        if ymin is not None and ymax is not None and ymax > ymin:
            self.ax_altcmp.set_ylim(ymin, ymax)
            self.ax_altcmp.set_yticks(self._yticks_with_floor(ymin, ymax, ytick))
        self.ax_altcmp.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
        self.ax_altcmp.grid(True, alpha=0.25)

    def _paper_rc(self):
        """User-configurable font (family / sizes / bold) for the Display figure,
        scoped here so other steps keep their default font."""
        weight = "bold" if self.font_bold.get() else "normal"
        fam = self.font_family.get().strip() or "Times New Roman"

        def _sz(var, default):
            v = _r4_float(var.get())
            return v if (v and v > 0) else default
        return {
            "font.family": fam,
            "mathtext.fontset": "stix",
            "font.weight": weight,
            "text.color": "black",
            "axes.edgecolor": "black", "axes.labelcolor": "black",
            "xtick.color": "black", "ytick.color": "black",
            "axes.titlesize": _sz(self.font_title_size, 17),
            "axes.titleweight": weight, "axes.titlecolor": "black",
            "axes.labelsize": _sz(self.font_label_size, 15), "axes.labelweight": weight,
            "xtick.labelsize": _sz(self.font_tick_size, 12),
            "ytick.labelsize": _sz(self.font_tick_size, 12),
            "legend.fontsize": _sz(self.font_legend_size, 12),
        }

    def refresh_all(self):
        with plt.rc_context(self._paper_rc()):
            self._draw_all()

    def _draw_all(self):
        for ax in (self.ax_rti1, self.ax_rti2, self.ax_prof, self.ax_altcmp,
                   self.ax_cloud, self.ax_rti_lines, self.ax_snr):
            ax.clear()
        self._reset_cbars(); self._apply_layout()
        cmap = self._get_cmap()
        interp = self.interp_var.get().strip()
        if interp not in INTERP_CHOICES:
            interp = "nearest"
        vmin = float(self.vmin_var.get())
        vmax = float(self.vmax_var.get())
        if vmax <= vmin:
            vmax = vmin + 1e-6
        norm = self._get_norm(vmin, vmax, cmap)
        self.im1 = None
        if (self.show_proto_var.get() and self.ax_rti1.get_visible()
                and self.Z_proto is not None and self.r_proto is not None):
            Zm = self._mask(np.clip(self.Z_proto, vmin, vmax)
                            if self.scale_mode_var.get() == "preset" else self.Z_proto)
            self.im1 = self.ax_rti1.imshow(
                Zm, aspect="auto", origin="lower",
                extent=[0, Zm.shape[1] - 1,
                        float(np.nanmin(self.r_proto)), float(np.nanmax(self.r_proto))],
                cmap=cmap, norm=norm, interpolation=interp, resample=False)
        if self.ax_rti1.get_visible():
            self.ax_rti1.set_title(self._title_with_date(self.title_rti_proto_var.get()), pad=8)
            self._rti_style(self.ax_rti1, show_xlabel=True, show_xticks=True)
        self.im2 = None
        if (self.show_mpl_var.get() and self.ax_rti2.get_visible()
                and self.Z_mpl_on_proto is not None and self.r_proto is not None):
            Zm = self._mask(np.clip(self.Z_mpl_on_proto, vmin, vmax)
                            if self.scale_mode_var.get() == "preset" else self.Z_mpl_on_proto)
            self.im2 = self.ax_rti2.imshow(
                Zm, aspect="auto", origin="lower",
                extent=[0, Zm.shape[1] - 1,
                        float(np.nanmin(self.r_proto)), float(np.nanmax(self.r_proto))],
                cmap=cmap, norm=norm, interpolation=interp, resample=False)
        if self.ax_rti2.get_visible():
            self.ax_rti2.set_title(self._title_with_date(self.title_rti_mpl_var.get()), pad=8)
            self._rti_style(self.ax_rti2, show_xlabel=True, show_xticks=True)
        if self.show_alt_overlay_var.get() and self.t_labels_hhmm:
            x = np.arange(len(self.t_labels_hhmm))
            # White line + black halo stays visible over ANY colormap value
            # (a plain red line vanishes inside the red high-NRB aerosol band).
            _alt_stroke = [mpe.withStroke(linewidth=3.4, foreground="black")]
            if self.ax_rti1.get_visible() and self.alt_proto_by_slot is not None:
                self.ax_rti1.plot(x, self.alt_proto_by_slot.astype(float),
                                  color="white", lw=1.6, alpha=1.0,
                                  path_effects=_alt_stroke)
                # colour each ALT point by its δ feature (aerosol/cloud)
                self._draw_alt_feature_markers(self.ax_rti1, self.alt_proto_by_slot)
            if self.ax_rti2.get_visible() and self.alt_mpl_by_slot is not None:
                # MPL ALT in a distinct colour (amber) so it is clearly the MPL
                # panel's own line, not confused with the proto white line.
                self.ax_rti2.plot(x, self.alt_mpl_by_slot.astype(float),
                                  color="#FFD400", lw=1.6, alpha=1.0,
                                  path_effects=_alt_stroke)
        if (self.show_profile_var.get() and self.ax_prof.get_visible()) or \
           (self.show_snr_var.get() and self.ax_snr.get_visible()):
            self.update_profile_plot(True)   # also refreshes the SNR line panel
        else:
            self._update_markers()
        if self.show_alt_compare_var.get() and self.ax_altcmp.get_visible():
            self._style_altcmp()
            if self.t_labels_hhmm and self.alt_proto_by_slot is not None:
                x = np.arange(len(self.t_labels_hhmm))
                # Prototype first, MPL dashed on top: guided ALT tracks MPL
                # closely (~140 m apart), so a thin MPL line drawn under the
                # thick proto line gets hidden. Dashed + on top keeps both.
                proto_lab = (self.alt_proto_legend.get().strip()
                             or f"Prototype ALT ({self.alt_proto_col.get()})")
                mpl_lab = self.alt_mpl_legend.get().strip() or "MPL ALT"
                self.ax_altcmp.plot(x, self.alt_proto_by_slot.astype(float),
                                    zorder=2, label=proto_lab,
                                    **self._line_style("proto"))
                if self.alt_mpl_by_slot is not None:
                    self.ax_altcmp.plot(x, self.alt_mpl_by_slot.astype(float),
                                        zorder=3, label=mpl_lab,
                                        **self._line_style("mpl"))
                # colour each prototype ALT point by δ feature (with legend here)
                self._draw_alt_feature_markers(self.ax_altcmp, self.alt_proto_by_slot,
                                               label=True)
                self.ax_altcmp.legend()

        # ── Cloud markers on RTI ───────────────────────────────────────────
        if self.show_cloud_markers_var.get():
            self._draw_cloud_markers()

        # ── Cloud layers panel ─────────────────────────────────────────────
        if self.show_cloud_panel_var.get() and self.ax_cloud.get_visible():
            self._draw_cloud_panel()

        # ── RTI-as-lines panel (stacked vertical profiles, proto vs MPL) ────
        if self.show_rti_lines_var.get() and self.ax_rti_lines.get_visible():
            self._render_rti_lines_panel(self.fig, self.ax_rti_lines)

        self._update_cbars()
        # Force the tick-label weight explicitly (rc font.weight does not always
        # survive tick regeneration). Honours the Bold toggle.
        _tw = "bold" if self.font_bold.get() else "normal"
        for ax in (self.ax_rti1, self.ax_rti2, self.ax_prof, self.ax_altcmp,
                   self.ax_cloud, self.ax_rti_lines, self.ax_snr):
            for lab in (ax.get_xticklabels() + ax.get_yticklabels()):
                lab.set_fontweight(_tw)
        for cb in (getattr(self, "cbar1", None), getattr(self, "cbar2", None),
                   getattr(self, "cbar_snr", None)):
            if cb is not None:
                for lab in cb.ax.get_yticklabels():
                    lab.set_fontweight(_tw)
        if self._mpl_cid is None:
            self._mpl_cid = self.canvas.mpl_connect("button_press_event", self._on_click)
        # Synchronous draw so tick labels are generated INSIDE the paper rc_context
        # (a deferred draw_idle would regenerate them with the default thin font).
        self.canvas.draw()

    def _get_sel(self):
        sel = list(self.time_list.curselection())
        if not sel and self.t_list_labels:
            sel = [0]; self.time_list.selection_set(0)
        return sel

    def _get_proto_mat(self):
        mode = self.profile_proto_mode_var.get().strip().lower()
        if mode == "denoised":
            if self.Z_proto_den is not None:
                return self.Z_proto_den, "Denoised", None
            return None, "Denoised", f"Load {DENOISED_SHEET} from ALT file first"
        return self.Z_proto, "Raw", None

    def update_profile_plot(self, redraw_markers=False):
        self.ax_prof.clear()
        self.ax_prof.set_title(self._title_with_date(self.title_prof_var.get()), pad=10)
        self.ax_prof.set_xlabel(self.prof_xlabel_var.get())
        self.ax_prof.set_ylabel(self.prof_ylabel_var.get())
        if self.r_proto is None or not self._proto_mats:
            self.ax_prof.text(0.5, 0.5, "Load Prototype first",
                              ha="center", va="center", transform=self.ax_prof.transAxes)
            return
        # Profile source picks the DATASET (Prototype / Mini MPL / Duo); Profile
        # channels picks a single channel (from the channel dropdown) or both
        # parallel + perpendicular of each shown dataset.
        series = self._profile_series()
        if not any(s[0] is not None for s in series):
            msg = ("Mini MPL: load / align MPL first"
                   if self.profile_source.get() == "Mini MPL"
                   else "No data (load / align first)")
            self.ax_prof.text(0.5, 0.5, msg, ha="center", va="center",
                              transform=self.ax_prof.transAxes)
            self._update_markers(); return
        r = self.r_proto
        both = self.profile_channels.get().startswith("Parallel")
        proto_pref = self.proto_legend_prefix.get().strip()
        mpl_pref = self.mpl_legend_prefix.get().strip()
        sel = self._get_sel()
        lc = 0
        all_norm = True
        for (Z, lbl, nm, dataset, ch) in series:
            if Z is None:
                continue
            all_norm = all_norm and nm
            style = self._profile_curve_style(dataset, ch, both)
            pref = proto_pref if dataset == "proto" else mpl_pref
            base = lbl if both else (pref or lbl)
            for idx in sel[:10]:
                if idx >= Z.shape[1]:
                    continue
                y = Z[:, idx]; m = np.isfinite(y)
                if m.any():
                    self.ax_prof.plot(r[m], y[m],
                                      label=f"{base} {self.t_labels_hhmm[idx]}", **style)
                    lc += 1
        self.ax_prof.grid(True, alpha=0.25)
        self.ax_prof.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))  # Range 2,000
        if lc > 0:
            self.ax_prof.legend()
        try:
            xmin = _r4_float(self.prof_xmin_var.get())
            xmax = _r4_float(self.prof_xmax_var.get())
            ymin = _r4_float(self.prof_ymin_var.get())
            ymax = _r4_float(self.prof_ymax_var.get())
            if xmin is not None and xmax is not None and xmax > xmin:
                self.ax_prof.set_xlim(xmin, xmax)
            # Clamp Y (default 0-1) only when EVERY shown curve is normalised;
            # a physical "(real)" or δ curve auto-scales to its own range.
            if all_norm and ymin is not None and ymax is not None and ymax > ymin:
                self.ax_prof.set_ylim(ymin, ymax)
        except Exception:
            pass
        self._update_markers()
        # The SNR line panel tracks the same time selection → refresh it here so a
        # profile-time change updates it too, then push the redraw to screen.
        self._refresh_snr_panel()
        self.canvas.draw_idle()

    @staticmethod
    def _source_is_normalized(src: str) -> bool:
        """A source is 0-1 normalised unless it is a physical '(real)' channel,
        a raw-counts channel, or the depolarization ratio δ."""
        s = str(src)
        return ("(real)" not in s) and ("δ" not in s) and ("raw" not in s)

    def _update_markers(self):
        for ln in self.sel_lines:
            try:
                ln.remove()
            except Exception:
                pass
        self.sel_lines = []
        for idx in self._get_sel():
            for ax in (self.ax_rti1, self.ax_rti2):
                if ax.get_visible():
                    self.sel_lines.append(ax.axvline(idx, color="white", lw=1.0, alpha=0.85))

    # ── Cloud visualization ────────────────────────────────────────────────
    # Per-layer colors (white / cyan / pink stand out on jet/turbo colormap).
    _CLOUD_COLORS = ["#FFFFFF", "#22D3EE", "#FF3257", "#FACC15"]

    def _draw_cloud_markers(self):
        """Overlay cloud base (circle) and top (triangle) markers on the RTI axes."""
        if self.cloud_base_by_slot is None or not self.t_labels_hhmm:
            return
        x = np.arange(len(self.t_labels_hhmm))
        rti_axes = [ax for ax in (self.ax_rti1, self.ax_rti2) if ax.get_visible()]
        for ax in rti_axes:
            for layer in range(1, self.cloud_max_layer + 1):
                base = self.cloud_base_by_slot.get(layer)
                top = self.cloud_top_by_slot.get(layer)
                if base is None:
                    continue
                color = self._CLOUD_COLORS[(layer - 1) % len(self._CLOUD_COLORS)]
                m_base = np.isfinite(base)
                if np.any(m_base):
                    # connect base-top with thin vertical lines
                    if top is not None:
                        for xi in x[m_base]:
                            b, t = base[xi], top[xi]
                            if np.isfinite(b) and np.isfinite(t):
                                ax.plot([xi, xi], [b, t], color=color,
                                        lw=0.7, alpha=0.55)
                    ax.scatter(x[m_base], base[m_base], s=18, marker="o",
                               facecolors="none", edgecolors=color, linewidths=1.1,
                               alpha=0.95, zorder=5,
                               label=f"Cloud {layer} base" if ax is rti_axes[0] else None)
                if top is not None:
                    m_top = np.isfinite(top)
                    if np.any(m_top):
                        ax.scatter(x[m_top], top[m_top], s=20, marker="v",
                                   facecolors=color, edgecolors=color, linewidths=0.8,
                                   alpha=0.85, zorder=5)

    def _draw_cloud_panel(self):
        """Dedicated panel: cloud base/top time series per layer + cloud fraction."""
        ax = self.ax_cloud
        ax.clear()
        ax.set_title("Cloud Layers", pad=8)
        ax.set_xlabel(self.rti_xlabel_var.get())
        ax.set_ylabel("Height (m)")

        if self.cloud_base_by_slot is None or not self.t_labels_hhmm:
            ax.text(0.5, 0.5, "Load ALT file with Cloud_results sheet",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            return

        x = np.arange(len(self.t_labels_hhmm))
        # x-axis ticks (reuse RTI tick step)
        stepN = max(1, _r4_int(self.rti_tickN_var.get()) or 4)
        xt = np.arange(0, len(self.t_labels_hhmm), stepN)
        labs = self._axis_tick_labels()
        ax.set_xticks(xt)
        ax.set_xticklabels([labs[i] for i in xt], rotation=30, ha="right")
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

        total_slots = len(x)
        any_cloud = np.zeros(total_slots, dtype=bool)
        for layer in range(1, self.cloud_max_layer + 1):
            base = self.cloud_base_by_slot.get(layer)
            top = self.cloud_top_by_slot.get(layer)
            if base is None:
                continue
            color = self._CLOUD_COLORS[(layer - 1) % len(self._CLOUD_COLORS)]
            # Use a darker variant for line visibility on white panel
            line_color = {"#FFFFFF": "#94A3B8"}.get(color, color)
            m = np.isfinite(base)
            any_cloud |= m
            if np.any(m):
                ax.plot(x[m], base[m], "o-", color=line_color, ms=3, lw=1.0,
                        alpha=0.9, label=f"Layer {layer} base")
                if top is not None:
                    mt = np.isfinite(top)
                    if np.any(mt):
                        ax.plot(x[mt], top[mt], "v--", color=line_color, ms=3, lw=0.8,
                                alpha=0.7, label=f"Layer {layer} top")
                        # fill between base and top
                        both = m & mt
                        if np.any(both):
                            ax.fill_between(x[both], base[both], top[both],
                                            color=line_color, alpha=0.12)

        # Cloud fraction annotation
        n_cloud = int(np.sum(any_cloud))
        frac = 100.0 * n_cloud / total_slots if total_slots else 0.0
        ax.text(0.015, 0.95, f"Cloud fraction: {frac:.0f}%  ({n_cloud}/{total_slots} slots)",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=theme.CREAM,
                          edgecolor=theme.ORANGE))

        ymin = _r4_float(self.rti_ymin_var.get())
        ymax = _r4_float(self.rti_ymax_var.get())
        if ymin is not None and ymax is not None and ymax > ymin:
            ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.25)
        if self.cloud_max_layer >= 1 and n_cloud > 0:
            ax.legend(fontsize=7, loc="upper right", ncol=2)

    def _on_click(self, event):
        if self.t_list_labels is None:
            return
        if event.inaxes not in (self.ax_rti1, self.ax_rti2):
            return
        if event.xdata is None:
            return
        idx = max(0, min(len(self.t_list_labels) - 1, int(round(event.xdata))))
        self.time_list.selection_clear(0, tk.END)
        self.time_list.selection_set(idx)
        self.time_list.see(idx)
        self.update_profile_plot(True)
        self.canvas.draw_idle()

    # ── Save (with figsize chooser) ────────────────────────────────────────
    def _savefig(self, fig, path, dpi, bbox="tight"):
        """Save a figure. CMYK mode → RGB rendered then converted to a CMYK TIFF
        (print), else a normal RGB PNG. Returns the path actually written."""
        if self.cmyk_var.get():
            import io
            from PIL import Image
            base, ext = os.path.splitext(path)
            if ext.lower() not in (".tif", ".tiff"):
                path = base + ".tif"
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches=bbox, pad_inches=0.15)
            buf.seek(0)
            Image.open(buf).convert("RGB").convert("CMYK").save(
                path, format="TIFF", dpi=(dpi, dpi))
        else:
            fig.savefig(path, dpi=dpi, bbox_inches=bbox, pad_inches=0.15)
        return path

    def _save_ext(self):
        return ".tif" if self.cmyk_var.get() else ".png"

    def save_png(self):
        dlg = FigsizeDialog(self.winfo_toplevel())
        if dlg.result is None:
            return
        w, h, dpi = dlg.result
        ext = self._save_ext()
        fp = filedialog.asksaveasfilename(
            defaultextension=ext,
            filetypes=[("TIFF (CMYK)", "*.tif")] if ext == ".tif" else [("PNG", "*.png")])
        if not fp:
            return
        try:
            orig = self.fig.get_size_inches()
            with plt.rc_context(self._paper_rc()):
                self.fig.set_size_inches(w, h)
                self.fig.canvas.draw()          # render ticks under the paper rc
                fp = self._savefig(self.fig, fp, dpi)
            self.fig.set_size_inches(orig)
            self.refresh_all()
            messagebox.showinfo("Saved", f"Saved ({w:g}×{h:g} in @ {dpi} DPI):\n{fp}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _render_rti_panel(self, fig, ax, Z, title_base, which):
        """Draw one RTI heatmap (+ colorbar + optional ALT overlay) on a fresh ax."""
        if Z is None or self.r_proto is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            return
        cmap = self._get_cmap()
        interp = self.interp_var.get().strip()
        if interp not in INTERP_CHOICES:
            interp = "nearest"
        vmin = float(self.vmin_var.get()); vmax = float(self.vmax_var.get())
        if vmax <= vmin:
            vmax = vmin + 1e-6
        norm = self._get_norm(vmin, vmax, cmap)
        Zm = self._mask(np.clip(Z, vmin, vmax)
                        if self.scale_mode_var.get() == "preset" else Z)
        im = ax.imshow(
            Zm, aspect="auto", origin="lower",
            extent=[0, Zm.shape[1] - 1,
                    float(np.nanmin(self.r_proto)), float(np.nanmax(self.r_proto))],
            cmap=cmap, norm=norm, interpolation=interp, resample=False)
        ax.set_title(self._title_with_date(title_base), pad=8)
        self._rti_style(ax, show_xlabel=True, show_xticks=True)
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(self.cbar_label_var.get())
        if self.show_alt_overlay_var.get() and self.t_labels_hhmm:
            x = np.arange(len(self.t_labels_hhmm))
            stroke = [mpe.withStroke(linewidth=3.4, foreground="black")]
            if which == "proto" and self.alt_proto_by_slot is not None:
                ax.plot(x, self.alt_proto_by_slot.astype(float),
                        color="white", lw=1.6, path_effects=stroke)
            elif which == "mpl" and self.alt_mpl_by_slot is not None:
                ax.plot(x, self.alt_mpl_by_slot.astype(float),
                        color="#FFD400", lw=1.6, path_effects=stroke)
        tw = "bold" if self.font_bold.get() else "normal"
        for lab in (ax.get_xticklabels() + ax.get_yticklabels()):
            lab.set_fontweight(tw)
        for lab in cb.ax.get_yticklabels():
            lab.set_fontweight(tw)

    def _refresh_snr_panel(self):
        """Re-render the SNR line panel if it is visible (selection-dependent,
        like the Profile view)."""
        if self.show_snr_var.get() and self.ax_snr.get_visible():
            self.ax_snr.clear()
            self._render_snr_panel(self.ax_snr)

    def _render_snr_panel(self, ax):
        """SNR as LINE profiles (like Step 2): SNR vs Height for the selected
        time(s), on a log-y axis, with the SNR=3 trusted threshold marked."""
        Z = self.Z_proto_snr
        if Z is None or self.r_proto is None:
            ax.text(0.5, 0.5, "no SNR data (needs the 'SNR' sheet)",
                    ha="center", va="center", transform=ax.transAxes)
            return None
        Z = np.asarray(Z, float); r = np.asarray(self.r_proto, float)
        lc = 0
        for idx in self._get_sel()[:10]:
            if idx >= Z.shape[1]:
                continue
            y = Z[:, idx]
            m = np.isfinite(r) & np.isfinite(y) & (y > 0)
            if m.any():
                ax.plot(r[m], y[m], lw=1.3, label=f"SNR {self.t_labels_hhmm[idx]}")
                lc += 1
        ax.axhline(3.0, color="#e24a3b", lw=1.2, ls="--",
                   label="trusted (SNR=3)" if lc else None)
        ax.set_yscale("log"); ax.set_ylim(0.3, 300)
        ax.set_xlabel("Height (m)"); ax.set_ylabel("SNR")
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
        xmin = _r4_float(self.prof_xmin_var.get()); xmax = _r4_float(self.prof_xmax_var.get())
        if xmin is not None and xmax is not None and xmax > xmin:
            ax.set_xlim(xmin, xmax)
        ax.set_title(self._title_with_date("Prototype SNR (parallel)"), pad=8)
        ax.grid(True, which="both", alpha=0.2)
        if lc:
            ax.legend()
        return None

    def _render_rti_lines_panel(self, fig, ax):
        """5th summary figure — the RTI drawn as LINES instead of a heatmap:
        each time slot's NRB profile is a vertical curve anchored at its own
        timestamp on the x-axis, deflecting rightward by the (normalised) NRB
        amplitude. Prototype (denoised) in teal, the nearest-sample MiniMPL in
        red, with the gap between them shaded — a compact 'small-multiples'
        comparison of every profile across the day."""
        import matplotlib.dates as mdates
        from matplotlib.lines import Line2D
        Zp = self.Z_proto_den if self.Z_proto_den is not None \
            else self._proto_mats.get("NRB parallel")
        if Zp is None or self.r_proto is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            return
        Zm = self.Z_mpl_on_proto
        r = np.asarray(self.r_proto, float)          # metres — same Y axis as the other RTIs
        ts = [pd.Timestamp(t) if not pd.isna(t) else None for t in (self.t_proto or [])]
        xnum = np.array([mdates.date2num(t.to_pydatetime()) if t is not None else np.nan
                         for t in ts], float)
        valid = np.sort(xnum[np.isfinite(xnum)])
        gap = float(np.median(np.diff(valid))) if valid.size >= 2 else (10.0 / 1440.0)
        # A full-amplitude (=1) profile deflects 0.8 of the slot spacing, so even the
        # widest near-field spike stops short of the next profile → no overlap.
        xscale = gap * 0.8
        ymin = _r4_float(self.rti_ymin_var.get()); ymax = _r4_float(self.rti_ymax_var.get())
        y0, y1 = (ymin, ymax) if (ymin is not None and ymax is not None and ymax > ymin) \
            else (float(np.nanmin(r)), float(np.nanmax(r)))
        TEAL, RED = "#2b8ca6", "#e24a3b"
        nT = int(Zp.shape[1])
        for i in range(nT):
            x0 = xnum[i] if i < xnum.size else np.nan
            if not np.isfinite(x0):
                continue
            yp = Zp[:, i].astype(float)
            mp = np.isfinite(yp) & (r >= y0) & (r <= y1)
            if not mp.any():
                continue
            ax.axvline(x0, color="#c9ced6", lw=0.5, zorder=1)
            xp = x0 + xscale * np.clip(yp, 0.0, None)
            if Zm is not None and i < Zm.shape[1]:
                ym = Zm[:, i].astype(float)
                mm = np.isfinite(ym) & (r >= y0) & (r <= y1)
                if mm.any():
                    xm = x0 + xscale * np.clip(ym, 0.0, None)
                    both = mp & mm
                    ax.fill_betweenx(r[both], xm[both], xp[both],
                                     color="#8a8f98", alpha=0.30, lw=0, zorder=2)
                    ax.fill_betweenx(r[mm], x0, xm[mm], color=RED, alpha=0.10, lw=0, zorder=2)
                    ax.plot(xm[mm], r[mm], color=RED, lw=0.8, zorder=4)
            ax.fill_betweenx(r[mp], x0, xp[mp], color=TEAL, alpha=0.10, lw=0, zorder=3)
            ax.plot(xp[mp], r[mp], color=TEAL, lw=0.8, zorder=5)
        if valid.size:
            ax.set_xlim(valid.min() - gap * 0.5, valid.max() + xscale + gap * 0.5)
        # Y axis in metres, matching the other RTIs (RTI ymin/ymax + ytick step)
        ax.set_ylim(y0, y1)
        ax.set_ylabel(self.rti_ylabel_var.get())
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
        ytick = _r4_float(self.rti_ytick_var.get())
        if ymin is not None and ymax is not None and ymax > ymin:
            ax.set_yticks(self._yticks_with_floor(ymin, ymax, ytick))
        # X ticks at the ACTUAL profile timestamps, stepped by 'RTI tickN' and
        # honouring the Show-date checkbox — not auto hour ticks.
        labs = self._axis_tick_labels()
        stepN = max(1, _r4_int(self.rti_tickN_var.get()) or 4)
        idxs = [i for i in range(0, nT, stepN)
                if i < xnum.size and np.isfinite(xnum[i]) and i < len(labs)]
        ax.set_xticks([xnum[i] for i in idxs])
        ax.set_xticklabels([labs[i] for i in idxs], rotation=30, ha="right")
        ax.set_xlabel(self.rti_xlabel_var.get())
        ax.set_title(self._title_with_date("Prototype vs Mini MPL"), pad=8)
        ax.grid(True, axis="y", alpha=0.20)
        # Legend in the upper-left empty region (above the aerosol layer) with an
        # opaque box, so it never sits on top of the profile lines.
        leg = ax.legend([Line2D([0], [0], color=TEAL, lw=1.4),
                         Line2D([0], [0], color=RED, lw=1.4)],
                        ["Prototype", "Mini MPL"], loc="upper left", framealpha=0.95)
        leg.set_zorder(20)
        tw = "bold" if self.font_bold.get() else "normal"
        for lab in (ax.get_xticklabels() + ax.get_yticklabels()):
            lab.set_fontweight(tw)
        for lab in ax.get_yticklabels():
            lab.set_fontweight(tw)

    def save_rti_pair(self):
        """Export Mini MPL + Prototype RTI side-by-side in a single figure."""
        if self.Z_proto is None:
            messagebox.showinfo("No data", "Load Prototype first."); return
        dlg = FigsizeDialog(self.winfo_toplevel())
        if dlg.result is None:
            return
        w, h, dpi = dlg.result
        ext = self._save_ext()
        fp = filedialog.asksaveasfilename(
            defaultextension=ext,
            filetypes=[("TIFF (CMYK)", "*.tif")] if ext == ".tif" else [("PNG", "*.png")])
        if not fp:
            return
        try:
            with plt.rc_context(self._paper_rc()):
                fig = plt.Figure(figsize=(w * 2, h))   # double width for 2 panels
                ax1 = fig.add_subplot(121)   # Mini MPL (left, like the example)
                ax2 = fig.add_subplot(122)   # Prototype (right)
                fig.subplots_adjust(wspace=0.14, left=0.06, right=0.96)
                self._render_rti_panel(fig, ax1, self.Z_mpl_on_proto,
                                       self.title_rti_mpl_var.get(), "mpl")
                self._render_rti_panel(fig, ax2, self.Z_proto,
                                       self.title_rti_proto_var.get(), "proto")
                fp = self._savefig(fig, fp, dpi)
            messagebox.showinfo("Saved", f"Saved RTI pair (MPL + Prototype):\n{fp}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_all_figures(self, folder=None, size=None, announce=True):
        """One round → 5 PNGs: Prototype RTI, Mini MPL RTI, ALT compare, the
        side-by-side RTI pair, and the RTI-as-lines profile overlay (single
        size/DPI prompt, one base name). When `folder` is given (batch mode) the
        files land there as <date>_<suffix>, using `size` and no messagebox."""
        if self.Z_proto is None:
            if announce:
                messagebox.showinfo("No data", "Load Prototype first.")
            return
        if size is None:
            dlg = FigsizeDialog(self.winfo_toplevel())
            if dlg.result is None:
                return
            size = dlg.result
        w, h, dpi = size
        ext = self._save_ext()
        date_tag = self._date_tag_for_filename()
        if folder is not None:
            stem = str(Path(folder) / (date_tag or "figure"))
        else:
            base = filedialog.asksaveasfilename(
                defaultextension=ext,
                filetypes=[("TIFF (CMYK)", "*.tif")] if ext == ".tif" else [("PNG", "*.png")],
                initialfile=(f"{date_tag}{ext}" if date_tag else None),
                title="Base name — 5 files saved with suffixes")
            if not base:
                return
            stem = os.path.splitext(base)[0]
        written = []
        # 1-3: isolate each stacked panel and save
        panels = [
            ("prototype_rti", (True,  False, False, False, False, False, False)),
            ("mpl_rti",       (False, True,  False, False, False, False, False)),
            ("alt_compare",   (False, False, False, True,  False, False, False)),
        ]
        vars_ = (self.show_proto_var, self.show_mpl_var, self.show_profile_var,
                 self.show_alt_compare_var, self.show_cloud_panel_var,
                 self.show_rti_lines_var, self.show_snr_var)
        saved = tuple(v.get() for v in vars_)
        orig = self.fig.get_size_inches()
        try:
            self.fig.set_size_inches(w, h)
            for suffix, cfg in panels:
                for v, val in zip(vars_, cfg):
                    v.set(val)
                self.refresh_all()
                with plt.rc_context(self._paper_rc()):
                    fp = self._savefig(self.fig, f"{stem}_{suffix}{ext}", dpi)
                written.append(fp)
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            for v, val in zip(vars_, saved):
                v.set(val)
            self.fig.set_size_inches(orig)
            self.refresh_all()
            self.canvas.draw_idle()
        # 4: side-by-side RTI pair — DOUBLE the width so EACH panel keeps the
        #    chosen (golden) aspect; extra wspace stops the colorbar/label overlap.
        try:
            with plt.rc_context(self._paper_rc()):
                fig = plt.Figure(figsize=(w * 2, h))
                ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122)
                fig.subplots_adjust(wspace=0.14, left=0.06, right=0.96)
                self._render_rti_panel(fig, ax1, self.Z_mpl_on_proto,
                                       self.title_rti_mpl_var.get(), "mpl")
                self._render_rti_panel(fig, ax2, self.Z_proto,
                                       self.title_rti_proto_var.get(), "proto")
                fp = self._savefig(fig, f"{stem}_rti_pair{ext}", dpi)
            written.append(fp)
        except Exception as e:
            messagebox.showerror("Error", str(e))
        # 5: RTI-as-lines — stacked vertical NRB profiles at each timestamp
        #    (Prototype denoised vs nearest-sample MiniMPL). Wide like the pair.
        try:
            with plt.rc_context(self._paper_rc()):
                fig = plt.Figure(figsize=(w * 2, h))
                ax = fig.add_subplot(111)
                fig.subplots_adjust(left=0.06, right=0.98, bottom=0.16, top=0.92)
                self._render_rti_lines_panel(fig, ax)
                fp = self._savefig(fig, f"{stem}_rti_lines{ext}", dpi)
            written.append(fp)
        except Exception as e:
            messagebox.showerror("Error", str(e))
        if written and announce:
            messagebox.showinfo(f"Saved {len(written)} images", "\n".join(written))
        return written

    def save_profile_png(self):
        if self.r_proto is None:
            messagebox.showinfo("No data", "Load Prototype first."); return
        sel = self._get_sel()
        if not sel:
            messagebox.showinfo("No selection", "Select at least one time."); return
        dlg = FigsizeDialog(self.winfo_toplevel())
        if dlg.result is None:
            return
        w, h, dpi = dlg.result
        ext = self._save_ext()
        fp = filedialog.asksaveasfilename(
            defaultextension=ext,
            filetypes=[("TIFF (CMYK)", "*.tif")] if ext == ".tif" else [("PNG", "*.png")])
        if not fp:
            return
        try:
            with plt.rc_context(self._paper_rc()):
                fig = plt.Figure(figsize=(w, h), facecolor=theme.CARD_BG)
                ax = fig.add_subplot(111)
                ax.set_title(self._title_with_date(self.title_prof_var.get()), pad=10)
                ax.set_xlabel(self.prof_xlabel_var.get())
                ax.set_ylabel(self.prof_ylabel_var.get())
                r = self.r_proto
                both = self.profile_channels.get().startswith("Parallel")
                proto_pref = self.proto_legend_prefix.get().strip()
                mpl_pref = self.mpl_legend_prefix.get().strip()
                for (Z, lbl, nm, dataset, ch) in self._profile_series():
                    if Z is None:
                        continue
                    style = self._profile_curve_style(dataset, ch, both)
                    pref = proto_pref if dataset == "proto" else mpl_pref
                    base = lbl if both else (pref or lbl)
                    for idx in sel:
                        if idx >= Z.shape[1]:
                            continue
                        y = Z[:, idx]; m = np.isfinite(y)
                        if m.any():
                            ax.plot(r[m], y[m], label=f"{base} {self.t_labels_hhmm[idx]}", **style)
                ax.grid(True, alpha=0.25); ax.legend()
                ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
                _tw = "bold" if self.font_bold.get() else "normal"
                for lab in (ax.get_xticklabels() + ax.get_yticklabels()):
                    lab.set_fontweight(_tw)
                fp = self._savefig(fig, fp, dpi)
            messagebox.showinfo("Saved", f"Saved ({w:g}×{h:g} in @ {dpi} DPI):\n{fp}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_mpl_set(self, folder=None, size=None, announce=True):
        """Batch-save the standard MPL figure set as separate profile PNGs:
        co-pol raw, cross-pol raw, NRB profile (parallel), NRB profile
        (perpendicular) — for the currently-selected time(s)."""
        if self.r_proto is None or not self._mpl_mats:
            if announce:
                messagebox.showinfo("No data", "Load Prototype + MPL first.")
            return
        sel = self._get_sel()
        if not sel:
            if announce:
                messagebox.showinfo("No selection", "Select at least one time.")
            return
        if folder is None:
            folder = filedialog.askdirectory(title="Folder to save MPL figures")
            if not folder:
                return
        if size is None:
            dlg = FigsizeDialog(self.winfo_toplevel())
            if dlg.result is None:
                return
            size = dlg.result
        w, h, dpi = size
        ext = self._save_ext()
        # (channel key, filename tag, title name)
        channels = [("raw parallel", "copol_raw", "Co-pol raw"),
                    ("raw perpendicular", "crosspol_raw", "Cross-pol raw"),
                    ("NRB parallel", "NRB_parallel", "NRB co-pol"),
                    ("NRB perpendicular", "NRB_perpendicular", "NRB cross-pol")]
        tlabel = (self.t_labels_hhmm[sel[0]].replace(":", "").replace(" ", "_").replace("-", "")
                  if sel and sel[0] < len(self.t_labels_hhmm) else "")
        # date + time for the per-figure titles
        date_str = (self.proto_date.strftime("%Y-%m-%d")
                    if getattr(self, "proto_date", None) is not None else "")
        _times = [self.t_labels_hhmm[i] for i in sel if i < len(self.t_labels_hhmm)]
        tt = _times[0] if len(_times) == 1 else (f"{_times[0]}-{_times[-1]}" if _times else "")
        saved = (self.profile_source.get(), self.profile_channels.get(), self.mpl_rti_source.get())
        r = self.r_proto
        n = 0
        try:
          with plt.rc_context(self._paper_rc()):   # paper style at axes creation
            self.profile_source.set("Mini MPL")
            self.profile_channels.set("Selected channel")
            for ch_key, tag, title_name in channels:
                if ch_key not in self._mpl_mats:
                    self._safe(self._log, f"[skip] MPL has no '{ch_key}' — re-run Step 1?")
                    continue
                self.mpl_rti_source.set(ch_key)
                fig = plt.Figure(figsize=(w, h), facecolor=theme.CARD_BG)
                ax = fig.add_subplot(111)
                for (Z, lbl, nm, dataset, ch) in self._profile_series():
                    if Z is None:
                        continue
                    for idx in sel:
                        if idx >= Z.shape[1]:
                            continue
                        y = Z[:, idx]; m = np.isfinite(y)
                        if m.any():
                            ax.plot(r[m], y[m], lw=1.6,
                                    label=f"{lbl} {self.t_labels_hhmm[idx]}")
                ax.set_xlabel(self.prof_xlabel_var.get())
                # Raw channels are the MPL photon count rate (counts/us); the NRB
                # channels are the normalised 0-1 backscatter.
                ax.set_ylabel("MPL raw signal (counts/µs)" if "raw" in ch_key
                              else self.prof_ylabel_var.get())
                ax.set_title(f"Mini MPL {title_name} · {date_str} ({tt})".strip(), pad=10)
                ax.grid(True, alpha=0.25)
                if ax.has_data():
                    ax.legend()
                ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
                if "raw" not in ch_key:   # NRB is normalised — hard 0-1
                    ax.set_ylim(0, 1)
                self._savefig(fig, Path(folder) / f"MPL_{tag}_{tlabel}{ext}", dpi)
                n += 1
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.profile_source.set(saved[0]); self.profile_channels.set(saved[1])
            self.mpl_rti_source.set(saved[2])
            self.refresh_all()
        self._log(f"Saved {n} MPL figures -> {folder}")
        if announce:
            messagebox.showinfo("Saved", f"Saved {n} MPL figures to:\n{folder}")
        return n

    def save_proto_set(self, folder=None, size=None, announce=True):
        """Batch-save the prototype figure set as separate profile PNGs: for BOTH
        channels (parallel + perpendicular) x Analog raw / Photon raw / Glue
        Overlay / NRB, from the signal sheets Step 3 writes to the workbook."""
        if self.r_proto is None or not self._proto_sig:
            if announce:
                messagebox.showinfo("No data",
                    "Load a Prototype workbook from Step 3 (one that has the signal "
                    "sheets Analog_*/Photon_*/Glued_*). Re-run Step 3 if missing.")
            return
        sel = self._get_sel()
        if not sel:
            if announce:
                messagebox.showinfo("No selection", "Select at least one time.")
            return
        if folder is None:
            folder = filedialog.askdirectory(title="Folder to save prototype figures")
            if not folder:
                return
        if size is None:
            dlg = FigsizeDialog(self.winfo_toplevel())
            if dlg.result is None:
                return
            size = dlg.result
        w, h, dpi = size
        ext = self._save_ext()
        r = self.r_proto
        date_str = (self.proto_date.strftime("%Y-%m-%d")
                    if getattr(self, "proto_date", None) is not None else "")
        _times = [self.t_labels_hhmm[i] for i in sel if i < len(self.t_labels_hhmm)]
        tt = _times[0] if len(_times) == 1 else (f"{_times[0]}-{_times[-1]}" if _times else "")
        tlabel = (self.t_labels_hhmm[sel[0]].replace(":", "").replace(" ", "_").replace("-", "")
                  if sel and sel[0] < len(self.t_labels_hhmm) else "")
        n = [0]

        def _save(curves, ylabel, mode, chan, fname, norm=False):
          with plt.rc_context(self._paper_rc()):   # paper style at axes creation
            fig = plt.Figure(figsize=(w, h), facecolor=theme.CARD_BG)
            ax = fig.add_subplot(111)
            for Z, lab in curves:
                if Z is None:
                    continue
                for idx in sel:
                    if idx >= Z.shape[1]:
                        continue
                    y = Z[:, idx]; m = np.isfinite(y)
                    if m.any():
                        ax.plot(r[m], y[m], lw=1.5, label=f"{lab} {self.t_labels_hhmm[idx]}")
            ax.set_xlabel(self.prof_xlabel_var.get()); ax.set_ylabel(ylabel)
            ax.set_title(f"Prototype {mode} {chan} · {date_str} ({tt})".strip(), pad=10)
            ax.grid(True, alpha=0.25)
            if ax.has_data():
                ax.legend()
            ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
            if norm:   # normalised NRB — hard 0-1
                ax.set_ylim(0, 1)
            self._savefig(fig, Path(folder) / fname, dpi)
            n[0] += 1

        try:
            sig = self._proto_sig
            for sfx, chan, nrb_key in [("_par", "parallel", "NRB parallel"),
                                       ("_perp", "perpendicular", "NRB perpendicular")]:
                _save([(sig.get("Analog" + sfx), "Analog")], "Analog raw (mV)",
                      "Analog raw", chan, f"prototype_{chan}_AnalogRaw_{tlabel}{ext}")
                _save([(sig.get("Photon" + sfx), "Photon")], "Photon raw (MHz)",
                      "Photon raw", chan, f"prototype_{chan}_PhotonRaw_{tlabel}{ext}")
                _save([(sig.get("PhotonDT" + sfx), "Photon DT"),
                       (sig.get("AnalogScaled" + sfx), "Scaled Analog"),
                       (sig.get("Glued" + sfx), "Glued")], "Signal (MHz)",
                      "Glue Overlay", chan, f"prototype_{chan}_GlueOverlay_{tlabel}{ext}")
                # Honour the Raw/Denoised radio so the saved NRB matches the screen
                den = self._denoised_active() and nrb_key == "NRB parallel"
                _save([(self._proto_mat_for(nrb_key), "NRB (Denoised)" if den else "NRB")],
                      self.prof_ylabel_var.get(),
                      "NRB (Denoised)" if den else "NRB", chan,
                      f"prototype_{chan}_NRB{'_Denoised' if den else ''}_{tlabel}{ext}",
                      norm=True)
        except Exception as e:
            messagebox.showerror("Error", str(e))
        self._log(f"Saved {n[0]} prototype figures -> {folder}")
        if announce:
            messagebox.showinfo("Saved", f"Saved {n[0]} prototype figures to:\n{folder}")
        return n[0]

    def save_nrb_compare_set(self, folder=None, size=None, announce=True):
        """For EACH selected time, save TWO NRB comparison figures (Prototype vs
        Mini MPL overlaid) — one for the parallel channel, one for perpendicular."""
        if self.r_proto is None or not self._proto_mats:
            if announce:
                messagebox.showinfo("No data", "Load Prototype + MPL first.")
            return
        sel = self._get_sel()
        if not sel:
            if announce:
                messagebox.showinfo("No selection", "Select at least one time.")
            return
        if folder is None:
            folder = filedialog.askdirectory(title="Folder to save NRB comparison figures")
            if not folder:
                return
        if size is None:
            dlg = FigsizeDialog(self.winfo_toplevel())
            if dlg.result is None:
                return
            size = dlg.result
        w, h, dpi = size
        ext = self._save_ext()
        r = self.r_proto
        date_str = (self.proto_date.strftime("%Y-%m-%d")
                    if getattr(self, "proto_date", None) is not None else "")
        n = 0
        try:
          with plt.rc_context(self._paper_rc()):
            for ch, chan in [("NRB parallel", "parallel"), ("NRB perpendicular", "perpendicular")]:
                # Honour the Raw/Denoised radio (denoise exists for parallel only)
                proto_Z = self._proto_mat_for(ch)
                den = self._denoised_active() and ch == "NRB parallel"
                den_tag = " (Denoised)" if den else ""
                mpl_Z = self._mpl_channel_on_proto(ch)
                for idx in sel:
                    if idx >= len(self.t_labels_hhmm):
                        continue
                    thm = self.t_labels_hhmm[idx]
                    fig = plt.Figure(figsize=(w, h), facecolor=theme.CARD_BG)
                    ax = fig.add_subplot(111)
                    if proto_Z is not None and idx < proto_Z.shape[1]:
                        y = proto_Z[:, idx]; m = np.isfinite(y)
                        if m.any():
                            ax.plot(r[m], y[m], label=f"Prototype{den_tag} {thm}",
                                    **self._line_style("proto"))
                    if mpl_Z is not None and idx < mpl_Z.shape[1]:
                        y = mpl_Z[:, idx]; m = np.isfinite(y)
                        if m.any():
                            ax.plot(r[m], y[m], label=f"Mini MPL {thm}", **self._line_style("mpl"))
                    ax.set_xlabel(self.prof_xlabel_var.get())
                    ax.set_ylabel(self.prof_ylabel_var.get())
                    ax.set_title(
                        f"Prototype{den_tag} vs Mini MPL · NRB {chan} · {date_str} ({thm})".strip(),
                        pad=10)
                    ax.grid(True, alpha=0.25)
                    if ax.has_data():
                        ax.legend()
                    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
                    ax.set_ylim(0, 1)   # NRB is normalised — hard 0-1 (far-range
                    #                     R^2 background would otherwise exceed 1)
                    tl = thm.replace(":", "").replace(" ", "_").replace("-", "")
                    self._savefig(
                        fig,
                        Path(folder) / f"compare_NRB_{chan}{'_Denoised' if den else ''}_{tl}{ext}",
                        dpi)
                    n += 1
        except Exception as e:
            messagebox.showerror("Error", str(e))
        self._log(f"Saved {n} NRB comparison figures -> {folder}")
        if announce:
            messagebox.showinfo("Saved", f"Saved {n} NRB comparison figures to:\n{folder}")
        return n

    def save_all_organized(self):
        """One button → a full dated picture archive on disk:

            picture-<date>/
              result summary <date>/   4 whole-day figures (Save all 4)
              prototype set/           per-profile-time prototype signals
              MPL set/                 per-profile-time MPL figures
              NRB compare/             per-profile-time Prototype vs MPL

        Asks ONCE for a parent folder + size/DPI. The result-summary set is the
        whole-day overview (selection-independent → saved once). The three profile
        sets are saved ONE FIGURE PER PROFILE TIME — exactly as if you selected
        each time on its own and clicked the button — by looping the time list and
        selecting a single time at a time. Sub-sets whose data is not loaded
        (MPL / prototype signals) are simply skipped."""
        if self.Z_proto is None:
            messagebox.showinfo("No data", "Load Prototype first."); return
        date_tag = self._date_tag_for_filename() or "unknown-date"
        parent = filedialog.askdirectory(
            title=f"Parent folder — will create picture-{date_tag} inside it")
        if not parent:
            return
        dlg = FigsizeDialog(self.winfo_toplevel())
        if dlg.result is None:
            return
        size = dlg.result
        root = Path(parent) / f"picture-{date_tag}"
        summary  = root / f"result summary {date_tag}"
        proto_dir = root / "prototype set"
        mpl_dir   = root / "MPL set"
        nrb_dir   = root / "NRB compare"
        for d in (summary, proto_dir, mpl_dir, nrb_dir):
            d.mkdir(parents=True, exist_ok=True)
        saved_sel = list(self.time_list.curselection())
        counts = {"result summary": 0, "MPL set": 0, "prototype set": 0, "NRB compare": 0}
        try:
            # 1) Whole-day overview — does not depend on the time selection → once.
            w = self.save_all_figures(folder=summary, size=size, announce=False)
            counts["result summary"] = len(w) if w else 0
            # 2) The 3 sets: one figure per profile time (select a single time,
            #    save, repeat) so each time gets its own image, not an overlay.
            for i in range(len(self.t_labels_hhmm)):
                self.time_list.selection_clear(0, "end")
                self.time_list.selection_set(i)
                m  = self.save_mpl_set(folder=mpl_dir, size=size, announce=False)
                p  = self.save_proto_set(folder=proto_dir, size=size, announce=False)
                nb = self.save_nrb_compare_set(folder=nrb_dir, size=size, announce=False)
                counts["MPL set"]       += (m or 0)
                counts["prototype set"] += (p or 0)
                counts["NRB compare"]   += (nb or 0)
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.time_list.selection_clear(0, "end")
            for i in saved_sel:
                self.time_list.selection_set(i)
            self.refresh_all()
        lines = [f"{name}: {c} figures"
                 + ("  (skipped — data not loaded)" if not c and name != "result summary" else "")
                 for name, c in counts.items()]
        self._log(f"Saved full picture archive -> {root}")
        messagebox.showinfo("Saved all", f"Saved to:\n{root}\n\n" + "\n".join(lines))


# ═════════════════════════════════════════════════════════════════════════════
# Step 5 Page  –  Fernald / Klett Aerosol Inversion
# ═════════════════════════════════════════════════════════════════════════════
FERNALD_PLOT_MODES = [
    "beta_aer profile",
    "alpha_aer profile",
    "AOD time series",
    "mass_aer profile",
    "AERONET vs Fernald AOD",
]


class Step5Page(ctk.CTkFrame):
    def __init__(self, master, app_state: AppState):
        super().__init__(master, fg_color="transparent")
        self.app_state = app_state

        # ── State ───────────────────────────────────────────────────────────
        self.nrb_path   = tk.StringVar()
        self.nrb_sheet  = tk.StringVar(value="NRB profile")
        self.out_path   = tk.StringVar()
        self.method     = tk.StringVar(value="Fernald 1984")
        self.r_ref_m    = tk.DoubleVar(value=5000.0)
        self.sa_sr      = tk.DoubleVar(value=50.0)
        # δ-driven lidar ratio: use depol δ_v to pick a height-resolved S_a(R)
        # instead of the single constant above (sa_sr becomes the fallback).
        self.use_delta_sa = tk.BooleanVar(value=False)
        self.delta_file   = tk.StringVar(value="")
        self.atm_mode   = tk.StringVar(value="US Standard Atm")
        self.t_k        = tk.DoubleVar(value=288.15)
        self.p_pa       = tk.DoubleVar(value=101325.0)
        # Mass concentration (Track 1, item 5)
        _preset_names = list(_SIGMA_PRESETS.keys()) or ["Mixed continental (default)"]
        self.mass_enable    = tk.BooleanVar(value=True)
        self.mass_preset    = tk.StringVar(value=_preset_names[0])
        _default_sig = float(_SIGMA_PRESETS.get(_preset_names[0], 3.5))
        self.sigma_ext      = tk.DoubleVar(value=_default_sig)
        # AERONET validation (Track 1, item 4)
        self.aeronet_enable = tk.BooleanVar(value=False)
        self.aeronet_path   = tk.StringVar(value="")
        self.aeronet_tol_min = tk.DoubleVar(value=15.0)

        self.progress   = tk.DoubleVar(value=0.0)
        self.plot_mode  = tk.StringVar(value="beta_aer profile")
        self.chart_title_var = tk.StringVar()

        # Results kept in memory for plotting
        self._res: Optional[Dict[str, pd.DataFrame]] = None
        self._ts_labels: List[str] = []

        if not _HAS_FERNALD:
            self._build_missing_view()
            return

        self._build_ui()
        self.app_state.register_cb(self._sync_from_state)
        self._sync_from_state()
        self._update_atm_ui()

    # ── Backend missing fallback ───────────────────────────────────────────
    def _build_missing_view(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        card = Card(self, title="Backend missing", icon="⚠️")
        card.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(
            card.body,
            text="fernald_engine.py was not found alongside this script.\n"
                 "Step 5 needs that module to run the aerosol inversion.",
            font=theme.F_BODY, text_color=theme.TEXT_SECONDARY, justify="left",
        ).pack(pady=20)

    # ── Layout ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = PageHeader(
            self,
            title="Step 4 · Fernald / Klett Inversion",
            subtitle="Retrieve aerosol backscatter, extinction and AOD from the NRB profile",
            badge=("v1.0", "navy"),
        )
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))

        self.paned = tk.PanedWindow(
            self, orient="horizontal", bg=theme.APP_BG,
            sashrelief="flat", sashwidth=8, sashpad=0, bd=0, showhandle=False,
        )
        self.paned.grid(row=1, column=0, sticky="nsew")

        left_pane = tk.Frame(self.paned, bg=theme.APP_BG, bd=0, highlightthickness=0)
        left_pane.grid_columnconfigure(0, weight=1)
        left_pane.grid_rowconfigure(0, weight=1)
        self.paned.add(left_pane, minsize=380, width=560, stretch="always")

        left = ctk.CTkScrollableFrame(left_pane, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_columnconfigure(0, weight=1)
        self._build_io_card(left).grid(row=0, column=0, sticky="ew", pady=(0, 12), padx=(0, 10))
        self._build_params_card(left).grid(row=1, column=0, sticky="ew", pady=(0, 12), padx=(0, 10))
        self._build_mass_aeronet_card(left).grid(row=2, column=0, sticky="ew", pady=(0, 12), padx=(0, 10))
        self._build_run_card(left).grid(row=3, column=0, sticky="ew", pady=(0, 12), padx=(0, 10))
        self._build_console_card(left).grid(row=4, column=0, sticky="nsew", padx=(0, 10))

        right_pane = tk.Frame(self.paned, bg=theme.APP_BG, bd=0, highlightthickness=0)
        right_pane.grid_columnconfigure(0, weight=1)
        right_pane.grid_rowconfigure(0, weight=1)
        self.paned.add(right_pane, minsize=380, stretch="always")

        right = ctk.CTkFrame(right_pane, fg_color="transparent")
        right.grid(row=0, column=0, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)
        self._build_plot_card(right).grid(row=0, column=0, sticky="nsew")

    # ── Card: I/O ──────────────────────────────────────────────────────────
    def _build_io_card(self, parent) -> "Card":
        card = Card(parent, title="Input / Output", icon="📁")
        body = card.body
        body.grid_columnconfigure(0, weight=1)

        # NRB workbook
        ctk.CTkLabel(body, text="NRB workbook (from Step 2)", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY, anchor="w").grid(row=0, column=0, sticky="w")
        row1 = ctk.CTkFrame(body, fg_color="transparent")
        row1.grid(row=1, column=0, sticky="ew", pady=(2, 8))
        row1.grid_columnconfigure(0, weight=1)
        ctk.CTkEntry(row1, textvariable=self.nrb_path, **theme.input_style()).grid(
            row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(row1, text="Browse…", command=self._pick_nrb, width=100,
                      **theme.secondary_button_style()).grid(row=0, column=1)

        # Sheet
        sh_row = ctk.CTkFrame(body, fg_color="transparent")
        sh_row.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        sh_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(sh_row, text="NRB sheet:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 8))
        self.sheet_box = ctk.CTkOptionMenu(
            sh_row, variable=self.nrb_sheet, values=["NRB profile"],
            fg_color=theme.CARD_BG, button_color=theme.CARD_BG, button_hover_color=theme.CREAM_SOFT,
            text_color=theme.TEXT_PRIMARY, dropdown_fg_color=theme.CARD_BG,
            dropdown_text_color=theme.TEXT_PRIMARY, dropdown_hover_color=theme.CREAM,
            corner_radius=theme.RADIUS_INPUT, height=32, font=theme.F_BODY, width=240,
        )
        self.sheet_box.grid(row=0, column=1, sticky="w")

        # Output
        ctk.CTkLabel(body, text="Output workbook", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY, anchor="w").grid(row=3, column=0, sticky="w")
        row3 = ctk.CTkFrame(body, fg_color="transparent")
        row3.grid(row=4, column=0, sticky="ew", pady=(2, 0))
        row3.grid_columnconfigure(0, weight=1)
        ctk.CTkEntry(row3, textvariable=self.out_path, **theme.input_style()).grid(
            row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(row3, text="Save As…", command=self._pick_output, width=100,
                      **theme.secondary_button_style()).grid(row=0, column=1)

        ctk.CTkLabel(
            body,
            text="Reads the NRB profile (range-corrected signal) and writes a separate "
                 "Fernald-<date>.xlsx with sheets: Fernald_beta_aer, Fernald_alpha_aer, Fernald_AOD.",
            font=(theme.FONT_FAMILY, 10), text_color=theme.TEXT_MUTED, anchor="w",
            wraplength=480, justify="left",
        ).grid(row=5, column=0, sticky="w", pady=(8, 0))

        return card

    # ── Card: Parameters ───────────────────────────────────────────────────
    def _build_params_card(self, parent) -> "Card":
        card = Card(parent, title="Inversion Parameters", icon="⚙️")
        body = card.body
        body.grid_columnconfigure(0, weight=1)

        # Method & reference range
        self._subsection(body, "Method & reference range").grid(row=0, column=0, sticky="ew")
        meth_row = ctk.CTkFrame(body, fg_color="transparent")
        meth_row.grid(row=1, column=0, sticky="ew", pady=(2, 6))
        meth_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(meth_row, text="Method:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 8))
        ctk.CTkOptionMenu(
            meth_row, variable=self.method, values=["Fernald 1984", "Klett 1981"],
            fg_color=theme.CARD_BG, button_color=theme.CARD_BG, button_hover_color=theme.CREAM_SOFT,
            text_color=theme.TEXT_PRIMARY, dropdown_fg_color=theme.CARD_BG,
            dropdown_text_color=theme.TEXT_PRIMARY, dropdown_hover_color=theme.CREAM,
            corner_radius=theme.RADIUS_INPUT, height=32, font=theme.F_BODY, width=180,
        ).grid(row=0, column=1, sticky="w")

        pr = ctk.CTkFrame(body, fg_color="transparent")
        pr.grid(row=2, column=0, sticky="ew", pady=(0, 4))
        for i in range(2):
            pr.grid_columnconfigure(i, weight=1)
        FieldRow(pr, "R_ref (m) — clean-air range", self.r_ref_m).grid(
            row=0, column=0, sticky="ew", padx=(0, 6))
        FieldRow(pr, "S_a (sr) — aerosol lidar ratio", self.sa_sr).grid(
            row=0, column=1, sticky="ew", padx=(6, 0))

        ctk.CTkLabel(
            body,
            text="R_ref: altitude where aerosol ≈ 0 (typically 4000–8000 m for nighttime TR40).  "
                 "S_a [sr] @ 532 nm: urban 50–80, marine ~25, biomass smoke 70–100.",
            font=theme.F_TINY, text_color=theme.TEXT_MUTED, anchor="w",
            wraplength=480, justify="left",
        ).grid(row=3, column=0, sticky="w", pady=(0, 8))

        # δ-driven height-resolved S_a (from Step 3 depol workbook)
        dsa = ctk.CTkFrame(body, fg_color="transparent")
        dsa.grid(row=4, column=0, sticky="ew", pady=(0, 6))
        dsa.grid_columnconfigure(1, weight=1)
        ctk.CTkCheckBox(
            dsa, text="Use δ for S_a (height-resolved; S_a above = fallback)",
            variable=self.use_delta_sa,
            font=theme.F_SMALL, text_color=theme.TEXT_PRIMARY,
            fg_color=theme.ORANGE, hover_color=theme.ORANGE_HOVER,
            border_color=theme.BORDER, checkmark_color="#fff",
            corner_radius=4, border_width=2,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 2))
        ctk.CTkLabel(dsa, text="Depol workbook:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=1, column=0, padx=(0, 8))
        ctk.CTkEntry(dsa, textvariable=self.delta_file, **theme.input_style()).grid(
            row=1, column=1, sticky="ew", padx=(0, 8))
        ctk.CTkButton(dsa, text="…", width=36,
                      command=self._pick_delta_file,
                      **theme.secondary_button_style(height=34, font=theme.F_SMALL)).grid(
            row=1, column=2)

        # Atmosphere
        self._subsection(body, "Atmospheric profile").grid(row=5, column=0, sticky="ew")
        atm_row = ctk.CTkFrame(body, fg_color="transparent")
        atm_row.grid(row=6, column=0, sticky="ew", pady=(2, 4))
        atm_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(atm_row, text="T / P source:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 8))
        ctk.CTkOptionMenu(
            atm_row, variable=self.atm_mode,
            values=["US Standard Atm", "Manual (uniform)"],
            command=lambda _v: self._update_atm_ui(),
            fg_color=theme.CARD_BG, button_color=theme.CARD_BG, button_hover_color=theme.CREAM_SOFT,
            text_color=theme.TEXT_PRIMARY, dropdown_fg_color=theme.CARD_BG,
            dropdown_text_color=theme.TEXT_PRIMARY, dropdown_hover_color=theme.CREAM,
            corner_radius=theme.RADIUS_INPUT, height=32, font=theme.F_BODY, width=220,
        ).grid(row=0, column=1, sticky="w")

        man = ctk.CTkFrame(body, fg_color="transparent")
        man.grid(row=7, column=0, sticky="ew", pady=(0, 4))
        for i in range(2):
            man.grid_columnconfigure(i, weight=1)
        self._t_field = FieldRow(man, "T (K)", self.t_k)
        self._t_field.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self._p_field = FieldRow(man, "P (Pa)", self.p_pa)
        self._p_field.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        ctk.CTkLabel(
            body,
            text="US Standard Atm (ISO 2533): T(h)=288.15−0.0065h K, P from the barometric "
                 "formula. Manual: one uniform T/P for the whole column.",
            font=theme.F_TINY, text_color=theme.TEXT_MUTED, anchor="w",
            wraplength=480, justify="left",
        ).grid(row=8, column=0, sticky="w", pady=(0, 0))

        return card

    @staticmethod
    def _subsection(parent, text: str) -> ctk.CTkLabel:
        return ctk.CTkLabel(
            parent, text=text.upper(), font=(theme.FONT_FAMILY, 10, "bold"),
            text_color=theme.TEXT_MUTED, anchor="w",
        )

    # ── Card: Mass concentration & AERONET validation ──────────────────────
    def _build_mass_aeronet_card(self, parent) -> "Card":
        card = Card(parent, title="Mass concentration & AERONET", icon="📡")
        body = card.body
        body.grid_columnconfigure(0, weight=1)

        check_kw = dict(
            font=theme.F_SMALL, text_color=theme.TEXT_PRIMARY,
            fg_color=theme.ORANGE, hover_color=theme.ORANGE_HOVER,
            border_color=theme.BORDER, checkmark_color="#fff",
            corner_radius=4, border_width=2,
            onvalue=True, offvalue=False,
        )

        # ── Mass concentration ─────────────────────────────────────────────
        self._subsection(body, "Mass concentration  (M = α_aer / σ_ext)").grid(
            row=0, column=0, sticky="ew", pady=(0, 4))
        mr0 = ctk.CTkFrame(body, fg_color="transparent")
        mr0.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        cb = ctk.CTkCheckBox(mr0, text="Compute mass profile [µg/m³]",
                             variable=self.mass_enable,
                             command=self._update_mass_aeronet_ui, **check_kw)
        cb.pack(side="left")
        if bool(self.mass_enable.get()):
            cb.select()
        else:
            cb.deselect()

        mr1 = ctk.CTkFrame(body, fg_color="transparent")
        mr1.grid(row=2, column=0, sticky="ew", pady=(0, 4))
        mr1.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(mr1, text="Particle type:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 8))
        preset_vals = list(_SIGMA_PRESETS.keys()) or ["Mixed continental (default)"]
        self._mass_preset_box = ctk.CTkOptionMenu(
            mr1, variable=self.mass_preset, values=preset_vals,
            command=lambda v: self._on_mass_preset(v),
            fg_color=theme.CARD_BG, button_color=theme.CARD_BG,
            button_hover_color=theme.CREAM_SOFT,
            text_color=theme.TEXT_PRIMARY, dropdown_fg_color=theme.CARD_BG,
            dropdown_text_color=theme.TEXT_PRIMARY,
            dropdown_hover_color=theme.CREAM,
            corner_radius=theme.RADIUS_INPUT, height=32,
            font=theme.F_BODY, width=320,
        )
        self._mass_preset_box.grid(row=0, column=1, sticky="w")

        mr2 = ctk.CTkFrame(body, fg_color="transparent")
        mr2.grid(row=3, column=0, sticky="ew", pady=(0, 4))
        mr2.grid_columnconfigure(1, weight=0)
        ctk.CTkLabel(mr2, text="σ_ext (m²/g):", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 8))
        self._sigma_entry = ctk.CTkEntry(
            mr2, textvariable=self.sigma_ext, width=80, **theme.input_style())
        self._sigma_entry.grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(
            mr2,
            text="(preset auto-fills; you can override)",
            font=theme.F_TINY, text_color=theme.TEXT_MUTED,
        ).grid(row=0, column=2, sticky="w", padx=(8, 0))

        # ── AERONET validation ─────────────────────────────────────────────
        self._subsection(body, "AERONET AOD validation").grid(
            row=4, column=0, sticky="ew", pady=(8, 4))

        if not _HAS_AERONET:
            ctk.CTkLabel(body, text="aeronet_loader.py not found — validation disabled.",
                         font=theme.F_TINY, text_color=theme.RED,
                         anchor="w").grid(row=5, column=0, sticky="w")
            return card

        ar0 = ctk.CTkFrame(body, fg_color="transparent")
        ar0.grid(row=5, column=0, sticky="ew", pady=(0, 4))
        self._aer_cb = ctk.CTkCheckBox(
            ar0, text="Compare Fernald AOD vs AERONET",
            variable=self.aeronet_enable,
            command=self._update_mass_aeronet_ui, **check_kw)
        self._aer_cb.pack(side="left")
        if bool(self.aeronet_enable.get()):
            self._aer_cb.select()
        else:
            self._aer_cb.deselect()

        ar1 = ctk.CTkFrame(body, fg_color="transparent")
        ar1.grid(row=6, column=0, sticky="ew", pady=(0, 4))
        ar1.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(ar1, text="AERONET file:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 8))
        self._aer_entry = ctk.CTkEntry(
            ar1, textvariable=self.aeronet_path, **theme.input_style())
        self._aer_entry.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self._aer_btn = ctk.CTkButton(
            ar1, text="Browse…", command=self._pick_aeronet, width=90,
            **theme.secondary_button_style(height=34, font=theme.F_SMALL))
        self._aer_btn.grid(row=0, column=2)

        ar2 = ctk.CTkFrame(body, fg_color="transparent")
        ar2.grid(row=7, column=0, sticky="ew", pady=(0, 4))
        ctk.CTkLabel(ar2, text="Time-match tolerance (min):", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 8))
        self._aer_tol_entry = ctk.CTkEntry(
            ar2, textvariable=self.aeronet_tol_min, width=70, **theme.input_style())
        self._aer_tol_entry.grid(row=0, column=1, sticky="w")

        ctk.CTkLabel(
            body,
            text="Accepts AERONET V3 Level 1.0/1.5/2.0 .lev* files. AOD_532nm used directly "
                 "if present, otherwise Ångström-interpolated from 500 + 440 nm.",
            font=theme.F_TINY, text_color=theme.TEXT_MUTED, anchor="w",
            wraplength=480, justify="left",
        ).grid(row=8, column=0, sticky="w", pady=(0, 0))

        self._update_mass_aeronet_ui()
        return card

    def _on_mass_preset(self, name: str):
        if name in _SIGMA_PRESETS:
            self.sigma_ext.set(float(_SIGMA_PRESETS[name]))

    def _pick_aeronet(self):
        p = filedialog.askopenfilename(
            title="AERONET AOD file",
            filetypes=[("AERONET", "*.lev10 *.lev15 *.lev20 *.txt *.csv"),
                       ("All files", "*.*")],
        )
        if p:
            self.aeronet_path.set(p)

    def _update_mass_aeronet_ui(self):
        if not hasattr(self, "_sigma_entry"):
            return
        mst = "normal" if bool(self.mass_enable.get()) else "disabled"
        for w in (self._mass_preset_box, self._sigma_entry):
            try: w.configure(state=mst)
            except Exception: pass
        if hasattr(self, "_aer_entry"):
            ast_ = "normal" if (bool(self.aeronet_enable.get()) and _HAS_AERONET) else "disabled"
            for w in (self._aer_entry, self._aer_btn, self._aer_tol_entry):
                try: w.configure(state=ast_)
                except Exception: pass

    # ── Card: Run ──────────────────────────────────────────────────────────
    def _build_run_card(self, parent) -> "Card":
        card = Card(parent)
        body = card.body
        body.grid_columnconfigure(1, weight=1)

        self.run_btn = ctk.CTkButton(
            body, text="▶  Run Fernald Inversion", command=self.run,
            **theme.primary_button_style(width=240),
        )
        self.run_btn.grid(row=0, column=0, sticky="w")

        self.pb = ctk.CTkProgressBar(
            body, variable=self.progress, progress_color=theme.ORANGE,
            fg_color=theme.LIGHT_GRAY, height=8, corner_radius=4,
        )
        self.pb.set(0)
        self.pb.grid(row=0, column=1, sticky="ew", padx=(16, 0))

        self.status = StatusBar(body)
        self.status.grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))
        return card

    def _build_console_card(self, parent) -> "ConsoleLog":
        self.console = ConsoleLog(parent, title="Console Log")
        return self.console

    # ── Card: Plot ─────────────────────────────────────────────────────────
    def _build_plot_card(self, parent) -> "Card":
        card = Card(parent, title="Aerosol Plot", icon="🔬")
        body = card.body
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(3, weight=1)

        ctrl = ctk.CTkFrame(body, fg_color="transparent")
        ctrl.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ctk.CTkLabel(ctrl, text="Plot:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).pack(side="left")
        ctk.CTkOptionMenu(
            ctrl, variable=self.plot_mode, values=FERNALD_PLOT_MODES,
            command=lambda _v: self.refresh_plot(),
            fg_color=theme.CARD_BG, button_color=theme.CARD_BG, button_hover_color=theme.CREAM_SOFT,
            text_color=theme.TEXT_PRIMARY, dropdown_fg_color=theme.CARD_BG,
            dropdown_text_color=theme.TEXT_PRIMARY, dropdown_hover_color=theme.CREAM,
            corner_radius=theme.RADIUS_INPUT, height=32, font=theme.F_BODY, width=180,
        ).pack(side="left", padx=(6, 12))
        ctk.CTkButton(ctrl, text="Refresh", command=self.refresh_plot,
                      **theme.primary_button_style(width=90, height=32)).pack(side="left", padx=(0, 6))
        ctk.CTkButton(ctrl, text="PNG", command=self.save_png,
                      **theme.ghost_button_style(width=60, height=32)).pack(side="left")

        ctk.CTkLabel(body, text="Chart title", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=1, column=0, sticky="w")
        ctk.CTkEntry(body, textvariable=self.chart_title_var, **theme.input_style()).grid(
            row=1, column=0, sticky="e")

        # Times list (for profile modes)
        sel_row = ctk.CTkFrame(body, fg_color="transparent")
        sel_row.grid(row=2, column=0, sticky="ew", pady=(6, 6))
        sel_row.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(sel_row, text="Profiles (multi-select)", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY, anchor="w").grid(row=0, column=0, sticky="w")
        list_wrap = ctk.CTkFrame(sel_row, fg_color=theme.CREAM_SOFT,
                                 corner_radius=theme.RADIUS_INPUT, border_width=1,
                                 border_color=theme.BORDER, height=110)
        list_wrap.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        list_wrap.grid_columnconfigure(0, weight=1)
        list_wrap.grid_rowconfigure(0, weight=1)
        list_wrap.grid_propagate(False)
        self.time_list = tk.Listbox(
            list_wrap, selectmode="extended", exportselection=False, height=5,
            bg=theme.CREAM_SOFT, fg=theme.TEXT_PRIMARY,
            selectbackground=theme.ORANGE_SOFT, selectforeground=theme.ORANGE,
            relief="flat", borderwidth=0, highlightthickness=0, font=(theme.FONT_FAMILY, 11),
        )
        self.time_list.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        btns = ctk.CTkFrame(list_wrap, fg_color="transparent")
        btns.grid(row=0, column=1, sticky="ns", padx=(0, 8), pady=8)
        ctk.CTkButton(btns, text="Select All",
                      command=lambda: self.time_list.selection_set(0, "end"),
                      **theme.ghost_button_style(width=90, height=28)).pack(pady=(0, 4))
        ctk.CTkButton(btns, text="Clear",
                      command=lambda: self.time_list.selection_clear(0, "end"),
                      **theme.ghost_button_style(width=90, height=28)).pack()

        canvas_wrap = ctk.CTkFrame(body, fg_color=theme.CARD_BG, corner_radius=theme.RADIUS_INPUT,
                                   border_width=1, border_color=theme.BORDER)
        canvas_wrap.grid(row=3, column=0, sticky="nsew", pady=(6, 0))
        canvas_wrap.grid_columnconfigure(0, weight=1)
        canvas_wrap.grid_rowconfigure(0, weight=1)
        self.fig = plt.Figure(figsize=(7, 5.0), dpi=100, facecolor=theme.CARD_BG)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("No data — run the inversion first")
        self.ax.grid(True, alpha=0.3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_wrap)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        return card

    # ── Helpers ────────────────────────────────────────────────────────────
    def _log(self, msg): self.console.log(msg)

    def _safe(self, fn, *a, **kw):
        self.after(0, lambda: fn(*a, **kw))

    def _safe_log(self, msg): self._safe(self._log, msg)

    def _update_atm_ui(self):
        if not hasattr(self, "_t_field"):
            return
        manual = self.atm_mode.get() == "Manual (uniform)"
        st = "normal" if manual else "disabled"
        for fld in (self._t_field, self._p_field):
            try:
                fld.entry.configure(state=st)
            except Exception:
                pass

    def _sync_from_state(self):
        """Auto-fill the NRB path from Step 2's output if still empty."""
        if not self.nrb_path.get().strip() and self.app_state.step2_output:
            self.nrb_path.set(self.app_state.step2_output)
            self._refresh_sheets()
            if not self.out_path.get().strip():
                self.out_path.set(self._suggest_out_path(self.app_state.step2_output))
        # Auto-chain: Depolarization (Step 3) workbook → δ source for S_a, and
        # turn the δ-driven S_a on by default once δ becomes available.
        dep = getattr(self.app_state, "depol_output", None)
        if dep and not self.delta_file.get().strip():
            self.delta_file.set(dep)
            self.use_delta_sa.set(True)

    @staticmethod
    def _suggest_out_path(nrb_path: str) -> str:
        p = Path(nrb_path)
        stem = p.stem
        if stem.upper().startswith("NRB"):
            new = "Fernald" + stem[3:]
        else:
            new = "Fernald-" + stem
        return str(p.with_name(new + ".xlsx"))

    def _pick_nrb(self):
        p = filedialog.askopenfilename(
            title="NRB workbook", filetypes=[("Excel", "*.xlsx *.xls")])
        if p:
            self.nrb_path.set(p)
            self._refresh_sheets()
            if not self.out_path.get().strip():
                self.out_path.set(self._suggest_out_path(p))

    def _pick_delta_file(self):
        p = filedialog.askopenfilename(
            title="Depol workbook (Step 3 output) for δ-driven S_a",
            filetypes=[("Excel", "*.xlsx *.xls")])
        if p:
            self.delta_file.set(p)
            self.use_delta_sa.set(True)

    def _pick_output(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if p:
            self.out_path.set(p)

    def _refresh_sheets(self):
        path = self.nrb_path.get().strip()
        if not path or not os.path.isfile(path):
            return
        try:
            names = pd.ExcelFile(path).sheet_names
        except Exception as e:
            self._log(f"[WARN] could not read sheets: {e}")
            return
        self.sheet_box.configure(values=names)
        if self.nrb_sheet.get() not in names:
            preferred = next((s for s in names if "nrb" in s.lower()), names[0])
            self.nrb_sheet.set(preferred)

    # ── Run ────────────────────────────────────────────────────────────────
    def run(self):
        if not _HAS_FERNALD:
            messagebox.showerror("Missing", "fernald_engine.py not found.")
            return
        nrb = self.nrb_path.get().strip()
        out = self.out_path.get().strip()
        sheet = self.nrb_sheet.get().strip()
        if not nrb or not os.path.isfile(nrb):
            messagebox.showerror("Error", "Choose a valid NRB workbook.")
            return
        if not out:
            messagebox.showerror("Error", "Choose an output path.")
            return
        try:
            r_ref = float(self.r_ref_m.get())
            sa = float(self.sa_sr.get())
        except Exception:
            messagebox.showerror("Error", "R_ref and S_a must be numbers.")
            return
        if sa <= 0:
            messagebox.showerror("Error", "S_a must be > 0 sr.")
            return

        self.run_btn.configure(state="disabled")
        self.progress.set(0.0); self.pb.set(0.0)
        self.status.set("Running…", "running")
        self._log("=== START Fernald inversion ===")

        method = "fernald" if "Fernald" in self.method.get() else "klett"
        manual = self.atm_mode.get() == "Manual (uniform)"
        T_sc = float(self.t_k.get()) if manual else None
        P_sc = float(self.p_pa.get()) if manual else None

        # Mass + AERONET options
        sig_ext = None
        if bool(self.mass_enable.get()):
            try:
                v = float(self.sigma_ext.get())
                if v > 0:
                    sig_ext = v
            except Exception:
                sig_ext = None

        aer_path = self.aeronet_path.get().strip()
        do_aer = (_HAS_AERONET and bool(self.aeronet_enable.get())
                  and aer_path and os.path.isfile(aer_path))
        try:
            aer_tol = float(self.aeronet_tol_min.get())
        except Exception:
            aer_tol = 15.0

        # δ-driven S_a: read the depol workbook's Depol_delta_v sheet if enabled.
        delta_path = self.delta_file.get().strip()
        do_delta = (bool(self.use_delta_sa.get())
                    and delta_path and os.path.isfile(delta_path))

        def worker():
            try:
                df_delta = None
                if do_delta:
                    try:
                        df_delta = pd.read_excel(delta_path, sheet_name="Depol_delta_v")
                        df_delta = df_delta.rename(columns={df_delta.columns[0]: "Range(m)"})
                        self._safe(self._log,
                                   f"δ-driven S_a ON — read δ_v from {Path(delta_path).name}")
                    except Exception as de:
                        df_delta = None
                        self._safe(self._log,
                                   f"[WARN] δ workbook unreadable ({de}) — using constant S_a")
                df = pd.read_excel(nrb, sheet_name=sheet)
                if df.shape[1] < 2:
                    raise ValueError(
                        f"Sheet '{sheet}' has fewer than 2 columns — "
                        "expected range + at least one profile.")
                df = df.rename(columns={df.columns[0]: "Range(m)"})
                n_prof = df.shape[1] - 1
                self._safe(self._log,
                           f"Loaded '{sheet}': {df.shape[0]} range bins · {n_prof} profiles")
                self._safe(self.progress.set, 20.0); self._safe(self.pb.set, 0.20)

                res = _fernald_run(
                    df, R_ref_m=r_ref, lidar_ratio_aer=sa,
                    method=method, T_K_scalar=T_sc, P_Pa_scalar=P_sc,
                    sigma_ext_m2_per_g=sig_ext,
                    df_delta=df_delta,
                )
                self._safe(self.progress.set, 55.0); self._safe(self.pb.set, 0.55)

                # ── AERONET comparison (optional) ──────────────────────────
                df_aer_cmp = None
                if do_aer:
                    try:
                        self._safe(self._log, f"Loading AERONET: {Path(aer_path).name}")
                        aer_df = _aeronet_load(aer_path)
                        df_aer_cmp = _aeronet_match(aer_df, res["AOD"], tol_min=aer_tol)
                        stats = _aeronet_stats(df_aer_cmp)
                        self._safe(self._log,
                                   f"AERONET match: N={stats['n']} · MAE={stats['MAE']:.3f} "
                                   f"· bias={stats['bias']:+.3f} · RMSE={stats['RMSE']:.3f}"
                                   + (f" · R²={stats['R2']:.2f}" if np.isfinite(stats['R2']) else ""))
                        res["AERONET_comparison"] = df_aer_cmp
                    except Exception as ae:
                        self._safe(self._log, f"[WARN] AERONET: {ae}")

                self._safe(self.progress.set, 80.0); self._safe(self.pb.set, 0.80)

                # ── Write Excel ────────────────────────────────────────────
                with pd.ExcelWriter(out, engine="openpyxl") as xw:
                    res["beta_aer"].to_excel(xw, index=False, sheet_name="Fernald_beta_aer")
                    res["alpha_aer"].to_excel(xw, index=False, sheet_name="Fernald_alpha_aer")
                    res["AOD"].to_excel(xw, index=False, sheet_name="Fernald_AOD")
                    if "lidar_ratio" in res:
                        res["lidar_ratio"].to_excel(xw, index=False, sheet_name="Fernald_lidar_ratio")
                    if "mass_aer" in res:
                        res["mass_aer"].to_excel(xw, index=False, sheet_name="Fernald_mass_aer")
                    if df_aer_cmp is not None:
                        df_aer_cmp.to_excel(xw, index=False, sheet_name="AERONET_comparison")

                aod_df = res["AOD"]
                meta = {"R_ref_m", "lidar_ratio_aer_sr", "method"}
                aod_vals = aod_df[[c for c in aod_df.columns
                                   if c not in meta]].to_numpy(float).flatten()
                aod_mean = float(np.nanmean(aod_vals)) if aod_vals.size else float("nan")

                self._res = res
                self._safe(self._populate_times)
                msg = (f"Fernald ({method}): R_ref={r_ref:.0f} m · S_a={sa:.0f} sr · "
                       f"mean AOD = {aod_mean:.3f}")
                if sig_ext:
                    mass_peak = float(np.nanmax(res["mass_aer"].iloc[:, 1:].to_numpy(float)))
                    msg += f" · σ_ext={sig_ext:.1f} m²/g · peak mass={mass_peak:.1f} µg/m³"
                self._safe(self._log, msg)
                self._safe(self._log, f"[OK] Saved: {Path(out).resolve()}")
                self._safe(self.progress.set, 100.0); self._safe(self.pb.set, 1.0)
                self._safe(self.status.set, "Done", "ok")
                self._safe(self.refresh_plot)
                self._safe(messagebox.showinfo, "Step 5 Complete", f"Saved:\n{out}")
            except Exception as e:
                self._safe(self._log, f"[FAILED] {e}")
                self._safe(self.status.set, "Failed", "error")
                self._safe(messagebox.showerror, "Failed", str(e))
            finally:
                self._safe(self.run_btn.configure, state="normal")
                self._safe(self._log, "=== END ===")

        threading.Thread(target=worker, daemon=True).start()

    # ── Plotting ───────────────────────────────────────────────────────────
    @staticmethod
    def _col_label(c) -> str:
        ts = pd.to_datetime(c, errors="coerce")
        if pd.notna(ts):
            return ts.strftime("%H:%M")
        return str(c)

    def _populate_times(self):
        self.time_list.delete(0, "end")
        self._ts_labels = []
        if self._res is None:
            return
        cols = [c for c in self._res["beta_aer"].columns if c != "Range(m)"]
        for c in cols:
            lbl = self._col_label(c)
            self._ts_labels.append(lbl)
            self.time_list.insert("end", lbl)
        if self._ts_labels:
            self.time_list.selection_set(0)

    def _selected_indices(self) -> List[int]:
        sel = list(self.time_list.curselection())
        if sel:
            return sel
        return [0] if self._ts_labels else []

    def refresh_plot(self):
        self.ax.clear()
        if self._res is None:
            self.ax.set_title("No data — run the inversion first")
            self.ax.grid(True, alpha=0.3)
            self.canvas.draw_idle()
            return

        mode = self.plot_mode.get()
        try:
            if mode == "AOD time series":
                self._plot_aod()
            elif mode == "mass_aer profile":
                self._plot_mass()
            elif mode == "AERONET vs Fernald AOD":
                self._plot_aeronet()
            elif mode.startswith("alpha"):
                self._plot_profiles("alpha_aer")
            else:
                self._plot_profiles("beta_aer")
        except Exception as e:
            self.ax.set_title(f"Plot error: {e}")

        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _plot_profiles(self, key: str):
        df = self._res[key]
        R = df["Range(m)"].to_numpy(float)
        cols = [c for c in df.columns if c != "Range(m)"]
        unit = "m⁻¹ sr⁻¹" if key == "beta_aer" else "m⁻¹"
        label = "β_aer" if key == "beta_aer" else "α_aer"
        for i in self._selected_indices():
            if i >= len(cols):
                continue
            y = df[cols[i]].to_numpy(float)
            m = np.isfinite(R) & np.isfinite(y)
            if m.any():
                self.ax.plot(R[m], y[m], linewidth=1.4, label=self._ts_labels[i])
        self.ax.set_xlabel("Range (m)")
        self.ax.set_ylabel(f"{label} ({unit})")
        self.ax.set_title(self.chart_title_var.get() or f"{label} profile · Fernald inversion")
        if self.ax.get_legend_handles_labels()[0]:
            self.ax.legend(fontsize=8, loc="best")

    def _plot_aod(self):
        aod_df = self._res["AOD"]
        meta = {"R_ref_m", "lidar_ratio_aer_sr", "method"}
        cols = [c for c in aod_df.columns if c not in meta]
        labels = [self._col_label(c) for c in cols]
        vals = aod_df[cols].to_numpy(float).flatten()
        x = np.arange(len(vals))
        self.ax.plot(x, vals, color=theme.ORANGE, marker="o", markersize=4, linewidth=1.4)
        self.ax.set_xticks(x)
        step = max(1, len(labels) // 12)
        self.ax.set_xticklabels(
            [labels[i] if i % step == 0 else "" for i in range(len(labels))],
            rotation=45, ha="right", fontsize=8)
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("AOD (532 nm)")
        self.ax.set_title(self.chart_title_var.get() or "Aerosol Optical Depth · time series")

    def _plot_mass(self):
        if "mass_aer" not in self._res:
            self.ax.set_title("Mass profile not computed — tick 'Compute mass' and rerun")
            return
        df = self._res["mass_aer"]
        R = df["Range(m)"].to_numpy(float)
        cols = [c for c in df.columns if c != "Range(m)"]
        for i in self._selected_indices():
            if i >= len(cols):
                continue
            y = df[cols[i]].to_numpy(float)
            m = np.isfinite(R) & np.isfinite(y)
            if m.any():
                self.ax.plot(R[m], y[m], linewidth=1.4, label=self._ts_labels[i])
        try:
            sig = float(self.sigma_ext.get())
            title_extra = f" · σ_ext = {sig:.1f} m²/g"
        except Exception:
            title_extra = ""
        self.ax.set_xlabel("Range (m)")
        self.ax.set_ylabel("Mass concentration (µg / m³)")
        self.ax.set_title(self.chart_title_var.get()
                          or f"Aerosol mass profile{title_extra}")
        if self.ax.get_legend_handles_labels()[0]:
            self.ax.legend(fontsize=8, loc="best")

    def _plot_aeronet(self):
        if "AERONET_comparison" not in self._res:
            self.ax.set_title("AERONET comparison not loaded — tick the box, pick the file, rerun")
            return
        df = self._res["AERONET_comparison"]
        t = pd.to_datetime(df["Time"], errors="coerce")
        f_aod = pd.to_numeric(df["AOD_Fernald_532"], errors="coerce")
        a_aod = pd.to_numeric(df["AOD_AERONET_532"], errors="coerce")
        self.ax.plot(t, f_aod, color=theme.ORANGE, lw=1.7, marker="o", markersize=4,
                     label="Fernald AOD (532 nm)", zorder=2)
        self.ax.plot(t, a_aod, color=theme.NAVY, lw=1.6, ls="--", marker="s", markersize=4,
                     label="AERONET AOD (532 nm)", zorder=3)
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("AOD (532 nm)")
        try:
            stats = _aeronet_stats(df) if _HAS_AERONET else {"n": 0}
            sub = (f"N={stats['n']}  MAE={stats['MAE']:.3f}  "
                   f"bias={stats['bias']:+.3f}  RMSE={stats['RMSE']:.3f}")
            if np.isfinite(stats.get("R2", np.nan)):
                sub += f"  R²={stats['R2']:.2f}"
        except Exception:
            sub = ""
        self.ax.set_title(self.chart_title_var.get()
                          or f"Fernald vs AERONET AOD  ·  {sub}")
        self.ax.legend(fontsize=9, loc="best")
        self.fig.autofmt_xdate()

    def save_png(self):
        out = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if out:
            self.fig.savefig(out, dpi=150, bbox_inches="tight")
            self._log(f"Saved PNG: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Step 6 Page  –  Depolarization (cross-pol, Track 2)
# ═════════════════════════════════════════════════════════════════════════════
DEPOL_PLOT_MODES = [
    "delta_v", "NRB parallel", "NRB perpendicular", "NRB parallel + perpendicular", "SNR parallel + perpendicular",
]
# Signal-inspection sources for Step 3 (mirror Step 2). "Depolarization" = the
# depol products above; "Processed"/"Raw .dat" show the per-channel signals
# (analog / glue / NRB) for the chosen channel, reusing Step 2's mode lists.
DEPOL_SOURCES = ["Depolarization", "Processed", "Raw .dat"]


class Step6Page(ctk.CTkFrame):
    def __init__(self, master, app_state: AppState):
        super().__init__(master, fg_color="transparent")
        self.app_state = app_state

        # ── State ───────────────────────────────────────────────────────────
        self.co_folder    = tk.StringVar()
        self.cross_folder = tk.StringVar()
        # Channel mode: "Pair (co + cross)" needs both; the single-channel modes
        # process one polarisation alone (separate-day experiment → no δ).
        self.channel_mode = tk.StringVar(value="Pair (co + cross)")
        self.out_path     = tk.StringVar()
        self.pattern      = tk.StringVar(value="*.dat")
        # Measurement date (YYYY-MM-DD). .dat filenames like "21.21co.dat" carry
        # no date → times default to 1900-01-01; set this so the workbook lines up
        # with Step 2 / ALT / Display (leave blank to keep the filename date).
        self.date_str     = tk.StringVar(value="")
        self.pretrigger_bins = tk.IntVar(value=1024)
        # Glue mode — kept in sync with Step 2 (default "auto"). photon_only=True
        # skips the analog+photon glue, matching Step 2's "Skip glue" checkbox.
        self.photon_only  = tk.BooleanVar(value=False)
        # Per-profile glue choice from the pretrigger BG (day → photon only,
        # night → glue). Overrides Skip glue when ticked. Default ON (safe in every
        # case; the prototype folders mix day+night).
        self.glue_auto_daynight = tk.BooleanVar(value=True)
        # ── Full raw-processing parameters (mirror Step 2) so Step 3 can be run
        #    STANDALONE for a single-channel experiment without touching Step 2.
        #    All of these are forwarded to build_single_profile via nrb_kwargs.
        self.bin_spacing_m        = tk.DoubleVar(value=3.75)
        self.dead_time_ns         = tk.DoubleVar(value=3.06)
        self.bg_mode              = tk.StringVar(value="pretrigger")
        self.first_signal_bin     = tk.IntVar(value=1025)
        self.first_signal_range_m = tk.DoubleVar(value=3.75)
        self.pretrigger_trim_bins = tk.IntVar(value=24)
        self.bg_start_m           = tk.DoubleVar(value=0.0)
        self.bg_end_m             = tk.DoubleVar(value=3750.0)
        # Toggle window: min 2.0 = ~2x the photon floor so the auto-blend descending
        # crossing exists; max 10.0 keeps the dead-time correction at ~3% (tau=3.06 ns).
        self.min_toggle_rate      = tk.DoubleVar(value=2.0)
        self.max_toggle_rate      = tk.DoubleVar(value=10.0)
        self.auto_toggle_selector = tk.BooleanVar(value=True)
        self.day_min_toggle_rate  = tk.DoubleVar(value=75.0)
        self.day_max_toggle_rate  = tk.DoubleVar(value=130.0)
        self.toggle_bg_switch_threshold_mhz = tk.DoubleVar(value=10.0)
        self.blend_r1_m           = tk.DoubleVar(value=1200.0)
        self.blend_r2_m           = tk.DoubleVar(value=1800.0)
        self.auto_blend           = tk.BooleanVar(value=True)
        self.sig_start_m          = tk.DoubleVar(value=0.0)
        self.sig_end_m            = tk.DoubleVar(value=15000.0)
        self.preset_name          = tk.StringVar(value="Custom")
        self.delta_mol_mode = tk.StringVar(value="Narrowband (0.0044)")
        self.delta_mol    = tk.DoubleVar(value=0.0044)
        self.cal_rmin_m   = tk.DoubleVar(value=4500.0)
        self.cal_rmax_m   = tk.DoubleVar(value=5500.0)
        self.snr_min      = tk.DoubleVar(value=3.0)   # mask δ where cross SNR below this
        self.snr_gate     = tk.BooleanVar(value=True)   # gate NRB co/cross by SNR (default ON)
        # Corrections (so per-channel NRB is fully comparable to MPL / Step 2)
        self.energy_mj    = tk.DoubleVar(value=25.0)
        # Overlap ON by default (analytical NARIT geometry), matching the NRB page.
        # O_min=0.99 → only the near field with essentially complete overlap
        # (R≳150 m) is kept; below that O(R)<0.99 → NaN.
        self.overlap_mode = tk.StringVar(value="Analytical (NARIT)")
        self.overlap_o_min = tk.DoubleVar(value=0.99)
        self.overlap_file = tk.StringVar(value="")
        self.afterpulse_co_file = tk.StringVar(value="")
        self.afterpulse_cross_file = tk.StringVar(value="")
        # Afterpulse .dat recorded WITHOUT a pretrigger block → read with
        # pretrigger_bins=0 and take the BG from the far range instead.
        self.ap_no_pretrigger = tk.BooleanVar(value=False)
        self.progress     = tk.DoubleVar(value=0.0)
        self.plot_mode    = tk.StringVar(value="delta_v")
        self.data_source  = tk.StringVar(value="Depolarization")
        self.signal_channel = tk.StringVar(value="Parallel")
        self.chart_title_var = tk.StringVar()
        self.x_title_var  = tk.StringVar()
        self.y_title_var  = tk.StringVar()
        self.xmin_var = tk.StringVar(); self.xmax_var = tk.StringVar()
        self.ymin_var = tk.StringVar(); self.ymax_var = tk.StringVar()

        # Results held in memory
        self._dv_df: Optional[pd.DataFrame] = None
        self._qc_df: Optional[pd.DataFrame] = None
        self._nrb_co_df: Optional[pd.DataFrame] = None
        self._nrb_cross_df: Optional[pd.DataFrame] = None
        self._snr_co_df: Optional[pd.DataFrame] = None
        self._snr_cross_df: Optional[pd.DataFrame] = None
        self._profiles_diag: List[dict] = []   # per-profile signal diagnostics
        self._ts_labels: List[str] = []

        if not _HAS_DEPOL:
            self._build_missing_view()
            return

        self._build_ui()
        self._update_delta_mol_ui()

    # ── Backend missing fallback ───────────────────────────────────────────
    def _build_missing_view(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        card = Card(self, title="Backend missing", icon="⚠️")
        card.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(
            card.body,
            text="depol_engine.py was not found alongside this script.\n"
                 "Step 6 needs that module for the depolarization (parallel / perpendicular) analysis.",
            font=theme.F_BODY, text_color=theme.TEXT_SECONDARY, justify="left",
        ).pack(pady=20)

    # ── Layout ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = PageHeader(
            self,
            title="Step 2 · Depolarization (parallel / perpendicular)",
            subtitle="Volume depolarization ratio δ + aerosol typing from paired parallel/perpendicular .dat files",
            badge=("v1.0", "navy"),
        )
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))

        self.paned = tk.PanedWindow(
            self, orient="horizontal", bg=theme.APP_BG,
            sashrelief="flat", sashwidth=8, sashpad=0, bd=0, showhandle=False,
        )
        self.paned.grid(row=1, column=0, sticky="nsew")

        left_pane = tk.Frame(self.paned, bg=theme.APP_BG, bd=0, highlightthickness=0)
        left_pane.grid_columnconfigure(0, weight=1)
        left_pane.grid_rowconfigure(0, weight=1)
        self.paned.add(left_pane, minsize=380, width=560, stretch="always")

        left = ctk.CTkScrollableFrame(left_pane, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_columnconfigure(0, weight=1)
        self._build_io_card(left).grid(row=0, column=0, sticky="ew", pady=(0, 12), padx=(0, 10))
        self._build_params_card(left).grid(row=1, column=0, sticky="ew", pady=(0, 12), padx=(0, 10))
        self._build_run_card(left).grid(row=2, column=0, sticky="ew", pady=(0, 12), padx=(0, 10))
        self._build_console_card(left).grid(row=3, column=0, sticky="nsew", padx=(0, 10))

        right_pane = tk.Frame(self.paned, bg=theme.APP_BG, bd=0, highlightthickness=0)
        right_pane.grid_columnconfigure(0, weight=1)
        right_pane.grid_rowconfigure(0, weight=1)
        self.paned.add(right_pane, minsize=380, stretch="always")

        right = ctk.CTkFrame(right_pane, fg_color="transparent")
        right.grid(row=0, column=0, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)
        self._build_plot_card(right).grid(row=0, column=0, sticky="nsew")

    # ── Card: I/O ──────────────────────────────────────────────────────────
    def _build_io_card(self, parent) -> "Card":
        card = Card(parent, title="Input / Output", icon="📁")
        body = card.body
        body.grid_columnconfigure(0, weight=1)

        def _input_row(r, label, var, sync_out=False):
            lbl = ctk.CTkLabel(body, text=label, font=theme.F_SMALL,
                               text_color=theme.TEXT_SECONDARY, anchor="w")
            lbl.grid(row=r, column=0, sticky="w")
            row = ctk.CTkFrame(body, fg_color="transparent")
            row.grid(row=r+1, column=0, sticky="ew", pady=(2, 8))
            row.grid_columnconfigure(0, weight=1)
            ent = ctk.CTkEntry(row, textvariable=var, **theme.input_style())
            ent.grid(row=0, column=0, sticky="ew", padx=(0, 8))
            bf = ctk.CTkButton(row, text="File…", width=64,
                               command=lambda: self._pick_file(var, sync_out=sync_out),
                               **theme.secondary_button_style(font=theme.F_SMALL))
            bf.grid(row=0, column=1, padx=(0, 4))
            bd = ctk.CTkButton(row, text="Folder…", width=72,
                               command=lambda: self._pick_folder(var, sync_out=sync_out),
                               **theme.secondary_button_style(font=theme.F_SMALL))
            bd.grid(row=0, column=2)
            return lbl, [ent, bf, bd]

        self._par_label_default = "Parallel  —  file or folder (co_pol_*.dat)"
        self._perp_label_default = "Perpendicular  —  file or folder (cross_pol_*.dat)"
        self._par_label, _ = _input_row(0, self._par_label_default, self.co_folder, sync_out=True)
        self._perp_label, self._perp_widgets = _input_row(2, self._perp_label_default, self.cross_folder)

        ctk.CTkLabel(body, text="Output workbook", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY, anchor="w").grid(row=4, column=0, sticky="w")
        row3 = ctk.CTkFrame(body, fg_color="transparent")
        row3.grid(row=5, column=0, sticky="ew", pady=(2, 0))
        row3.grid_columnconfigure(0, weight=1)
        ctk.CTkEntry(row3, textvariable=self.out_path, **theme.input_style()).grid(
            row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(row3, text="Save As…", command=self._pick_output, width=100,
                      **theme.secondary_button_style()).grid(row=0, column=1)

        # Measurement date — stamps dateless (HH.MM) filenames so the workbook
        # lines up with Step 2 / ALT / Display (blank = keep the filename date).
        ctk.CTkLabel(body, text="Measurement date (YYYY-MM-DD) — blank = from filename",
                     font=theme.F_SMALL, text_color=theme.TEXT_SECONDARY, anchor="w").grid(
            row=6, column=0, sticky="w", pady=(8, 0))
        drow = ctk.CTkFrame(body, fg_color="transparent")
        drow.grid(row=7, column=0, sticky="ew", pady=(2, 0))
        ctk.CTkEntry(drow, textvariable=self.date_str, width=160,
                     **theme.input_style()).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            body,
            text="Parallel and perpendicular may live in the same folder (matched by prefix) or two folders. "
                 "Files pair by their shared suffix, e.g. co_pol_23-06-2026-21.37.dat ↔ "
                 "cross_pol_23-06-2026-21.37.dat.",
            font=(theme.FONT_FAMILY, 10), text_color=theme.TEXT_MUTED, anchor="w",
            wraplength=480, justify="left",
        ).grid(row=8, column=0, sticky="w", pady=(8, 0))

        # Channel mode — pair (needs both) vs a single-channel run (one folder).
        mrow = ctk.CTkFrame(body, fg_color="transparent")
        mrow.grid(row=9, column=0, sticky="ew", pady=(8, 0))
        ctk.CTkLabel(mrow, text="Channel mode:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 8))
        ctk.CTkOptionMenu(
            mrow, variable=self.channel_mode,
            values=["Pair (co + cross)", "2-PMT file (par+perp)",
                    "Parallel only", "Perpendicular only"],
            command=self._on_channel_mode_change,
            fg_color=theme.CARD_BG, button_color=theme.CARD_BG, button_hover_color=theme.CREAM_SOFT,
            text_color=theme.TEXT_PRIMARY, dropdown_fg_color=theme.CARD_BG,
            dropdown_text_color=theme.TEXT_PRIMARY, dropdown_hover_color=theme.CREAM,
            corner_radius=theme.RADIUS_INPUT, height=30, font=theme.F_SMALL, width=200,
        ).grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(
            body,
            text="2-PMT file = one folder where each .dat holds BOTH channels "
                 "(parallel cols A-D, perpendicular E-H) → uses the Parallel input only. "
                 "Single-channel = process parallel OR perpendicular alone (one folder, "
                 "no δ) — for separate-day experiments (uses the matching input).",
            font=(theme.FONT_FAMILY, 10), text_color=theme.TEXT_MUTED, anchor="w",
            wraplength=480, justify="left",
        ).grid(row=10, column=0, sticky="w", pady=(4, 0))
        self._on_channel_mode_change()   # set initial enable/label state
        return card

    def _on_channel_mode_change(self, *_):
        """Grey out the (unused) Perpendicular input in 2-PMT mode and retitle the
        Parallel input, so it's obvious only one input is needed."""
        dual = self.channel_mode.get().startswith("2-PMT")
        for w in getattr(self, "_perp_widgets", []):
            try:
                w.configure(state="disabled" if dual else "normal")
            except Exception:
                pass
        if getattr(self, "_par_label", None) is not None:
            self._par_label.configure(
                text="2-PMT file  —  file or folder (par cols A-D + perp E-H)"
                if dual else self._par_label_default)
        if getattr(self, "_perp_label", None) is not None:
            self._perp_label.configure(
                text="Perpendicular  —  (not used in 2-PMT mode)"
                if dual else self._perp_label_default,
                text_color=theme.TEXT_MUTED if dual else theme.TEXT_SECONDARY)

    # ── Card: Parameters ───────────────────────────────────────────────────
    def _build_params_card(self, parent) -> "Card":
        """Parameters laid out in PIPELINE ORDER — the same order the engine
        applies them:
            read .dat -> dead-time -> BG -> glue -> afterpulse -> overlap
            -> xR^2 -> /energy -> SNR gate -> depolarization calibration
        """
        card = Card(parent, title="Parameters (pipeline order)", icon="⚙️")
        body = card.body
        body.grid_columnconfigure(0, weight=1)
        r = 0

        def sub(text):
            nonlocal r
            self._subsection(body, text).grid(row=r, column=0, sticky="ew", pady=(10, 4))
            r += 1

        def frame(pady=(0, 4), cols=1):
            nonlocal r
            f = ctk.CTkFrame(body, fg_color="transparent")
            f.grid(row=r, column=0, sticky="ew", pady=pady)
            for i in range(cols):
                f.grid_columnconfigure(i, weight=1)
            r += 1
            return f

        def note(text, pady=(0, 6)):
            nonlocal r
            ctk.CTkLabel(body, text=text, font=theme.F_TINY, text_color=theme.TEXT_MUTED,
                         anchor="w", wraplength=480, justify="left").grid(
                row=r, column=0, sticky="w", pady=pady)
            r += 1

        def cb(parent_, text, var, **grid):
            return ctk.CTkCheckBox(
                parent_, text=text, variable=var,
                font=theme.F_SMALL, text_color=theme.TEXT_PRIMARY,
                fg_color=theme.ORANGE, hover_color=theme.ORANGE_HOVER,
                border_color=theme.BORDER, checkmark_color="#fff",
                corner_radius=4, border_width=2).grid(**grid)

        def menu(parent_, var, values, command=None, width=None, **grid):
            kw = dict(variable=var, values=values,
                      fg_color=theme.CARD_BG, button_color=theme.CARD_BG,
                      button_hover_color=theme.CREAM_SOFT, text_color=theme.TEXT_PRIMARY,
                      dropdown_fg_color=theme.CARD_BG, dropdown_text_color=theme.TEXT_PRIMARY,
                      dropdown_hover_color=theme.CREAM, corner_radius=theme.RADIUS_INPUT,
                      height=32, font=theme.F_BODY)
            if command:
                kw["command"] = command
            if width:
                kw["width"] = width
            m = ctk.CTkOptionMenu(parent_, **kw)
            m.grid(**grid)
            return m

        # ── 1. Read the .dat: raw layout & dead time ───────────────────────
        sub("1 · Raw layout & dead time")
        pr = frame(pady=(0, 2), cols=3)
        for i, (short, full) in enumerate([("Pre-trigger", "Licel pretrigger 1024"),
                                           ("No-pretrigger", "Legacy no pretrigger"),
                                           ("Auto-detect", "Auto detect")]):
            cmd = (self.auto_detect_bins if full == "Auto detect"
                   else (lambda n=full: self.apply_preset(n)))
            ctk.CTkButton(pr, text=short, command=cmd,
                          fg_color=theme.CREAM, hover_color=theme.ORANGE_SOFT,
                          text_color=theme.TEXT_PRIMARY, corner_radius=theme.RADIUS_BUTTON,
                          height=32, font=theme.F_SMALL,
                          ).grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 4, 0))
        ps = frame()
        ps.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(ps, text="Active preset:", font=(theme.FONT_FAMILY, 10),
                     text_color=theme.TEXT_MUTED, anchor="e").grid(row=0, column=0, sticky="e", padx=(0, 6))
        ctk.CTkLabel(ps, textvariable=self.preset_name, font=theme.F_SMALL,
                     text_color=theme.ORANGE, anchor="e").grid(row=0, column=1, sticky="e")

        f = frame(cols=3)
        FieldRow(f, "bin_spacing_m", self.bin_spacing_m).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        FieldRow(f, "dead_time_ns", self.dead_time_ns).grid(row=0, column=1, sticky="ew", padx=3)
        FieldRow(f, "Pattern", self.pattern).grid(row=0, column=2, sticky="ew", padx=(6, 0))
        f = frame(cols=3)
        FieldRow(f, "pretrigger_bins", self.pretrigger_bins).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        FieldRow(f, "first_signal_bin", self.first_signal_bin).grid(row=0, column=1, sticky="ew", padx=3)
        FieldRow(f, "first_signal_range_m", self.first_signal_range_m).grid(row=0, column=2, sticky="ew", padx=(6, 0))
        note("All three describe the SAME raw layout: the file starts with "
             "pretrigger_bins baseline bins, real signal begins at first_signal_bin "
             "(= pretrigger_bins + 1) at range first_signal_range_m. Use the presets above.")

        # ── 2. Background subtraction ──────────────────────────────────────
        sub("2 · Background (BG) subtraction")
        f = frame(cols=3)
        bgm = ctk.CTkFrame(f, fg_color="transparent")
        bgm.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        bgm.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(bgm, text="BG mode", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY, anchor="w").grid(row=0, column=0, sticky="w")
        menu(bgm, self.bg_mode, ["pretrigger", "fixed", "far_range"],
             row=1, column=0, sticky="ew", pady=(2, 0))
        FieldRow(f, "pretrigger trim bins", self.pretrigger_trim_bins).grid(row=0, column=1, sticky="ew", padx=3)
        f2 = frame(cols=2)
        FieldRow(f2, "bg_start_m (fixed mode)", self.bg_start_m).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        FieldRow(f2, "bg_end_m (fixed mode)", self.bg_end_m).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        # ── 3. Glue: analog <-> photon ─────────────────────────────────────
        sub("3 · Glue (analog ↔ photon)")
        f = frame(cols=4)
        FieldRow(f, "min_toggle_rate", self.min_toggle_rate).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        FieldRow(f, "max_toggle_rate", self.max_toggle_rate).grid(row=0, column=1, sticky="ew", padx=3)
        FieldRow(f, "day_min_toggle", self.day_min_toggle_rate).grid(row=0, column=2, sticky="ew", padx=3)
        FieldRow(f, "day_max_toggle", self.day_max_toggle_rate).grid(row=0, column=3, sticky="ew", padx=(6, 0))
        f = frame(cols=2)
        FieldRow(f, "BG switch threshold (MHz)", self.toggle_bg_switch_threshold_mhz).grid(
            row=0, column=0, sticky="ew", padx=(0, 6))
        cb(f, "Auto day/night toggle rates", self.auto_toggle_selector,
           row=0, column=1, sticky="sw", padx=(6, 0), pady=(18, 0))
        f = frame(cols=3)
        FieldRow(f, "blend_r1_m", self.blend_r1_m).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        FieldRow(f, "blend_r2_m", self.blend_r2_m).grid(row=0, column=1, sticky="ew", padx=3)
        cb(f, "Auto blend r1/r2", self.auto_blend,
           row=0, column=2, sticky="sw", padx=(6, 0), pady=(18, 0))
        f = frame(cols=1)
        cb(f, "Skip glue (photon only)", self.photon_only,
           row=0, column=0, sticky="w")
        cb(f, "Auto day/night glue: day → photon only, night → glue (overrides Skip glue)",
           self.glue_auto_daynight, row=1, column=0, sticky="w", pady=(6, 0))
        note("max_toggle = ε/τ_d (10 MHz → 3% dead-time correction at τ=3.06 ns; the "
             "Licel model fails near 326 MHz). min_toggle must sit ABOVE the photon "
             "floor or the auto-blend crossing is never found (0.5 is below it — use ~2).")

        # ── 4. Corrections: afterpulse -> overlap -> energy ────────────────
        sub("4 · Corrections (afterpulse → overlap → R² → energy)")
        f = frame()
        f.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(f, text="Afterpulse parallel:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 8))
        ctk.CTkEntry(f, textvariable=self.afterpulse_co_file, **theme.input_style()).grid(
            row=0, column=1, sticky="ew", padx=(0, 8))
        ctk.CTkButton(f, text="…", width=36,
                      command=lambda: self._pick_into(self.afterpulse_co_file, "Afterpulse parallel",
                                                      [("Cal", "*.dat *.csv *.xlsx *.xls")]),
                      **theme.secondary_button_style(height=34, font=theme.F_SMALL)).grid(row=0, column=2)
        f = frame()
        f.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(f, text="Afterpulse perpendicular:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 8))
        ctk.CTkEntry(f, textvariable=self.afterpulse_cross_file, **theme.input_style()).grid(
            row=0, column=1, sticky="ew", padx=(0, 8))
        ctk.CTkButton(f, text="…", width=36,
                      command=lambda: self._pick_into(self.afterpulse_cross_file, "Afterpulse perpendicular",
                                                      [("Cal", "*.dat *.csv *.xlsx *.xls")]),
                      **theme.secondary_button_style(height=34, font=theme.F_SMALL)).grid(row=0, column=2)
        cb(f, "Calibration .dat has NO pretrigger (BG from far range)", self.ap_no_pretrigger,
           row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))

        f = frame(cols=2)
        ov = ctk.CTkFrame(f, fg_color="transparent")
        ov.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ov.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(ov, text="Overlap O(R)", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY, anchor="w").grid(row=0, column=0, sticky="w")
        menu(ov, self.overlap_mode, ["Disabled", "Analytical (NARIT)", "Load file"],
             command=lambda _v: self._update_corr_ui(), row=1, column=0, sticky="ew", pady=(2, 0))
        FieldRow(f, "O_min", self.overlap_o_min).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self._ov_file_row = ctk.CTkFrame(body, fg_color="transparent")
        self._ov_file_row.grid(row=r, column=0, sticky="ew", pady=(0, 4))
        r += 1
        self._ov_file_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self._ov_file_row, text="O(R) file:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 8))
        self._ov_entry = ctk.CTkEntry(self._ov_file_row, textvariable=self.overlap_file,
                                      **theme.input_style())
        self._ov_entry.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        ctk.CTkButton(self._ov_file_row, text="…", width=36,
                      command=lambda: self._pick_into(self.overlap_file, "Overlap O(R)",
                                                      [("Tabular", "*.csv *.xlsx *.xls")]),
                      **theme.secondary_button_style(height=34, font=theme.F_SMALL)).grid(row=0, column=2)

        f = frame(cols=3)
        FieldRow(f, "Energy (mJ)", self.energy_mj).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        FieldRow(f, "sig_start_m", self.sig_start_m).grid(row=0, column=1, sticky="ew", padx=3)
        FieldRow(f, "sig_end_m", self.sig_end_m).grid(row=0, column=2, sticky="ew", padx=(6, 0))
        note("Overlap & energy cancel in δ (same telescope/shot) but are applied so the "
             "NRB parallel/perpendicular plots are fully corrected — comparable to MPL. "
             "Afterpulse is per-PMT (does NOT cancel) → give a separate cal file per channel.")

        # ── 5. Signal quality (SNR gate) ───────────────────────────────────
        sub("5 · Signal quality (SNR)")
        f = frame()
        f.grid_columnconfigure(2, weight=1)
        ctk.CTkLabel(f, text="Min perpendicular SNR (mask δ):", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 8))
        ctk.CTkEntry(f, textvariable=self.snr_min, width=70,
                     **theme.input_style()).grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(f, text="(δ set NaN where perpendicular SNR < this)",
                     font=theme.F_TINY, text_color=theme.TEXT_MUTED).grid(
            row=0, column=2, sticky="w", padx=(8, 0))
        cb(f, "Normalise NRB co/cross in the lower troposphere (BL → 1.0, full curve, no cut)",
           self.snr_gate, row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))

        # ── 6. Depolarization calibration (last: needs the finished NRB) ───
        sub("6 · Depolarization calibration")
        f = frame(pady=(2, 6))
        f.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(f, text="δ_mol (532 nm):", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 8))
        menu(f, self.delta_mol_mode,
             ["Narrowband (0.0044)", "Broadband (0.0144)", "Custom"],
             command=lambda _v: self._update_delta_mol_ui(), width=200,
             row=0, column=1, sticky="w")
        f = frame()
        ctk.CTkLabel(f, text="δ_mol value:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).grid(row=0, column=0, padx=(0, 8))
        self._dm_entry = ctk.CTkEntry(f, textvariable=self.delta_mol, width=90,
                                      **theme.input_style())
        self._dm_entry.grid(row=0, column=1, sticky="w")
        f = frame(cols=2)
        FieldRow(f, "cal R_min (m)", self.cal_rmin_m).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        FieldRow(f, "cal R_max (m)", self.cal_rmax_m).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        note("Narrowband (filter rejects rotational-Raman wings) = 0.0044; broadband = "
             "0.0144 (Behrendt & Nakamura 2002). The cal window is assumed aerosol-free "
             "(purely molecular): C = ⟨δ*⟩_clean / δ_mol. Typical 4000–6000 m at night.",
             pady=(0, 8))

        self._update_corr_ui()
        return card

    def _pick_into(self, var, title, filetypes):
        p = filedialog.askopenfilename(title=title, filetypes=filetypes + [("All", "*.*")])
        if p:
            var.set(p)

    def apply_preset(self, name: str):
        """Raw-layout presets — mirrors Step 2 so a no-pretrigger acquisition can
        be tried without hand-editing every field."""
        if name == "Licel pretrigger 1024":
            self.bg_mode.set("pretrigger"); self.pretrigger_bins.set(1024)
            self.first_signal_bin.set(1025); self.first_signal_range_m.set(3.75)
            self.pretrigger_trim_bins.set(24)
            self.bg_start_m.set(0.0); self.bg_end_m.set(3750.0)
        elif name == "Legacy no pretrigger":
            self.bg_mode.set("fixed"); self.pretrigger_bins.set(0)
            self.first_signal_bin.set(1); self.first_signal_range_m.set(3.75)
            self.pretrigger_trim_bins.set(0)
            self.bg_start_m.set(13000.0); self.bg_end_m.set(14500.0)
        else:
            return
        self.blend_r1_m.set(1200.0); self.blend_r2_m.set(1800.0)
        self.min_toggle_rate.set(2.0); self.max_toggle_rate.set(10.0)
        self.day_min_toggle_rate.set(75.0); self.day_max_toggle_rate.set(130.0)
        self.toggle_bg_switch_threshold_mhz.set(10.0)
        self.sig_start_m.set(0.0); self.sig_end_m.set(15000.0)
        self.bin_spacing_m.set(3.75); self.dead_time_ns.set(3.06)
        self.preset_name.set(name)
        self._log(f"Preset applied: {name}")

    def auto_detect_bins(self):
        """Pick the preset from the first input file's bin count
        (>=5024 → 1024 pretrigger + 4000 signal; otherwise no pretrigger)."""
        src = (self.co_folder.get().strip() or self.cross_folder.get().strip())
        if not src or not os.path.exists(src):
            messagebox.showerror("Error", "Choose a parallel/perpendicular input first.")
            return
        p = Path(src)
        files = [p] if p.is_file() else _depol_glob(p, self.pattern.get().strip() or "*.dat")
        if not files:
            messagebox.showerror("Error", "No .dat or raw files found in the input.")
            return
        try:
            bins = int(_read_tr40_dat_ascii_array(files[0]).shape[0])
            self._log(f"Auto detect: {files[0].name} -> {bins} bins")
            self.apply_preset("Licel pretrigger 1024" if bins >= 5024
                              else "Legacy no pretrigger")
        except Exception as e:
            messagebox.showerror("Auto detect failed", str(e))

    @staticmethod
    def _range_axis_for(path: Path, pretrigger_bins: int, dr_m: float = 3.75,
                        first_signal_range_m: float = 3.75,
                        first_signal_bin: Optional[int] = None) -> np.ndarray:
        """Range axis a .dat file yields after the first-signal cut — to build
        O(R)/A(R) on the same grid the engine will use. dr_m / first_signal_*
        must match the values handed to the engine or the arrays land on a
        different grid."""
        arr = _read_tr40_dat_ascii_array(Path(path))
        total = int(arr.shape[0])
        fsb = int(first_signal_bin) if first_signal_bin else int(pretrigger_bins) + 1
        n = total - (fsb - 1)
        return float(first_signal_range_m) + np.arange(n, dtype=float) * float(dr_m)

    def _update_corr_ui(self):
        if not hasattr(self, "_ov_entry"):
            return
        st = "normal" if self.overlap_mode.get().startswith("Load") else "disabled"
        try:
            self._ov_entry.configure(state=st)
        except Exception:
            pass

    @staticmethod
    def _subsection(parent, text: str) -> ctk.CTkLabel:
        return ctk.CTkLabel(
            parent, text=text.upper(), font=(theme.FONT_FAMILY, 10, "bold"),
            text_color=theme.TEXT_MUTED, anchor="w",
        )

    # ── Card: Run ──────────────────────────────────────────────────────────
    def _build_run_card(self, parent) -> "Card":
        card = Card(parent)
        body = card.body
        body.grid_columnconfigure(1, weight=1)
        self.run_btn = ctk.CTkButton(
            body, text="▶  Run Depolarization", command=self.run,
            **theme.primary_button_style(width=240),
        )
        self.run_btn.grid(row=0, column=0, sticky="w")
        self.pb = ctk.CTkProgressBar(
            body, variable=self.progress, progress_color=theme.ORANGE,
            fg_color=theme.LIGHT_GRAY, height=8, corner_radius=4,
        )
        self.pb.set(0)
        self.pb.grid(row=0, column=1, sticky="ew", padx=(16, 0))
        self.status = StatusBar(body)
        self.status.grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))
        return card

    def _build_console_card(self, parent) -> "ConsoleLog":
        self.console = ConsoleLog(parent, title="Console Log")
        return self.console

    # ── Card: Plot ─────────────────────────────────────────────────────────
    def _build_plot_card(self, parent) -> "Card":
        card = Card(parent, title="Depolarization Plot", icon="🧭")
        body = card.body
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(4, weight=1)

        # Controls row 1: mode + actions
        ctrl = ctk.CTkFrame(body, fg_color="transparent")
        ctrl.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        _menu_kw = dict(
            fg_color=theme.CARD_BG, button_color=theme.CARD_BG, button_hover_color=theme.CREAM_SOFT,
            text_color=theme.TEXT_PRIMARY, dropdown_fg_color=theme.CARD_BG,
            dropdown_text_color=theme.TEXT_PRIMARY, dropdown_hover_color=theme.CREAM,
            corner_radius=theme.RADIUS_INPUT, height=32, font=theme.F_BODY,
        )
        ctk.CTkLabel(ctrl, text="Source:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).pack(side="left")
        ctk.CTkOptionMenu(ctrl, variable=self.data_source, values=DEPOL_SOURCES,
                          command=lambda _v: self._update_plot_mode_options(),
                          width=140, **_menu_kw).pack(side="left", padx=(6, 10))
        ctk.CTkLabel(ctrl, text="Ch:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).pack(side="left")
        ctk.CTkOptionMenu(ctrl, variable=self.signal_channel,
                          values=["Parallel", "Perpendicular"],
                          command=lambda _v: self.refresh_plot(),
                          width=130, **_menu_kw).pack(side="left", padx=(6, 10))
        ctk.CTkLabel(ctrl, text="Plot:", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY).pack(side="left")
        self._plot_mode_menu = ctk.CTkOptionMenu(
            ctrl, variable=self.plot_mode, values=DEPOL_PLOT_MODES,
            command=lambda _v: self.refresh_plot(), width=200, **_menu_kw)
        self._plot_mode_menu.pack(side="left", padx=(6, 12))
        ctk.CTkButton(ctrl, text="Refresh", command=self.refresh_plot,
                      **theme.primary_button_style(width=90, height=32)).pack(side="left", padx=(0, 6))
        ctk.CTkButton(ctrl, text="Auto Scale", command=self.auto_scale,
                      **theme.secondary_button_style(width=90, height=32)).pack(side="left", padx=(0, 6))
        ctk.CTkButton(ctrl, text="PNG", command=self.save_png,
                      **theme.ghost_button_style(width=60, height=32)).pack(side="left", padx=(8, 0))
        ctk.CTkButton(ctrl, text="CSV", command=self.save_csv,
                      **theme.ghost_button_style(width=60, height=32)).pack(side="left")
        ctk.CTkButton(ctrl, text="Save set", command=self.save_signal_set,
                      **theme.secondary_button_style(width=90, height=32)).pack(side="left", padx=(8, 0))

        # Controls row 2: axis limits
        ctrl2 = ctk.CTkFrame(body, fg_color="transparent")
        ctrl2.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        for lbl, var in [("x min", self.xmin_var), ("x max", self.xmax_var),
                         ("y min", self.ymin_var), ("y max", self.ymax_var)]:
            ctk.CTkLabel(ctrl2, text=lbl, font=theme.F_SMALL,
                         text_color=theme.TEXT_SECONDARY).pack(side="left")
            ctk.CTkEntry(ctrl2, textvariable=var, width=80,
                         **theme.input_style()).pack(side="left", padx=(4, 8))
        ctk.CTkButton(ctrl2, text="Apply Axis", command=self.apply_axes,
                      **theme.secondary_button_style(width=100, height=32)).pack(side="left")

        # Controls row 3: titles
        ctrl3 = ctk.CTkFrame(body, fg_color="transparent")
        ctrl3.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        for lbl, var, w in [("Chart title", self.chart_title_var, 200),
                            ("X title", self.x_title_var, 140),
                            ("Y title", self.y_title_var, 140)]:
            ctk.CTkLabel(ctrl3, text=lbl, font=theme.F_SMALL,
                         text_color=theme.TEXT_SECONDARY).pack(side="left")
            ctk.CTkEntry(ctrl3, textvariable=var, width=w,
                         **theme.input_style()).pack(side="left", padx=(4, 10))

        sel_row = ctk.CTkFrame(body, fg_color="transparent")
        sel_row.grid(row=3, column=0, sticky="ew", pady=(6, 6))
        sel_row.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(sel_row, text="Profiles (multi-select, for profile mode)", font=theme.F_SMALL,
                     text_color=theme.TEXT_SECONDARY, anchor="w").grid(row=0, column=0, sticky="w")
        list_wrap = ctk.CTkFrame(sel_row, fg_color=theme.CREAM_SOFT,
                                 corner_radius=theme.RADIUS_INPUT, border_width=1,
                                 border_color=theme.BORDER, height=110)
        list_wrap.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        list_wrap.grid_columnconfigure(0, weight=1)
        list_wrap.grid_rowconfigure(0, weight=1)
        list_wrap.grid_propagate(False)
        self.time_list = tk.Listbox(
            list_wrap, selectmode="extended", exportselection=False, height=5,
            bg=theme.CREAM_SOFT, fg=theme.TEXT_PRIMARY,
            selectbackground=theme.ORANGE_SOFT, selectforeground=theme.ORANGE,
            relief="flat", borderwidth=0, highlightthickness=0, font=(theme.FONT_FAMILY, 11),
        )
        self.time_list.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        btns = ctk.CTkFrame(list_wrap, fg_color="transparent")
        btns.grid(row=0, column=1, sticky="ns", padx=(0, 8), pady=8)
        ctk.CTkButton(btns, text="Select All",
                      command=lambda: self.time_list.selection_set(0, "end"),
                      **theme.ghost_button_style(width=90, height=28)).pack(pady=(0, 4))
        ctk.CTkButton(btns, text="Clear",
                      command=lambda: self.time_list.selection_clear(0, "end"),
                      **theme.ghost_button_style(width=90, height=28)).pack()

        canvas_wrap = ctk.CTkFrame(body, fg_color=theme.CARD_BG, corner_radius=theme.RADIUS_INPUT,
                                   border_width=1, border_color=theme.BORDER)
        canvas_wrap.grid(row=4, column=0, sticky="nsew", pady=(6, 0))
        canvas_wrap.grid_columnconfigure(0, weight=1)
        canvas_wrap.grid_rowconfigure(0, weight=1)
        self.fig = plt.Figure(figsize=(7, 5.0), dpi=100, facecolor=theme.CARD_BG)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("No data — run the depolarization first")
        self.ax.grid(True, alpha=0.3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_wrap)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        return card

    # ── Helpers ────────────────────────────────────────────────────────────
    def _log(self, msg): self.console.log(msg)
    def _safe(self, fn, *a, **kw): self.after(0, lambda: fn(*a, **kw))
    def _safe_log(self, msg): self._safe(self._log, msg)

    def _update_delta_mol_ui(self):
        mode = self.delta_mol_mode.get()
        if mode.startswith("Narrowband"):
            self.delta_mol.set(0.0044)
            self._dm_entry.configure(state="disabled")
        elif mode.startswith("Broadband"):
            self.delta_mol.set(0.0144)
            self._dm_entry.configure(state="disabled")
        else:
            self._dm_entry.configure(state="normal")

    def _autofill_after_pick(self, picked: str, sync_out: bool):
        """Mirror co-pol selection to cross + suggest an output path."""
        base = Path(picked)
        out_dir = base if base.is_dir() else base.parent
        if sync_out and not self.cross_folder.get().strip():
            self.cross_folder.set(picked)
        if sync_out and not self.out_path.get().strip():
            self.out_path.set(str(out_dir / "Depol-output.xlsx"))

    def _pick_folder(self, var, sync_out=False):
        p = filedialog.askdirectory(title="Select folder")
        if p:
            var.set(p)
            self._autofill_after_pick(p, sync_out)

    def _pick_file(self, var, sync_out=False):
        p = filedialog.askopenfilename(
            title="Select .dat file",
            filetypes=[("TR40 data", "*.dat"), ("All files", "*.*")])
        if p:
            var.set(p)
            self._autofill_after_pick(p, sync_out)

    def _pick_output(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if p:
            self.out_path.set(p)

    # ── Run ────────────────────────────────────────────────────────────────
    def run(self):
        if not _HAS_DEPOL:
            messagebox.showerror("Missing", "depol_engine.py not found.")
            return
        co = self.co_folder.get().strip()
        cross = self.cross_folder.get().strip() or co
        out = self.out_path.get().strip()
        mode = self.channel_mode.get()
        dual = mode.startswith("2-PMT")
        single = None if dual else ("par" if mode.startswith("Parallel")
                  else "perp" if mode.startswith("Perpendicular") else None)
        if dual:
            # 2-PMT: one folder/file, each .dat holds both channels (uses Parallel input).
            if not co or not os.path.exists(co):
                messagebox.showerror(
                    "Error", "Choose a valid 2-PMT file or folder in the Parallel input.")
                return
        elif single is not None:
            # Single-channel: only the chosen channel's input must exist.
            src = (self.cross_folder.get().strip() or co) if single == "perp" else co
            if not src or not os.path.exists(src):
                messagebox.showerror(
                    "Error", f"Choose a valid {'parallel' if single == 'par' else 'perpendicular'} "
                             "file or folder for the single-channel run.")
                return
        else:
            if not co or not os.path.exists(co):
                messagebox.showerror("Error", "Choose a valid parallel file or folder.")
                return
            if not os.path.exists(cross):
                messagebox.showerror("Error", "Choose a valid perpendicular file or folder.")
                return
        if not out:
            messagebox.showerror("Error", "Choose an output path.")
            return
        try:
            dmol = float(self.delta_mol.get())
            cmin = float(self.cal_rmin_m.get())
            cmax = float(self.cal_rmax_m.get())
            pbins = int(self.pretrigger_bins.get())
            energy = float(self.energy_mj.get())
            o_min = float(self.overlap_o_min.get())
            snr_min_val = float(self.snr_min.get())
        except Exception:
            messagebox.showerror("Error", "Invalid numeric parameter.")
            return
        if not (dmol > 0 and cmax > cmin):
            messagebox.showerror("Error", "Need δ_mol > 0 and cal R_max > R_min.")
            return

        ov_mode = self.overlap_mode.get()
        ov_file = self.overlap_file.get().strip()
        ap_co_file = self.afterpulse_co_file.get().strip()
        ap_cr_file = self.afterpulse_cross_file.get().strip()
        glue_mode = ("auto_day_night" if bool(self.glue_auto_daynight.get())
                     else "photon_only" if bool(self.photon_only.get()) else "auto")
        # Read the raw-processing params on the UI thread (tk vars are not
        # thread-safe) and hand the worker a plain dict.
        try:
            raw_p = {
                "dr_m": float(self.bin_spacing_m.get()),
                "dead_time_ns": float(self.dead_time_ns.get()),
                "bg_mode": self.bg_mode.get().strip().lower(),
                "bg_start_m": float(self.bg_start_m.get()),
                "bg_end_m": float(self.bg_end_m.get()),
                "first_signal_bin": int(self.first_signal_bin.get()),
                "first_signal_range_m": float(self.first_signal_range_m.get()),
                "pretrigger_trim_bins": int(self.pretrigger_trim_bins.get()),
                "min_toggle_rate": float(self.min_toggle_rate.get()),
                "max_toggle_rate": float(self.max_toggle_rate.get()),
                "auto_toggle_selector": bool(self.auto_toggle_selector.get()),
                "day_min_toggle_rate": float(self.day_min_toggle_rate.get()),
                "day_max_toggle_rate": float(self.day_max_toggle_rate.get()),
                "toggle_bg_switch_threshold_mhz": float(self.toggle_bg_switch_threshold_mhz.get()),
                "blend_r1_m": float(self.blend_r1_m.get()),
                "blend_r2_m": float(self.blend_r2_m.get()),
                "auto_blend": bool(self.auto_blend.get()),
                "sig_start_m": float(self.sig_start_m.get()),
                "sig_end_m": float(self.sig_end_m.get()),
            }
        except Exception as e:
            messagebox.showerror("Invalid parameter", str(e)); return

        self.run_btn.configure(state="disabled")
        self.progress.set(0.0); self.pb.set(0.0)
        self.status.set("Running…", "running")
        self._log("=== START Depolarization ===")
        pat = self.pattern.get().strip() or "*.dat"
        date_str_val = self.date_str.get().strip()

        def progress_cb(v):
            self._safe(self.progress.set, v)
            self._safe(self.pb.set, max(0.0, min(1.0, v / 100.0)))

        def worker():
            try:
                # Build O(R) + A(R) once from the first pair's range axis
                # (same instrument → same grid for every profile).
                ov_arr = ap_co_arr = ap_cr_arr = None
                try:
                    pairs0 = _depol_resolve(Path(co), Path(cross), pattern=pat)
                    if pairs0:
                        _, first_co, first_cr, _ = pairs0[0]
                        r_axis = self._range_axis_for(
                            first_co, pbins, dr_m=raw_p["dr_m"],
                            first_signal_range_m=raw_p["first_signal_range_m"],
                            first_signal_bin=raw_p["first_signal_bin"])
                        if not ov_mode.startswith("Disabled") and _HAS_OVERLAP:
                            mk = "load" if ov_mode.startswith("Load") else "analytical"
                            ov_arr = _get_overlap_array(mode=mk, r_m_target=r_axis,
                                                        file_path=ov_file or None, hardware=None)
                            self._safe(self._log, f"Overlap O(R): {mk} mode applied")
                        # Calibration .dat may have no pretrigger block (laser
                        # firing, telescope covered) → 0 pretrigger bins + far-range BG.
                        no_pre = bool(self.ap_no_pretrigger.get())
                        ap_kw = dict(
                            pretrigger_bins=0 if no_pre else pbins,
                            first_signal_bin=1 if no_pre else None,
                            pretrigger_trim_bins=0 if no_pre else 24,
                            bg_mode="far_range" if no_pre else "pretrigger",
                        )
                        if ap_co_file and _HAS_AFTERPULSE:
                            ap_co_arr = _get_afterpulse_array(
                                mode="load", r_m_target=r_axis, file_path=ap_co_file, **ap_kw)
                            self._safe(self._log, "Afterpulse parallel applied"
                                       + (" (no-pretrigger cal)" if no_pre else ""))
                        if ap_cr_file and _HAS_AFTERPULSE:
                            ap_cr_arr = _get_afterpulse_array(
                                mode="load", r_m_target=r_axis, file_path=ap_cr_file, **ap_kw)
                            self._safe(self._log, "Afterpulse perpendicular applied"
                                       + (" (no-pretrigger cal)" if no_pre else ""))
                except Exception as ce:
                    self._safe(self._log, f"[WARN] correction setup: {ce}")

                res = _depol_run(
                    Path(co), Path(cross), Path(out),
                    delta_mol=dmol, cal_rmin_m=cmin, cal_rmax_m=cmax,
                    pattern=pat, pretrigger_bins=pbins,
                    date_str=date_str_val,
                    gluing_mode=glue_mode,
                    energy_mj=energy,
                    overlap_O_R=ov_arr, overlap_O_min=o_min,
                    afterpulse_co=ap_co_arr, afterpulse_cross=ap_cr_arr,
                    snr_min=snr_min_val, snr_gate=bool(self.snr_gate.get()),
                    single_channel=single,
                    dual_channel_file=dual,
                    # Full raw-processing parameters (forwarded to build_single_profile
                    # via nrb_kwargs) so Step 3 runs standalone without Step 2.
                    dr_m=raw_p["dr_m"],
                    dead_time_ns=raw_p["dead_time_ns"],
                    bg_mode=raw_p["bg_mode"],
                    bg_start_m=raw_p["bg_start_m"], bg_end_m=raw_p["bg_end_m"],
                    first_signal_bin=raw_p["first_signal_bin"],
                    first_signal_range_m=raw_p["first_signal_range_m"],
                    pretrigger_trim_bins=raw_p["pretrigger_trim_bins"],
                    min_toggle_rate=raw_p["min_toggle_rate"],
                    max_toggle_rate=raw_p["max_toggle_rate"],
                    auto_toggle_selector=raw_p["auto_toggle_selector"],
                    day_min_toggle_rate=raw_p["day_min_toggle_rate"],
                    day_max_toggle_rate=raw_p["day_max_toggle_rate"],
                    toggle_bg_switch_threshold_mhz=raw_p["toggle_bg_switch_threshold_mhz"],
                    blend_r1_m=raw_p["blend_r1_m"], blend_r2_m=raw_p["blend_r2_m"],
                    auto_blend=raw_p["auto_blend"],
                    sig_start_m=raw_p["sig_start_m"], sig_end_m=raw_p["sig_end_m"],
                    strict=False, logger=self._safe_log, progress_cb=progress_cb,
                )
                dv = res["delta_v"]
                qc = res["qc"]
                self._dv_df = dv
                self._qc_df = qc
                self._nrb_co_df = res["nrb_co"]
                self._nrb_cross_df = res["nrb_cross"]
                self._snr_co_df = res.get("snr_co")
                self._snr_cross_df = res.get("snr_cross")
                self._profiles_diag = res.get("profiles_diag", []) or []
                n_ok = int((qc["status"] == "ok").sum()) if "status" in qc else len(qc)
                cmean = float(pd.to_numeric(qc.get("C"), errors="coerce").mean()) if "C" in qc else float("nan")
                self._safe(self._populate_times)
                self._safe(self._log,
                           f"Depol done: {n_ok} pair(s) · mean C = {cmean:.3f} · "
                           f"δ_mol={dmol} · cal {cmin:.0f}-{cmax:.0f} m")
                self._safe(self._log, f"[OK] Saved: {Path(out).resolve()}")
                self._safe(self.progress.set, 100.0); self._safe(self.pb.set, 1.0)
                self._safe(self.status.set, "Done", "ok")
                self._safe(self.refresh_plot)
                self._safe(self.app_state.set_depol, str(Path(out).resolve()))
                # This workbook also carries a normalised "NRB profile" + "SNR"
                # sheet, so publish it as the NRB source — ALT/Fernald can read it
                # directly (auto-fills their NRB field only when still empty, so a
                # dedicated Step-2 run is never clobbered).
                self._safe(self.app_state.set_step2, str(Path(out).resolve()))
                self._safe(self._log,
                           "Depol workbook also provides 'NRB profile' + 'SNR' "
                           "→ usable directly in ALT / Fernald.")
                self._safe(messagebox.showinfo, "Step 3 Complete", f"Saved:\n{out}")
            except Exception as e:
                self._safe(self._log, f"[FAILED] {e}")
                self._safe(self.status.set, "Failed", "error")
                self._safe(messagebox.showerror, "Failed", str(e))
            finally:
                self._safe(self.run_btn.configure, state="normal")
                self._safe(self._log, "=== END ===")

        threading.Thread(target=worker, daemon=True).start()

    # ── Plotting ───────────────────────────────────────────────────────────
    @staticmethod
    def _col_label(c) -> str:
        ts = pd.to_datetime(c, errors="coerce")
        return ts.strftime("%m-%d %H:%M") if pd.notna(ts) else str(c)

    def _populate_times(self):
        self.time_list.delete(0, "end")
        self._ts_labels = []
        if self._dv_df is None:
            return
        for c in [c for c in self._dv_df.columns if c != "Range(m)"]:
            lbl = self._col_label(c)
            self._ts_labels.append(lbl)
            self.time_list.insert("end", lbl)
        if self._ts_labels:
            self.time_list.selection_set(0)

    def _selected_indices(self) -> List[int]:
        sel = list(self.time_list.curselection())
        return sel if sel else ([0] if self._ts_labels else [])

    @staticmethod
    def _parse_lim(s):
        s = str(s).strip()
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
            if redraw:
                self.canvas.draw_idle()
        except Exception:
            pass

    def auto_scale(self):
        for v in [self.xmin_var, self.xmax_var, self.ymin_var, self.ymax_var]:
            v.set("")
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    def _current_plot_df(self):
        """The DataFrame behind the current plot mode (for CSV export)."""
        mode = self.plot_mode.get()
        if mode == "delta_v":
            return self._dv_df
        if mode == "NRB parallel":
            return self._nrb_co_df
        if mode == "NRB perpendicular":
            return self._nrb_cross_df
        # both-channel modes -> merge parallel + perpendicular on Range(m)
        if mode == "SNR parallel + perpendicular":
            a, b = self._snr_co_df, self._snr_cross_df
        else:
            a, b = self._nrb_co_df, self._nrb_cross_df
        if a is None:
            return b
        if b is None:
            return a
        try:
            aa = a.add_suffix(" [parallel]").rename(columns={"Range(m) [parallel]": "Range(m)"})
            bb = b.add_suffix(" [perpendicular]").rename(columns={"Range(m) [perpendicular]": "Range(m)"})
            return aa.merge(bb, on="Range(m)", how="outer")
        except Exception:
            return a

    def save_csv(self):
        df = self._current_plot_df()
        if df is None:
            messagebox.showinfo("No data", "Run the depolarization first.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not out:
            return
        try:
            df.to_csv(out, index=False)
            self._log(f"Saved CSV: {out}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _update_plot_mode_options(self):
        """Switch the Plot dropdown modes when Source changes (mirror Step 2)."""
        src = self.data_source.get()
        if src == "Raw .dat":
            vals = RAW_PLOT_MODES
        elif src == "Processed":
            vals = PROCESSED_PLOT_MODES
        else:
            vals = DEPOL_PLOT_MODES
        if getattr(self, "_plot_mode_menu", None) is not None:
            self._plot_mode_menu.configure(values=vals)
        if self.plot_mode.get() not in vals:
            self.plot_mode.set(vals[0])
        self.refresh_plot()

    def _chan_key(self) -> str:
        return "par" if self.signal_channel.get().startswith("Parallel") else "perp"

    def refresh_plot(self):
        self.ax.clear()
        src = self.data_source.get()
        try:
            if src == "Depolarization":
                if self._dv_df is None:
                    self.ax.set_title("No data — run the depolarization first")
                    self.ax.grid(True, alpha=0.3); self.canvas.draw_idle()
                    return
                mode = self.plot_mode.get()
                if mode == "delta_v":
                    self._plot_delta_v()
                elif mode == "NRB parallel":
                    self._plot_nrb(self._nrb_co_df, "NRB parallel", theme.NAVY)
                elif mode == "NRB perpendicular":
                    self._plot_nrb(self._nrb_cross_df, "NRB perpendicular", theme.ORANGE)
                elif mode == "SNR parallel + perpendicular":
                    self._plot_snr_both()
                else:  # NRB parallel + perpendicular
                    self._plot_nrb_both()
            else:
                self._plot_signal(src)
            # user-typed axis titles override the per-mode defaults
            if self.x_title_var.get().strip():
                self.ax.set_xlabel(self.x_title_var.get())
            if self.y_title_var.get().strip():
                self.ax.set_ylabel(self.y_title_var.get())
            self.apply_axes(redraw=False)
        except Exception as e:
            self.ax.set_title(f"Plot error: {e}")
        self.fig.tight_layout(); self.canvas.draw_idle()

    # ── Per-channel signal views (analog / glue / NRB), like Step 2 ──────────
    def _plot_signal(self, src):
        if not self._profiles_diag:
            self.ax.set_title("No data — run the depolarization first")
            self.ax.grid(True, alpha=0.3); return
        chan = self._chan_key()
        ch_label = self.signal_channel.get()
        mode = self.plot_mode.get()
        sel = self._selected_indices()
        if not sel:
            self.ax.set_title("Select a profile (time) on the left")
            self.ax.grid(True, alpha=0.3); return
        if mode == "Glue Overlay":
            self._plot_signal_glue_overlay(sel[0], chan, ch_label)
        else:
            col = {"Analog Raw": "analog_mV", "Photon Raw": "photon_MHz",
                   "Photon DT": "photon_deadtime_corr_MHz",
                   "Analog Scaled": "analog_scaled_MHz",
                   "Glue": "glued_profile_MHz", "NRB": "nrb"}.get(mode)
            for i in sel:
                if i >= len(self._profiles_diag):
                    continue
                d = self._profiles_diag[i].get(chan)
                if not d or col not in d:
                    continue
                R = np.asarray(d["range_m"], float); y = np.asarray(d[col], float)
                m = np.isfinite(R) & np.isfinite(y)
                if m.any():
                    self.ax.plot(R[m], y[m], lw=1.3,
                                 label=f"{ch_label} {self._ts_labels[i]}")
            self.ax.set_title(self.chart_title_var.get() or f"{ch_label} · {src} · {mode}")
        self.ax.set_xlabel("Range (m)")
        ylab = {"Analog Raw": "Analog (mV)", "Photon Raw": "Photon (MHz)",
                "Photon DT": "Photon DT (MHz)", "Analog Scaled": "Analog scaled (MHz)",
                "Glue": "Glued (MHz)", "NRB": "NRB (norm.)",
                "Glue Overlay": "Signal (MHz)"}.get(mode, "Signal")
        self.ax.set_ylabel(ylab)
        self.ax.grid(True, alpha=0.3)
        if self.ax.get_legend_handles_labels()[0]:
            self.ax.legend(fontsize=8, loc="best")

    def _plot_signal_glue_overlay(self, i, chan, ch_label):
        d = self._profiles_diag[i].get(chan) if i < len(self._profiles_diag) else None
        if not d:
            self.ax.set_title("No data for this profile"); return
        R = np.asarray(d["range_m"], float)

        def _pf(y, **kw):
            y = np.asarray(y, float); m = np.isfinite(R) & np.isfinite(y)
            if m.any():
                self.ax.plot(R[m], y[m], **kw)

        _pf(d["photon_deadtime_corr_MHz"], color=theme.NAVY, lw=1.0, alpha=0.85, label="Photon DT")
        _pf(d["analog_scaled_MHz"], color=theme.GREEN, lw=1.0, ls="--", alpha=0.85, label="Scaled Analog")
        _pf(d["glued_profile_MHz"], color=theme.ORANGE, lw=1.8, label="Glued")
        mt = d.get("meta", {}) or {}
        for key in ("min_toggle_rate", "max_toggle_rate"):
            v = mt.get(key)
            if v is not None and np.isfinite(v):
                self.ax.axhline(float(v), color=theme.TEXT_MUTED, ls=":", lw=0.9, alpha=0.7,
                                label=f"{key.replace('_rate','')} = {float(v):g}")
        for key, color in [("blend_r1_used_m", theme.BROWN), ("blend_r2_used_m", theme.RED)]:
            v = mt.get(key)
            if v is not None and np.isfinite(v):
                self.ax.axvline(float(v), color=color, ls="--", lw=0.9, alpha=0.85,
                                label=f"{key[6:8]} = {float(v):.0f} m")
        self.ax.set_title(self.chart_title_var.get()
                          or f"Glue Overlay · {ch_label} · {self._ts_labels[i]}")

    # All Step-6 plots use X = Range (m), Y = value (like Step 4 profile view).
    def _plot_delta_v(self):
        df = self._dv_df
        R = df["Range(m)"].to_numpy(float)
        cols = [c for c in df.columns if c != "Range(m)"]
        # aerosol-type bands as horizontal spans (δ_v is on the Y axis now)
        band_colors = ["#D9E9F4", "#E5F4EA", "#FFE9DD", "#FFD3B9", "#F3D0D7"]
        for (lo, hi, _lab), col in zip(AEROSOL_TYPE_BANDS or [], band_colors):
            self.ax.axhspan(lo, min(hi, 0.4), color=col, alpha=0.40, zorder=0)
        for i in self._selected_indices():
            if i >= len(cols):
                continue
            y = df[cols[i]].to_numpy(float)
            m = np.isfinite(R) & np.isfinite(y)
            if m.any():
                self.ax.plot(R[m], y[m], lw=1.5, label=self._ts_labels[i])
        self.ax.set_xlabel("Range (m)")
        self.ax.set_ylabel("Volume depolarization ratio  δ_v")
        self.ax.set_xlim(0, 5000)
        self.ax.set_ylim(0, 0.4)
        self.ax.set_title(self.chart_title_var.get() or "Depolarization δ_v")
        self.ax.grid(True, alpha=0.3)
        # Only legend when something was actually drawn — a single-channel run has
        # no delta (all-NaN), so nothing is plotted and legend() would warn.
        if self.ax.get_legend_handles_labels()[0]:
            self.ax.legend(fontsize=8, loc="best")

    def _plot_nrb(self, df, label, color):
        if df is None:
            self.ax.set_title(f"{label}: no data"); return
        R = df["Range(m)"].to_numpy(float)
        cols = [c for c in df.columns if c != "Range(m)"]
        for k, i in enumerate(self._selected_indices()):
            if i >= len(cols):
                continue
            y = df[cols[i]].to_numpy(float)
            m = np.isfinite(R) & np.isfinite(y)
            if m.any():
                self.ax.plot(R[m], y[m], lw=1.4,
                             color=(color if len(self._selected_indices()) == 1 else None),
                             label=self._ts_labels[i])
        self.ax.set_xlabel("Range (m)")
        self.ax.set_ylabel(f"{label}  (MHz·m²·J⁻¹, physical)")
        self.ax.set_xlim(0, 6000)
        self.ax.set_title(self.chart_title_var.get() or f"{label} profile")
        self.ax.grid(True, alpha=0.3)
        if self.ax.get_legend_handles_labels()[0]:
            self.ax.legend(fontsize=8, loc="best")

    def _plot_nrb_both(self):
        if self._nrb_co_df is None or self._nrb_cross_df is None:
            self.ax.set_title("No data"); return
        Rco = self._nrb_co_df["Range(m)"].to_numpy(float)
        Rcr = self._nrb_cross_df["Range(m)"].to_numpy(float)
        co_cols = [c for c in self._nrb_co_df.columns if c != "Range(m)"]
        cr_cols = [c for c in self._nrb_cross_df.columns if c != "Range(m)"]
        for i in self._selected_indices():
            if i < len(co_cols):
                y = self._nrb_co_df[co_cols[i]].to_numpy(float)
                m = np.isfinite(Rco) & np.isfinite(y)
                if m.any():
                    self.ax.plot(Rco[m], y[m], lw=1.5, color=theme.NAVY,
                                 label=f"parallel {self._ts_labels[i]}")
            if i < len(cr_cols):
                y = self._nrb_cross_df[cr_cols[i]].to_numpy(float)
                m = np.isfinite(Rcr) & np.isfinite(y)
                if m.any():
                    self.ax.plot(Rcr[m], y[m], lw=1.5, color=theme.ORANGE, ls="--",
                                 label=f"perpendicular {self._ts_labels[i]}")
        self.ax.set_xlabel("Range (m)")
        self.ax.set_ylabel("NRB  (MHz·m²·J⁻¹, physical)")
        self.ax.set_xlim(0, 6000)
        self.ax.set_title(self.chart_title_var.get() or "NRB parallel (navy) vs perpendicular (orange)")
        self.ax.grid(True, alpha=0.3)
        if self.ax.get_legend_handles_labels()[0]:
            self.ax.legend(fontsize=8, loc="best")

    def _plot_snr_both(self):
        if self._snr_co_df is None or self._snr_cross_df is None:
            self.ax.set_title("No SNR data — re-run Step 6"); return
        Rco = self._snr_co_df["Range(m)"].to_numpy(float)
        Rcr = self._snr_cross_df["Range(m)"].to_numpy(float)
        co_cols = [c for c in self._snr_co_df.columns if c != "Range(m)"]
        cr_cols = [c for c in self._snr_cross_df.columns if c != "Range(m)"]
        for i in self._selected_indices():
            if i < len(co_cols):
                y = self._snr_co_df[co_cols[i]].to_numpy(float)
                m = np.isfinite(Rco) & np.isfinite(y) & (y > 0)
                if m.any():
                    self.ax.plot(Rco[m], y[m], lw=1.5, color=theme.NAVY,
                                 label=f"parallel {self._ts_labels[i]}")
            if i < len(cr_cols):
                y = self._snr_cross_df[cr_cols[i]].to_numpy(float)
                m = np.isfinite(Rcr) & np.isfinite(y) & (y > 0)
                if m.any():
                    self.ax.plot(Rcr[m], y[m], lw=1.5, color=theme.ORANGE, ls="--",
                                 label=f"perpendicular {self._ts_labels[i]}")
        # δ-mask threshold line (cross SNR below this → δ masked)
        try:
            thr = float(self.snr_min.get())
        except Exception:
            thr = 3.0
        self.ax.axhline(thr, color=theme.RED, lw=1.0, ls=":",
                        label=f"δ mask (SNR={thr:g})")
        self.ax.set_xlabel("Range (m)")
        self.ax.set_ylabel("SNR")
        self.ax.set_yscale("log")
        self.ax.set_xlim(0, 6000)
        self.ax.set_title(self.chart_title_var.get() or "SNR parallel (navy) vs perpendicular (orange)")
        self.ax.grid(True, alpha=0.3, which="both")
        if self.ax.get_legend_handles_labels()[0]:
            self.ax.legend(fontsize=8, loc="best")

    def save_png(self):
        out = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if out:
            self.fig.savefig(out, dpi=150, bbox_inches="tight")
            self._log(f"Saved PNG: {out}")

    def save_signal_set(self):
        """Batch-save the standard prototype figure set: for BOTH channels
        (parallel + perpendicular) x Analog raw / Photon raw / Gluing / NRB, as
        separate PNGs, for the currently-selected profile time(s)."""
        if not self._profiles_diag:
            messagebox.showinfo("No data", "Run the depolarization first."); return
        sel = self._selected_indices()
        if not sel:
            messagebox.showinfo("No selection", "Select a profile (time) on the left."); return
        folder = filedialog.askdirectory(title="Folder to save prototype figures")
        if not folder:
            return
        combos = [("Raw .dat", "Analog Raw"), ("Raw .dat", "Photon Raw"),
                  ("Processed", "Glue Overlay"), ("Processed", "NRB")]
        tlabel = self._ts_labels[sel[0]].replace(":", "").replace(" ", "_").replace("-", "") if sel else ""
        saved = (self.data_source.get(), self.signal_channel.get(), self.plot_mode.get())
        n = 0
        try:
            for ch in ("Parallel", "Perpendicular"):
                self.signal_channel.set(ch)
                for src, mode in combos:
                    self.data_source.set(src)
                    self._update_plot_mode_options()   # repoint the Plot menu for this source
                    self.plot_mode.set(mode)
                    self.refresh_plot()
                    tag = mode.replace(" ", "")
                    fp = Path(folder) / f"prototype_{ch}_{tag}_{tlabel}.png"
                    self.fig.savefig(fp, dpi=150, bbox_inches="tight")
                    n += 1
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.data_source.set(saved[0]); self.signal_channel.set(saved[1])
            self._update_plot_mode_options(); self.plot_mode.set(saved[2])
            self.refresh_plot()
        self._log(f"Saved {n} prototype figures -> {folder}")
        messagebox.showinfo("Saved", f"Saved {n} prototype figures to:\n{folder}\n\n"
                            "(Glue Overlay is meaningful only if Step 3 was run with "
                            "'Skip glue' OFF.)")


# ═════════════════════════════════════════════════════════════════════════════
# Placeholder (no longer needed — all 4 steps migrated)
# ═════════════════════════════════════════════════════════════════════════════
class PlaceholderPage(ctk.CTkFrame):
    def __init__(self, master, step_label: str, description: str):
        super().__init__(master, fg_color="transparent")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = PageHeader(self, title=step_label, subtitle=description,
                            badge=("Coming next phase", "navy"))
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))

        card = Card(self)
        card.grid(row=1, column=0, sticky="nsew")

        inner = ctk.CTkFrame(card.body, fg_color="transparent")
        inner.pack(expand=True)

        ctk.CTkLabel(inner, text="🚧", font=(theme.FONT_FAMILY, 56)).pack(pady=(40, 12))
        ctk.CTkLabel(
            inner, text="Migration in progress", font=theme.F_H1, text_color=theme.TEXT_PRIMARY,
        ).pack()
        ctk.CTkLabel(
            inner,
            text="This step will be ported to the new theme in the next phase.\n"
                 "All logic and outputs remain available in main.py for now.",
            font=theme.F_BODY, text_color=theme.TEXT_SECONDARY, justify="center",
        ).pack(pady=(8, 40))


# ═════════════════════════════════════════════════════════════════════════════
# Main Application
# ═════════════════════════════════════════════════════════════════════════════
class App(ctk.CTk):
    def __init__(self):
        super().__init__(fg_color=theme.APP_BG)
        self.title("LiDAR Analysis Suite")
        self.geometry("1400x880")
        self.minsize(1100, 700)

        theme.apply_matplotlib_style()

        self.app_state = AppState()

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = Sidebar(self, on_navigate=self.show_step)
        self.sidebar.grid(row=0, column=0, sticky="nsw")

        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.grid(row=0, column=1, sticky="nsew", padx=24, pady=24)
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        self.pages: Dict[str, ctk.CTkFrame] = {}
        self._current_page: Optional[str] = None

        # Real pages (all 5 steps migrated)
        self.pages["step1"] = Step1Page(self.content, self.app_state)
        self.pages["step2"] = Step2Page(self.content, self.app_state)
        self.pages["step3"] = Step3Page(self.content, self.app_state)
        self.pages["step4"] = Step4Page(self.content, self.app_state)
        self.pages["step5"] = Step5Page(self.content, self.app_state)
        self.pages["step6"] = Step6Page(self.content, self.app_state)

        self.show_step("step1")

    @staticmethod
    def _desc_for(step_id: str) -> str:
        return {
            "step1": "Build the rmin-rmax search table from MPL data.",
            "step2": "Convert TR40 .dat files into a daily NRB profile workbook.",
            "step3": "Detect aerosol layer top (ALT) using FFT + HWCT.",
            "step4": "Display / Visualize — RTI, profile comparison, ALT overlay.",
            "step5": "Fernald/Klett inversion — aerosol backscatter, extinction, AOD.",
            "step6": "Depolarization ratio + aerosol typing from parallel / perpendicular .dat files.",
        }.get(step_id, "")

    def show_step(self, step_id: str):
        if step_id not in self.pages:
            return
        if self._current_page == step_id:
            return
        for sid, page in self.pages.items():
            page.grid_forget()
        self.pages[step_id].grid(row=0, column=0, sticky="nsew")
        self._current_page = step_id
        self.sidebar.set_active(step_id)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
