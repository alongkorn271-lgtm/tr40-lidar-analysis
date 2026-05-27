# modern_theme.py
# Shared modern tkinter/ttk theme for LiDAR GUI suite
# Drop this file in the same folder as the GUI scripts.

import platform
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, Tuple

# ── Color Palette ─────────────────────────────────────────────────────────────
DARK      = "#1A2D4F"    # header / sidebar dark blue
PRIMARY   = "#2563EB"    # primary blue (buttons, accents)
PLIGHT    = "#3B82F6"    # lighter primary (hover)
LBLUE     = "#DBEAFE"    # very light blue (hover bg)
BG        = "#F1F5F9"    # window background
CARD      = "#FFFFFF"    # card / panel background
BORDER    = "#E2E8F0"    # border / separator
TEXT      = "#1E293B"    # primary text
SUBTEXT   = "#64748B"    # secondary / hint text
SUCCESS   = "#10B981"    # green
WARNING   = "#F59E0B"    # amber
ERROR     = "#EF4444"    # red
TERMINAL  = "#0F172A"    # dark log background
TERM_FG   = "#94A3B8"    # log default text

# ── Fonts ─────────────────────────────────────────────────────────────────────
_sys = platform.system()
FF = "Segoe UI" if _sys == "Windows" else ("SF Pro Display" if _sys == "Darwin" else "Ubuntu")
F_H1    = (FF, 15, "bold")
F_H2    = (FF, 11, "bold")
F_BODY  = (FF, 10)
F_SMALL = (FF, 9)
F_MONO  = ("Consolas" if _sys == "Windows" else "Courier New", 9)


# ── Apply TTK Theme ────────────────────────────────────────────────────────────
def apply_theme(root: tk.Tk) -> ttk.Style:
    """Configure ttk styles for a modern look.  Call once per Tk root."""
    root.configure(bg=BG)
    s = ttk.Style(root)
    s.theme_use("clam")

    # Frame
    s.configure("TFrame",        background=BG)
    s.configure("Card.TFrame",   background=CARD)
    s.configure("Dark.TFrame",   background=DARK)

    # Label
    s.configure("TLabel",        background=BG,   foreground=TEXT,    font=F_BODY)
    s.configure("H2.TLabel",     background=BG,   foreground=TEXT,    font=F_H2)
    s.configure("Sub.TLabel",    background=BG,   foreground=SUBTEXT, font=F_SMALL)
    s.configure("Card.TLabel",   background=CARD, foreground=TEXT,    font=F_BODY)
    s.configure("CardH2.TLabel", background=CARD, foreground=PRIMARY, font=F_H2)

    # LabelFrame
    s.configure("TLabelframe",         background=CARD, bordercolor=BORDER, relief="flat")
    s.configure("TLabelframe.Label",   background=CARD, foreground=PRIMARY, font=F_H2)
    s.configure("Card.TLabelframe",    background=CARD, bordercolor=BORDER, relief="flat")
    s.configure("Card.TLabelframe.Label", background=CARD, foreground=PRIMARY, font=F_H2)

    # Buttons
    s.configure("P.TButton",   background=PRIMARY, foreground="white",
                font=F_BODY, borderwidth=0, focusthickness=0, padding=(14, 7))
    s.map("P.TButton",
          background=[("active", PLIGHT), ("pressed", DARK), ("disabled", BORDER)],
          foreground=[("active", "white"),  ("disabled", SUBTEXT)])

    s.configure("S.TButton",   background=CARD, foreground=PRIMARY,
                font=F_BODY, borderwidth=1, focusthickness=0, padding=(14, 7))
    s.map("S.TButton",
          background=[("active", LBLUE)],
          foreground=[("active", PRIMARY)])

    s.configure("Sm.TButton",  background=CARD, foreground=PRIMARY,
                font=F_SMALL, borderwidth=1, focusthickness=0, padding=(10, 5))
    s.map("Sm.TButton",
          background=[("active", LBLUE)])

    # Entry
    s.configure("TEntry",
                fieldbackground=CARD, foreground=TEXT, font=F_BODY,
                padding=(8, 5), relief="flat",
                bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
                insertcolor=TEXT)
    s.map("TEntry",
          bordercolor=[("focus", PRIMARY), ("!focus", BORDER)],
          lightcolor=[("focus", PRIMARY)],
          darkcolor=[("focus", PRIMARY)])

    # Progressbar
    s.configure("TProgressbar",
                troughcolor=BORDER, background=PRIMARY,
                borderwidth=0, thickness=7, relief="flat")
    s.configure("G.TProgressbar",
                troughcolor=BORDER, background=SUCCESS,
                borderwidth=0, thickness=7)

    # Checkbutton / Radiobutton
    for w in ("TCheckbutton", "TRadiobutton"):
        s.configure(w, background=BG,   foreground=TEXT, font=F_BODY, focusthickness=0)
        s.map(w,   background=[("active", BG)],
                   indicatorcolor=[("selected", PRIMARY), ("!selected", BORDER)])
    for w in ("Card.TCheckbutton", "Card.TRadiobutton"):
        base = w.split(".")[1]
        s.configure(w, background=CARD, foreground=TEXT, font=F_BODY, focusthickness=0)
        s.map(w,   background=[("active", CARD)],
                   indicatorcolor=[("selected", PRIMARY), ("!selected", BORDER)])

    # Combobox
    s.configure("TCombobox",
                fieldbackground=CARD, foreground=TEXT, font=F_BODY,
                background=CARD, padding=(8, 5), relief="flat",
                bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
    s.map("TCombobox",
          fieldbackground=[("readonly", CARD)],
          bordercolor=[("focus", PRIMARY)])

    # Scrollbar (thin / subtle)
    s.configure("TScrollbar",
                background=BORDER, troughcolor=BG,
                borderwidth=0, arrowsize=12, relief="flat")
    s.map("TScrollbar", background=[("active", SUBTEXT)])

    return s


# ── Widget Helpers ─────────────────────────────────────────────────────────────
def make_card(parent: tk.Widget, title: Optional[str] = None,
              padx: int = 14, pady: int = 12,
              bg: str = CARD) -> Tuple[tk.Frame, tk.Frame]:
    """Return (outer_frame, inner_frame).  outer has the card border."""
    outer = tk.Frame(parent, bg=bg,
                     highlightbackground=BORDER, highlightthickness=1)
    inner = tk.Frame(outer, bg=bg)
    inner.pack(fill="both", expand=True, padx=padx, pady=pady)
    if title:
        tk.Label(inner, text=title, bg=bg, fg=PRIMARY, font=F_H2).pack(anchor="w", pady=(0, 10))
    return outer, inner


def make_separator(parent: tk.Widget, bg: str = BG) -> tk.Frame:
    """1 px horizontal separator."""
    return tk.Frame(parent, bg=BORDER, height=1)


def make_header(parent: tk.Widget, title: str, subtitle: str = "") -> tk.Frame:
    """Dark top header bar."""
    bar = tk.Frame(parent, bg=DARK, height=54)
    bar.pack_propagate(False)
    inner = tk.Frame(bar, bg=DARK)
    inner.pack(fill="both", expand=True, padx=20)
    tk.Label(inner, text=title,    bg=DARK, fg="white",  font=F_H1).pack(side="left", pady=14)
    if subtitle:
        tk.Label(inner, text=subtitle, bg=DARK, fg=SUBTEXT, font=F_SMALL).pack(
            side="left", padx=(10, 0), pady=14)
    return bar


def make_status_bar(parent: tk.Widget):
    """Dark bottom status bar.  Returns (bar_frame, string_var, set_fn)."""
    bar = tk.Frame(parent, bg=DARK, height=28)
    bar.pack_propagate(False)
    var = tk.StringVar(value="Ready.")
    dot = tk.Label(bar, text="●", bg=DARK, fg=SUBTEXT, font=F_SMALL)
    dot.pack(side="right", padx=12, pady=4)
    tk.Label(bar, textvariable=var, bg=DARK, fg="#CBD5E1",
             font=F_SMALL, anchor="w").pack(side="left", padx=12, pady=4)

    _dot_colors = {"ok": SUCCESS, "error": ERROR, "running": WARNING, "normal": SUBTEXT}

    def set_fn(msg: str, state: str = "normal"):
        var.set(msg)
        dot.configure(fg=_dot_colors.get(state, SUBTEXT))

    return bar, var, set_fn


def make_log(parent: tk.Widget, height: int = 12) -> Tuple[tk.Frame, tk.Text]:
    """Dark terminal log.  Returns (outer_frame, text_widget)."""
    outer = tk.Frame(parent, bg=TERMINAL,
                     highlightbackground=BORDER, highlightthickness=1)
    txt = tk.Text(outer, height=height, wrap="word",
                  bg=TERMINAL, fg=TERM_FG, font=F_MONO,
                  relief="flat", borderwidth=0,
                  insertbackground="white",
                  selectbackground=PRIMARY,
                  exportselection=False)
    sb = ttk.Scrollbar(outer, orient="vertical", command=txt.yview)
    txt.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    txt.pack(side="left", fill="both", expand=True, padx=6, pady=6)
    txt.configure(state="disabled")
    # colour tags
    txt.tag_configure("ok",   foreground="#10B981")
    txt.tag_configure("err",  foreground="#EF4444")
    txt.tag_configure("warn", foreground="#F59E0B")
    txt.tag_configure("info", foreground="#60A5FA")
    txt.tag_configure("sep",  foreground="#334155")
    return outer, txt


def append_log(txt: tk.Text, msg: str, tag: str = "") -> None:
    """Append a line to a log text widget, auto-tagging by content."""
    txt.configure(state="normal")
    if not tag:
        m = msg.lower()
        if   "[ok]"    in m or "✅" in m: tag = "ok"
        elif "[failed]" in m or "❌" in m or "[error]" in m: tag = "err"
        elif "[warn]"  in m or "⚠"  in m: tag = "warn"
        elif "==="     in m:              tag = "sep"
        elif m.strip().startswith("["):   tag = "info"
    if tag:
        txt.insert("end", msg + "\n", tag)
    else:
        txt.insert("end", msg + "\n")
    txt.see("end")
    txt.configure(state="disabled")


# ── Convenience row builders ───────────────────────────────────────────────────
def lbl_entry(parent: tk.Widget, label: str, var, width: int = 14,
              bg: str = CARD) -> ttk.Entry:
    """Label + Entry in a row Frame, packed into parent."""
    row = tk.Frame(parent, bg=bg)
    row.pack(fill="x", pady=(0, 6))
    tk.Label(row, text=label, bg=bg, fg=SUBTEXT, font=F_SMALL,
             width=22, anchor="w").pack(side="left")
    e = ttk.Entry(row, textvariable=var, width=width)
    e.pack(side="left", fill="x", expand=True)
    return e


def lbl_check(parent: tk.Widget, label: str, var,
              bg: str = CARD, **kw) -> ttk.Checkbutton:
    cb = ttk.Checkbutton(parent, text=label, variable=var,
                         style="Card.TCheckbutton", **kw)
    cb.pack(anchor="w", pady=(0, 4))
    return cb
