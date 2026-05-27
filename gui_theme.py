# gui_theme.py
"""
Modern Dashboard Theme for LiDAR GUI Tools
Blue / White card-based design inspired by modern analytics dashboards.
"""
import tkinter as tk
from tkinter import ttk

# ── Colour palette ────────────────────────────────────────────────────────────
BG        = "#EEF2F7"   # app background (light blue-gray)
CARD      = "#FFFFFF"   # card / panel background
BORDER    = "#D0DAE8"   # card border
SIDEBAR   = "#1A2B4A"   # dark sidebar
PRIMARY   = "#2563EB"   # blue accent
PRIMARY_H = "#1D4ED8"   # blue hover
ACCENT    = "#EFF6FF"   # blue tint background
TEXT      = "#0F172A"   # primary text
MUTED     = "#64748B"   # secondary text
SUCCESS   = "#10B981"
WARN      = "#F59E0B"
DANGER    = "#EF4444"
LOG_BG    = "#F8FAFC"
STRIPE    = "#F1F5F9"   # table/log stripe

# ── Typography ────────────────────────────────────────────────────────────────
FF       = "Segoe UI"
FONT     = (FF, 10)
FONT_SM  = (FF, 9)
FONT_B   = (FF, 10, "bold")
FONT_LG  = (FF, 12, "bold")
FONT_XL  = (FF, 14, "bold")
FONT_H   = (FF, 16, "bold")
FONT_LOG = ("Consolas", 9)

# ── ttk Style Setup ───────────────────────────────────────────────────────────
def apply_theme(root):
    """Call once after creating the Tk root window."""
    root.configure(bg=BG)
    s = ttk.Style(root)
    s.theme_use("clam")

    # Base
    s.configure(".", background=BG, foreground=TEXT, font=FONT,
                relief="flat", borderwidth=0)
    s.configure("TFrame", background=BG)
    s.configure("TLabel", background=BG, foreground=TEXT, font=FONT)

    # Card frame (white)
    s.configure("Card.TFrame", background=CARD)
    s.configure("Card.TLabel", background=CARD, foreground=TEXT, font=FONT)
    s.configure("CardMuted.TLabel", background=CARD, foreground=MUTED, font=FONT_SM)
    s.configure("CardBold.TLabel", background=CARD, foreground=TEXT, font=FONT_B)
    s.configure("CardPrimary.TLabel", background=CARD, foreground=PRIMARY, font=FONT_B)
    s.configure("CardH.TLabel", background=CARD, foreground=TEXT, font=FONT_LG)

    # LabelFrame (card-section)
    s.configure("Card.TLabelframe",
                background=CARD, bordercolor=BORDER,
                lightcolor=BORDER, darkcolor=BORDER, relief="solid")
    s.configure("Card.TLabelframe.Label",
                background=CARD, foreground=PRIMARY, font=FONT_B)

    # ── Buttons ───────────────────────────────────────────────────────────────
    # Primary blue
    s.configure("P.TButton",
                background=PRIMARY, foreground="white",
                font=FONT_B, padding=(18, 10),
                relief="flat", borderwidth=0, focuscolor=PRIMARY_H)
    s.map("P.TButton",
          background=[("active", PRIMARY_H), ("disabled", "#93C5FD")],
          foreground=[("disabled", "#E0F2FE")])

    # Secondary outlined
    s.configure("S.TButton",
                background=CARD, foreground=PRIMARY,
                font=FONT, padding=(12, 8),
                relief="solid", borderwidth=1, focuscolor=CARD)
    s.map("S.TButton",
          background=[("active", ACCENT)],
          foreground=[("active", PRIMARY_H)])

    # Ghost (subtle)
    s.configure("G.TButton",
                background=BG, foreground=MUTED,
                font=FONT_SM, padding=(10, 6), relief="flat")
    s.map("G.TButton",
          background=[("active", CARD)],
          foreground=[("active", PRIMARY)])

    # ── Entry ─────────────────────────────────────────────────────────────────
    s.configure("TEntry",
                fieldbackground=CARD, foreground=TEXT,
                bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
                insertcolor=PRIMARY, padding=(8, 7))
    s.map("TEntry",
          bordercolor=[("focus", PRIMARY), ("!focus", BORDER)],
          lightcolor=[("focus", PRIMARY), ("!focus", BORDER)],
          darkcolor=[("focus", PRIMARY), ("!focus", BORDER)])

    # ── Combobox ──────────────────────────────────────────────────────────────
    s.configure("TCombobox",
                fieldbackground=CARD, foreground=TEXT,
                background=CARD, arrowcolor=PRIMARY,
                bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
                padding=(6, 6))
    s.map("TCombobox",
          fieldbackground=[("readonly", CARD)],
          bordercolor=[("focus", PRIMARY)])

    # ── Check / Radio ─────────────────────────────────────────────────────────
    s.configure("TCheckbutton", background=CARD, foreground=TEXT, font=FONT)
    s.map("TCheckbutton", background=[("active", CARD)])
    s.configure("TRadiobutton", background=CARD, foreground=TEXT, font=FONT)
    s.map("TRadiobutton", background=[("active", CARD)])

    # ── Progressbar ───────────────────────────────────────────────────────────
    s.configure("Modern.Horizontal.TProgressbar",
                troughcolor="#DBEAFE", background=PRIMARY,
                thickness=8, borderwidth=0, relief="flat")

    # ── Scale ─────────────────────────────────────────────────────────────────
    s.configure("TScale", background=CARD,
                troughcolor=BORDER, sliderlength=16)

    # ── Scrollbar ─────────────────────────────────────────────────────────────
    s.configure("TScrollbar",
                background=BORDER, troughcolor=CARD,
                arrowcolor=MUTED, borderwidth=0, relief="flat",
                arrowsize=12)
    s.map("TScrollbar", background=[("active", MUTED)])

    # ── Separator ─────────────────────────────────────────────────────────────
    s.configure("TSeparator", background=BORDER)

    return s


# ── Widget helpers ────────────────────────────────────────────────────────────

def header_bar(parent, title, subtitle=None, bg=PRIMARY):
    """Blue header strip at the top of a window."""
    bar = tk.Frame(parent, bg=bg)
    tk.Label(bar, text=title, bg=bg, fg="white", font=FONT_H,
             padx=20, pady=14).pack(anchor="w")
    if subtitle:
        tk.Label(bar, text=subtitle, bg=bg, fg="#BFDBFE",
                 font=FONT_SM, padx=20).pack(anchor="w", pady=(0, 10))
    return bar


def make_card(parent, title=None, subtitle=None, pad=14, fill=False, expand=False):
    """
    Returns (outer_wrap, inner_frame).
    outer_wrap  – use .pack() / .grid() on this
    inner_frame – place widgets inside this (bg=CARD)
    """
    outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
    inner = tk.Frame(outer, bg=CARD, padx=pad, pady=pad)
    inner.pack(fill="both" if fill else "none", expand=expand)
    if title:
        tk.Label(inner, text=title, bg=CARD, fg=PRIMARY,
                 font=FONT_B).pack(anchor="w", pady=(0, 6))
    if subtitle:
        tk.Label(inner, text=subtitle, bg=CARD, fg=MUTED,
                 font=FONT_SM).pack(anchor="w", pady=(0, 8))
    return outer, inner


def divider(parent, bg=BG, pady=6):
    f = tk.Frame(parent, bg=BORDER, height=1)
    return f


def section_title(parent, text, bg=BG):
    """Blue bold section label on app background."""
    return tk.Label(parent, text=text, bg=bg, fg=PRIMARY, font=FONT_B)


def muted_label(parent, text, bg=CARD):
    return tk.Label(parent, text=text, bg=bg, fg=MUTED, font=FONT_SM)


def log_widget(parent, height=12):
    """Returns (outer_frame, text_widget)."""
    outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
    inner = tk.Frame(outer, bg=LOG_BG)
    inner.pack(fill="both", expand=True)
    text = tk.Text(inner, height=height, bg=LOG_BG, fg="#334155",
                   font=FONT_LOG, relief="flat", bd=8,
                   wrap="word", insertbackground=PRIMARY,
                   selectbackground=ACCENT, selectforeground=TEXT)
    sb = ttk.Scrollbar(inner, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=sb.set)
    text.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")
    text.configure(state="disabled")
    return outer, text


def status_row(parent, status_var, bg=BG):
    """Returns a frame with a coloured indicator dot + status text."""
    f = tk.Frame(parent, bg=bg)
    dot = tk.Label(f, text="●", bg=bg, fg=SUCCESS, font=(FF, 10))
    dot.pack(side="left", padx=(0, 6))
    tk.Label(f, textvariable=status_var, bg=bg, fg=MUTED,
             font=FONT_SM).pack(side="left")
    return f, dot


def badge(parent, text, color=PRIMARY, bg=CARD):
    """Small colour badge/pill."""
    f = tk.Frame(parent, bg=color, padx=8, pady=2)
    tk.Label(f, text=text, bg=color, fg="white",
             font=(FF, 8, "bold")).pack()
    return f


class ScrollableFrame(tk.Frame):
    """A vertically scrollable frame container."""
    def __init__(self, master, width=None, bg=BG, **kwargs):
        super().__init__(master, bg=bg, **kwargs)
        kw = {}
        if width:
            kw["width"] = width
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, **kw)
        self.vsb = ttk.Scrollbar(self, orient="vertical",
                                 command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window((0, 0), window=self.inner,
                                              anchor="nw")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_canvas)
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)

    def _on_inner(self, _e=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, e):
        self.canvas.itemconfig(self._win, width=e.width)

    def _on_wheel(self, e):
        w = self.winfo_containing(e.x_root, e.y_root)
        if w and str(w).startswith(str(self.canvas)):
            self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
