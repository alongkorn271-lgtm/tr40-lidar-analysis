"""Bao Solutions inspired theme for the LiDAR Analysis Suite.

Color palette and typography mirror the Bao Solutions dashboard design system
(see project Behance reference). All UI modules should pull tokens from here so
the look stays consistent.
"""
from __future__ import annotations

import platform

# ── Color tokens ────────────────────────────────────────────────────────────
# Main brand colors
NAVY       = "#35375B"   # primary text, headings, sidebar background option
ORANGE     = "#FF5632"   # primary accent (CTAs, active state, logo)
CHARCOAL   = "#2B2727"   # secondary deep text

# Secondary
GREEN      = "#00BE7C"   # success
RED        = "#FF3257"   # error / destructive
LIGHT_GRAY = "#EFEFEF"   # dividers, disabled
BROWN      = "#B15224"   # tertiary accent (rare)
CREAM      = "#F0E7DF"   # soft hover / chip background

# Surfaces
APP_BG     = "#F5F4F2"   # off-white app background (cream tint)
CARD_BG    = "#FFFFFF"   # card surface
SIDEBAR_BG = "#FFFFFF"   # sidebar surface
BORDER     = "#E7E2DC"   # subtle borders
DIVIDER    = "#EFEFEF"

# Text
TEXT_PRIMARY   = NAVY
TEXT_SECONDARY = "#6B6E84"
TEXT_MUTED     = "#9CA0B4"
TEXT_ON_ORANGE = "#FFFFFF"

# State tints (soft backgrounds for badges / inputs)
ORANGE_SOFT = "#FFE4DC"
GREEN_SOFT  = "#D7F4E8"
RED_SOFT    = "#FFE0E6"
NAVY_SOFT   = "#E6E7EE"
CREAM_SOFT  = "#FAF6F2"

# Hover variants
ORANGE_HOVER = "#E94A2A"
NAVY_HOVER   = "#2A2C49"


# ── Typography ──────────────────────────────────────────────────────────────
# Nimbus Sans is rarely available; pick the closest installed sans-serif.
def _pick_font_family() -> str:
    candidates = ["Nimbus Sans", "Inter", "Segoe UI Variable", "Segoe UI", "Helvetica Neue", "Arial"]
    # On Windows, Segoe UI is always available; on macOS Helvetica Neue; on Linux DejaVu Sans.
    sys_name = platform.system()
    if sys_name == "Windows":
        for c in candidates:
            if c in ("Segoe UI Variable", "Segoe UI"):
                return c
    elif sys_name == "Darwin":
        return "Helvetica Neue"
    return "Segoe UI"


FONT_FAMILY = _pick_font_family()

# Sizes (matching the Bao spec: 36 / 20 / 16 / 14 / 12)
F_DISPLAY = (FONT_FAMILY, 36, "bold")    # page title
F_H1      = (FONT_FAMILY, 20, "bold")    # section heading
F_H2      = (FONT_FAMILY, 16, "bold")    # card heading
F_BODY    = (FONT_FAMILY, 14)            # default body
F_BODY_B  = (FONT_FAMILY, 14, "bold")
F_SMALL   = (FONT_FAMILY, 12)            # captions, labels
F_TINY    = (FONT_FAMILY, 11)


# ── Layout constants ────────────────────────────────────────────────────────
SIDEBAR_WIDTH       = 240
SIDEBAR_NAV_HEIGHT  = 44
RADIUS_CARD         = 16
RADIUS_BUTTON       = 12
RADIUS_INPUT        = 10
RADIUS_PILL         = 999

CARD_PAD_X          = 20
CARD_PAD_Y          = 18
SECTION_GAP         = 16


# ── CustomTkinter widget defaults ───────────────────────────────────────────
# Used by the application to create widgets consistently. Pass these as
# **STYLES["primary_button"] when constructing a CTkButton, for example.

def primary_button_style(**overrides) -> dict:
    d = dict(
        fg_color=ORANGE,
        hover_color=ORANGE_HOVER,
        text_color=TEXT_ON_ORANGE,
        corner_radius=RADIUS_BUTTON,
        height=40,
        font=F_BODY_B,
    )
    d.update(overrides)
    return d


def secondary_button_style(**overrides) -> dict:
    d = dict(
        fg_color=NAVY_SOFT,
        hover_color=CREAM,
        text_color=TEXT_PRIMARY,
        corner_radius=RADIUS_BUTTON,
        height=40,
        font=F_BODY,
    )
    d.update(overrides)
    return d


def ghost_button_style(**overrides) -> dict:
    d = dict(
        fg_color="transparent",
        hover_color=CREAM_SOFT,
        text_color=TEXT_SECONDARY,
        corner_radius=RADIUS_BUTTON,
        height=36,
        font=F_BODY,
    )
    d.update(overrides)
    return d


def nav_item_style(active: bool = False) -> dict:
    """Sidebar nav item — pill background when active."""
    if active:
        return dict(
            fg_color=ORANGE_SOFT,
            hover_color=ORANGE_SOFT,
            text_color=ORANGE,
            anchor="w",
            corner_radius=RADIUS_BUTTON,
            height=SIDEBAR_NAV_HEIGHT,
            font=F_BODY_B,
        )
    return dict(
        fg_color="transparent",
        hover_color=CREAM_SOFT,
        text_color=TEXT_PRIMARY,
        anchor="w",
        corner_radius=RADIUS_BUTTON,
        height=SIDEBAR_NAV_HEIGHT,
        font=F_BODY,
    )


def card_style() -> dict:
    return dict(
        fg_color=CARD_BG,
        corner_radius=RADIUS_CARD,
        border_width=1,
        border_color=BORDER,
    )


def input_style() -> dict:
    return dict(
        fg_color=CARD_BG,
        border_color=BORDER,
        border_width=1,
        text_color=TEXT_PRIMARY,
        corner_radius=RADIUS_INPUT,
        height=36,
        font=F_BODY,
    )


def chip_style(tone: str = "default") -> dict:
    """Small badge/chip — tone in {default, orange, green, red, navy}."""
    palette = {
        "default": (CREAM, TEXT_PRIMARY),
        "orange":  (ORANGE_SOFT, ORANGE),
        "green":   (GREEN_SOFT, GREEN),
        "red":     (RED_SOFT, RED),
        "navy":    (NAVY_SOFT, NAVY),
    }
    bg, fg = palette.get(tone, palette["default"])
    return dict(fg_color=bg, text_color=fg, corner_radius=RADIUS_PILL, font=F_TINY)


# ── matplotlib styling (use after import matplotlib.pyplot as plt) ──────────
def apply_matplotlib_style():
    """Apply Bao-style defaults to matplotlib."""
    try:
        import matplotlib as mpl
        mpl.rcParams.update({
            "figure.facecolor": APP_BG,
            "axes.facecolor": CARD_BG,
            "axes.edgecolor": BORDER,
            "axes.labelcolor": TEXT_PRIMARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "grid.color": DIVIDER,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.family": FONT_FAMILY,
            "font.size": 10,
        })
    except Exception:
        pass


# ── Convenience: status / log color mapping for the legacy modern_theme API ──
SUCCESS = GREEN
WARNING = "#E0900E"
ERROR   = RED
PRIMARY = ORANGE
