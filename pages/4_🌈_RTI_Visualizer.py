"""Step 4 — Range-Time-Intensity visualizer (Prototype vs MPL + ALT overlay)."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

st.set_page_config(page_title="Step 4 · RTI Visualizer", page_icon="🌈", layout="wide")
st.title("🌈 Step 4 · RTI Visualizer")
st.caption("Prototype RTI vs Mini-MPL RTI · ALT overlay (white line) · ALT compare panel")


def _read_wide_nrb(uploaded_or_bytes, candidate_sheets):
    """Read first column = Range(m), rest = timestamps. Try sheet candidates."""
    src = io.BytesIO(uploaded_or_bytes) if isinstance(uploaded_or_bytes, (bytes, bytearray)) else uploaded_or_bytes
    xl = pd.ExcelFile(src)
    sheet = next((s for s in candidate_sheets if s in xl.sheet_names), xl.sheet_names[0])
    df = pd.read_excel(xl, sheet_name=sheet)
    R = df.iloc[:, 0].to_numpy(float)
    Z = df.iloc[:, 1:].to_numpy(float)
    ts = [pd.to_datetime(c, errors="coerce") for c in df.columns[1:]]
    return R, np.array(ts), Z


def _pick(label, sess_key, types=("xlsx",)):
    has_prev = sess_key in st.session_state
    if has_prev:
        use = st.checkbox(f"Use `{st.session_state[sess_key + '_name']}` from session", value=True, key=f"{sess_key}_use")
        if use:
            return st.session_state[sess_key]
    up = st.file_uploader(label, type=list(types), key=f"{sess_key}_up")
    return up.getvalue() if up else None


# ── Inputs ──────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Inputs")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Prototype NRB**")
        proto_b = _pick("Prototype NRB workbook", "step2_xlsx_bytes")
    with c2:
        st.markdown("**MPL NRB**")
        mpl_b = _pick("MPL workbook (Step 1 output)", "step1_xlsx_bytes")
    with c3:
        st.markdown("**ALT results** (optional)")
        alt_b = _pick("ALT_results workbook (Step 3 output)", "step3_xlsx_bytes")

if proto_b is None or mpl_b is None:
    st.warning("Need both Prototype and MPL workbooks.")
    st.stop()

# Read all
R_p, t_p, Z_p = _read_wide_nrb(proto_b, ["NRB profile", "Input_NRB_raw"])
R_m, t_m, Z_m = _read_wide_nrb(mpl_b, ["copol_nrb_norm", "NRB profile"])

# ── Display controls ────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Display")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        show_proto = st.checkbox("Prototype RTI", value=True)
        show_mpl = st.checkbox("Mini-MPL RTI", value=True)
    with c2:
        show_alt_overlay = st.checkbox("ALT overlay (white line)", value=alt_b is not None)
        show_alt_cmp = st.checkbox("ALT Compare panel", value=alt_b is not None)
    with c3:
        rmax = st.number_input("y-max (m)", value=6000.0, step=500.0)
        cmap = st.selectbox("Colormap", ["jet", "turbo", "viridis"], index=0)
    with c4:
        vmin = st.number_input("NRB vmin", value=0.0, step=0.05)
        vmax = st.number_input("NRB vmax", value=1.0, step=0.05)
        tol_min = st.number_input("Match ± (min)", value=3, step=1)

# Align MPL → prototype timeline
def _interp_to_grid(rs, Zs, r_target):
    """Resample MPL Z (in src range) onto r_target."""
    Z_out = np.full((len(r_target), Zs.shape[1]), np.nan)
    for j in range(Zs.shape[1]):
        Z_out[:, j] = np.interp(r_target, rs, Zs[:, j], left=np.nan, right=np.nan)
    return Z_out

# match each t_p to nearest t_m within tol
def _match_idx(t_a, t_b, tol_min):
    out = [None] * len(t_a)
    for i, ta in enumerate(t_a):
        if pd.isna(ta):
            continue
        dt = np.abs((t_b - ta).astype("timedelta64[s]").astype(float) / 60.0)
        if not np.isfinite(dt).any():
            continue
        j = int(np.nanargmin(dt))
        if dt[j] <= tol_min:
            out[i] = j
    return out

idx_map = _match_idx(t_p, t_m, tol_min)
Z_mpl_on_p = np.full_like(Z_p, np.nan)
Z_m_on_grid = _interp_to_grid(R_m, Z_m, R_p)
for i, j in enumerate(idx_map):
    if j is not None:
        Z_mpl_on_p[:, i] = Z_m_on_grid[:, j]

n_matched = sum(j is not None for j in idx_map)
st.info(f"Aligned: {n_matched}/{len(t_p)} prototype profiles matched MPL within ±{tol_min} min")

# ── ALT alignment ───────────────────────────────────────────────────────────
alt_proto = alt_mpl = None
if alt_b is not None:
    alt_df = pd.read_excel(io.BytesIO(alt_b), sheet_name="ALT_results")
    alt_df["Time"] = pd.to_datetime(alt_df["Time"], errors="coerce")
    times = alt_df["Time"]
    ap = np.full(len(t_p), np.nan); am = np.full(len(t_p), np.nan)
    for i, ts in enumerate(t_p):
        if pd.isna(ts): continue
        dt = (times - ts).dt.total_seconds().abs() / 60.0
        j = int(dt.idxmin())
        if float(dt.loc[j]) <= tol_min:
            if "ALT_TR40_m" in alt_df.columns:
                ap[i] = pd.to_numeric(alt_df.loc[j, "ALT_TR40_m"], errors="coerce")
            if "ALT_MPL_m" in alt_df.columns:
                am[i] = pd.to_numeric(alt_df.loc[j, "ALT_MPL_m"], errors="coerce")
    alt_proto, alt_mpl = ap, am

# ── Plots ───────────────────────────────────────────────────────────────────
n_panels = int(show_proto) + int(show_mpl) + int(show_alt_cmp and alt_b is not None)
if n_panels == 0:
    st.warning("Toggle at least one panel.")
    st.stop()

heights = []
if show_proto: heights.append(2.4)
if show_mpl:   heights.append(2.4)
if show_alt_cmp and alt_b is not None: heights.append(1.6)
fig, axes = plt.subplots(n_panels, 1, figsize=(13, sum(heights)),
                         dpi=110, gridspec_kw={"height_ratios": heights})
if n_panels == 1:
    axes = [axes]
ax_iter = iter(axes)

x = np.arange(len(t_p))
extent = [0, len(t_p) - 1, float(np.nanmin(R_p)), float(np.nanmax(R_p))]
stroke = [mpe.withStroke(linewidth=3.0, foreground="black")]

if show_proto:
    a = next(ax_iter)
    a.imshow(np.clip(Z_p, vmin, vmax), aspect="auto", origin="lower",
             extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
             interpolation="nearest")
    if show_alt_overlay and alt_proto is not None and np.isfinite(alt_proto).any():
        a.plot(x, alt_proto, color="white", lw=1.6, path_effects=stroke)
    a.set_ylim(0, rmax); a.set_ylabel("Range (m)"); a.set_title("LiDAR Prototype")

if show_mpl:
    a = next(ax_iter)
    a.imshow(np.clip(Z_mpl_on_p, vmin, vmax), aspect="auto", origin="lower",
             extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
             interpolation="nearest")
    if show_alt_overlay and alt_mpl is not None and np.isfinite(alt_mpl).any():
        a.plot(x, alt_mpl, color="white", lw=1.6, path_effects=stroke)
    a.set_ylim(0, rmax); a.set_ylabel("Range (m)"); a.set_title("Mini MPL")

if show_alt_cmp and alt_b is not None:
    a = next(ax_iter)
    if alt_proto is not None:
        a.plot(x, alt_proto, color="#FF5632", lw=1.8, label="Prototype ALT (ALT_TR40_m)", zorder=2)
    if alt_mpl is not None:
        a.plot(x, alt_mpl, color="#35375B", lw=1.7, ls="--", label="MPL ALT", zorder=3)
    a.set_ylabel("Height (m)"); a.set_title("MPL ALT vs Prototype ALT")
    a.legend(fontsize=8, loc="best"); a.grid(True, alpha=0.3)

# x-tick labels = HH:MM, every Nth
step = max(1, len(t_p) // 14)
xticks = list(range(0, len(t_p), step))
xlabs  = [pd.Timestamp(t_p[i]).strftime("%m-%d %H:%M") if pd.notna(t_p[i]) else "" for i in xticks]
for a in axes:
    a.set_xticks(xticks); a.set_xticklabels(xlabs, rotation=40, ha="right", fontsize=8)
    a.grid(True, alpha=0.25)
axes[-1].set_xlabel("Time")
fig.tight_layout()
st.pyplot(fig)

# Download PNG
png_buf = io.BytesIO()
fig.savefig(png_buf, dpi=160, bbox_inches="tight")
st.download_button("⬇ Download PNG", data=png_buf.getvalue(),
                   file_name="RTI_visualization.png", mime="image/png")
