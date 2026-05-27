"""Step 3 — Aerosol Layer Top (ALT) detection via FFT low-pass + HWCT."""
from __future__ import annotations

import io
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

st.set_page_config(page_title="Step 3 · ALT Detection", page_icon="📊", layout="wide")
st.title("📊 Step 3 · ALT Detection")
st.caption("FFT low-pass → HWCT Haar step → most-negative W peak = aerosol layer top.")


def _pick_file(label: str, sess_key: str, types=("xlsx", "xls")):
    """Use Step-N output from session state OR let user upload."""
    has_prev = sess_key in st.session_state
    src = st.radio(
        f"{label} source",
        ["From previous step (session)" if has_prev else "Upload",
         "Upload"],
        horizontal=True, key=f"{sess_key}_src",
        disabled=not has_prev,
    )
    if has_prev and src.startswith("From"):
        st.success(f"Using `{st.session_state[sess_key + '_name']}` from session")
        return st.session_state[sess_key], st.session_state[sess_key + "_name"]
    up = st.file_uploader(f"{label} (.xlsx)", type=list(types), key=f"{sess_key}_up")
    if up is None:
        return None, None
    return up.getvalue(), up.name


# ── Inputs ──────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Inputs")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**NRB workbook** (from Step 2)")
        nrb_bytes, nrb_name = _pick_file("NRB", "step2_xlsx_bytes")
    with c2:
        st.markdown("**rmin-rmax workbook** (from Step 1)")
        rr_bytes, rr_name = _pick_file("rmin-rmax", "step1_xlsx_bytes")

# ── Parameters ──────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Options")
    c1, c2, c3 = st.columns(3)
    with c1:
        det_mode = st.selectbox("Detection mode", ["dual", "mpl_guided", "nrb_profile"], index=0)
    with c2:
        prmin = st.number_input("profile rmin (m)", value=0.0, step=50.0)
        prmax = st.number_input("profile rmax (m)", value=2500.0, step=50.0)
    with c3:
        tol_m_val = st.number_input("HWCT tol_m (m)", value=250.0, step=10.0,
                                     help="Half-window of Haar step. 250 m fits the broad TR40 BL top.")
        fc = st.number_input("FFT cutoff fc (cyc/m)", value=0.015, step=0.001, format="%.4f")

with st.expander("Track A — algorithm improvements (optional)", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        adaptive_window = st.checkbox("Adaptive window (diurnal)", value=False)
        peak_anchored = st.checkbox("Peak-anchored search", value=False)
    with c2:
        lowest_edge = st.checkbox("Prefer lowest stable edge", value=False)
        lowest_thr = st.number_input("Lowest-edge thr factor", value=0.30, step=0.05)
    with c3:
        ts_win = st.number_input("Temporal smoothing window (slots, 0=off)", value=0, step=1)
        cs_thr = st.number_input("Cloud screen threshold (0=off)", value=0.0, step=0.05)

with st.expander("Track 3 — cloud detection (extra Cloud_results sheet)", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        cd_enable = st.checkbox("Detect cloud layers", value=False)
        cd_thr = st.number_input("Cloud NRB threshold", value=0.30, step=0.05)
    with c2:
        cd_thick = st.number_input("Min thickness (m)", value=100.0, step=10.0)
    with c3:
        cd_max = st.number_input("Max layers per profile", value=3, step=1)

ready = nrb_bytes is not None and rr_bytes is not None
run = st.button("▶ Run ALT detection", type="primary", disabled=not ready)

# ── Run ─────────────────────────────────────────────────────────────────────
if run:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # Engine needs ONE Excel file with both the NRB profile sheet AND the rmin-rmax sheet.
        # Build a combined workbook from the two uploads.
        nrb_df = pd.read_excel(io.BytesIO(nrb_bytes), sheet_name="NRB profile")
        rr_df  = pd.read_excel(io.BytesIO(rr_bytes), sheet_name="rmin-rmax")

        combined = td_path / "_combined.xlsx"
        with pd.ExcelWriter(combined, engine="openpyxl") as xw:
            nrb_df.to_excel(xw, index=False, sheet_name="NRB profile")
            rr_df.to_excel(xw, index=False, sheet_name="rmin-rmax")

        out = td_path / "ALT_output.xlsx"
        cmd = [
            sys.executable, str(_ROOT / "pbl_engine.py"),
            "--nrb", str(combined), "--sheet", "NRB profile",
            "--rminrmax_sheet", "rmin-rmax", "--out", str(out),
            "--detection_mode", det_mode,
            "--profile_rmin", str(prmin), "--profile_rmax", str(prmax),
            "--tol_m", str(tol_m_val), "--fc", str(fc),
        ]
        if adaptive_window: cmd.append("--adaptive_window")
        if peak_anchored:   cmd.append("--peak_anchored")
        if lowest_edge:
            cmd += ["--lowest_edge", "--lowest_edge_thr_factor", str(lowest_thr)]
        if float(cs_thr) > 0:
            cmd += ["--cloud_screen_threshold", str(cs_thr)]
        if int(ts_win) > 0:
            cmd += ["--temporal_smooth_window", str(ts_win)]
        if cd_enable:
            cmd += ["--cloud_detect", "--cloud_threshold", str(cd_thr),
                    "--cloud_min_thickness_m", str(cd_thick),
                    "--cloud_max_layers", str(cd_max)]

        with st.spinner("Running pbl_engine.py …"):
            r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            st.error("Engine failed — see log below")
            st.code(r.stderr or r.stdout, language="text")
            st.stop()

        st.session_state["step3_xlsx_bytes"] = out.read_bytes()
        st.session_state["step3_xlsx_name"]  = "ALT_NRB-output.xlsx"
        st.success("✅ ALT detection complete")

# ── Result ──────────────────────────────────────────────────────────────────
if "step3_xlsx_bytes" in st.session_state:
    buf = io.BytesIO(st.session_state["step3_xlsx_bytes"])
    df = pd.read_excel(buf, sheet_name="ALT_results")
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

    st.markdown("### Result")
    c1, c2, c3 = st.columns(3)
    for col, lab in [(c1, "ALT_TR40_m"), (c2, "ALT_guided_m"), (c3, "ALT_profile_m")]:
        if lab in df.columns:
            s = pd.to_numeric(df[lab], errors="coerce")
            col.metric(lab, f"{s.notna().sum()}/{len(s)}",
                       help=f"mean = {s.mean():.0f} m" if s.notna().any() else "no valid values")

    st.download_button(
        "⬇ Download ALT_results Excel",
        data=st.session_state["step3_xlsx_bytes"],
        file_name=st.session_state["step3_xlsx_name"],
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4), dpi=110)
    if "ALT_MPL_m" in df.columns:
        s = pd.to_numeric(df["ALT_MPL_m"], errors="coerce")
        ax.plot(df["Time"], s, "--", color="#35375B", lw=1.6, label="MPL ALT", zorder=3)
    for col, color, ls in [
        ("ALT_TR40_m", "#FF5632", "-"),
        ("ALT_guided_m", "#00BE7C", ":"),
        ("ALT_profile_m", "#5B6CFF", ":"),
    ]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            ax.plot(df["Time"], s, ls, color=color, lw=1.6, label=col)
    ax.set_xlabel("Time"); ax.set_ylabel("Height (m)")
    ax.set_title("ALT vs MPL"); ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=9)
    fig.autofmt_xdate(); fig.tight_layout()
    st.pyplot(fig)

    with st.expander("Per-profile table", expanded=False):
        show_cols = ["Time", "ALT_TR40_m", "ALT_guided_m", "ALT_profile_m",
                     "ALT_MPL_m", "ALT_profile_status", "window_source"]
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(df[show_cols], hide_index=True, use_container_width=True, height=320)
