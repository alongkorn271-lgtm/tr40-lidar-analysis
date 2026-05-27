"""Step 2 — build daily NRB profile workbook from TR40 .dat files."""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from nrb_engine import build_daily_profile_from_folder, read_tr40_dat_ascii  # noqa: E402

try:
    from overlap import get_overlap as _get_overlap, OverlapHardware  # noqa: E402
    _HAS_OV = True
except Exception:
    _HAS_OV = False

try:
    from afterpulse import get_afterpulse as _get_afterpulse  # noqa: E402
    _HAS_AP = True
except Exception:
    _HAS_AP = False


st.set_page_config(page_title="Step 2 · NRB Profile", page_icon="📈", layout="wide")
st.title("📈 Step 2 · NRB Profile Builder")
st.caption("Convert TR40 .dat files (raw analog + photon) into a daily NRB profile workbook.")

# ── Input files ─────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Input")
    uploads = st.file_uploader(
        "TR40 .dat files (drag many — one per profile time)",
        type=["dat", "DAT"], accept_multiple_files=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1: date_text = st.text_input("Date (YYYY-MM-DD)", value="")
    with c2: start_time = st.text_input("Start time HH:MM (blank=all)", value="")
    with c3: pattern = st.text_input("Pattern", value="*.dat")

# ── Parameters ──────────────────────────────────────────────────────────────
with st.expander("⚙ Parameters (defaults are TR40-tuned)", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        dr_m = st.number_input("dr_m (bin spacing)", value=3.75, step=0.05)
        dead_time_ns = st.number_input("dead_time_ns", value=3.06, step=0.05)
    with c2:
        bg_mode = st.selectbox("bg_mode", ["pretrigger", "fixed", "far_range"], index=0)
        pretrigger_bins = st.number_input("pretrigger_bins", value=1024, step=1)
    with c3:
        sig_start_m = st.number_input("sig_start_m", value=0.0, step=50.0)
        sig_end_m = st.number_input("sig_end_m", value=15000.0, step=100.0)
    with c4:
        energy_mj = st.number_input("energy_mJ", value=25.0, step=0.5)
        pretrigger_trim_bins = st.number_input("pretrigger_trim_bins", value=24, step=1)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        bg_start_m = st.number_input("bg_start_m (fixed mode)", value=0.0, step=50.0)
    with c2:
        bg_end_m = st.number_input("bg_end_m (fixed mode)", value=3750.0, step=50.0)
    with c3:
        photon_only = st.checkbox("Skip glue (photon only)", value=False)
    with c4:
        auto_blend = st.checkbox("Auto blend r1/r2", value=True)

with st.expander("⚙ Overlap correction", expanded=False):
    ov_mode = st.selectbox("Overlap mode", ["Disabled", "Analytical (NARIT)", "Upload file"], index=0)
    ov_omin = st.number_input("O_min (mask below)", value=0.10, step=0.05)
    ov_file = None
    if ov_mode == "Upload file":
        ov_file = st.file_uploader("Overlap file (CSV/Excel: range_m, O)", type=["csv", "xlsx"])

with st.expander("⚙ Afterpulse correction", expanded=False):
    ap_mode = st.selectbox("Afterpulse mode", ["Disabled", "Upload file"], index=0)
    ap_file = None
    if ap_mode == "Upload file":
        ap_file = st.file_uploader("Afterpulse file (.dat / CSV / xlsx)", type=["dat", "csv", "xlsx"])

run = st.button("▶ Build NRB profile", type="primary", disabled=not uploads)

# ── Run ─────────────────────────────────────────────────────────────────────
if run and uploads:
    log = st.empty()
    progress = st.progress(0.0, text="Starting…")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for u in uploads:
            (td_path / u.name).write_bytes(u.getvalue())

        # Build O(R), A(R) using first file's range axis
        first_file = next(td_path.glob(pattern or "*.dat"))
        ov_arr = None; ap_arr = None
        if ov_mode != "Disabled" and _HAS_OV:
            try:
                df_raw = read_tr40_dat_ascii(first_file, dr_m=dr_m,
                                             pretrigger_bins=int(pretrigger_bins))
                r_axis = df_raw["range_m"].to_numpy(float)
                if ov_mode == "Analytical (NARIT)":
                    ov_arr = _get_overlap(r_axis, OverlapHardware())
                elif ov_file is not None:
                    tmp_ov = td_path / ("_ov" + Path(ov_file.name).suffix)
                    tmp_ov.write_bytes(ov_file.getvalue())
                    ov_arr = _get_overlap(r_axis, str(tmp_ov))
            except Exception as e:
                st.warning(f"Overlap setup: {e}")

        if ap_mode != "Disabled" and _HAS_AP and ap_file is not None:
            try:
                df_raw = read_tr40_dat_ascii(first_file, dr_m=dr_m,
                                             pretrigger_bins=int(pretrigger_bins))
                r_axis = df_raw["range_m"].to_numpy(float)
                tmp_ap = td_path / ("_ap" + Path(ap_file.name).suffix)
                tmp_ap.write_bytes(ap_file.getvalue())
                ap_arr = _get_afterpulse(r_axis, str(tmp_ap))
            except Exception as e:
                st.warning(f"Afterpulse setup: {e}")

        out_path = td_path / f"NRB-{date_text or 'output'}.xlsx"

        def _log(msg): log.text(msg)
        def _prog(v): progress.progress(min(1.0, v / 100.0), text=f"{v:.0f}%")

        try:
            profile_df, qc_df, params_df = build_daily_profile_from_folder(
                td_path,
                date_str=date_text, pattern=pattern or "*.dat",
                out_path=out_path, start_time=start_time,
                dr_m=float(dr_m), dead_time_ns=float(dead_time_ns),
                bg_mode=bg_mode, bg_start_m=float(bg_start_m), bg_end_m=float(bg_end_m),
                pretrigger_bins=int(pretrigger_bins),
                sig_start_m=float(sig_start_m), sig_end_m=float(sig_end_m),
                pretrigger_trim_bins=int(pretrigger_trim_bins),
                energy_mj=float(energy_mj),
                auto_blend=bool(auto_blend),
                gluing_mode="photon_only" if photon_only else "auto",
                overlap_O_R=ov_arr, overlap_O_min=float(ov_omin),
                afterpulse_A_R=ap_arr,
                strict=False, logger=_log, progress_cb=_prog,
            )
            st.session_state["step2_xlsx_bytes"] = out_path.read_bytes()
            st.session_state["step2_xlsx_name"] = out_path.name
            progress.empty()
            st.success(f"✅ Built  ·  {len(profile_df.columns) - 1} profiles")
        except Exception as e:
            progress.empty()
            st.error(f"Failed: {e}")
            st.stop()

# ── Result ──────────────────────────────────────────────────────────────────
if "step2_xlsx_bytes" in st.session_state:
    buf = io.BytesIO(st.session_state["step2_xlsx_bytes"])
    df = pd.read_excel(buf, sheet_name="NRB profile")

    st.markdown("### Result")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.metric("Profiles", df.shape[1] - 1)
        st.metric("Range bins", df.shape[0])
        st.download_button(
            "⬇ Download NRB Excel",
            data=st.session_state["step2_xlsx_bytes"],
            file_name=st.session_state["step2_xlsx_name"],
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
        st.info("➡️ Result is in session state — Step 3 / 5 will pick it up.")

    with c2:
        import matplotlib.pyplot as plt
        # RTI heatmap preview
        R = df.iloc[:, 0].to_numpy(float)
        Z = df.iloc[:, 1:].to_numpy(float)
        ts_cols = list(df.columns[1:])
        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=110)
        im = ax.imshow(Z, aspect="auto", origin="lower",
                       extent=[0, Z.shape[1] - 1, R.min(), min(R.max(), 6000)],
                       cmap="jet", vmin=0, vmax=1)
        ax.set_ylim(R.min(), min(R.max(), 6000))
        ax.set_xlabel("Profile index"); ax.set_ylabel("Range (m)")
        ax.set_title("NRB (0–1) preview")
        fig.colorbar(im, ax=ax, label="NRB")
        fig.tight_layout()
        st.pyplot(fig)
