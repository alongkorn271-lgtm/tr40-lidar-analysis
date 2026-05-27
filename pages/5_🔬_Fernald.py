"""Step 5 — Fernald / Klett aerosol inversion (β_aer, α_aer, AOD)."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fernald_engine import compute_fernald_from_nrb_df  # noqa: E402


st.set_page_config(page_title="Step 5 · Fernald", page_icon="🔬", layout="wide")
st.title("🔬 Step 5 · Fernald / Klett Inversion")
st.caption("Retrieve aerosol backscatter β_aer, extinction α_aer, and AOD from the NRB profile.")


# ── Input ───────────────────────────────────────────────────────────────────
def _pick(label, sess_key):
    if sess_key in st.session_state:
        use = st.checkbox(f"Use `{st.session_state[sess_key + '_name']}` from session",
                          value=True, key=f"{sess_key}_use")
        if use:
            return st.session_state[sess_key]
    up = st.file_uploader(label, type=["xlsx"], key=f"{sess_key}_up")
    return up.getvalue() if up else None


with st.container(border=True):
    st.subheader("Input")
    nrb_b = _pick("NRB workbook (Step 2 output)", "step2_xlsx_bytes")
    sheet = st.text_input("NRB sheet name", value="NRB profile")

# ── Parameters ──────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Inversion parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        method = st.selectbox("Method", ["Fernald 1984", "Klett 1981"], index=0)
    with c2:
        r_ref_m = st.number_input("R_ref (m) — clean-air range",
                                   min_value=1000.0, value=5000.0, step=100.0,
                                   help="Where atmosphere is assumed purely molecular.")
    with c3:
        sa_sr = st.number_input("S_a (sr) — aerosol lidar ratio",
                                 min_value=10.0, max_value=120.0, value=50.0, step=5.0,
                                 help="Typical 532 nm: urban 50–80, marine ~25, smoke 70–100.")

    atm_mode = st.radio("T / P source",
                        ["US Standard Atmosphere (recommended)", "Manual (uniform T, P)"],
                        horizontal=True)
    T_sc = P_sc = None
    if atm_mode.startswith("Manual"):
        c1, c2 = st.columns(2)
        with c1: T_sc = st.number_input("T (K)", value=288.15, step=0.5)
        with c2: P_sc = st.number_input("P (Pa)", value=101325.0, step=100.0)

run = st.button("▶ Run inversion", type="primary", disabled=nrb_b is None)

# ── Run ─────────────────────────────────────────────────────────────────────
if run and nrb_b:
    try:
        df_nrb = pd.read_excel(io.BytesIO(nrb_b), sheet_name=sheet)
    except Exception as e:
        st.error(f"Cannot read sheet `{sheet}`: {e}")
        st.stop()
    df_nrb = df_nrb.rename(columns={df_nrb.columns[0]: "Range(m)"})

    with st.spinner("Running Fernald inversion …"):
        try:
            res = compute_fernald_from_nrb_df(
                df_nrb,
                R_ref_m=float(r_ref_m),
                lidar_ratio_aer=float(sa_sr),
                method=("fernald" if method.startswith("Fernald") else "klett"),
                T_K_scalar=T_sc, P_Pa_scalar=P_sc,
            )
        except Exception as e:
            st.error(f"Inversion failed: {e}")
            st.stop()

    # Write to in-memory Excel
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        res["beta_aer"].to_excel(xw, index=False, sheet_name="Fernald_beta_aer")
        res["alpha_aer"].to_excel(xw, index=False, sheet_name="Fernald_alpha_aer")
        res["AOD"].to_excel(xw, index=False, sheet_name="Fernald_AOD")

    st.session_state["step5_xlsx_bytes"] = buf.getvalue()
    st.session_state["step5_xlsx_name"]  = "Fernald-output.xlsx"
    st.session_state["step5_res"] = res
    st.success("✅ Done")

# ── Result ──────────────────────────────────────────────────────────────────
if "step5_res" in st.session_state:
    res = st.session_state["step5_res"]
    df_beta = res["beta_aer"]
    df_aod = res["AOD"]

    meta = {"R_ref_m", "lidar_ratio_aer_sr", "method"}
    ts_cols = [c for c in df_aod.columns if c not in meta]
    aod_vals = df_aod[ts_cols].to_numpy(float).flatten()

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean AOD", f"{np.nanmean(aod_vals):.3f}")
    c2.metric("Max AOD", f"{np.nanmax(aod_vals):.3f}" if np.isfinite(aod_vals).any() else "—")
    c3.metric("Profiles", len(ts_cols))

    st.download_button(
        "⬇ Download Fernald Excel",
        data=st.session_state["step5_xlsx_bytes"],
        file_name=st.session_state["step5_xlsx_name"],
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

    plot_mode = st.radio("Plot",
                          ["β_aer profile (select time)", "AOD time series"],
                          horizontal=True)

    if plot_mode.startswith("β_aer"):
        ts_labels = [str(c) for c in ts_cols]
        choice = st.selectbox("Profile time", ts_labels, index=0)
        R = df_beta["Range(m)"].to_numpy(float)
        y = df_beta[ts_cols[ts_labels.index(choice)]].to_numpy(float)

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=110)
        m = np.isfinite(R) & np.isfinite(y)
        ax.plot(R[m], y[m], color="#FF5632", lw=1.6)
        ax.set_xlabel("Range (m)"); ax.set_ylabel("β_aer (m⁻¹ sr⁻¹)")
        ax.set_title(f"β_aer profile · {choice}")
        ax.grid(True, alpha=0.3); fig.tight_layout()
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=110)
        x = np.arange(len(aod_vals))
        ax.plot(x, aod_vals, color="#FF5632", marker="o", ms=4, lw=1.6)
        ax.set_xticks(x[:: max(1, len(x) // 12)])
        ax.set_xticklabels(
            [str(ts_cols[i])[:16] for i in range(0, len(x), max(1, len(x) // 12))],
            rotation=45, ha="right", fontsize=8,
        )
        ax.set_xlabel("Time"); ax.set_ylabel("AOD (532 nm)")
        ax.set_title("Aerosol Optical Depth time series")
        ax.grid(True, alpha=0.3); fig.tight_layout()
        st.pyplot(fig)
