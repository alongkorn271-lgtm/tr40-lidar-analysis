"""TR40 LiDAR Analysis Suite — Streamlit web front-end.

Run locally:
    streamlit run streamlit_app.py

Streamlit auto-discovers files in pages/ for the multi-page navigation
(Step 1 → Step 5).  This file is the entry / Home page only.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="TR40 LiDAR Analysis Suite",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#35375B"
ORANGE = "#FF5632"

st.markdown(
    f"""
    <h1 style='color:{NAVY}; margin-bottom:4px'>TR40 LiDAR Analysis Suite</h1>
    <p style='color:{ORANGE}; font-size:14px; margin-top:0'>
      Elastic Mie LiDAR 532 nm &nbsp;·&nbsp; NARIT Chiang Mai &nbsp;·&nbsp; Processing pipeline
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

col_a, col_b = st.columns([1.25, 1])
with col_a:
    st.markdown(
        """
        ### Five-step workflow

        | # | Page | Input | Output |
        |---|------|-------|--------|
        | **1** | MPL rmin-rmax | Mini-MPL CSV files | `rmin-rmax-*.xlsx` |
        | **2** | NRB Profile | TR40 `.dat` files | `NRB-*.xlsx` |
        | **3** | ALT Detection | NRB + rmin-rmax | `ALT_NRB-*.xlsx` |
        | **4** | RTI Visualizer | Prototype + MPL + ALT | RTI / overlay plots |
        | **5** | Fernald Inversion | NRB profile | `Fernald-*.xlsx` (β_aer, AOD) |

        Use the **sidebar** on the left to jump between steps.
        Output of each step is held in browser session state so the next step
        can pick it up automatically — no need to re-upload between steps.
        """
    )

with col_b:
    try:
        st.image("LiDAR_pipeline_flowchart.png",
                 caption="Processing pipeline",
                 use_container_width=True)
    except Exception:
        st.info("`LiDAR_pipeline_flowchart.png` not found — run the desktop GUI once or regenerate the diagram.")

st.markdown("---")

with st.expander("Hardware & science notes", expanded=False):
    st.markdown(
        """
        **Hardware (NARIT TR40):**
        - Telescope: Celestron EdgeHD 800 (D = 203 mm, f = 2032 mm)
        - Laser: Quantel VIRON, 532 nm
        - Detector: Licel PM-HV + Hamamatsu R9880U PMT
        - Geometry: biaxial, d⊥ = 156.0 mm  ·  bin spacing 3.75 m (25 ns)

        **NRB definition (SigmaMPL convention):**
        `NRB = (Raw·DT − Afterpulse − BG) / (Overlap · Energy) × R²`, then ÷ max.

        **ALT detection:** FFT low-pass (cutoff `fc`) → HWCT Haar step (half-window
        `tol_m`) → most-negative W peak inside the search window.

        **Fernald inversion** is self-calibrating via Rayleigh normalisation
        `β_total(R_ref) = β_mol(R_ref)`, so the unknown lidar constant cancels —
        the NRB-engine `÷ energy` and `÷ max` normalisations carry through.

        References: Fernald (1984), Klett (1981), Brooks (2003), Weitkamp (2005).
        """
    )

with st.expander("Session state", expanded=False):
    keys = sorted(k for k in st.session_state.keys() if not k.startswith("_"))
    if not keys:
        st.write("_(nothing stored yet — run a step to populate)_")
    else:
        for k in keys:
            v = st.session_state[k]
            preview = (
                f"`bytes ({len(v):,})`" if isinstance(v, (bytes, bytearray))
                else f"`{type(v).__name__}` — {str(v)[:80]}"
            )
            st.write(f"• **{k}**: {preview}")
    if st.button("🗑️ Clear all session state"):
        for k in list(st.session_state.keys()):
            if not k.startswith("_"):
                del st.session_state[k]
        st.rerun()
