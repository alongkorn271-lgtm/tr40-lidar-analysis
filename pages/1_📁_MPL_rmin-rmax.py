"""Step 1 — build the rmin-rmax search-window table from Mini-MPL CSV files."""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# project root on sys.path so we can import the legacy helpers
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from main import (  # noqa: E402
    _s1_collect_actual_files, _s1_read_copol, _s1_read_pbl_km,
)

st.set_page_config(page_title="Step 1 · MPL rmin-rmax", page_icon="📁", layout="wide")
st.title("📁 Step 1 · MPL rmin-rmax")
st.caption("Build the rmin / rmax search-window table from Mini-MPL CSV files (one CSV per profile, filename contains the YYYYMMDDHHMM timestamp).")

# ── Inputs ──────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Input")
    uploads = st.file_uploader(
        "Mini-MPL CSV files (drag multiple)", type=["csv"],
        accept_multiple_files=True,
        help="Filenames must contain the timestamp like `MPL_5038_202603250006.csv`.",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        date_text = st.text_input("Date (YYYY-MM-DD, blank = auto)", value="")
    with c2:
        start_time = st.text_input("Start time (HH:MM, blank = all)", value="")
    with c3:
        margin = st.number_input("Window margin (m) — rmin/rmax = PBL ± margin",
                                  min_value=50.0, max_value=3000.0, value=300.0, step=25.0)

run_clicked = st.button("▶ Build rmin-rmax table", type="primary",
                        disabled=not uploads)

# ── Run ─────────────────────────────────────────────────────────────────────
if run_clicked and uploads:
    log_box = st.container()

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for u in uploads:
            (td_path / u.name).write_bytes(u.getvalue())

        try:
            target_date, ordered = _s1_collect_actual_files(td_path, date_text, start_time)
        except Exception as e:
            st.error(f"Failed to collect files: {e}")
            st.stop()

        log_box.success(f"Date detected: **{target_date.strftime('%Y-%m-%d')}** · {len(ordered)} profiles")

        rows = []
        copol_cols: dict = {}
        range_master = None
        copol_n = 498
        progress = st.progress(0.0, text="Reading MPL files…")

        for i, (ts, fp) in enumerate(ordered, 1):
            try:
                rng_m, cop = _s1_read_copol(fp, 0, copol_n)
                if range_master is None and np.isfinite(rng_m).any():
                    range_master = rng_m
                copol_cols[ts] = cop
            except Exception:
                copol_cols[ts] = np.full((copol_n,), np.nan)

            pbl_km = _s1_read_pbl_km(fp)
            if np.isfinite(pbl_km):
                pbl_m = float(pbl_km) * 1000.0
                rmin = max(0.0, pbl_m - margin); rmax = pbl_m + margin; status = "ok"
            else:
                pbl_m = rmin = rmax = np.nan; status = "read_fail"

            rows.append({
                "Time": ts, "PBL from MPL (m)": pbl_m,
                "rmin": rmin, "rmax": rmax,
                "Status": status, "Source file": fp.name,
            })
            progress.progress(i / len(ordered), text=f"[{i}/{len(ordered)}] {ts.strftime('%H:%M')}")

        df = pd.DataFrame(rows)
        if range_master is None:
            range_master = np.full((copol_n,), np.nan)
        copol_df = pd.DataFrame({"Range(m)": range_master})
        for ts in [r["Time"] for r in rows]:
            copol_df[ts.strftime("%H:%M")] = copol_cols.get(ts, np.full((copol_n,), np.nan))

        # Write to in-memory Excel
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            df.to_excel(xw, index=False, sheet_name="rmin-rmax")
            copol_df.to_excel(xw, index=False, sheet_name="copol_nrb_norm")
        out_name = f"rmin-rmax-{target_date.strftime('%Y-%m-%d')}.xlsx"

        st.session_state["step1_xlsx_bytes"] = buf.getvalue()
        st.session_state["step1_xlsx_name"] = out_name

        progress.empty()
        st.success(f"✅ Built  ·  {len(df)} profiles  ·  "
                   f"{int(df['Status'].eq('ok').sum())} valid PBL reads")

# ── Show results (from session state, persists across reruns) ──────────────
if "step1_xlsx_bytes" in st.session_state:
    buf = io.BytesIO(st.session_state["step1_xlsx_bytes"])
    df = pd.read_excel(buf, sheet_name="rmin-rmax")
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

    st.markdown("### Result")
    c1, c2 = st.columns([1, 1.1])
    with c1:
        st.dataframe(df, hide_index=True, use_container_width=True, height=320)
        st.download_button(
            "⬇ Download Excel",
            data=st.session_state["step1_xlsx_bytes"],
            file_name=st.session_state["step1_xlsx_name"],
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )

    with c2:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=110)
        m = df["PBL from MPL (m)"].notna()
        if m.any():
            ax.fill_between(df["Time"][m], df["rmin"][m], df["rmax"][m],
                            color="#FFCBB6", alpha=0.65, label="rmin–rmax window")
            ax.plot(df["Time"][m], df["PBL from MPL (m)"][m],
                    color="#35375B", lw=1.8, marker="o", ms=4, label="PBL (MPL)")
        ax.set_xlabel("Time"); ax.set_ylabel("Height (m)")
        ax.set_title("MPL PBL  &  search window")
        ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=9)
        fig.autofmt_xdate(); fig.tight_layout()
        st.pyplot(fig)

    st.info("➡️ Result is held in session state — Step 3 can pick it up automatically.")
