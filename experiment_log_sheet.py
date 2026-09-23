"""Build the DCT-04 style experiment log of a raw folder straight from the file headers.

A measurement day often mixes several settings (bin width, number of bins, shots). This
reads every Licel raw file, groups the files into setting blocks, and writes a sheet in
the layout of the LiDAR-DCT-04 case sheets: one row per file with what the recorder
actually used, plus the columns the operator fills in by hand (QS, laser energy, roof,
weather, humidity, person, note).

    python experiment_log_sheet.py "<raw folder>" [-o out.xlsx] [--case "C01-03"] [--qc]

``--qc`` adds the Raw QC measurements (analog peak, PMT current, background, usability).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

import licel_binary_reader as lbr
import nrb_engine as ne

FONT = "Leelawadee UI"
THIN = Side(style="thin", color="808080")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)
NOWRAP = Alignment(horizontal="left", vertical="center", wrap_text=False)
HEAD_FILL = PatternFill("solid", fgColor="1F3864")
SUB_FILL = PatternFill("solid", fgColor="D9E1F2")
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")
TEST_FILL = PatternFill("solid", fgColor="F2F2F2")
BLOCK_FILLS = ["C6EFCE", "DDEBF7", "FCE4D6", "E4DFEC", "FFF2CC"]   # one tint per setting block
MIN_SHOTS = 1500.0
LASER_HZ = 10.0     # header line 2 field 2 on this system


def read_folder(folder: Path) -> pd.DataFrame:
    """Header facts of every readable raw file, sorted by acquisition end."""
    rows: List[Dict] = []
    for f in sorted(Path(folder).iterdir()):
        if f.is_dir():
            continue
        try:
            rf = lbr.parse_raw_file(f)
        except Exception:
            continue
        by = {m.device: m for m in rf.datasets}
        a, p = by.get("BT0"), by.get("BC0")
        b = by.get("BT1")
        if a is None:
            continue
        bw = float(a.binwidth_m)
        pre = int(round(ne.PRETRIGGER_TRACE_DISTANCE_M / bw))
        sig = int(a.ndata) - pre
        ts = ne.parse_licel_raw_timestamp(f.name)
        start = pd.to_datetime(rf.start_time, format="%d/%m/%Y %H:%M:%S", errors="coerce")
        stop = pd.to_datetime(rf.stop_time, format="%d/%m/%Y %H:%M:%S", errors="coerce")
        rows.append(dict(
            file=f.name, end=pd.Timestamp(ts) if ts is not None else stop,
            start=start, stop=stop,
            window_min=(stop - start).total_seconds() / 60.0 if pd.notna(start) and pd.notna(stop) else np.nan,
            shots=int(rf.shots), laser_s=int(rf.shots) / LASER_HZ,
            hv_par=int(a.hv), hv_perp=int(b.hv) if b else np.nan,
            bin_m=bw, pretrigger=pre, signal_bins=sig, total_bins=int(a.ndata),
            altitude_km=sig * bw / 1000.0,
            disc_mv=float(p.input_range) if p else np.nan,
            range_mv=float(a.input_range) * 1000.0,
        ))
    D = pd.DataFrame(rows)
    if D.empty:
        return D
    D = D.sort_values("end").reset_index(drop=True)
    D["test"] = D["shots"] < MIN_SHOTS
    # a setting block = a run of files sharing the recorder configuration
    key = D[["hv_par", "hv_perp", "bin_m", "total_bins", "disc_mv", "range_mv"]].astype(str).agg("|".join, axis=1)
    D["block"] = (key != key.shift()).cumsum()
    # a block is only a real change of setting if some file in it is a measurement
    return D


def block_table(D: pd.DataFrame) -> pd.DataFrame:
    """One row per setting block: what was set and when it ran."""
    out = []
    for blk, g in D.groupby("block"):
        real = g[~g["test"]]
        shots = sorted(set(real["shots"].round(-2).astype(int))) if len(real) else []
        out.append(dict(
            block=int(blk), files=len(g), measurements=len(real), tests=int(g["test"].sum()),
            first=g["end"].min(), last=g["end"].max(),
            bin_m=g["bin_m"].iloc[0], pretrigger=int(g["pretrigger"].iloc[0]),
            signal_bins=int(g["signal_bins"].iloc[0]), total_bins=int(g["total_bins"].iloc[0]),
            altitude_km=g["altitude_km"].iloc[0], hv=int(g["hv_par"].iloc[0]),
            disc_mv=g["disc_mv"].iloc[0], range_mv=g["range_mv"].iloc[0],
            shots=", ".join(str(s) for s in shots) if shots else "—",
            laser_s=", ".join(f"{s / LASER_HZ:.0f}" for s in shots) if shots else "—",
            cadence_min=round(float(real["end"].diff().dt.total_seconds().median() / 60), 0) if len(real) > 2 else np.nan,
        ))
    return pd.DataFrame(out)


def block_metrics(D: pd.DataFrame) -> pd.DataFrame:
    """Per setting block, the Raw QC numbers that say how the block went (needs --qc)."""
    out = []
    for blk, g in D[~D["test"]].groupby("block"):
        night = g[g["night"].fillna(False).astype(bool)] if "night" in g else g.iloc[0:0]

        def med(frame, col):
            return float(frame[col].median()) if col in frame and len(frame[col].dropna()) else np.nan
        out.append(dict(block=int(blk), files=len(g), night_files=len(night),
                        par_peak=med(g, "par_peak_mv"), perp_peak=med(g, "perp_peak_mv"),
                        pmt_max=float(g["par_pmt_uA"].max()) if "par_pmt_uA" in g else np.nan,
                        night_bg_par=med(night, "par_bg_photon_mhz"), night_bg_perp=med(night, "perp_bg_photon_mhz"),
                        night_snr5_perp=med(night, "perp_snr_5km"), snr3_par=med(g, "par_snr_3km"),
                        par_top_km=med(g, "par_top_m") / 1000, perp_top_km=med(g, "perp_top_m") / 1000,
                        clouds=int((g["G1"] == "WARN").sum()) if "G1" in g else 0))
    return pd.DataFrame(out)


def _style(c, *, bold=False, size=9, fill=None, color="000000", align=CENTER, border=True, fmt=None):
    c.font = Font(name=FONT, size=size, bold=bold, color=color)
    # Thai text and ∥/⊥ in one cell: Excel drops the symbols unless they get their own run
    from raw_quality_sheet import _has_thai, _rich, SYMBOLS
    v = c.value
    if isinstance(v, str) and not v.startswith("=") and _has_thai(v) and any(x in v for x in SYMBOLS):
        c.value = _rich(v, bold=bold, size=size, color=color)
    c.alignment = align
    if fill is not None:
        c.fill = fill
    if border:
        c.border = BOX
    if fmt:
        c.number_format = fmt


COLS = [
    ("Time of experiment\n(สิ้นสุด)", "end", 15, "dd/mm hh:mm"),
    ("ช่วงบันทึก (header)", "_window", 20, None),
    ("นาที", "window_min", 6, "0"),
    ("ไฟล์", "file", 16, None),
    ("ชุดค่าตั้ง", "block", 8, "0"),
    ("Signal (QS)\nP&S channel", "_qs", 10, None),
    ("พลังงานเลเซอร์ (mJ)", "_energy", 11, None),
    ("Applied PMT (V)\nP&S channel", "hv_par", 11, "0"),
    ("ND filter (OD)\nช่อง ∥", "_nd", 10, None),
    ("Bin Range (m)", "bin_m", 10, "0.00"),
    ("Pretrigger (bin)", "pretrigger", 10, "0"),
    ("signal (bin)", "signal_bins", 9, "0"),
    ("Total (bin)", "total_bins", 9, "0"),
    ("Altitude (km)", "altitude_km", 10, "0.0"),
    ("laser time (s)", "laser_s", 10, "0"),
    ("Shot (pulse)", "shots", 10, "0"),
    ("Discriminator (mV)", "disc_mv", 11, "0.0000"),
    ("Input range (mV)", "range_mv", 10, "0"),
    ("Sliding roof", "_roof", 10, None),
    ("สภาพอากาศ / % ฝน", "_weather", 13, None),
    ("ความชื้น", "_humid", 9, None),
    ("ผู้รับผิดชอบ", "_person", 11, None),
    ("หมายเหตุ", "_note", 34, None),
]
MANUAL = {"_qs", "_energy", "_nd", "_roof", "_weather", "_humid", "_person", "_note"}
QC_COLS = [("∥ peak (mV)", "par_peak_mv", 10, "0"), ("⊥ peak (mV)", "perp_peak_mv", 10, "0"),
           ("∥ PMT (µA)", "par_pmt_uA", 10, "0"), ("⊥ PMT (µA)", "perp_pmt_uA", 10, "0"),
           ("∥ bg (MHz)", "par_bg_photon_mhz", 10, "0.00"), ("⊥ bg (MHz)", "perp_bg_photon_mhz", 10, "0.00"),
           ("∥ ถึง (km)", "_ptop", 9, "0.0"), ("⊥ ถึง (km)", "_stop", 9, "0.0"),
           ("NRB / PBL", "_use_nrb", 14, None), ("δ", "_use_delta", 14, None)]


def write_sheet(D: pd.DataFrame, out: Path, *, case: str = "", folder: str = "", qc: bool = False,
                block_notes: Optional[Dict[int, str]] = None, file_notes: Optional[Dict[str, str]] = None,
                findings: Optional[List[str]] = None) -> None:
    B = block_table(D)
    block_notes = block_notes or {}
    file_notes = file_notes or {}
    wb = Workbook(); ws = wb.active; ws.title = "Log"
    ws.sheet_view.showGridLines = False
    cols = COLS + (QC_COLS if qc else [])
    n_col = len(cols)

    def hdr(r, c, text, span=1, bold=True, fill=None):
        _style(ws.cell(row=r, column=c, value=text), bold=bold, fill=fill, align=LEFT)
        if span > 1:
            ws.merge_cells(start_row=r, start_column=c, end_row=r, end_column=c + span - 1)
            for k in range(c, c + span):
                ws.cell(row=r, column=k).border = BOX

    date_txt = f"{D['end'].min():%d-%b-%y}" if len(D) else ""
    hdr(1, 1, "Project :"); hdr(1, 2, "polarization system optimization", span=6, bold=False)
    hdr(1, n_col - 1, "Ref:"); hdr(1, n_col, "LiDAR-DCT-04")
    hdr(2, 1, "Testing Conditions :"); hdr(2, 2, case or "ดูชุดค่าตั้งด้านล่าง", span=6, bold=False)
    hdr(2, n_col - 1, "Date:"); hdr(2, n_col, date_txt)
    hdr(3, 1, "Raw folder :"); hdr(3, 2, folder, span=n_col - 2, bold=False)

    # ---- settings blocks -------------------------------------------------------------
    r = 5
    _style(ws.cell(row=r, column=1, value="ชุดค่าตั้งที่ใช้ในวันนี้ (แยกอัตโนมัติจาก header)"),
           bold=True, size=11, align=LEFT, border=False)
    r += 1
    bh = ["ชุด", "ช่วงเวลา", "ไฟล์วัด", "ไฟล์ทดสอบ", "Bin (m)", "Pretrigger", "signal", "Total",
          "Altitude (km)", "Shot", "laser time (s)", "ทุกๆ (นาที)", "HV (V)", "Disc (mV)", "Range (mV)",
          "ทดลองอะไร"]
    for j, h in enumerate(bh, start=1):
        _style(ws.cell(row=r, column=j, value=h), bold=True, fill=HEAD_FILL, color="FFFFFF")
    for i, row in B.iterrows():
        rr = r + 1 + i
        vals = [int(row["block"]), f"{row['first']:%H:%M} – {row['last']:%H:%M}", int(row["measurements"]),
                int(row["tests"]), row["bin_m"], row["pretrigger"], row["signal_bins"], row["total_bins"],
                round(row["altitude_km"], 1), row["shots"], row["laser_s"],
                None if not np.isfinite(row["cadence_min"]) else row["cadence_min"],
                row["hv"], row["disc_mv"], row["range_mv"]]
        fill = PatternFill("solid", fgColor=BLOCK_FILLS[(int(row["block"]) - 1) % len(BLOCK_FILLS)])
        vals.append(block_notes.get(int(row["block"]), ""))
        for j, v in enumerate(vals, start=1):
            _style(ws.cell(row=rr, column=j, value=v), fill=fill if j == 1 else None,
                   bold=(j == 1), align=LEFT if j == 16 else CENTER,
                   fmt="0.00" if j == 5 else ("0.0" if j == 9 else None))
        ws.merge_cells(start_row=rr, start_column=16, end_row=rr, end_column=16 + 6)
    r += len(B) + 2

    # ---- per-file rows ---------------------------------------------------------------
    for j, (label, key, w, fmt) in enumerate(cols, start=1):
        _style(ws.cell(row=r, column=j, value=label), bold=True,
               fill=INPUT_FILL if key in MANUAL else HEAD_FILL,
               color="000000" if key in MANUAL else "FFFFFF")
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.row_dimensions[r].height = 34
    first = r + 1
    for i, row in D.iterrows():
        rr = first + i
        blk_fill = PatternFill("solid", fgColor=BLOCK_FILLS[(int(row["block"]) - 1) % len(BLOCK_FILLS)])
        for j, (label, key, w, fmt) in enumerate(cols, start=1):
            if key == "_window":
                v = (f"{row['start']:%H:%M:%S} – {row['stop']:%H:%M:%S}"
                     if pd.notna(row["start"]) and pd.notna(row["stop"]) else "")
            elif key == "_note":
                hhmm = f"{row['end']:%H:%M}" if pd.notna(row["end"]) else ""
                v = "; ".join(x for x in [file_notes.get(hhmm, ""),
                                          "ไฟล์ทดสอบ (< 1500 shot)" if row["test"] else ""] if x)
            elif key in ("_ptop", "_stop", "_use_nrb", "_use_delta"):
                src = {"_ptop": "par_top_m", "_stop": "perp_top_m",
                       "_use_nrb": "use_nrb", "_use_delta": "use_delta"}[key]
                v = row.get(src)
                if key in ("_ptop", "_stop"):
                    v = None if v is None or not np.isfinite(v) or row["test"] else float(v) / 1000
                elif v is not None:
                    v = {"USABLE": "ใช้ได้", "LIMITED": "ใช้ได้มีข้อจำกัด", "UNUSABLE": "ใช้ไม่ได้"}.get(v, v)
            elif key in MANUAL:
                v = None
            else:
                v = row.get(key)
                if isinstance(v, (float, np.floating)) and not np.isfinite(v):
                    v = None
                elif isinstance(v, (np.integer,)):
                    v = int(v)
                elif isinstance(v, pd.Timestamp):
                    v = v.to_pydatetime()
            fill = (blk_fill if key == "block" else
                    INPUT_FILL if key in MANUAL else
                    TEST_FILL if row["test"] else None)
            _style(ws.cell(row=rr, column=j, value=v), fill=fill, fmt=fmt,
                   align=LEFT if key in ("file", "_note", "_weather") else CENTER,
                   bold=(key == "block"))
    last = first + len(D) - 1
    ws.freeze_panes = ws.cell(row=first, column=5)
    ws.auto_filter.ref = f"A{r}:{get_column_letter(n_col)}{last}"
    fr = last + 2
    for k, txt in enumerate(["Report By : ……………………………  Date : ………",
                             "Checked By : ……………………………  Date : ………",
                             "Approved By : ……………………………  Date : ………"]):
        _style(ws.cell(row=fr, column=1 + k * 6, value=txt), bold=True, align=LEFT, border=False)
    _style(ws.cell(row=fr + 2, column=1,
                   value="ค่าที่อ่านจาก header ของไฟล์ (สีขาว) · ช่องสีเหลือง = กรอกเอง · แถวสีเทา = ไฟล์ทดสอบ (< 1500 shot) · "
                         f"laser time = shot ÷ {LASER_HZ:g} Hz · Altitude = signal bin × bin range"),
           size=8, align=LEFT, border=False)
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1

    if qc:
        write_summary(wb, D, B, block_notes, findings or [], case=case)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)


def write_summary(wb: Workbook, D: pd.DataFrame, B: pd.DataFrame, block_notes: Dict[int, str],
                  findings: List[str], *, case: str = "") -> None:
    """A sheet that answers "how did the day go": the day verdict, each block's numbers,
    and the findings worth recording."""
    import raw_quality_check as rq
    ws = wb.create_sheet("สรุป (Summary)", 0)
    ws.sheet_view.showGridLines = False
    for j, w in enumerate([8, 15, 8, 46, 12, 12, 13, 13, 13, 15, 11, 11, 12], start=1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws["A1"] = f"สรุปการทดลอง {D['end'].min():%d/%m/%Y}" + (f" — {case}" if case else "")
    _style(ws["A1"], bold=True, size=14, align=NOWRAP, border=False)
    r = 3
    dd = rq.day_decision(D)
    n_cal = int(D["day_cal_files"].iloc[0]) if "day_cal_files" in D else 0
    for txt in [f"ไฟล์ทั้งหมด {dd['n_files']} (+{dd['n_test']} ไฟล์ทดสอบ) · ชุดค่าตั้ง {len(B)} ชุด",
                f"NRB / PBL: {dd['nrb_verdict']} — ใช้ได้ {dd['nrb'][rq.USABLE]} · มีข้อจำกัด {dd['nrb'][rq.LIMITED]} · "
                f"ใช้ไม่ได้ {dd['nrb'][rq.UNUSABLE]} · ช่วงที่ใช้ได้ {', '.join(dd['nrb_periods']) or '-'}",
                f"δ: {dd['delta_verdict']} — ใช้ได้ {dd['delta'][rq.USABLE]} · มีข้อจำกัด {dd['delta'][rq.LIMITED]} · "
                f"ใช้ไม่ได้ {dd['delta'][rq.UNUSABLE]} · ไฟล์ที่คาลิเบรต δ ได้ {n_cal}",
                "ข้อจำกัดของทั้งวัน: " + ("; ".join(sorted(set(dd["nrb_common"]) | set(dd["delta_common"]))) or "-")]:
        _style(ws.cell(row=r, column=1, value=txt), size=10, align=NOWRAP, border=False)
        r += 1
    r += 1
    M = block_metrics(D)
    heads = ["ชุด", "ช่วงเวลา", "ไฟล์", "ทดลองอะไร", "peak ∥ (mV)", "peak ⊥ (mV)", "PMT สูงสุด (µA)",
             "bg คืน ∥ (MHz)", "bg คืน ⊥ (MHz)", "SNR ⊥ 5 km (คืน)", "∥ ถึง (km)", "⊥ ถึง (km)", "เจอเมฆ (ไฟล์)"]
    for j, h in enumerate(heads, start=1):
        _style(ws.cell(row=r, column=j, value=h), bold=True, fill=HEAD_FILL, color="FFFFFF")
    for i, row in M.iterrows():
        b = B[B.block == row["block"]].iloc[0]
        vals = [int(row["block"]), f"{b['first']:%H:%M} – {b['last']:%H:%M}", int(row["files"]),
                block_notes.get(int(row["block"]), ""), row["par_peak"], row["perp_peak"], row["pmt_max"],
                row["night_bg_par"], row["night_bg_perp"], row["night_snr5_perp"],
                row["par_top_km"], row["perp_top_km"], int(row["clouds"])]
        for j, v in enumerate(vals, start=1):
            if isinstance(v, float) and not np.isfinite(v):
                v = "-"
            _style(ws.cell(row=r + 1 + i, column=j, value=v), align=LEFT if j == 4 else CENTER,
                   fill=(PatternFill("solid", fgColor=BLOCK_FILLS[(int(row["block"]) - 1) % len(BLOCK_FILLS)])
                         if j == 1 else None),
                   fmt="0.00" if j in (8, 9) else ("0.0" if j in (5, 6, 7, 10, 11, 12) else None))
    r += len(M) + 2
    if findings:
        _style(ws.cell(row=r, column=1, value="สิ่งที่ได้จากวันนี้"), bold=True, size=11, align=LEFT, border=False)
        r += 1
        for k, t in enumerate(findings, start=1):
            c = ws.cell(row=r, column=1, value=f"{k}. {t}")
            _style(c, size=10, align=NOWRAP, border=False)
            ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=13)
            ws.row_dimensions[r].height = 30
            r += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("-o", "--out")
    ap.add_argument("--case", default="")
    ap.add_argument("--qc", action="store_true", help="add the Raw QC measurements (slower: reads the data)")
    ap.add_argument("--block-note", action="append", default=[], metavar="N=text",
                    help="what setting block N was testing")
    ap.add_argument("--note", action="append", default=[], metavar="HH:MM=text", help="note for one file")
    ap.add_argument("--finding", action="append", default=[], help="a line for the summary sheet")
    a = ap.parse_args()
    folder = Path(a.folder)
    D = read_folder(folder)
    if D.empty:
        raise SystemExit("No readable Licel raw file in this folder.")
    if a.qc:
        import raw_quality_check as rq
        Q = rq.check_folder(folder)
        keep = ["file", "night", "par_peak_mv", "perp_peak_mv", "par_pmt_uA", "perp_pmt_uA",
                "par_bg_photon_mhz", "perp_bg_photon_mhz", "par_top_m", "perp_top_m",
                "use_nrb", "use_delta", "use_nrb_reason", "use_delta_reason",
                "perp_snr_5km", "par_snr_3km",
                "day_cal_files", "G1", "time"]
        D = D.merge(Q[[c for c in keep if c in Q.columns]], on="file", how="left")
        D["night"] = D["night"].fillna(False).astype(bool)
    bn = {int(x.split("=", 1)[0]): x.split("=", 1)[1] for x in a.block_note}
    fn_notes = {x.split("=", 1)[0]: x.split("=", 1)[1] for x in a.note}
    out = Path(a.out) if a.out else folder / f"Log-{D['end'].min():%Y-%m-%d}.xlsx"
    write_sheet(D, out, case=a.case, folder=str(folder), qc=a.qc,
                block_notes=bn, file_notes=fn_notes, findings=a.finding)
    B = block_table(D)
    print(B.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
