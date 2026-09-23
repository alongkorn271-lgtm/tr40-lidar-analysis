"""Excel check sheet for raw-data quality, in the layout of LiDAR-DCT-04.

Sheets:
  เกณฑ์ (Criteria)     every criterion: what is measured, PASS/WARN/FAIL, why, reference, fix.
                        The PASS/WARN limits are cells; every verdict formula reads them.
  สรุป (Summary)        per case: % PASS night / day, P/W/F counts, status, score.
  Session               once-per-session items (dark file, ±45° calibration, log sheet ...).
  QC <case>             one row per raw file: measured values from raw_quality_check.py and
                        verdict formulas, laid out like the DCT-04 case sheets.
  QC Template           the same sheet, empty, for a new case.

Usage (one --case per session):
    python raw_quality_sheet.py out.xlsx --case "C01-01|<raw folder>|2026-09-08|2400|3.75|15"
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.workbook.defined_name import DefinedName

import raw_quality_check as rq

FONT = "Leelawadee UI"   # Thai + ∥/⊥ glyphs (Tahoma drops the symbols)
THIN = Side(style="thin", color="808080")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT = Alignment(horizontal="left", vertical="top", wrap_text=True)
NOWRAP = Alignment(horizontal="left", vertical="center", wrap_text=False)
HEAD_FILL = PatternFill("solid", fgColor="1F3864")
SUB_FILL = PatternFill("solid", fgColor="D9E1F2")
PAR_FILL = PatternFill("solid", fgColor="DDEBF7")
PERP_FILL = PatternFill("solid", fgColor="FCE4D6")
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")
VERDICT_STYLE = {   # text, fill, font colour (Excel's good/neutral/bad cell styles)
    "PASS": ("C6EFCE", "006100"), "ผ่าน": ("C6EFCE", "006100"),
    "WARN": ("FFEB9C", "9C5700"), "บางส่วน": ("FFEB9C", "9C5700"),
    "FAIL": ("FFC7CE", "9C0006"), "ไม่ผ่าน": ("FFC7CE", "9C0006"),
    "NA": ("F2F2F2", "808080"),
    "ใช้ได้": ("C6EFCE", "006100"), "ใช้ได้มีข้อจำกัด": ("FFEB9C", "9C5700"), "ใช้ไม่ได้": ("FFC7CE", "9C0006"),
}

# ---- Criteria text ----------------------------------------------------------------
# id, group, name, applies, how, pass, warn, fail, lim1, lim2, why, ref, impact, fix
CRITERIA_ROWS = [
    ("A1", "A · การเก็บข้อมูล", "จำนวน shot ต่อไฟล์", "ทุกไฟล์",
     "อ่านจาก header (TR0 และ TR1)",
     "≥ 95 % ของแผน", "≥ 1500 shot", "< 1500 shot", 0.95, 1500,
     "noise แบบ Poisson ลดลงตาม √shot ช่อง ⊥ มีสัญญาณน้อยที่สุดจึงเสียก่อน ไฟล์ที่ต่ำกว่า 1500 shot มักเป็นการยิงทดสอบหรือหยุดกลางคัน pipeline จะข้ามไฟล์นั้น",
     "Licel TR manual §9.3 (stderr ∝ 1/√shots); nrb_engine min_shots = 1500",
     "SNR ต่ำ, profile หายจาก RTI", "ตรวจ shot ก่อนบันทึก แยกไฟล์ทดสอบออกจากโฟลเดอร์ข้อมูลจริง"),
    ("A2", "A · การเก็บข้อมูล", "Header ตรงกับแผน", "ทุกไฟล์",
     "HV, bin width, discriminator (3.1746 mV = ระดับ 8), input range 500 mV, ระยะสูงสุด และ ∥ = ⊥",
     "ตรงทุกค่า", "–", "ค่าใดค่าหนึ่งต่างจากแผน", None, None,
     "gain ของ PMT เปลี่ยนแรงมากตาม HV ส่วน glue gain (87/96), dead time (4.7/5.0 ns) และค่าคาลิเบรต C วัดไว้ที่ HV 750 V, disc 8, bin นี้ ถ้าเปลี่ยนค่าเหล่านี้ ค่าคงที่ทั้งหมดต้องวัดใหม่",
     "Licel PM-HV manual §5.6 (dead time ขึ้นกับ HV/disc)",
     "ค่าคงที่ของเครื่องใช้ไม่ได้ ผล δ และ NRB เพี้ยนแบบไม่รู้ตัว", "ล็อก preset ACQ ต่อ case, ตรวจ header ทุกวัน"),
    ("B1", "B · Analog", "Analog peak (หัก background)", "ทุกไฟล์, แยก ∥/⊥",
     "ค่าสูงสุดของ analog − ค่าเฉลี่ย pretrigger",
     "≤ 200 mV", "200–250 mV", "> 250 mV (ครึ่งหนึ่งของ range 500 mV)", 200, 250,
     "ภาคขยาย analog ต้องทำงานเชิงเส้น คู่มือ PM-HV ให้ peak ไม่เกินครึ่ง input range และค่าราว 494 mV คือเพดาน ADC (ตัด) ส่วนที่ตัดคือ near-field ซึ่ง glue และ PBL ต้องใช้",
     "Licel PM-HV manual §5.2; docs/analog_range_and_hv.md",
     "near-field ∥ ถูกตัด, glue window หาย, δ ใกล้พื้นผิด", "ลด HV ของ ∥ หรือใส่ ND filter เฉพาะ ∥ (ไม่ลดพลังงานเลเซอร์ เพราะ ⊥ อ่อนอยู่แล้ว)"),
    ("B2", "B · Analog", "ADC overflow bins (ข้อมูลประกอบ ไม่นับคะแนน)", "ทุกไฟล์, แยก ∥/⊥",
     "นับ bin ที่ Licel ตั้ง overflow flag (dataset OF0)",
     "0 bin", "–", "≥ 1 bin", 0, None,
     "ถ้า flag หมายถึง ADC ออกนอกช่วง ค่าใน bin นั้นไม่ใช่สัญญาณจริง แต่ข้อมูลจริงมี flag ⊥ ขึ้นไกลถึง 1.9 km และ flag ∥ ถึง 7.6 km ทั้งที่ analog เฉลี่ยที่ระยะนั้นต่ำกว่าไม่กี่ mV ความหมายของ flag (ค่าเฉลี่ยหรือราย shot, bit ไหนคือช่องไหน) จึงต้องยืนยันกับ Licel ก่อน ใช้ B1 เป็นตัวตัดสินการอิ่มตัวแทน",
     "Licel TR manual (overflow dataset) — รอยืนยันความหมาย bit",
     "ต้องแทนค่า ทำให้ความไม่แน่นอนเพิ่ม", "เหมือน B1"),
    ("B3", "B · Analog", "กระแสเฉลี่ยของ PMT", "ทุกไฟล์, แยก ∥/⊥",
     "(analog pretrigger − dark offset กลางคืน) × 20 µA/mV (50 Ω)",
     "≤ 80 µA", "80–100 µA", "> 100 µA", 80, 100,
     "คู่มือ PM-HV จำกัดกระแส anode เฉลี่ยไว้ที่ 100 µA (5 mV ที่ 50 Ω) ถ้าเกิน gain จะลอย, PMT ล้าและอายุสั้นลง แสงอาทิตย์คือแหล่งหลักของกระแสนี้",
     "Licel PM-HV manual §5.1; เอกสาร PMT current (artifact)",
     "กลางวัน gain ไม่คงที่, photon ใช้ไม่ได้", "Alluxa ultra-narrow filter, ลด FOV (field stop) หรือลด HV ตอนกลางวัน"),
    ("C1", "C · Photon", "Sky background ของ photon", "ทุกไฟล์, แยก ∥/⊥",
     "ค่าเฉลี่ย photon ใน pretrigger (MHz, ยังไม่แก้ dead time)",
     "< 10 MHz", "10–40 MHz", "≥ 40 MHz", 10, 40,
     "ช่วง toggle ที่ใช้ glue คือ 10–40 MHz ถ้า background สูงกว่า min toggle แล้ว photon จะไม่เหลือช่วงเชิงเส้นให้ต่อกับ analog (Licel ให้ใช้ analog ที่ scale แล้วแทน) และค่า dead-time correction จะโตมาก",
     "Licel TR manual §9.7.5; toggle study 08–15 ก.ย.",
     "กลางวัน photon ใช้ไม่ได้ ต้องพึ่ง analog อย่างเดียว", "filter แคบลง (Alluxa), ลด FOV"),
    ("C2", "C · Photon", "Photon background กลางคืน (dark + แสงรบกวน)", "กลางคืน 19:00–05:30",
     "เหมือน C1 แต่ดูเฉพาะไฟล์กลางคืน",
     "≤ 0.5 MHz", "0.5–1 MHz", "> 1 MHz", 0.5, 1.0,
     "เป็นพื้น noise ของระยะไกล 22 ก.ย. 2569 วัดได้ 0.00 MHz เมื่อปิดไฟห้องและจอคอมพิวเตอร์ ตรงกับคู่มือ PM-HV บทที่ 4 ที่ว่า PMT ในที่มืดแทบไม่นับอะไรเลย ค่าที่สูงกว่านี้จึงคือแสงรบกวนในห้อง ไม่ใช่ dark count (14–15 ก.ย. เปิดจอไว้ ได้ 1.5–1.8 MHz)",
     "วัดจาก raw 08/09/14/15/22 ก.ย. (เกณฑ์ภายใน)",
     "SNR ของ ⊥ ที่ 5 km ตกหลายเท่า คาลิเบรต δ ไม่ได้", "ปิดไฟห้องและจอ (S9) ตรวจรอยรั่วของกล่องตัวรับ แล้ววัด dark file"),
    ("D1", "D · Glue", "มีช่วง glue ที่ใช้ได้", "กลางคืน, แยก ∥/⊥",
     "bin ที่ photon (แก้ dead time) อยู่ใน 10–40 MHz, analog > 200σ, ไม่มี overflow, อยู่ใต้เมฆที่ทำให้ analog อิ่มตัว",
     "≥ 20 bin และยาว ≥ 300 m", "–", "ไม่มีช่วง", 20, 300,
     "gain analog→photon ต้องวัดจากช่วงที่ทั้งสองช่องเป็นเชิงเส้นพร้อมกัน ถ้าไม่มี pipeline ต้องใช้ gain คงที่ ซึ่งผิดได้ถ้าระบบลอย",
     "nrb_engine GLUE_ANALOG_SNR_MIN, GLUE_MIN_SPAN_M; Licel §9.7",
     "ใช้ fallback gain ตรวจสอบตัวเองไม่ได้", "แก้ B1 (analog ∥ อิ่มตัว) และเลี่ยงคืนที่มีเมฆต่ำ"),
    ("D2", "D · Glue", "Glue gain ตรงกับค่าของเครื่อง", "กลางคืน, แยก ∥/⊥",
     "median(photon/analog) เทียบกับ 3.75 m: ∥ 87, ⊥ 96; 30 m: ∥ 90, ⊥ 97 MHz/mV",
     "ต่างไม่เกิน ±5 %", "±5–15 %", "> ±15 %", 5, 15,
     "gain เป็นค่าคงที่ของเครื่อง ถ้าเพี้ยนแปลว่ามีช่องใดไม่เชิงเส้น (analog อิ่มตัว, photon pile-up, dead time ผิด) หรือ background ผิด",
     "robust glue study 08–15 ก.ย.; GLUE_GAIN_TOLERANCE = 0.15",
     "NRB near-field และ δ มี bias", "ตรวจ B1/C1 และ dead time"),
    ("E1", "E · SNR", "SNR ของ ∥ ที่ 2.5–3.5 km", "ทุกไฟล์",
     "median SNR ต่อ 30 m (เอาค่าที่ดีกว่าระหว่าง photon กับ analog)",
     "≥ 10", "3–10", "< 3", 10, 3,
     "SNR 3 คือขีดตรวจจับได้ ส่วน 10 (noise 10 %) คือระดับที่หาขอบชั้น aerosol/PBL ได้ ใช้ 30 m เพื่อให้เทียบ MPL ได้โดยตรงไม่ว่าจะใช้ bin เท่าไร",
     "เกณฑ์ SNR ที่ใช้ทั่วไปใน lidar; snr_gate ใน pipeline",
     "ชั้นบนหาย, PBL ผิด", "เพิ่ม shot, ลด background (C1/C2)"),
    ("E2", "E · SNR", "SNR ของ ⊥ ที่ 4.5–5.5 km", "กลางคืน",
     "median SNR ต่อ 30 m ในช่วงคาลิเบรต",
     "≥ 3", "1–3", "< 1", 3, 1,
     "ค่าคาลิเบรต C หาจากช่วงนี้ ถ้า SNR < 1 จะไม่มีสัญญาณ ⊥ ให้คาลิเบรต (pipeline ใช้ C ค่ามัธยฐานกลางคืนแทน) ส่วน SNR ≥ 3 ทำให้ C จาก median ราว 30 block คลาดไม่เกินประมาณ 6 %",
     "depol_engine CAL_MIN_CROSS_SNR = 1; Freudenthaler et al. 2016 (AMT)",
     "δ ต้องใช้ C ยืม ความไม่แน่นอนของ δ สูง", "เพิ่ม shot (C02 6000 shot ได้ SNR 12–15), แก้ C2, เลี่ยงคืนที่มีเมฆ"),
    ("F1", "F · Polarization", "อัตราส่วน ⊥/∥ ที่ 4.5–5.5 km คงที่", "กลางคืน",
     "|ratio / median ของคืนนั้น − 1| ใช้เฉพาะไฟล์ที่ SNR ทั้งสองช่อง ≥ 3 และต้องมีอย่างน้อย 3 ไฟล์",
     "±10 %", "±10–20 %", "> ±20 %", 10, 20,
     "ในอากาศสะอาด อัตราส่วนนี้ = C × δ_mol จึงควรคงที่ ถ้าเปลี่ยน แปลว่าแนวแสงลอย, polarization หมุน หรือมี aerosol/เมฆบางในช่วงอ้างอิง",
     "Freudenthaler 2016; Behrendt & Nakamura 2002 (δ_mol)",
     "C ไม่เสถียร δ ลอยตามเวลา", "ตรวจ alignment, ทำ ±45° calibration"),
    ("G1", "G · สภาพอากาศ", "เมฆต่ำกว่า 3 km", "ทุกไฟล์ (ข้อมูลประกอบ ไม่นับคะแนน)",
     "สัญญาณ analog หลังคูณ r² กระโดด ≥ 4 เท่า และยอด ≥ 10σ (ไวกว่าตัวตรวจของ glue ที่ต้องถึง 100 mV)",
     "ไม่มี", "มีเมฆ", "–", 3000, None,
     "ไม่ใช่ความผิดของเครื่อง แต่อธิบายว่าทำไม D1/E2/F1 ไม่ผ่าน ใช้แยกปัญหาฤดูฝนออกจากปัญหาเครื่อง",
     "glue_cloud_base_m; MPL cloud product",
     "–", "จดสภาพฟ้าใน log (S2)"),
]

SESSION_ROWS = [
    ("S1", "แผน, log sheet และ header ตรงกัน", "ค่าใน DCT-04 (bin, จำนวน bin, laser time) ตรงกับ header จริง",
     "sheet ต้องตรงกับไฟล์ ไม่งั้นย้อนตรวจไม่ได้ว่าเก็บด้วยค่าอะไร"),
    ("S2", "กรอก log sheet ครบ", "Sliding roof, สภาพอากาศ/% ฝน, ความชื้น, ผู้รับผิดชอบ ครบทุกแถว",
     "ใช้แยกผลของสภาพอากาศ (G1) ออกจากปัญหาเครื่อง"),
    ("S3", "เก็บได้ครบตามตาราง", "สัดส่วนช่วงเวลาตามแผนที่มีไฟล์ ≥ 1500 shot ภายใน ±30 นาที ต้อง ≥ 90 %",
     "ช่วงที่หายไปทำให้ RTI/PBL ขาดความต่อเนื่อง"),
    ("S4", "มี dark file ในรอบนั้น", "ปิดเลเซอร์หรือปิดกล้อง, HV/disc เท่าตอนวัด, shot ≥ ไฟล์วัด ทั้ง ∥ และ ⊥",
     "ใช้แยก dark count กับแสงรบกวน (C2) และแก้ baseline analog; dark file ที่มีตอนนี้มาจาก setup เดือน ก.ค."),
    ("S5", "±45° calibration / cross-talk", "ทำ Δ90 calibration อย่างน้อยครั้งต่อการเปลี่ยน optics และวัด ε ได้ ≤ 1 %",
     "cross-talk ประเมินจากการเทียบ MPL ได้ราว 1.5 % ซึ่งมากกว่า δ_mol (0.0042) หลายเท่า; Freudenthaler 2016"),
    ("S6", "บันทึกพลังงานเลเซอร์ / QS", "ค่าพลังงาน (หรือ monitor) และ QS delay ทุกไฟล์",
     "แยกได้ว่าสัญญาณเปลี่ยนเพราะบรรยากาศหรือเพราะเลเซอร์"),
    ("S7", "แยกไฟล์ทดสอบออก", "ไม่มีไฟล์ < 1500 shot ปนในโฟลเดอร์ข้อมูล หรือตั้งชื่อ/ย้ายไว้แยก",
     "กันไม่ให้ pipeline หรือคนอื่นหยิบไปใช้ผิด"),
    ("S9", "ปิดไฟห้องและจอคอมพิวเตอร์ระหว่างวัด", "ห้องมืดสนิท จอดับหรืออยู่นอกห้อง ตลอดการวัดกลางคืน",
     "22 ก.ย. 2569: ปิดจอแล้ว background กลางคืนลงจาก 1.5 MHz เหลือ 0.00 MHz และ SNR ของ ⊥ ที่ 5 km ขึ้นจาก <1 เป็น 11–20"),
    ("S8", "เวลาเครื่องตรงกับ MPL", "Step 6 จับคู่ได้ดีที่สุดที่ lag 0 (เวลาในไฟล์ Licel = เวลาจบการเก็บ)",
     "ถ้าเวลาไม่ตรง การเทียบกับ MPL จะผิดทั้งชุด"),
]

# Per-file verdict columns scored in the summary: (label, criterion id, verdict column key)
SCORED = rq.SCORED
CONTEXT = rq.CONTEXT


def _name(cid: str, which: str) -> str:
    return f"LIM_{cid}_{which}"


SYMBOLS = "∥⊥"
SYMBOL_FONT = "Segoe UI Symbol"


def _has_thai(text: str) -> bool:
    return any("฀" <= ch <= "๿" for ch in text)


def _rich(text: str, *, bold: bool, size: float, color: str) -> CellRichText:
    """Thai and ∥/⊥ in one cell: Excel drops the symbols unless they get their own font run."""
    parts, buf, cur = [], "", None
    for ch in text:
        kind = ch in SYMBOLS
        if cur is not None and kind != cur:
            parts.append((buf, cur)); buf = ""
        buf += ch; cur = kind
    if buf:
        parts.append((buf, cur))
    return CellRichText([TextBlock(InlineFont(rFont=SYMBOL_FONT if sym else FONT, sz=size, b=bold, color=color), t)
                         for t, sym in parts])


def _style(cell, *, bold=False, size=9, fill=None, color="000000", align=CENTER, border=True, fmt=None):
    cell.font = Font(name=FONT, size=size, bold=bold, color=color)
    v = cell.value
    if isinstance(v, str) and not v.startswith("=") and _has_thai(v) and any(c in v for c in SYMBOLS):
        cell.value = _rich(v, bold=bold, size=size, color=color)
    cell.alignment = align
    if fill is not None:
        cell.fill = fill
    if border:
        cell.border = BOX
    if fmt:
        cell.number_format = fmt


def _verdict_colours(ws, rng: str) -> None:
    for text, (bg, fg) in VERDICT_STYLE.items():
        ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=[f'"{text}"'],
                                                      fill=PatternFill("solid", fgColor=bg, bgColor=bg),
                                                      font=Font(name=FONT, color=fg, bold=True)))


# ---- Criteria sheet ------------------------------------------------------------------
def build_criteria_sheet(wb: Workbook) -> None:
    ws = wb.create_sheet("เกณฑ์ (Criteria)")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "เกณฑ์สัญญาณที่มีคุณภาพ — Raw data TR40 dual-PMT (∥ = TR0, ⊥ = TR1, 532 nm)"
    _style(ws["A1"], bold=True, size=14, align=NOWRAP, border=False)
    ws["A2"] = ("ช่อง 'ค่าผ่าน' และ 'ค่าเตือน' (สีเหลือง) ถูกอ้างอิงโดยสูตรตัดสินในทุก sheet QC — แก้ตรงนี้แล้วผลทั้งไฟล์เปลี่ยนตาม. "
                "PASS = ใช้ได้เต็มที่, WARN = ใช้ได้แต่มีความไม่แน่นอนเพิ่ม, FAIL = ข้อมูลส่วนนั้นเชื่อไม่ได้, NA = ไม่เกี่ยวกับไฟล์นั้น (เช่น เกณฑ์กลางคืนกับไฟล์กลางวัน)")
    _style(ws["A2"], size=9, align=LEFT, border=False)
    ws.merge_cells("A2:N2"); ws.row_dimensions[2].height = 30
    heads = ["ID", "กลุ่ม", "เกณฑ์", "ใช้กับ", "วัดอย่างไร", "PASS", "WARN", "FAIL", "ค่าผ่าน", "ค่าเตือน",
             "เหตุผล (ทำไมต้องผ่าน)", "อ้างอิง", "ถ้าไม่ผ่าน → ผลต่อข้อมูล", "แนวทางแก้"]
    widths = [6, 14, 24, 16, 30, 13, 12, 16, 8, 8, 52, 30, 26, 32]
    for j, (h, w) in enumerate(zip(heads, widths), start=1):
        c = ws.cell(row=4, column=j, value=h)
        _style(c, bold=True, fill=HEAD_FILL, color="FFFFFF")
        ws.column_dimensions[get_column_letter(j)].width = w
    for i, row in enumerate(CRITERIA_ROWS, start=5):
        cid, grp, name, applies, how, p, w_, f, l1, l2, why, ref, impact, fix = row
        vals = [cid, grp, name, applies, how, p, w_, f, l1, l2, why, ref, impact, fix]
        for j, v in enumerate(vals, start=1):
            c = ws.cell(row=i, column=j, value=v)
            _style(c, bold=(j == 1), align=CENTER if j in (1, 6, 7, 8, 9, 10) else LEFT,
                   fill=INPUT_FILL if j in (9, 10) and v is not None else None)
        for which, col in (("PASS", 9), ("WARN", 10)):
            if ws.cell(row=i, column=col).value is not None:
                wb.defined_names[_name(cid, which)] = DefinedName(
                    _name(cid, which), attr_text=f"'{ws.title}'!${get_column_letter(col)}${i}")
        ws.row_dimensions[i].height = 78
    r0 = 5 + len(CRITERIA_ROWS) + 1
    ws.cell(row=r0, column=1, value="เกณฑ์ระดับ session (ตรวจครั้งเดียวต่อรอบการวัด — กรอกใน sheet Session)")
    _style(ws.cell(row=r0, column=1), bold=True, size=11, align=NOWRAP, border=False)
    for j, h in enumerate(["ID", "เกณฑ์", "", "", "ผ่านเมื่อ", "", "", "", "", "", "เหตุผล"], start=1):
        if h:
            _style(ws.cell(row=r0 + 1, column=j, value=h), bold=True, fill=HEAD_FILL, color="FFFFFF")
    for i, (sid, name, rule, why) in enumerate(SESSION_ROWS, start=r0 + 2):
        ws.cell(row=i, column=1, value=sid); ws.cell(row=i, column=2, value=name)
        ws.cell(row=i, column=5, value=rule); ws.cell(row=i, column=11, value=why)
        ws.merge_cells(start_row=i, start_column=2, end_row=i, end_column=4)
        ws.merge_cells(start_row=i, start_column=5, end_row=i, end_column=10)
        ws.merge_cells(start_row=i, start_column=11, end_row=i, end_column=14)
        for j in range(1, 15):
            _style(ws.cell(row=i, column=j), bold=(j == 1), align=CENTER if j == 1 else LEFT)
        ws.row_dimensions[i].height = 36
    ws.freeze_panes = "D5"
    r1 = r0 + 2 + len(SESSION_ROWS) + 1
    refs = [
        "อ้างอิง",
        "Licel GmbH, Transient Recorder TR40 manual — §9.3 standard error, §9.7 gluing / toggle rates, §9.7.5 high background.",
        "Licel GmbH, PM-HV (Hamamatsu R9880U) photomultiplier module manual — §5.1 average current ≤ 100 µA, §5.2 peak ≤ ½ input range, §5.6 dead time.",
        "Freudenthaler V. et al. (2016), Depolarization ratio profiling at several wavelengths in pure Saharan dust during SAMUM 2006 / Δ90 calibration, AMT / Tellus B (2009).",
        "Behrendt A. & Nakamura T. (2002), Calculation of the calibration constant of polarization lidar and its dependency on atmospheric temperature, Opt. Express 10, 805.",
        "Flynn C. J. et al. (2007), Novel polarization-sensitive micropulse lidar measurement technique, Opt. Express 15, 2785; Córdoba-Jabonero C. et al. (2021), MPL depolarization.",
        "ข้อมูลภายใน: toggle/dead-time/glue study, PMT current study และ raw_quality_check.py (08, 09, 14, 15 ก.ย. 2569).",
    ]
    for k, t in enumerate(refs):
        c = ws.cell(row=r1 + k, column=1, value=t)
        _style(c, bold=(k == 0), size=9, align=LEFT, border=False)
        ws.merge_cells(start_row=r1 + k, start_column=1, end_row=r1 + k, end_column=14)


# ---- Case sheet ------------------------------------------------------------------------
def _case_columns() -> List[Dict]:
    """Column specs. kind: val (dataframe column), f (formula template), static (dataframe verdict)."""
    P, S = PAR_FILL, PERP_FILL
    def v(key, grp, label, src, fmt="0.0", fill=None, w=8):
        return dict(key=key, grp=grp, label=label, kind="val", src=src, fmt=fmt, fill=fill, w=w)
    def f(key, grp, label, tmpl, fill=None, w=7, verdict=True, fmt=None):
        return dict(key=key, grp=grp, label=label, kind="f", tmpl=tmpl, fill=fill, w=w, verdict=verdict, fmt=fmt)

    def maxrule(cid, val, night=False):
        cond = f',{{dn}}<>"Night"' if night else ""
        return (f'=IF(OR({{use}}="N",{{{val}}}=""{cond}),"NA",IF({{{val}}}<={_name(cid,"PASS")},"PASS",'
                f'IF({{{val}}}<={_name(cid,"WARN")},"WARN","FAIL")))')

    def minrule(cid, val, night=False):
        cond = f',{{dn}}<>"Night"' if night else ""
        return (f'=IF(OR({{use}}="N",{{{val}}}=""{cond}),"NA",IF({{{val}}}>={_name(cid,"PASS")},"PASS",'
                f'IF({{{val}}}>={_name(cid,"WARN")},"WARN","FAIL")))')

    cols = [
        v("time", "เวลา", "Time (สิ้นสุด)", "time", fmt="dd/mm hh:mm", w=11),
        v("dn", "เวลา", "Day/Night", "dn", fmt="@", w=7),
        v("file", "ไฟล์", "File", "file", fmt="@", w=15),
        v("shots", "A · การเก็บข้อมูล", "Shots", "shots", fmt="0", w=7),
        f("use", "A · การเก็บข้อมูล", "ใช้ประเมิน", f'=IF({{shots}}>={_name("A1","WARN")},"Y","N")', verdict=False, w=6),
        f("A1", "A · การเก็บข้อมูล", "A1", f'=IF({{shots}}="","NA",IF({{shots}}>={_name("A1","PASS")}*$E$3,"PASS",IF({{shots}}>={_name("A1","WARN")},"WARN","FAIL")))'),
        dict(key="A2", grp="A · การเก็บข้อมูล", label="A2", kind="static", src="A2", w=7),
        v("A2n", "A · การเก็บข้อมูล", "A2 หมายเหตุ", "A2_note", fmt="@", w=12),
        v("pk_p", "B · Analog ∥", "Peak mV", "par_peak_mv", fill=P),
        f("B1p", "B · Analog ∥", "B1", maxrule("B1", "pk_p"), fill=P),
        v("of_p", "B · Analog ∥", "Overflow bin", "par_overflow_bins", fmt="0", fill=P, w=7),
        f("B2p", "B · Analog ∥", "B2", f'=IF(OR({{use}}="N",{{of_p}}=""),"NA",IF({{of_p}}<={_name("B2","PASS")},"PASS","FAIL"))', fill=P),
        v("ua_p", "B · Analog ∥", "PMT µA", "par_pmt_uA", fmt="0", fill=P, w=7),
        f("B3p", "B · Analog ∥", "B3", maxrule("B3", "ua_p"), fill=P),
        v("pk_s", "B · Analog ⊥", "Peak mV", "perp_peak_mv", fill=S),
        f("B1s", "B · Analog ⊥", "B1", maxrule("B1", "pk_s"), fill=S),
        v("of_s", "B · Analog ⊥", "Overflow bin", "perp_overflow_bins", fmt="0", fill=S, w=7),
        f("B2s", "B · Analog ⊥", "B2", f'=IF(OR({{use}}="N",{{of_s}}=""),"NA",IF({{of_s}}<={_name("B2","PASS")},"PASS","FAIL"))', fill=S),
        v("ua_s", "B · Analog ⊥", "PMT µA", "perp_pmt_uA", fmt="0", fill=S, w=7),
        f("B3s", "B · Analog ⊥", "B3", maxrule("B3", "ua_s"), fill=S),
        v("bg_p", "C · Photon", "∥ bg MHz", "par_bg_photon_mhz", fmt="0.00", fill=P),
        f("C1p", "C · Photon", "C1 ∥", maxrule("C1", "bg_p"), fill=P),
        f("C2p", "C · Photon", "C2 ∥", maxrule("C2", "bg_p", night=True), fill=P),
        v("bg_s", "C · Photon", "⊥ bg MHz", "perp_bg_photon_mhz", fmt="0.00", fill=S),
        f("C1s", "C · Photon", "C1 ⊥", maxrule("C1", "bg_s"), fill=S),
        f("C2s", "C · Photon", "C2 ⊥", maxrule("C2", "bg_s", night=True), fill=S),
        v("gb_p", "D · Glue ∥", "bin", "par_glue_bins", fmt="0", fill=P, w=6),
        v("gs_p", "D · Glue ∥", "span m", "par_glue_span_m", fmt="0", fill=P, w=7),
        f("D1p", "D · Glue ∥", "D1", f'=IF(OR({{use}}="N",{{dn}}<>"Night"),"NA",IF(AND({{gb_p}}>={_name("D1","PASS")},{{gs_p}}>={_name("D1","WARN")}),"PASS","FAIL"))', fill=P),
        v("gg_p", "D · Glue ∥", "gain", "par_glue_gain", fill=P, w=7),
        v("gr_p", "D · Glue ∥", "ref", "par_ref_gain", fmt="0", fill=P, w=5),
        f("gd_p", "D · Glue ∥", "ต่าง %", '=IF(OR({gg_p}="",{gr_p}=""),"",ABS({gg_p}/{gr_p}-1)*100)', fill=P, verdict=False, fmt="0.0", w=7),
        f("D2p", "D · Glue ∥", "D2", maxrule("D2", "gd_p", night=True), fill=P),
        v("gb_s", "D · Glue ⊥", "bin", "perp_glue_bins", fmt="0", fill=S, w=6),
        v("gs_s", "D · Glue ⊥", "span m", "perp_glue_span_m", fmt="0", fill=S, w=7),
        f("D1s", "D · Glue ⊥", "D1", f'=IF(OR({{use}}="N",{{dn}}<>"Night"),"NA",IF(AND({{gb_s}}>={_name("D1","PASS")},{{gs_s}}>={_name("D1","WARN")}),"PASS","FAIL"))', fill=S),
        v("gg_s", "D · Glue ⊥", "gain", "perp_glue_gain", fill=S, w=7),
        v("gr_s", "D · Glue ⊥", "ref", "perp_ref_gain", fmt="0", fill=S, w=5),
        f("gd_s", "D · Glue ⊥", "ต่าง %", '=IF(OR({gg_s}="",{gr_s}=""),"",ABS({gg_s}/{gr_s}-1)*100)', fill=S, verdict=False, fmt="0.0", w=7),
        f("D2s", "D · Glue ⊥", "D2", maxrule("D2", "gd_s", night=True), fill=S),
        v("snr3", "E · SNR", "∥ SNR 3 km", "par_snr_3km", fmt="0.0", fill=P),
        f("E1", "E · SNR", "E1", minrule("E1", "snr3"), fill=P),
        v("snr5", "E · SNR", "⊥ SNR 5 km", "perp_snr_5km", fmt="0.0", fill=S),
        f("E2", "E · SNR", "E2", minrule("E2", "snr5", night=True), fill=S),
        v("rdev", "F · Pol.", "⊥/∥ ต่าง %", "ratio_dev_pct", fmt="0.0", w=7),
        f("F1", "F · Pol.", "F1", maxrule("F1", "rdev", night=True)),
        v("cb", "G · อากาศ", "เมฆ ∥ m", "par_cloud_base_m", fmt="0", w=7),
        f("G1", "G · อากาศ", "G1", f'=IF({{use}}="N","NA",IF(AND({{cb}}<>"",{{cb}}<{_name("G1","PASS")}),"WARN","PASS"))'),
    ]
    return cols


SUMMARY_COLS = ["PASS", "WARN", "FAIL", "NA"]
LOG_COLS = [("Sliding roof", 9), ("สภาพอากาศ / % ฝน", 12), ("ความชื้น", 8), ("ผู้รับผิดชอบ", 10), ("หมายเหตุ", 24)]
FIRST_ROW = 7


def build_case_sheet(wb: Workbook, title: str, meta: Dict, D: Optional[pd.DataFrame], n_blank: int = 0) -> Dict:
    """One QC sheet. Returns {'sheet', 'first', 'last', 'col': {key: letter}}."""
    ws = wb.create_sheet(title)
    ws.sheet_view.showGridLines = False
    cols = _case_columns()
    letters = {c["key"]: get_column_letter(j) for j, c in enumerate(cols, start=1)}
    ncol = len(cols)
    j_res = ncol + 1
    res_letters = {k: get_column_letter(j_res + i) for i, k in enumerate(SUMMARY_COLS)}
    j_pct = j_res + len(SUMMARY_COLS)
    j_log = j_pct + 1
    last_col = j_log + len(LOG_COLS) - 1

    # header block (DCT-04 style)
    def hdr(r, c, text, bold=True, fill=None, span=1, color="000000"):
        cell = ws.cell(row=r, column=c, value=text)
        _style(cell, bold=bold, size=9, fill=fill, color=color, align=Alignment(horizontal="left", vertical="center", wrap_text=True))
        if span > 1:
            ws.merge_cells(start_row=r, start_column=c, end_row=r, end_column=c + span - 1)
            for k in range(c, c + span):
                ws.cell(row=r, column=k).border = BOX
    hdr(1, 1, "Project :"); hdr(1, 3, "Polarization system optimization — Raw signal quality check", span=12)
    hdr(1, 16, "Ref:"); hdr(1, 17, "LiDAR-DCT-04-QC", span=4)
    hdr(2, 1, "Testing Conditions :"); hdr(2, 3, meta.get("condition", ""), bold=False, span=12)
    hdr(2, 16, "Date:"); hdr(2, 17, meta.get("date", ""), bold=False, span=4)
    hdr(3, 1, "Plan shots"); c = ws.cell(row=3, column=5, value=meta.get("shots")); _style(c, fill=INPUT_FILL)
    hdr(3, 6, "Bin (m)"); c = ws.cell(row=3, column=8, value=meta.get("bin_m")); _style(c, fill=INPUT_FILL)
    hdr(3, 9, "HV ∥/⊥ (V)")
    c = ws.cell(row=3, column=10, value=f"{meta.get('hv_par', meta.get('hv', 750))} / "
                                       f"{meta.get('hv_perp', meta.get('hv', 750))}")
    _style(c, fill=INPUT_FILL)
    hdr(3, 11, "Altitude (km)"); c = ws.cell(row=3, column=13, value=meta.get("altitude_km")); _style(c, fill=INPUT_FILL)
    hdr(3, 14, "Dark offset ∥/⊥ (mV)"); hdr(3, 17, meta.get("dark", ""), bold=False, span=4)
    hdr(4, 1, "Raw folder"); hdr(4, 3, meta.get("folder", ""), bold=False, span=18)
    for r in (1, 2, 3):
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=2 if r < 3 else 4)

    # two-row column header
    hr1, hr2 = FIRST_ROW - 2, FIRST_ROW - 1
    j = 1
    while j <= ncol:
        grp = cols[j - 1]["grp"]; k = j
        while k + 1 <= ncol and cols[k]["grp"] == grp:
            k += 1
        c = ws.cell(row=hr1, column=j, value=grp)
        _style(c, bold=True, fill=HEAD_FILL, color="FFFFFF")
        if k > j:
            ws.merge_cells(start_row=hr1, start_column=j, end_row=hr1, end_column=k)
        j = k + 1
    for j, col in enumerate(cols, start=1):
        c = ws.cell(row=hr2, column=j, value=col["label"])
        _style(c, bold=True, fill=col.get("fill") or SUB_FILL)
        ws.column_dimensions[get_column_letter(j)].width = col["w"]
    _style(ws.cell(row=hr1, column=j_res, value="ผลต่อไฟล์"), bold=True, fill=HEAD_FILL, color="FFFFFF")
    ws.merge_cells(start_row=hr1, start_column=j_res, end_row=hr1, end_column=j_pct)
    for i, k in enumerate(SUMMARY_COLS + ["% PASS"]):
        _style(ws.cell(row=hr2, column=j_res + i, value=k), bold=True, fill=SUB_FILL)
        ws.column_dimensions[get_column_letter(j_res + i)].width = 6
    _style(ws.cell(row=hr1, column=j_log, value="บันทึกหน้างาน (กรอกเอง)"), bold=True, fill=HEAD_FILL, color="FFFFFF")
    ws.merge_cells(start_row=hr1, start_column=j_log, end_row=hr1, end_column=last_col)
    for i, (lab, w) in enumerate(LOG_COLS):
        _style(ws.cell(row=hr2, column=j_log + i, value=lab), bold=True, fill=INPUT_FILL)
        ws.column_dimensions[get_column_letter(j_log + i)].width = w
    ws.row_dimensions[hr2].height = 30

    rows = [] if D is None else [r for _, r in D.iterrows()]
    n = len(rows) + n_blank
    first, last = FIRST_ROW, FIRST_ROW + max(n, 1) - 1
    verdict_letters = [letters[c["key"]] for c in cols if (c["kind"] == "f" and c.get("verdict")) or c["kind"] == "static"]
    for i in range(n):
        r = FIRST_ROW + i
        src = rows[i] if i < len(rows) else None
        for j, col in enumerate(cols, start=1):
            cell = ws.cell(row=r, column=j)
            if col["kind"] == "f":
                tmpl = col["tmpl"]
                for key, let in letters.items():
                    tmpl = tmpl.replace("{" + key + "}", f"{let}{r}")
                cell.value = tmpl
                _style(cell, fill=col.get("fill"), fmt=col.get("fmt"))
            else:
                val = None
                if src is not None:
                    if col["src"] == "dn":
                        val = "Night" if bool(src.get("night")) else "Day"
                    elif col["src"] == "time":
                        val = pd.Timestamp(src["time"]).to_pydatetime() if pd.notna(src["time"]) else None
                    else:
                        x = src.get(col["src"])
                        if isinstance(x, (float, np.floating)):
                            val = None if not np.isfinite(x) else float(x)
                        elif isinstance(x, (np.integer,)):
                            val = int(x)
                        else:
                            val = None if (x is None or (isinstance(x, str) and x == "")) else x
                cell.value = val
                fill = col.get("fill") if src is not None else INPUT_FILL
                _style(cell, fill=fill, fmt=col.get("fmt"))
        rng = f"{verdict_letters[0]}{r}:{verdict_letters[-1]}{r}"
        for k, name in enumerate(SUMMARY_COLS):
            c = ws.cell(row=r, column=j_res + k, value=f'=COUNTIF({rng},"{name}")')
            _style(c)
        p, w_, f_ = (f"{res_letters[x]}{r}" for x in ("PASS", "WARN", "FAIL"))
        c = ws.cell(row=r, column=j_pct, value=f'=IF(({p}+{w_}+{f_})=0,"",{p}/({p}+{w_}+{f_}))')
        _style(c, fmt="0%")
        for k in range(len(LOG_COLS)):
            _style(ws.cell(row=r, column=j_log + k), fill=INPUT_FILL, align=LEFT)
    for let in verdict_letters:
        _verdict_colours(ws, f"{let}{first}:{let}{last}")

    fr = last + 2
    for k, txt in enumerate(["Report By : ……………………………………  Date : ……………",
                             "Checked By : ……………………………………  Date : ……………",
                             "Approved By : ……………………………………  Date : ……………"]):
        c = ws.cell(row=fr, column=1 + k * 12, value=txt)
        _style(c, bold=True, align=NOWRAP, border=False)
    ws.cell(row=fr + 2, column=1, value=("หมายเหตุ: ค่าวัดมาจาก raw_quality_check.py (อ่านไฟล์ Licel โดยตรง) · Time = เวลาสิ้นสุดการเก็บ · "
                                         "Night = 19:00–05:30 · ใช้ประเมิน = N เมื่อ shot < 1500 (ไฟล์ทดสอบ นับแค่ A1) · "
                                         "SNR คิดต่อ 30 m · ช่องสีเหลือง = กรอกเอง"))
    _style(ws.cell(row=fr + 2, column=1), size=8, align=NOWRAP, border=False)
    ws.freeze_panes = ws.cell(row=FIRST_ROW, column=4)
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    return dict(sheet=title, first=first, last=last, col=letters)


# ---- Session sheet --------------------------------------------------------------------
def build_session_sheet(wb: Workbook, cases: List[Dict]) -> Dict[str, str]:
    """Rows S1..S8, a status + evidence column per case. Returns {case: status column letter}."""
    ws = wb.create_sheet("Session")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "เกณฑ์ระดับ session — ตรวจครั้งเดียวต่อรอบการวัด (PASS / WARN / FAIL / NA)"
    _style(ws["A1"], bold=True, size=12, align=NOWRAP, border=False)
    _style(ws.cell(row=3, column=1, value="ID"), bold=True, fill=HEAD_FILL, color="FFFFFF")
    _style(ws.cell(row=3, column=2, value="เกณฑ์"), bold=True, fill=HEAD_FILL, color="FFFFFF")
    ws.column_dimensions["A"].width = 5; ws.column_dimensions["B"].width = 26
    out = {}
    for k, case in enumerate(cases):
        js = 3 + 2 * k
        _style(ws.cell(row=3, column=js, value=case["name"]), bold=True, fill=HEAD_FILL, color="FFFFFF")
        _style(ws.cell(row=3, column=js + 1, value="หลักฐาน / หมายเหตุ"), bold=True, fill=HEAD_FILL, color="FFFFFF")
        ws.column_dimensions[get_column_letter(js)].width = 8
        ws.column_dimensions[get_column_letter(js + 1)].width = 34
        out[case["name"]] = get_column_letter(js)
    for i, (sid, name, _rule, _why) in enumerate(SESSION_ROWS, start=4):
        _style(ws.cell(row=i, column=1, value=sid), bold=True)
        _style(ws.cell(row=i, column=2, value=name), align=LEFT)
        for k, case in enumerate(cases):
            status, note = case.get("session", {}).get(sid, (None, ""))
            _style(ws.cell(row=i, column=3 + 2 * k, value=status), bold=True, fill=INPUT_FILL if status is None else None)
            _style(ws.cell(row=i, column=4 + 2 * k, value=note), align=LEFT, size=8)
        ws.row_dimensions[i].height = 48
    last = 3 + len(SESSION_ROWS)
    _verdict_colours(ws, f"C4:{get_column_letter(2 + 2 * len(cases))}{last}")
    ws.freeze_panes = "C4"
    return out


# ---- Summary sheet ---------------------------------------------------------------------
def build_summary_sheet(wb: Workbook, cases: List[Dict], infos: Dict[str, Dict], sess_cols: Dict[str, str]) -> None:
    ws = wb.create_sheet("สรุป (Summary)", 0)
    ws.sheet_view.showGridLines = False
    ws["A1"] = "สรุปคุณภาพ raw data ต่อ case — ผ่านกี่ข้อ"
    _style(ws["A1"], bold=True, size=14, align=NOWRAP, border=False)
    ws["A2"] = ("สถานะต่อเกณฑ์: ผ่าน = PASS ≥ 90 % ของไฟล์ที่เกี่ยวข้อง · บางส่วน = PASS+WARN ≥ 50 % · ไม่ผ่าน = ต่ำกว่านั้น · "
                "นับเฉพาะไฟล์ที่ใช้ประเมิน (≥ 1500 shot) ยกเว้น A1 ที่นับทุกไฟล์ · B2 และ G1 เป็นข้อมูลประกอบ ไม่นับคะแนน")
    _style(ws["A2"], size=9, align=LEFT, border=False)
    ws.merge_cells("A2:R2"); ws.row_dimensions[2].height = 28
    ws.column_dimensions["A"].width = 8; ws.column_dimensions["B"].width = 34; ws.column_dimensions["C"].width = 12
    hr = 5
    for j, h in enumerate(["เกณฑ์", "ชื่อ", "ใช้กับ"], start=1):
        _style(ws.cell(row=hr, column=j, value=h), bold=True, fill=HEAD_FILL, color="FFFFFF")
        ws.merge_cells(start_row=hr - 1, start_column=j, end_row=hr, end_column=j)
        _style(ws.cell(row=hr - 1, column=j), fill=HEAD_FILL)
        ws.cell(row=hr - 1, column=j).value = h
    sub = ["คืน %PASS", "วัน %PASS", "P / W / F", "สถานะ"]
    crit = {r[0]: r for r in CRITERIA_ROWS}
    status_cols = {}
    for k, case in enumerate(cases):
        j0 = 4 + k * len(sub)
        c = ws.cell(row=hr - 1, column=j0, value=f"{case['name']} · {case.get('label', '')}")
        _style(c, bold=True, fill=HEAD_FILL, color="FFFFFF")
        ws.merge_cells(start_row=hr - 1, start_column=j0, end_row=hr - 1, end_column=j0 + len(sub) - 1)
        for i, s in enumerate(sub):
            _style(ws.cell(row=hr, column=j0 + i, value=s), bold=True, fill=SUB_FILL)
            ws.column_dimensions[get_column_letter(j0 + i)].width = 9 if i < 3 else 10
        status_cols[case["name"]] = get_column_letter(j0 + 3)

    def countifs(info, vcol, val, extra=""):
        sh = f"'{info['sheet']}'"
        rng = f"{sh}!${vcol}${info['first']}:${vcol}${info['last']}"
        return f'COUNTIFS({rng},"{val}"{extra})'

    r = hr + 1
    score_rows = []
    for label, cid, vkey in SCORED + CONTEXT:
        _style(ws.cell(row=r, column=1, value=label), bold=True)
        _style(ws.cell(row=r, column=2, value=crit[cid][2]), align=LEFT)
        _style(ws.cell(row=r, column=3, value=crit[cid][3]), align=LEFT, size=8)
        for k, case in enumerate(cases):
            info = infos[case["name"]]; j0 = 4 + k * len(sub)
            L = info["col"][vkey]; dn = info["col"]["dn"]; use = info["col"]["use"]
            sh = f"'{info['sheet']}'"
            dn_rng = f"{sh}!${dn}${info['first']}:${dn}${info['last']}"
            use_rng = f"{sh}!${use}${info['first']}:${use}${info['last']}"
            use_cond = "" if vkey == "A1" else f',{use_rng},"Y"'
            for i, part in enumerate(("Night", "Day")):
                ex = f',{dn_rng},"{part}"{use_cond}'
                p = countifs(info, L, "PASS", ex)
                tot = "+".join(countifs(info, L, v, ex) for v in ("PASS", "WARN", "FAIL"))
                c = ws.cell(row=r, column=j0 + i, value=f'=IF(({tot})=0,"NA",{p}/({tot}))')
                _style(c, fmt="0%")
            P = countifs(info, L, "PASS", use_cond); W = countifs(info, L, "WARN", use_cond); F = countifs(info, L, "FAIL", use_cond)
            _style(ws.cell(row=r, column=j0 + 2, value=f'={P}&" / "&{W}&" / "&{F}'))
            stat = (f'=IF(({P}+{W}+{F})=0,"NA",IF({P}/({P}+{W}+{F})>=0.9,"ผ่าน",'
                    f'IF(({P}+{W})/({P}+{W}+{F})>=0.5,"บางส่วน","ไม่ผ่าน")))')
            _style(ws.cell(row=r, column=j0 + 3, value=stat), bold=True)
        if (label, cid, vkey) in SCORED:
            score_rows.append(r)
        else:
            _style(ws.cell(row=r, column=3, value=f"{crit[cid][3]} · ไม่นับคะแนน"), align=LEFT, size=8)
        ws.row_dimensions[r].height = 24
        r += 1
    # session rows
    _style(ws.cell(row=r, column=1, value="Session"), bold=True, fill=SUB_FILL)
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=3 + len(cases) * len(sub))
    r += 1
    for i, (sid, name, _rule, _why) in enumerate(SESSION_ROWS):
        _style(ws.cell(row=r, column=1, value=sid), bold=True)
        _style(ws.cell(row=r, column=2, value=name), align=LEFT)
        _style(ws.cell(row=r, column=3, value="ต่อ session"), size=8)
        for k, case in enumerate(cases):
            j0 = 4 + k * len(sub)
            src = f"Session!${sess_cols[case['name']]}${4 + i}"
            for q in range(3):
                _style(ws.cell(row=r, column=j0 + q, value="" if q < 2 else f'=IF({src}="","–",{src})'))
            ws.merge_cells(start_row=r, start_column=j0, end_row=r, end_column=j0 + 2)
            stat = f'=IF({src}="PASS","ผ่าน",IF({src}="WARN","บางส่วน",IF({src}="FAIL","ไม่ผ่าน","NA")))'
            _style(ws.cell(row=r, column=j0 + 3, value=stat), bold=True)
        score_rows.append(r)
        r += 1
    # score
    r += 1
    _style(ws.cell(row=r, column=1, value="คะแนน"), bold=True, fill=HEAD_FILL, color="FFFFFF")
    _style(ws.cell(row=r, column=2, value="ผ่าน / บางส่วน / ไม่ผ่าน  (จากเกณฑ์ที่ประเมินได้)"), bold=True, fill=HEAD_FILL, color="FFFFFF", align=LEFT)
    ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=3)
    for k, case in enumerate(cases):
        j0 = 4 + k * len(sub); L = status_cols[case["name"]]
        cells = ",".join(f"{L}{x}" for x in score_rows)
        def cnt(val):
            return "+".join(f'COUNTIF({L}{x},"{val}")' for x in score_rows)
        f = (f'="ผ่าน "&({cnt("ผ่าน")})&" / "&(({cnt("ผ่าน")})+({cnt("บางส่วน")})+({cnt("ไม่ผ่าน")}))'
             f'&"   (บางส่วน "&({cnt("บางส่วน")})&", ไม่ผ่าน "&({cnt("ไม่ผ่าน")})&")"')
        c = ws.cell(row=r, column=j0, value=f)
        _style(c, bold=True, size=10, fill=SUB_FILL)
        ws.merge_cells(start_row=r, start_column=j0, end_row=r, end_column=j0 + 3)
    ws.row_dimensions[r].height = 30
    for case in cases:
        L = status_cols[case["name"]]
        _verdict_colours(ws, f"{L}{hr + 1}:{L}{r - 2}")
    ws.freeze_panes = ws.cell(row=hr + 1, column=4)


def build_usability_sheet(wb: Workbook, cases: List[Dict]) -> None:
    """Per-file decision: can the profile be used for backscatter (NRB/PBL) and for
    depolarization, up to what height, and why not — with the day's usable periods."""
    ws = wb.create_sheet("ใช้ได้ไหม (Usability)")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "ไฟล์ไหนใช้ได้ — ตัดสินต่อ profile แยก NRB/PBL กับ δ"
    _style(ws["A1"], bold=True, size=14, align=NOWRAP, border=False)
    ws["A2"] = ("ใช้ได้ = ไม่มีข้อจำกัด · ใช้ได้มีข้อจำกัด = ใช้ได้ภายใต้เหตุผลที่ระบุ (เช่น ถึงความสูงที่บอก) · "
                "ใช้ไม่ได้ = ไม่ควรนำไปคำนวณ · ใช้ได้ถึง = ความสูงแรกเหนือ 300 m ที่ SNR ต่อ 30 m < 3 · "
                "เวลา profile = เวลาสิ้นสุดการเก็บ (ชื่อไฟล์ Licel)")
    _style(ws["A2"], size=9, align=NOWRAP, border=False)
    heads = ["Case", "เวลา profile (สิ้นสุด)", "ช่วงบันทึก (header)", "Day/Night", "Shots",
             "∥ ใช้ได้ถึง (km)", "⊥ ใช้ได้ถึง (km)", "NRB / PBL", "เหตุผล NRB / PBL", "δ", "เหตุผล δ"]
    widths = [11, 16, 20, 9, 7, 11, 11, 16, 50, 16, 60]
    for j, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(j)].width = w
    r = 4
    for case in cases:
        D = case["df"]
        if D is None or not len(D):
            continue
        dd = rq.day_decision(D)
        lines = [
            f"{case['name']} · {case.get('date', '')} — {dd['n_files']} ไฟล์ (+{dd['n_test']} ไฟล์ทดสอบ)",
            f"NRB / PBL: {dd['nrb_verdict']} · ใช้ได้ {dd['nrb'][rq.USABLE]} · มีข้อจำกัด {dd['nrb'][rq.LIMITED]} · "
            f"ใช้ไม่ได้ {dd['nrb'][rq.UNUSABLE]} · ช่วงที่ใช้ได้ {', '.join(dd['nrb_periods']) or '–'}",
            f"δ: {dd['delta_verdict']} · ใช้ได้ {dd['delta'][rq.USABLE]} · มีข้อจำกัด {dd['delta'][rq.LIMITED]} · "
            f"ใช้ไม่ได้ {dd['delta'][rq.UNUSABLE]} · ช่วงที่ใช้ได้ {', '.join(dd['delta_periods']) or '–'}",
        ]
        common = sorted(set(dd["nrb_common"]) | set(dd["delta_common"]))
        if common:
            lines.append("ข้อจำกัดของทั้งวัน: " + "; ".join(common))
        for k, t in enumerate(lines):
            c = ws.cell(row=r, column=1, value=t)
            _style(c, bold=(k == 0), size=11 if k == 0 else 9, align=NOWRAP, border=False,
                   fill=SUB_FILL if k == 0 else None)
            if k == 0:
                for j in range(2, len(heads) + 1):
                    ws.cell(row=r, column=j).fill = SUB_FILL
            r += 1
        for j, h in enumerate(heads, start=1):
            _style(ws.cell(row=r, column=j, value=h), bold=True, fill=HEAD_FILL, color="FFFFFF")
        r += 1
        first = r
        for _, row in D.sort_values("time").iterrows():
            t = pd.Timestamp(row["time"]) if pd.notna(row["time"]) else None
            a0, a1 = row.get("acq_start"), row.get("acq_stop")
            acq = (f"{pd.Timestamp(a0):%H:%M:%S}–{pd.Timestamp(a1):%H:%M:%S}"
                   if pd.notna(a0) and pd.notna(a1) else "")
            test = row["shots"] < rq.CRITERIA["A1"]["min_shots"]
            vals = [case["name"], t.to_pydatetime() if t is not None else None, acq,
                    "Night" if row["night"] else "Day", int(row["shots"]),
                    None if test else round(float(row["par_top_m"]) / 1000, 1),
                    None if test else round(float(row["perp_top_m"]) / 1000, 1),
                    rq.USE_TEXT[row["use_nrb"]], row["use_nrb_reason"],
                    rq.USE_TEXT[row["use_delta"]], row["use_delta_reason"]]
            for j, v in enumerate(vals, start=1):
                c = ws.cell(row=r, column=j, value=v)
                _style(c, align=LEFT if j in (9, 11) else CENTER, bold=j in (8, 10),
                       fmt="dd/mm hh:mm" if j == 2 else ("0.0" if j in (6, 7) else None))
            r += 1
        for L in ("H", "J"):
            _verdict_colours(ws, f"{L}{first}:{L}{r - 1}")
        r += 2
    ws.freeze_panes = "C4"


def build_workbook(cases: List[Dict], out: Path) -> None:
    wb = Workbook()
    wb.remove(wb.active)
    build_criteria_sheet(wb)
    infos = {}
    for case in cases:
        infos[case["name"]] = build_case_sheet(wb, f"QC {case['name']}", case, case["df"])
    build_case_sheet(wb, "QC Template", dict(condition="(กรอก case)", folder="python raw_quality_check.py <folder> --out ... แล้ววางค่า หรือกรอกเอง"),
                     None, n_blank=40)
    sess = build_session_sheet(wb, cases)
    build_summary_sheet(wb, cases, infos, sess)
    build_usability_sheet(wb, cases)
    wb.move_sheet("ใช้ได้ไหม (Usability)", offset=1 - wb.sheetnames.index("ใช้ได้ไหม (Usability)"))
    wb.move_sheet("Session", offset=2 - wb.sheetnames.index("Session"))
    wb.save(out)


def auto_session(D: pd.DataFrame) -> Dict[str, tuple]:
    """Session items the raw files alone can decide; the rest stay blank for the operator."""
    test = D[D["shots"] < rq.CRITERIA["A1"]["min_shots"]]
    s7 = (("WARN", f"มีไฟล์ทดสอบ {len(test)} ไฟล์ปนอยู่: "
           + ", ".join(f"{f} ({int(n)} shot)" for f, n in zip(test["file"], test["shots"])))
          if len(test) else ("PASS", "ไม่มี"))
    return {"S7": s7}


def write_case_workbook(D: pd.DataFrame, out: Path, *, name: str, folder: str, plan: Dict,
                        date: str = "", condition: str = "") -> None:
    """Check sheet for a single case (what the GUI Raw QC step saves)."""
    dark = (f"{D.par_dark_offset_mv.iloc[0]:.2f} / {D.perp_dark_offset_mv.iloc[0]:.2f}"
            if len(D) and "par_dark_offset_mv" in D else "")
    case = dict(name=name, label=date, date=date, folder=folder, df=D, dark=dark,
                condition=condition or name, session=auto_session(D), **plan)
    build_workbook([case], Path(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--case", action="append", required=True,
                    help='"name|folder|date|plan_shots|plan_bin_m|plan_altitude_km"')
    a = ap.parse_args()
    cases = []
    for spec in a.case:
        name, folder, date, shots, bin_m, alt = spec.split("|")
        plan = dict(shots=int(shots), bin_m=float(bin_m), hv=750, altitude_km=float(alt))
        D = rq.check_folder(Path(folder), plan=plan)
        dark = f"{D.par_dark_offset_mv.iloc[0]:.2f} / {D.perp_dark_offset_mv.iloc[0]:.2f}" if len(D) else ""
        cases.append(dict(name=name, label=date, date=date, folder=folder, df=D, dark=dark,
                          condition=name, **plan))
    build_workbook(cases, Path(a.out))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
