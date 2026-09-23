"""Raw-data quality check for the TR40 dual-PMT (TR0 = parallel, TR1 = perpendicular).

Reads the Licel raw binary files directly (no NRB pipeline) and measures, per file
and per channel, the numbers that decide whether the acquisition itself is good:

  A  acquisition   shots, header configuration (HV, bin width, discriminator, range)
  B  analog        peak vs half the input range, overflow flags, PMT average current
  C  photon        sky background vs the minimum toggle rate
  D  glue          a clean toggle window exists and its gain matches the instrument's
  E  SNR           parallel at 3 km, perpendicular in the calibration window
  F  polarization  stability of the perp/par ratio in the calibration window (night)
  G  environment   analog-overloading cloud below 3 km (context, not a fault)

Thresholds live in CRITERIA so the check sheet, the GUI and this script agree.
Usage:
    python raw_quality_check.py <raw folder> [--plan-shots 2400 --plan-bin 3.75 ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import licel_binary_reader as lbr
import nrb_engine as ne

C_LIGHT = 299_792_458.0

# ---- Criteria -------------------------------------------------------------------
# (id, group, name, unit, pass_rule, warn_rule, source)
# pass/warn rules are applied by verdict(); the text is what the check sheet prints.
CRITERIA = {
    "A1": dict(name="Shots per file", unit="shots",
               pass_="≥ 95 % of plan", warn="≥ 1500 (pipeline minimum)", fail="< 1500",
               plan_frac=0.95, min_shots=1500),
    "A2": dict(name="Header matches plan (HV, bin, discriminator, range, altitude)", unit="-",
               pass_="all match, ∥ = ⊥", warn="-", fail="any differs"),
    "B1": dict(name="Analog peak (background removed)", unit="mV",
               pass_="≤ 200 mV (≤ 40 % of range)", warn="200–250 mV", fail="> 250 mV (half of 500 mV range)",
               pass_max=200.0, warn_max=250.0),
    "B2": dict(name="ADC overflow bins", unit="bins", pass_="0", warn="-", fail="≥ 1", pass_max=0),
    "B3": dict(name="PMT average current", unit="µA",
               pass_="≤ 80 µA", warn="80–100 µA", fail="> 100 µA", pass_max=80.0, warn_max=100.0),
    "C1": dict(name="Photon sky background", unit="MHz",
               pass_="< 10 MHz (below min toggle)", warn="10–40 MHz", fail="≥ 40 MHz (no night toggle window)",
               pass_max=10.0, warn_max=40.0),
    # 22 Sep 2026: with the room dark and the monitor off this reads 0.00 MHz, so anything
    # above ~0.1 is stray light in the room, not the PMT (PM-HV manual ch. 4: a dark PMT
    # counts almost nothing). 08-09 Sep read 0.12-0.18 (a little light), 14-15 Sep 1.5-1.8
    # (the monitor left on).
    "C2": dict(name="Night photon background (dark + sky, 19:00–05:30)", unit="MHz",
               pass_="≤ 0.1 MHz (dark room)", warn="0.1–0.5 MHz", fail="> 0.5 MHz (stray light)",
               pass_max=0.1, warn_max=0.5),
    "D1": dict(name="Glue window, night (10–40 MHz, analog > 200σ, no overflow, below cloud)", unit="m",
               pass_="span ≥ 300 m and ≥ 20 bins", warn="-", fail="no window", min_span=300.0),
    "D2": dict(name="Glue gain vs instrument gain (3.75 m: ∥ 87, ⊥ 96; 30 m: ∥ 90, ⊥ 97 MHz/mV)", unit="%",
               pass_="within ±5 %", warn="±5–15 %", fail="> ±15 % or no gain", pass_max=5.0, warn_max=15.0),
    "E1": dict(name="∥ SNR at 2.5–3.5 km (per 30 m, better of photon/analog)", unit="-",
               pass_="≥ 10", warn="3–10", fail="< 3", pass_min=10.0, warn_min=3.0),
    "E2": dict(name="⊥ SNR at 4.5–5.5 km (per 30 m, better of photon/analog, calibration window)", unit="-",
               pass_="≥ 3", warn="1–3", fail="< 1", pass_min=3.0, warn_min=1.0),
    "F1": dict(name="⊥/∥ ratio at 4.5–5.5 km vs night median (C stability)", unit="%",
               pass_="within ±10 %", warn="±10–20 %", fail="> ±20 %", pass_max=10.0, warn_max=20.0),
    "G1": dict(name="Cloud below 3 km (jump ≥ 4× in the range-corrected analog)", unit="m",
               pass_="none", warn="cloud (weather, not a fault)", fail="-", max_base=3000.0),
}

# Night gain measured per bin width (robust glue, 08/14/15 Sep at 3.75 m, 09 Sep at 30 m)
REF_GAIN = {3.75: {"par": 87.0, "perp": 96.0}, 30.0: {"par": 90.0, "perp": 97.0}}
NIGHT_HOURS = (19.0, 5.5)   # local clock: after astronomical twilight, before dawn (Chiang Mai, Sep)


def is_night(t) -> bool:
    if pd.isna(t):
        return False
    h = t.hour + t.minute / 60.0
    return h >= NIGHT_HOURS[0] or h <= NIGHT_HOURS[1]


def ref_gain(bin_m: float, ch: str) -> float:
    key = min(REF_GAIN, key=lambda b: abs(b - bin_m))
    return REF_GAIN[key][ch]
DEAD_TIME = {"par": ne.DEAD_TIME_NS_PAR, "perp": ne.DEAD_TIME_NS_PERP}
CHANNELS = (("par", 0, 2, 1, 0), ("perp", 4, 6, 2, 1))   # name, analog col, photon col, flag bit, TR
NIGHT_BG_MHZ = 5.0          # photon background below this = night (dark offset + ratio reference)
SNR_BLOCK_M = 30.0
MIN_RATIO_FILES = 3         # F1 needs at least this many clean night files for its reference
MUA_PER_MV = 20.0           # 50 ohm input: 1 mV = 20 µA
TOP_SNR_MIN = 3.0           # usable signal: SNR per 30 m at least this
TOP_CAP_M = 15000.0


def verdict(cid: str, value) -> str:
    """PASS / WARN / FAIL / NA for a measured value against CRITERIA[cid]."""
    c = CRITERIA[cid]
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if "pass_max" in c and "warn_max" in c:
        return "PASS" if value <= c["pass_max"] else "WARN" if value <= c["warn_max"] else "FAIL"
    if "pass_max" in c:
        return "PASS" if value <= c["pass_max"] else "FAIL"
    if "pass_min" in c:
        return "PASS" if value >= c["pass_min"] else "WARN" if value >= c["warn_min"] else "FAIL"
    return "NA"


def _block_snr(counts: np.ndarray, bg_counts: float, n_bg: int, nb: int) -> np.ndarray:
    """Poisson SNR of background-subtracted photon counts summed over nb bins."""
    m = counts.size // nb
    c = counts[:m * nb].reshape(m, nb).sum(axis=1)
    s = c - nb * bg_counts
    var = c + nb ** 2 * bg_counts / max(n_bg, 1)
    return s / np.sqrt(np.where(var > 0, var, np.nan))


def cloud_base_m(r_m: np.ndarray, analog_raw: np.ndarray, *, bg_analog_mv: float,
                 analog_noise_mv: float, min_ratio: float = 4.0, min_peak_snr: float = 10.0,
                 max_range_m: float = 8000.0) -> float:
    """Range of the first cloud, or NaN — the same jump test as nrb_engine.glue_cloud_base_m
    but without its "the cloud must overload the analog (>= 100 mV)" condition, which is
    right for the glue and too blunt for a weather flag: on 2026-09-22 15:37 a cloud at
    1.7 km peaked at 175 mV raw (97.8 mV after the 90 m median) and went unflagged."""
    r = np.asarray(r_m, float); a = np.asarray(analog_raw, float)
    if r.size < 10 or not np.isfinite(bg_analog_mv):
        return np.nan
    dr = float(np.nanmedian(np.diff(r)))
    if not np.isfinite(dr) or dr <= 0:
        return np.nan
    n_s = max(1, int(round(ne.GLUE_CLOUD_SMOOTH_M / dr)))
    n_b = max(1, int(round(ne.GLUE_CLOUD_BELOW_M / dr)))
    n_j = max(1, int(round(ne.GLUE_CLOUD_JUMP_M / dr)))
    sig = pd.Series(a - float(bg_analog_mv)).rolling(n_s, center=True, min_periods=1).median()
    rc = sig * r ** 2
    below = rc.rolling(n_b, min_periods=1).median().shift(1)
    above = rc[::-1].rolling(n_j, min_periods=1).max()[::-1]
    sig_above = sig[::-1].rolling(n_j, min_periods=1).max()[::-1]
    noise = float(analog_noise_mv) if np.isfinite(analog_noise_mv) and analog_noise_mv > 0 else 0.0
    noise_s = noise / np.sqrt(n_s)
    # only the troposphere the flag is about: past ~8 km the range-corrected noise itself
    # jumps by more than 4x and every profile would "have a cloud"
    hit = ((r >= ne.GLUE_CLOUD_MIN_RANGE_M) & (r <= max_range_m) & (below.to_numpy() > 0)
           & (sig_above.to_numpy() >= min_peak_snr * noise_s)
           & (above.to_numpy() >= min_ratio * below.to_numpy()))
    idx = np.flatnonzero(hit)
    return float(r[idx[0]]) if idx.size else np.nan


def measure_file(path: Path) -> Optional[Dict]:
    """Raw metrics of one Licel file; None if it is not a readable raw file."""
    try:
        rf = lbr.parse_raw_file(path)
        a2 = lbr.raw_file_to_array(rf)
    except Exception:
        return None
    ts = ne.parse_licel_raw_timestamp(path.name)
    by_dev = {m.device: m for m in rf.datasets}
    bw = float(rf.datasets[0].binwidth_m)
    pre = int(round(ne.PRETRIGGER_TRACE_DISTANCE_M / bw))
    trim = max(1, int(round(90.0 / bw)))
    if a2.shape[0] <= pre + 10:
        return None
    n_pre = pre - 2 * trim
    dt_s = 2.0 * bw / C_LIGHT
    out = dict(file=path.name, time=pd.Timestamp(ts) if ts is not None else pd.NaT,
               start=rf.start_time, stop=rf.stop_time, shots=int(rf.shots), bin_m=bw)
    for ch, ca, cp, bit, tr in CHANNELS:
        ma, mp = by_dev.get(f"BT{tr}"), by_dev.get(f"BC{tr}")
        out[f"{ch}_hv"] = ma.hv if ma else np.nan
        out[f"{ch}_disc_mv"] = float(mp.input_range) if mp else np.nan
        out[f"{ch}_range_mv"] = float(ma.input_range) * 1000.0 if ma else np.nan
        out[f"{ch}_shots"] = ma.shots if ma else np.nan
        out[f"{ch}_bins"] = ma.ndata if ma else np.nan
        shots = float(ma.shots) if ma else float(rf.shots)

        A = a2[pre:, ca]; N = a2[pre:, cp]
        r = bw * (1 + np.arange(A.size))
        bgA = float(np.mean(a2[trim:pre - trim, ca])); sA = float(np.std(a2[trim:pre - trim, ca]))
        bgN = float(np.mean(a2[trim:pre - trim, cp]))
        flag = (a2[pre:, 8].astype(np.int64) & bit) > 0
        P = ne.dead_time_correct_mhz(N, DEAD_TIME[ch])
        bgP = float(np.mean(ne.dead_time_correct_mhz(a2[trim:pre - trim, cp], DEAD_TIME[ch])))
        out[f"{ch}_bg_analog_mv"] = bgA
        out[f"{ch}_bg_photon_mhz"] = bgN
        out[f"{ch}_peak_mv"] = float(np.max(A - bgA))
        out[f"{ch}_peak_r_m"] = float(r[int(np.argmax(A - bgA))])
        out[f"{ch}_overflow_bins"] = int(flag.sum())

        cb = ne.glue_cloud_base_m(r, A, bg_analog_mv=bgA, analog_noise_mv=sA)   # overloading cloud (glue)
        out[f"{ch}_cloud_base_m"] = cb
        out[f"{ch}_cloud_any_m"] = cloud_base_m(r, A, bg_analog_mv=bgA, analog_noise_mv=sA)
        cand = (~flag & (P >= ne.NIGHT_MIN_TOGGLE_MHZ) & (P <= ne.NIGHT_MAX_TOGGLE_MHZ)
                & (N * DEAD_TIME[ch] * 1e-3 < ne.DT_VALID_MAX_FRACTION))
        if np.isfinite(cb):
            cand &= r < cb
        g = ne.robust_glue_gain(A, P, cand, bg_analog_mv=bgA, bg_photon_mhz=bgP,
                                analog_noise_mv=sA, r_m=r)
        keep = cand & ((A - bgA) > ne.GLUE_ANALOG_SNR_MIN * sA)
        out[f"{ch}_glue_bins"] = int(g["n"])
        out[f"{ch}_glue_span_m"] = float(r[keep].max() - r[keep].min()) if keep.any() and g["n"] else 0.0
        out[f"{ch}_glue_gain"] = g["slope"]
        out[f"{ch}_glue_r2"] = g["r2"]

        counts = N * 1e6 * dt_s * shots
        bgc = bgN * 1e6 * dt_s * shots
        nb = max(1, int(round(SNR_BLOCK_M / bw)))
        snr = _block_snr(counts, bgc, n_pre, nb)
        rb = bw * nb * (0.5 + np.arange(snr.size))
        # analog: signal over the pretrigger noise averaged down to the block
        m = A.size // nb
        snr_a = (A[:m * nb].reshape(m, nb).mean(axis=1) - bgA) / (sA / np.sqrt(nb)) if sA > 0 else np.full(m, np.nan)
        # usable-signal top: first height above 300 m where the better of photon/analog SNR,
        # smoothed over 150 m, falls below 3 (a cloud or the end of the signal)
        best = np.fmax(snr[:m], snr_a) if m else snr
        sm = pd.Series(best).rolling(5, center=True, min_periods=1).median().to_numpy()
        above = rb[:m] >= 300.0
        low = np.flatnonzero(above & ~(sm >= TOP_SNR_MIN))
        out[f"{ch}_top_m"] = float(min(rb[low[0]] if low.size else rb[:m][-1], TOP_CAP_M))
        for lo, hi, key in ((750, 1250, "snr_1km"), (2500, 3500, "snr_3km"), (4500, 5500, "snr_5km")):
            w = (rb >= lo) & (rb <= hi)
            ps = float(np.nanmedian(snr[w])) if w.any() and np.isfinite(snr[w]).any() else np.nan
            asn = float(np.nanmedian(snr_a[w[:m]])) if w[:m].any() else np.nan
            out[f"{ch}_photon_{key}"] = ps
            out[f"{ch}_analog_{key}"] = asn
            out[f"{ch}_{key}"] = np.nanmax([ps, asn]) if np.isfinite([ps, asn]).any() else np.nan
        w = (r >= 4500) & (r <= 5500)
        out[f"{ch}_sig_5km_mhz"] = float(np.nanmedian(P[w] - bgP)) if w.any() else np.nan
    s_par, s_perp = out["par_sig_5km_mhz"], out["perp_sig_5km_mhz"]
    out["ratio_5km"] = s_perp / s_par if (np.isfinite(s_par) and s_par > 0) else np.nan
    return out


def check_folder(folder: Path, *, plan: Optional[Dict] = None) -> pd.DataFrame:
    """Measure every raw file in *folder* and add the verdict columns."""
    plan = plan or {}
    rows = [m for f in ne.glob_lidar_files(Path(folder), "*") if (m := measure_file(f))]
    if not rows:
        return pd.DataFrame()
    D = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

    D["night"] = D.time.apply(is_night)
    night = D[(D.par_bg_photon_mhz < NIGHT_BG_MHZ) & (D.shots >= CRITERIA["A1"]["min_shots"])]
    for ch, *_ in CHANNELS:
        off = float(night[f"{ch}_bg_analog_mv"].median()) if len(night) else np.nan
        D[f"{ch}_dark_offset_mv"] = off
        D[f"{ch}_pmt_uA"] = (D[f"{ch}_bg_analog_mv"] - off) * MUA_PER_MV
    snr_ok = (night.par_snr_5km >= 3) & (night.perp_snr_5km >= 3)
    # a reference needs a few clean nights' files, otherwise every file is compared with itself
    ref_ratio = float(night.loc[snr_ok, "ratio_5km"].median()) if snr_ok.sum() >= MIN_RATIO_FILES else np.nan
    D["ratio_ref"] = ref_ratio

    plan_shots = plan.get("shots"); plan_bin = plan.get("bin_m"); plan_hv = plan.get("hv")
    plan_alt = plan.get("altitude_km"); plan_disc = plan.get("disc_mv", 3.1746); plan_rng = plan.get("range_mv", 500.0)
    V = {}
    for i, row in D.iterrows():
        v = {}
        c = CRITERIA["A1"]
        v["A1"] = ("PASS" if plan_shots and row.shots >= c["plan_frac"] * plan_shots
                   else "PASS" if not plan_shots and row.shots >= c["min_shots"]
                   else "WARN" if row.shots >= c["min_shots"] else "FAIL")
        diffs = []
        for ch in ("par", "perp"):
            if plan_hv and row[f"{ch}_hv"] != plan_hv:
                diffs.append(f"{ch} HV {row[f'{ch}_hv']}")
            if abs(row[f"{ch}_disc_mv"] - plan_disc) > 0.01:
                diffs.append(f"{ch} disc {row[f'{ch}_disc_mv']}")
            if abs(row[f"{ch}_range_mv"] - plan_rng) > 0.5:
                diffs.append(f"{ch} range {row[f'{ch}_range_mv']:.0f}")
        if plan_bin and abs(row.bin_m - plan_bin) > 1e-6:
            diffs.append(f"bin {row.bin_m:g} m")
        if plan_alt and row.par_bins * row.bin_m < 0.99 * plan_alt * 1000 - ne.PRETRIGGER_TRACE_DISTANCE_M:
            diffs.append(f"alt {row.par_bins * row.bin_m / 1000:.1f} km")
        if row.par_hv != row.perp_hv or row.par_bins != row.perp_bins:
            diffs.append("∥≠⊥")
        v["A2"] = "FAIL" if diffs else "PASS"
        D.loc[i, "A2_note"] = "; ".join(diffs)
        for ch in ("par", "perp"):
            k = "p" if ch == "par" else "s"
            v[f"B1{k}"] = verdict("B1", row[f"{ch}_peak_mv"])
            v[f"B2{k}"] = "PASS" if row[f"{ch}_overflow_bins"] == 0 else "FAIL"
            v[f"B3{k}"] = verdict("B3", row[f"{ch}_pmt_uA"])
            v[f"C1{k}"] = verdict("C1", row[f"{ch}_bg_photon_mhz"])
            v[f"C2{k}"] = verdict("C2", row[f"{ch}_bg_photon_mhz"]) if row.night else "NA"
            ok_win = row[f"{ch}_glue_bins"] >= ne.GLUE_ROBUST_MIN_BINS and row[f"{ch}_glue_span_m"] >= CRITERIA["D1"]["min_span"]
            # by day the glue uses the night gain, so only the night needs its own window
            v[f"D1{k}"] = ("PASS" if ok_win else "FAIL") if row.night else "NA"
            dev = abs(row[f"{ch}_glue_gain"] / ref_gain(row.bin_m, ch) - 1) * 100 if ok_win else np.nan
            D.loc[i, f"{ch}_gain_dev_pct"] = dev
            D.loc[i, f"{ch}_ref_gain"] = ref_gain(row.bin_m, ch)
            v[f"D2{k}"] = verdict("D2", dev) if row.night else "NA"
        v["E1"] = verdict("E1", row.par_snr_3km)
        v["E2"] = verdict("E2", row.perp_snr_5km) if row.night else "NA"   # calibration is a night job
        dev_r = abs(row.ratio_5km / ref_ratio - 1) * 100 if (np.isfinite(ref_ratio) and row.perp_snr_5km >= 3 and row.par_snr_5km >= 3) else np.nan
        D.loc[i, "ratio_dev_pct"] = dev_r
        v["F1"] = verdict("F1", dev_r) if row.night else "NA"
        cb = row.par_cloud_any_m
        v["G1"] = "WARN" if np.isfinite(cb) and cb < CRITERIA["G1"]["max_base"] else "PASS"
        V[i] = v
    VD = pd.DataFrame.from_dict(V, orient="index")
    # a test fire (< min shots) is not a measurement: only A1/A2 describe it
    test = D["shots"] < CRITERIA["A1"]["min_shots"]
    VD.loc[test, [c for c in VD.columns if c not in ("A1", "A2")]] = "NA"
    D = pd.concat([D, VD], axis=1)
    return add_usability(D)


# ---- Per-file decision: can this profile be used, and for what? ------------------------
# Two products, judged separately, because one file can be fine for backscatter (NRB/PBL)
# and useless for depolarization. USABLE / LIMITED (usable, with the listed caveat) /
# UNUSABLE. Nothing is deleted: the decision travels with the file to Step 3 and Validation.
USABLE, LIMITED, UNUSABLE = "USABLE", "LIMITED", "UNUSABLE"
USE_TEXT = {USABLE: "ใช้ได้", LIMITED: "ใช้ได้มีข้อจำกัด", UNUSABLE: "ใช้ไม่ได้"}

# Heights that decide use: the boundary layer (~1 km) must be visible at all; a profile
# that stops below the aerosol layer top (~3 km for backscatter, ~2 km for delta) is
# usable only up to that height.
NRB_MIN_TOP_M, NRB_FULL_TOP_M = 1000.0, 3000.0
DELTA_MIN_TOP_M, DELTA_FULL_TOP_M = 1000.0, 2000.0

# (verdict column, value, night only, reason) — reasons are what the operator reads
NRB_LIMIT = [("B1p", "FAIL", False, "analog ∥ อิ่มตัว (near field)"),
             ("B3p", "FAIL", False, "กระแส PMT ∥ > 100 µA"),
             ("D1p", "FAIL", True, "ไม่มีช่วง glue ∥ (ใช้ gain คงที่)")]
DELTA_LIMIT = [("B1p", "FAIL", False, "analog ∥ อิ่มตัว → δ ใกล้พื้นเพี้ยน"),
               ("B3s", "FAIL", False, "กระแส PMT ⊥ > 100 µA"),
               ("E2", "FAIL", True, "คาลิเบรตเองไม่ได้ (ใช้ C ของคืน)"),
               ("C2s", "FAIL", True, "background ⊥ กลางคืนสูง"),
               ("F1", "FAIL", True, "อัตราส่วน ⊥/∥ ไม่คงที่"),
               ("D1s", "FAIL", True, "ไม่มีช่วง glue ⊥ (ใช้ gain คงที่)")]


def _hits(row, rules) -> List[str]:
    return [why for col, val, night_only, why in rules
            if row.get(col) == val and (row.get("night") or not night_only)]


def _top_reason(ch_sym: str, top: float, cloud: float) -> str:
    txt = f"สัญญาณ {ch_sym} ใช้ได้ถึง {top / 1000:.1f} km"
    return txt


def add_usability(D: pd.DataFrame) -> pd.DataFrame:
    """Add use_nrb / use_delta (USABLE | LIMITED | UNUSABLE), their reasons, the usable
    signal tops and the acquisition window to every file row."""
    D = D.copy()
    nrb, dlt, nrb_why, dlt_why = [], [], [], []
    # delta needs a calibration constant: at least one dark-sky file of the day whose
    # perpendicular signal reaches the calibration window, otherwise C must come from
    # another day and no delta of this day stands on its own
    dark = (D["shots"] >= CRITERIA["A1"]["min_shots"]) & (D["par_bg_photon_mhz"] < NIGHT_BG_MHZ)
    n_cal = int((dark & (D["perp_snr_5km"] >= CRITERIA["E2"]["warn_min"])).sum())
    D["day_cal_files"] = n_cal
    for _, row in D.iterrows():
        r = row.to_dict()
        cloud = r.get("par_cloud_any_m", np.nan)
        cloud = float(cloud) if cloud is not None and np.isfinite(cloud) else np.nan
        blk: List[str] = []
        if r.get("A1") == "FAIL":
            blk.append("ไฟล์ทดสอบ (< 1500 shot)")
        ptop, stop_ = float(r.get("par_top_m", np.nan)), float(r.get("perp_top_m", np.nan))
        lim = []
        # a header that differs from the plan does not make the signal bad, but the
        # instrument constants (glue gain, dead time, C) belong to the planned setting
        plan_txt = ("header ไม่ตรงแผน" + (f" ({r.get('A2_note')})" if r.get("A2_note") else "")
                    if r.get("A2") == "FAIL" else "")
        if plan_txt:
            lim.append(plan_txt)
        if not blk:
            if not (ptop >= NRB_MIN_TOP_M):
                blk.append(_top_reason("∥", ptop if np.isfinite(ptop) else 0.0, cloud))
            elif ptop < NRB_FULL_TOP_M:
                lim.append(_top_reason("∥", ptop, cloud))
        # the overload detector also fires on thin cloud the signal passes through, so a
        # cloud limits the profile (attenuated above it) rather than rejecting it
        cloud_txt = (f"มีเมฆที่ {cloud / 1000:.1f} km (เหนือเมฆสัญญาณถูกลดทอน)"
                     if np.isfinite(cloud) and cloud < NRB_FULL_TOP_M else "")
        if not blk and cloud_txt:
            lim.append(cloud_txt)
        lim += _hits(r, NRB_LIMIT)
        n_state = UNUSABLE if blk else LIMITED if lim else USABLE

        dblk = list(blk)
        dlim = [plan_txt] if plan_txt else []
        if not dblk and n_cal == 0:
            dblk.append("วันนี้ไม่มีไฟล์ที่คาลิเบรต δ ได้ (⊥ ไม่ถึง 4.5–5.5 km) — ต้องใช้ C จากวันอื่น")
        if not dblk:
            if not (stop_ >= DELTA_MIN_TOP_M):
                dblk.append(_top_reason("⊥", stop_ if np.isfinite(stop_) else 0.0, cloud))
            elif stop_ < DELTA_FULL_TOP_M:
                dlim.append(_top_reason("⊥", stop_, cloud))
        if not dblk and 0 < n_cal < MIN_RATIO_FILES:
            dlim.append(f"C ของวันนี้มาจาก {n_cal} ไฟล์เท่านั้น")
        if not dblk and cloud_txt:
            dlim.append(cloud_txt)
        dlim += _hits(r, DELTA_LIMIT)
        d_state = UNUSABLE if dblk else LIMITED if dlim else USABLE
        nrb.append(n_state); dlt.append(d_state)
        nrb_why.append("; ".join(blk or lim)); dlt_why.append("; ".join(dblk or dlim))
    D["use_nrb"], D["use_delta"] = nrb, dlt
    D["use_nrb_reason"], D["use_delta_reason"] = nrb_why, dlt_why
    for col, src in (("acq_start", "start"), ("acq_stop", "stop")):
        D[col] = (pd.to_datetime(D[src], format="%d/%m/%Y %H:%M:%S", errors="coerce")
                  if src in D else pd.NaT)
    return D


def common_limits(D: pd.DataFrame, col: str, share: float = 0.8) -> List[str]:
    """Limitations present in at least ``share`` of the usable files of the day — shown
    once for the day instead of on every row."""
    ok = D[(D["shots"] >= CRITERIA["A1"]["min_shots"]) & (D[f"use_{col}"] == LIMITED)]
    n_use = int(((D["shots"] >= CRITERIA["A1"]["min_shots"]) & (D[f"use_{col}"] != UNUSABLE)).sum())
    if not n_use:
        return []
    counts: Dict[str, int] = {}
    for txt in ok[f"use_{col}_reason"]:
        for part in [x.strip() for x in str(txt).split(";") if x.strip()]:
            if not part.startswith("สัญญาณ"):
                counts[part] = counts.get(part, 0) + 1
    return [k for k, n in counts.items() if n >= share * n_use]


def usable_periods(D: pd.DataFrame, col: str, *, allow_limited: bool = True) -> List[str]:
    """Runs of consecutive files usable for one product, as 'HH:MM–HH:MM' (profile end
    times). Test fires are ignored; a single file gives 'HH:MM'."""
    ok_states = {USABLE, LIMITED} if allow_limited else {USABLE}
    rows = D[D["shots"] >= CRITERIA["A1"]["min_shots"]].sort_values("time")
    runs, cur = [], []
    for _, r in rows.iterrows():
        if r[col] in ok_states:
            cur.append(pd.Timestamp(r["time"]))
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    def _fmt(c):
        # a run that crosses midnight needs its dates, otherwise 09:30-09:45 reads as 15 minutes
        same_day = c[0].date() == c[-1].date()
        if len(c) == 1:
            return f"{c[0]:%H:%M}"
        return f"{c[0]:%H:%M}–{c[-1]:%H:%M}" if same_day else f"{c[0]:%d/%m %H:%M}–{c[-1]:%d/%m %H:%M}"
    return [_fmt(c) for c in runs]


def day_decision(D: pd.DataFrame) -> Dict[str, object]:
    """Counts per product and the usable periods of the day."""
    ok = D[D["shots"] >= CRITERIA["A1"]["min_shots"]]
    out = {"n_files": int(len(ok)), "n_test": int(len(D) - len(ok))}
    for prod, col in (("nrb", "use_nrb"), ("delta", "use_delta")):
        c = ok[col].value_counts()
        out[prod] = {k: int(c.get(k, 0)) for k in (USABLE, LIMITED, UNUSABLE)}
        out[f"{prod}_common"] = common_limits(D, prod)
        out[f"{prod}_periods"] = usable_periods(D, col)
        out[f"{prod}_verdict"] = ("ใช้ไม่ได้ทั้งวัน" if out[prod][UNUSABLE] == len(ok)
                                  else "ใช้ได้ทั้งวัน" if out[prod][UNUSABLE] == 0
                                  else "ใช้ได้บางช่วง")
    return out


# ---- Summary shared by the GUI step, the check sheet and the pipeline ---------------
# (label, criterion id, verdict column). SCORED counts toward the score; CONTEXT is shown
# but not scored (B2: bit meaning unconfirmed; G1: weather, not a fault).
SCORED = [("A1", "A1", "A1"), ("A2", "A2", "A2"),
          ("B1 ∥", "B1", "B1p"), ("B1 ⊥", "B1", "B1s"),
          ("B3 ∥", "B3", "B3p"), ("B3 ⊥", "B3", "B3s"),
          ("C1 ∥", "C1", "C1p"), ("C1 ⊥", "C1", "C1s"),
          ("C2 ∥", "C2", "C2p"), ("C2 ⊥", "C2", "C2s"),
          ("D1 ∥", "D1", "D1p"), ("D1 ⊥", "D1", "D1s"),
          ("D2 ∥", "D2", "D2p"), ("D2 ⊥", "D2", "D2s"),
          ("E1 ∥", "E1", "E1"), ("E2 ⊥", "E2", "E2"), ("F1", "F1", "F1")]
CONTEXT = [("B2 ∥", "B2", "B2p"), ("B2 ⊥", "B2", "B2s"), ("G1", "G1", "G1")]
VERDICT_COLS = [v for _, _, v in SCORED + CONTEXT]
STATUS_PASS, STATUS_PARTIAL, STATUS_FAIL = "ผ่าน", "บางส่วน", "ไม่ผ่าน"


def item_status(v: pd.Series) -> str:
    """Case status of one criterion from its per-file verdicts."""
    P, W, F = int((v == "PASS").sum()), int((v == "WARN").sum()), int((v == "FAIL").sum())
    n = P + W + F
    if n == 0:
        return "NA"
    return STATUS_PASS if P / n >= 0.9 else STATUS_PARTIAL if (P + W) / n >= 0.5 else STATUS_FAIL


def summarize(D: pd.DataFrame, *, min_shots: float = CRITERIA["A1"]["min_shots"]) -> pd.DataFrame:
    """One row per criterion: % PASS night / day, P/W/F counts, status, scored flag.

    A1 counts every file; the rest only files with at least ``min_shots`` (test fires
    are not measurements)."""
    rows = []
    ok = D["shots"] >= min_shots
    for group, scored in ((SCORED, True), (CONTEXT, False)):
        for label, cid, vcol in group:
            sel = slice(None) if vcol == "A1" else ok
            v = D.loc[sel, vcol]
            night = D.loc[sel, "night"].astype(bool)

            def pct(m):
                vv = v[m]
                n = int(vv.isin(["PASS", "WARN", "FAIL"]).sum())
                return float((vv == "PASS").sum()) / n if n else np.nan
            rows.append(dict(item=label, criterion=cid, column=vcol, scored=scored,
                             name=CRITERIA[cid]["name"],
                             night_pass=pct(night), day_pass=pct(~night),
                             n_pass=int((v == "PASS").sum()), n_warn=int((v == "WARN").sum()),
                             n_fail=int((v == "FAIL").sum()), status=item_status(v)))
    return pd.DataFrame(rows)


def score(summary: pd.DataFrame) -> Dict[str, int]:
    s = summary[summary["scored"] & (summary["status"] != "NA")]["status"]
    return dict(passed=int((s == STATUS_PASS).sum()), partial=int((s == STATUS_PARTIAL).sum()),
                failed=int((s == STATUS_FAIL).sum()), total=int(len(s)))


def profile_flags(D: pd.DataFrame) -> pd.DataFrame:
    """Per-file verdicts for the pipeline, keyed by file name: rawqc_<column> plus the
    list of failed scored items (what Step 2 attaches to its QC sheet and Step 6 filters on)."""
    out = pd.DataFrame({"key": D["file"].astype(str)})
    for c in VERDICT_COLS:
        out[f"rawqc_{c}"] = D[c].astype(str).values
    scored_cols = [(label, v) for label, _, v in SCORED]
    out["rawqc_n_fail"] = [sum(D.iloc[i][v] == "FAIL" for _, v in scored_cols) for i in range(len(D))]
    out["rawqc_n_warn"] = [sum(D.iloc[i][v] == "WARN" for _, v in scored_cols) for i in range(len(D))]
    out["rawqc_nrb"] = D["use_nrb"].values
    out["rawqc_delta"] = D["use_delta"].values
    out["rawqc_nrb_reason"] = D["use_nrb_reason"].values
    out["rawqc_delta_reason"] = D["use_delta_reason"].values
    out["rawqc_fails"] = [", ".join(label for label, v in scored_cols if D.iloc[i][v] == "FAIL")
                          for i in range(len(D))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--plan-shots", type=int)
    ap.add_argument("--plan-bin", type=float)
    ap.add_argument("--plan-hv", type=int)
    ap.add_argument("--plan-alt", type=float, help="planned altitude [km]")
    ap.add_argument("--out", help="write the table to this .xlsx")
    a = ap.parse_args()
    D = check_folder(Path(a.folder), plan=dict(shots=a.plan_shots, bin_m=a.plan_bin, hv=a.plan_hv,
                                                  altitude_km=a.plan_alt))
    vcols = [c for c in D.columns if len(c) <= 3 and c[0] in "ABCDEFG"]
    pd.set_option("display.width", 250)
    print(D[["file", "shots"] + vcols].to_string(index=False))
    sm = score(summarize(D))
    print(f"score: {sm['passed']} / {sm['total']} passed ({sm['partial']} partial, {sm['failed']} failed)")
    if a.out:
        D.to_excel(a.out, index=False)


if __name__ == "__main__":
    main()
