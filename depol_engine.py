"""
Depolarization engine for the TR40 polarization LiDAR (Track 2, 532 nm).

The TR40 measures co- and cross-polarized returns SIMULTANEOUSLY with TWO PMTs
on the two output ports of a polarizing beamsplitter cube (Thorlabs CCM1-PBS25-
532, extinction ratio Tp:Ts > 3000:1). The transmitted (p) port is the co-pol
channel; the reflected (s) port is the cross-pol channel. The transmitter beam
is verified linearly polarized. Each PMT is recorded to an ordinary TR40 `.dat`
file:

    co_pol_<DD-MM-YYYY-HH.MM>.dat       (parallel,  P_par,  PMT 1, p port)
    cross_pol_<DD-MM-YYYY-HH.MM>.dat    (perpendicular, P_perp, PMT 2, s port)

Both files share the standard 4-column TR40 ASCII layout, so each is read by
nrb_engine.build_single_profile() unchanged. Because the two channels are
acquired on the same laser shots, the depolarization ratio is free of any
time-gap / atmospheric-change error.

Volume linear depolarization ratio
-----------------------------------
    delta*(R)  =  P_perp(R) / P_par(R)            (raw, background-subtracted)

Because both channels share the same telescope / spatial filter, the range^2,
overlap O(R) and energy terms cancel in the ratio.  The two PMTs have different
gains, leaving a single gain/calibration constant C, found by Rayleigh
(molecular) calibration in a clean, aerosol-free reference region where the true
depolarization is known (delta_mol ~ 0.0044 at 532 nm for a narrowband
receiver):

    C            =  < delta*(R) >_clean  /  delta_mol
    delta_v(R)   =  delta*(R) / C        (calibrated volume depolarization)

Aerosol type is then inferred from delta_v.

References
----------
Cairo F. et al. (1999), Appl. Opt. 38(21), 4425-4432.
Behrendt A., Nakamura T. (2002), Opt. Express 10(16), 805-817.
Freudenthaler V. et al. (2009), Tellus B 61(1), 165-179.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from nrb_engine import (
    build_single_profile, snr_gate_nrb, snr_trusted_top, _looks_like_licel_binary,
)

# Molecular (Rayleigh) linear depolarization ratio at 532 nm.
# ~0.0044 for a narrowband receiver that rejects rotational-Raman wings;
# ~0.0144 for a broadband receiver (Behrendt & Nakamura 2002, Table 1).
DELTA_MOL_532_NARROW = 0.0044
DELTA_MOL_532_BROAD = 0.0144

# Range window (m) used to normalise the cross NRB profile to 0-1. The cross
# channel's global max is usually a far-range R^2-amplified noise spike, so the
# normaliser is the max within the lower troposphere (aerosol/BL region) — this
# keeps the BL peak near 1.0 and the 0-1 profile view readable.
NRB_NORM_MAX_RANGE_M = 6000.0

# Aerosol-type thresholds on the volume linear depolarization ratio (532 nm).
# Boundaries are indicative — overlapping in reality (Freudenthaler 2009,
# Burton et al. 2012). Tuned for Chiang Mai biomass-burning context.
AEROSOL_TYPE_BANDS = [
    (0.00, 0.05, "spherical (urban/haze, hygroscopic)"),
    (0.05, 0.10, "mixed / weakly depolarizing"),
    (0.10, 0.20, "biomass smoke (aged)"),
    (0.20, 0.35, "dust / coarse non-spherical"),
    (0.35, 1.00, "ice crystals / cloud"),
]


def classify_aerosol(delta_v: float) -> str:
    """Map a volume depolarization value to an aerosol-type label."""
    if not np.isfinite(delta_v):
        return "n/a"
    for lo, hi, label in AEROSOL_TYPE_BANDS:
        if lo <= delta_v < hi:
            return label
    return "n/a"


# ---------------------------------------------------------------------------
# File pairing
# ---------------------------------------------------------------------------

# co / cross tokens, longest first. Matched ANYWHERE in the filename so both
# prefix style (co_pol_<t>.dat) and suffix style (<t>co.dat) work. "cross" is
# checked before "co" so a cross file is never mis-read as co.
_CROSS_TOKENS = ("cross_pol_", "crosspol_", "cross_", "cross", "depol_", "depol")
_CO_TOKENS = ("co_pol_", "copol_", "co_", "co")


def _classify_and_key(name: str) -> Tuple[Optional[str], Optional[str]]:
    """Classify a filename as 'co'/'cross' and return a pairing key (the name
    with the matched polarisation token removed once, lower-cased). co and cross
    members of a pair yield the SAME key."""
    low = name.lower()
    for tok in _CROSS_TOKENS:
        if tok in low:
            return "cross", low.replace(tok, "", 1)
    for tok in _CO_TOKENS:
        if tok in low:
            return "co", low.replace(tok, "", 1)
    return None, None


def _parse_ts_from_suffix(suffix: str) -> Optional[pd.Timestamp]:
    """Parse a timestamp from the shared key. Handles full DD-MM-YYYY-HH.MM and
    bare HH.MM / HH:MM (time-only → dummy 1900-01-01 date for sorting)."""
    m = re.search(r"(\d{2})-(\d{2})-(\d{4})-(\d{2})[.:](\d{2})", suffix)
    if m:
        dd, mm, yyyy, hh, mi = map(int, m.groups())
        try:
            return pd.Timestamp(year=yyyy, month=mm, day=dd, hour=hh, minute=mi)
        except ValueError:
            pass
    m2 = re.search(r"(?<!\d)(\d{1,2})[.:](\d{2})(?!\d)", suffix)
    if m2:
        hh, mi = int(m2.group(1)), int(m2.group(2))
        if 0 <= hh <= 23 and 0 <= mi <= 59:
            return pd.Timestamp(year=1900, month=1, day=1, hour=hh, minute=mi)
    return None


# Licel raw filename: <prefix><yy><Mhex><dd><HH>.<MM><SS><ms>, e.g. a2682711.150166
# = 2026-08(hex 8)-27 11:15:01. Month is a single hex digit (Oct-Dec = A/B/C).
_LICEL_RAW_RE = re.compile(r"^[A-Za-z]{1,2}(\d{2})([0-9A-Ca-c])(\d{2})(\d{2})\.(\d{2})(\d{2})(\d{2,})$")


def _parse_licel_raw_name(name: str) -> Optional[pd.Timestamp]:
    """Parse the acquisition timestamp encoded in a Licel raw filename, or None
    if the name is not in that format (e.g. an ASCII ``.dat`` export)."""
    m = _LICEL_RAW_RE.match(name)
    if not m:
        return None
    yy, mhex, dd, hh, mm, ss, _ms = m.groups()
    try:
        return pd.Timestamp(year=2000 + int(yy), month=int(mhex, 16), day=int(dd),
                            hour=int(hh), minute=int(mm), second=int(ss))
    except ValueError:
        return None


def _lidar_ts_key(f: Path) -> Tuple[pd.Timestamp, str]:
    """(timestamp, grouping key) for one lidar file, handling both ASCII ``.dat``
    exports and extensionless Licel raw files. Raw files carry no co/cross token,
    so their full (unique) filename is the key and the encoded time the timestamp
    — Path.stem would collide (three 11:xx files all stem to 'a2682711')."""
    raw_ts = _parse_licel_raw_name(f.name)
    if raw_ts is not None:
        return raw_ts, f.name
    key = _classify_and_key(f.name)[1] or f.stem
    ts = _parse_ts_from_suffix(key) or _parse_ts_from_suffix(f.stem) or pd.Timestamp(1970, 1, 1)
    return ts, key


def _glob_lidar_files(folder: Path, pattern: str) -> List[Path]:
    """Files in ``folder`` matching ``pattern``; when the pattern matches nothing
    (e.g. the default ``*.dat`` against a folder of extensionless raw files) fall
    back to every file that sniffs as a Licel raw binary. This keeps existing
    ``.dat`` folders byte-for-byte unchanged while letting raw folders load."""
    folder = Path(folder)
    matched = sorted(folder.glob(pattern))
    if matched:
        return matched
    return [f for f in sorted(folder.iterdir()) if f.is_file() and _looks_like_licel_binary(f)]


def pair_co_cross_files(
    co_folder: Path,
    cross_folder: Optional[Path] = None,
    *,
    pattern: str = "*.dat",
    pair_tol_min: float = 30.0,
    **_ignored,
) -> List[Tuple[pd.Timestamp, Path, Path, str]]:
    """
    Pair co- and cross-pol .dat files. Works for THREE naming conventions:
      • prefix style : co_pol_<t>.dat  ↔ cross_pol_<t>.dat
      • suffix style : <t>co.dat       ↔ <t>cross.dat
      • folder style : par/<t>.dat     ↔ perp/<t>.dat  (channel = the folder,
                       filename carries no co/cross token, e.g. par/10.25.dat)
    and whether co/cross live in one folder or two.

    When co_folder and cross_folder are TWO DISTINCT folders the folder itself
    names the channel, so filename tokens are not required.

    Matching: exact key (token-removed name) first, then nearest acquisition
    time within `pair_tol_min`, then a 1:1 fallback when each side has one file.
    Returns a time-sorted list of (timestamp, co_path, cross_path, key) tuples.
    """
    co_folder = Path(co_folder)
    cross_folder = Path(cross_folder) if cross_folder else co_folder
    # Two DISTINCT folders => the folder names the channel; filenames need not
    # carry a co/cross token. Same folder => fall back to token classification.
    try:
        two_folders = cross_folder.resolve() != co_folder.resolve()
    except OSError:
        two_folders = str(cross_folder) != str(co_folder)

    def _collect(folder, want):
        items = []
        for f in _glob_lidar_files(folder, pattern):
            raw_ts = _parse_licel_raw_name(f.name)
            if two_folders:
                # Channel given by the folder; raw files have no token, so use the
                # full (unique) name as the key (f.stem would collide).
                kind, key = want, (f.name if raw_ts is not None else f.stem)
            else:
                kind, key = _classify_and_key(f.name)
            if kind == want:
                ts = raw_ts or _parse_ts_from_suffix(key or "") or _parse_ts_from_suffix(f.stem)
                items.append((ts, key, f))
        return items

    co_items = _collect(co_folder, "co")
    cr_items = _collect(cross_folder, "cross")
    if not co_items or not cr_items:
        return []

    pairs: List[Tuple[pd.Timestamp, Path, Path, str]] = []

    # 1) exact key match
    cr_by_key: Dict[str, list] = {}
    for ts, key, p in cr_items:
        cr_by_key.setdefault(key, []).append((ts, p))
    remaining_co = []
    for ts_co, key, co_p in co_items:
        bucket = cr_by_key.get(key)
        if bucket:
            ts_cr, cr_p = bucket.pop(0)
            pairs.append((ts_co or ts_cr or pd.Timestamp(1970, 1, 1), co_p, cr_p, key))
        else:
            remaining_co.append((ts_co, key, co_p))

    # 2) nearest-time for leftovers
    remaining_cross = [(ts, p) for lst in cr_by_key.values() for ts, p in lst]
    tol = pd.Timedelta(minutes=float(pair_tol_min))
    for ts_co, key, co_p in remaining_co:
        if ts_co is None:
            continue
        best_j, best_dt = None, None
        for j, (ts_cr, _p) in enumerate(remaining_cross):
            if ts_cr is None:
                continue
            dt = abs(ts_cr - ts_co)
            if dt <= tol and (best_dt is None or dt < best_dt):
                best_dt, best_j = dt, j
        if best_j is not None:
            _, cr_p = remaining_cross.pop(best_j)
            pairs.append((ts_co, co_p, cr_p, key))

    # 3) 1:1 fallback when nothing matched but exactly one file each
    if not pairs and len(co_items) == 1 and len(cr_items) == 1:
        ts_co, key, co_p = co_items[0]
        ts_cr, _, cr_p = cr_items[0]
        pairs.append((ts_co or ts_cr or pd.Timestamp(1970, 1, 1), co_p, cr_p, key or "pair"))

    pairs.sort(key=lambda t: t[0])
    return pairs


# Subfolder names that identify a polarisation channel (case-insensitive).
_PAR_DIR_NAMES = ("par", "parallel", "co", "co_pol", "copol", "p")
_PERP_DIR_NAMES = ("perp", "perpendicular", "cross", "cross_pol", "crosspol", "depol", "s")


def _find_channel_subdirs(folder: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """If `folder` holds par/perp (parallel/perpendicular) subfolders, return
    (parallel_dir, perpendicular_dir); otherwise (None, None)."""
    if not folder.is_dir():
        return None, None
    subs = {p.name.lower(): p for p in folder.iterdir() if p.is_dir()}
    par = next((subs[n] for n in _PAR_DIR_NAMES if n in subs), None)
    perp = next((subs[n] for n in _PERP_DIR_NAMES if n in subs), None)
    return par, perp


def resolve_co_cross_pairs(
    co_input: Path,
    cross_input: Optional[Path] = None,
    *,
    pattern: str = "*.dat",
    **_ignored,
) -> List[Tuple[pd.Timestamp, Path, Path, str]]:
    """
    Accept FILES or FOLDERS for co/cross and return paired tuples.
      • both files   -> one direct pair (channel = argument position)
      • two folders  -> pair_co_cross_files (channel = the folder)
      • ONE folder holding par/perp subfolders -> auto-split into the two channels
      • file given where a folder is expected -> use its parent folder
    """
    co_input = Path(co_input)
    cross_input = Path(cross_input) if cross_input else co_input

    if co_input.is_file() and cross_input.is_file():
        ts, key = _lidar_ts_key(co_input)
        return [(ts, co_input, cross_input, key)]

    co_folder = co_input if co_input.is_dir() else co_input.parent
    cross_folder = cross_input if cross_input.is_dir() else cross_input.parent

    # A single parent folder that holds par/perp subfolders -> use them as the
    # two channels (so the user can just pick the parent).
    try:
        same_folder = co_folder.resolve() == cross_folder.resolve()
    except OSError:
        same_folder = str(co_folder) == str(cross_folder)
    if same_folder and not list(co_folder.glob(pattern)):
        par_dir, perp_dir = _find_channel_subdirs(co_folder)
        if par_dir is not None and perp_dir is not None:
            co_folder, cross_folder = par_dir, perp_dir

    return pair_co_cross_files(co_folder, cross_folder, pattern=pattern)


# ---------------------------------------------------------------------------
# Core depolarization computation
# ---------------------------------------------------------------------------

def compute_depolarization(
    r_m: np.ndarray,
    P_par: np.ndarray,
    P_perp: np.ndarray,
    *,
    delta_mol: float = DELTA_MOL_532_NARROW,
    cal_rmin_m: float = 4500.0,
    cal_rmax_m: float = 5500.0,
    min_par_signal: float = 1e-6,
) -> Dict[str, object]:
    """
    Compute the calibrated volume depolarization ratio profile.

    Parameters
    ----------
    r_m              : range grid [m]
    P_par, P_perp    : background-subtracted co- / cross-pol signals (same grid)
    delta_mol        : molecular depolarization in the clean region (532 nm)
    cal_rmin_m/max_m : clean-air calibration window [m]
    min_par_signal   : floor below which the ratio is set NaN (noise guard)

    Returns dict with keys:
      delta_star : raw ratio P_perp / P_par
      delta_v    : calibrated volume depolarization
      C          : calibration constant
      cal_ratio  : mean raw ratio in the calibration window
      n_cal      : number of bins used for calibration
    """
    r = np.asarray(r_m, float)
    par = np.asarray(P_par, float)
    perp = np.asarray(P_perp, float)

    with np.errstate(divide="ignore", invalid="ignore"):
        delta_star = np.where(par > float(min_par_signal), perp / par, np.nan)

    cal = (r >= float(cal_rmin_m)) & (r <= float(cal_rmax_m)) & np.isfinite(delta_star)
    n_cal = int(np.sum(cal))
    if n_cal >= 3:
        cal_ratio = float(np.nanmean(delta_star[cal]))
        C = cal_ratio / float(delta_mol) if delta_mol > 0 else np.nan
    else:
        cal_ratio = np.nan
        C = np.nan

    if np.isfinite(C) and C > 0:
        delta_v = delta_star / C
    else:
        delta_v = np.full_like(r, np.nan)

    return {
        "delta_star": delta_star,
        "delta_v": delta_v,
        "C": C,
        "cal_ratio": cal_ratio,
        "n_cal": n_cal,
    }


def _glue_qc(prof_meta: Dict[str, object], prefix: str) -> Dict[str, object]:
    """Per-channel analog<->photon glue-fit QC fields, prefixed 'par_'/'perp_'
    (mirrors the Step-2 NRB QC: blend_r1/r2_used_m, slope, offset, fit_quality_r2,
    n_toggle_points, glue_mode)."""
    keys = ("blend_r1_used_m", "blend_r2_used_m", "slope", "offset",
            "fit_quality_r2", "fit_rmse", "n_toggle_points",
            "auto_blend_ok", "toggle_mode", "glue_mode",
            "day_night_glue_pick")
    return {f"{prefix}_{k}": prof_meta.get(k) for k in keys}


def _signal_diag(prof_df: pd.DataFrame, prof_meta: Dict[str, object]) -> Dict[str, object]:
    """Extract the raw/processed signal columns (+ glue markers) of ONE channel
    for the signal-inspection plots, mirroring Step 2's per-profile views."""
    def col(name):
        return (prof_df[name].to_numpy(float) if name in prof_df.columns
                else np.full(len(prof_df), np.nan))
    return {
        "range_m": col("range_m"),
        "analog_mV": col("analog_mV"),
        "photon_MHz": col("photon_MHz"),
        "photon_deadtime_corr_MHz": col("photon_deadtime_corr_MHz"),
        "analog_scaled_MHz": col("analog_scaled_MHz"),
        "glued_profile_MHz": col("glued_profile_MHz"),
        "nrb": col("nrb"),
        "meta": {k: prof_meta.get(k) for k in (
            "min_toggle_rate", "max_toggle_rate", "blend_r1_used_m",
            "blend_r2_used_m", "glue_mode", "gluing_mode")},
    }


def build_depol_for_pair(
    co_path: Path,
    cross_path: Path,
    *,
    delta_mol: float = DELTA_MOL_532_NARROW,
    cal_rmin_m: float = 4500.0,
    cal_rmax_m: float = 5500.0,
    afterpulse_co: Optional[np.ndarray] = None,
    afterpulse_cross: Optional[np.ndarray] = None,
    snr_min: float = 3.0,
    snr_gate: bool = True,
    co_channel: str = "parallel",
    cross_channel: str = "parallel",
    **nrb_kwargs,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Process one co/cross pair into a depolarization profile DataFrame.

    `nrb_kwargs` are forwarded identically to build_single_profile for BOTH
    channels (overlap_O_R, energy_mj, etc. are SHARED — same telescope/shot).
    Afterpulse is the ONE per-channel correction, so co and cross use their own
    `afterpulse_co` / `afterpulse_cross` arrays. Returns (df, meta).

    delta is taken from the afterpulse-corrected signal (glued_afterpulse_corr):
    overlap / R^2 / energy cancel in the cross/co ratio, but afterpulse is
    channel-specific and does NOT cancel, so it must be removed before dividing.
    The per-channel NRB columns carry the FULL correction chain (overlap, R^2,
    energy) in PHYSICAL units (NOT max-normalised) so co/cross share one scale
    and are comparable to MPL's physical-unit NRB.
    """
    # Default to the simplest symmetric processing unless overridden.
    nrb_kwargs.setdefault("gluing_mode", "photon_only")
    nrb_kwargs.setdefault("bg_mode", "pretrigger")
    nrb_kwargs.setdefault("pretrigger_bins", 1024)

    # co_channel / cross_channel select the polarization column block within each
    # file. Separate par/perp files (old workflow) are each single-channel, so both
    # default to "parallel". A 2-PMT file passes the SAME path for both with
    # co_channel="parallel", cross_channel="perpendicular".
    nrb_kwargs.pop("channel", None)
    co_df, co_meta = build_single_profile(
        Path(co_path), afterpulse_A_R=afterpulse_co, channel=co_channel, **nrb_kwargs)
    cr_df, cr_meta = build_single_profile(
        Path(cross_path), afterpulse_A_R=afterpulse_cross, channel=cross_channel, **nrb_kwargs)

    r = co_df["range_m"].to_numpy(float)
    r_cr = cr_df["range_m"].to_numpy(float)
    # delta from the BG- and afterpulse-corrected signal (before overlap/R^2/
    # energy, which cancel in the ratio anyway).
    P_par = co_df["glued_afterpulse_corr_MHz"].to_numpy(float)
    P_perp = cr_df["glued_afterpulse_corr_MHz"].to_numpy(float)
    # Fully-corrected NRB per channel in PHYSICAL units (NOT max-normalised) —
    # like MPL's (counts/us/uJ)*km^2. Use nrb_final, not nrb (=nrb_final/max),
    # because ÷max on the weak cross channel divides by a far-range R^2-amplified
    # noise spike, making the scale meaningless. Real units keep co/cross on a
    # common physical scale so cross/co is directly readable.
    nrb_co = co_df["nrb_final"].to_numpy(float)
    nrb_cr = cr_df["nrb_final"].to_numpy(float)
    # Normalised co NRB (0-1) is set below (both snr_gate branches), never from
    # co_df["nrb"] (global-max) — in daytime the global max is a far-range
    # R^2-amplified noise spike that squashes the real boundary layer.
    snr_co = co_df["snr"].to_numpy(float)
    snr_cr = cr_df["snr"].to_numpy(float)
    co_trusted_top = snr_trusted_top(r, snr_co, snr_min)

    # Align cross onto the co range grid if they ever differ.
    if len(r_cr) != len(r) or not np.allclose(r_cr, r, atol=1e-6):
        P_perp = np.interp(r, r_cr, P_perp, left=np.nan, right=np.nan)
        nrb_cr = np.interp(r, r_cr, nrb_cr, left=np.nan, right=np.nan)
        snr_cr = np.interp(r, r_cr, snr_cr, left=np.nan, right=np.nan)

    res = compute_depolarization(
        r, P_par, P_perp,
        delta_mol=delta_mol, cal_rmin_m=cal_rmin_m, cal_rmax_m=cal_rmax_m,
    )

    # Mask delta where the WEAK cross channel is too noisy (cross SNR < snr_min).
    # Calibration C is computed first (above) on the raw ratio, so the mask only
    # affects the reported/plotted delta — not the calibration itself.
    delta_v = np.asarray(res["delta_v"], float).copy()
    trust = np.isfinite(snr_cr) & (snr_cr >= float(snr_min))
    delta_v[~trust] = np.nan
    n_trust = int(np.sum(trust & np.isfinite(res["delta_v"])))

    # Normalised cross NRB — divided by the cross channel's max WITHIN the
    # aerosol/boundary-layer range window (rmin..NRB_NORM_MAX_RANGE_M), NOT the
    # global max. Because nrb includes x R^2, far ranges get amplified: the cross
    # channel's global max is a far-range (~12 km) noise/R^2 spike that squashes
    # the real BL signal to ~0.1, so a 0-1 profile view of cross looks empty
    # (SNR-trusting alone does not help — SNR can stay >3 out to ~10 km). Capping
    # the normalising max to the lower troposphere puts the BL peak near 1.0 so
    # cross is visible and shape-comparable to co / MPL. Falls back to the global
    # max only if the window is empty.
    cr_trusted_top = snr_trusted_top(r, snr_cr, snr_min)
    if snr_gate:
        # Normalise within the SNR-trusted lower troposphere for BOTH channels
        # (replaces the 0-6 km cap for cross and the global-max normalise for co),
        # so the boundary layer sits near 1.0 instead of a far-range R² noise
        # spike. mask_above_trusted=False keeps the FULL profile visible — Step 5's
        # Prototype-vs-Mini-MPL overlay needs the whole curve, not a line cut short
        # at the trusted top.
        nrb_co_norm, co_trusted_top = snr_gate_nrb(r, nrb_co, snr_co, snr_min,
                                                   mask_above_trusted=False)
        nrb_cross_norm, cr_trusted_top = snr_gate_nrb(r, nrb_cr, snr_cr, snr_min,
                                                      mask_above_trusted=False)
    else:
        # snr_gate off: still normalise BOTH channels by the max within the lower
        # troposphere (r <= NRB_NORM_MAX_RANGE_M), NOT the global max — the daytime
        # global max is a far-range R^2-amplified noise spike that would squash the
        # real boundary layer for the parallel channel too. Falls back to the
        # global max only when the window is empty.
        def _cap_norm(y):
            win = np.isfinite(y) & (r <= NRB_NORM_MAX_RANGE_M)
            if win.any() and np.isfinite(y[win]).any():
                mx = float(np.nanmax(y[win]))
            else:
                mx = float(np.nanmax(y)) if np.any(np.isfinite(y)) else np.nan
            return (y / mx if (np.isfinite(mx) and mx > 0) else np.full_like(y, np.nan))
        nrb_co_norm = _cap_norm(nrb_co)
        nrb_cross_norm = _cap_norm(nrb_cr)

    df = pd.DataFrame({
        "range_m": r,
        "P_par_MHz": P_par,
        "P_perp_MHz": P_perp,
        "nrb_co": nrb_co,
        "nrb_cross": nrb_cr,
        "nrb_co_norm": nrb_co_norm,
        "nrb_cross_norm": nrb_cross_norm,
        "snr_co": snr_co,
        "snr_cross": snr_cr,
        "delta_star": res["delta_star"],
        "delta_v": delta_v,
        "aerosol_type": [classify_aerosol(d) for d in delta_v],
    })
    meta = {
        "C": res["C"],
        "cal_ratio": res["cal_ratio"],
        "n_cal": res["n_cal"],
        "delta_mol": float(delta_mol),
        "cal_rmin_m": float(cal_rmin_m),
        "cal_rmax_m": float(cal_rmax_m),
        "snr_min": float(snr_min),
        "snr_gate": 1.0 if snr_gate else 0.0,
        "co_trusted_range_m": float(co_trusted_top),
        "cross_trusted_range_snrmin_m": float(cr_trusted_top),
        "n_delta_trusted": n_trust,
        "cross_trusted_range_m": float(cr_meta.get("trusted_range_snr3_m", np.nan)),
        "bg_par_mhz": float(co_meta.get("bg_glued", np.nan)),
        "bg_perp_mhz": float(cr_meta.get("bg_glued", np.nan)),
        # Per-channel analog<->photon glue-fit QC (mirrors the Step-2 NRB QC), so
        # the depol workbook shows how each channel was glued.
        **_glue_qc(co_meta, "par"),
        **_glue_qc(cr_meta, "perp"),
        # Per-channel raw/processed signal diagnostics (analog, glue, NRB) so the
        # GUI can show the same signal views as Step 2 for each channel.
        "diag": {
            "par":  _signal_diag(co_df, co_meta),
            "perp": _signal_diag(cr_df, cr_meta),
        },
    }
    return df, meta


# ---------------------------------------------------------------------------
# Single-channel (no pair) processing
# ---------------------------------------------------------------------------
# The prototype currently measures ONE polarisation per day (single PMT, analyser
# swapped between days), so a simultaneous co/cross pair may not exist. These
# helpers let Step 3 process just parallel OR just perpendicular: no δ (needs
# both), but the channel's NRB / SNR / signal diagnostics + figures are produced.

def _collect_single_channel(inp: Path, pattern: str = "*.dat"):
    """List (timestamp, key, file) for ONE channel folder (or a single file).
    Handles both ASCII ``.dat`` exports and extensionless Licel raw files."""
    inp = Path(inp)
    if inp.is_file():
        ts, key = _lidar_ts_key(inp)
        return [(ts, key, inp)]
    folder = inp if inp.is_dir() else inp.parent
    items = []
    for f in _glob_lidar_files(folder, pattern):
        ts, key = _lidar_ts_key(f)
        items.append((ts, key, f))
    return items


def build_single_channel_for_file(
    path: Path,
    channel: str,          # "par" or "perp"
    *,
    afterpulse: Optional[np.ndarray] = None,
    snr_min: float = 3.0,
    snr_gate: bool = True,
    **nrb_kwargs,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Process ONE channel's file alone (no pair → no depolarization).

    Returns the SAME DataFrame/meta shape as build_depol_for_pair, but only the
    channel's columns are populated (the co_* set for "par", the cross_* set for
    "perp"); the other channel and delta_v are NaN. This lets the daily builder,
    the workbook writer and the Step-6 figure tools stay unchanged."""
    nrb_kwargs.setdefault("gluing_mode", "photon_only")
    nrb_kwargs.setdefault("bg_mode", "pretrigger")
    nrb_kwargs.setdefault("pretrigger_bins", 1024)

    prof_df, prof_meta = build_single_profile(
        Path(path), afterpulse_A_R=afterpulse, **nrb_kwargs)
    r = prof_df["range_m"].to_numpy(float)
    nrb_final = prof_df["nrb_final"].to_numpy(float)
    snr = prof_df["snr"].to_numpy(float)
    if snr_gate:
        nrb_norm, trusted = snr_gate_nrb(r, nrb_final, snr, snr_min)
    else:
        nrb_norm = prof_df["nrb"].to_numpy(float)
        trusted = snr_trusted_top(r, snr, snr_min)

    is_par = (channel == "par")
    nan = np.full(len(r), np.nan)
    P = prof_df["glued_afterpulse_corr_MHz"].to_numpy(float)
    df = pd.DataFrame({
        "range_m": r,
        "P_par_MHz":     P if is_par else nan,
        "P_perp_MHz":    nan if is_par else P,
        "nrb_co":        nrb_final if is_par else nan,
        "nrb_cross":     nan if is_par else nrb_final,
        "nrb_co_norm":   nrb_norm if is_par else nan,
        "nrb_cross_norm": nan if is_par else nrb_norm,
        "snr_co":        snr if is_par else nan,
        "snr_cross":     nan if is_par else snr,
        "delta_star":    nan,
        "delta_v":       nan,
        "aerosol_type":  ["Undetected"] * len(r),
    })
    meta = {
        "C": np.nan, "cal_ratio": np.nan, "n_cal": 0,
        "delta_mol": np.nan, "cal_rmin_m": np.nan, "cal_rmax_m": np.nan,
        "snr_min": float(snr_min), "snr_gate": 1.0 if snr_gate else 0.0,
        "co_trusted_range_m": float(trusted) if is_par else np.nan,
        "cross_trusted_range_snrmin_m": np.nan if is_par else float(trusted),
        "n_delta_trusted": 0,
        "cross_trusted_range_m": (np.nan if is_par
                                  else float(prof_meta.get("trusted_range_snr3_m", np.nan))),
        "bg_par_mhz": float(prof_meta.get("bg_glued", np.nan)) if is_par else np.nan,
        "bg_perp_mhz": np.nan if is_par else float(prof_meta.get("bg_glued", np.nan)),
        # Glue-fit QC on the channel that was actually processed; the other stays NaN.
        **_glue_qc(prof_meta if is_par else {}, "par"),
        **_glue_qc({} if is_par else prof_meta, "perp"),
        "single_channel": channel,
        "diag": {
            "par":  _signal_diag(prof_df, prof_meta) if is_par else {},
            "perp": {} if is_par else _signal_diag(prof_df, prof_meta),
        },
    }
    return df, meta


# ---------------------------------------------------------------------------
# Daily batch processor
# ---------------------------------------------------------------------------

def build_daily_depol_from_folders(
    co_folder: Path,
    cross_folder: Optional[Path],
    out_path: Path,
    *,
    delta_mol: float = DELTA_MOL_532_NARROW,
    cal_rmin_m: float = 4500.0,
    cal_rmax_m: float = 5500.0,
    pattern: str = "*.dat",
    date_str: str = "",
    afterpulse_co: Optional[np.ndarray] = None,
    afterpulse_cross: Optional[np.ndarray] = None,
    snr_min: float = 3.0,
    snr_gate: bool = True,
    single_channel: Optional[str] = None,
    dual_channel_file: bool = False,
    strict: bool = False,
    logger: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
    **nrb_kwargs,
) -> Dict[str, pd.DataFrame]:
    """
    Pair every co/cross file (accepts FILES or FOLDERS), compute
    depolarization, and write a workbook.

    `single_channel` = "par" or "perp" processes ONE channel alone (no pairing,
    no δ) — for experiments that record parallel and perpendicular on separate
    days. In that mode the given folder is that channel's data.

    Returns a dict of DataFrames:
      "delta_v"      : Range(m) + one column per timestamp (volume depol)
      "aerosol_type" : Range(m) + one column per timestamp (type label)
      "nrb_co"       : Range(m) + one column per timestamp (co-pol NRB, physical units)
      "nrb_cross"    : Range(m) + one column per timestamp (cross-pol NRB, physical units)
      "qc"           : per-profile calibration constants and metadata
    """
    single = single_channel if single_channel in ("par", "perp") else None
    if single is not None:
        # One channel only: enumerate that channel's folder (par → co_folder,
        # perp → cross_folder, falling back to co_folder). No pairing.
        src = cross_folder if (single == "perp" and cross_folder is not None) else co_folder
        items = _collect_single_channel(src, pattern=pattern)
        if not items:
            raise ValueError(
                f"No .dat files found for single-channel ({single}) run in: {src}")
        pairs = [(ts, f, f, key) for (ts, key, f) in items]
    elif dual_channel_file:
        # 2-PMT: ONE folder, each .dat holds both channels (parallel cols 0-3,
        # perpendicular cols 4-7). co and cross are the SAME file.
        items = _collect_single_channel(co_folder, pattern=pattern)
        if not items:
            raise ValueError(f"No .dat files found for 2-PMT run in: {co_folder}")
        pairs = [(ts, f, f, key) for (ts, key, f) in items]
    else:
        pairs = resolve_co_cross_pairs(co_folder, cross_folder, pattern=pattern)
        if not pairs:
            raise ValueError(
                "No matching co/cross .dat pairs found. Check filenames "
                "(expected co_pol_*.dat and cross_pol_*.dat with identical suffix). "
                "For a one-channel experiment use single_channel='par'/'perp'."
            )

    # Re-date the (time-only) timestamps to the given day, keeping time-of-day.
    # Filenames like "21.21co.dat" carry no date → default to 1900-01-01; setting
    # date_str stamps the real day so the workbook lines up with Step 2 / ALT.
    if str(date_str).strip():
        try:
            base_date = pd.to_datetime(date_str).normalize()
            pairs = [(base_date + (ts - ts.normalize()), co, cr, key)
                     for (ts, co, cr, key) in pairs]
        except Exception:
            pass

    ref_r: Optional[np.ndarray] = None
    ts_list: List[pd.Timestamp] = []
    dv_cols: List[np.ndarray] = []
    type_cols: List[np.ndarray] = []
    nrb_co_cols: List[np.ndarray] = []
    nrb_cr_cols: List[np.ndarray] = []
    nrb_co_norm_cols: List[np.ndarray] = []
    nrb_cr_norm_cols: List[np.ndarray] = []
    snr_co_cols: List[np.ndarray] = []
    snr_cr_cols: List[np.ndarray] = []
    diag_list: List[Dict[str, object]] = []   # per-profile signal diagnostics
    qc_rows: List[Dict[str, object]] = []
    total = len(pairs)

    for idx, (ts, co_path, cr_path, key) in enumerate(pairs, start=1):
        if logger:
            if single is not None:
                logger(f"[{idx}/{total}] {key}  ({single}={co_path.name})")
            else:
                logger(f"[{idx}/{total}] {key}  (co={co_path.name}, cross={cr_path.name})")
        try:
            if single is not None:
                df, meta = build_single_channel_for_file(
                    co_path, single,
                    afterpulse=(afterpulse_co if single == "par" else afterpulse_cross),
                    snr_min=snr_min, snr_gate=snr_gate, **nrb_kwargs)
            else:
                df, meta = build_depol_for_pair(
                    co_path, cr_path,
                    delta_mol=delta_mol, cal_rmin_m=cal_rmin_m, cal_rmax_m=cal_rmax_m,
                    afterpulse_co=afterpulse_co, afterpulse_cross=afterpulse_cross,
                    snr_min=snr_min, snr_gate=snr_gate,
                    co_channel="parallel",
                    cross_channel="perpendicular" if dual_channel_file else "parallel",
                    **nrb_kwargs,
                )
        except Exception as e:
            if strict:
                raise
            qc_rows.append({"key": key, "time": ts, "status": f"error: {e}"})
            if logger:
                logger(f"   error: {e}")
            if progress_cb:
                progress_cb(100.0 * idx / total)
            continue

        r = df["range_m"].to_numpy(float)
        dv = df["delta_v"].to_numpy(float)
        tp = df["aerosol_type"].to_numpy(object)
        nco = df["nrb_co"].to_numpy(float)
        ncr = df["nrb_cross"].to_numpy(float)
        ncn = df["nrb_co_norm"].to_numpy(float)
        ncrn = df["nrb_cross_norm"].to_numpy(float)
        sco = df["snr_co"].to_numpy(float)
        scr = df["snr_cross"].to_numpy(float)

        if ref_r is None:
            ref_r = r.copy()
        elif len(r) != len(ref_r) or np.nanmax(np.abs(r - ref_r)) > 1e-6:
            dv = np.interp(ref_r, r, dv, left=np.nan, right=np.nan)
            nco = np.interp(ref_r, r, nco, left=np.nan, right=np.nan)
            ncr = np.interp(ref_r, r, ncr, left=np.nan, right=np.nan)
            ncn = np.interp(ref_r, r, ncn, left=np.nan, right=np.nan)
            ncrn = np.interp(ref_r, r, ncrn, left=np.nan, right=np.nan)
            sco = np.interp(ref_r, r, sco, left=np.nan, right=np.nan)
            scr = np.interp(ref_r, r, scr, left=np.nan, right=np.nan)
            tp = np.array([classify_aerosol(v) for v in dv], dtype=object)

        ts_list.append(ts)
        diag_list.append(meta.get("diag", {}))
        dv_cols.append(dv.astype(float))
        type_cols.append(tp)
        nrb_co_cols.append(nco.astype(float))
        nrb_cr_cols.append(ncr.astype(float))
        nrb_co_norm_cols.append(ncn.astype(float))
        nrb_cr_norm_cols.append(ncrn.astype(float))
        snr_co_cols.append(sco.astype(float))
        snr_cr_cols.append(scr.astype(float))
        _glue_keys = ("blend_r1_used_m", "blend_r2_used_m", "slope", "offset",
                      "fit_quality_r2", "fit_rmse", "n_toggle_points",
                      "auto_blend_ok", "toggle_mode", "glue_mode", "day_night_glue_pick")
        qc_rows.append({
            "key": key, "time": ts, "status": "ok",
            "C": meta["C"], "cal_ratio": meta["cal_ratio"], "n_cal": meta["n_cal"],
            "snr_min": meta.get("snr_min", snr_min),
            "n_delta_trusted": meta.get("n_delta_trusted", np.nan),
            "cross_trusted_range_m": meta.get("cross_trusted_range_m", np.nan),
            "bg_par_mhz": meta["bg_par_mhz"], "bg_perp_mhz": meta["bg_perp_mhz"],
            # Per-channel glue-fit QC (mirrors Step-2 NRB QC)
            **{f"par_{k}": meta.get(f"par_{k}") for k in _glue_keys},
            **{f"perp_{k}": meta.get(f"perp_{k}") for k in _glue_keys},
        })
        if progress_cb:
            progress_cb(100.0 * idx / total)

    if ref_r is None or not ts_list:
        raise ValueError("No valid profiles processed."
                         if single is not None else "No valid co/cross pairs processed.")

    def _frame(cols):
        d = pd.DataFrame(np.column_stack(cols), columns=ts_list)
        d.insert(0, "Range(m)", ref_r)
        return d

    df_delta_v = _frame(dv_cols)
    df_type = pd.DataFrame(np.column_stack(type_cols), columns=ts_list)
    df_type.insert(0, "Range(m)", ref_r)
    df_nrb_co = _frame(nrb_co_cols)
    df_nrb_cr = _frame(nrb_cr_cols)
    df_nrb_profile = _frame(nrb_co_norm_cols)  # normalised co NRB = Step-2 product
    df_nrb_cr_norm = _frame(nrb_cr_norm_cols)  # cross ÷ its own max (0-1, per-channel)
    df_snr_co = _frame(snr_co_cols)
    df_snr_cr = _frame(snr_cr_cols)
    df_qc = pd.DataFrame(qc_rows)

    # Per-channel raw/processed SIGNAL sheets (range x time) so Step 6 can draw
    # the prototype signal figures (Analog / Photon / Glue) from the workbook.
    def _sig_frame(channel: str, key: str) -> pd.DataFrame:
        cols = []
        for diag in diag_list:
            d = ((diag or {}).get(channel) or {})
            rr = np.asarray(d.get("range_m", ref_r), float)
            vv = np.asarray(d.get(key, np.full(len(rr), np.nan)), float)
            if len(rr) == len(ref_r) and np.allclose(rr, ref_r, equal_nan=True):
                col = vv
            else:
                col = np.interp(ref_r, rr, vv, left=np.nan, right=np.nan)
            cols.append(np.asarray(col, float))
        fr = pd.DataFrame(np.column_stack(cols), columns=ts_list) if cols else pd.DataFrame()
        fr.insert(0, "Range(m)", ref_r)
        return fr

    sig_sheets = {
        "Analog_par":        _sig_frame("par",  "analog_mV"),
        "Analog_perp":       _sig_frame("perp", "analog_mV"),
        "Photon_par":        _sig_frame("par",  "photon_MHz"),
        "Photon_perp":       _sig_frame("perp", "photon_MHz"),
        "PhotonDT_par":      _sig_frame("par",  "photon_deadtime_corr_MHz"),
        "PhotonDT_perp":     _sig_frame("perp", "photon_deadtime_corr_MHz"),
        "AnalogScaled_par":  _sig_frame("par",  "analog_scaled_MHz"),
        "AnalogScaled_perp": _sig_frame("perp", "analog_scaled_MHz"),
        "Glued_par":         _sig_frame("par",  "glued_profile_MHz"),
        "Glued_perp":        _sig_frame("perp", "glued_profile_MHz"),
    }

    params = pd.DataFrame([{
        "engine": "depol_engine v1",
        "n_pairs": int(len(ts_list)),
        "mode": f"single_channel:{single}" if single is not None else "co_cross_pair",
        "delta_mol": float(delta_mol),
        "cal_rmin_m": float(cal_rmin_m),
        "cal_rmax_m": float(cal_rmax_m),
        "snr_min": float(snr_min),
        "snr_gate": "applied" if snr_gate else "not_applied",
        "pattern": pattern,
        "note": (f"Single-channel ({single}) run — no pairing, delta_v is NaN; "
                 f"the {single} NRB/SNR live in the "
                 f"{'co' if single == 'par' else 'cross'} sheets."
                 if single is not None else
                 "Simultaneous co/cross (2 PMT); ratio cancels R^2/overlap/energy; "
                 "molecular-calibrated C; delta masked where cross SNR < snr_min."),
    }])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        # "NRB profile" + "SNR" mirror the Step-2 workbook so ALT / Fernald can
        # read this depol workbook directly (no separate Step-2 run needed).
        df_nrb_profile.to_excel(xw, index=False, sheet_name="NRB profile")
        df_snr_co.to_excel(xw, index=False, sheet_name="SNR")
        df_delta_v.to_excel(xw, index=False, sheet_name="Depol_delta_v")
        df_type.to_excel(xw, index=False, sheet_name="Aerosol_type")
        df_nrb_co.to_excel(xw, index=False, sheet_name="NRB_co")
        df_nrb_cr.to_excel(xw, index=False, sheet_name="NRB_cross")
        # normalised (÷max) pair — NRB_co_norm is the SAME data as "NRB profile"
        # above; written under this name too so the co/cross norm sheets are
        # symmetric and easy to find.
        df_nrb_profile.to_excel(xw, index=False, sheet_name="NRB_co_norm")
        df_nrb_cr_norm.to_excel(xw, index=False, sheet_name="NRB_cross_norm")
        df_snr_co.to_excel(xw, index=False, sheet_name="SNR_co")
        df_snr_cr.to_excel(xw, index=False, sheet_name="SNR_cross")
        df_qc.to_excel(xw, index=False, sheet_name="QC_calibration")
        params.to_excel(xw, index=False, sheet_name="Parameters")
        for sheet, d in sig_sheets.items():
            d.to_excel(xw, index=False, sheet_name=sheet)

    return {
        "delta_v": df_delta_v,
        "aerosol_type": df_type,
        "nrb_profile": df_nrb_profile,
        "nrb_co": df_nrb_co,
        "nrb_cross": df_nrb_cr,
        "snr_co": df_snr_co,
        "snr_cross": df_snr_cr,
        "qc": df_qc,
        # per-profile signal diagnostics, ordered like the delta_v columns
        "profiles_diag": diag_list,
    }


# ---------------------------------------------------------------------------
# Self-test on the real 23-06-2026 pair (if present on the Desktop)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    co = Path(r"C:\Users\Alongkorn\Desktop\co_pol_23-06-2026-21.37.dat")
    cr = Path(r"C:\Users\Alongkorn\Desktop\cross_pol_23-06-2026-21.37.dat")
    if co.exists() and cr.exists():
        df, meta = build_depol_for_pair(co, cr)
        print(f"Calibration constant C = {meta['C']:.3f} "
              f"(cal window {meta['cal_rmin_m']:.0f}-{meta['cal_rmax_m']:.0f} m, "
              f"n={meta['n_cal']})")
        r = df["range_m"].to_numpy(float)
        for rr in (500, 800, 1200, 2000, 3000):
            i = int(np.argmin(np.abs(r - rr)))
            row = df.iloc[i]
            print(f"  R={r[i]:>5.0f} m   delta_v={row['delta_v']:.4f}   {row['aerosol_type']}")
    else:
        print("Sample files not found on Desktop; skipping self-test.")
