"""Layer classification first, aerosol layer top second.

Why this exists
---------------
The legacy ALT (pbl_engine) runs a Haar wavelet over the whole search window and
takes the STRONGEST falling edge. Two things fool it:

1. A cloud edge is always the strongest edge, so a cloud that slips into the
   window becomes the "ALT".
2. At night the aerosol profile often has several stacked layers; the strongest
   edge is frequently a lower one, while the reference instruments (MPLNET,
   CALIPSO, Mini-MPL) report the TOP of the uppermost aerosol layer.

So, like those products, the profile is classified bin by bin first, and the
layer top is searched only in the cloud-free aerosol part below the lowest
cloud base.

Method (calibration-free; works on the physical co-polar NRB)
-------------------------------------------------------------
Cloud
    A cloud base is where the signal JUMPS: the maximum within
    ``CLOUD_JUMP_DIST_M`` above a bin is at least ``CLOUD_JUMP_RATIO`` times the
    level just below it. Water/ice clouds backscatter one to two orders of
    magnitude more than aerosol, so a 4x rise within 300 m is a cloud signature,
    while a boundary-layer top is a FALL and an elevated smoke layer rises more
    gently. The top is where the signal returns to the pre-cloud level, or
    where SNR is lost (opaque cloud: everything above it is unobservable).
    Depolarization labels a detected cloud as ice (delta >= ICE_DELTA) or water;
    it does not decide whether a cloud is there, because a water cloud has low
    delta and the prototype's delta calibration is not yet validated.

Aerosol layer top (the ALT)
    Below min(lowest cloud base, SNR-trusted top, search ceiling), compute the
    RELATIVE change between the mean signal in a window above and below each
    height. Every local minimum where the signal falls by at least
    ``TOP_MIN_DROP`` is a layer top; the UPPERMOST one is reported. A relative
    (not absolute) drop keeps a weak upper layer from being out-voted by the
    strong near-surface gradient.

Per-bin labels: 'cloud_water', 'cloud_ice', 'aerosol', 'clear', 'unobservable'.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---- parameters, fixed from physical reasoning (not tuned to the reference) --
MIN_RANGE_M = 200.0          # below: incomplete overlap
SMOOTH_M = 90.0              # running-median length before any test
CLOUD_JUMP_RATIO = 4.0       # peak within JUMP_DIST above >= ratio * level below
CLOUD_JUMP_DIST_M = 300.0
CLOUD_BELOW_M = 150.0        # length of the "level just below" window
CLOUD_MIN_SNR = 5.0          # SNR at the cloud peak
OPAQUE_SNR = 3.0             # SNR lost within OPAQUE_WITHIN_M above top -> opaque
OPAQUE_WITHIN_M = 500.0
ICE_DELTA = 0.35
SNR_TRUST_MIN = 5.0          # ALT search stays where SNR is solid
SEARCH_TOP_M = 5000.0        # boundary layer interest ends here
TOP_HALF_WINDOW_M = 250.0    # half-window of the above/below means
TOP_MIN_DROP = 0.25          # >= 25 % relative fall marks a layer top
CLOUD_MARGIN_M = 60.0        # ALT search stops this far below a cloud base
NOISE_WINDOW_M = 300.0       # window for the signal-derived noise estimate
NEAR_GUARD_M = 100.0         # a cloud base must sit this far above the first valid bin


def _bins(length_m: float, dr: float) -> int:
    return max(1, int(round(length_m / dr)))


def _running(y: np.ndarray, n: int, how: str = "median") -> np.ndarray:
    s = pd.Series(y)
    roll = s.rolling(n, center=True, min_periods=max(1, n // 2))
    return (roll.median() if how == "median" else roll.mean()).to_numpy(float)


def signal_snr(r_m, nrb) -> np.ndarray:
    """SNR of the smoothed signal, estimated from the signal itself.

    The photon-channel SNR is unusable in daylight: the solar background drives
    the photon counter into saturation and the profile is carried by the analog
    channel, so the recorded SNR reads ~0 even where a cloud is obvious. Here the
    noise is the robust scatter (MAD) of bin-to-bin differences in a local window,
    divided by sqrt(2), so the estimate works for analog, photon or glued signal
    alike. (Residuals from a short running median are unusable: with a 3-bin
    median at 30 m half of them are exactly zero and the MAD collapses to 0.)
    Bins without a usable noise estimate get NaN, i.e. they never pass an SNR test.
    """
    r = np.asarray(r_m, float)
    y = np.asarray(nrb, float)
    dr = float(np.nanmedian(np.diff(r)))
    n = _bins(SMOOTH_M, dr)
    ys = _running(np.where(np.isfinite(y), y, np.nan), n)
    w = max(7, _bins(NOISE_WINDOW_M, dr))
    dif = np.abs(np.diff(y, prepend=np.nan))
    roll = pd.Series(dif).rolling(w, center=True, min_periods=max(4, w // 3))
    sig = roll.median().to_numpy(float) * 1.4826 / np.sqrt(2.0)
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nan)
    with np.errstate(all="ignore"):
        return ys / (sig / np.sqrt(n))


def _snr_top(r: np.ndarray, snr: np.ndarray, snr_min: float, dr: float) -> float:
    """Contiguous trusted top: walk up from the SNR peak until smoothed SNR < min."""
    sm = _running(np.where(np.isfinite(snr), snr, np.nan), _bins(SMOOTH_M * 2, dr))
    ok = np.isfinite(sm) & (sm >= snr_min) & (r >= MIN_RANGE_M)
    if not ok.any():
        return float("nan")
    i = int(np.nanargmax(np.where(ok, sm, -np.inf)))
    top = float(r[i])
    while i < r.size and ok[i]:
        top = float(r[i])
        i += 1
    return top


def detect_clouds(r_m, nrb, snr=None, delta=None, *, max_range_m: Optional[float] = None
                  ) -> List[Dict[str, float]]:
    """Clouds in one profile, lowest first.

    ``snr=None`` (recommended) uses :func:`signal_snr`, which also works in
    daylight; pass a photon SNR profile only to reproduce the first version.

    Each item: base_m, peak_m, top_m, peak_ratio, delta_median, phase
    ('ice' | 'water' | 'unknown'), opaque (bool).
    """
    r = np.asarray(r_m, float)
    y = np.asarray(nrb, float)
    s = signal_snr(r, y) if snr is None else np.asarray(snr, float)
    d = np.asarray(delta, float) if delta is not None else np.full(r.shape, np.nan)
    if r.size < 10 or not np.isfinite(y).any():
        return []
    first_valid = float(r[np.isfinite(y)][0])
    dr = float(np.nanmedian(np.diff(r)))
    ys = _running(np.where(np.isfinite(y), y, np.nan), _bins(SMOOTH_M, dr))
    ss = _running(np.where(np.isfinite(s), s, np.nan), _bins(SMOOTH_M, dr))

    n_up, n_dn = _bins(CLOUD_JUMP_DIST_M, dr), _bins(CLOUD_BELOW_M, dr)
    top_limit = float(max_range_m) if max_range_m is not None else float(np.nanmax(r))

    clouds: List[Dict[str, float]] = []
    i = int(np.searchsorted(r, MIN_RANGE_M))
    while i < r.size - n_up and r[i] <= top_limit:
        below = ys[max(0, i - n_dn):i]
        below = below[np.isfinite(below) & (below > 0)]
        # Near the start of valid data the "level below" is the overlap edge, not
        # clear air: a jump there is the data starting, not a cloud base.
        if below.size < 0.8 * n_dn or r[i] - CLOUD_BELOW_M < first_valid + NEAR_GUARD_M:
            i += 1
            continue
        level = float(np.median(below))
        seg = ys[i:i + n_up]
        if not np.isfinite(seg).any():
            i += 1
            continue
        j = i + int(np.nanargmax(seg))
        ratio = float(ys[j] / level)
        if ratio < CLOUD_JUMP_RATIO or not (np.isfinite(ss[j]) and ss[j] >= CLOUD_MIN_SNR):
            i += 1
            continue

        # Base: first bin on the way up that exceeds the geometric midpoint.
        thr = level * np.sqrt(ratio)
        b = i
        while b < j and not (np.isfinite(ys[b]) and ys[b] >= thr):
            b += 1
        # Top: back down to 1.5x the pre-cloud level, or SNR lost.
        k = j
        while k < r.size - 1 and np.isfinite(ys[k]) and ys[k] > 1.5 * level \
                and not (np.isfinite(ss[k]) and ss[k] < OPAQUE_SNR):
            k += 1
        above = (r > r[k]) & (r <= r[k] + OPAQUE_WITHIN_M)
        opaque = bool(above.any() and np.nanmedian(np.where(above, ss, np.nan)) < OPAQUE_SNR)
        dsel = d[b:k + 1]
        dsel = dsel[np.isfinite(dsel)]
        dmed = float(np.median(dsel)) if dsel.size else float("nan")
        phase = "unknown" if not np.isfinite(dmed) else ("ice" if dmed >= ICE_DELTA else "water")
        clouds.append({"base_m": float(r[b]), "peak_m": float(r[j]), "top_m": float(r[k]),
                       "peak_ratio": ratio, "delta_median": dmed, "phase": phase,
                       "opaque": opaque})
        if opaque:
            break
        i = k + 1
    return clouds


def aerosol_layer_top(r_m, nrb, snr, *, ceiling_m: float) -> Dict[str, object]:
    """Top of the uppermost aerosol layer below ``ceiling_m``.

    Returns alt_m, drop (relative fall at the top), n_tops (all tops found,
    lowest first in tops_m), status.
    """
    r = np.asarray(r_m, float)
    y = np.asarray(nrb, float)
    out = {"alt_m": float("nan"), "drop": float("nan"), "tops_m": [], "status": "no_top"}
    if r.size < 10 or not np.isfinite(ceiling_m) or ceiling_m <= MIN_RANGE_M + 2 * TOP_HALF_WINDOW_M:
        out["status"] = "window_too_short"
        return out
    dr = float(np.nanmedian(np.diff(r)))
    ys = _running(np.where(np.isfinite(y), y, np.nan), _bins(SMOOTH_M, dr))
    h = _bins(TOP_HALF_WINDOW_M, dr)

    lower = pd.Series(ys).rolling(h, min_periods=max(1, h // 2)).mean().to_numpy(float)
    upper = pd.Series(ys[::-1]).rolling(h, min_periods=max(1, h // 2)).mean().to_numpy(float)[::-1]
    # lower[i] = mean over (i-h, i]; upper[i] = mean over [i, i+h)
    with np.errstate(all="ignore"):
        rel = (upper - lower) / lower
    ok = (r >= MIN_RANGE_M + TOP_HALF_WINDOW_M) & (r <= ceiling_m - TOP_HALF_WINDOW_M) \
        & np.isfinite(rel) & (lower > 0)
    if not ok.any():
        out["status"] = "window_too_short"
        return out

    rr = np.where(ok, rel, np.nan)
    tops = []
    for i in range(1, r.size - 1):
        if not ok[i] or rr[i] > -TOP_MIN_DROP:
            continue
        lo, hi = max(0, i - h), min(r.size, i + h + 1)
        if rr[i] <= np.nanmin(rr[lo:hi]):
            tops.append((float(r[i]), float(rr[i])))
    if not tops:
        return out
    alt, drop = tops[-1]
    out.update(alt_m=alt, drop=drop, tops_m=[t for t, _ in tops], status="ok")
    return out


# ---- scattering-ratio layer top (molecular reference, CALIPSO-style) ---------
SITE_ALTITUDE_M = 300.0      # Chiang Mai site, added to range for the atmosphere
SR_AEROSOL_MIN = 1.2         # attenuated scattering ratio above which a bin is aerosol
SR_SUSTAIN_M = 150.0         # the aerosol run must last this long
REF_WINDOW_M = 1000.0        # Rayleigh-fit window length
REF_SEARCH_M = (3000.0, 8000.0)


def attenuated_molecular(r_m) -> np.ndarray:
    """beta_mol * T_mol^2 on the range grid (US Standard Atmosphere, 532 nm)."""
    from fernald_engine import molecular_backscatter_532, molecular_extinction
    r = np.asarray(r_m, float)
    beta = molecular_backscatter_532(r + SITE_ALTITUDE_M)
    alpha = molecular_extinction(beta)
    tau = np.concatenate(([0.0], np.cumsum(0.5 * (alpha[1:] + alpha[:-1]) * np.diff(r))))
    return beta * np.exp(-2.0 * tau)


def scattering_ratio(r_m, nrb, snr, *, top_m: float) -> Dict[str, object]:
    """Attenuated scattering ratio SR = NRB / (C * beta_mol * T^2).

    C comes from a Rayleigh fit: among 1 km windows between REF_SEARCH_M and the
    SNR-trusted top, the one where NRB / molecular is flattest (lowest CV) is
    taken as aerosol-free and C is the median ratio there.
    """
    r = np.asarray(r_m, float)
    y = np.asarray(nrb, float)
    dr = float(np.nanmedian(np.diff(r)))
    ys = _running(np.where(np.isfinite(y), y, np.nan), _bins(SMOOTH_M, dr))
    mol = attenuated_molecular(r)
    q = ys / mol
    lo, hi = REF_SEARCH_M[0], min(REF_SEARCH_M[1], top_m if np.isfinite(top_m) else REF_SEARCH_M[1])
    best = None
    z = lo
    while z + REF_WINDOW_M <= hi:
        m = (r >= z) & (r < z + REF_WINDOW_M) & np.isfinite(q) & (q > 0)
        if m.sum() >= 8:
            cv = float(np.std(q[m]) / np.median(q[m]))
            if best is None or cv < best[0]:
                best = (cv, float(np.median(q[m])), z)
        z += REF_WINDOW_M / 4
    if best is None:
        return {"sr": np.full(r.shape, np.nan), "C": np.nan, "ref_m": np.nan, "ref_cv": np.nan}
    return {"sr": q / best[1], "C": best[1], "ref_m": best[2], "ref_cv": best[0]}


def aerosol_layer_top_sr(r_m, nrb, snr, *, ceiling_m: float, snr_top_m: float
                         ) -> Dict[str, object]:
    """Top of the uppermost aerosol layer: the highest height below the ceiling
    under which SR stays >= SR_AEROSOL_MIN for SR_SUSTAIN_M."""
    r = np.asarray(r_m, float)
    out = {"alt_m": float("nan"), "status": "no_aerosol", "ref_m": np.nan, "C": np.nan}
    if not np.isfinite(ceiling_m) or ceiling_m <= MIN_RANGE_M:
        out["status"] = "window_too_short"
        return out
    srd = scattering_ratio(r, nrb, snr, top_m=snr_top_m)
    out.update(ref_m=srd["ref_m"], C=srd["C"])
    sr = srd["sr"]
    if not np.isfinite(srd["C"]):
        out["status"] = "no_reference_window"
        return out
    dr = float(np.nanmedian(np.diff(r)))
    n = _bins(SR_SUSTAIN_M, dr)
    aer = (sr >= SR_AEROSOL_MIN) & (r >= MIN_RANGE_M) & (r <= ceiling_m)
    idx = np.where(aer)[0]
    for i in idx[::-1]:
        if i - n + 1 >= 0 and aer[i - n + 1:i + 1].all():
            out.update(alt_m=float(r[i]), status="ok")
            break
    return out


def classify_profile(r_m, nrb, snr, delta=None, *, top_method: str = "sr") -> Dict[str, object]:
    """Full classification of one profile.

    Returns clouds, alt (dict from aerosol_layer_top), snr_top_m, ceiling_m and
    labels (per-bin strings on r_m).
    """
    r = np.asarray(r_m, float)
    s = np.asarray(snr, float)
    dr = float(np.nanmedian(np.diff(r)))
    snr_top = _snr_top(r, s, SNR_TRUST_MIN, dr)
    obs_top = _snr_top(r, s, OPAQUE_SNR, dr)

    clouds = detect_clouds(r, nrb, snr, delta,
                           max_range_m=obs_top if np.isfinite(obs_top) else None)
    ceiling = SEARCH_TOP_M
    if np.isfinite(snr_top):
        ceiling = min(ceiling, snr_top)
    if clouds:
        ceiling = min(ceiling, clouds[0]["base_m"] - CLOUD_MARGIN_M)
    if top_method == "sr":
        alt = aerosol_layer_top_sr(r, nrb, snr, ceiling_m=ceiling, snr_top_m=snr_top)
    else:
        alt = aerosol_layer_top(r, nrb, snr, ceiling_m=ceiling)

    labels = np.full(r.size, "clear", dtype=object)
    labels[r < MIN_RANGE_M] = "unobservable"
    if np.isfinite(alt["alt_m"]):
        labels[(r >= MIN_RANGE_M) & (r <= alt["alt_m"])] = "aerosol"
    for c in clouds:
        m = (r >= c["base_m"]) & (r <= c["top_m"])
        labels[m] = "cloud_ice" if c["phase"] == "ice" else "cloud_water"
        if c["opaque"]:
            labels[r > c["top_m"]] = "unobservable"
    if np.isfinite(obs_top):
        labels[r > obs_top] = "unobservable"
    return {"clouds": clouds, "alt": alt, "snr_top_m": snr_top,
            "ceiling_m": ceiling, "labels": labels}
