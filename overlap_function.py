"""Where the TR40's profile starts: the overlap function O(R).

Why this file exists
--------------------
Cutting the near field needs a reason. "The ADC clips below 146 m" is an
electronics reason and moves with the high voltage; the reason that holds is
geometry - below full overlap the telescope does not see the whole laser beam,
so the profile there is not a measurement of the atmosphere at all.

The number comes from the hardware
----------------------------------
``overlap.py`` already carries the NARIT TR40 geometry and the Gaussian-beam +
Rice-CDF model (Kumar & Rocadenbosch 2013):

    telescope   EdgeHD 800, f = 2032 mm, field stop 8 mm  -> FOV 3.94 mrad
    laser       VIRON 532 nm, D4sigma 3.8 mm, divergence 1.5 mrad
    biaxial     axes 156 mm apart (2026 CAD)

    hard-edge check   R = (d + a0) / (FOV/2 - div/2) = 130 m
    Rice model        O = 0.90 at 108 m, 0.95 at 120 m, **0.99 at 149 m**, 0.999 at 208 m

So the profile starts at **150 m**. That also happens to clear every ADC-clipped
bin (clipping reaches 146 m at 750 V, 105 m at 680 V), but the clipping is not
the reason - the geometry is.

Sensitivity, if the hardware numbers are ever re-measured: the axis separation
barely matters (140-180 mm -> 134-172 m), the field stop and the divergence
matter a lot (6 mm -> 297 m, 10 mm -> 101 m; 1.0 mrad -> 115 m, 2.0 mrad ->
223 m). Those two are worth measuring before quoting 150 m in a paper.

The measured cross-check, and where it disagrees
------------------------------------------------
Measured 2026-09-24 from 112 night profiles over 8 nights, two independent ways:

1. Against the Mini-MPL's own overlap-corrected NRB: parallel reaches 0.99 at
   150 m, perpendicular at 45 m. Agrees with the geometry.
2. Homogeneous-layer extrapolation (no Raman channel, no horizontal scan): fit
   ln(NRB) to a straight line over a fully overlapped stretch, extrapolate down,
   O_apparent = NRB / extrapolation, then take the 90th percentile over all the
   nights so the aerosol averages out. **Perpendicular comes out at 82-112 m,
   agreeing with the geometry. Parallel comes out at 255-405 m** - and it stays
   there for every fit window tried between 250 m and 2400 m, so it is not a
   fitting artefact.

Both channels look through the same telescope, so the geometry cannot differ
between them: the parallel deficit between ~150 m and ~340 m is something in the
parallel detection chain, not overlap. It is NOT simple amplitude compression -
HV 750 V and HV 680 V, where the analog is 1.8x smaller, give 330 m and 338 m,
and an amplitude threshold would have moved. Candidates still open: PMT gain
suppression after the huge near-field pulse (the reason an ND filter is wanted
on the parallel channel anyway), or leakage of the strong parallel signal into
the perpendicular channel, which would lift the perpendicular curve instead.
Until that is settled, the cut follows the geometry and the MPL, and
``CURVE_PARALLEL_MEASURED`` is kept here as the evidence.

Reproduce: ``measure_from_folders`` below, or the session scripts overlap3.py /
overlap4.py. The definitive measurement is a horizontal scan through a
homogeneous layer - that is the one experiment that would settle it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

MEASURED_ON = "2026-09-24"
MEASURED_FROM = "112 night profiles, 8 nights (8-24 Sep 2026), HV 750/750 and 680/700"
GEOMETRY_SOURCE = "overlap.OverlapHardware (NARIT TR40, 2026 CAD) + Kumar & Rocadenbosch 2013"

#: Range [m] where the GEOMETRIC overlap reaches 0.99. Both channels share the
#: telescope, so this is one number, from overlap.compute_analytical_overlap.
FULL_OVERLAP_M = 149.0

#: Where every product starts. Rounded up from the geometry.
DEFAULT_MIN_RANGE_M = 150.0

#: The geometric curve, range [m] -> O (overlap.py, NARIT hardware).
CURVE_GEOMETRIC: Tuple[Tuple[float, float], ...] = (
    (30.0, 0.00), (60.0, 0.04), (81.0, 0.50), (105.0, 0.88), (108.0, 0.90),
    (120.0, 0.95), (149.0, 0.99), (208.0, 0.999), (250.0, 1.00),
)

#: What the extrapolation method measured per channel - the cross-check, not the
#: cut. Perpendicular agrees with the geometry; parallel does not (see above).
CURVE_PARALLEL_MEASURED: Tuple[Tuple[float, float], ...] = (
    (30.0, 0.08), (60.0, 0.20), (105.0, 0.51), (150.0, 0.63), (225.0, 0.83),
    (270.0, 0.90), (300.0, 0.95), (330.0, 0.98), (345.0, 0.99), (405.0, 1.00),
)
CURVE_PERPENDICULAR_MEASURED: Tuple[Tuple[float, float], ...] = (
    (30.0, 0.23), (60.0, 0.52), (90.0, 0.90), (105.0, 0.97), (120.0, 0.99),
    (150.0, 1.00),
)

FIT_WINDOW_M = (700.0, 1500.0)
NORM_WINDOW_M = (900.0, 1500.0)
ENVELOPE_QUANTILE = 0.90
MAX_FIT_RESIDUAL_LN = 0.12


def overlap_curve(channel: str = "parallel", source: str = "geometry"
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """(range [m], O). ``source='geometry'`` is the hardware model - one curve for
    both channels. ``source='measured'`` returns that channel's measured curve."""
    if str(source).lower().startswith("geom"):
        pts = CURVE_GEOMETRIC
    elif str(channel).lower().startswith("perp"):
        pts = CURVE_PERPENDICULAR_MEASURED
    else:
        pts = CURVE_PARALLEL_MEASURED
    a = np.asarray(pts, dtype=float)
    return a[:, 0], a[:, 1]


def overlap_at(r_m, channel: str = "parallel", source: str = "geometry") -> np.ndarray:
    """O(R) for the given ranges: interpolated inside the curve, 1.0 above it."""
    r = np.asarray(r_m, dtype=float)
    x, y = overlap_curve(channel, source)
    return np.interp(r, x, y, left=float(y[0]), right=1.0)


def full_overlap_range_m(channel: str = "parallel", threshold: float = 0.99,
                         source: str = "geometry") -> float:
    """Lowest range whose O is at least ``threshold``."""
    x, y = overlap_curve(channel, source)
    ok = np.where(y >= float(threshold))[0]
    return float(x[ok[0]]) if ok.size else float("nan")


def geometric_overlap(r_m, hardware=None) -> np.ndarray:
    """O(R) straight from the hardware model, for a range grid of your own."""
    from overlap import OverlapHardware, compute_analytical_overlap
    return compute_analytical_overlap(np.asarray(r_m, float),
                                      hardware or OverlapHardware())


# ── re-measuring the cross-check ──────────────────────────────────────────────

def _apparent_overlap(r_m: np.ndarray, nrb: np.ndarray, grid: np.ndarray,
                      fit: Tuple[float, float] = FIT_WINDOW_M) -> Optional[pd.Series]:
    S = np.interp(grid, np.asarray(r_m, float), np.asarray(nrb, float),
                  left=np.nan, right=np.nan)
    m = (grid >= fit[0]) & (grid <= fit[1]) & np.isfinite(S) & (S > 0)
    if int(m.sum()) < 20:
        return None
    k, b = np.polyfit(grid[m], np.log(S[m]), 1)
    if float(np.std(np.log(S[m]) - (k * grid[m] + b))) > MAX_FIT_RESIDUAL_LN:
        return None
    with np.errstate(all="ignore"):
        return pd.Series(S / np.exp(k * grid + b), index=grid)


def measure_from_folders(folders: Iterable[Path], *, channel: str = "parallel",
                         night_hours: Tuple[int, int] = (20, 5),
                         fit: Tuple[float, float] = FIT_WINDOW_M,
                         grid: Optional[np.ndarray] = None,
                         logger=None) -> pd.Series:
    """Re-measure the envelope from raw folders. Returns O indexed by range [m].

    ``nrb_engine`` is imported here rather than at module import time, so this
    module stays cheap for the engines that only want the constant."""
    import nrb_engine as ne
    import licel_binary_reader as lbr

    g = np.arange(15.0, 3001.0, 7.5) if grid is None else np.asarray(grid, float)
    rows: List[pd.Series] = []
    for folder in folders:
        for f in sorted(Path(folder).glob("a26*")):
            try:
                rf = lbr.parse_raw_file(f)
            except Exception:
                continue
            if rf.shots < 1500:
                continue
            t = ne.parse_licel_raw_timestamp(f.name)
            if t is None:
                continue
            hh = pd.Timestamp(t).hour
            if not (hh >= night_hours[0] or hh <= night_hours[1]):
                continue
            try:
                df, _ = ne.build_single_profile(
                    f, channel=channel, dr_m=float(rf.datasets[0].binwidth_m),
                    min_range_m=0.0)          # no cut: this is what measures it
            except Exception:
                continue
            s = _apparent_overlap(df["range_m"].to_numpy(float),
                                  df["nrb_final"].to_numpy(float), g, fit)
            if s is not None:
                rows.append(s)
            if logger:
                logger(f"{f.name}: {'ok' if s is not None else 'skipped'}")
    if not rows:
        raise ValueError("no usable night profile found")
    env = pd.DataFrame(rows).quantile(ENVELOPE_QUANTILE, axis=0)
    env = env.rolling(9, center=True, min_periods=3).median()
    env = env / float(np.nanmedian(env.loc[NORM_WINDOW_M[0]:NORM_WINDOW_M[1]]))
    return env.clip(upper=1.0)
