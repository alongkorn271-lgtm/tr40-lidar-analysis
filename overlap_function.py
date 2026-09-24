"""The TR40's overlap function O(r), measured from our own night profiles.

Why this file exists
--------------------
Cutting the near field needs a reason. "The ADC is clipped below 146 m" is an
electronics reason and changes with the high voltage; the honest reason is
geometry: below full overlap the telescope does not see the whole laser beam, so
the profile there is not a measurement of the atmosphere at all. This module
holds the measured curve and the range where it reaches 0.99, so every product
can start at one documented height.

How it was measured (2026-09-24, 81 night profiles over 8 nights, 8-24 Sep)
--------------------------------------------------------------------------
No Raman channel and no horizontal scan, so the standard homogeneous-layer
method was used per profile:

  1. fit ln(NRB) to a straight line over 700-1500 m, where the layer is smooth
     and overlap is certainly complete (fits with residual scatter above 0.12
     in ln were rejected);
  2. extrapolate that line down to the ground;
  3. O_apparent(r) = NRB(r) / extrapolation(r).

A single profile only gives a LOWER bound for O, because real aerosol below the
fit is usually thinner than the extrapolation. Pooling 8 nights of different
aerosol and taking the 90th percentile at each range (the upper envelope) lets
the instrument's own ceiling show through. The envelope is normalised on
900-1500 m and capped at 1.

Why the answer is trusted
-------------------------
The same curve comes out at HV 750 V and at HV 680 V, where the analog signal is
1.8x smaller: parallel reaches 0.99 at 330 m and 338 m respectively. Amplifier
compression would have moved with the signal level; geometry does not. The
Mini-MPL cross-check (our NRB divided by the MPL's own overlap-corrected NRB)
puts the parallel crossing at >= 150 m, which is a lower bound because the MPL's
own overlap correction is least certain in exactly that region.

The two channels differ - perpendicular is complete by ~90-120 m, parallel only
by ~340 m. They share the telescope, so this is differential overlap between the
two detector paths after the polarising splitter, and it is a known cause of a
biased depolarisation ratio in the near field.

The parallel curve below ~150 m is not trustworthy on the 750 V nights (the
analog is against the ADC rail there), but the 0.99 crossing sits far above it.

Reproduce with the script in the session scratchpad (overlap3.py) or by feeding
new nights to ``measure_from_folders``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

MEASURED_ON = "2026-09-24"
MEASURED_FROM = "81 night profiles, 8 nights (8-24 Sep 2026), HV 750/750 and 680/700"

#: Range [m] where the measured overlap first reaches 0.99, per channel.
FULL_OVERLAP_M: Dict[str, float] = {"parallel": 345.0, "perpendicular": 120.0}

#: What every product starts from. The parallel channel is the binding one:
#: the depolarisation ratio needs both channels, and mixing two different start
#: heights between products would make them incomparable.
DEFAULT_MIN_RANGE_M = 345.0

#: The measured envelope, range [m] -> O. Linear between the listed points.
CURVE_PARALLEL: Tuple[Tuple[float, float], ...] = (
    (30.0, 0.08), (60.0, 0.20), (105.0, 0.51), (150.0, 0.63), (225.0, 0.83),
    (270.0, 0.90), (300.0, 0.95), (330.0, 0.98), (345.0, 0.99), (405.0, 1.00),
)
CURVE_PERPENDICULAR: Tuple[Tuple[float, float], ...] = (
    (30.0, 0.23), (60.0, 0.52), (90.0, 0.90), (105.0, 0.97), (120.0, 0.99),
    (150.0, 1.00),
)

FIT_WINDOW_M = (700.0, 1500.0)
NORM_WINDOW_M = (900.0, 1500.0)
ENVELOPE_QUANTILE = 0.90
MAX_FIT_RESIDUAL_LN = 0.12


def overlap_curve(channel: str = "parallel") -> Tuple[np.ndarray, np.ndarray]:
    """(range [m], O) of the measured curve for a channel."""
    pts = CURVE_PERPENDICULAR if str(channel).lower().startswith("perp") else CURVE_PARALLEL
    a = np.asarray(pts, dtype=float)
    return a[:, 0], a[:, 1]


def overlap_at(r_m, channel: str = "parallel") -> np.ndarray:
    """O(r) for the given ranges: interpolated inside the measured curve, 1.0 above it."""
    r = np.asarray(r_m, dtype=float)
    x, y = overlap_curve(channel)
    return np.interp(r, x, y, left=float(y[0]), right=1.0)


def full_overlap_range_m(channel: str = "parallel", threshold: float = 0.99) -> float:
    """Lowest range whose measured O is at least ``threshold``."""
    x, y = overlap_curve(channel)
    ok = np.where(y >= float(threshold))[0]
    return float(x[ok[0]]) if ok.size else float("nan")


# ── re-measuring it ────────────────────────────────────────────────────────────

def _apparent_overlap(r_m: np.ndarray, nrb: np.ndarray, grid: np.ndarray
                      ) -> Optional[pd.Series]:
    S = np.interp(grid, np.asarray(r_m, float), np.asarray(nrb, float),
                  left=np.nan, right=np.nan)
    m = ((grid >= FIT_WINDOW_M[0]) & (grid <= FIT_WINDOW_M[1])
         & np.isfinite(S) & (S > 0))
    if int(m.sum()) < 20:
        return None
    k, b = np.polyfit(grid[m], np.log(S[m]), 1)
    if float(np.std(np.log(S[m]) - (k * grid[m] + b))) > MAX_FIT_RESIDUAL_LN:
        return None
    with np.errstate(all="ignore"):
        return pd.Series(S / np.exp(k * grid + b), index=grid)


def measure_from_folders(folders: Iterable[Path], *, channel: str = "parallel",
                         night_hours: Tuple[int, int] = (20, 5),
                         grid: Optional[np.ndarray] = None,
                         logger=None) -> pd.Series:
    """Re-measure the envelope from raw folders. Returns O indexed by range [m].

    Import-time cost is kept out of this module: ``nrb_engine`` is only imported
    when someone actually re-measures."""
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
                    f, channel=channel, dr_m=float(rf.datasets[0].binwidth_m))
            except Exception:
                continue
            s = _apparent_overlap(df["range_m"].to_numpy(float),
                                  df["nrb_final"].to_numpy(float), g)
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
