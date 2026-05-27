"""
Fernald / Klett inversion engine for TR40 elastic Mie LiDAR (532 nm).

Takes the NRB profile produced by nrb_engine (already range-corrected,
BG / overlap / afterpulse corrected) and runs a backward Fernald (1984)
or Klett (1981) inversion to retrieve aerosol optical properties.

The inversion is self-calibrating via Rayleigh normalisation: the boundary
condition beta_total(R_ref) = beta_mol(R_ref) (assuming beta_aer ~ 0 at the
reference range) provides the absolute scale, so no separate lidar constant
is needed.  The normalisation by energy and max used inside nrb_engine
cancels algebraically.

Output quantities
-----------------
beta_aer  [m^-1 sr^-1]  Aerosol backscatter coefficient
alpha_aer [m^-1]        Aerosol extinction  = S_a * beta_aer
AOD       [-]           Aerosol optical depth = integral(alpha_aer dR)

References
----------
Fernald F.G. (1984) Appl. Opt. 23(5), 652-653.
Klett J.D.   (1981) Appl. Opt. 20(2), 211-220.
Collis R.T.H., Russell P.B. (1976) in Meteorological Monographs 15(37).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_KB = 1.380649e-23           # J K^-1  Boltzmann constant
_SIGMA_MOL_532 = 5.45e-32    # m^2 sr^-1  Rayleigh backscatter XS @ 532 nm
_SA_MOL = 8.0 * np.pi / 3.0  # ~8.378 sr  molecular lidar ratio (Rayleigh)

# US Standard Atmosphere (ISO 2533) — troposphere only
_T0 = 288.15    # K    sea-level temperature
_P0 = 101325.0  # Pa   sea-level pressure
_L  = 0.0065    # K m^-1  temperature lapse rate (6.5 K / km)
_G  = 9.80665   # m s^-2  standard gravity
_RD = 287.05    # J kg^-1 K^-1  dry-air specific gas constant


# ---------------------------------------------------------------------------
# Atmospheric profiles
# ---------------------------------------------------------------------------

def us_standard_atmosphere(R_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    ISO 2533 troposphere temperature and pressure vs range.

    Parameters
    ----------
    R_m : geometric range (treated as altitude above sea level) in metres.

    Returns
    -------
    T_K  : temperature [K]
    P_Pa : pressure    [Pa]
    """
    h = np.clip(np.asarray(R_m, float), 0.0, 11000.0)
    T = _T0 - _L * h
    P = _P0 * (T / _T0) ** (_G / (_L * _RD))
    return T, P


def molecular_backscatter_532(
    R_m: np.ndarray,
    T_K:  Optional[np.ndarray] = None,
    P_Pa: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Rayleigh backscatter coefficient beta_mol(R) [m^-1 sr^-1] at 532 nm.

    Uses US Standard Atmosphere if T_K / P_Pa are not provided.
    """
    R = np.asarray(R_m, float)
    if T_K is None or P_Pa is None:
        T, P = us_standard_atmosphere(R)
    else:
        T = np.broadcast_to(np.asarray(T_K, float), R.shape)
        P = np.broadcast_to(np.asarray(P_Pa, float), R.shape)
    N = P / (_KB * T)           # number density [m^-3]
    return N * _SIGMA_MOL_532   # [m^-1 sr^-1]


def molecular_extinction(beta_mol: np.ndarray) -> np.ndarray:
    """Rayleigh extinction alpha_mol = (8*pi/3) * beta_mol [m^-1]."""
    return _SA_MOL * np.asarray(beta_mol, float)


# ---------------------------------------------------------------------------
# Inversion — Fernald 1984
# ---------------------------------------------------------------------------

def fernald_inversion(
    R_m: np.ndarray,
    S_R: np.ndarray,
    R_ref_m: float,
    beta_mol: np.ndarray,
    lidar_ratio_aer: float = 50.0,
    lidar_ratio_mol: float = _SA_MOL,
    beta_aer_ref: float = 0.0,
) -> np.ndarray:
    """
    Fernald (1984) backward inversion separating aerosol from molecular.

    Parameters
    ----------
    R_m           : range array [m], monotonically increasing.
    S_R           : range-corrected signal S(R) = P(R)*R^2 (arbitrary units).
    R_ref_m       : reference range [m] — must be in clean (aerosol-free) air.
    beta_mol      : molecular backscatter [m^-1 sr^-1] on same grid as R_m.
    lidar_ratio_aer : S_a, aerosol extinction-to-backscatter ratio [sr].
    lidar_ratio_mol : S_m = 8*pi/3 sr (Rayleigh).
    beta_aer_ref  : assumed beta_aer at R_ref (typically 0).

    Returns
    -------
    beta_aer : aerosol backscatter [m^-1 sr^-1].
    """
    R  = np.asarray(R_m,    float)
    S  = np.asarray(S_R,    float)
    bm = np.asarray(beta_mol, float)
    Sa, Sm = float(lidar_ratio_aer), float(lidar_ratio_mol)

    if not (R[0] <= float(R_ref_m) <= R[-1]):
        raise ValueError(
            f"R_ref_m={R_ref_m:.0f} m outside profile range "
            f"[{R[0]:.0f}, {R[-1]:.0f}] m"
        )

    i_ref = int(np.searchsorted(R, float(R_ref_m)))

    beta_total = np.full_like(R, np.nan, float)
    beta_total[i_ref] = float(beta_aer_ref) + float(bm[i_ref])

    for i in range(i_ref - 1, -1, -1):
        dR    = float(R[i + 1] - R[i])
        A     = (Sa - Sm) * float(bm[i + 1] + bm[i]) * dR
        X_i   = float(S[i])     * np.exp(A)
        X_ip1 = float(S[i + 1])
        bt_ip1 = beta_total[i + 1]
        if not (np.isfinite(bt_ip1) and bt_ip1 > 0.0):
            continue
        denom = (X_ip1 / bt_ip1) + Sa * (X_i + X_ip1) * dR
        if abs(denom) < 1e-300:
            continue
        beta_total[i] = X_i / denom

    return beta_total - bm


# ---------------------------------------------------------------------------
# Inversion — Klett 1981
# ---------------------------------------------------------------------------

def klett_inversion(
    R_m: np.ndarray,
    S_R: np.ndarray,
    R_ref_m: float,
    beta_ref: float = 1e-7,
    k: float = 1.0,
) -> np.ndarray:
    """
    Klett (1981) power-law inversion for total backscatter (no mol. separation).

    beta_ref should equal beta_mol(R_ref) for clean-air Rayleigh normalisation.
    Returns beta_total [m^-1 sr^-1].
    """
    R = np.asarray(R_m, float)
    S = np.asarray(S_R, float)

    if not (R[0] <= float(R_ref_m) <= R[-1]):
        raise ValueError(
            f"R_ref_m={R_ref_m:.0f} m outside profile range "
            f"[{R[0]:.0f}, {R[-1]:.0f}] m"
        )

    i_ref  = int(np.searchsorted(R, float(R_ref_m)))
    S_ref  = float(S[i_ref])
    S_safe = np.where((S > 0) & np.isfinite(S), S, np.nan)
    ref_safe = max(abs(S_ref), 1e-300)

    with np.errstate(invalid="ignore", divide="ignore"):
        integrand = np.exp(np.log(S_safe / ref_safe) / k)

    beta = np.full_like(R, np.nan, float)
    for i in range(len(R)):
        if i > i_ref:
            v = integrand[i]
            beta[i] = float(beta_ref * v) if np.isfinite(v) else np.nan
            continue
        seg   = integrand[i: i_ref + 1]
        seg_r = R[i: i_ref + 1]
        if not np.all(np.isfinite(seg)):
            continue
        integral = float(np.trapezoid(seg, seg_r))
        denom = (1.0 / beta_ref) + (2.0 / k) * integral
        beta[i] = float(integrand[i]) / denom if abs(denom) > 1e-300 else np.nan

    return beta


# ---------------------------------------------------------------------------
# Batch processor — daily NRB profile DataFrame
# ---------------------------------------------------------------------------

def compute_fernald_from_nrb_df(
    df_nrb: pd.DataFrame,
    R_ref_m: float = 5000.0,
    lidar_ratio_aer: float = 50.0,
    method: str = "fernald",
    T_K_scalar: Optional[float] = None,
    P_Pa_scalar: Optional[float] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run Fernald / Klett inversion on every profile column of an NRB DataFrame.

    Parameters
    ----------
    df_nrb          : DataFrame produced by nrb_engine.build_daily_profile_from_folder.
                      Must have a "Range(m)" column followed by timestamp columns.
    R_ref_m         : Reference range [m] in clean (aerosol-free) air.
                      Typical: 4000-8000 m for TR40 nighttime profiles.
    lidar_ratio_aer : Aerosol lidar ratio S_a [sr].
                      Typical 532 nm values: urban ~70, marine ~25, smoke ~80.
    method          : "fernald" (Fernald 1984, recommended) or "klett".
    T_K_scalar      : Uniform temperature [K] for the entire profile (None = Standard Atm).
    P_Pa_scalar     : Uniform pressure [Pa] (None = Standard Atm).

    Returns
    -------
    dict with three DataFrames:
      "beta_aer"  : Range(m) + timestamp columns — aerosol backscatter [m^-1 sr^-1]
      "alpha_aer" : Range(m) + timestamp columns — aerosol extinction  [m^-1]
      "AOD"       : 1-row DataFrame — per-timestamp column-integrated AOD
    """
    R = df_nrb["Range(m)"].to_numpy(float)
    ts_cols = [c for c in df_nrb.columns if c != "Range(m)"]

    if len(ts_cols) == 0:
        raise ValueError("df_nrb has no timestamp columns.")

    # Molecular profile (shared — same hardware range grid for all profiles)
    if T_K_scalar is not None and P_Pa_scalar is not None:
        T_arr = np.full_like(R, float(T_K_scalar))
        P_arr = np.full_like(R, float(P_Pa_scalar))
    else:
        T_arr, P_arr = us_standard_atmosphere(R)
    beta_mol = molecular_backscatter_532(R, T_arr, P_arr)

    # Rayleigh beta at reference range (used as beta_ref for Klett)
    i_ref_global = int(np.searchsorted(R, float(R_ref_m)))
    beta_mol_ref = float(beta_mol[min(i_ref_global, len(beta_mol) - 1)])

    beta_aer_cols:  Dict = {}
    alpha_aer_cols: Dict = {}
    aod_vals:       Dict = {}

    for col in ts_cols:
        S = df_nrb[col].to_numpy(float)
        # Zero-fill non-positive / non-finite bins before inversion
        S_in = np.where(np.isfinite(S) & (S > 0.0), S, 0.0)
        try:
            if method.lower() == "fernald":
                ba = fernald_inversion(
                    R, S_in, R_ref_m, beta_mol,
                    lidar_ratio_aer=lidar_ratio_aer,
                )
            else:
                bt = klett_inversion(
                    R, S_in, R_ref_m,
                    beta_ref=beta_mol_ref,
                )
                ba = bt - beta_mol

            ba = np.where(np.isfinite(ba), np.maximum(ba, 0.0), np.nan)
            aa = lidar_ratio_aer * ba

            valid = np.isfinite(aa)
            aod = float(np.trapezoid(np.where(valid, aa, 0.0), R)) if valid.any() else np.nan

        except Exception:
            ba  = np.full_like(R, np.nan)
            aa  = np.full_like(R, np.nan)
            aod = np.nan

        beta_aer_cols[col]  = ba
        alpha_aer_cols[col] = aa
        aod_vals[col]       = aod

    def _frame(cols_dict: Dict) -> pd.DataFrame:
        df = pd.DataFrame({"Range(m)": R})
        for col, arr in cols_dict.items():
            df[col] = arr
        return df

    df_beta  = _frame(beta_aer_cols)
    df_alpha = _frame(alpha_aer_cols)
    df_aod   = pd.DataFrame([{
        "R_ref_m":            R_ref_m,
        "lidar_ratio_aer_sr": lidar_ratio_aer,
        "method":             method,
        **aod_vals,
    }])

    return {"beta_aer": df_beta, "alpha_aer": df_alpha, "AOD": df_aod}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    R = np.linspace(50, 9000, 2000)
    T_k, P_pa = us_standard_atmosphere(R)
    bm = molecular_backscatter_532(R, T_k, P_pa)

    # Synthetic aerosol layer 500-1500 m
    ba_true = np.zeros_like(R)
    ba_true[(R > 500) & (R < 1500)] = 5e-6

    Sa = 50.0
    alpha_total = _SA_MOL * bm + Sa * ba_true
    tau = np.cumsum(alpha_total) * (R[1] - R[0])
    P = (bm + ba_true) * np.exp(-2 * tau) / R**2
    S = P * R**2

    ba_rec = fernald_inversion(R, S, R_ref_m=7000.0, beta_mol=bm,
                               lidar_ratio_aer=Sa)

    i1000 = int(np.argmin(np.abs(R - 1000)))
    print(f"True beta_aer @ 1000 m : {ba_true[i1000]:.3e} m^-1 sr^-1")
    print(f"Recovered             : {ba_rec[i1000]:.3e} m^-1 sr^-1")
    err = abs(ba_rec[i1000] - ba_true[i1000]) / ba_true[i1000] * 100
    print(f"Relative error        : {err:.1f}%")
