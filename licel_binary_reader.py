"""Decode Licel transient-recorder raw binary files (the native files the
Advanced Viewer opens) directly into physical units, with no ASCII export step.

Validated byte-for-byte against Advanced Viewer ``.dat`` exports: every signal,
standard-error and overflow column reproduces to a relative error of ~1e-5
(i.e. the ASCII export's own rounding).

File layout (see https://licel.com/raw_data_format.html and the official
``Licel_TCPIP_Python`` acquisition library, which is the authoritative spec):

    line 0 : " <filename>"
    line 1 : " <site> <start dd/mm/yyyy hh:mm:ss> <stop ...> <alt> <lon> <lat> <zen> <azi>"
    line 2 : " <shots> <repRate0> ... <numDatasets> ..."
    line 3..N : one descriptor line per active dataset, e.g.
        " 1 0 1 09024 0 0750 3.75 00532.p 0 0 01 000 16 002393 0.500 BT0 \"TR0_Parallel\""
          |  |  |  |     |   |    |     |               |   |      |     |
          |  |  |  |     |   HV  binw  wl.pol          adc shots range  device
          |  |  |  ndata
          |  dtype (0=analog 1=photon 2=analog^2 3=photon^2 5=overflow)
          active
    blank line
    <binary body> : each dataset = ``ndata`` * uint32 (little-endian), then b"\\r\\n"

Physical scaling (identical to the acquisition library):
    analog  -> mV  : count / shots * (fullscale_mV / 2**adc)
    photon  -> MHz : count / shots * (150.0 / binwidth_m)
    std-err        : the squared dataset stores sqd_bin = sqrt(N*sum(x^2)-sum(x)^2)
                     already; StErr = sqd_bin / sqrt(N*(N-1)) / sqrt(N) * scale
                     (N = shots). Squared datasets cover only the first bins;
                     beyond them StErr = 0.

This project uses dual-PMT files only (parallel on TR0, cross on TR1), but the
parser is descriptor-driven and does not hard-code that.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Distance light travels in 1 microsecond for a lidar (double pass), in metres.
_PHOTON_RANGE_CONST = 150.0

# dtype codes in the descriptor line (2nd token).
DT_ANALOG = 0
DT_PHOTON = 1
DT_ANALOG_SQ = 2
DT_PHOTON_SQ = 3
DT_OVERFLOW = 5


@dataclass
class DatasetMeta:
    """Descriptor-line metadata for one dataset in a raw file."""
    active: int
    dtype: int
    laser: int
    ndata: int
    hv: int
    binwidth_m: float
    wavelength_nm: int
    polarization: str            # o|p|s|r|l  (none|parallel|cross|right|left)
    adc_bits: int
    shots: int
    input_range: str             # analog: "0.500" (V full-scale); photon: discriminator
    device: str                  # e.g. BT0, BC0, S2A0, S2P0, OF0
    name: str = ""               # optional quoted label, e.g. "TR0_Parallel"
    raw_line: str = ""

    @property
    def tr_index(self) -> Optional[int]:
        """Transient-recorder index parsed from the trailing digit of ``device``."""
        for ch in reversed(self.device):
            if ch.isdigit():
                return int(ch)
        return None


@dataclass
class LicelRawFile:
    """A decoded raw file: header info + every dataset as raw uint32 counts."""
    path: Path
    filename: str
    site: str
    start_time: str
    stop_time: str
    altitude_m: int
    longitude: float
    latitude: float
    zenith: float
    azimuth: float
    shots: int
    num_datasets: int
    datasets: List[DatasetMeta]
    raw: Dict[str, np.ndarray]           # device token -> uint32 counts (float64)
    header_lines: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Low-level parsing
# -----------------------------------------------------------------------------

def _split_header(data: bytes) -> Tuple[List[str], int]:
    """Return (header_lines, body_start_byte).

    Header lines are the mostly-printable ASCII lines up to and including the
    blank line that precedes the binary body.
    """
    pos = 0
    lines: List[str] = []
    n = len(data)
    while pos < n:
        nl = data.find(b"\n", pos)
        if nl == -1:
            raise ValueError("No line terminator found while reading header.")
        line = data[pos:nl]
        # Binary body bytes are mostly non-printable; a header line is ASCII.
        printable = sum(1 for b in line if 9 <= b <= 126) / max(1, len(line))
        if printable < 0.85 or len(line) > 4000:
            break
        lines.append(line.rstrip(b"\r").decode("latin1"))
        pos = nl + 1
        if lines[-1].strip() == "":  # blank line -> body starts next
            break
    return lines, pos


def _parse_descriptor(line: str) -> Optional[DatasetMeta]:
    """Parse one dataset descriptor line; return None if it is not one."""
    toks = line.split()
    if len(toks) < 16:
        return None
    try:
        active = int(toks[0])
        dtype = int(toks[1])
        laser = int(toks[2])
        ndata = int(toks[3])
        hv = int(toks[5])
        binwidth = float(toks[6])
        wl_pol = toks[7]                       # e.g. "00532.p"
        adc = int(toks[12])
        shots = int(toks[13])
        input_range = toks[14]
        device = toks[15]
    except (ValueError, IndexError):
        return None
    wl_str, _, pol = wl_pol.partition(".")
    try:
        wavelength = int(wl_str)
    except ValueError:
        wavelength = 0
    name = ""
    if '"' in line:
        name = line.split('"')[1]
    return DatasetMeta(
        active=active, dtype=dtype, laser=laser, ndata=ndata, hv=hv,
        binwidth_m=binwidth, wavelength_nm=wavelength, polarization=pol,
        adc_bits=adc, shots=shots, input_range=input_range, device=device,
        name=name, raw_line=line.strip(),
    )


def parse_raw_file(path: str | Path) -> LicelRawFile:
    """Decode a Licel raw binary file into header metadata + raw uint32 datasets."""
    path = Path(path)
    data = path.read_bytes()
    header_lines, body_start = _split_header(data)
    if len(header_lines) < 4:
        raise ValueError(f"{path.name}: header too short to be a Licel raw file.")

    filename = header_lines[0].strip()

    # Line 1: site, start, stop, altitude, longitude, latitude, zenith, azimuth
    l1 = header_lines[1].split()
    site = l1[0] if l1 else ""
    start_time = f"{l1[1]} {l1[2]}" if len(l1) >= 3 else ""
    stop_time = f"{l1[3]} {l1[4]}" if len(l1) >= 5 else ""

    def _f(seq, i, cast, default):
        try:
            return cast(seq[i])
        except (IndexError, ValueError):
            return default

    altitude = _f(l1, 5, int, 0)
    longitude = _f(l1, 6, float, 0.0)
    latitude = _f(l1, 7, float, 0.0)
    zenith = _f(l1, 8, float, 0.0)
    azimuth = _f(l1, 9, float, 0.0)

    # Line 2: shots ... num_datasets ...
    l2 = header_lines[2].split()
    shots = _f(l2, 0, int, 0)
    num_datasets = _f(l2, 4, int, 0)

    # Descriptor lines (skip the blank trailing line).
    datasets: List[DatasetMeta] = []
    for line in header_lines[3:]:
        if line.strip() == "":
            continue
        meta = _parse_descriptor(line)
        if meta is not None and meta.active:
            datasets.append(meta)

    if not datasets:
        raise ValueError(f"{path.name}: no dataset descriptors found in header.")

    # Binary body: each dataset = ndata * uint32 (LE), separated by b"\r\n".
    raw: Dict[str, np.ndarray] = {}
    p = body_start
    for meta in datasets:
        nbytes = meta.ndata * 4
        if p + nbytes > len(data):
            raise ValueError(
                f"{path.name}: truncated body reading dataset {meta.device} "
                f"(need {nbytes} bytes at offset {p}, have {len(data) - p})."
            )
        arr = np.frombuffer(data[p:p + nbytes], dtype="<u4").astype(np.float64)
        raw[meta.device] = arr
        p += nbytes
        # Skip the CR/LF separator (be lenient: 0, 1 or 2 of \r \n).
        j = 0
        while p + j < len(data) and data[p + j] in (13, 10) and j < 2:
            j += 1
        p += j

    return LicelRawFile(
        path=path, filename=filename, site=site, start_time=start_time,
        stop_time=stop_time, altitude_m=altitude, longitude=longitude,
        latitude=latitude, zenith=zenith, azimuth=azimuth, shots=shots,
        num_datasets=num_datasets, datasets=datasets, raw=raw,
        header_lines=header_lines,
    )


# -----------------------------------------------------------------------------
# Physical scaling
# -----------------------------------------------------------------------------

def _analog_scale_mv_per_count(meta: DatasetMeta) -> float:
    """mV per count: fullscale_mV / 2**adc_bits. input_range like '0.500' = 0.5 V."""
    fullscale_mv = float(meta.input_range) * 1000.0
    return fullscale_mv / (1 << meta.adc_bits)


def scale_analog(raw_counts: np.ndarray, meta: DatasetMeta) -> np.ndarray:
    """Accumulated analog counts -> mV."""
    shots = meta.shots if meta.shots > 0 else 1
    return raw_counts / shots * _analog_scale_mv_per_count(meta)


def scale_photon(raw_counts: np.ndarray, meta: DatasetMeta) -> np.ndarray:
    """Accumulated photon counts -> MHz."""
    shots = meta.shots if meta.shots > 0 else 1
    return raw_counts / shots * (_PHOTON_RANGE_CONST / meta.binwidth_m)


def scale_stderr(sqd_bin: np.ndarray, meta_signal: DatasetMeta, unit_scale: float) -> np.ndarray:
    """Squared dataset (already sqrt(N*sum(x^2)-sum(x)^2)) -> StErr in signal units.

    ``meta_signal`` supplies the shot count; ``unit_scale`` is the same physical
    scale factor used for the corresponding signal (mV/count or MHz per count).
    """
    n = meta_signal.shots
    if n <= 1:
        return np.zeros_like(sqd_bin)
    divider = np.sqrt(n * (n - 1)) * np.sqrt(n)
    return sqd_bin / divider * unit_scale


# -----------------------------------------------------------------------------
# High-level: raw file -> the same ASCII-style column array / DataFrame
# -----------------------------------------------------------------------------

def _find_channel_devices(rf: LicelRawFile) -> Dict[int, Dict[str, str]]:
    """Group device tokens by TR index into {tr: {'analog','photon','asq','psq'}}."""
    groups: Dict[int, Dict[str, str]] = {}
    for meta in rf.datasets:
        tr = meta.tr_index
        if tr is None:
            continue
        g = groups.setdefault(tr, {})
        if meta.dtype == DT_ANALOG:
            g["analog"] = meta.device
        elif meta.dtype == DT_PHOTON:
            g["photon"] = meta.device
        elif meta.dtype == DT_ANALOG_SQ:
            g["asq"] = meta.device
        elif meta.dtype == DT_PHOTON_SQ:
            g["psq"] = meta.device
        elif meta.dtype == DT_OVERFLOW:
            g["overflow"] = meta.device
    return groups


def raw_file_to_array(rf: LicelRawFile) -> np.ndarray:
    """Build the (bins, 9) array matching the Advanced Viewer ASCII layout:

        col 0-3 : analog_par, analog_stderr_par, photon_par, photon_stderr_par
        col 4-7 : analog_cross, analog_stderr_cross, photon_cross, photon_stderr_cross
        col 8   : overflow flag

    TR0 is taken as parallel, TR1 as cross (project convention). Std-error
    columns are zero-padded to the full bin count. An overflow dataset, if
    present, fills col 8; otherwise col 8 is zeros.
    """
    meta_by_dev = {m.device: m for m in rf.datasets}
    groups = _find_channel_devices(rf)
    tr_indices = sorted(groups.keys())
    if not tr_indices:
        raise ValueError(f"{rf.filename}: no transient-recorder datasets found.")

    # Number of bins = length of the analog signal of the first TR.
    first_analog_dev = groups[tr_indices[0]].get("analog")
    if first_analog_dev is None:
        raise ValueError(f"{rf.filename}: first TR has no analog dataset.")
    nbins = int(rf.raw[first_analog_dev].size)

    def _col_signal(dev: Optional[str], kind: str) -> np.ndarray:
        out = np.zeros(nbins)
        if dev is None:
            return out
        m = meta_by_dev[dev]
        vals = scale_analog(rf.raw[dev], m) if kind == "analog" else scale_photon(rf.raw[dev], m)
        out[:vals.size] = vals[:nbins]
        return out

    def _col_stderr(sq_dev: Optional[str], sig_dev: Optional[str], kind: str) -> np.ndarray:
        out = np.zeros(nbins)
        if sq_dev is None or sig_dev is None:
            return out
        sig_meta = meta_by_dev[sig_dev]
        scale = (_analog_scale_mv_per_count(sig_meta) if kind == "analog"
                 else _PHOTON_RANGE_CONST / sig_meta.binwidth_m)
        vals = scale_stderr(rf.raw[sq_dev], sig_meta, scale)
        out[:vals.size] = vals[:nbins]
        return out

    cols = [np.zeros(nbins) for _ in range(9)]
    # Parallel = lowest TR index, cross = next.
    layout = [(tr_indices[0], 0)]
    if len(tr_indices) > 1:
        layout.append((tr_indices[1], 4))

    overflow_dev = None
    for tr, base in layout:
        g = groups[tr]
        cols[base + 0] = _col_signal(g.get("analog"), "analog")
        cols[base + 1] = _col_stderr(g.get("asq"), g.get("analog"), "analog")
        cols[base + 2] = _col_signal(g.get("photon"), "photon")
        cols[base + 3] = _col_stderr(g.get("psq"), g.get("photon"), "photon")
        if overflow_dev is None:
            overflow_dev = g.get("overflow")

    if overflow_dev is not None:
        of = rf.raw[overflow_dev]
        cols[8][:of.size] = of[:nbins]

    return np.column_stack(cols)


def read_licel_raw_array(path: str | Path) -> np.ndarray:
    """Convenience: decode a raw file straight to the (bins, 9) column array."""
    return raw_file_to_array(parse_raw_file(path))


def read_licel_raw_dataframe(
    path: str | Path,
    *,
    dr_m: float = 3.75,
    channel: str = "parallel",
    first_signal_range_m: float = 3.75,
) -> pd.DataFrame:
    """Decode a raw file to a DataFrame matching ``read_tr40_dat_ascii``'s output
    (plus an ``overflow`` column) for the requested polarization channel."""
    arr = read_licel_raw_array(path)
    ch = str(channel).strip().lower()
    if ch in ("perpendicular", "perp", "cross", "s", "l"):
        c0 = 4
    elif ch in ("parallel", "par", "co", "p", ""):
        c0 = 0
    else:
        raise ValueError(f"Unknown channel {channel!r}; use 'parallel' or 'perpendicular'.")

    n = arr.shape[0]
    range_m = float(first_signal_range_m) + np.arange(n, dtype=float) * float(dr_m)
    return pd.DataFrame({
        "bin_index": np.arange(1, n + 1),
        "source_bin_index": np.arange(1, n + 1),
        "range_m": range_m,
        "analog_mV": arr[:, c0 + 0],
        "analog_stderr_mV": arr[:, c0 + 1],
        "photon_MHz": arr[:, c0 + 2],
        "photon_stderr_MHz": arr[:, c0 + 3],
        "overflow": arr[:, 8],
    })


# -----------------------------------------------------------------------------
# Self-test / CLI: decode a file and (optionally) compare to an ASCII export
# -----------------------------------------------------------------------------

def _compare_to_ascii(raw_path: Path, ascii_path: Path) -> None:
    arr = read_licel_raw_array(raw_path)
    nbins = arr.shape[0]
    rows = []
    for line in ascii_path.read_text(errors="replace").splitlines():
        toks = line.split()
        if len(toks) < 9:
            continue
        try:
            rows.append([float(x) for x in toks[:9]])
        except ValueError:
            continue
    A = np.asarray(rows[-nbins:])
    names = ["an_par", "anErr_par", "ph_par", "phErr_par",
             "an_cross", "anErr_cross", "ph_cross", "phErr_cross", "OF"]
    print(f"decoded {raw_path.name}  vs  {ascii_path.name}")
    print(f"{'col':<13}{'max|diff|':>14}{'max rel err':>14}")
    ok = True
    for i in range(9):
        d, a = arr[:, i], A[:, i]
        absmax = float(np.nanmax(np.abs(d - a)))
        denom = np.where(np.abs(a) > 1e-9, np.abs(a), np.nan)
        rel = float(np.nanmax(np.abs(d - a) / denom))
        flag = "" if (not np.isfinite(rel) or rel < 1e-3) else "  <-- CHECK"
        ok = ok and (not np.isfinite(rel) or rel < 1e-3)
        print(f"{names[i]:<13}{absmax:>14.4g}{rel:>14.3g}{flag}")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Decode a Licel raw binary file.")
    ap.add_argument("raw", help="path to a raw file (e.g. a2682711.150166)")
    ap.add_argument("--ascii", help="optional ASCII .dat export to validate against")
    args = ap.parse_args()

    rf = parse_raw_file(args.raw)
    print(f"file      : {rf.filename}")
    print(f"site      : {rf.site}   shots={rf.shots}   datasets={rf.num_datasets}")
    print(f"time      : {rf.start_time} -> {rf.stop_time}")
    print("datasets  :")
    for m in rf.datasets:
        print(f"  {m.device:<5} dtype={m.dtype} ndata={m.ndata:>6} "
              f"adc={m.adc_bits:>2} shots={m.shots} range={m.input_range} "
              f"wl={m.wavelength_nm}.{m.polarization}")
    arr = raw_file_to_array(rf)
    print(f"array shape: {arr.shape}  (bins x 9 columns)")
    if args.ascii:
        print()
        _compare_to_ascii(Path(args.raw), Path(args.ascii))
