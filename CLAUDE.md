# CLAUDE.md — TR40 LiDAR Analysis Suite

Context for anyone (and any Claude) working on this repo. Read before changing code.
`README.md` has the science write-up + screenshots; this file is the dev map +
the hard-won gotchas. Detailed decision history lives in Claude **memory**
(`~/.claude/projects/.../memory/`), which loads automatically — check it too.

## What this is
Research-grade processing for the **TR40 elastic Mie LiDAR (532 nm)** at NARIT,
Chiang Mai. Turns raw photon-counting / analog `.dat` (and Licel binary, and
Mini-MPL `.nc`) into **NRB** (normalized relative backscatter), **ALT** (aerosol
layer top), **RTI** imagery, and **Fernald/Klett** retrievals (β_aer, α_aer, AOD).
Desktop (Tkinter) + Streamlit web, same engines.

Repo: https://github.com/alongkorn271-lgtm/tr40-lidar-analysis

## Run
```bash
pip install -r requirements.txt
python main_v2.py                   # desktop GUI (current; main.py is older)
streamlit run streamlit_app.py      # web (pages/ = Step 1–5 auto-nav)
```
Raw measurement data (`MPL data/`, `Licel_TCPIP_Python/`) is **gitignored** — it
is local input, not on GitHub. Copy it separately when moving machines.

## Pipeline (see README for the science)
Step 1 MPL rmin-rmax → Step 2 NRB Profile → Step 3 ALT Detection →
Step 4 RTI Visualizer → Step 5 Fernald. (The GUI sidebar groups these; Raw QC is
its own step — displayed step numbers ≠ internal ids, see memory.)

NRB chain (Step 2): raw analog+photon → dead-time correction → toggle fit + cosine
glue → −background → −afterpulse A(R) → ÷overlap O(R) → ×R² → ÷energy → ÷max → NRB.

## Code map
```
main_v2.py / main.py       desktop GUI (Tkinter)      streamlit_app.py + pages/  web
nrb_engine.py              NRB derivation (+ auto-detect .dat vs Licel vs .nc dispatch)
licel_binary_reader.py     decode raw Licel TR binary directly (dual-PMT)
mpl_reader.py              Mini-MPL .nc ingest (netCDF4)
overlap.py / overlap_function.py   overlap O(R);  afterpulse.py  afterpulse A(R)
depol_engine.py            depolarization (co/cross, δ)
pbl_engine.py              PBL/ALT edge detection (FFT low-pass + HWCT Haar)
layer_classifier.py        cloud / layer classification
fernald_engine.py          Fernald/Klett inversion → β_aer, α_aer, AOD
aeronet_loader.py          AERONET AOD for constraint/validation
raw_quality_check.py / raw_quality_sheet.py   Raw QC step + workbook
validation_engine.py       cross-checks;  calc_theory.py  Rayleigh/molecular theory
bao_theme.py / gui_theme.py / modern_theme.py   UI themes
docs/                      analog range & HV, depol calibration, Licel stderr notes
```

## Critical gotchas (do NOT relearn the hard way — from memory)
- **Licel bin shift is firmware-applied** — do NOT re-shift analog before glue
  (shift 0 always gave best r² across 33 files).
- **MPL `.nc` → read with `netCDF4`, never xarray** (xarray was ~580× slower).
  MPL `lidar_ratio` is a fixed 30 sr.
- **Depol definition differs**: ours δ = cross/co; the SigmaMPL manual's MPL depol
  = cross/(cross+co). Convert before comparing.
- **UI labels are ∥/⊥**, but sheet names / columns / filenames / variables stay
  `co`/`cross`. Keep that split.
- **Profiles start at 150 m** (overlap O≈0.99 at 149 m); the ∥ chain reads low to
  ~340 m — still open.
- **Glue slope is an instrument constant** (~86 ∥ / ~95 ⊥); use robust median +
  night-gain pass. δ still ~2× low = calibration, not a bug.
- **Dead-time τ ≈ 4.8 ns** (measured); stderr propagates through dead-time as
  σ/(1−Nτ)² (Licel §9.3).
- **SNR gate** (Step 2/3) masks low-SNR bins + normalises within the SNR-trusted
  lower troposphere; default OFF (daytime NRB path defaults ON — check the file).
- New 2-PMT `.dat` carries ∥ (cols A–D) + ⊥ (E–H) in one file → reader `channel`
  param + depol `dual_channel_file` mode.
- Independent pipeline tracks get their **own GUI step**, not sub-cards (user pref).

## Conventions
- Money/units aside, this is scientific code — preserve physical correctness and
  keep the Excel workbook column meanings stable (downstream reads them).
- Commit only when asked; branch off `main`; don't push without being asked.
- Large raw data stays out of git (already gitignored).
