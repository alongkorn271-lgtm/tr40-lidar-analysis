# Licel StErr / Squared-Data Limitation at High Shot Counts (30 m acquisitions)

**Status:** confirmed and worked around · **Applies to:** TR40‑16bit‑3U, squared‑data (standard‑deviation) acquisition · **First seen:** Case C02 (30 m, 300 s, 2026‑09‑09)

---

## TL;DR

When an acquisition runs a **large number of shots combined with high counts‑per‑bin**
(our 30 m / 300 s case: 5 988 shots, ~40 counts/shot in the near field), the Licel
**photon squared data is corrupted**, so the **photon standard error (StErr) — and therefore SNR — is unusable**.

- This is **not a bug in our reader.** Licel's own *Advanced Viewer* produces the same
  garbage StErr (a constant ≈ 36 618 MHz).
- The **raw signals (photon MHz and analog mV) are 100 % correct** — only the StErr is affected.
- The **analog** StErr is fine; only the **photon** StErr breaks.
- **Fix applied:** the processing now falls back to a **Poisson shot‑noise StErr** for the
  photon channel when the hardware squared data is corrupt. 3.75 m / low‑shot files are unchanged.

---

## What is affected vs. what is trustworthy

| Quantity | 3.75 m (C01, 2 392 shots) | 30 m (C02, 5 988 shots) |
|---|---|---|
| Photon signal (MHz) | ✅ correct | ✅ correct |
| Analog signal (mV) | ✅ correct | ✅ correct (a few near‑field bins ADC‑clipped, flagged by the overflow dataset) |
| NRB, profile shape, correlation vs Mini‑MPL | ✅ correct | ✅ correct (r = 0.98 at night) |
| Analog StErr | ✅ correct | ✅ correct |
| **Photon StErr / SNR (from hardware)** | ✅ correct | ❌ **corrupt** → use Poisson |

Verification: our reader reproduces the Advanced Viewer ASCII export to a relative error
of ~5 × 10⁻⁶ across every column, including the corrupt photon StErr — i.e. we read exactly
what Licel writes.

---

## Root cause — 32‑bit overflow in the squared sum

Per the *Ethernet Controller* manual (pages 66, 170–171), the squared datasets **S2A / S2P**
store, for each bin,

```
sqd_bin = s · √(N(N−1)) = √( N·Σx² − (Σx)² )
```

where `N` = number of shots, `x` = counts per shot, `s` = sample standard deviation. The
standard error of the mean is then `σ_µ = sqd_bin / (√(N(N−1))·√N)`. These are 32‑bit unsigned
integers stored little‑endian.

For the **photon** channel at high `N` and high counts the intermediate term overflows:

| Case | shots `N` | counts/bin | **N·Σx²** | 32‑bit limit (2³²) | overflow |
|---|---|---|---|---|---|
| C01 3.75 m | 2 392 | ~5 | 1.7 × 10⁸ | 4.29 × 10⁹ | no ✅ |
| **C02 30 m** | 5 988 | ~40 | **5.9 × 10¹⁰** | 4.29 × 10⁹ | **yes (14×)** ❌ |

The *final* result `N·Σx² − (Σx)² ≈ N·c_bin ≈ 1.4 × 10⁹` would fit in 32 bits, but the
**intermediate `N·Σx²` (≈ 5.9 × 10¹⁰) overflows before the subtraction**, so the value written
to the file is garbage (a near‑constant ≈ 3.36 × 10⁹ that scales to StErr ≈ 36 618 MHz).

**Why analog survives:** analog ADC values are large, so the variance is a large fraction of
`N·Σx²` (no catastrophic cancellation), and the analog squared uses a wider (48‑bit / 3‑word)
representation.

**Threshold.** To keep `N·Σx² < 2³²` at 30 m (~40 counts/bin) the shots must be below
**≈ 1 600**. The Licel *Shot Limit* setting (4 094 vs 65 534) is not enough on its own — at
4 094 shots the 30 m case still overflows. In short: **30 m + long integration + hardware
photon StErr cannot coexist** on this hardware.

---

## Noise floor: pre‑trigger vs. far range (measured)

Night‑time photon noise level (mean and bin‑to‑bin std over all valid night profiles):

| | Pre‑trigger region | Far range | Difference |
|---|---|---|---|
| **C02 30 m** (far = 25–30 km) | mean 0.111 MHz, std 0.020 MHz | mean 0.140 MHz, std 0.032 MHz | −0.03 MHz (≈ equal) |
| **C01 3.75 m** (far = 13–15 km) | mean 0.135 MHz, std 0.208 MHz | mean 1.101 MHz, std 0.901 MHz | −0.97 MHz |

Interpretation:

- **C02 (30 m → 30 km):** the pre‑trigger baseline (0.11 MHz) and the true far‑range background
  (0.14 MHz) are **essentially the same**, and both are very low/clean. The 30 km range is long
  enough to reach the real background floor beyond the atmospheric return — so either region is a
  valid background reference (the pre‑trigger is marginally more stable, std 0.020 vs 0.032).
- **C01 (3.75 m → 15 km):** the "far range" at 13–15 km is **not background** — it still carries
  ~1.1 MHz of atmospheric signal (8× the pre‑trigger, with 0.9 MHz of scatter). At night the
  15 km ceiling cuts off **before** the signal has decayed to the background floor.
- Daytime: pre‑trigger and far range are both ~90–102 MHz (solar background dominates both,
  difference ~0), for both cases.

Practical consequence: for background subtraction, C01's short range makes the far‑range
background unreliable at night (use the pre‑trigger); C02's long range gives a clean far‑range
floor that agrees with the pre‑trigger.

---

## Fix applied (processing side)

`licel_binary_reader.py` now auto‑detects a corrupt photon squared dataset (photon StErr ≥ the
signal itself at strong‑signal bins) and replaces the photon StErr with the **Poisson shot‑noise
estimate**

```
σ_µ (photon) = √(accumulated counts) / shots × (150 / bin_width)
```

which uses only the always‑valid accumulated counts. 3.75 m / low‑shot files keep their hardware
StErr unchanged (detection leaves them alone — verified bit‑identical with the repair on and off).

**It is a user setting, not a hidden behaviour.** The Depolarization page has a checkbox,
*"Repair corrupt photon StErr with the Poisson shot‑noise estimate"*, **ticked by default**.
Untick it to process exactly what the recorder wrote — the same values the Advanced Viewer shows,
corruption included. In code the policy is `licel_binary_reader.set_poisson_stderr_mode(...)` or the
per‑call `poisson_stderr=` argument, with three modes:

| mode | behaviour |
|---|---|
| `auto` (default) | repair only where the hardware StErr is detectably corrupt |
| `off` | never repair — report the raw hardware value |
| `always` | always use the Poisson estimate |

Measured on the 2026‑09‑09 30 m batch: 11 of 12 non‑empty files repaired (StErr 36 160 → 0.408 MHz,
0.20 % of a 200 MHz signal); all 3.75 m files unchanged.

**Result on C02 (30 m):**

- Photon StErr at the peak: **36 618 MHz → 0.41 MHz** (0.20 % of a 200 MHz signal — physically correct).
- Peak SNR now lands at the real signal peak (3 990 m) instead of a spurious far‑range bin.
- Night profiles: valid, SNR ~240, trusted range 15–29 km.
- **Dawn profiles (06:00–07:00) recovered** — they had been wrongly discarded by the SNR gate that
  was fed the corrupt StErr.
- Full‑daylight profiles (07:30–16:00) remain empty — but now for the correct physical reason
  (solar background ~102 MHz buries the signal), not a StErr artifact.

**Caveat — the Poisson StErr is conservative at high count rates, not a lower bound.**
Photon-counting dead time makes the counts sub-Poissonian, so in the strong near-field signal
the plain Poisson StErr is larger than the true one. Measured on Case 01 (3.75 m), where the
hardware StErr is valid:

| count rate | Poisson / hardware | 1 / (1 − n·τ), τ = 3.06 ns | Poisson × (1 − n·τ) / hardware |
|---|---|---|---|
| 100–150 MHz | 1.68 | 1.53 | 1.09 |
| 150–250 MHz | 1.94 | 2.01 | 0.95 |

So on Case 02 the repaired SNR near the signal peak is ~1.5–2× **pessimistic**. Multiplying the
Poisson value by (1 − n·τ) tracks the hardware StErr to within ~10 %. At low count rates the
dead-time factor is ≈ 1 and the two estimates coincide.

**Which files are affected.** Only Case 02 (30 m, ~6 000 shots) hits the 32-bit limit. Case 01
(3.75 m, ~2 400 shots) is unaffected: all its files keep their valid hardware StErr. A few very
short files at the start of a Case 02 run (51 and 92 shots, with a dataset reporting shots = 0)
also carry a garbage StErr, but for a different reason — they are aborted/partial acquisitions,
far below the overflow threshold.

---

## Options at acquisition (if hardware StErr is required)

Reduce `N·Σx²` below 2³²:

1. **Shot Limit → 4k** (config dialog) *and* keep shots low (≈ ≤ 1 600 for 30 m count rates).
2. **Shorter integration** (fewer shots) — costs SNR averaging.
3. **Lower PMT HV** — fewer counts/bin, but weaker signal.
4. **Turn off Squared Data** for the run (config dialog) — clean files, no garbage; rely on the
   Poisson StErr in processing.
5. **Smaller bins** (data reduction / frequency divider) — fewer counts/bin, but abandons 30 m.

None of these keep *30 m + high SNR + valid hardware photon StErr* together.

---

## To report to Licel

- The squared‑sum standard‑deviation computation overflows 32‑bit for the **photon** channel when
  `N·Σx²` exceeds 2³² (high shots × high counts/bin). The final result fits in 32 bits — only the
  intermediate does not, so a 64‑bit intermediate would fix it.
- Analog is unaffected (48‑bit / 3‑word path).
- Reproduced with 30 m / 300 s / 5 988 shots; Advanced Viewer shows the same constant garbage StErr.
- Manual references: squared‑data format & standard‑deviation derivation, *Ethernet Controller*
  manual pp. 66, 170–171; `programmingManual.pdf §5.3` (raw → physical conversion).
