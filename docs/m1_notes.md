# M1 notes: preprocessing, denoising, features

## Pipeline per file
load (mono, resample to 16 kHz, no amplitude normalisation) -> remove DC -> zero-phase band-pass
100 Hz to 7 kHz (Butterworth, 4th order per side, SOS) -> denoise variant -> 1.0 s windows, 0.5 s hop
(full windows only) -> 146 features per window -> `data/features/features_<variant>.npz`.

Live engine (M5) uses `StreamingBandpass`: causal `sosfilt` with state carried between chunks,
tested to match one-shot `sosfilt` exactly.

## Denoising variants (all from scratch, `src/sentinel/denoise.py`)
- `none`: band-pass only.
- `specsub`: Boll-style power spectral subtraction, over-subtraction alpha 2.0, floor beta 0.02, gain capped at 1.
- `wiener`: gain xi/(1+xi), decision-directed a priori SNR (0.98), minimum gain 0.1.
- `svd_energy`: truncated SVD of the magnitude spectrogram per 2 s block, rank = smallest k reaching 95 % of sum of s^2.
- `svd_elbow`: same, rank = elbow of the singular value curve (point farthest from the chord).
- specsub and wiener share a rolling noise estimate: mean power of the quietest 20 % of frames in a trailing 2 s window,
  causal after a 0.5 s warm-up (the first 0.5 s learn from the first 0.5 s).

All numeric defaults are untuned engineering choices. The ablation (M3) and the SVD sweep (M2) decide what is kept.

## Known weaknesses (measured or tested, not hypothetical)
1. **A steady hum looks like noise.** The quiet-frame noise estimate absorbs any sound that is present for a whole window.
   `tests/test_denoise.py::test_stationary_tone_is_absorbed_into_noise_estimate` shows a constant tone is mostly removed by
   specsub. On the real ESC-50 airplane example in `results/figures/denoise_examples.png`, specsub and Wiener cut the
   clip power substantially while a bursty hen clip is untouched. Drones are steady tonal sources, so these two variants may
   remove what the SVM needs. M3 measures it. Mitigations to try later: a longer noise window, and freezing the noise
   estimate while an alert is active (M5).
2. **Clips shorter than one window give zero windows.** They are counted in `files_without_a_full_window` in
   `results.json` and contribute nothing. DADS drone clips average 0.6 s, so this rule must be revisited once DADS is
   configured (concatenate neighbours of one source, pad, or drop).
3. **Level features.** MFCC 0, RMS and spectral contrast carry loudness, which can differ between data sources. M3 should
   include an ablation without them to check for source shortcuts.
4. **SVD energy mode on a non-centred magnitude spectrogram** keeps the mean level component, so 95 % energy is reached at a
   small rank and the denoising is mild. The M2 sweep (90 to 99 %) quantifies it.
5. **Padded and near-silent windows (measured with `scripts/audit_silence.py` on the full ESC-50 cache).**
   Some clips are padded with digital silence (3 of 960 files are at least 10 % exact zeros, 1 at least 50 %, all ambient).
   Counting windows with mean RMS below 1e-4 (-80 dBFS): aircraft 8 of 720 (1.1 %, in 1 file), ambient 338 of 5760 (5.9 %,
   26 files with half or more such windows), hard negatives 21 of 2160 (1.0 %). Silence labelled ambient is semantically fine,
   but a silent window labelled aircraft (or drone) is label noise. Suggested M3 handling: drop near-silent windows from the
   aircraft and drone classes only, and report the ambient ones. Not applied yet.
   The SVD rank code used to divide by zero on an all-zero block (the cause of the RuntimeWarning in the first full run, and it
   inflated the mean rank for those blocks). Fixed in this patch and covered by tests.

## Not done yet
- The drone row of the before/after figure is skipped until drone clips are in the manifest.
- Own recordings are included in the cache if present in the manifest, but they are held out from training by config.

## Update after M2
Weaknesses 1 and 4 above were measured in M2 (see `docs/m2_notes.md`): the hum-absorption problem of specsub and Wiener on
steady harmonics, and the dominance of the first uncentred singular value on real clips.
