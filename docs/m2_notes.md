# M2 notes: SVDPCA, SVD denoising experiment, variance analysis

Reproduce: `make m2` (about 2.5 min for the experiment on 2 cores, 8 s for PCA). Every number below is in
`results/results.json` (`svd_experiment`, `pca_variance`) and in `results/tables/*.csv`.

## SVDPCA (`src/sentinel/svdpca.py`)
PCA from scratch with `numpy.linalg.svd` on the centred matrix, plus `UncenteredSVD` (the TruncatedSVD quantity). Tests compare
mean, components (up to sign), singular values, explained variance and ratio, transform, inverse transform, float
`n_components` and whitening against scikit-learn, and check orthonormality, Eckart-Young, wide matrices and bad input.
Provenance: `svdpca.py` and `tests/test_svdpca.py` were in the project tree when M2 started and were not written in the
conversation. They were read in full and tested before use. One assertion was wrong (it claimed the uncentred first
component holds more than 90 % of the energy, the value is 0.88) and was replaced with a property that holds
(Rayleigh quotient bound).

## What the SVD denoising experiment is, and is not
- **Controlled part.** A synthetic clean signal is mixed with noise at a fixed SNR (measured after the band-pass), and each
  method is scored against the known clean magnitude spectrogram. Two synthetic signals: steady rotor-like harmonics
  (f0 120 Hz with drift and 4 Hz amplitude modulation) and impulsive bursts. Noise: white (3 draws per cell) and 8 real ESC-50
  ambient clips (rain, wind, sea waves, chirping birds, 2 each, padded clips skipped; 24 values per cell).
  **The synthetic signals are not drone recordings.** They show how each method treats steady and impulsive content.
- **Real clips.** There is no clean reference, so only rank kept, retained power and centring statistics are reported (30 clips
  per group). These are not quality scores.
- Metric: magnitude SNR gain in dB (positive is better). The time-domain gain is also saved. Noisy phase is kept by all
  methods, so time-domain gains are smaller. Neither metric is classification accuracy; M3 measures that.

## Findings (magnitude SNR gain, dB, mean over cells; full tables in the CSVs)
Steady harmonics, input SNR 0 / 10 / 20 dB:
| method | white | ambient |
|---|---|---|
| specsub | +1.98 / -7.64 / -17.80 | +0.27 / -7.74 / -17.75 |
| Wiener | -0.09 / -10.64 / -20.92 | -1.01 / -10.75 / -20.89 |
| SVD elbow | +2.10 / +2.68 / +2.98 | +1.07 / +1.86 / +2.82 |
| SVD 95 % energy (default) | +0.85 / +2.53 / -1.85 | +0.70 / +2.59 / -1.13 |

1. **Confirmed: specsub and Wiener destroy steady harmonic content when the SNR is good.** At 20 dB input they lose 17.8 and
   20.9 dB, in white and in real ambient noise alike. This is the hum-absorption weakness from M1, now measured. They help
   at -5 dB (about +5 dB in white noise) and on the impulsive signal (+3.8 dB specsub at 20 dB in white noise).
2. **The energy-share SVD rule has a built-in ceiling.** It always discards the same energy fraction, so at high SNR it
   throws away signal: the 90 % rule loses 4.5 dB and the 95 % rule 1.9 dB at 20 dB on steady harmonics, and on bursts
   the 90 % rule loses 6.9 dB. The 99 % rule does little at any SNR (+0.2 dB at 0 dB on steady harmonics).
3. **The elbow rule is the safest SVD setting on steady content** (positive at every SNR, +1.9 to +3.0 dB in white noise)
   but still loses on bursts at 10 dB and above (-0.9 and -7.9 dB at 10 and 20 dB, white). Impulsive content is not low rank.
4. **Rank sweep (0 dB input).** The best fixed rank was 2 and 1 for steady harmonics (white, ambient) and 6 and 4 for bursts.
   The elbow rule keeps about 3 (steady) and 5.6 (bursts) on average, close to those. The 95 % energy rule keeps about 37 at
   0 dB, far past the best. Full rank gives 0.00 dB gain, as it must.
5. **On real ESC-50 clips the SVD denoisers barely change the signal.** Mean rank kept is 9 to 11 for the elbow rule and
   12 to 15 for 95 % energy (out of up to 250), and the waveform power drops by only 0.3 to 0.5 dB. Expect the SVD variants
   to land close to `none` in M3.
6. **Centring matters.** On real clips the first uncentred singular value holds 76 to 82 % of the block energy; after removing the
   per-frequency time mean it holds 34 to 48 %, and the rank for 95 % energy rises from 12 to 14 to 24 to 35. The 95 % rule
   on the raw spectrogram therefore mostly counts the mean level.
   Not tested: whether the noise floor sitting in that mean component is why the best SVD gain is only about 2 to 3 dB.

## PCA variance of the cached features (z-scored, descriptive, no labels, scaler and PCA on all windows)
- Needed for 90 / 95 / 99 % of the variance, band-pass only: 66 / 85 / 119 of 146 components. The other variants are within
  about 5 components of that (specsub the highest at 71 / 88 / 123). The features are not low dimensional.
- PC1 holds 30 % (band-pass only) and loads on the standard deviations of the delta MFCCs (temporal variability). Its
  Spearman correlation with `mfcc00_std` is 0.67 and with `rms_mean` 0.06. PC2 (5 %) correlates with `rms_mean` at -0.58 and
  with `mfcc00_mean` at -0.50, so it is partly a loudness axis.
- Removing the 18 level features (`mfcc00_*`, `rms_*`, `contrast*`) lowers the 95 % count from 85 to 77. Level information is
  spread across several components, so PCA alone will not remove a loudness shortcut. The M3 level-feature ablation does that.
- **Near-silent windows distort the scaling.** 367 windows have `rms_mean` below 1e-4 (359 ambient, 8 aircraft; this equals the
  M1 audit total). All 308 windows with PC1 below -15 are near-silent, and 84 % of the near-silent windows are in that stack.
  Refitting without them cuts PC1 from 30.1 to 25.8 % (band-pass only) and from 27.6 to 18.6 % for specsub.
- Aircraft windows form a tight cluster at low PC1 that overlaps ambient and the hard negatives (`pca_scatter.png`).
  Separation is visible but not clean in two dimensions. This is descriptive, not a result.

## Consequences for M3
- Fit the scaler and PCA inside each training fold. Sweep the PCA size (a few values from 10 up to about 90), and include
  "no PCA".
- Drop near-silent windows from the aircraft and drone classes (planned in M1), and decide how to treat near-silent ambient
  windows before fitting the scaler.
- Keep all five denoising variants in the grid. This experiment says specsub and Wiener are risky for steady sources and the
  SVD variants are mild on real audio. It does not say which wins on the classification task.
- The default `svd.energy` of 0.95 is not supported by the synthetic results (the elbow rule did better). It is left as is
  and will be decided by the M3 grid, not here.

## Limits
- Synthetic signals are simple. A real rotor has more structure, and a real drone recording could behave differently.
- Real-clip statistics have no ground truth. White-noise cells have 3 draws, ambient cells 24 values from 8 clips.
- The PCA numbers describe the ESC-50 feature matrix only. Drone data is still missing.
