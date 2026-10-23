# M4 notes: noise robustness, the final pipeline, calibrated window probabilities, latency and size

Reproduce: `make robustness` (about 22 min on 2 cores the first time, of which 19 min is the noisy features; 3 min when they are cached) and `make final`
(about 20 s). Every number below is in `results/results.json` (`robustness`, `final_model`) and `results/tables/m4_*.csv`.

**Scope: ESC-50 only, aircraft versus ambient.** There is no drone class yet. Nothing here is a drone result.
Numbers are quoted from the sandbox run. A second run on another machine agreed to within about 0.003 on balanced accuracy and
the rates (the largest gap was 0.009, the ambient -5 noise-only false-alarm rate); a few paired fold counts differ by one.

## Decision recorded: the final pipeline is the untuned default
`none` (band-pass only), all 146 features, PCA 40, RBF C=1, `class_weight=balanced`. It was fixed before the M3 grid ran, so its
cross-validated estimate (balanced accuracy 0.792 +/- 0.042 over folds) carries no selection bias. The nested selection did not
beat it (0.764), and the grid top is a tie. `m4.final.config` in `configs/default.yaml` switches to another setting with no
code change. The final model is fitted on all 8273 non-holdout windows (706 source recordings). The only honest estimate of how
it performs on unseen recordings is the cross-validated one; no number here is a held-out test of the saved file itself.

## Noise robustness
**Protocol.** Noise is added to the raw audio of every recording and the whole chain is rerun (band-pass, denoising variant,
features). The model of each fold is fitted on clean training windows only and scores the noisy windows of the recordings that
fold held out, with the same folds as `make train`. The `clean` row reproduces `make train` exactly (fold-mean balanced accuracy
0.792336 in both).
- **Noise level.** One absolute in-band (100 to 7000 Hz) noise power per condition, set against the median aircraft clip:
  `P_noise = P_ref / 10^(snr/10)`, `P_ref = 1.323e-02` (median of 80 aircraft clips). Every clip gets the same noise level, as a
  microphone in a noisy place would. **The SNR on the axes is therefore nominal.** What clips actually received (in-band, in dB,
  from `robustness.clip_snr_db`): at nominal +15, aircraft clips ranged from -6.4 to +26.4 (10th to 90th percentile +6.7 to +20.2),
  median ambient clip +12.1. At nominal +5, 19 % of aircraft clips and 40 % of ambient clips are below 0 dB. At nominal -5, 86 %
  and 90 %. The noise is the same shape at every SNR and for every denoising variant (paired).
- **Kinds.** White and pink (power 1/f) are synthetic. `ambient` is a real ESC-50 recording (rain, wind, sea waves or birds) from
  a different source group, 158 usable recordings. The SVM was trained with those classes as negatives, so it has heard this kind
  of sound; read that column as optimistic.
- **Near-silent windows.** The silence gate removes 359 near-silent ambient windows from the main metrics, as in M3. Under noise
  they contain only the added noise, so they are reported separately as `noise_only_far` (share called aircraft).

**Pooled out-of-fold balanced accuracy, trained on clean audio** (95 % group-bootstrap CI in the CSV; chance is 0.5):

| variant | noise | clean | +15 | +5 | -5 |
|---|---|---|---|---|---|
| none | white | 0.792 | 0.698 | 0.592 | 0.527 |
| none | pink | 0.792 | 0.747 | 0.673 | 0.569 |
| none | ambient | 0.792 | 0.729 | 0.684 | 0.601 |
| wiener | white | 0.782 | 0.689 | 0.580 | 0.517 |
| wiener | pink | 0.782 | 0.747 | 0.663 | 0.554 |
| wiener | ambient | 0.782 | 0.711 | 0.656 | 0.584 |
| svd_elbow | white | 0.808 | 0.730 | 0.616 | 0.510 |
| svd_elbow | pink | 0.808 | 0.758 | 0.699 | 0.570 |
| svd_elbow | ambient | 0.808 | 0.760 | 0.695 | 0.604 |
| none, no level features | white | 0.757 | 0.669 | 0.551 | 0.503 |
| none, no level features | pink | 0.757 | 0.720 | 0.620 | 0.518 |
| none, no level features | ambient | 0.757 | 0.703 | 0.646 | 0.575 |

**1. Noise costs a lot, and the damage is not the same for every noise.** Balanced accuracy falls from 0.79 to 0.70 at nominal
+15 white and reaches chance at -5. For the default model, with the native decision boundary:
- White noise pushes the decision toward ambient: aircraft recall 0.751 (clean), 0.494, 0.243, 0.084; ambient specificity rises
  from 0.872 to 0.983. Noise-only windows are never called aircraft. AUC falls from 0.878 to 0.852 at +15 (mostly a threshold
  shift, the ranking survives) and to 0.693 at -5 (information is lost too).
- Pink and real ambient noise push the other way at high levels: specificity falls (pink -5: 0.506, ambient -5: 0.712) and
  noise-only windows are called aircraft 41.2 % (pink -5) and 34.5 % (ambient -5) of the time. Low-frequency broadband noise
  reads as aircraft rumble. AUC is 0.600 and 0.644 there.
- A threshold set on clean audio does not hold its false-alarm rate under noise. That matters for M5 and M6.

**2. Denoising: SVD elbow helps a little at mild noise, Wiener does not help.** Paired over the same folds (variant minus `none`):
- svd_elbow: clean +0.015 (4 of 5 folds); white +0.031 (+15) and +0.024 (+5), both 5 of 5 folds; ambient +0.031 (+15, 5 of 5) and
  +0.011 (+5, 4 of 5); pink +0.011 and +0.025 (4 of 5 each). At -5 it gives nothing: white -0.017 (0 of 5), pink +0.001, ambient
  +0.003. Part of the gain is the +0.015 it already has on clean audio.
- wiener: below `none` in almost every condition (-0.009 to -0.027), better in at most 3 of 5 folds, and 2 of 5 at the one cell
  where it ties.
- The effect sizes (0.01 to 0.03) are small next to the fold spread (0.04 to 0.07). Consistent in sign, not large. Neither
  denoiser fixes the noise-only false alarms or the white-noise recall collapse.

**3. The loudness features are not what noise breaks.** Dropping the 18 level features lowers accuracy in every condition
(clean 0.757 against 0.792) and white-noise recall still collapses (0.007 at -5). With them dropped, the noise-only
false-alarm rate is identical at all three SNRs (pink 0.011, ambient 0.214), as expected once gain no longer enters the
features, which is a sanity check on the noise scaling. The shape features themselves move with the noise. Paired against `none`, dropping the level features is worse on average in
all 10 conditions and in 0 of 5 folds in each of the 9 noisy ones (-0.024 to -0.055).

**4. Training with other noise kinds does not transfer reliably** (clean + the other kinds, tested on the held-out kind; each
training recording is clean or corrupted by one condition drawn at random; paired with clean-only training):
white held out +0.034, +0.048, +0.019 (4, 4, 3 of 5 folds); pink -0.019, +0.011, -0.008 (2, 2, 3 of 5); ambient -0.058, -0.041,
-0.002 (0, 0, 2 of 5). It helped on the noise kind that is the most unlike the others and hurt on real ambient noise. This is a
negative result about transfer to an unseen noise, not about noise-aware training with matched noise (not tested). The final model
stays clean-trained, as decided before looking.

## Calibrated window probabilities
`SVC(probability=True)` is not used. A Platt calibrator (`p = sigmoid(1.235 * score - 1.906)` for aircraft, from `final_model`) is
fitted on out-of-fold scores from the grouped CV (windows scored by a model that never saw their recording).
- Cross-fitted quality (the calibrator scoring a fold was fitted on the other folds): Brier 0.0616, ECE 0.0045, log loss 0.208.
  Baselines: constant prevalence Brier 0.0787, ECE 0.0172; the raw score through a sigmoid Brier 0.1173, ECE 0.1918.
  Calibration is good (`results/figures/m4_calibration_aircraft.png`); the Brier gain over the prevalence baseline is modest, which is
  the same limited discrimination as before.
- **The probabilities hold at ESC-50's window prevalence (8.6 % aircraft).** For another prevalence use
  `calibration.adjust_prior` (odds shift, tested against the Bayes formula). The calibration also assumes clean audio; under noise
  the score distribution shifts (finding 1) and it no longer holds.
- Multi-class (drone, aircraft, ambient) will be one-vs-rest probabilities per target class, which do not sum to 1.

## Operating points
Window-level false-alarm rate (FAR) is measured on ambient windows, plain and hard negatives together. The native SVM boundary
gives FAR 0.167 and recall 0.751. Table from pooled out-of-fold scores, with the cross-fitted check (threshold taken from the
other four folds' ambient windows, applied to the held-out fold: does a FAR target transfer to unseen recordings?):

| target FAR | threshold (p) | pooled recall | achieved FAR on held-out folds (mean, sd, max) | recall on held-out folds (mean, sd, min) |
|---|---|---|---|---|
| 0.01 | 0.489 | 0.169 | 0.010, 0.004, 0.016 | 0.170, 0.104, 0.042 |
| 0.02 | 0.427 | 0.232 | 0.019, 0.007, 0.032 | 0.231, 0.135, 0.083 |
| 0.05 | 0.302 | 0.416 | 0.049, 0.010, 0.066 | 0.410, 0.131, 0.278 |
| 0.10 | 0.203 | 0.587 | 0.101, 0.011, 0.116 | 0.585, 0.106, 0.479 |
| 0.20 | 0.108 | 0.785 | 0.202, 0.030, 0.249 | 0.792, 0.079, 0.722 |

The targets transfer well on clean ESC-50 audio, but the price is high: at a 5 % window FAR only 41 % of aircraft windows are kept.
The saved model uses `m4.operating.target_window_far = 0.05`, an engineering default: threshold p >= 0.3018, pooled FAR 0.050
(plain 0.036, hard negatives 0.086), recall 0.416 (95 % CI 0.293 to 0.547). **M6 sets the real operating point** from event-level
false alarms; the alert hysteresis, not this number, is the main lever.

## The saved model: `models/pipeline.joblib` (+ `pipeline.json`)
`SentinelModel`: FeaturePipeline (z-score, SVDPCA 40, SVM), calibrators, thresholds, silence gate, a snapshot of the feature
settings and a hash of them. `load_model(path, cfg)` raises if the config would compute different features, and warns on a
different sklearn or numpy minor version. The file reloads to identical probabilities (max difference 0.0), and `score_audio` on a
clip matches the feature cache (max difference below 1e-7 on two machines: 0.0 and 5.6e-8, 3 clips). The gate is the band-passed mean frame RMS < 1e-4, computed exactly
as the `rms_mean` feature (tested). The live engine (M5) will use causal filtering and a streaming denoiser; `score_audio` is the
offline chain.

**Size and latency** (this sandbox: 2 CPU cores, x86_64, Python 3.13, 1 thread, one window at a time; your machine will differ
and `make final` measures yours): file 963 KiB compressed, arrays 1004 KiB, 2985 support vectors (344 aircraft, 2641 ambient) of 40
dimensions. Per 1 s window, median / p95 (ms): features 6.62 / 9.79, decision 0.80 / 1.21, calibration 0.02 / 0.02, total 7.46 /
11.12, 1.5 % of the 500 ms hop. On another machine the same run gave a total of 4.67 / 5.42 ms (0.9 % of the hop). The gate statistic costs 1.03 ms more if computed on its own; for `none` it is the `rms_mean`
feature and costs nothing. The denoiser is not included (none needs none; M1 measured the offline cost of the others).

## What this means for M5 and M6
- The detector's false alarms depend on the background, and a fixed threshold calibrated on clean audio will not hold under
  wind-like noise. M6 must evaluate false alarms on noisy backgrounds, not only on the clean ESC-50 negatives. A noise-floor
  estimate in the engine (M5) could feed the gate or the threshold; this has not been tested.
- Event-level false alarms are not window-level ones. Windows from steady noise are strongly correlated, so a hysteresis that waits
  for several consecutive windows helps less than independent-window arithmetic suggests.

## Limits
- ESC-50 only, 47 aircraft recordings (9 or 10 per test fold). Intervals are wide and fold spread is as large as most effects here.
- The nominal SNR is against the median aircraft clip; individual aircraft clips span about 30 dB around it. Noise is added to
  clips, not to a continuous scene, and it is stationary within a clip.
- The `ambient` noise kind is made of classes the SVM was trained on. White and pink are synthetic, not field noise.
- Calibration is for clean audio at ESC-50 prevalence. The threshold is not a deployment threshold.
- The robustness grid has 3 variants, 3 kinds and 3 SNRs. specsub and svd_energy are not in it (add them to
  `m4.robustness.variants`; only the missing noisy features are computed).

## Housekeeping
- `configs/default.yaml` shipped in the M3 package with two `m3:` blocks. YAML keeps the last one, so the first (with `near_silent`,
  a different grid and `positive`) was dead, and nothing but the unshipped `models.py` read it. It is removed, and
  `tests/test_config.py` fails on any duplicate key. M3 results are unaffected (the surviving block is the one the grid ran with).
- `src/sentinel/models.py` and `tests/test_models.py` are still in the sandbox only, still unexplained, and not part of any package.
