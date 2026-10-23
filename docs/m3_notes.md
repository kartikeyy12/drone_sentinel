# M3 notes: SVM with grouped cross-validation, ablation grid

Reproduce: `make train` (about 40 s) and `make ablation` (about 35 min on 2 cores; resumable, rerun picks up stored cells;
`python scripts/run_ablation.py --quick` is a 3 min smoke test). Every number below is in `results/results.json`
(`cv_default`, `ablation`) and `results/tables/m3_*.csv`.

**Scope: ESC-50 only, aircraft versus ambient.** There is no drone class yet. Nothing here is a drone result.

## Protocol
- Windows: 8640 cached, 8273 used. Rows removed by the silence gate (band-pass-only mean frame RMS below 1e-4, defined on the
  `none` cache so all five variants see identical windows): 359 near-silent ambient windows never reach the SVM and are not scored
  (crediting them as correct would inflate specificity), and 8 near-silent aircraft windows are dropped as label noise.
  Used: 712 aircraft windows from 47 source recordings, 7561 ambient windows from 659 recordings.
- Folds: `StratifiedGroupKFold(5, shuffle, seed 1337)` on the source recording id, identical for every configuration.
  `check_folds` raises if any recording is on both sides of a split or a class is missing from a side. Each test fold holds
  only 9 or 10 aircraft recordings, so fold-to-fold spread is large.
- Everything learned is fitted on the training part of the fold only: z-score, SVDPCA (from scratch), SVC with
  `class_weight=balanced`. Tests check that the fast sliced-PCA path used by the grid equals a standalone pipeline, that a
  different test set changes nothing that was learned, and that nested selection never reads outer-test labels.
- Sanity: shuffling labels across whole recordings and rerunning the default pipeline gives balanced accuracy 0.577, 0.468, 0.499
  (chance, as it should); the majority-class baseline is 0.500.

## Default pipeline (untuned: no denoising beyond band-pass, all 146 features, PCA 40, RBF C=1)
Balanced accuracy 0.792 +/- 0.042 over folds, 95 % group-bootstrap CI 0.734 to 0.852.
Aircraft recall 0.751 (CI 0.637 to 0.869), ambient recall 0.833, aircraft precision 0.299, F1 0.427, AP 0.436, AUC 0.878.
Ambient specificity 0.871 on plain ambient and 0.731 on the hard negatives (engine, train, vacuum cleaner, washing machine, chainsaw,
hand saw).

## Ablation grid (150 configurations x 5 folds)
Grid: 5 denoising variants x level features kept or dropped x PCA (none, 10, 20, 40, 80) x SVM (linear C=1, RBF C=1, RBF C=10).
- Balanced accuracy spans 0.650 to 0.815 across the grid; the median fold std of a cell is 0.062.
- **The top of the grid is a tie.** The best cell (svd_elbow, level features kept, PCA 20, RBF C=1) scores 0.815 +/- 0.033, and the
  top ten lie between 0.799 and 0.815. The untuned default scores 0.792 +/- 0.042 on the same folds. These differences are well inside
  the fold spread.
- Paired effects (same configuration, one factor changed, mean difference in balanced accuracy and the number of matched
  configurations won out of the total):
  | factor | effect | wins |
  |---|---|---|
  | RBF C=10 versus C=1 | -0.050 | 2 of 50 |
  | level features dropped | -0.018 | 5 of 75 |
  | spectral subtraction versus band-pass only | -0.013 | 7 of 30 |
  | Wiener versus band-pass only | -0.014 | 8 of 30 |
  | SVD elbow versus band-pass only | +0.017 | 26 of 30 |
  | SVD energy 95 % versus band-pass only | +0.007 | 20 of 30 |
  | PCA 10 / 20 / 40 / 80 versus no PCA | +0.015 / +0.015 / +0.007 / -0.011 | 18, 17, 18, 5 of 30 |
  | linear versus RBF at C=1 | -0.004 | 21 of 50 |
  Read these as directions, not as a ranking. Their size (0.01 to 0.02) is small against the 0.04 to 0.07 fold spread, but the
  sign is consistent across matched configurations for RBF C=10, level features, specsub, Wiener and the elbow rule.
  This agrees with M2: specsub and Wiener risk removing steady content, the SVD variants are mild. The strong RBF C=10 loss (0.727 mean
  over its 50 cells against 0.777 for C=1) is most simply read as overfitting to individual recordings.
- Level features: dropping the 18 loudness-related features costs 0.018 on average. That does not show whether they carry a
  real signal or a source shortcut (see the feature-group result below).

## Nested selection (the number to quote)
In each outer fold the configuration is chosen by 3-fold grouped CV inside the outer training part, then scored once on the outer
test part. Balanced accuracy **0.764 +/- 0.069** (pooled 95 % CI 0.697 to 0.829), aircraft recall 0.746 (CI 0.617 to 0.871),
aircraft precision 0.253, ambient specificity 0.823 plain and 0.671 on hard negatives.
- The best grid cell (0.815) overstates the selected procedure by 0.051: it was picked on the data it is scored on.
- The selection did not beat the fixed default. The default (0.792) was decided before the grid ran, so its estimate carries no
  selection bias, and 0.792 versus 0.764 is inside the noise. Selecting among 150 near-equal cells adds variance and no
  demonstrated skill. The five outer folds chose four different configurations (svd_elbow with PCA 20 in three of five, PCA 20 in
  four of five), and the fold that chose `none`, PCA 10, linear scored worst (0.655).
- **Fold-to-fold spread is as large as any effect here:** nested outer-fold balanced accuracy runs from 0.655 to 0.833.

## Feature groups (fixed pipeline: band-pass only, no PCA, RBF C=1; paired per-fold differences to all 146 features)
| subset | features | balanced accuracy | paired diff (mean +/- std) | folds better |
|---|---|---|---|---|
| all | 146 | 0.790 | 0 | 0 |
| without MFCC | 106 | 0.765 | -0.025 +/- 0.020 | 1 of 5 |
| without delta MFCC | 106 | 0.778 | -0.012 +/- 0.011 | 1 of 5 |
| without delta-delta MFCC | 106 | 0.812 | +0.022 +/- 0.019 | 5 of 5 |
| without spectral shape | 138 | 0.792 | +0.001 +/- 0.008 | 3 of 5 |
| without contrast | 132 | 0.782 | -0.008 +/- 0.015 | 2 of 5 |
| without ZCR and RMS | 142 | 0.792 | +0.002 +/- 0.008 | 3 of 5 |
| only MFCC | 40 | 0.792 | +0.002 +/- 0.039 | 3 of 5 |
| only delta MFCC | 40 | 0.698 | -0.092 +/- 0.038 | 0 of 5 |
| only delta-delta MFCC | 40 | 0.662 | -0.128 +/- 0.047 | 0 of 5 |
| only spectral shape | 8 | 0.746 | -0.044 +/- 0.073 | 1 of 5 |
| only contrast | 14 | 0.692 | -0.099 +/- 0.099 | 1 of 5 |
| only ZCR and RMS | 4 | 0.776 | -0.014 +/- 0.068 | 3 of 5 |
- **The static MFCCs carry the result.** Forty MFCC means and stds match all 146 features (0.792), and removing them costs the most.
- **Four features (ZCR and RMS, mean and std) reach 0.776**, within noise of the full set. Either aircraft clips are easy to tell
  from this ambient set by loudness and noisiness alone, or ESC-50 contains a source shortcut. This data cannot separate the two. Drone clips from
  a different source, and your own recordings, are the test.
- Removing the delta-delta MFCCs helps in all five folds (+0.022). One consistent group result on five folds; treat it as a
  lead to re-check on the final data, not as a decision.

## Where the errors are
Nested out-of-fold false-alarm rates on ambient windows: wind 77.5 %, train 63.1 %, thunderstorm 61.7 %, washing machine 50.7 %,
engine 39.3 %, car horn 32.5 %. Near zero: rooster, hand saw, laughing, clapping. The false alarms are steady broadband or
rumbling sounds, which is also what a drone detector will meet outdoors.
Aircraft precision of 0.25 to 0.30 is at ESC-50's class ratio (aircraft is 9 % of windows). A real perimeter has a far lower
prevalence of targets, so precision there will be lower. M6 (false-alarm evaluation) is where that gets measured; the SVM score
threshold and the alert hysteresis are the levers, not the grid.

## Decisions this leaves open
1. **M4 final pipeline.** The grid does not identify a winner. Options: keep the default, or take the configuration the nested
   selection chose most often (svd_elbow, PCA 20, RBF C=1). Both are within noise on ESC-50; the choice should be revisited
   once drone data exists.
2. **DADS and drone data** (still open). Without a drone class, M4 onward is built and tested on aircraft versus ambient.
3. **Score calibration and threshold.** M4 needs window probabilities; the plan is to calibrate decision scores on out-of-fold
   scores (not `SVC(probability=True)`), then set the operating threshold from a false-alarm target.

## Limits
- ESC-50 only, 47 aircraft source recordings, 9 or 10 per test fold. Confidence intervals are wide.
- Clips in ESC-50 are 5 s, so windows from one clip are correlated; grouping by source recording handles leakage, not that dependence.
- The silence gate uses the band-passed mean frame RMS from the features. The live engine will compute RMS in the time domain, which
  is close but not identical; M5 must test that definition.
- Nested selection used 3 inner folds, so selection ran on smaller training sets than the final fit.

## Provenance
`src/sentinel/models.py` and `tests/test_models.py` were in the sandbox tree when M3 work started (file times 2026-10-18 10:17 UTC) and
were not written in the conversation. They overlap with `dataset.py`, `cv.py` and `pipeline.py`, nothing imports them, and their
silence-mask test fails because it reads a config key (`m3.near_silent`) that `configs/default.yaml` does not define. They were
read, not used, and are not part of the M3 package. Remove them if you did not put them there.
