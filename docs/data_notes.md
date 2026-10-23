# Data notes (M0)

Status legend: VERIFIED = read from the source while writing M0. UNVERIFIED = could not be reached, do not rely on it.

## ESC-50 (VERIFIED, github.com/karolpiczak/ESC-50)
- Layout: `audio/*.wav` (2000 files), `meta/esc50.csv` with columns `filename,fold,target,category,esc10,src_file,take`.
- README: 5 s clips, 44.1 kHz, mono. 40 clips per category. All 24 categories requested in the spec exist (counts checked in the CSV).
- Licence (README and LICENSE): CC BY-NC 3.0 for the whole dataset, CC BY for the ESC-10 subset. Per-clip attributions in `LICENSE`.
- `src_file` is the Freesound source clip id: 1524 unique values over 2000 clips. `group_id = esc50_<src_file>` so clips cut from the same source never straddle train and test. The official ESC-50 folds are not used for the primary split.
- Counts for the configured classes (from the CSV): aircraft 80 clips (airplane 40, helicopter 40), ambient 640 (16 categories x 40), hard negatives 240 (6 categories x 40).
- Aircraft exists only in ESC-50, so the held-out-dataset test cannot measure aircraft recall on a second source.

## Al-Emadi drone audio dataset (VERIFIED, github.com/saraalemadi/DroneAudioDataset) -> NOT USED
- README and LICENSE.md state: educational and non-commercial academic research only, and the dataset "may not be used, directly or indirectly, for military, defence, intelligence, security, weapons, surveillance, targeting, or other defence-related applications".
- This project is framed as perimeter surveillance for defence. That conflicts with the licence, so `datasets.al_emadi.enabled` is false and no download code exists.
- Its `unknown` class is cut from ESC-50 (README says so) plus Speech Commands white noise, so it would also leak ESC-50 audio into any ESC-50 split.
- Its drone files (`Binary_Drone_Audio/yes_drone`, 1332 files) are the same file names as `Multiclass_Drone_Audio/bebop_1` + `membo_1` (666 + 666), i.e. duplicated across the two folders.
- DADS (below) lists this dataset as a drone source. See "Open decision" at the bottom.

## DADS, gr1047/drone-audio-detection-samples (card and file list VERIFIED, audio contents NOT YET INSPECTED)
- Card: licence tag `mit`, not gated, last modified 2026-07-14. The card also says "Data from various sources is licensed under Creative Commons Attribution and similar open licenses. Please verify individual licenses". An MIT tag on a compilation does not remove the upstream terms, including the Al-Emadi defence-use ban.
- Size: 41 repo files, 39 parquet shards `data/train-XXXXX-of-00039.parquet`, 6.81 GB download. Over the 1 GB limit, so it needs your explicit confirmation.
- Structure: one `train` split of 180,320 rows. Features are only `audio` (stored inside parquet) and `label` (0 = no drone, 1 = drone). The card lists no file name, source dataset or group column. There are no folders to regex on, so the `audio_glob`/regex settings in the config do not apply and the manifest builder needs a parquet-aware path.
- Card statistics: no-drone 16,729 files, 33.81 h, avg 7.28 s. Drone 163,591 files, 27.10 h, avg 0.60 s. All 16 kHz, 16-bit, mono WAV.
- Drone sources on the card: Al-Emadi Drone Audio Dataset, SPCup19 Egonoise, DREGON, DroneNoise Database, AUDROK, Yi et al. 2023 (Zenodo drone fault classification).
- Non-drone sources on the card: UrbanSound8K, TUT Acoustic Scenes 2017, ESC-50, DNC.
- Consequences to resolve before building the manifest:
  1. ESC-50 is a DADS non-drone source. The same ESC-50 audio can sit in DADS as "no drone" (ambient) and in ESC-50 as aircraft, ambient or hard negative. That is duplicate audio across datasets and possibly conflicting labels (an ESC-50 airplane clip labelled no-drone in DADS).
  2. Drone clips average 0.6 s, shorter than the 1.0 s window. Windows need a rule (concatenate neighbours from one source, pad, or drop short clips).
  3. Without per-row provenance, `group_id` cannot be derived from metadata. Whether `audio.path` inside the parquet carries original file names is unknown until a shard is inspected.
  4. No source-dataset tag means the held-out-dataset test and hard-negative tagging need provenance recovered some other way.
- Next step: `python scripts/peek_dads_shards.py` downloads three small shards (about 275 MB total), reports row order, `audio.path`, durations and sample rates, and fills the generated block below.

## DroneNoise Database, University of Salford (UNVERIFIED, optional)
- figshare was blocked. Not scripted. Send me the article URL or DOI and the licence text if you want it added.

## Own recordings
- `data/own/<place>_<yyyymmdd>_<hhmm>.wav`, label ambient, one group per file, held out from training (`splits.holdout_datasets`).
- Files with other names are skipped with a warning. Put bench-distance drone clips in `data/bench/`, never in `data/own/`.

## Known risks to keep in mind
- Source confound: drone comes from DADS, aircraft only from ESC-50. A classifier can partly learn the recording chain instead of the sound. The held-out-dataset and own-recording tests are there to expose this.
- Aircraft is the smallest class (80 clips, 400 s), so its per-class metrics will be noisy.

## Open decision
Whether to use DADS at all given the Al-Emadi restriction. The DADS card lists Al-Emadi as a source and gives no per-row provenance, so the Al-Emadi part may not be separable. Options: (1) exclude the Al-Emadi-derived part if the shard peek shows it can be identified, (2) ask the course instructor in writing, (3) reframe the project as non-defence. This is your call, not something the code can settle.

<!-- DADS:BEGIN (generated by scripts/inspect_dads.py) -->
Not generated yet. Run `python scripts/inspect_dads.py` after downloading DADS.
<!-- DADS:END -->
