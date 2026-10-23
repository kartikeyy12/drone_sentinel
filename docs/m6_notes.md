# M6 notes: scenarios, false alarms per hour, operating point

Reproduce: `make eval-fa` (runs `make scenarios` first). About 20 s for the scenarios and 5 to 7 minutes for the evaluation on 2 cores.
Every number below is in `results/results.json` (`false_alarms`) and `results/tables/m6_*.csv`; figures `results/figures/m6_*.png`.

**Scope: ESC-50 only, aircraft versus ambient.** There is no drone class, so the "targets" are ESC-50 aircraft clips. Nothing here is a
drone result. **`data/own/` was empty, so no false-alarm rate here comes from real long recordings.** The rate below is from joined
ESC-50 ambient clips and cannot validate the 1-per-hour target on real audio. Put your own recordings in `data/own/` and rerun
`make eval-fa`: if they total at least 1 hour (`m6.false_alarms.min_own_hours`) the operating point is chosen on them instead.

## Honest protocol (no leakage)
The saved model was fitted on all ESC-50 windows, so scoring ESC-50 audio with it would be in-sample. Everything is scored by a
cross-fitted model: for each of the 5 M3 folds, a pipeline fitted on the other folds scores only audio made from that fold's
held-out recordings. The calibrator and silence gate are the saved ones (two fitted parameters from out-of-fold scores).
- **False-alarm streams** (`esc50_crossfit`): per fold, all held-out ESC-50 ambient clips (hard negatives such as engine, train,
  vacuum cleaner included) joined into one stream. 1.22 hours in total (0.89 h other ambient, 0.33 h hard negatives). These are 5 s clips
  joined with abrupt cuts, a stress test with many scene changes, not a continuous recording.
- **Scenario streams** (`make scenarios`, `data/scenarios/*.wav`, ground truth in `scenarios.json`): per fold, a 240 s background cut
  from that fold's held-out ambient clips, scaled to a fixed in-band rms of 0.02 so its level is the same at every SNR, with 8 held-out
  aircraft clips embedded at random times (at least 12 s apart). SNR = in-band power of the clip over the in-band power of the whole
  background stream, at 20, 10, 5, 0 and -5 dB. Times, clips and background are the same at every SNR (paired). 5 folds x 8 clips = 40
  clips per SNR. Only the +20 dB streams (and one +10 dB stream) needed a global gain below 1 to avoid 16 bit clipping (recorded in the json).
- The engine runs on the real audio once; the window probabilities are stored and the alert logic (`alerting.detect_events`) is swept over
  them, so the sweep needs no model re-runs. The alert logic is the M5 default except `t_on`; `t_off = 0.6 * t_on`.
- Event matching: a clip counts as detected if an alert event overlaps the clip interval widened by 1 s. Latency = alert time minus the
  clip start. Alert events matching no clip are false alarms.

## Results (cross-fitted, ESC-50)
Operating point rule: at most 1 false alarm per hour, then the best mean event recall over the 5 scenario SNRs.
| Alert threshold `t_on` | False alarms (esc50_crossfit, 1.22 h) | Mean event recall |
|---|---|---|
| 0.302 (saved model threshold) | 37 = 30.3 per hour (95% CI 21.3 to 41.7) | 0.120 |
| **0.56 (operating point)** | **0 = 0.0 per hour (95% CI 0.0 to 3.0)** | **0.015** |

Event recall by scenario SNR (detected / 40 clips, median latency):
| SNR | t_on 0.302 | t_on 0.56 |
|---|---|---|
| +20 dB | 0.28 (11/40), 2.9 s | 0.07 (3/40), 2.8 s |
| +10 dB | 0.17 (7/40), 2.9 s | 0.00 (0/40) |
| +5 dB | 0.07 (3/40), 2.8 s | 0.00 (0/40) |
| 0 dB | 0.05 (2/40), 0.3 s | 0.00 (0/40) |
| -5 dB | 0.03 (1/40), -2.3 s | 0.00 (0/40) |

Mean event recall for other false-alarm budgets (`recall_by_false_alarm_budget`): 1/h: 0.015 (t_on 0.56); 5/h: 0.025 (0.46); 10/h: 0.045 (0.42);
30/h: 0.100 (0.32); 100/h: 0.555 (0.14).

## What this says
- The detector as trained is weak at event level. ESC-50 aircraft clips are short, often faint, and the window classifier has balanced
  accuracy 0.79 (M3). To keep false alarms near 1 per hour the threshold has to be so high that almost no aircraft clip is alerted.
  This is a property of the model and data, not a bug in the alert logic (which is unit tested); it is the honest operating-point result.
- The 1-per-hour point is only weakly determined: 0 alarms in 1.22 h has a 95% upper bound of 3.0 per hour, and the point estimate
  jumps from 1.6 per hour at t_on 0.54 to 0 at 0.56 (counts of 2 and 0).
- Part of the low-SNR recall at low thresholds is chance. The background itself raises false alarms (30 per hour at 0.302), and an
  alarm within 1 s of a clip counts for it. That is also why the -5 dB latency at 0.302 is negative (an alarm that started just before the clip).
- Alert logic parameters (EMA, N of M, K, merge gap) were not tuned here; a sweep over them could move the trade-off curve. That
  is not done.
- Hard negatives: 0 false alarms at the operating point in both groups (hard negatives 0 in 0.33 h, other ambient 0 in 0.89 h), so no
  breakdown is informative at that point. `results/tables/m6_false_alarm_events.csv` is written with the events at the operating point.

## Using the operating point
`t_on` acts on the smoothed probability inside the alert logic, not on the window threshold of M4. To use 0.56, set
`m5.alert.t_on: 0.56` in `configs/default.yaml` (null means the model's saved 0.302). The config is not changed by this script.
The monitor also takes `--t-on 0.56`.
