# M5 notes: streaming engine, alert logic, event logging, console monitor

Reproduce the checks: `make stream-check` (about 2 min on 2 cores, 75 s of it building the causal feature cache the first time).
Every number below is in `results/results.json` (`stream_check`) and `results/tables/m5_*.csv`.

**Scope: ESC-50 only, aircraft versus ambient.** There is no drone class. Nothing here is a drone result, and nothing here is an
event-level detection rate; M6 measures that on scenarios.

## What was built
| Piece | File | Notes |
|---|---|---|
| Alert logic | `src/sentinel/alerting.py` | Pure functions and an immutable state, no clock, no audio. Unit tested, including a randomised invariant test. |
| Sources | `src/sentinel/sources.py` | `FileSource` (default, `--realtime` paces it), `SoundDeviceSource`, `TcpSource`. Live sources queue at most `m5.stream.queue_s` seconds and count dropped audio. |
| Windows microphone | `tools/win_mic_stream.py` | Run with Windows Python. Raw int16 mono 16 kHz to `tcp 5005`. `--wav` streams a file instead (stdlib only), for testing the link. |
| Engine | `src/sentinel/engine.py` | Causal band-pass with carried state, 1 s windows every 0.5 s, 146 features, model, silence gate, one tracker per target class, rolling noise floor. |
| Run logs | `src/sentinel/runlog.py` | `runs/<UTC timestamp>/events.jsonl, events.csv, windows.csv, run.json`. Events are flushed to disk when they close. |
| Monitor | `src/sentinel/monitor.py` | `python -m sentinel.monitor --source {mic,tcp,file}`. `rich` live view on a terminal, plain lines otherwise (`--no-ui`). Ctrl+C or SIGTERM closes the open event and writes it. |
| Checks | `scripts/check_streaming.py`, `src/sentinel/streamcheck.py` | `make stream-check`. |

## Alert logic (defaults in `m5.alert`, engineering values, not tuned)
EMA of the window probability (alpha 0.5). ON when the smoothed value is at or above `t_on` in 3 of the last 5 windows; the
event start is back-dated to the first of those windows. OFF after 4 consecutive windows below `t_off`. `t_on` is the model's
saved threshold (0.302) and `t_off = 0.6 * t_on`. Events under 2 s are dropped. Events less than 3 s apart are merged. Consequences
you will see in the logs:
- The alert is raised at the end of the 3rd flagged window, so at least about 2 s after the sound starts, more with the EMA.
- An event is written `merge_gap_s` (3 s) after it ends, because a following event may still merge into it. Ctrl+C or end of
  stream releases it at once.
- The reject option (`m5.alert.reject_below`) changes the displayed window label only, not the events.

## Event record
The fields of the spec, in that order. `t_start` and `t_end` have millisecond resolution (`2026-10-12T10:15:03.500Z`) because the
hop is 0.5 s. `model_version` is the config hash stored in the model file. `snr_est_db` is `null` when no noise floor exists yet
(first second of a stream, or digital silence). Stream time maps to UTC from the moment the engine started (`--t0` overrides).

## SNR estimate
Power subtraction: `10 log10(P/N - 1)` with P the mean in-band power of the event's windows and N the 10th percentile of 20 ms
frame power over the last 60 s of windows that are not flagged and not inside an event, floored at -20 dB. Only windows the
engine does not flag feed the floor, so a sound it misses leaks into the floor and pulls the estimate down.

## Checks (all from `make stream-check`, this sandbox)
**A. Chunk invariance.** Chunks of 0.25 s, 1 s and 17.3 ms give the same probabilities and the same events on 4 real clips
(max |dp| = 0.00e+00 here; the test allows 1e-6).

**B. Causal filter versus the zero-phase filter used for training.** Features of all 8273 non-holdout windows recomputed the live
way, scored out of fold with the M3 folds. Balanced accuracy, pooled out-of-fold, 95% group-bootstrap CI:
| Training features | Test features | Balanced accuracy | Aircraft recall |
|---|---|---|---|
| zero-phase | zero-phase | 0.792 [0.733, 0.854] | 0.751 |
| zero-phase | causal (live) | 0.791 [0.731, 0.852] | 0.744 |
| causal | causal (live) | 0.790 [0.732, 0.851] | 0.747 |

Fold-paired change from zero-phase to causal test features: -0.0014 (fold std 0.0128, worse in 2 of 5 folds). That is inside the
noise, so the live filter costs no measurable accuracy and retraining on causal features gains none. The saved model on the same
windows (trained on all of them, so descriptive only): mean |dp| 0.014, 95th percentile 0.064, largest 0.333; the alert decision
at the saved threshold changes for 216 of 8273 windows (2.6%); the silence gate changes for none.

**C. Speed.** 60 s of audio in 0.80 s (real-time factor 0.013). Per window: median 5.9 ms, 95th percentile 8.6 ms, max 13.7 ms,
against a 500 ms hop. Wall time on this 2-core machine, one process; yours will differ.

**D. SNR estimate on known mixtures** (6 aircraft clips in pink noise, in-band noise power 1e-4, per-window comparison):
| Nominal SNR | Windows | Bias | Mean abs error |
|---|---|---|---|
| 0 dB | 53 | +1.4 dB | 1.9 dB |
| 5 dB | 54 | +1.0 dB | 1.2 dB |
| 10 dB | 54 | +0.8 dB | 0.9 dB |
| 20 dB | 54 | +0.7 dB | 0.7 dB |

The estimate reads high by about 1 dB. The likely cause is that a low percentile of frame power sits below the mean noise power;
this was not isolated. The error is larger at low SNR, where the signal power itself is close to the noise. One noise colour,
6 clips, so treat the table as a sanity check, not a calibration.

## Not done or not verified
- **Microphone paths were not exercised on real hardware.** PortAudio is not installed in the sandbox. `SoundDeviceSource` is tested
  against a fake `sounddevice`. The TCP path is tested over loopback with the real wire format and with `win_mic_stream.py --wav`.
  The Windows microphone itself and WSL2 localhost forwarding are untested here.
- Only the `none` variant runs live; the denoisers work on whole clips.
- `--zmq` is accepted and exits with a message; the publisher is part of M8.
- Dropped audio (consumer slower than real time) is counted and shown, but event times after a gap are late by that much.
- Probabilities are calibrated for clean ESC-50 audio at ESC-50 prevalence. Live false-alarm rates depend on the real background;
  M6 sets the operating point.
