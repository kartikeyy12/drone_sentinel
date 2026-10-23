# SENTINEL-A: acoustic sentinel (PCA, SVD, SVM mini project)

Classifies 1 s audio windows (0.5 s hop, 16 kHz mono) and turns the window probabilities into alerts and events for a person on watch.
Pipeline: band-pass and noise removal, 146 MFCC and spectral features, PCA via SVD (written from scratch), RBF SVM, Platt calibration,
alert logic, event feed, false-alarm and range analysis, HTML report.

**Scope.** The data are ESC-50 only, aircraft versus ambient. There is no drone class (the DADS download failed and the Al-Emadi set forbids
defence use). No number in this project is a drone result. The drone source level in the range model is a config parameter marked TODO_CITE.

## Setup (WSL2 Ubuntu, CPU only)

```bash
cd ~/ml_mini/p1_drone_sentinel
bash scripts/setup_wsl.sh          # apt packages, Python 3.11 venv, pip install -r requirements.txt, pip install -e .
source .venv/bin/activate
make check                         # environment check
```

Keep the project on the Linux filesystem (`~/ml_mini/...`), not under `/mnt/c`. Windows files are at `/mnt/c/Users/<you>/...`.

## Make targets

| target | what it does | output |
| --- | --- | --- |
| `make data` | download ESC-50, build `data/manifest.csv` | `data/raw/`, `data/manifest.csv` |
| `make features` | preprocessing variants, feature cache, before/after spectrograms | `data/features/`, `results/figures/denoise_examples.png` |
| `make m2` | SVD denoising experiment, PCA variance plots | `results/figures/pca_*.png` |
| `make train` | grouped CV of the default pipeline | `results.json` `cv_default` |
| `make ablation` | full ablation grid, nested selection, feature groups | `results/tables/m3_*.csv` |
| `make robustness` | noise robustness sweep | `results/tables/m4_robustness*.csv` |
| `make final` | fit and save the final pipeline | `models/pipeline.joblib`, `pipeline.json` |
| `make stream-check` | streaming checks (chunk invariance, causal filter, speed) | `results.json` `stream_check` |
| `make scenarios` | synthetic test streams with known event times | `data/scenarios/` |
| `make eval-fa` | false alarms per hour versus threshold, operating point | `results.json` `false_alarms` |
| `make range` | detection range model and, if present, the bench test | `results.json` `range`, `bench` |
| `make report` | `report_assets/report_notes.md` and `runs/<timestamp>/report.html` | see below |
| `make test` | whole test suite | |

Every number in a table, figure or report comes from `results/results.json`, written by the scripts through `sentinel.config.update_results`.
All tunables are in `configs/default.yaml`.

## Live monitor

```bash
python -m sentinel.monitor --source file --file clip.wav [--realtime]
python -m sentinel.monitor --source tcp      # microphone of Windows, see below
python -m sentinel.monitor --source mic      # PortAudio input device inside WSL
```

Options: `--t-on 0.56` sets the alert threshold (see the operating point in the report), `--zmq` publishes events, `--no-ui` prints plain lines.
Every closed event is appended at once to `runs/<UTC timestamp>/events.jsonl` and `events.csv`; `windows.csv` and `run.json` are written too.
Ctrl+C closes the open event and writes it.

### Microphone options

1. **File** (default for development): `--source file --file x.wav --realtime` replays a recording at its own speed.
2. **Windows microphone over TCP** (the reliable live option under WSL2). In WSL: `python -m sentinel.monitor --source tcp`. In Windows
   (Windows Python, not WSL): copy the tool over once with `cp tools/win_mic_stream.py /mnt/c/Users/<you>/Downloads/`, then in Windows
   `py -m pip install sounddevice`, `py win_mic_stream.py --list-devices`, `py win_mic_stream.py --device <index>`. `--wav file.wav` streams a 16 kHz mono 16-bit WAV instead, a quick test of the link.
   If localhost does not reach WSL2, run `hostname -I` in WSL and pass it with `--host` (Windows side) and `--tcp-host 0.0.0.0` (monitor).
3. **sounddevice inside WSL** (through WSLg PulseAudio): check `python -c "import sounddevice as sd; print(sd.query_devices())"`.
   If no input device is listed, use option 2 and do not debug it for long.

The two live microphone paths are unit-tested with a fake device and over loopback, but they have not been tried on real hardware by the author
of this code. Test them on your machine before you rely on them.

## ZeroMQ event feed

```bash
python tools/zmq_listen.py                   # terminal 1: prints events (start it first)
python -m sentinel.monitor --source file --file clip.wav --zmq     # terminal 2
```

PUB socket on `tcp://127.0.0.1:5556` (`m5.zmq` in the config, `--zmq-endpoint` overrides). Each closed event is a two-frame message: topic
`acoustic`, then the event as JSON:

```json
{"sensor_id": "acoustic-01", "event_id": 12, "label": "aircraft", "t_start": "2026-10-12T10:15:03.000Z", "t_end": "2026-10-12T10:15:41.000Z",
 "duration_s": 38.0, "peak_prob": 0.97, "mean_prob": 0.91, "top_alt_class": "ambient", "snr_est_db": 9.4, "model_version": "<config hash>"}
```

PUB/SUB drops events for subscribers that are not connected yet. `events.jsonl` always has all of them. A failing publisher never stops the monitor.

## Report

```bash
make report
```

Writes `report_assets/report_notes.md` (results mapped to the seven report sections) and `runs/<timestamp>/report.html` for the newest run
(one self-contained file, open it in a browser). With no run in `runs/` it first replays a scenario stream through the live engine
(`make scenarios` first). To report on a specific run: `python scripts/make_report.py --run runs/<timestamp> --audio clip.wav`
(the audio is only needed for the spectrogram).

## Range model and bench test

`make range` reads the recall-versus-SNR curves from `results.json` and solves `L_src - 20 log10(d/d0) - alpha d - L_ambient >= snr_min`.
`snr_min` is the lowest SNR from which recall stays at or above `m7.recall_target` (0.9). If the curves never get there, the output says
**not reached** and gives only a sensitivity table for hypothetical required SNRs. It is a model estimate, not a measurement.
To get a range number you need a cited source level and a site ambient level in `configs/default.yaml` (`m7`), and a curve that reaches the target.

Bench test: play one clip through a speaker at fixed volume, record with the same microphone at 1, 2, 4 and 8 m, with a few seconds of silence
before the clip starts, and save `data/bench/bench_1m.wav`, `bench_2m.wav`, `bench_4m.wav`, `bench_8m.wav`. Optional `bench_noise.wav`: the room with
the speaker silent, same gain. Then `make range` (or `make bench`) writes `results/tables/m7_bench.csv` and `results/figures/m7_bench.png`.
Indoor reverberation breaks the free-field assumption, so expect the measured SNR to fall more slowly than the prediction.

## False alarms on your own recordings

Put at least one hour of no-target audio (traffic, fans, birds, crowd; any sample rate) in `data/own/` as `<place>_<yyyymmdd>_<hhmm>.wav`, then
`make eval-fa`. The operating point is then chosen on your recordings instead of ESC-50 clips. These files are never used for training.

## Layout

`configs/` settings, `src/sentinel/` package, `scripts/` entry points, `tools/` helpers (Windows microphone, ZeroMQ listener), `tests/`, `docs/`
(notes per milestone), `data/`, `models/`, `results/`, `runs/`, `report_assets/`.
