"""M8: build runs/<timestamp>/report.html (one self-contained file) from a run directory and results/results.json.

    make report                       # newest run in runs/; if there is none, a demo run on a scenario stream is made first
    python scripts/make_report.py --run runs/20261022-101500 [--audio clip.wav]
    python scripts/make_report.py --demo [--audio data/scenarios/fold0_snr+20.wav]

The audio is needed only for the spectrogram: it is taken from --audio, else from the file the run replayed (run.json
source_path), else the spectrogram is left out with a note. The demo run replays a file through the live engine exactly like
`python -m sentinel.monitor --source file`; it never invents data.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

from rich.console import Console

from sentinel import monitor as mon
from sentinel.bundle import load_model
from sentinel.config import load_config, project_path, setup_logging
from sentinel.preprocess import load_audio
from sentinel.report import build_html, latest_run
from sentinel.sources import FileSource


def demo_audio(cfg) -> Path | None:
    d = project_path(cfg, "data_raw").parent / "scenarios"
    for name in ("fold0_snr+20.wav", "fold0_snr+10.wav"):
        if (d / name).exists():
            return d / name
    return next(iter(sorted(d.glob("*.wav"))), None) if d.exists() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="run directory (default: newest in runs/)")
    ap.add_argument("--audio", default=None, help="audio of the run, for the spectrogram (and for --demo)")
    ap.add_argument("--demo", action="store_true", help="make a new run by replaying --audio (default: a scenario stream) first")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    setup_logging(logging.WARNING)
    cfg = load_config(args.config)
    runs_dir = project_path(cfg, "runs")
    run_dir = None if args.demo else (Path(args.run) if args.run else latest_run(runs_dir))
    audio_path = Path(args.audio) if args.audio else None
    if run_dir is None:
        audio_path = audio_path or demo_audio(cfg)
        if audio_path is None:
            print("no run in runs/ and no audio for a demo run. Run `make scenarios` or pass --audio file.wav, "
                  "or record a run with: python -m sentinel.monitor --source file --file clip.wav")
            return 1
        model = load_model(project_path(cfg, "models") / "pipeline.joblib", cfg)
        print(f"no run selected: replaying {audio_path.name} through the live engine for the report ...")
        s = mon.run(FileSource(audio_path, cfg["audio"]["sr"], chunk_s=cfg["m5"]["stream"]["chunk_s"]), model, cfg, runs_dir=runs_dir, ui=False,
                    console=Console(file=io.StringIO()))
        run_dir = Path(s["run_dir"])
    if not (run_dir / "run.json").exists():
        print(f"not a run directory: {run_dir}")
        return 1
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    audio = None
    cand = audio_path or (Path(meta["source_path"]) if meta.get("source_path") else None)
    if cand is not None and cand.exists():
        audio = load_audio(cand, cfg["audio"]["sr"])
    res = json.loads(project_path(cfg, "results_json").read_text(encoding="utf-8"))
    out = run_dir / "report.html"
    out.write_text(build_html(run_dir, cfg, res, audio, cfg["audio"]["sr"]), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    if audio is None:
        print("note: no audio found for this run, so the spectrogram section says so (pass --audio to add it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
