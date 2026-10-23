#!/usr/bin/env python
"""M6 `make scenarios`: synthetic test streams with ground truth.

For each M3 fold a 240 s background is cut from that fold's held-out ESC-50 ambient clips (scaled to a fixed in-band rms, m6.scenarios.bg_inband_rms), and 8 aircraft clips of the same
fold are embedded at known times at each scenario SNR (in-band clip power over the in-band power of the background stream).
Times, clips and background are the same for every SNR, so the SNRs are paired. Streams go to data/scenarios/*.wav (16 bit)
with data/scenarios/scenarios.json holding the ground truth. Replay one with
    python -m sentinel.monitor --source file --file data/scenarios/fold0_snr+10.wav --realtime
The targets are ESC-50 aircraft clips; there is no drone class. The streams are concatenations of 5 s clips with abrupt joins.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentinel import scenarios as sc  # noqa: E402
from sentinel.config import PROJECT_ROOT, load_config, project_path, set_seeds, setup_logging  # noqa: E402
from sentinel.cv import make_folds  # noqa: E402
from sentinel.dataset import build_dataset  # noqa: E402
from sentinel.noise import inband_power  # noqa: E402
from sentinel.preprocess import load_audio  # noqa: E402

log = logging.getLogger("make_scenarios")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    set_seeds(cfg["seed"])
    sr, sp = cfg["audio"]["sr"], cfg["m6"]["scenarios"]
    out_dir = PROJECT_ROOT / sp["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = build_dataset(cfg, "none")
    folds = make_folds(ds.y, ds.groups, cfg["splits"]["cv_folds"], cfg["seed"])
    man = pd.read_csv(project_path(cfg, "manifest"))
    man["fold"] = sc.manifest_folds(man, ds, folds)
    man = man[~man["dataset"].isin(cfg["splits"]["holdout_datasets"])]
    index = []
    for f in range(len(folds)):
        rng = np.random.default_rng(cfg["seed"] + 1000 + f)
        amb = man[(man["label"] == "ambient") & (man["fold"] == f)].sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
        air = man[(man["label"] == "aircraft") & (man["fold"] == f)]
        clips = [(load_audio(PROJECT_ROOT / r.path, sr), r.category, bool(r.hard_negative)) for r in amb.itertuples()]
        bg, _ = sc.concat_clips(clips, sr, max_s=sp["duration_s"])
        if len(bg) < sp["duration_s"] * sr:
            log.error("fold %d has only %.0f s of ambient audio", f, len(bg) / sr)
            return 1
        n = min(int(sp["n_targets"]), len(air))
        pick = air.iloc[rng.choice(len(air), n, replace=False)]
        targets = [load_audio(PROJECT_ROOT / p, sr) for p in pick["path"]]
        starts = sc.place_times(rng, n, sp["duration_s"], max(len(t) for t in targets) / sr, sp["min_gap_s"], sp["margin_s"])
        bg_scale = float(sp["bg_inband_rms"]) / float(np.sqrt(inband_power(bg, sr, cfg)))
        bg = (bg * bg_scale).astype(np.float32)          # fixed absolute background level at every SNR
        p_bg = inband_power(bg, sr, cfg)
        for snr in sp["snr_db"]:
            mix = sc.embed(bg, targets, starts, float(snr), sr, cfg, p_bg)
            peak = float(np.max(np.abs(mix)))
            gain = 0.99 / peak if peak > 0.99 else 1.0           # keeps 16 bit from clipping; scales clip and background alike
            name = f"fold{f}_snr{snr:+g}.wav"
            sf.write(out_dir / name, mix * gain, sr, subtype="PCM_16")
            truth = [{"t_start": float(t), "t_end": float(t + len(c) / sr), "clip": str(p), "category": str(cat)}
                     for t, c, p, cat in zip(starts, targets, pick["path"], pick["category"])]
            index.append({"file": name, "fold": f, "snr_db": float(snr), "duration_s": len(mix) / sr, "gain": gain,
                          "background_inband_power": p_bg, "background_scale": bg_scale, "truth": truth})
            print(f"fold {f}  snr {snr:+g} dB  {name}  {len(truth)} clips  gain {gain:.3f}", flush=True)
    (out_dir / "scenarios.json").write_text(json.dumps({"sr": sr, "targets": "ESC-50 aircraft clips (no drone class)", "streams": index},
                                                       indent=1), encoding="utf-8")
    print(f"\nwrote {len(index)} streams and scenarios.json to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
