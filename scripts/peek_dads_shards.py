#!/usr/bin/env python
"""Download a few small DADS shards (each under ~130 MB) and report what is inside.

Run this BEFORE committing to the full 6.8 GB download. Writes docs/data_notes.md (generated
block) and results/results.json (key `dads_peek`).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentinel.config import PROJECT_ROOT, load_config, project_path, setup_logging, update_results  # noqa: E402
from sentinel.dads_peek import render_markdown, replace_block, summarize_shard  # noqa: E402

N_SHARDS = 39  # from the dataset card file list: data/train-XXXXX-of-00039.parquet


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=int, nargs="+", default=[4, 12, 30],
                    help="shard indices (4, 12, 30 are 75-125 MB each; 0-3 are 0.5-1.3 GB, avoid)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    from huggingface_hub import hf_hub_download

    ds = cfg["datasets"]["dads"]
    dest = project_path(cfg, "data_raw") / ds["dir"]
    summaries = []
    for i in args.shards:
        name = f"data/train-{i:05d}-of-{N_SHARDS:05d}.parquet"
        try:
            local = hf_hub_download(ds["repo"], name, repo_type="dataset", local_dir=str(dest))
        except Exception as e:  # noqa: BLE001
            logging.error("download of %s failed: %s", name, str(e).splitlines()[0])
            logging.error("If this is a certificate error, something on your network is intercepting TLS. "
                          "Run the openssl diagnostic from the M0 notes. Do not disable verification.")
            return 1
        s = summarize_shard(Path(local))
        summaries.append(s)
        print(f"\n== {s['file']}: {s['rows']} rows, labels {s['label_counts']}, label runs {s['label_runs_in_row_order']}")
        print(f"   audio.path non-empty {s['audio_path_non_empty']}, unique {s['audio_path_unique']}")
        print(f"   examples: {s['audio_path_examples']}")
        print(f"   sample rates {s['sample_rates']}, decode errors {s['decode_errors']}")
        print(f"   duration by label: {s['duration_s_by_label']}")
    update_results(cfg, "dads_peek", {"shards": summaries})
    notes = PROJECT_ROOT / "docs" / "data_notes.md"
    notes.write_text(replace_block(notes.read_text(encoding="utf-8"), render_markdown(summaries)), encoding="utf-8")
    print(f"\nupdated {notes} and results/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
