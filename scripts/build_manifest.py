#!/usr/bin/env python
"""Build data/manifest.csv, validate it, write counts to results.json and plot class balance."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentinel.config import PROJECT_ROOT, load_config, project_path, set_seeds, setup_logging, update_results  # noqa: E402
from sentinel.manifest import COLUMNS, dads_rows, esc50_rows, own_rows, summarize, validate_manifest  # noqa: E402

log = logging.getLogger("build_manifest")


def plot_balance(summary: dict, out: Path) -> None:
    """Two panels: files per class and minutes per class, split by dataset/bucket."""
    items = summary["by_dataset_bucket"]
    classes = ["drone", "aircraft", "ambient", "ambient_hard_negative"]
    datasets = sorted({k.split("/")[0] for k in items})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, key, title in zip(axes, ["files", "minutes", "windows"],
                              ["Files", "Minutes of audio", "1 s windows (0.5 s hop)"]):
        bottom = [0.0] * len(classes)
        for ds in datasets:
            vals = [items.get(f"{ds}/{c}", {}).get(key, 0) for c in classes]
            ax.bar(classes, vals, bottom=bottom, label=ds)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
    axes[0].legend(title="dataset")
    fig.suptitle("Class balance (hard negatives shown separately)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-missing", action="store_true",
                    help="warn instead of failing when ESC-50 audio files listed in meta are absent")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    set_seeds(cfg["seed"])
    raw, ds = project_path(cfg, "data_raw"), cfg["datasets"]
    rows: list[dict] = []
    try:
        if ds["esc50"]["enabled"]:
            rows += esc50_rows(raw / ds["esc50"]["dir"], ds["esc50"], args.allow_missing)
        if ds["dads"]["enabled"]:
            if ds["dads"].get("audio_glob"):
                rows += dads_rows(raw / ds["dads"]["dir"], ds["dads"])
            else:
                log.warning("DADS skipped: layout not configured (see docs/data_notes.md). "
                            "The manifest has NO drone class until this is done.")
        if ds["own"]["enabled"]:
            rows += own_rows(PROJECT_ROOT / ds["own"]["dir"])
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 1
    df = pd.DataFrame(rows, columns=COLUMNS)
    problems = validate_manifest(df)
    if problems:
        for p in problems:
            log.error("manifest problem: %s", p)
        return 1
    out = project_path(cfg, "manifest")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary = summarize(df, cfg["audio"]["window_s"], cfg["audio"]["hop_s"])
    update_results(cfg, "data", summary)
    plot_balance(summary, project_path(cfg, "figures") / "class_balance.png")
    print(f"wrote {out} ({len(df)} files, {df['group_id'].nunique()} groups)")
    for k, v in summary["by_class"].items():
        print(f"  {k:9s} files={v['files']:5d} groups={v['groups']:5d} minutes={v['minutes']:8.2f} windows={v['windows']}")
    if "drone" not in summary["by_class"]:
        print("WARNING: no drone rows. Configure DADS before M1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
