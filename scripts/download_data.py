#!/usr/bin/env python
"""Download public datasets into data/raw. Sizes are printed first; >1 GB needs confirmation.

ESC-50: per-file download from raw.githubusercontent.com with retries; re-run resumes.
DADS:   huggingface_hub.snapshot_download (no `datasets` library, no torch).
Al-Emadi is intentionally NOT downloadable here: its licence forbids defence/security use.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentinel.config import load_config, project_path, setup_logging  # noqa: E402
from sentinel.fetch import count_audio, fetch_esc50  # noqa: E402

log = logging.getLogger("download_data")


def confirm_size(name: str, size_gb: float | None, limit_gb: float, yes: bool, dry_run: bool = False) -> bool:
    """Print the size and decide whether to proceed."""
    print(f"[{name}] size: {'unknown' if size_gb is None else f'{size_gb:.2f} GB'} (limit without confirmation: {limit_gb} GB)")
    if dry_run:
        return False
    if size_gb is not None and size_gb <= limit_gb:
        return True
    if yes:
        return True
    if sys.stdin.isatty():
        return input(f"Download {name}? [y/N] ").strip().lower() == "y"
    print(f"[{name}] not downloading: size is unknown or over the limit. Re-run with --yes to confirm.")
    return False


def esc50(cfg: dict, args: argparse.Namespace) -> None:
    """Download ESC-50 into data/raw/esc50."""
    ds = cfg["datasets"]["esc50"]
    dest = project_path(cfg, "data_raw") / ds["dir"]
    if count_audio(dest) >= 2000 and (dest / "meta" / "esc50.csv").exists():
        print(f"[esc50] already present at {dest} (2000 wav files)")
        return
    size_gb = None
    try:
        with urllib.request.urlopen(f"https://api.github.com/repos/{ds['repo']}", timeout=20) as r:
            size_gb = json.load(r)["size"] / 1024 / 1024  # GitHub reports KB (whole repo incl. history)
    except Exception as e:  # noqa: BLE001
        log.warning("could not read repo size from GitHub API: %s", e)
    if not confirm_size("esc50", size_gb, cfg["download"]["max_gb_without_confirmation"], args.yes, args.dry_run):
        return
    n = fetch_esc50(ds["repo"], dest, tuple(args.esc50_folds))
    print(f"[esc50] {n} wav files at {dest}")
    if n < 2000 and len(args.esc50_folds) == 5:
        raise RuntimeError(f"expected 2000 wav files, found {n}")


def dads(cfg: dict, args: argparse.Namespace) -> None:
    """Download the DADS snapshot into data/raw/dads."""
    from huggingface_hub import HfApi, snapshot_download

    ds = cfg["datasets"]["dads"]
    dest = project_path(cfg, "data_raw") / ds["dir"]
    info = HfApi().dataset_info(ds["repo"], files_metadata=True)
    size = sum((s.size or 0) for s in info.siblings)
    print(f"[dads] files: {len(info.siblings)}; licence on card: {getattr(info.card_data, 'license', None)}; "
          f"gated: {info.gated}; last modified: {info.last_modified}")
    if not confirm_size("dads", size / 1e9, cfg["download"]["max_gb_without_confirmation"], args.yes, args.dry_run):
        return
    snapshot_download(repo_id=ds["repo"], repo_type="dataset", local_dir=str(dest))
    print(f"[dads] downloaded to {dest}. Next: python scripts/inspect_dads.py")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["esc50", "dads", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true", help="print sizes only")
    ap.add_argument("--yes", action="store_true", help="confirm downloads over the size limit")
    ap.add_argument("--esc50-folds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                    help="ESC-50 folds to fetch (each is about 176 MB); re-run to resume")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    rc = 0
    for name, fn in (("esc50", esc50), ("dads", dads)):
        if args.only in (name, "all") and cfg["datasets"][name]["enabled"]:
            try:
                fn(cfg, args)
            except Exception as e:  # noqa: BLE001
                log.error("%s failed: %s", name, e)
                rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
