"""Write report_assets/report_notes.md from results/results.json (numbers only from the file)."""
from __future__ import annotations

import json
import sys

from sentinel.config import PROJECT_ROOT, load_config, project_path
from sentinel.notes import build_notes


def main() -> int:
    cfg = load_config()
    res = json.loads(project_path(cfg, "results_json").read_text(encoding="utf-8"))
    out = PROJECT_ROOT / "report_assets" / "report_notes.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_notes(res, cfg), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
