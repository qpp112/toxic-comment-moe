#!/usr/bin/env python
"""Run every configuration in an ablation file, skipping runs that already finished.

Examples
--------
    # full study (8 runs)
    python scripts/run_ablation.py

    # only the headline runs, written to Google Drive so a Colab disconnect loses nothing
    python scripts/run_ablation.py --only bert_bce moe_cbfocal --set output_dir=/content/drive/MyDrive/toxic-moe/results

    # 2-minute pipeline check on 1% of the data
    python scripts/run_ablation.py --set train_fraction=0.01 epochs=1 output_dir=results_smoke
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toxic_moe.config import load_ablation, parse_overrides  # noqa: E402
from toxic_moe.experiment import is_complete, run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/ablation.yaml")
    parser.add_argument("--only", nargs="*", help="run names to execute (default: all)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config keys for every run")
    parser.add_argument("--force", action="store_true", help="re-run even if metrics.json exists")
    parser.add_argument("--list", action="store_true", help="print the runs and exit")
    args = parser.parse_args()

    configs = load_ablation(args.config, parse_overrides(args.set))
    if args.only:
        unknown = set(args.only) - {c.name for c in configs}
        if unknown:
            parser.error(f"unknown run names: {sorted(unknown)}")
        configs = [c for c in configs if c.name in args.only]

    for cfg in configs:
        status = "done" if is_complete(cfg) else "todo"
        print(f"  [{status}] {cfg.name:<24} arch={cfg.arch:<4} loss={cfg.loss:<12} -> {cfg.run_dir}")
    if args.list:
        return 0

    failures = []
    for cfg in configs:
        if is_complete(cfg) and not args.force:
            print(f"skip {cfg.name} (already complete)")
            continue
        t0 = time.time()
        try:
            run(cfg)
            print(f"finished {cfg.name} in {(time.time() - t0) / 60:.1f} min\n", flush=True)
        except KeyboardInterrupt:
            raise
        except Exception:  # keep going so one bad run doesn't waste a GPU session
            traceback.print_exc()
            failures.append(cfg.name)
    if failures:
        print(f"FAILED runs: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
