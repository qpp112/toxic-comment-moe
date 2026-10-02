#!/usr/bin/env python
"""Run every configuration in an ablation file, skipping runs that already finished.

Examples
--------
    # full study: 8 settings x 3 seeds = 24 runs
    python scripts/run_ablation.py

    # stronger-encoder study (DeBERTa-v3, 2 settings x 3 seeds)
    python scripts/run_ablation.py --config configs/encoders.yaml

    # only the headline comparison, one seed, results on Google Drive
    python scripts/run_ablation.py --only bert_bce moe_cbfocal --seeds 42 \\
        --set output_dir=/content/drive/MyDrive/toxic-moe/results

    # 2-minute pipeline check on 1% of the data
    python scripts/run_ablation.py --only moe_cbfocal --seeds 42 \\
        --set train_fraction=0.01 epochs=1 output_dir=results_smoke

``--only`` accepts setting names (``bert_bce``: every seed) or full run names
(``bert_bce_s43``: one run).
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toxic_moe.config import is_complete, load_ablation, parse_overrides  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/ablation.yaml")
    parser.add_argument("--only", nargs="*", help="setting or run names to execute (default: all)")
    parser.add_argument("--seeds", nargs="*", type=int, help="override the seed list in the config file")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config keys for every run")
    parser.add_argument("--force", action="store_true", help="re-run even if metrics.json exists")
    parser.add_argument("--list", action="store_true", help="print the runs and exit")
    args = parser.parse_args()

    configs = load_ablation(args.config, parse_overrides(args.set), seeds=args.seeds)
    if args.only:
        known = {c.name for c in configs} | {c.group_name for c in configs}
        unknown = set(args.only) - known
        if unknown:
            parser.error(f"unknown names: {sorted(unknown)}. Known settings: {sorted({c.group_name for c in configs})}")
        configs = [c for c in configs if c.name in args.only or c.group_name in args.only]

    todo = [c for c in configs if args.force or not is_complete(c)]
    for cfg in configs:
        status = "todo" if cfg in todo else "done"
        print(f"  [{status}] {cfg.name:<28} {cfg.encoder:<26} head={cfg.arch:<4} loss={cfg.loss:<12} -> {cfg.run_dir}")
    print(f"{len(todo)} of {len(configs)} runs to do")
    if args.list:
        return 0

    from toxic_moe.experiment import run  # imported here so --list works without torch

    failures, durations = [], []
    for i, cfg in enumerate(todo, 1):
        t0 = time.time()
        try:
            run(cfg)
            durations.append((time.time() - t0) / 60)
            remaining = len(todo) - i
            eta = sum(durations) / len(durations) * remaining
            print(
                f"finished {cfg.name} in {durations[-1]:.1f} min "
                f"({i}/{len(todo)} done; ~{eta:.0f} min left at this pace)\n",
                flush=True,
            )
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
