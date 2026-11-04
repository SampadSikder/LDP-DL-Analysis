#!/usr/bin/env python
"""Server side of Codex mode: pack every Stage-1 unit that has a WSP run (mode 'wsp') with exactly the probe and
validation data the harness uses, so Codex can edit the weights on another machine (no API key needed).

  python scripts/wsp/export_codex.py                    # tier A (+ budget curve): results/wsp_codex/bundle/
  python scripts/wsp/export_codex.py --set tier_b
  python scripts/wsp/export_codex.py --pilot            # only the 2 pilot runs (small, quick to copy)
Then: tar czf results/wsp_codex_bundle.tar.gz results/wsp_codex/bundle   (run_wsp_server.sh codex_export does both)
"""
import argparse
import sys

import _common  # noqa: F401  (sets sys.path / cwd)

from attacker_detector.wsp.codex import bundle_dir, export_bundle
from attacker_detector.wsp.jobs import STAGE2, SWEEP, all_jobs, load_yaml, select
from attacker_detector.wsp.runner import device_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', nargs='+', default=['tier_a'])
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--kind', nargs='+', choices=['mv', 'freq'])
    ap.add_argument('--device', default='auto')
    ap.add_argument('--force', action='store_true', help='rewrite units already in the bundle')
    args = ap.parse_args()
    jobs = select(all_jobs(), args.set, ['wsp'], args.pilot, args.kind)
    print(f'{len(jobs)} Codex runs -> {bundle_dir()}', flush=True)
    man = export_bundle(jobs, load_yaml(SWEEP), load_yaml(STAGE2), device_for(args.device),
                        log=lambda s: print(s, flush=True), force=args.force)
    print(f"bundle: {len(man['units'])} units, {len(man['cells'])} mean/variance cells, {len(man['jobs'])} runs")
    return 0


if __name__ == '__main__':
    sys.exit(main())
