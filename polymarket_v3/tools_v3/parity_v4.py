"""Parity test: the fast cache reader must agree with the authoritative loader.

    python tools_v3/parity_v4.py

This exists because of a real bug. ``slim.SlimMarket`` keeps ``ticks`` in
chronological order (countdowns descending) and a separate ascending mirror of the
countdowns for bisecting. Indexing both lists with the same number is a silent
57-second error that still returns a plausible price, and it survived a full
validation run without anyone noticing.

The lesson is that a "cache" is only safe if it is proven equal to the thing it
caches. This walks the same markets through both readers and compares the fields
every experiment depends on.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import slim  # noqa: E402
from engine_v3 import hf  # noqa: E402

CHECKPOINTS = (300.0, 240.0, 180.0, 120.0, 60.0, 30.0)


def main(argv=None) -> int:
    workspace = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join(workspace, "micro_v3"))
    parser.add_argument("--markets", type=int, default=150)
    args = parser.parse_args(argv)
    root = args.root

    if not slim.has_cache(root):
        print("no slim cache present - nothing to compare")
        return 1

    resolutions = hf.load_resolutions(root)
    failures = 0
    compared = 0

    print("=" * 92)
    print("CACHE PARITY vs THE AUTHORITATIVE LOADER")
    print("=" * 92)
    print(f"  comparing up to {args.markets} markets at "
          f"{len(CHECKPOINTS)} checkpoints each")
    print()

    cached = slim.load_ticks(root, horizons=("5m",))
    raw = hf.iter_markets(root, horizons=["5m"], resolutions=resolutions,
                          limit=args.markets)

    for i, (fast, slow) in enumerate(zip(cached, raw), start=1):
        if i > args.markets:
            break
        if fast.market_id != slow.market_id:
            print(f"  [FAIL] market order diverges at {i}: "
                  f"{fast.market_id} vs {slow.market_id}")
            failures += 1
            continue
        # tick count
        if len(fast.ticks) != len(slow.ticks):
            print(f"  [FAIL] {fast.market_id}: tick count "
                  f"{len(fast.ticks)} vs {len(slow.ticks)}")
            failures += 1
            continue
        # first tick
        if fast.first_tick().to_deadline_ms != slow.first_tick().to_deadline_ms:
            print(f"  [FAIL] {fast.market_id}: first tick deadline differs")
            failures += 1
        # every checkpoint must land on the SAME tick
        for cp in CHECKPOINTS:
            a = fast.tick_at_t_minus(cp, tolerance_s=1.5)
            b = slow.tick_at_t_minus(cp, tolerance_s=1.5)
            compared += 1
            if (a is None) != (b is None):
                print(f"  [FAIL] {fast.market_id} T-{cp:.0f}: one is None "
                      f"({a is None} vs {b is None})")
                failures += 1
                continue
            if a is None:
                continue
            if b.up_mid is None or a.up_mid is None:
                continue
            if abs(a.to_deadline_ms - b.to_deadline_ms) > 0.5:
                print(f"  [FAIL] {fast.market_id} T-{cp:.0f}: deadline "
                      f"{a.to_deadline_ms:,.0f} vs {b.to_deadline_ms:,.0f} "
                      f"(off by {a.to_deadline_ms-b.to_deadline_ms:,.0f} ms)")
                failures += 1
            if abs(a.up_mid - b.up_mid) > 1e-9:
                print(f"  [FAIL] {fast.market_id} T-{cp:.0f}: up_mid "
                      f"{a.up_mid} vs {b.up_mid}")
                failures += 1
        # resolution agreement
        if fast.up_won != slow.up_won:
            print(f"  [FAIL] {fast.market_id}: up_won {fast.up_won} vs {slow.up_won}")
            failures += 1

    print(f"  markets compared : {i if 'i' in dir() else 0}")
    print(f"  checkpoint lookups: {compared}")
    print(f"  failures         : {failures}")
    print()
    if failures:
        print("  The cache does NOT match the authoritative loader. Every result")
        print("  produced through the cache is suspect until this is fixed.")
        return 1
    print("  The cache reproduces the authoritative loader exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
