#!/usr/bin/env python3
"""Compare tracemalloc snapshots to find memory leaks."""

import pickle
import tracemalloc
from pathlib import Path


def load_snapshot(filepath):
    """Load a tracemalloc snapshot from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def format_bytes(b):
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(b) < 1024:
            return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b:.2f} TB"


def compare_snapshots(snap1, snap2, top_n=30):
    """Compare two snapshots and show growing allocations."""
    # Compare the snapshots
    diff = snap2.compare_to(snap1, 'lineno')

    print(f"\nTop {top_n} growing allocations:")
    print("=" * 100)

    for i, stat in enumerate(diff[:top_n], 1):
        if stat.size_diff <= 0:
            break
        print(f"\n{i}. {stat.traceback}")
        print(f"   Size: {format_bytes(stat.size)} ({format_bytes(stat.size_diff):+})")
        print(f"   Count: {stat.count} ({stat.count_diff:+})")


def main():
    import sys

    snapshots_dir = Path("/workspace/rlee-tools/kv-bug/snapshots")
    files = sorted(snapshots_dir.glob("*.pickle"))

    print("Available snapshots:")
    for f in files:
        print(f"  {f.name}")

    if len(sys.argv) >= 3:
        snap1_path = sys.argv[1]
        snap2_path = sys.argv[2]
    else:
        # Default: compare first and last benchmark snapshots
        snap1_path = str(snapshots_dir / "002_after_bench_1.pickle")
        snap2_path = str(snapshots_dir / "012_after_bench_6.pickle")

    print(f"\nComparing: {snap1_path}")
    print(f"      vs: {snap2_path}")

    snap1 = load_snapshot(snap1_path)
    snap2 = load_snapshot(snap2_path)

    # Get traced memory stats
    stats1 = snap1.statistics('lineno')
    stats2 = snap2.statistics('lineno')

    total1 = sum(s.size for s in stats1)
    total2 = sum(s.size for s in stats2)

    print(f"\nSnapshot 1 total: {format_bytes(total1)}")
    print(f"Snapshot 2 total: {format_bytes(total2)}")
    print(f"Difference: {format_bytes(total2 - total1)}")

    compare_snapshots(snap1, snap2)


if __name__ == "__main__":
    main()
