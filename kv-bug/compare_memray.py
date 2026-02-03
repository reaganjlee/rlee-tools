#!/usr/bin/env python3
"""Compare two memray HTML table files to find growing allocations."""

import re
import json
from pathlib import Path
from collections import defaultdict


def parse_memray_html(filepath):
    """Extract allocation data from memray table HTML."""
    content = Path(filepath).read_text()

    # Find packed_data array
    match = re.search(r'const packed_data = (\[.*?\]);', content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return data
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            pass

    return None


def format_bytes(b):
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(b) < 1024:
            return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b:.2f} TB"


def main():
    import sys

    if len(sys.argv) < 3:
        print("Usage: compare_memray.py <bench1.html> <bench2.html>")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]

    print(f"Parsing {file1}...")
    data1 = parse_memray_html(file1)

    print(f"Parsing {file2}...")
    data2 = parse_memray_html(file2)

    if data1 is None or data2 is None:
        print("Failed to parse HTML files. Checking structure...")
        # Print first few KB to understand structure
        content = Path(file1).read_text()[:5000]
        print("First 5KB of file:")
        print(content)
        return

    print(f"Bench1: {len(data1)} allocation sites")
    print(f"Bench2: {len(data2)} allocation sites")

    # Build lookup by stack_trace - aggregate by trace
    def aggregate(data):
        agg = defaultdict(lambda: {"size": 0, "n_allocations": 0})
        for item in data:
            trace = item.get('stack_trace', '')
            agg[trace]["size"] += item.get('size', 0)
            agg[trace]["n_allocations"] += item.get('n_allocations', 0)
        return agg

    alloc1 = aggregate(data1)
    alloc2 = aggregate(data2)

    # Find growing allocations
    growth = []
    for trace, a2 in alloc2.items():
        a1 = alloc1.get(trace, {"size": 0, "n_allocations": 0})
        size1 = a1["size"]
        size2 = a2["size"]
        delta = size2 - size1
        if delta > 0:
            growth.append((trace, size1, size2, delta, a2["n_allocations"] - a1["n_allocations"]))

    # Sort by growth
    growth.sort(key=lambda x: -x[3])

    print("\n" + "="*80)
    print("TOP 30 GROWING ALLOCATIONS (by memory increase)")
    print("="*80)

    for i, (trace, s1, s2, delta, n_delta) in enumerate(growth[:30], 1):
        print(f"\n{i}. {trace[:100]}")
        print(f"   Bench1: {format_bytes(s1)}")
        print(f"   Bench2: {format_bytes(s2)}")
        print(f"   Growth: +{format_bytes(delta)} (+{n_delta} allocations)")


if __name__ == "__main__":
    main()
