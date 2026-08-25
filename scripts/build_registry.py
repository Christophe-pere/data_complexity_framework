"""Build the Phase B dataset registry — `datasets/registry.json`.

Loads every spec in qmlc.datasets.all_phase_b_specs(), computes SHA-256, runs
pre-screening, and writes the frozen registry. Class A (13 synthetic) and the
offline tier of Class C (4 sklearn-bundled real datasets) are populated; the
network tier of Class C and Classes B/D return empty spec lists pending
follow-up implementation.

Usage:
    source ../../research-dev/bin/activate
    python -m scripts.build_registry                 # writes datasets/registry.json
    python -m scripts.build_registry --output X      # custom output path
    python -m scripts.build_registry --version V     # set registry version label
    python -m scripts.build_registry --quiet         # suppress per-dataset progress
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable when running as `python scripts/build_registry.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmlc.registry import build_registry, write_registry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Phase B dataset registry")
    parser.add_argument(
        "--output",
        default="datasets/registry.json",
        help="output path (default: datasets/registry.json)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="pre-screening retries before marking a v_i as failed (default: 3)",
    )
    parser.add_argument(
        "--version",
        default="0.1.0-class-a",
        help="registry version label (default: 0.1.0-class-a)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-dataset progress")
    args = parser.parse_args()

    entries = build_registry(max_retries=args.max_retries, verbose=not args.quiet)
    if not entries:
        print("No dataset specs to build (Class A empty?)", file=sys.stderr)
        return 1

    payload = write_registry(entries, args.output, version=args.version)
    print()
    print(f"Wrote {args.output}")
    print(f"  version           : {payload['version']}")
    print(f"  frozen_at         : {payload['frozen_at']}")
    print(f"  total datasets    : {payload['n_datasets']}")
    print(f"  passed screening  : {payload['n_passed']}")
    if payload["n_passed"] < payload["n_datasets"]:
        failed = [d["name"] for d in payload["datasets"] if not d["screening"]["passed"]]
        print(f"  FAILED            : {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
