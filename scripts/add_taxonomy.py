"""Inject the data-access-mode taxonomy (framework_v2 §2.6 metadata schema) into
the frozen registry, per dataset. Metadata-only: does NOT recompute hashes or
pre-screening, so it is safe to run while Phase B is in progress (the sweep does
not read registry.json). Idempotent.

    python -m scripts.add_taxonomy            # updates datasets/registry.json in place
    python -m scripts.add_taxonomy --output X
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmlc.taxonomy import dataset_taxonomy  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="datasets/registry.json")
    ap.add_argument("--output", default=None, help="default: in place")
    args = ap.parse_args()

    path = Path(args.registry)
    reg = json.loads(path.read_text())
    from collections import Counter
    modes: Counter = Counter()
    for d in reg["datasets"]:
        tax = dataset_taxonomy(d["klass"])
        d["taxonomy"] = tax
        modes[tax["data_access_mode"]] += 1
    reg["taxonomy_schema"] = (
        "data_access_mode/qram_required/sparse_block_encoding_admissible/"
        "bqp_complete_subroutine/classical_baseline_family per framework_v2 §2.6; "
        "see qmlc/taxonomy.py. Part II: no Mode-Q pairs (no QRAM encoding)."
    )

    out = Path(args.output) if args.output else path
    out.write_text(json.dumps(reg, indent=2))
    print(f"Wrote {out}: tagged {len(reg['datasets'])} datasets — modes {dict(modes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
