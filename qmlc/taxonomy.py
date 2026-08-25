"""Data-access-mode taxonomy (framework_v2 §2.5 / §2.6 metadata schema).

Each (D, φ) pair is classified by the data-loading regime required, since this
determines the honest classical comparator for the budget-matched edge and the
H6 dequantization-stratified test:

  - Mode S (sparse / structured): φ loads classical data without assuming coherent
    random-access (QRAM). Encoding cost is paid explicitly (Eq. 2.9).
  - Mode Q (qRAM-dependent): φ *assumes* a bucket-brigade QRAM oracle preparing
    |v⟩ ∝ Σ vᵢ|i⟩ from a classical vector in polylog time. Such pairs MUST be
    compared against a matched quantum-inspired (ℓ²-sampling) classical baseline.
  - Mode N (quantum-native): the input is a physical quantum state; no classical-
    data-loading assumption applies (Class B).

Part II finding (frozen): **no encoding in the suite assumes QRAM.** Angle, ZZ and
re-uploading encodings are structured poly-depth circuits; amplitude encoding is
costed *without* QRAM at O(2ⁿ) state preparation (framework_v2 §2.5: "amplitude
encoding without QRAM, which requires O(2ⁿ) gates"), i.e. it pays the full
non-QRAM cost rather than assuming the polylog oracle. Hence `qram_required = False`
for every φ, every classical-data pair is **Mode S**, and Class B is **Mode N** —
there are **no Mode-Q pairs**. Consequence: H6's (Mode Q ∖ sparse-survivor)
stratum is empty, so H6 is descriptive / underpowered by construction in Part II
(per the N<10-per-stratum clause), and the mode is a per-*dataset* property
(independent of φ in this suite). The classifier still takes the encoding so it
stays correct if a QRAM-dependent encoding is ever added.
"""
from __future__ import annotations

from typing import Optional

# Encodings that ASSUME coherent random-access (QRAM) loading of unstructured
# classical data → Mode Q. Empty for Part II: amplitude is costed without QRAM.
QRAM_ENCODINGS: frozenset[str] = frozenset()

# Encodings whose pipeline contains a sparse-input QLSA/QSVT (BQP-complete) call
# → "sparse-survivor" tag (exempt from the dequantized-baseline rule). None here:
# the Phase B pipeline is variational classification, no QLSS/HHL primitive.
BQP_COMPLETE_ENCODINGS: frozenset[str] = frozenset()


def data_access_mode(klass: str, encoding_id: Optional[str] = None) -> str:
    """Return the data-access mode ∈ {"S", "Q", "N"} for a (dataset, encoding) pair.

    Class B (quantum-native) → N regardless of encoding. Otherwise Q iff the
    encoding assumes QRAM (none do in Part II), else S."""
    if klass == "B":
        return "N"
    if encoding_id is not None and encoding_id in QRAM_ENCODINGS:
        return "Q"
    return "S"


def qram_required(encoding_id: Optional[str]) -> bool:
    return encoding_id in QRAM_ENCODINGS if encoding_id is not None else False


def bqp_complete_subroutine(encoding_id: Optional[str]) -> bool:
    return encoding_id in BQP_COMPLETE_ENCODINGS if encoding_id is not None else False


def classical_baseline_family(klass: str, encoding_id: Optional[str] = None) -> str:
    """Which classical baseline family the matched-budget comparison must use:
    "dequantized" for Mode Q (ℓ²-sampling comparator, mandatory baseline rule),
    "sparse-classical" for sparse-survivor pairs, else "standard"."""
    mode = data_access_mode(klass, encoding_id)
    if mode == "Q" and not bqp_complete_subroutine(encoding_id):
        return "dequantized"
    if bqp_complete_subroutine(encoding_id):
        return "sparse-classical"
    return "standard"


def dataset_taxonomy(klass: str) -> dict:
    """Per-dataset taxonomy block for the registry. In the Part II suite the mode
    is φ-independent (no QRAM encodings), so a per-dataset record is faithful:
    Class B → N, all others → S; all qram/bqp flags False."""
    mode = data_access_mode(klass)
    return {
        "data_access_mode": mode,                 # "N" for Class B, else "S"
        "qram_required": False,                   # no encoding assumes QRAM
        "sparse_block_encoding_admissible": False,  # variational pipeline, no block-encoding
        "bqp_complete_subroutine": False,         # no sparse QLSA/QSVT call
        "classical_baseline_family": "standard",  # no Mode-Q ⇒ no dequantized baseline
        "note": ("φ-independent in Part II: no encoding assumes QRAM, so every "
                 "classical-data pair is Mode S and Class B is Mode N; H6 Mode-Q "
                 "stratum is empty (descriptive/underpowered)."),
    }
