"""causalarc — Phase 0 harness for "Causally Useful Compression on ARC".

This package is the trustworthy instrument that every later gate depends on.
It is intentionally dependency-light (stdlib + numpy) and CPU-only: Phase 0 is
specification & audit, not training. See COMPRESSION_INTELLIGENCE_RESEARCH_PLAN.

Public surface:
    Role, Variable, CausalLedger, Family      (ledger.py)
    FAMILIES                                   (families.py)
    do, effect_of_intervention, recover_causal_mask   (interventions.py)
    make_splits, Split                         (splits.py)
    lz78_bits, mdl_bits_of_grid                (mdl.py)
    RunRecord, write_record                    (records.py)
"""
from __future__ import annotations

__version__ = "0.0.1"
