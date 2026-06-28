# Data Leakage Audit Run Log

This file records the public command sequence used to generate the released
leakage-audit artifacts. The commands assume that the MP-grounded train/test
JSONL files and symmetrized CIF files are available locally. Paths should be
adapted to the user's local checkout.

## Inputs

- MP-grounded training split JSONL.
- Initial MP-grounded held-out test split JSONL with 2,000 rows.
- Symmetrized CIF files indexed by MP-ID.

## Commands

Run composition overlap audit:

```bash
python data/data-audit/scripts/audit_composition_leakage.py
```

Run AFLOW-style protostructure-label overlap audit:

```bash
python data/data-audit/scripts/audit_prototype_leakage.py --workers 64
```

Run structure-similarity audit with global fingerprint retrieval followed by
StructureMatcher confirmation:

```bash
python data/data-audit/scripts/audit_structure_similarity_leakage.py \
  --workers 64 \
  --top-k 200 \
  --fingerprint-distance-threshold 0.9
```

Regenerate the final row-level flags, summary tables, and clean held-out splits:

```bash
python data/data-audit/scripts/summarize_and_filter_leakage.py
```

## Released Outputs

- `leakage_row_flags.jsonl`: 2,000 rows.
- `leakage_metric_summary_table.csv`: reported leakage metrics.
- `clean-tests/mp_cif_design_property_test_no_identity_structure_full_protostructure_overlap.jsonl`: 1,662 rows.
- `clean-tests/mp_cif_design_property_test_no_any_audit_overlap.jsonl`: 1,308 rows.

The audit scripts expose path arguments or constants for local source locations;
users should point them to their local train/test JSONL files and CIF directory
before rerunning the full audit.
