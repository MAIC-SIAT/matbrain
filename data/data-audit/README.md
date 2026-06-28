# Data Leakage Audit

This directory contains the public artifacts for the training-test leakage audit
used in the MatBrain manuscript revision.

## Contents

| Path | Description |
| --- | --- |
| `scripts/` | Audit scripts for MP-ID/exact-CIF identity checks, raw-formula overlap, AFLOW-style protostructure-label overlap, StructureMatcher-confirmed structure-similarity overlap, and final aggregation. |
| `leakage_row_flags.jsonl` | Sample-level audit flags for the initial 2,000 held-out test records. |
| `leakage_metric_summary_table.csv` | Summary counts and fractions for the reported leakage metrics. |
| `clean-tests/mp_cif_design_property_test_no_identity_structure_full_protostructure_overlap.jsonl` | Primary leakage-controlled held-out set after removing identity, exact-CIF, full-protostructure-label, and StructureMatcher-confirmed structural-similarity overlaps. This file contains 1,662 examples. |
| `clean-tests/mp_cif_design_property_test_no_any_audit_overlap.jsonl` | More conservative composition-inclusive clean set after also excluding raw-formula overlaps. This file contains 1,308 examples. |
| `train_mp_ids.json`, `val_mp_ids.json`, `test_mp_ids.json` | Timestamp-based MP-ID split lists used to construct the MP-grounded training, validation, and initial held-out test partitions. |
| `audit_run_log.md` | Reproducibility notes and command sequence for the public audit artifacts. |

## Reported Audit Metrics

| Audit level | Criterion | Overlapping test samples | Fraction |
| --- | --- | ---: | ---: |
| Material identity | MP-ID overlap | 0 / 2000 | 0.00% |
| Exact structure identity | Exact CIF-hash overlap | 0 / 2000 | 0.00% |
| Prototype-level overlap | Full AFLOW-style protostructure-label overlap | 217 / 2000 | 10.85% |
| Structure-similarity overlap | Fingerprint retrieval followed by StructureMatcher confirmation | 144 / 2000 | 7.20% |
| Composition-level overlap | Raw-formula composition overlap | 692 / 2000 | 34.60% |

Raw-formula composition overlap is reported as an audit statistic and retained as
a sample-level flag. It is not used as the primary duplicate definition because
identical formulas can correspond to different polymorphs, space groups,
stability regimes, and properties.

## Primary Clean Split

The primary leakage-controlled benchmark excludes held-out rows with any of the
following flags:

- MP-ID overlap.
- Exact CIF-hash overlap.
- Full AFLOW-style protostructure-label overlap.
- StructureMatcher-confirmed structural-similarity overlap.

This filtering retains 1,662 of the initial 2,000 held-out examples.
