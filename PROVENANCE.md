# Provenance and scope of the portable implementation

Prepared 2026-09-26 from the existing Multi-VEP research workspace. The source
repository's recorded HEAD was `ea9ca5c9d4c62dd5c8912679b052c0a4fa775981`, but the
workspace contained uncommitted diagnostic repairs and newer slide/report files.
Therefore the HEAD alone does not identify every inspected source. Hashes below
record the actual inspected files. The original working tree and completed
results were not rewritten by this extraction.

The primary presentation reference was
`cross_family_VEP_five_part_20260918.pptx` (SHA256
`68db631195bace3c2b619343c860ae367fa3434fab655552f4861c6646a437f0`).
Its five acts organize the questions, not a promise that this package recreates
every experimental branch. No slides, figures, numerical result tables, training
checkpoints or raw data are bundled.

## Included methods and their source lineage

These paths refer to the originating research repository; they are provenance
identifiers, not required files or imports in this standalone package.

| Portable tool | Source implementation / evidence |
|---|---|
| Source-only alpha selection | `experiment/_shared/source_selection.py`; the source-selection families under `experiment/diagnostics/source_selected_bstar/` and `experiment/interfaces/source_alpha_selection/` |
| Embedding distances and reference calibration | `experiment/diagnostics/transfer/diagnose_transfer_shift.py`; `experiment/diagnostics/transfer/analysis/mmd_traintrain_assaylevel.py`; `reports/mmd_traintrain_assaylevel/report.md` |
| Position-support diagnostic | `experiment/diagnostics/position_support/run.py`; `reports/paper_revision_execution_20260907/position_support_v2/REPORT.md` |
| Composition and coverage | `experiment/diagnostics/variant_composition/metadata_reaudit.py`; `experiment/diagnostics/variant_composition/raw_metadata_coverage.py`; `reports/paper_revision_execution_20260907/variant_composition/REPORT.md` |
| Context concordance | `experiment/diagnostics/paired_context/run.py`; `reports/paper_revision_execution_20260907/paired_context/REPORT.md` |
| Conditional paired cluster intervals | `experiment/reproducibility/cluster_uncertainty/cluster_sensitivity.py` |
| Evaluation guardrails and interpretation | `experiment/DECISION_LOG.md`; `reports/diagnostic_repairs_20260918/REPORT.md`; the repository's `AGENTS.md` |

### Source hashes

```text
a893f3c2de4829e971d33cb26360f4964e7e9da7b4837f793112ec7926fc682f  experiment/diagnostics/transfer/diagnose_transfer_shift.py
e0d5fc59dae10b391881b38c6a57b454c2f1862ab7bf904c3fe20d6256d9ba08  experiment/diagnostics/transfer/analysis/mmd_traintrain_assaylevel.py
6560cc47de0f90374e42b4fc1915bd2c1e89e504acedc1710c042f140c93b97a  experiment/diagnostics/position_support/run.py
4d41e3a5ce3157184fc1948636fed5dc6da480899e41873e9d5d9dbcbcabe7a1  experiment/diagnostics/paired_context/run.py
ce6ea153e91909b00d2f893435c7f06adaff8d59e65f26075706f66cc975249f  experiment/diagnostics/variant_composition/metadata_reaudit.py
4c6f855c4a6a3b0781e61cce75794efbe3085d0bfc59308387a1276ec954915d  experiment/diagnostics/variant_composition/raw_metadata_coverage.py
1c37897d68ede5e1b7e9e643e722dd0d99c4d0667adb9adf9487f97c4d4f980f  experiment/_shared/source_selection.py
b0e322fb4822a33a9d90ed2f5f9efa61cd918e27b47aced3c36cf9a801349684  experiment/reproducibility/cluster_uncertainty/cluster_sensitivity.py
```

## What is preserved and what differs

This is a standalone adaptation to user-provided keyed arrays. It is not a copy
of private-path producer scripts with their original result/cache dependencies.

- The canonical split CSV is preserved byte for byte; see
  [split provenance](splits/README.md). No split regeneration is asserted.
- Outer folds and source-only inner selection preserve the no-target-label
  boundary. The generic probe is inspired by B* but is not a B* replay: it uses
  user-defined rows, exact SVD, equal-row fitting, explicit preprocessing and the
  documented rank convention, rather than reconstructing the study's cached
  populations, numerical solver/thread contracts, caps and fitted artifacts.
- The maximum-based source alpha tie rule uses the repaired semantics. All
  source validation assays must have finite correlations for selection. There
  is no target-driven representation/method-family selection.
- Position support follows the corrected v2 construction: replacement rows must
  occur at actual evaluation positions and preserve substitution-type counts.
  It retains the support `rank/n` convention, larger-alpha tie preference and
  position-grouped support validation. Portable code uses SVD, a new seeded
  hash convention and explicit full-budget eligibility. It does not load the
  original five-assay contract, frozen common population or source predictions.
- MMD preserves the multi-scale biased statistic, pair-adaptive bandwidth and
  assay-versus-source-pool reference with whole-cluster exclusion. Here,
  reference and held-out statistics are generated together, without historical
  prediction artifacts. The label-free input universe is exactly the supplied
  metadata/embedding keys; the historical producer filtered for label
  availability. Default projection and sampling use this package's recorded
  seeds and float64 arrays, not historical cache identities.
- Composition implements assay-equal source histograms, coverage and JSD; it
  does not fit the incomplete source-reweighting campaign or infer raw-versus-
  retained row lineage. Users supply those populations separately.
- Context analysis validates full backgrounds and mutated sequences, exact
  variant joins, and position-block bootstrap ranks. It omits the study's
  publication/replicate metadata audit and background-equal task aggregation.
  Resamples are expanded explicitly for clarity, which can be slower on large
  assays than the original weighted-rank implementation.
- Comparison intervals preserve the assay-macro paired contrast while resampling
  clusters. They do not reproduce study-specific registration, simultaneous
  bounds, multiple-testing procedures or two-level source-refit uncertainty.
- Direct coefficient-cosine, SAE training/interpretability, PLM fine-tuning,
  residual-network sweeps, test-time training, external corpus ingestion,
  biological database queries and mutation-level mechanistic ranking are not
  part of the exported toolset.

Changing the population, rank convention, pooling, solver, selection grid,
support design or metric changes the estimand or numerical procedure. Declare
these choices rather than attaching the original study's numerical claims to
new outputs.

## Validation and distribution

Validation uses synthetic arrays and the small split lookup only. Tests cover
identity joins, known split counts and checksum, group leakage rejection,
source alpha selection, target-label perturbation invariance, corrected support
replacement geometry, descriptive analysis without effect labels, conditional
cluster aggregation, output preservation and every analysis command. These are
software/protocol checks, not fresh evidence for scientific performance.

No new project-level scientific conclusion is introduced. Decision impact:
none. No model was fitted to the originating study's biological data during
packaging, and no existing completed result table was regenerated.

Code attribution and a distribution license should be finalized by the
repository owner. This package does not assert a new license over the source
project. ProteinGym and individual assay data remain subject to their original
terms. No external API was queried and no external biological payload was
downloaded for this extraction.
