# Slide/code coverage and deliberate exclusions

This inventory was checked against both the original 36-slide storyline and the
27-slide September 18 five-part deck, the diagnostic producers, the decision
log and the September correctness repairs. Later corrections take precedence
over older interpretations. This is a reusable toolkit, not numerical study replay.

## Coverage

| Slide/code analysis | Portable entry point | Important boundary |
|---|---|---|
| Protein-disjoint evaluation contract | `check-split`, `attach-split`, `validate` | Frozen lookup included; operational clusters do not prove remote-homology exclusion |
| Fixed versus source-selected readout | `probe`, `evaluate` | Inner scaler/rank/alpha fits use source training labels only |
| Fixed-prior and substitution-floor comparison | `baselines`, `evaluate`, `compare` | Precomputed LLR/MSA scores; optional BLOSUM62 single-mutant floor |
| Random-ranking reference | `evaluate` | Tie-preserving evaluation-label permutation; exchangeability assumption explicit |
| Shuffled-source-label control | `probe --shuffle-source-labels` | Distinct refit control; repeat prespecified seeds |
| Source/oracle performance, gap and recovery R | `diagnose`, `evaluate --oracle-column` | Same held-out rows, support access, weak denominator handling, no ceiling claim |
| Position-overlap versus disjoint local signal | `support` | Corrected replacements, matched substitution counts, fixed/tuned readouts |
| Embedding shift and source calibration | `shift` | MMD², centroid/covariance distances, assay-level cluster-excluding reference |
| MMD versus source performance/oracle gap | `associate` | Within-task descriptive associations; audited assay joins, no causal attribution |
| Embedding visual panels | `geometry` | Shared capped-sample PCA and quality checks; no result-selected panels |
| Probe-weight similarity | `cosine`; separate field in `diagnose` | Fold-local and source/oracle models are named separately |
| Cosine estimation/reference checks | `cosine --reference-repeats` | Same-pool assay-bootstrap and label-shuffle references, not upper bounds |
| Calibration and error modes | `errors` | Original-prediction bias is distinct from evaluation-fitted intercept residuals |
| Pair ordering and tail retrieval | `errors` | Tie-aware pair accuracy, top/bottom AP and AUROC, declared tail fraction |
| Variant-library composition | `composition` | Coverage, position/type JSD, unsupported type mass; input rows only |
| Exact-background context agreement | `context` | Sequence/variant identity and position-block uncertainty |
| Pooling/modality/interface contrasts | independent `probe` runs + `compare` | Matched rows and selection; complete interfaces, not isolated causal modalities |
| Add-information versus standalone performance | `probe --aux-embeddings` + `compare` | Exact-key concatenation and separate source selection; no target-selected best family |
| Cross-task matrix and prior increments | `cross-task` | Ranking Holm family separate from unadjusted increment intervals |
| Context/subgroup heterogeneity | `evaluate --group-by` | Supplied assay annotations, task-separated assay-macro summaries |
| Provenance concentration | `provenance` | Assay and row concentration both reported, not a causal source-effect test |
| External-corpus performance summaries | `evaluate`, `compare` | Saved predictions; external generation/overlap auditing remain user responsibilities |
| Conditional paired uncertainty | `compare`, `evaluate`, `cross-task` | Supercluster resampling preserves assay-macro weighting; no refit uncertainty |

## Deliberate exclusions

- **Automatic causal driver categories and their threshold sweeps.** Historical
  mapping/covariate/noisy-label heuristics do not establish a unique cause.
  Continuous measurements, error strata and within-task associations are included.
- **Legacy post-calibration `residual_bias`.** Fitting an intercept on those same
  evaluation rows makes it approximately zero by construction. `errors` uses
  original-prediction rank bias with an explicit prediction-scale flag.
- **Cosine normalized to a supposed aligned upper bound, or intervals treating
  overlapping fold pairs as independent.** Reference distributions are retained
  descriptively without these unsupported interpretations.
- **SAE feature-mechanism inference, feature ablation, NLA, deep task architectures,
  residual-network capacity sweeps, LoReFT/LESS and test-time training.** These are
  research/training branches, not portable embedding diagnostics. Their saved
  predictions can still be compared on a permitted matched scope.
- **The incomplete fitted variant-balancing campaign.** The completed metadata
  audit is included, not efficacy inferred from successful shards.
- **MSA/PLM extraction, biological APIs and external-corpus acquisition.** Users
  supply their vectors, fixed priors and metadata.
- **Replicate noise ceilings without replicate data/provenance, or automatic
  claims of clean cross-corpus exclusion.** Neither follows from embeddings and
  one aggregate effect column.

## Source locations

These are upstream provenance identifiers, not dependencies or missing files
here. Hashes appear in [PROVENANCE](../PROVENANCE.md).

- Main diagnosis, moment distances and corrected bias:
  `experiment/diagnostics/transfer/diagnose_transfer_shift.py`.
- Baseline/random/recovery summaries:
  `experiment/diagnostics/transfer/diagnostic_statistical_baselines.py` and
  `consolidate_diagnostic_baselines.py` in that directory.
- PCA/shift views: `experiment/diagnostics/transfer/diagnostic_embedding_visualization.py`.
- Error strata: `experiment/diagnostics/transfer/diagnostic2_calibration_errormode.py`.
- Matched cosine references:
  `experiment/diagnostics/matched_calibration/cosine/run_matched_fold_local_calibration.py`.
- Cross-task selection/null: `experiment/cross_task/source_alpha_selection/core.py`.
- Interpretation: `experiment/DECISION_LOG.md`,
  `reports/diagnostic_repairs_20260918/REPORT.md`, and
  `reports/paper_storyline_completion_20260918/REPORT.md`.

These are adaptations, not wholesale copies with private cache dependencies.
No study result was rerun or changed. Decision impact: none.

[Back to README](../README.md) · [Runnable recipes](RECIPES.md)
