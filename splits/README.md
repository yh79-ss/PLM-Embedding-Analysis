# Frozen ProteinGym protein-level split

Use [protein_fold_lookup_c08_v3_seed893.csv](protein_fold_lookup_c08_v3_seed893.csv)
as the authority for exact assignments. This is a byte-identical copy of the
source project's `dataset/ProteinGym/protein_fold_lookup_c08_v3_seed893.csv`.

SHA256:

```text
fca42ef22d6c305982c64f3eb0032b68757251262357edb9fcf231d4c4ed5280
```

The lookup contains 217 unique assay records, 186 distinct saved protein IDs,
and 163 operational superclusters, including 16 with multiple distinct protein
IDs and 147 single-protein clusters. `UniProt_ID` values are the historical saved
identifiers, not necessarily current UniProt accessions; preserve their spelling.

| Column | Definition |
|---|---|
| `DMS_id` | Exact assay identifier; one record per assay |
| `UniProt_ID` | Saved protein identifier |
| `super_cluster` | Operational connected sequence group |
| `fold_protein_5` | Integer 0–4 outer fold |

For outer evaluation fold `k`, train and select using permitted rows from other
folds. All assays and variants for a protein stay together, as do all proteins
in a supercluster. No protein or supercluster crosses folds. A fold contains
many superclusters and is not itself a biological family.

## Recorded construction

1. The historical input used the wild-type target-protein sequence associated
   with each of the 217 assay records.
2. Initial MMseqs2 `easy-cluster` used
   `--min-seq-id 0.30 --cov-mode 1 -c 0.8 --cluster-mode 0`, producing 167 initial
   clusters. Coverage mode 1 refers to **target sequence coverage**, not
   bidirectional coverage.
3. An all-versus-all search supplied qualifying directed relationships at
   at least 30% identity. Qualifying directed hits were treated as undirected
   edges; Union-Find merged related initial clusters. Four recorded bridges
   were KCNH2_HUMAN–PHOT_CHLRE (33.0%), RAF1_HUMAN–MET_HUMAN (31.1%),
   SRC_HUMAN–MET_HUMAN (37.5%), and ANCSZ–MET_HUMAN (36.4%). Merging yielded
   163 superclusters. These are recorded operational edges, not new biological
   homology assertions by this package.
4. Whole superclusters were assigned to five folds. A metadata-only search over
   10,000 candidate seeds selected seed 893 with recorded objective
   `10 * MutantRatio + 2 * TypeImbal + 0.5 * TaxonImbal` and assay-count balance
   constraints. `MutantRatio` is maximum/minimum fold mutant count;
   `TypeImbal` concerns single- versus multi-mutant composition, and
   `TaxonImbal` concerns taxonomic composition. Labels, embeddings and downstream
   performance were not used for this recorded split-selection procedure.

| Fold | Assays | Superclusters |
|---|---:|---:|
| 0 | 43 | 30 |
| 1 | 44 | 33 |
| 2 | 44 | 34 |
| 3 | 43 | 33 |
| 4 | 43 | 33 |

The initial clustering version and complete bridge-search coverage flags were
not preserved. The metadata imbalance function is not fully specified by the
saved description. Accordingly, this package distributes the verified lookup,
not a purported exact regeneration script. The label-free historical procedure
is documented from the author's specification and subsequent lineage audit; it
has not been independently rerun here. Lookup counts and separation are directly
checkable and covered by tests.

This split establishes held-out proteins and separation of the **recorded
operational sequence relationships**. It does not establish exclusion of all
remote homologs, shared domains, structural folds or biological families.
"Cross-family" in the originating project's terminology refers to these
operational groups and must retain that qualification.

## Provenance

Source-project artifacts (paths identify the original research repository;
research reports are intentionally not bundled here):

- `dataset/ProteinGym/protein_fold_lookup_c08_v3_seed893.csv`
- `paper_audit/long_track_diagnosis_first/canonical_split_lineage_final_closure_v1/audit/final_split_microcontract.yaml`
- `paper_audit/long_track_diagnosis_first/canonical_split_lineage_final_closure_v1/audit/canonical_split_lookup_identity.tsv`
- `paper_audit/codex_audit_20260718T175744Z/science_paper_reset/ppt_storyline_revision/protein_level_split_revision/author_verified_split_specification.yaml`

The later verified microcontract gives 16 multi-protein clusters; the earlier
author specification listed 17. The released lookup and later audit determine
the count used here. No historical artifact was changed.

Bundled on 2026-09-26 from the existing local study artifact. No external API
query, sequence download or new split generation was performed. Assay/protein
identifiers derive from the study's ProteinGym inputs; underlying dataset and
assay-source terms remain applicable. No effect labels, sequence files or
embedding caches accompany this split.
