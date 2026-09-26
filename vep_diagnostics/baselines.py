"""Package fixed scalar priors, optionally calculating a BLOSUM62 single-mutant floor."""
from .data import IDENTITY, KEY, Run, load_metadata, scores, single_substitutions


def run_baselines(args):
    frame = load_metadata(args.metadata)
    output = frame[list(dict.fromkeys(KEY + IDENTITY))].copy()
    for column in args.score_columns:
        if column in KEY + IDENTITY + ["score", "analysis_scope", "label_protocol"]:
            raise ValueError(f"Reserved baseline column: {column}")
        output[column] = scores(frame, column)
    if args.blosum62:
        try:
            from Bio.Align import substitution_matrices
        except ImportError as exc:
            raise ValueError("BLOSUM62 needs the optional dependency: pip install -e '.[baseline]'") from exc
        single = single_substitutions(frame)
        if len(single) != len(frame):
            raise ValueError("BLOSUM62 supports canonical nonsynonymous single substitutions only; declare a filtered population first")
        matrix = substitution_matrices.load("BLOSUM62")
        output["blosum62"] = [float(matrix[w, m]) for w, m in zip(single.wt_aa, single.mut_aa)]
    if not args.score_columns and not args.blosum62:
        raise ValueError("Supply --score-columns or --blosum62")
    run = Run(args, "no-target-label fixed scores as declared by user; no fitting", [args.metadata])
    run.table("predictions.csv", output)
    run.finish(score_direction="larger means larger expected effect score; caller declares fixed-prior provenance and direction",
               blosum="BioPython bundled BLOSUM62 substitution entry, not an effect probability; no sign selection using evaluation labels" if args.blosum62 else None)
