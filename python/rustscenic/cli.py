"""rustscenic CLI entry point — covers all 4 SCENIC+ stages.

    rustscenic grn       --expression --tfs --output [--seed ...]
    rustscenic aucell    --expression --regulons --output [--top-frac ...]
    rustscenic topics    --expression --output [--n-topics --n-passes ...]
    rustscenic cistarget --rankings --regulons --output [--top-frac --auc-threshold]
    rustscenic --version
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _load_expression(path: Path):
    import anndata as ad
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".h5ad":
        adata = ad.read_h5ad(path)
        return adata, list(adata.var_names), adata.n_obs
    elif suffix in (".tsv", ".csv"):
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, index_col=0)
        return df, list(df.columns), len(df)
    else:
        raise SystemExit(f"error: unsupported format {suffix}. Use .h5ad, .tsv, or .csv")


def _save(df, path: Path) -> None:
    ext = path.suffix.lower()
    if ext == ".parquet":
        df.to_parquet(path, index=False)
    elif ext in (".tsv", ".txt"):
        df.to_csv(path, sep="\t", index=False)
    elif ext == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path.with_suffix(".parquet"), index=False)


def cmd_grn(args: argparse.Namespace) -> int:
    from . import __version__
    from . import grn as rs_grn

    expr_path = Path(args.expression)
    if not expr_path.exists():
        print(f"error: expression file not found: {expr_path}", file=sys.stderr)
        return 2

    expression, gene_names, n_cells = _load_expression(expr_path)
    tfs = rs_grn.load_tfs(Path(args.tfs))
    tfs_in = [t for t in tfs if t in set(gene_names)]
    if not tfs_in:
        print(f"error: no TFs in {args.tfs} found in expression data", file=sys.stderr)
        return 2

    print(f"rustscenic {__version__}  grn  cells={n_cells}  genes={len(gene_names)}  tfs={len(tfs_in)}  seed={args.seed}",
          file=sys.stderr, flush=True)

    t0 = time.monotonic()
    out = rs_grn.infer(
        expression, tfs_in,
        n_estimators=args.n_estimators, learning_rate=args.learning_rate,
        max_features=args.max_features, subsample=args.subsample,
        max_depth=args.max_depth, early_stop_window=args.early_stop_window,
        seed=args.seed,
    )
    wall = time.monotonic() - t0
    output_path = Path(args.output)
    _save(out, output_path)
    meta = {
        "wall_clock_s": round(wall, 2), "n_edges": int(len(out)),
        "n_cells": int(n_cells), "n_genes": int(len(gene_names)),
        "n_tfs_used": len(tfs_in), "seed": args.seed,
        "rustscenic_version": __version__, "stage": "grn",
    }
    output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {output_path}  ({len(out)} edges, wall {wall:.1f}s)", file=sys.stderr)
    return 0


def cmd_aucell(args: argparse.Namespace) -> int:
    from . import __version__
    from . import aucell as rs_aucell
    import pandas as pd

    expr_path = Path(args.expression)
    reg_path = Path(args.regulons)
    if not expr_path.exists() or not reg_path.exists():
        print(f"error: input file missing", file=sys.stderr); return 2

    expression, gene_names, n_cells = _load_expression(expr_path)
    # Regulons expected as TSV: regulon_name\tgene1,gene2,...  OR  regulon_name\tgene  (long form)
    # Accept either format; auto-detect by checking first line.
    regulons: list[tuple[str, list[str]]] = []
    # Load by extension — the common workflow is `rustscenic grn --output grn.parquet`
    # followed by `rustscenic aucell --regulons grn.parquet`.
    if reg_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(reg_path)
        if not {"TF", "target", "importance"}.issubset(df.columns):
            print(f"error: {reg_path} doesn't look like a GRN adjacencies parquet "
                  f"(missing TF/target/importance columns). Got: {df.columns.tolist()}",
                  file=sys.stderr)
            return 2
        for tf, group in df.groupby("TF"):
            top_targets = group.nlargest(args.top_n_targets, "importance")["target"].tolist()
            if len(top_targets) >= args.min_genes:
                regulons.append((f"{tf}_regulon", top_targets))
    else:
        lines = reg_path.read_text().splitlines()
        if not lines:
            print(f"error: {reg_path} is empty", file=sys.stderr)
            return 2
        header = lines[0]
        if "TF" in header and "target" in header and "importance" in header:
            # GRN-adjacencies TSV/CSV
            df = pd.read_csv(reg_path, sep="\t" if reg_path.suffix == ".tsv" else ",")
            for tf, group in df.groupby("TF"):
                top_targets = group.nlargest(args.top_n_targets, "importance")["target"].tolist()
                if len(top_targets) >= args.min_genes:
                    regulons.append((f"{tf}_regulon", top_targets))
        else:
            # Plain regulons TSV: name\tgene,gene,...
            for ln in lines:
                if "\t" in ln:
                    name, genes_str = ln.split("\t", 1)
                    genes = [g.strip() for g in genes_str.split(",") if g.strip()]
                    if len(genes) >= args.min_genes:
                        regulons.append((name.strip(), genes))

    if not regulons:
        print(f"error: no regulons loaded from {reg_path}", file=sys.stderr)
        return 2

    print(f"rustscenic {__version__}  aucell  cells={n_cells}  regulons={len(regulons)}  top_frac={args.top_frac}",
          file=sys.stderr, flush=True)

    t0 = time.monotonic()
    auc = rs_aucell.score(expression, regulons, top_frac=args.top_frac)
    wall = time.monotonic() - t0

    output_path = Path(args.output)
    # AUC is (cells x regulons); save as parquet with index, or TSV
    if output_path.suffix.lower() == ".parquet":
        auc.to_parquet(output_path)
    else:
        auc.to_csv(output_path, sep="\t" if output_path.suffix == ".tsv" else ",")
    meta = {
        "wall_clock_s": round(wall, 2), "n_cells": int(n_cells),
        "n_regulons": len(regulons), "top_frac": args.top_frac,
        "rustscenic_version": __version__, "stage": "aucell",
    }
    output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {output_path}  ({auc.shape[0]}x{auc.shape[1]}, wall {wall:.1f}s)", file=sys.stderr)
    return 0


def cmd_topics(args: argparse.Namespace) -> int:
    from . import __version__
    from . import topics as rs_topics

    expr_path = Path(args.expression)
    expression, gene_names, n_cells = _load_expression(expr_path)
    print(f"rustscenic {__version__}  topics  cells={n_cells}  peaks={len(gene_names)}  K={args.n_topics}  passes={args.n_passes}",
          file=sys.stderr, flush=True)

    t0 = time.monotonic()
    res = rs_topics.fit(
        expression, n_topics=args.n_topics, n_passes=args.n_passes,
        batch_size=args.batch_size, seed=args.seed,
    )
    wall = time.monotonic() - t0

    output_path = Path(args.output)
    out_ct = output_path.with_name(output_path.stem + "_cell_topic" + output_path.suffix)
    out_tp = output_path.with_name(output_path.stem + "_topic_peak" + output_path.suffix)
    if output_path.suffix.lower() == ".parquet":
        res.cell_topic.to_parquet(out_ct)
        res.topic_peak.to_parquet(out_tp)
    else:
        res.cell_topic.to_csv(out_ct, sep="\t" if output_path.suffix == ".tsv" else ",")
        res.topic_peak.to_csv(out_tp, sep="\t" if output_path.suffix == ".tsv" else ",")
    meta = {
        "wall_clock_s": round(wall, 2), "n_cells": int(n_cells),
        "n_peaks": int(len(gene_names)), "n_topics": args.n_topics,
        "rustscenic_version": __version__, "stage": "topics",
    }
    output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_ct} + {out_tp}  (wall {wall:.1f}s)", file=sys.stderr)
    return 0


def cmd_cistarget(args: argparse.Namespace) -> int:
    from . import __version__
    from . import cistarget as rs_cistarget
    import pandas as pd

    rank_path = Path(args.rankings)
    reg_path = Path(args.regulons)

    if rank_path.suffix.lower() == ".feather":
        rankings = rs_cistarget.load_aertslab_feather(rank_path)
    elif rank_path.suffix.lower() in (".tsv", ".csv"):
        sep = "\t" if rank_path.suffix == ".tsv" else ","
        rankings = pd.read_csv(rank_path, sep=sep, index_col=0)
    else:
        raise SystemExit(f"error: unsupported rankings format {rank_path.suffix}")

    regulons = []
    for ln in reg_path.read_text().strip().splitlines():
        if "\t" in ln:
            name, genes_str = ln.split("\t", 1)
            genes = [g.strip() for g in genes_str.split(",") if g.strip()]
            if genes:
                regulons.append((name.strip(), genes))

    print(f"rustscenic {__version__}  cistarget  motifs={rankings.shape[0]}  genes={rankings.shape[1]}  regulons={len(regulons)}",
          file=sys.stderr, flush=True)

    t0 = time.monotonic()
    out = rs_cistarget.enrich(rankings, regulons, top_frac=args.top_frac, auc_threshold=args.auc_threshold)
    wall = time.monotonic() - t0

    output_path = Path(args.output)
    _save(out, output_path)
    print(f"wrote {output_path}  ({len(out)} enriched, wall {wall:.1f}s)", file=sys.stderr)
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    """End-to-end: preproc + topics + grn + cistarget + aucell, one call."""
    from . import pipeline

    if (args.fragments is None) != (args.peaks is None):
        print("error: --fragments and --peaks must be supplied together", file=sys.stderr)
        return 2

    result = pipeline.run(
        rna=Path(args.rna),
        output_dir=Path(args.output),
        fragments=Path(args.fragments) if args.fragments else None,
        peaks=Path(args.peaks) if args.peaks else None,
        tfs=Path(args.tfs),
        motif_rankings=Path(args.motif_rankings) if args.motif_rankings else None,
        grn_n_estimators=args.grn_n_estimators,
        grn_top_targets=args.grn_top_targets,
        aucell_top_frac=args.aucell_top_frac,
        topics_n_topics=args.topics_n_topics,
        topics_n_passes=args.topics_n_passes,
        cistarget_top_frac=args.cistarget_top_frac,
        cistarget_auc_threshold=args.cistarget_auc_threshold,
        seed=args.seed,
        verbose=True,
    )
    print(f"pipeline done → {result.output_dir}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    from . import __version__
    p = argparse.ArgumentParser(prog="rustscenic", description="Fast SCENIC+ stage replacements (Rust + PyO3)")
    p.add_argument("--version", action="version", version=f"rustscenic {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pg = sub.add_parser("grn", help="GRN inference (GRNBoost2 replacement)")
    pg.add_argument("--expression", required=True); pg.add_argument("--tfs", required=True)
    pg.add_argument("--output", required=True); pg.add_argument("--seed", type=int, default=777)
    pg.add_argument("--n-estimators", type=int, default=5000)
    pg.add_argument("--learning-rate", type=float, default=0.01)
    pg.add_argument("--max-features", type=float, default=0.1); pg.add_argument("--subsample", type=float, default=0.9)
    pg.add_argument("--max-depth", type=int, default=3); pg.add_argument("--early-stop-window", type=int, default=25)
    pg.set_defaults(func=cmd_grn)

    pa = sub.add_parser("aucell", help="Regulon activity scoring (AUCell replacement)")
    pa.add_argument("--expression", required=True)
    pa.add_argument("--regulons", required=True, help="Path to grn.tsv/parquet or regulons TSV (name\\tgene,gene,...)")
    pa.add_argument("--output", required=True); pa.add_argument("--top-frac", type=float, default=0.05)
    pa.add_argument("--top-n-targets", type=int, default=50,
                    help="When regulons input is a grn output, keep top-N targets per TF (default 50)")
    pa.add_argument("--min-genes", type=int, default=10,
                    help="Drop regulons with fewer than this many genes (default 10)")
    pa.set_defaults(func=cmd_aucell)

    pt = sub.add_parser("topics", help="Topic modeling (pycisTopic LDA replacement, online VB)")
    pt.add_argument("--expression", required=True, help="Cells × peaks / cells × genes matrix")
    pt.add_argument("--output", required=True, help="Output prefix; writes *_cell_topic and *_topic_peak")
    pt.add_argument("--n-topics", type=int, default=50); pt.add_argument("--n-passes", type=int, default=10)
    pt.add_argument("--batch-size", type=int, default=256); pt.add_argument("--seed", type=int, default=42)
    pt.set_defaults(func=cmd_topics)

    pc = sub.add_parser("cistarget", help="Motif enrichment (pycistarget replacement, core algorithm)")
    pc.add_argument("--rankings", required=True, help=".feather (aertslab) or .tsv/.csv motif × gene rankings")
    pc.add_argument("--regulons", required=True, help="Regulons TSV (name\\tgene,gene,...)")
    pc.add_argument("--output", required=True); pc.add_argument("--top-frac", type=float, default=0.05)
    pc.add_argument("--auc-threshold", type=float, default=0.05)
    pc.set_defaults(func=cmd_cistarget)

    pp = sub.add_parser(
        "pipeline",
        help="End-to-end SCENIC+ (preproc + topics + grn + cistarget + aucell)",
    )
    pp.add_argument("--rna", required=True, help="RNA expression (.h5ad)")
    pp.add_argument("--output", required=True, help="Output directory for all artifacts")
    pp.add_argument("--tfs", required=True, help="Newline-separated TF names file")
    pp.add_argument("--fragments", default=None, help="Optional: 10x fragments.tsv[.gz]")
    pp.add_argument("--peaks", default=None, help="Optional: consensus peaks BED (required with --fragments)")
    pp.add_argument("--motif-rankings", default=None, help="Optional: motif ranking parquet/feather")
    pp.add_argument("--grn-n-estimators", type=int, default=500)
    pp.add_argument("--grn-top-targets", type=int, default=50)
    pp.add_argument("--aucell-top-frac", type=float, default=0.05)
    pp.add_argument("--topics-n-topics", type=int, default=30)
    pp.add_argument("--topics-n-passes", type=int, default=3)
    pp.add_argument("--cistarget-top-frac", type=float, default=0.05)
    pp.add_argument("--cistarget-auc-threshold", type=float, default=0.05)
    pp.add_argument("--seed", type=int, default=777)
    pp.set_defaults(func=cmd_pipeline)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
