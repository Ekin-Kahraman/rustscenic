from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def test_aucell_cli_groups_long_form_regulons_by_name(tmp_path, monkeypatch):
    import rustscenic.aucell
    import rustscenic.cli

    captured = {}

    def fake_load_expression(_path):
        return object(), ["g1", "g2", "g3", "g4"], 2

    def fake_score(_expression, regulons, *, top_frac):
        captured["regulons"] = regulons
        captured["top_frac"] = top_frac
        return pd.DataFrame(
            {"R": [0.1, 0.2], "S": [0.3, 0.4]},
            index=["c0", "c1"],
        )

    monkeypatch.setattr(rustscenic.cli, "_load_expression", fake_load_expression)
    monkeypatch.setattr(rustscenic.aucell, "score", fake_score)

    expr = tmp_path / "expr.tsv"
    expr.write_text("cell\tg1\nc0\t1\n")
    regs = tmp_path / "regulons.tsv"
    regs.write_text("R\tg1\nR\tg2\nS\tg3,g4\n")
    out = tmp_path / "auc.tsv"

    rc = rustscenic.cli.cmd_aucell(
        argparse.Namespace(
            expression=str(expr),
            regulons=str(regs),
            output=str(out),
            top_n_targets=50,
            min_genes=1,
            top_frac=0.05,
        )
    )

    assert rc == 0
    assert captured["regulons"] == [("R", ["g1", "g2"]), ("S", ["g3", "g4"])]
    assert captured["top_frac"] == 0.05
    assert out.exists()


def test_pipeline_cli_forwards_full_multiome_stage_options(tmp_path, monkeypatch):
    import rustscenic.cli
    import rustscenic.pipeline

    captured = {}

    class FakeResult:
        output_dir = tmp_path / "out"

    def fake_run(**kwargs):
        captured.update(kwargs)
        return FakeResult()

    monkeypatch.setattr(rustscenic.pipeline, "run", fake_run)

    rc = rustscenic.cli.main(
        [
            "pipeline",
            "--rna", str(tmp_path / "rna.h5ad"),
            "--output", str(tmp_path / "out"),
            "--tfs", str(tmp_path / "tfs.txt"),
            "--adata-atac", str(tmp_path / "atac.h5ad"),
            "--motif-rankings", str(tmp_path / "motif_rankings.parquet"),
            "--motif-annotations", str(tmp_path / "motif_annotations.tsv"),
            "--region-motif-rankings", str(tmp_path / "region_motifs.feather"),
            "--gene-coords", str(tmp_path / "gene_coords.parquet"),
            "--grn-n-estimators", "17",
            "--grn-max-features", "0.25",
            "--grn-early-stop-window", "13",
            "--grn-early-stop-mode", "legacy_inbag",
            "--grn-target-block-size", "11",
            "--grn-top-targets", "9",
            "--grn-regulon-polarities", "activating",
            "--grn-rho-threshold", "0.07",
            "--grn-rho-mask-dropouts",
            "--aucell-top-frac", "0.2",
            "--topics-n-topics", "7",
            "--topics-n-passes", "3",
            "--topics-method", "gibbs",
            "--topics-n-iters", "19",
            "--topics-n-threads", "4",
            "--cistarget-top-frac", "0.15",
            "--cistarget-auc-threshold", "0.01",
            "--cistarget-nes-threshold", "2.5",
            "--enhancer-max-distance", "123456",
            "--enhancer-min-abs-corr", "0.33",
            "--eregulon-min-target-genes", "6",
            "--eregulon-min-enhancer-links", "3",
            "--skip-integrated-adata",
            "--seed", "123",
        ]
    )

    assert rc == 0
    assert captured["rna"] == Path(tmp_path / "rna.h5ad")
    assert captured["output_dir"] == Path(tmp_path / "out")
    assert captured["tfs"] == Path(tmp_path / "tfs.txt")
    assert captured["adata_atac"] == Path(tmp_path / "atac.h5ad")
    assert captured["fragments"] is None
    assert captured["peaks"] is None
    assert captured["motif_rankings"] == Path(tmp_path / "motif_rankings.parquet")
    assert captured["motif_annotations"] == Path(tmp_path / "motif_annotations.tsv")
    assert captured["region_motif_rankings"] == Path(tmp_path / "region_motifs.feather")
    assert captured["gene_coords"] == Path(tmp_path / "gene_coords.parquet")
    assert captured["grn_n_estimators"] == 17
    assert captured["grn_max_features"] == 0.25
    assert captured["grn_early_stop_window"] == 13
    assert captured["grn_early_stop_mode"] == "legacy_inbag"
    assert captured["grn_target_block_size"] == 11
    assert captured["grn_top_targets"] == 9
    assert captured["grn_regulon_polarities"] == "activating"
    assert captured["grn_rho_threshold"] == 0.07
    assert captured["grn_rho_mask_dropouts"] is True
    assert captured["aucell_top_frac"] == 0.2
    assert captured["topics_n_topics"] == 7
    assert captured["topics_n_passes"] == 3
    assert captured["topics_method"] == "gibbs"
    assert captured["topics_n_iters"] == 19
    assert captured["topics_n_threads"] == 4
    assert captured["cistarget_top_frac"] == 0.15
    assert captured["cistarget_auc_threshold"] == 0.01
    assert captured["cistarget_nes_threshold"] == 2.5
    assert captured["enhancer_max_distance"] == 123456
    assert captured["enhancer_min_abs_corr"] == 0.33
    assert captured["eregulon_min_target_genes"] == 6
    assert captured["eregulon_min_enhancer_links"] == 3
    assert captured["write_integrated_adata"] is False
    assert captured["seed"] == 123
    assert captured["verbose"] is True


def test_pipeline_cli_rejects_adata_atac_with_fragments(tmp_path, capsys):
    import rustscenic.cli

    rc = rustscenic.cli.main(
        [
            "pipeline",
            "--rna", str(tmp_path / "rna.h5ad"),
            "--output", str(tmp_path / "out"),
            "--tfs", str(tmp_path / "tfs.txt"),
            "--adata-atac", str(tmp_path / "atac.h5ad"),
            "--fragments", str(tmp_path / "fragments.tsv.gz"),
            "--peaks", str(tmp_path / "peaks.bed"),
        ]
    )

    assert rc == 2
    assert "--adata-atac cannot be combined" in capsys.readouterr().err


def test_add_cor_cli_emits_signed_adjacencies(tmp_path):
    import numpy as np
    import rustscenic.cli

    expression_path = tmp_path / "expression.csv"
    pd.DataFrame(
        {
            "TF": [0.0, 1.0, 2.0, 3.0],
            "up": [0.0, 2.0, 4.0, 6.0],
            "down": [3.0, 2.0, 1.0, 0.0],
        },
        index=["c0", "c1", "c2", "c3"],
    ).to_csv(expression_path)
    adjacency_path = tmp_path / "grn.parquet"
    pd.DataFrame(
        {
            "TF": ["TF", "TF"],
            "target": ["up", "down"],
            "importance": [2.0, 1.0],
        }
    ).to_parquet(adjacency_path, index=False)
    output_path = tmp_path / "signed.parquet"

    rc = rustscenic.cli.main(
        [
            "add-cor",
            "--expression", str(expression_path),
            "--adjacencies", str(adjacency_path),
            "--output", str(output_path),
        ]
    )

    assert rc == 0
    out = pd.read_parquet(output_path)
    assert out["regulation"].tolist() == [1, -1]
    np.testing.assert_allclose(out["rho"], [1.0, -1.0], atol=1e-12)

    auc_path = tmp_path / "signed_auc.parquet"
    rc = rustscenic.cli.main(
        [
            "aucell",
            "--expression", str(expression_path),
            "--regulons", str(output_path),
            "--output", str(auc_path),
            "--top-frac", "0.75",
            "--top-n-targets", "2",
            "--min-genes", "1",
        ]
    )
    assert rc == 0
    assert pd.read_parquet(auc_path).columns.tolist() == [
        "TF_activator",
        "TF_repressor",
    ]


def test_doctor_cli_reports_backend_capabilities(monkeypatch, capsys):
    import rustscenic.backend
    import rustscenic.cli

    payload = {
        "ok": True,
        "extension_error": None,
        "required_symbols": {"grn": ["grn_infer"]},
        "missing_symbols": [],
    }
    monkeypatch.setattr(rustscenic.backend, "backend_capabilities", lambda: payload)

    rc = rustscenic.cli.main(["doctor"])

    assert rc == 0
    assert '"ok": true' in capsys.readouterr().out

    rc = rustscenic.cli.main(["doctor", "--json"])

    assert rc == 0
    assert '"ok": true' in capsys.readouterr().out


def test_pipeline_cli_runs_full_multiome_smoke(tmp_path):
    import gzip

    import anndata as ad
    import numpy as np
    import rustscenic.cli

    rng = np.random.default_rng(42)
    n_cells = 80
    cells = [f"cell{i}" for i in range(n_cells)]
    cluster = np.array([i % 2 for i in range(n_cells)], dtype=np.int8)
    signal = cluster.astype(np.float32)
    genes = ["TF_A", "TF_B"] + [f"G{i:03d}" for i in range(10)]
    X = rng.normal(0.2, 0.03, size=(n_cells, len(genes))).astype(np.float32)
    X[:, 0] += 1.0 - signal
    X[:, 1] += signal
    X[:, 2:7] += (1.0 - signal)[:, None]
    X[:, 7:12] += signal[:, None]
    X = np.clip(X, 0, None)
    rna_path = tmp_path / "rna.h5ad"
    ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=genes),
    ).write_h5ad(rna_path)

    tfs_path = tmp_path / "tfs.txt"
    tfs_path.write_text("TF_A\nTF_B\n")

    frag_lines = []
    for cell, group in zip(cells, cluster, strict=True):
        base = 10_000 if group == 0 else 110_000
        for _ in range(6):
            start = base + int(rng.integers(0, 1_000))
            frag_lines.append(f"chr1\t{start}\t{start + 120}\t{cell}\t1")
    fragments_path = tmp_path / "fragments.tsv.gz"
    with gzip.open(fragments_path, "wt") as handle:
        handle.write("\n".join(frag_lines) + "\n")

    peaks_path = tmp_path / "peaks.bed"
    peaks_path.write_text(
        "chr1\t9500\t11500\tpeak_A\n"
        "chr1\t109500\t111500\tpeak_B\n"
    )

    gene_coords_path = tmp_path / "gene_coords.parquet"
    pd.DataFrame(
        {
            "gene": genes[2:],
            "chrom": ["chr1"] * 10,
            "tss": [10_100] * 5 + [110_100] * 5,
        }
    ).to_parquet(gene_coords_path, index=False)

    motif_rankings_path = tmp_path / "motif_rankings.parquet"
    ranks = np.tile(np.arange(len(genes), dtype=np.int32), (2, 1))
    ranks[0, :] = len(genes) - 1
    ranks[1, :] = len(genes) - 1
    ranks[0, [2, 3, 4, 5, 6]] = np.arange(5, dtype=np.int32)
    ranks[1, [7, 8, 9, 10, 11]] = np.arange(5, dtype=np.int32)
    pd.DataFrame(ranks, index=["motif_A", "motif_B"], columns=genes).to_parquet(
        motif_rankings_path
    )

    out = tmp_path / "out"
    rc = rustscenic.cli.main(
        [
            "pipeline",
            "--rna", str(rna_path),
            "--output", str(out),
            "--tfs", str(tfs_path),
            "--fragments", str(fragments_path),
            "--peaks", str(peaks_path),
            "--motif-rankings", str(motif_rankings_path),
            "--gene-coords", str(gene_coords_path),
            "--grn-n-estimators", "8",
            "--grn-max-features", "1.0",
            "--grn-top-targets", "2",
            "--aucell-top-frac", "0.2",
            "--topics-n-topics", "2",
            "--topics-n-passes", "1",
            "--cistarget-top-frac", "0.5",
            "--cistarget-auc-threshold", "0.0",
            "--enhancer-min-abs-corr", "0.0",
            "--eregulon-min-target-genes", "1",
            "--eregulon-min-enhancer-links", "1",
            "--skip-integrated-adata",
            "--seed", "7",
        ]
    )

    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_cistarget_rows"] > 0
    assert manifest["n_enhancer_links"] > 0
    assert Path(manifest["eregulons_path"]).exists()
    assert manifest["backend_execution"]["preproc"]["engine"] == "rust"
    assert manifest["backend_execution"]["topics"]["engine"] == "rust"
    assert manifest["backend_execution"]["grn"]["engine"] == "rust"
    assert manifest["backend_execution"]["grn_correlation"]["engine"] == "rust"
    assert manifest["grn_fit"]["early_stop_mode"] == "arboreto"
    assert manifest["grn_fit"]["target_count"] == len(genes)
    assert manifest["grn_correlation"]["method"] == "pearson"
    assert manifest["backend_execution"]["cistarget"]["engine"] == "rust"
    assert manifest["backend_execution"]["enhancer"]["engine"] == "rust"
    assert manifest["backend_execution"]["eregulons"]["engine"] == "rust"
    assert manifest["backend_execution"]["aucell"]["engine"] == "rust"
    assert manifest["backend_execution"]["integrated_adata"]["engine"] == "skipped"
    candidates = json.loads((out / "candidate_regulons.json").read_text())
    assert candidates
    assert all(
        name.endswith(("_activator", "_repressor")) for name in candidates
    )
    assert {int(v) for v in pd.read_parquet(out / "grn.parquet")["regulation"]} <= {
        -1,
        0,
        1,
    }
