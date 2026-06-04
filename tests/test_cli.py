from __future__ import annotations

import argparse
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
            "--grn-target-block-size", "11",
            "--grn-top-targets", "9",
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
    assert captured["grn_target_block_size"] == 11
    assert captured["grn_top_targets"] == 9
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
