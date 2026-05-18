from __future__ import annotations

import argparse

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
