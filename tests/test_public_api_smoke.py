from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd


def test_data_tfs_aliases_are_available():
    import rustscenic.data as data

    human = data.tfs("human")
    mouse = data.tfs("mm10")
    assert len(human) > 1000
    assert len(mouse) > 1000
    assert "SPI1" in human
    assert "Gata1" in mouse


def test_download_motif_rankings_uses_cache_without_real_network(tmp_path, monkeypatch):
    import rustscenic.data as data
    import urllib.request

    expected_name = "hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"

    def fake_urlretrieve(url, local_path):
        assert url.endswith(expected_name)
        pd.DataFrame(
            {
                "motif": ["m1", "m2"],
                "GENE1": [1, 2],
                "GENE2": [2, 1],
            }
        ).to_feather(local_path)
        return local_path, None

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

    out = data.download_motif_rankings(cache_dir=tmp_path, verbose=False)
    assert out.index.tolist() == ["m1", "m2"]
    assert out.columns.tolist() == ["GENE1", "GENE2"]
    assert (tmp_path / expected_name).exists()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("cache hit should not download again")

    monkeypatch.setattr(urllib.request, "urlretrieve", fail_if_called)
    cached = data.download_motif_rankings(cache_dir=tmp_path, verbose=False)
    pd.testing.assert_frame_equal(out, cached)


def test_pipeline_run_rna_only_smoke(tmp_path):
    import rustscenic.pipeline

    rng = np.random.default_rng(7)
    genes = ["SPI1", "PAX5", "TCF7"] + [f"GENE{i:03d}" for i in range(40)]
    X = rng.lognormal(mean=0.3, sigma=0.7, size=(80, len(genes))).astype("float32")
    rna = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(X.shape[0])]),
        var=pd.DataFrame(index=genes),
    )

    result = rustscenic.pipeline.run(
        rna,
        tmp_path,
        tfs=["SPI1", "PAX5", "TCF7"],
        grn_n_estimators=10,
        grn_top_targets=10,
        verbose=False,
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["n_cells"] == 80
    assert result.grn_path is not None and result.grn_path.exists()
    assert result.aucell_path is not None and result.aucell_path.exists()
    assert result.integrated_adata_path is not None and result.integrated_adata_path.exists()
    assert pd.read_parquet(result.grn_path).shape[0] > 0
    assert pd.read_parquet(result.aucell_path).shape == (80, 3)


def test_quickstart_synthetic_fallback_runs(monkeypatch, capsys):
    import rustscenic.quickstart as quickstart

    def offline(_scanpy, attempts=3):
        raise RuntimeError("forced offline")

    monkeypatch.setattr(quickstart, "_load_pbmc3k_with_retry", offline)
    assert quickstart.main() == 0
    out = capsys.readouterr().out
    assert "source=synthetic" in out
    assert "top-10 TF -> target edges" in out


def test_download_motif_rankings_accepts_aliases_and_mouse(tmp_path, monkeypatch):
    """Aliases (`human`/`mouse`/`hg38`/`mm10`) resolve to the canonical
    species, and mouse mm10 has a default filename mapped."""
    import rustscenic.data as data
    import urllib.request

    seen_urls = []

    def fake_urlretrieve(url, local_path):
        seen_urls.append(url)
        pd.DataFrame({"motif": ["m1"], "GENE1": [1]}).to_feather(local_path)
        return local_path, None

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

    # human alias
    out = data.download_motif_rankings(species="human", cache_dir=tmp_path / "h", verbose=False)
    assert any("hg38" in u for u in seen_urls)
    # mouse alias resolves to mm10
    seen_urls.clear()
    out = data.download_motif_rankings(species="mouse", cache_dir=tmp_path / "m", verbose=False)
    assert any("mm10" in u for u in seen_urls)
    # mm10 alias works too
    seen_urls.clear()
    out = data.download_motif_rankings(species="mm10", cache_dir=tmp_path / "mm10", verbose=False)
    assert any("mm10" in u for u in seen_urls)


def test_download_motif_rankings_filename_override(tmp_path, monkeypatch):
    """Explicit `filename=` bypasses the canonical-name lookup."""
    import rustscenic.data as data
    import urllib.request

    captured = {}

    def fake_urlretrieve(url, local_path):
        captured["url"] = url
        pd.DataFrame({"motif": ["m1"], "GENE1": [1]}).to_feather(local_path)
        return local_path, None

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

    data.download_motif_rankings(
        species="hs",
        filename="custom_2026_build.feather",
        cache_dir=tmp_path,
        verbose=False,
    )
    assert captured["url"].endswith("custom_2026_build.feather")


def test_motif_ranking_urls_resolve_live(monkeypatch):
    """Real-network smoke check that the URLs the data module builds
    actually exist on aertslab. Skipped by default (CI offline policy);
    enable locally with `RUSTSCENIC_LIVE_NETWORK=1 pytest`. This is the
    test that would have caught the v0.1.0 URL regression earlier — the
    prior `urlretrieve` mock made every URL look fine.
    """
    import os
    import urllib.request

    if not os.environ.get("RUSTSCENIC_LIVE_NETWORK"):
        import pytest
        pytest.skip("set RUSTSCENIC_LIVE_NETWORK=1 to enable")

    import rustscenic.data as data

    captured = {}

    def fake_urlretrieve(url, local_path):
        captured["url"] = url
        # Don't download — issue a HEAD instead and surface the status.
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=15) as resp:
            captured["status"] = resp.status
        # Write a tiny placeholder so the rest of the call doesn't try
        # to read an empty path.
        pd.DataFrame({"motif": ["m1"], "GENE1": [1]}).to_feather(local_path)
        return local_path, None

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

    for species in ("hs", "mm"):
        captured.clear()
        data.download_motif_rankings(
            species=species,
            cache_dir=os.path.join(os.environ.get("TMPDIR", "/tmp"), f"rustscenic_aertslab_{species}"),
            verbose=False,
        )
        assert captured["status"] == 200, (
            f"aertslab URL for species={species!r} returned "
            f"{captured.get('status')}: {captured['url']}"
        )
