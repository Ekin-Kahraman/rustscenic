"""Bundled reference data and download helpers.

Everything in this module eliminates a step between ``pip install
rustscenic`` and running the full pipeline. TF lists ship with the
wheel (small text files). Motif-ranking databases are fetched on
first use with local caching.

Public API:

    rustscenic.data.tfs(species="hs") -> list[str]
        Bundled TF names. "hs" = human (1,839 TFs, HGNC), "mm" = mouse
        (1,721 TFs, MGI). Lists are from aertslab/pySCENIC resources/.

    rustscenic.data.download_motif_rankings(species, genome, version, cache_dir=None)
        Fetch + cache an aertslab motif ranking database (feather). Returns
        a pandas DataFrame with motifs as index, genes as columns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

_DATA_DIR = Path(__file__).parent


def tfs(species: Literal["hs", "mm"] = "hs") -> list[str]:
    """Return the bundled transcription-factor list for ``species``.

    Parameters
    ----------
    species
        ``"hs"`` for human (HGNC, 1,839 TFs, hg38) or ``"mm"`` for mouse
        (MGI, 1,721 TFs, mm10). Sourced from ``aertslab/pySCENIC`` resources.

    Returns
    -------
    Plain Python list of TF gene symbols, suitable to pass directly as
    the ``tf_names`` argument to ``rustscenic.grn.infer``.
    """
    filename = {"hs": "allTFs_hg38.txt", "mm": "allTFs_mm.txt"}.get(species)
    if filename is None:
        raise ValueError(f"unknown species {species!r} — use 'hs' or 'mm'")
    path = _DATA_DIR / filename
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


_AERTSLAB_RANKINGS_BASE = "https://resources.aertslab.org/cistarget/databases"


def download_motif_rankings(
    species: Literal["hs", "mm"] = "hs",
    genome: str = "hg38",
    version: str = "v10nr_clust_public",
    region: str = "gene_based",
    score_type: str = "rankings",
    cache_dir: Optional[Path] = None,
    verbose: bool = True,
):
    """Download (and cache) an aertslab motif-ranking database.

    On first use this downloads a large feather file (hundreds of MB to
    tens of GB depending on the DB). Subsequent calls read from the local
    cache at ``cache_dir`` (default: ``~/.cache/rustscenic/cistarget/``).

    Parameters
    ----------
    species
        ``"hs"`` or ``"mm"``.
    genome
        e.g. ``"hg38"`` (with species="hs") or ``"mm10"`` (with species="mm").
    version
        aertslab DB version string, e.g. ``"v10nr_clust_public"``.
    region
        ``"gene_based"`` (recommended for rustscenic.cistarget) or
        ``"region_based"``.
    score_type
        ``"rankings"`` for cistarget; ``"scores"`` for alternate analyses.
    cache_dir
        Override the default cache directory.

    Returns
    -------
    pandas.DataFrame
        Motif × gene ranking matrix suitable for ``rustscenic.cistarget.enrich``.

    Notes
    -----
    aertslab hosts rankings at ``resources.aertslab.org/cistarget/databases/``.
    See https://resources.aertslab.org/ for the authoritative file list.
    """
    import pandas as pd
    import urllib.request

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "rustscenic" / "cistarget"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # aertslab paths follow:
    #   <base>/<version>/<genome>/<species>_hgnc__<genome>__<region>__<score_type>.feather
    # e.g. v10nr_clust_public/hg38/hg38_refseq__10kb_up_and_down_tss.regions_vs_motifs.rankings.feather
    # The exact pattern varies by version — resolve at call time.
    # We do the simplest default: documented filename for v10nr_clust_public hg38 gene-based rankings.
    default_name = {
        ("hs", "hg38", "v10nr_clust_public", "gene_based", "rankings"):
            "hg38_refseq_r80__10kb_up_and_down_tss.genes_vs_motifs.rankings.feather",
        ("hs", "hg38", "v10nr_clust_public", "region_based", "rankings"):
            "hg38_refseq_r80__10kb_up_and_down_tss.regions_vs_motifs.rankings.feather",
    }
    key = (species, genome, version, region, score_type)
    if key not in default_name:
        raise ValueError(
            f"no canonical filename mapped for {key!r}. See "
            f"https://resources.aertslab.org/cistarget/databases/ for the "
            f"authoritative list and pass the URL by calling "
            f"`urllib.request.urlretrieve(...)` directly."
        )

    fname = default_name[key]
    url = f"{_AERTSLAB_RANKINGS_BASE}/{version}/{genome}/{fname}"
    local_path = cache_dir / fname

    if not local_path.exists():
        if verbose:
            print(f"downloading {fname} → {local_path}", flush=True)
        urllib.request.urlretrieve(url, local_path)

    return pd.read_feather(local_path).set_index(pd.read_feather(local_path).columns[0])


__all__ = ["tfs", "download_motif_rankings"]
