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
    # Accept the common long-form aliases the codebase suggests elsewhere
    # so a user following the species-hint diagnostic doesn't hit a new error.
    alias = {
        "hs": "hs", "human": "hs", "homo_sapiens": "hs", "hg38": "hs",
        "mm": "mm", "mouse": "mm", "mus_musculus": "mm", "mm10": "mm",
    }
    canonical = alias.get(str(species).lower())
    if canonical is None:
        raise ValueError(
            f"unknown species {species!r} — use 'hs' / 'human' / 'hg38' "
            f"for human, 'mm' / 'mouse' / 'mm10' for mouse"
        )
    filename = {"hs": "allTFs_hg38.txt", "mm": "allTFs_mm.txt"}[canonical]
    path = _DATA_DIR / filename
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


_AERTSLAB_RANKINGS_BASE = "https://resources.aertslab.org/cistarget/databases"


def download_motif_rankings(
    species: Literal["hs", "mm", "human", "mouse", "hg38", "mm10"] = "hs",
    genome: Optional[str] = None,
    motif_collection: str = "mc_v10_clust",
    refseq_release: str = "refseq_r80",
    region: str = "gene_based",
    window: str = "10kbp_up_10kbp_down",
    score_type: str = "rankings",
    cache_dir: Optional[Path] = None,
    filename: Optional[str] = None,
    url: Optional[str] = None,
    verbose: bool = True,
):
    """Download (and cache) an aertslab motif-ranking database.

    On first use this downloads a large feather file (hundreds of MB to
    tens of GB depending on the DB). Subsequent calls read from the local
    cache at ``cache_dir`` (default: ``~/.cache/rustscenic/cistarget/``).

    Resolves to URLs of the form::

        https://resources.aertslab.org/cistarget/databases/
            <species_dir>/<genome>/<refseq_release>/<motif_collection>/<region>/
            <genome>_<window>_full_tx_<motif_collection_short>.
            <region>s_vs_motifs.<score_type>.feather

    e.g. ``homo_sapiens/hg38/refseq_r80/mc_v10_clust/gene_based/
    hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather``.

    Parameters
    ----------
    species
        ``"hs"`` / ``"human"`` / ``"hg38"`` for human, or
        ``"mm"`` / ``"mouse"`` / ``"mm10"`` for mouse.
    genome
        Defaults to ``"hg38"`` for human, ``"mm10"`` for mouse.
    motif_collection
        aertslab motif collection slug. Default ``"mc_v10_clust"``.
    refseq_release
        RefSeq release directory. Default ``"refseq_r80"``.
    region
        ``"gene_based"`` (recommended for rustscenic.cistarget) or
        ``"region_based"``.
    window
        Region window slug. ``"10kbp_up_10kbp_down"`` (default, broad
        20kb total) or ``"500bp_up_100bp_down"`` (promoter-only).
    score_type
        ``"rankings"`` for cistarget; ``"scores"`` for alternate analyses.
    cache_dir
        Override the default cache directory.
    filename
        Escape hatch — pass an aertslab feather filename directly to
        bypass the auto-built name. Combined with the auto-built dir.
    url
        Full URL escape hatch — bypasses both name and dir construction.

    Returns
    -------
    pandas.DataFrame
        Motif × gene ranking matrix suitable for ``rustscenic.cistarget.enrich``.

    Notes
    -----
    aertslab hosts rankings at ``resources.aertslab.org/cistarget/databases/``.
    Browse https://resources.aertslab.org/cistarget/databases/ for the
    authoritative directory.
    """
    import pandas as pd
    import urllib.request

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "rustscenic" / "cistarget"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Normalise the species alias the same way `tfs()` does so users
    # following the diagnostic hints from `_gene_resolution` don't hit
    # an inconsistent error here.
    species_alias = {
        "hs": ("hs", "homo_sapiens"),
        "human": ("hs", "homo_sapiens"),
        "homo_sapiens": ("hs", "homo_sapiens"),
        "hg38": ("hs", "homo_sapiens"),
        "mm": ("mm", "mus_musculus"),
        "mouse": ("mm", "mus_musculus"),
        "mus_musculus": ("mm", "mus_musculus"),
        "mm10": ("mm", "mus_musculus"),
    }
    norm = species_alias.get(str(species).lower())
    if norm is None:
        raise ValueError(
            f"unknown species {species!r}. Use 'hs'/'human'/'hg38' for "
            f"human, 'mm'/'mouse'/'mm10' for mouse."
        )
    canonical_species, species_dir = norm
    if genome is None:
        genome = "hg38" if canonical_species == "hs" else "mm10"

    # aertslab filename template, derived from a directory listing of
    # https://resources.aertslab.org/cistarget/databases/{species_dir}/
    #   {genome}/refseq_r80/mc_v10_clust/gene_based/
    # The motif-collection short slug ("v10_clust") is the trailing
    # piece of the collection ("mc_v10_clust") after stripping the "mc_"
    # prefix. region_token is "genes" or "regions" (singular→plural).
    if filename is None:
        mc_short = motif_collection.split("_", 1)[1] if motif_collection.startswith("mc_") else motif_collection
        region_token = {"gene_based": "genes", "region_based": "regions"}.get(region, region)
        filename = (
            f"{genome}_{window}_full_tx_{mc_short}."
            f"{region_token}_vs_motifs.{score_type}.feather"
        )

    if url is None:
        url = (
            f"{_AERTSLAB_RANKINGS_BASE}/{species_dir}/{genome}/"
            f"{refseq_release}/{motif_collection}/{region}/{filename}"
        )

    local_path = cache_dir / filename

    if not local_path.exists():
        if verbose:
            print(f"downloading {filename} → {local_path}", flush=True)
        try:
            urllib.request.urlretrieve(url, local_path)
        except urllib.error.HTTPError as e:
            # Clean up the partial file so a retry with the right
            # filename doesn't get short-circuited by the cache check.
            if local_path.exists():
                local_path.unlink()
            raise RuntimeError(
                f"failed to download {url} ({e}). Browse "
                f"https://resources.aertslab.org/cistarget/databases/"
                f"{species_dir}/{genome}/ for the directory and pass the "
                f"exact `filename=` (or full `url=`) you find there."
            ) from e

    return pd.read_feather(local_path).set_index(pd.read_feather(local_path).columns[0])


__all__ = ["tfs", "download_motif_rankings"]
