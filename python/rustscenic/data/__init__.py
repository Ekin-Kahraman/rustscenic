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
    version: str = "v10nr_clust_public",
    region: str = "gene_based",
    score_type: str = "rankings",
    cache_dir: Optional[Path] = None,
    filename: Optional[str] = None,
    verbose: bool = True,
):
    """Download (and cache) an aertslab motif-ranking database.

    On first use this downloads a large feather file (hundreds of MB to
    tens of GB depending on the DB). Subsequent calls read from the local
    cache at ``cache_dir`` (default: ``~/.cache/rustscenic/cistarget/``).

    Parameters
    ----------
    species
        ``"hs"`` / ``"human"`` / ``"hg38"`` for human, or
        ``"mm"`` / ``"mouse"`` / ``"mm10"`` for mouse.
    genome
        Defaults to ``"hg38"`` for human, ``"mm10"`` for mouse. Override
        if you want a different aertslab DB build.
    version
        aertslab DB version string, e.g. ``"v10nr_clust_public"``.
    region
        ``"gene_based"`` (recommended for rustscenic.cistarget) or
        ``"region_based"``.
    score_type
        ``"rankings"`` for cistarget; ``"scores"`` for alternate analyses.
    cache_dir
        Override the default cache directory.
    filename
        Escape hatch: pass an aertslab feather filename directly to
        bypass the canonical-name lookup. Useful when aertslab releases
        a new build whose name we haven't mapped yet.

    Returns
    -------
    pandas.DataFrame
        Motif × gene ranking matrix suitable for ``rustscenic.cistarget.enrich``.

    Notes
    -----
    aertslab hosts rankings at ``resources.aertslab.org/cistarget/databases/``.
    See https://resources.aertslab.org/ for the authoritative file list.
    Mouse (mm10) ranking filenames are best-effort defaults — if a 404
    fires, pass ``filename=`` explicitly with the exact aertslab name.
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
        "hs": "hs", "human": "hs", "homo_sapiens": "hs", "hg38": "hs",
        "mm": "mm", "mouse": "mm", "mus_musculus": "mm", "mm10": "mm",
    }
    canonical_species = species_alias.get(str(species).lower())
    if canonical_species is None:
        raise ValueError(
            f"unknown species {species!r}. Use 'hs'/'human'/'hg38' for "
            f"human, 'mm'/'mouse'/'mm10' for mouse."
        )
    if genome is None:
        genome = "hg38" if canonical_species == "hs" else "mm10"

    # aertslab paths follow:
    #   <base>/<version>/<genome>/<species>_hgnc__<genome>__<region>__<score_type>.feather
    # e.g. v10nr_clust_public/hg38/hg38_refseq__10kb_up_and_down_tss.regions_vs_motifs.rankings.feather
    # The exact pattern varies by version — resolve at call time.
    default_name = {
        ("hs", "hg38", "v10nr_clust_public", "gene_based", "rankings"):
            "hg38_refseq_r80__10kb_up_and_down_tss.genes_vs_motifs.rankings.feather",
        ("hs", "hg38", "v10nr_clust_public", "region_based", "rankings"):
            "hg38_refseq_r80__10kb_up_and_down_tss.regions_vs_motifs.rankings.feather",
        # Mouse mm10 — naming follows the same aertslab convention. If
        # the URL 404s, pass `filename=` directly with the exact name
        # from https://resources.aertslab.org/cistarget/databases/.
        ("mm", "mm10", "v10nr_clust_public", "gene_based", "rankings"):
            "mm10_refseq-r80__10kb_up_and_down_tss.genes_vs_motifs.rankings.feather",
        ("mm", "mm10", "v10nr_clust_public", "region_based", "rankings"):
            "mm10_refseq-r80__10kb_up_and_down_tss.regions_vs_motifs.rankings.feather",
    }

    if filename is not None:
        fname = filename
    else:
        key = (canonical_species, genome, version, region, score_type)
        if key not in default_name:
            raise ValueError(
                f"no canonical filename mapped for {key!r}. See "
                f"https://resources.aertslab.org/cistarget/databases/ for "
                f"the authoritative list, then re-call with `filename=...`."
            )
        fname = default_name[key]

    url = f"{_AERTSLAB_RANKINGS_BASE}/{version}/{genome}/{fname}"
    local_path = cache_dir / fname

    if not local_path.exists():
        if verbose:
            print(f"downloading {fname} → {local_path}", flush=True)
        try:
            urllib.request.urlretrieve(url, local_path)
        except urllib.error.HTTPError as e:
            # Clean up the partial file so a retry with the right
            # filename doesn't get short-circuited by the cache check.
            if local_path.exists():
                local_path.unlink()
            raise RuntimeError(
                f"failed to download {url} ({e}). Check the canonical "
                f"name at https://resources.aertslab.org/cistarget/databases/ "
                f"and pass it via `filename=...`."
            ) from e

    return pd.read_feather(local_path).set_index(pd.read_feather(local_path).columns[0])


__all__ = ["tfs", "download_motif_rankings"]
