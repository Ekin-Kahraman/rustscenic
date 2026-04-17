Subject: rustscenic v0.1 — following your steer, Moha's stack

Hi Kuan,

Short update on the Rust-SCENIC+ direction we talked about.

Profiling (Apr 16) showed what you and Moha flagged is real: the adaptive-QC / Yates angle is a seconds-level problem nobody's bottlenecked on, but the pyscenic stack itself doesn't install at all on modern Python — arboreto.grnboost2 crashes in Dask's `from_delayed` on any 2024+ dask, and pyscenic's ctxcore imports `pkg_resources` which setuptools removed in November. On a fresh Python 3.12 + numpy 2 + pandas 3 venv, neither arboreto nor pyscenic runs.

rustscenic v0.1 is a Rust+PyO3 replacement for `arboreto.grnboost2` — same output schema, drop-in for the rest of the SCENIC+ pipeline (pycisTarget, AUCell, etc. remain on pyscenic). `pip install` on modern Python works clean, 5 MB wheel, 4 deps total.

Validated on PBMC-3k and PBMC-10k:
  - Biological hit rate on 43 literature-curated edges: 74% vs arboreto 51% (random baseline 0.15%).
  - External CollecTRI recall (independent 17,798 edges via decoupler-py): 2× arboreto at top-10, 1.3× at top-100.
  - All 8 canonical PBMC lineages discriminate correctly via downstream pyscenic.aucell on our output (PAX5 B-cell regulon: 15.8× fold; TBX21 NK: 9.5×).
  - Null shuffle test: importance collapses to 3% of real.
  - Seed stability: 92% top-10 TF overlap across 3 seeds.
  - Speed: ~2× faster than arboreto on PBMC-3k (207s vs 393s, single machine).

Repo is private while I finish the remaining stages — AUCell, pycisTopic LDA, and pycisTarget are scoped as v0.2–v0.4. Everything in v0.1 is committed and reproducible from `validation/*.py`.

Happy to jump on a call if any of this is useful to the lab. Also happy to open the repo up whenever you'd like to take a look — just wanted to get the first stage solid before making noise.

Thanks for pointing me at this.

Best,
Ekin
