# Scope

RustScenic focuses on the regulatory-network compute path that benefits most
from a small Rust-backed Python package: matrix-level inference, per-cell
scoring, motif enrichment, topic modelling, enhancer-gene links and eRegulon
assembly.

## Designed For

- Local CPU runs on laptops and workstations.
- Python 3.10 to 3.13 environments.
- Researchers who want fewer moving parts than the legacy SCENIC stack.
- Benchmarked, reproducible workflows with commands and artefacts committed in
  the repository.

## Current Boundary

- Motif ranking databases are external inputs because public databases can be
  hundreds of megabytes to tens of gigabytes.
- GRN edge rankings are independently implemented and can differ from arboreto
  at fine grain because RustScenic uses an independent histogram-tree builder;
  the early-stop monitor and fitted-tree distribution are validated separately.
- Topic modelling ships both Online VB and collapsed Gibbs paths. The Gibbs path
  is the stronger sparse scATAC option at larger topic counts. Parallel AD-LDA
  is reproducible at a fixed thread count, but thread count is part of the model
  configuration because changing it can change the fitted posterior mode.
- Larger real multiome runs and repeated measurements on a second machine are
  the next benchmark tier.
- Full workflow coverage from raw fragments plus external motif databases is in
  active validation.

## Positioning

The strongest current message is direct: RustScenic gives a faster, deterministic
regulatory-network compute path with a much simpler install than the legacy
stack, and with measured head-to-head speedups on the tested real-data core E2E
rows.
