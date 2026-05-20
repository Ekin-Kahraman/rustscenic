# Scope

RustScenic focuses on the SCENIC-style compute path that benefits most from a
small Rust-backed Python package: matrix-level regulatory-network inference,
per-cell scoring, motif enrichment, topic modelling, enhancer-gene links and
eRegulon assembly.

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
  at fine grain; downstream cell-level agreement is stronger.
- Topic modelling ships both Online VB and collapsed Gibbs paths. The Gibbs path
  is the stronger sparse scATAC option at larger topic counts.
- Larger real multiome runs and repeated measurements on a second machine are
  the next benchmark tier.
- Full workflow coverage from raw fragments plus external motif databases is in
  active validation.

## Positioning

The strongest current message is direct: RustScenic gives a fast, deterministic
SCENIC-style compute path with a much simpler install than the legacy stack, and
with measured head-to-head speedups on the tested real-data core E2E rows.
