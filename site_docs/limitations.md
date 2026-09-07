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
  is the stronger sparse scATAC option at larger topic counts.
- The million-cell benchmark measures RNA gene-network inference, not a complete
  spatial or RNA/chromatin workflow. Preparation memory is reported separately.
- Full workflow coverage from raw fragments plus external motif databases is in
  active validation.

## Positioning

RustScenic combines gene-regulation analysis stages in a CPU-based Python
package. Benchmarks show faster execution on the stated workloads; output
agreement varies by stage. Reproducibility requires the same input, version,
seed, thread count and settings.
