# Limitations

RustScenic is alpha research software. Use it deliberately and check outputs.

## Current Limitations

- Adoption is early. The project has external-user reports, but not broad community use.
- The GRN implementation is not bit-identical to arboreto. Fine-grained edge rankings can differ.
- Topic modelling has algorithmic tradeoffs. Collapsed Gibbs improves topic diversity, while Mallet remains a strong reference.
- Full SCENIC+ parity on region-ranking databases still needs more real-data comparison.
- Some validation claims are currently based on canonical TF recovery, not full biological replication.
- A formal methods paper or preprint does not exist yet.

## What Not To Claim

Do not claim:

- "RustScenic is universally better than pySCENIC."
- "RustScenic fully replaces every SCENIC+ analysis without caveats."
- "GRN edge rankings are identical to arboreto."
- "Community validation is the same as independent publication."

The accurate claim is narrower and stronger:

RustScenic is an installable, deterministic, CPU-focused implementation of the practical SCENIC compute path, with measured agreement on key downstream outputs and real-data validation evidence.
