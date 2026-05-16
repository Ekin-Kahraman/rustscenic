# Validation

RustScenic validation is designed to answer two questions:

1. Does the implementation agree with established SCENIC ecosystem outputs where a fair comparison is possible?
2. Does it remain usable on real atlas-scale single-cell data?

## Headline Results

| Test | Result |
| --- | --- |
| AUCell vs pySCENIC on Ziegler 2021 airway atlas | Mean per-cell Pearson `0.984`; `91.7%` cells above `0.95`. |
| Canonical airway TF benchmark | RustScenic and pySCENIC-unit both recover `8/14`; same miss set. |
| cisTarget AUC kernel vs `ctxcore.recovery.aucs` | Pearson `1.0000`; mean absolute difference about `2.4e-5`. |
| Real multiome pipeline runs | PBMC 3k, mouse brain E18 5k, PBMC granulocyte 10k. |
| Local unit/integration suite | 197 tests passed, 1 skipped in the 2026-05-15 portfolio audit. |

## Community Reports

Two external-user validation reports are currently surfaced in the README:

| Reporter | Dataset | Signal |
| --- | --- | --- |
| `@Skycr` | Kamath dopaminergic neurons | 266,805 GRN edges, 9 regulons, 9 of 9 expected DA-neuron TFs recovered. |
| `@lmVl12` | 10x human brain multiome | 4,293,902 GRN edges, 1,748 regulons, non-empty AUCell and topic outputs. |

These reports are adoption evidence, not a substitute for a fully controlled benchmark paper. Treat them as directional until commands, environment and artefacts are fully reproduced by maintainers.

## Known Validation Caveats

- GRN edge rankings do not exactly match arboreto at fine grain; downstream cell-level AUCell agreement is stronger than edge-level agreement.
- Topic modelling is not always a speed win. Mallet remains a strong reference for topic diversity and coherence.
- Some real-data biological checks are name-presence checks, not full cell-type enrichment validations.
- The six-dataset parity sweep is a planned v0.5+ credibility gate.

## Where To Look

- `site_docs/benchmarks.md`
- `validation/VALIDATION_SUMMARY.md`
- `validation/ziegler_headtohead_2026-04-19.md`
- `validation/community/`
- `validation/scaling/`
- `docs/v0.4.x-benchmark-plan.md`
