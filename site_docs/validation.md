# Validation

RustScenic validation tracks implementation agreement, runtime, memory and
real-data usability against established SCENIC ecosystem outputs where a fair
comparison is possible.

The standard is publication-minded: every serious claim should point to a
dataset, command, version, hardware context, runtime, memory measurement and
output sanity check.

## Headline Results

| Test | Result |
| --- | --- |
| AUCell vs pySCENIC on Ziegler 2021 airway atlas | Mean per-cell Pearson `0.984`; `91.7%` cells above `0.95`. |
| Canonical airway TF benchmark | RustScenic and pySCENIC-unit both recover `8/14`; same miss set. |
| cisTarget AUC kernel vs `ctxcore.recovery.aucs` | Pearson `1.0000`; mean absolute difference about `2.4e-5`. |
| Human brain GEM-X SCENIC+ comparison | Region-to-gene Jaccard `1.000`; region AUCell mean Pearson `0.823`. |
| Real multiome pipeline runs | PBMC 3k, mouse brain E18 5k, PBMC granulocyte 10k. |
| Local unit/integration suite | 197 tests passed, 1 skipped in the 2026-05-15 portfolio audit. |

## Community Reports

| Reporter | Dataset | Signal |
| --- | --- | --- |
| `@Skycr` | Kamath dopaminergic neurons | 266,805 GRN edges, 9 regulons, 9 of 9 expected DA-neuron TFs recovered. |
| `@lmVl12` | 10x human brain multiome | 4,293,902 GRN edges, 1,748 regulons, non-empty AUCell and topic outputs. |

These reports show the package running outside the maintainer benchmark path.
They complement the controlled head-to-head scripts and saved validation
artefacts in `validation/`.

## Validation Notes

- GRN edge rankings are not expected to be bit-identical to arboreto because the
  implementation uses an independent histogram-GBM path.
- Downstream cell-level AUCell agreement is stronger than fine-grained GRN edge
  agreement.
- Some real-data biological checks currently use expected TF recovery by name;
  cell-type enrichment checks are part of the next validation tier.
- The next benchmark tier adds more real multiome datasets, repeated runs and a
  second machine.

## Where To Look

- `site_docs/benchmarks.md`
- `validation/VALIDATION_SUMMARY.md`
- `validation/ziegler_headtohead_2026-04-19.md`
- `validation/community/`
- `validation/scaling/`
- `docs/v0.4.x-benchmark-plan.md`
