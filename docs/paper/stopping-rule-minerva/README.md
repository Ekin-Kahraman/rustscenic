# Stopping-rule replication at 20,000 cells and the binning fix

This folder holds the compact evidence for the manuscript sections "A pre-specified
test at 20,000 cells met 8 of 12 checks, and the original rule's cost grew with cell
number" and "The fidelity shortfall traces to 255-bin quantisation and is corrected by
exact binning in post hoc refits" (Figure 3).

## What is confirmatory and what is post hoc

- **Confirmatory.** The protocol ([`../stopping-rule/minerva/PROTOCOL.md`](../stopping-rule/minerva/PROTOCOL.md),
  SHA-256 `cd35e196ac8a52df6ab57bfdc00d7727d6a40053e6a46b2398e35edabccea55b`)
  was fixed before the package ran. A collaborator executed it on 16-core HPC
  nodes, 13 to 18 November 2026. The returned archive matched its SHA-256
  (`audit/SEND_BACK.sha256`). `audit/analyse_return.py` recomputed every check
  independently of the package's own summary; values agree to about 2e-15.
  Verdict: 8 of 12 checks met. Fidelity was not met in three of four tests, and
  pySCENIC's stability ratio in the atlas subset was 1.48 against 1.5.
- **Post hoc.** Everything in `diagnosis/` was done after the results were seen:
  the cause (255-bin TF quantisation), the exact-binning refits and their cost.
  None of it changes the verdict above.

## Files

| File | Content |
| --- | --- |
| `audit/checks-recomputed.csv` | The twelve checks plus descriptive scale rows |
| `audit/recovery-differences-recomputed.csv` | Average-precision differences with 2,000-draw TF-bootstrap intervals |
| `audit/recovery-points-recomputed.csv` | Average precision per arm and seed |
| `audit/scale-recomputed.csv` | Trees, time and peak memory per arm up to 1,306,127 cells |
| `audit/AUDIT.json` | Archive and hash audit |
| `audit/stopping-rule-at-scale.png` | Descriptive figure of the returned results |
| `diagnosis/DIAGNOSIS.md` | Diagnosis narrative and all post hoc results |
| `diagnosis/agreement-by-side.csv` | Regulator-side versus target-side agreement |
| `diagnosis/tree-count-effect.csv` | Stopping-rule tree counts between and within tools |
| `diagnosis/per-tf-binning-bmmc.csv`, `unique-values-atlas20000.csv` | Distinct values per TF |
| `diagnosis/binning-causal-test.csv` | arboreto on RustScenic's bin indices, 1,000 BMMC targets |
| `diagnosis/fidelity-bmmc-exact.txt` | Protocol fidelity for the bone-marrow exact-bin refits shown in Figure 3 |
| `diagnosis/fidelity-queue.txt` | Protocol fidelity for local refits (255, 1,023, 4,095 and exact bins) |
| `diagnosis/queue-summary.csv`, `queue-fits.tsv` | Time and peak memory of local refits |
| `diagnosis/verify-default.json` | Bit-for-bit checks of the finished `max_bins` implementation |
| `diagnosis/published-wheel-check/` | Bit-for-bit checks of the published 0.7.0 and 0.6.1 wheels |

## Reproducing the post hoc analyses

The scripts expect the returned archive extracted next to them and the
replication package (its code is in [`package/`](package/); inputs and layout
are in [REPRODUCE.md](REPRODUCE.md)). Local refits use a build of the
`exp/finer-bins` branch; `fit_native_bins.py` and `verify_default.py` read the
BMMC inputs from the package and the atlas subset rebuilt by
`rebuild_atlas.py` from the public 10x file. Run Python with `-I`.

The full returned archive (networks for every arm and seed) and the local
refit networks are not in this repository.
