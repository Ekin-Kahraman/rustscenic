# Fidelity gap at scale: diagnosis (post hoc, 19 November 2026)

Everything here was done after the Minerva results were seen. None of it changes
the protocol's verdict, which stands as returned: fidelity was not met for bone
marrow under either rule (0.763, 0.742) or for the 20,000-cell atlas subset
under the current rule (0.837); it was met for the atlas subset under the original rule
(0.911).

## What was found

1. **The gap is on the target side only.** Measured from the regulator side
   (top 10 TFs per target), RustScenic agrees with pySCENIC at 0.904 to 0.927
   of pySCENIC's seed-to-seed agreement in all four cases
   (`agreement-by-side.csv`).
2. **It is not the stopping rule.** Per-target tree counts agree between tools
   as well as within each tool (Spearman 0.97 under the original rule), and
   dividing importances by tree count does not close the gap
   (`tree-count-effect.csv`).
3. **It is confined to TFs that RustScenic compresses.** RustScenic keeps at
   most 255 bins per TF. For the 58 bone-marrow TFs with at most 255 distinct
   values (binned exactly) the ratio is 1.063 (current) and 0.918 (original);
   for the 1,011 compressed TFs it is 0.746 and 0.732
   (`per-tf-binning-bmmc.csv`).
4. **Binning is the cause.** arboreto 0.1.6 on scikit-learn 1.3.2, fitted on
   RustScenic's own bin indices (exact port of `histogram.rs`) for 1,000
   random bone-marrow targets and three seeds, diverges from arboreto on raw
   values as much as RustScenic does (0.902 vs 0.892, different-seed pairs
   only), and RustScenic agrees with binned arboreto at 0.987
   (`binning-causal-test.csv`). Local arboreto reproduced the Minerva
   networks to the bit.
5. **Exact bins restore fidelity.** A prototype with 16-bit bins
   (a local working copy of the repository, branch `exp/finer-bins`, cut-point
   budget from `RUSTSCENIC_MAX_BINS`, default 255) reproduces the Minerva
   RustScenic network bit for bit at 255. With the budget at 65,535 (exact for
   every bone-marrow TF), bone-marrow fidelity under the current rule is
   0.966, scored against the returned pySCENIC networks with the protocol's
   metric. Edges per network: 426k (pySCENIC 426k; 255 bins gave 448k).
   Under the original rule it is 0.985 (336 trees per target on average;
   255 bins gave 0.742). The scorer reproduces the returned protocol values
   (0.763, 0.837, 0.911) when pointed at the returned RustScenic networks.

6. **The atlas shows the same split.** In the 20,000-cell atlas subset, 66% of
   the 256 TFs have more than 255 distinct values (median 500, maximum
   6,798; counted on the rebuilt input below). Scoring the returned networks,
   fidelity for TFs binned exactly is 0.948 (current) and 0.974 (original);
   for compressed TFs it is 0.772 and 0.879
   (`unique-values-atlas20000.csv`).
7. **Why BEELINE passed (inferred).** The four BEELINE confirmation datasets
   have 425 to 1,071 cells. Most of their genes also exceed 255 distinct
   values (59% to 98%), but each bin then holds only a few cells, so little
   is lost. At 20,000 cells each bin holds about 80.

## Atlas 20,000 rebuilt locally

`rebuild_atlas.py` rebuilds the first 500,000 cells of the Minerva atlas input
from the local 10x source (`mouse-1m.h5`, sha256 matches the archived source;
panel and `tfs.txt` hashes match) without loading all 1.3M cells. The 20,000-cell
subset has the same cell and gene lists as the Minerva run (`cells_sha256`,
`genes_sha256`) and reproduces Minerva's Pearson baseline for all 767,744
TF-target pairs to 3e-8. It is not bit-identical: the local 255-bin RustScenic
network for seed 777 shares 0.843 of top-50 targets with Minerva's (different
seeds share 0.39), consistent with last-bit differences in the normalised
values between platforms (inferred, not shown). Atlas exact-bin results are
therefore compared with local 255-bin fits on the same rebuilt input as well as
with the returned networks.

## Results with exact bins (queue.sh, local, 19 November)

Fidelity, the protocol's metric against the returned pySCENIC networks
(`fidelity-queue.txt`; pass mark 0.90):

| Setting | 255 bins (returned) | 255 bins (local) | Exact bins (local) |
|---|---|---|---|
| Bone marrow, current rule | 0.763 | 0.763 (bit-identical) | 0.966 |
| Bone marrow, original rule | 0.742 | | 0.985 |
| Atlas 20,000, current rule | 0.837 | 0.836 | 0.949 |
| Atlas 20,000, original rule | 0.911 | 0.913 | 0.985 |

The local 255-bin fits on the rebuilt atlas input reproduce the returned
values, so the rebuild does not move the metric. With exact bins the
compressed TFs reach the exactly binned ones (atlas 0.949 vs 0.948 current,
0.987 vs 0.982 original). On bone marrow a 1,023-bin budget gives 0.921 and
4,095 gives 0.967, the same as exact.

Cost on this laptop (10 cores, shared; seed 777, current rule;
`queue-summary.csv`). "Released" is the v0.6.1 kernel with 8-bit bins
(`queue-release.sh`; its bone-marrow network is bit-identical to Minerva's);
the prototype stores 16-bit bins at every budget.

| Input | Released, 255 bins | Prototype, 255 bins | Prototype, exact bins |
|---|---|---|---|
| Bone marrow 20,000 cells, 1,069 TFs | 273 s, 0.65 GB | 280 s, 0.61 GB | 311 s, 0.66 GB |
| Atlas 100,000 cells, 256 TFs (two runs each) | 234 / 244 s, 0.57 GB | 245 / 255 s, 0.6 GB | 251 / 257 s, 0.6 GB |
| Atlas 500,000 cells | 1,654 s, 2.03 GB | 1,642 s, 2.06 GB | 1,596 s, 2.32 GB |

Exact bins against the released kernel: +14% time on bone marrow, about +6% at
100,000 atlas cells, none measurable at 500,000 cells (-3.5%), and +0.29 GB
peak memory at 500,000 cells. Repeat runs differed by up to 4%. Bone-marrow
1,023 and 4,095-bin fits took 305 to 328 s, no faster than exact.

## Implementation (19 November)

In a local working copy of the repository (branch `exp/finer-bins`, uncommitted)
the environment switch is replaced by a `max_bins` setting: Rust `GrnConfig`,
Python `grn.infer(max_bins=...)`, `grn --max-bins`, pipeline `grn_max_bins`.
The default is 65,535 (exact for every TF here); 2 to 65,535 accepted. Rust
tests and clippy pass; Python suite 769 passed, 1 skipped. `verify_default.py`
(`verify-default.json`): the default reproduces the exact-bin prototype
networks and `max_bins=255` reproduces the 255-bin networks, bit for bit, on
atlas 20,000 and bone marrow (the latter against Minerva's returned network).

## Not measured here

- The cost of exact bins on Linux (Minerva); unique values per TF grow with cells.
  The 1.3M-cell cost on a laptop is in the release-candidate timings below.

## Scripts

`per_target_agreement.py`, `tree_count_effect.py`, `per_tf_binning.py`,
`binning_experiment.py`, `compare_binning.py`, `fit_native_bins.py`,
`score_fidelity.py`, `rebuild_atlas.py`, `queue.sh`, `queue-release.sh`,
`summarise_queue.py`. Run with `python3 -I`; refits use
the Python of a local virtual environment with the `exp/finer-bins` build
installed (see [`../REPRODUCE.md`](../REPRODUCE.md)).

## Release-candidate timings (19 November, local, post hoc)

`final-timing.sh` → `final-timings.tsv`, `native-bins/rc070-*` and `release-u8-bmmc-arboreto-777-{b,c}`.
Builds: a v0.7.0 release-candidate wheel built from the working tree at 21:48 (the published
wheel adds only a buffer-size assert and the `max_bins >= 3` check made after it), against a
local v0.6.1 build with the uncommitted paired_oob patch (a local virtual environment,
255 bins), current rule, seed 777, 10-core laptop, sequential.

| Input | v0.7.0 RC (exact bins) | v0.6.1 build (255 bins) | Ratio |
| --- | --- | --- | --- |
| Bone marrow 20,000 cells, 1,069 TFs | 285.5, 284.6 s; 0.66-0.67 GB | 272.0, 271.7 s; 0.66-0.67 GB | +4.9% (pairwise 4.7-5.1%) |
| Atlas 1,306,127 cells, 256 TFs | 5,494.5 s; 4.71 GB | 5,065.1 s; 4.65 GB | +8.5% time, +1.3% memory |

The RC bone-marrow networks equal `cap65535-arboreto-777` bit for bit. An independent audit (20 Nov)
re-ran the published PyPI wheels: v0.7.0 default and `max_bins=255`, and v0.6.1, reproduced the
reference and HPC-returned bone-marrow networks and both atlas 20,000 networks bit for bit.
