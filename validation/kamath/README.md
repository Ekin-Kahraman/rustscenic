# Kamath 2022 silent-zero fix validation

The exact scenario Fuaad reported (AUCell scoring all-zero on a
cellxgene-convention dataset), reproduced and validated on the
actual Kamath et al. 2022 dataset he was testing against.

## Run

```bash
# Download the OPC cells subset (106 MB, 13,691 cells)
curl -L -o validation/kamath/kamath_opc.h5ad \
  https://datasets.cellxgene.cziscience.com/f25a8375-1db5-49a0-9c85-b72dbe5e2a92.h5ad

python validation/kamath/validate_kamath_fix.py
```

## What it checks

1. **Convention** — verifies the loaded AnnData has ENSEMBL IDs in
   `var_names` and HGNC symbols in `var["feature_name"]` (the exact
   cellxgene / 10x convention that broke Fuaad).
2. **Auto-swap fires** — the ENSEMBL-detected warning from PR #18 is
   emitted, proving the fix ran (not that it silently matched by
   coincidence).
3. **AUCell non-zero** — output values > 0 across every regulon; the
   bug's symptom was all-zero output.
4. **Regulon coverage** — PR #18's `regulon_coverage` diagnostic
   round-trips via `auc.attrs` and reports the resolved fraction.
5. **GRN parity** — `rustscenic.grn.infer` also respects the
   cellxgene convention and recovers requested HGNC-symbol TFs.

## Result (2026-04-23)

```
✓ ENSEMBL var_names detected (5/5 sampled)
✓ Auto-swap warning fired (MALAT1, MT-RNR2, LINC00486 swapped from ENSG… IDs)
✓ AUCell output is non-zero across 5/5 regulons
✓ Mean AUC > 0 on 5/5 regulons (0.0081–0.0350)
✓ GRN recovered 3/3 requested TFs (MALAT1, MT-RNR2, LINC00486)
✓ GRN auto-swap warning fired
```

Bug fixed and validated on the exact dataset class Fuaad reported.

## Dataset provenance

- **Collection**: Kamath et al. 2022, "Single-cell genomic profiling of
  human dopamine neurons identifies a population that selectively
  degenerates in Parkinson's disease", *Nature Neuroscience*.
- **Collection ID**: `b0f0b447-ac37-45b0-b1bf-5c0b7d871120`
- **Subset used**: Human OPC cells (13,691 cells, 33,295 genes)
- **Dataset ID**: `4dd1cd23-fc4d-4fd1-9709-602540f3ca6f`
- **H5AD asset**: `f25a8375-1db5-49a0-9c85-b72dbe5e2a92`

Downloaded from the CellxGene Discover portal, which normalises all
datasets to the ENSEMBL-in-var_names / symbols-in-feature_name
convention — the exact class of data that broke before PR #18.
