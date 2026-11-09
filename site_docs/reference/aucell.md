# AUCell

Per-cell regulon activity scores.

For comparisons with pySCENIC, use the same gene axis and an explicit scoring
seed in both tools:

```python
activity = rustscenic.aucell.score(expression, regulons, seed=777)
```

The seed controls the ordering of genes with equal expression, including sparse
zeros. It does not shuffle or densify the expression matrix. Omitting it retains
the previous deterministic gene-order behaviour. Seeded tie agreement alone is
not a biological-accuracy claim; match gene coverage, signatures and thresholds
as well. Native scores retain float32 precision.

::: rustscenic.aucell

## Reexported helpers

These names are also available from this module. They share the underlying
implementation and source documentation of their original definitions.

::: rustscenic.aucell.dedupe_backend_symbol_for_matrix
    options:
      heading_level: 3

::: rustscenic.aucell.dedupe_by_symbol
    options:
      heading_level: 3

::: rustscenic.aucell.duplicate_gene_summary
    options:
      heading_level: 3

::: rustscenic.aucell.warn_if_max_likely_unnormalized
    options:
      heading_level: 3

::: rustscenic.aucell.warn_if_poor_coverage
    options:
      heading_level: 3

::: rustscenic.aucell.as_float32_array
    options:
      heading_level: 3

::: rustscenic.aucell.as_float32_contiguous
    options:
      heading_level: 3

::: rustscenic.aucell.prepare_regulon_indices_with_coverage
    options:
      heading_level: 3

::: rustscenic.aucell.string_list
    options:
      heading_level: 3
