# Python API reference

This reference lists the Python functions, classes, methods and attributes in
RustScenic, including their source signatures, parameter defaults and available
docstrings. Start with the [API map](../api.md) to choose a stage or the
[quickstart](../quickstart.md) to prepare a workflow.

| Module | Contents |
| --- | --- |
| [`rustscenic.grn`](grn.md) | Gene-network inference, correlation, regulon construction and TF-list loading. |
| [`rustscenic.aucell`](aucell.md) | Per-cell regulon activity scores. |
| [`rustscenic.cistarget`](cistarget.md) | Motif enrichment, annotation-based pruning and ranking-file loading. |
| [`rustscenic.topics`](topics.md) | Topic fitting, coherence, assignment summaries and result containers. |
| [`rustscenic.preproc`](preproc.md) | Fragment matrices, peak calling and the `qc` namespace of quality-control functions. |
| [`rustscenic.enhancer`](enhancer.md) | Peak-to-gene links from RNA, ATAC and gene coordinates. |
| [`rustscenic.eregulon`](eregulon.md) | eRegulon objects, dataframe assembly and conversion helpers. |
| [`rustscenic.specificity`](specificity.md) | Regulon specificity scores and candidate enhancers per topic. |
| [`rustscenic.pipeline`](pipeline.md) | The staged orchestrator and its result manifest. |
| [`rustscenic.data`](data.md) | Bundled TF lists and cached motif-ranking and gene-coordinate resources. |
| [`rustscenic.backend`](backend.md) | Installed-backend checks and the required Rust symbol groups. |
| [`rustscenic.cli`](cli.md) | Python entry points behind the `rustscenic` command. Handlers receive the `argparse.Namespace` assembled by `main`; use the command line for normal interactive use. |
| [`rustscenic.quickstart`](quickstart.md) | Entry point for `python -m rustscenic.quickstart`. |

## Reading the reference

Each module page includes its own public definitions and first-party helpers
reexported under public names. Private names beginning with `_` and imported
third-party APIs are excluded. The explicitly implemented `ERegulon.__len__`
protocol is included. Reexports describe the same underlying implementation;
they do not introduce additional analysis stages.

Signatures include all parameters, their available type annotations and
defaults. Dataclass constructor signatures include their fields; `list()` and
`dict()` represent a fresh value from a default factory.
The array and dictionary result subclasses inherit their base constructors.

The reference is collected statically from this source tree. It does not import
RustScenic or load its compiled extension during the documentation build.
Other documentation projects can use the generated [Sphinx-compatible object
inventory](https://ekin-kahraman.github.io/rustscenic/objects.inv) to cross-reference these APIs.

## Package exports

Analysis modules are documented on the pages above. The package also reexports
[`backend_capabilities`][rustscenic.backend_capabilities] and the installed
extension's `__version__` value.

::: rustscenic
    options:
      filters: public
