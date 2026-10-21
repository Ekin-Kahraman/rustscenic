# Explore an endothelial-cell regulatory network

Download [starter.zip](https://raw.githubusercontent.com/Ekin-Kahraman/rustscenic/main/examples/cellxgene-endothelial/starter.zip) and extract it first. The instructions below apply inside the extracted folder.

**No installation or HPC needed.** Open `explore.html` in a browser to choose
a regulator and inspect its predicted targets.

Or use Python 3.10+:

```bash
python3 example.py --verify
python3 example.py --tf ERG
python3 example.py --list
```

The default example shows **ETS1**. The browser and Python examples use the
same saved results; neither downloads data or fits a model.

## What is included

- One complete endothelial-cell cohort (`CL:0000115`): **5,000 sampled cells**,
  **234 contributing datasets**, **184,000 retained edges**, **85 motif-supported regulons**.
- Original network, motif evidence, activity summaries and metadata in `data/`.
- A convenient JSON view of all 85 motif-supported regulons in `preview.json`.
- Source-study citations, methods, provenance and checksums.

To analyse the full signed network, install `pandas` and `pyarrow`, then:

```python
import pandas as pd
network = pd.read_parquet("data/artifacts/grn_signed.parquet")
print(network.query("TF == 'ETS1'").nlargest(10, "importance"))
```

**Interpretation:** these are pooled, exploratory predictions across tissues
and disease annotations. Motif support applies to a gene set, not experimental
proof of each interaction. Model importance is not a probability; correlation
is not causation. Activity scores should not be compared directly across
cohorts with different gene sets.

This is the existing endothelial-cell example extracted from the verified
704-label collection, not a new validation experiment. All nine original
cohort files are unchanged. See `provenance.json` and `METHODS.md`.

Public example; the full CELLxGENE resource release is pending.
Data licensing is not yet assigned; see `LICENSING.txt`. Cite contributing studies in
`source-citations.json` and the CELLxGENE platform when reusing the results.
Example code: Apache-2.0; see `LICENSE-SOFTWARE`.
