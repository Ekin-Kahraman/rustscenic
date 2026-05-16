# Installation

## Standard Install

```bash
pip install rustscenic
```

Supported Python versions: `3.10` to `3.13`.

Supported platforms in the release workflow:

| Platform | Architecture |
| --- | --- |
| Linux | `x86_64`, `aarch64` |
| macOS | `x86_64`, `aarch64` |
| Windows | `x64` |

Core runtime dependencies are intentionally small:

- `numpy`
- `pandas`
- `pyarrow`
- `scipy`
- `anndata`

## Optional Extras

```bash
pip install "rustscenic[examples]"
pip install "rustscenic[validation]"
pip install "rustscenic[benchmarks]"
```

The `reference` extra includes pySCENIC ecosystem dependencies and is treated as informational because parts of that stack can fail on current Python packaging combinations. For strict reference comparisons, prefer the pinned Docker path under `validation/reference/`.

## Check The Install

```bash
python - <<'PY'
import rustscenic
import rustscenic.grn
import rustscenic.aucell
import rustscenic.cistarget
import rustscenic.topics

print(rustscenic.__version__)
print("core imports OK")
PY
```

## Common Failure Modes

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| TF list has zero overlap with expression matrix | Gene-symbol mismatch, for example ENSEMBL IDs in `var_names` | Use `var["feature_name"]` or pass matching TF names. |
| cisTarget NES is `NaN` | Motif universe too small or zero variance across motifs | Use a real motif ranking database with enough motifs. |
| pySCENIC reference install fails | Upstream dependency conflict | Use the pinned reference Docker path instead of relying on a fresh pip install. |
