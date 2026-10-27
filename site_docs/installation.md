# Installation

RustScenic **v0.5.1** supports regular CPython **3.10–3.14** on Linux, macOS and Windows.
Prebuilt packages need no Rust setup. Core analysis runs on CPUs without Java,
Dask, CUDA or Snakemake.

## Install

Use a clean environment. With Conda:

```bash
conda create -n rustscenic python=3.12 -y
conda activate rustscenic
```

Install and check the version:

```bash
python -m pip install rustscenic
python -c "import rustscenic; print(rustscenic.__version__)"
```

Expected version: `0.5.1`.

<details markdown="1">
<summary>Using venv instead of Conda</summary>

With regular CPython 3.10–3.14 installed, create an environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then run the install and check commands above.

</details>

Next, [run the PBMC3k example](https://github.com/Ekin-Kahraman/rustscenic/blob/main/QUICKSTART.md)
or [analyse your own data](quickstart.md).

<details markdown="1">
<summary>Supported platforms and dependencies</summary>

| Platform | Architecture |
| --- | --- |
| Linux | `x86_64`, `aarch64` |
| macOS | `x86_64`, `aarch64` |
| Windows | `x64` |

The core dependencies (`numpy`, `pandas`, `pyarrow`, `scipy` and `anndata`)
install automatically.

Python 3.14 support refers to CPython with the GIL enabled. The standard audit
matrix tests source builds on Linux, macOS and Windows.
The public v0.5.1 macOS `arm64` ABI3 wheel was also installed and exercised on
CPython 3.14.5 with synthetic software fixtures. The other released platform
wheels were not individually requalified for Python 3.14 in that local check.
Free-threaded Python builds have not been qualified.

</details>

<details markdown="1">
<summary>Optional extras and reference comparisons</summary>

The PBMC3k walkthrough uses the `examples` extra. Validation and benchmark
extras add packages for comparison scripts.

```bash
python -m pip install "rustscenic[examples]"
python -m pip install "rustscenic[validation]"
python -m pip install "rustscenic[benchmarks]"
```

For pySCENIC reference comparisons, prefer the pinned
[Docker environment](https://github.com/Ekin-Kahraman/rustscenic/tree/main/validation/reference).
The `reference` extra can encounter upstream dependency conflicts.

</details>

<details markdown="1">
<summary>Troubleshooting</summary>

| Symptom | What to do |
| --- | --- |
| Package not found after installing | Use `python -m pip` in the same environment as your script. In a notebook, select that environment's kernel. |
| No compatible wheel, or pip tries to compile Rust | Check the Python version and platform above, then update pip with `python -m pip install --upgrade pip`. |
| TFs do not match your genes | Check species and gene identifiers in the [input guide](quickstart.md#prepare-your-input). |

</details>
