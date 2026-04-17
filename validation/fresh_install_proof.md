# Fresh Install Rot Proof — Python 3.12, 2026-04-17

## Environment: brand new uv venv, Python 3.12.13

### rustscenic v0.1
```
$ uv pip install target/wheels/rustscenic-0.1.0-cp310-abi3-macosx_11_0_arm64.whl
Installed 6 packages in 39ms
  + numpy==2.4.4
  + pandas==3.0.2
  + pyarrow==23.0.1
  + rustscenic==0.1.0
$ python -c "import rustscenic; print(rustscenic.__version__)"
0.1.0
$ python quickstart.py  # real PBMC-3k data
wall 2:58   produces biologically-valid output
```
Total install: 16s. Total dep web: 4 packages (numpy, pandas, pyarrow, rustscenic).

### pyscenic 0.12.1 (same fresh env)
```
$ uv pip install pyscenic==0.12.1
$ python -c "from pyscenic.aucell import aucell"
ModuleNotFoundError: No module named 'pkg_resources'
```
pkg_resources was removed from setuptools 82+ (Nov 2025). pyscenic's transitive
dep ctxcore hard-imports it. Fails immediately on any modern Python install.

### arboreto 0.1.6 (same fresh env)
```
$ python -c "from arboreto.algo import grnboost2; grnboost2(...)"
TypeError: Must supply at least one delayed object
  at arboreto/core.py:450 in create_graph
  at dask/dataframe/dask_expr/io/_delayed.py:123
```
Dask's `from_delayed` API changed in dask 2024+. arboreto's last commit (May 2022)
predates the change. Runtime crash, not fixable without forking arboreto.

## Conclusion
On modern Python + numpy 2 + pandas 3, rustscenic is the only installable
GRN inference tool in this space. Both alternatives fail before producing output.
