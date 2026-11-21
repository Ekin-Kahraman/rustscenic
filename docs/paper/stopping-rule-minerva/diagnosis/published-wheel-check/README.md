# Published-wheel bit-identity check (20 November 2026)

An independent audit installed the published PyPI wheels (rustscenic 0.7.0, macOS
arm64, SHA-256 in `wheel-sha256.txt`, matching `pypi-0.7.0-digests.txt`; and
rustscenic 0.6.1) into clean environments and refitted seed 777 under the current
rule:

- `result-default.json`: 0.7.0 default (exact bins), BMMC, against the local
  exact-bin refit `cap65535-arboreto-777`: identical edges and float32 bits.
- `result-255.json`: 0.7.0 with `max_bins=255`, BMMC, against the returned HPC
  network `bmmc-777/native_cross`: identical.
- `result-v061.json`: published 0.6.1, BMMC, against the same returned network:
  identical.
- `result-atlas20000.json`: 0.7.0 default and `max_bins=255` on the rebuilt
  20,000-cell atlas subset, against the local refits `atlas20000-cap65535-arboreto-777`
  and `atlas20000-cap255-arboreto-777`: identical.

`bitcheck.py` and `bitcheck_atlas.py` are the scripts as run; `run-*.log` are
their logs. The recorded inference times come from single runs made while other
jobs shared the laptop and are not used as timing evidence.
