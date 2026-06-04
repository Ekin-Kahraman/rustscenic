"""Run the RustScenic command-line interface with ``python -m rustscenic``."""
from __future__ import annotations

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
