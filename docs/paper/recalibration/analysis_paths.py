"""Relocatable roots for the preserved confirmation package and return."""
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
EVIDENCE_ROOT = Path(os.environ.get('RUSTSCENIC_EVIDENCE_ROOT', str(BASE.parent)))
PACKAGE = EVIDENCE_ROOT/'RustScenic-Biology-Confirm-20261112/rustscenic-biology-confirm'
AUDIT = EVIDENCE_ROOT/'RustScenic-Biology-Confirm-Results-Audit-20261112'
RETURNED = AUDIT/'returned'
