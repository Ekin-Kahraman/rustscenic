"""Stage the four BEELINE datasets no RustScenic analysis has used.

Uses the 28 October prepare_inputs.py unchanged (same checksum-pinned archives,
same gene/TF selection rule, no reference links used in selection). Only the
dataset list differs. Run only after CONFIRMATION-PROTOCOL.md is hashed.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'RustScenic-Biological-Validation-20261028'
sys.path.insert(0, str(SOURCE))
import prepare_inputs  # noqa: E402

HUMAN = ['HepG2-ChIP-seq-network.csv', 'Non-specific-ChIP-seq-network.csv', 'STRING-network.csv']
MOUSE = ['mHSC-ChIP-seq-network.csv', 'Non-Specific-ChIP-seq-network.csv', 'STRING-network.csv']
prepare_inputs.SPECS = {
    'hHep': ('human', HUMAN),
    'mHSC-E': ('mouse', MOUSE),
    'mHSC-GM': ('mouse', MOUSE),
    'mHSC-L': ('mouse', MOUSE),
}

if __name__ == '__main__':
    sys.argv = [sys.argv[0], '--cache', str(SOURCE / 'source-cache'), '--out', str(HERE / 'inputs/beeline-confirmation')]
    prepare_inputs.main()
