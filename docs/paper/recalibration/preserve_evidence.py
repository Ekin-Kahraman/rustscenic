"""Hash the full local analysis and make a verified compact paper evidence copy."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import subprocess

B = Path(__file__).resolve().parent
DEST = Path('/Users/ekin/Projects/bio/rustscenic-biological-fidelity/docs/paper/recalibration')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


verification = json.loads((B/'VERIFICATION.json').read_text())
assert verification['status'] == 'passed'
for name, digest in verification['scripts'].items():
    assert sha(B/name) == digest, f'Verified analysis code changed: {name}'
assert subprocess.check_output(['git', 'status', '--porcelain'], cwd=B/'sceptre-source').strip() == b''
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=B/'sceptre-source', text=True).strip()
assert commit == '20c33349c2c5a0df66589e37f67238e67c53bcb0'

exclude_top = {'SOURCE-MANIFEST.json', 'COMPLETION.json', 'PIPELINE-STATUS.json',
               'RESUME-STATE.json', 'finish_when_ready.py', 'progress.py'}
paths = [p for p in B.iterdir() if p.is_file() and p.name not in exclude_top]
for folder in ['inputs', 'results', 'summary', 'figures', 'verification']:
    paths.extend(p for p in (B/folder).rglob('*') if p.is_file())

records = []
for path in sorted(paths):
    rel = path.relative_to(B)
    include = len(rel.parts) == 1 or rel.parts[0] in {'summary', 'figures'}
    if rel.parts[0] == 'inputs':
        include = path.name in {'metadata.json', 'genes.txt', 'primary-tfs.txt'}
    if rel.parts[0] == 'results':
        include = path.name not in {'object.rds', 'pair-qc.csv', 'discovery.csv', 'corrected-pair-results.parquet'}
    records.append(dict(path=str(rel), bytes=path.stat().st_size, sha256=sha(path), included_in_compact_copy=include))
    if include:
        target = DEST/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        assert sha(target) == records[-1]['sha256']

manifest = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                local_analysis_directory=str(B), compact_copy=str(DEST),
                sceptre_commit=commit, sceptre_source_unchanged=True,
                sceptre_binary_sha256=sha(B/'r-library/sceptre/libs/sceptre.so'),
                scope='Working evidence copy; full raw/per-pair files remain local; no public deposit or message sent',
                files=records)
(B/'SOURCE-MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
shutil.copy2(B/'SOURCE-MANIFEST.json', DEST/'SOURCE-MANIFEST.json')
completion = dict(status='complete', discovery_tests=2120690, ranking_rows=26922,
                  negative_control_pairs=52000, negative_control_BH_discoveries=0,
                  regression_tests_passed=6, independent_AP_rows_verified=2037,
                  macro_summaries_verified=126, paired_intervals_verified=144,
                  figures_visually_reviewed=['calibration', 'response-recovery'],
                  compact_files_verified=sum(r['included_in_compact_copy'] for r in records),
                  source_manifest_sha256=sha(B/'SOURCE-MANIFEST.json'),
                  report_sha256=sha(B/'REPORT.md'), sent_messages=0, published=False,
                  publication_scope='Scalability and useful activity signals with limited calibrated response recovery; no general accuracy upgrade')
(B/'COMPLETION.json').write_text(json.dumps(completion, indent=2)+'\n')
shutil.copy2(B/'COMPLETION.json', DEST/'COMPLETION.json')
print(json.dumps(completion, indent=2))
