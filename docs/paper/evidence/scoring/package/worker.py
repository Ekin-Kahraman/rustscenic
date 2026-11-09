"""Fresh-process scoring checks. No network fitting or parameter search."""
import argparse
import json
import os
from pathlib import Path
import traceback

import h5py
import numpy as np
import pandas as pd
from common import ROOT, digest, read_json, write_json
from matrix_io import csr_slice, strings
from monitoring import environment
from polarity import signed_contrast
from scoring import score_block, score_file, signatures
from utility import evaluate


def verify_engine(arm):
    import rustscenic
    build = read_json(ROOT / 'build.json')
    engine = 'baseline' if arm == 'baseline' else 'candidate'
    expected = build['wheels'][engine]
    module = Path(rustscenic.__file__).parent
    actual = digest(module / Path(expected['extension']).name)
    if actual != expected['extension_sha256']:
        raise RuntimeError('Wrong native engine loaded: ' + str(module))


def biology(arm, out, threads):
    matrix = ROOT / 'external/rna.h5'
    with h5py.File(matrix) as h:
        genes, cells = strings(h['features'][:]), strings(h['cells'][:])
        x = csr_slice(h, 0, len(cells))
    meta = pd.read_csv(ROOT / 'external/cell-metadata.tsv', sep='\t', dtype={'donor':str}).set_index('cell').loc[cells]
    records = []
    scorer = 'seeded' if arm == 'seeded' else 'native'
    for n in (5000, 20000):
        for seed in (777, 778, 779):
            pair = ROOT / 'expected' / f'{n}-{seed}'
            previous = read_json(pair / 'assessment.json')
            edges = pd.read_parquet(ROOT / 'networks' / f'bmmc-train{n}-seed{seed}' / 'rustscenic.parquet')
            for mode in ('union', 'signed'):
                sets = signatures(edges, 'rustscenic', mode)
                frame = pd.concat([score_block(x[i:i+2048], genes, cells[i:i+2048], sets, scorer, threads)
                                   for i in range(0, len(cells), 2048)])
                if mode == 'signed':
                    frame = signed_contrast(frame)
                expected = pd.read_parquet(pair / f'{mode}-{"shared" if arm == "seeded" else "native"}.parquet')
                if not frame.index.equals(expected.index) or set(frame.columns) != set(expected.columns):
                    raise ValueError('Scoring axes changed')
                frame = frame[expected.columns]
                error = float(np.max(np.abs(frame.to_numpy() - expected.to_numpy())))
                record = {'training_cells':n, 'inference_seed':seed, 'mode':mode,
                          'max_absolute_error':error, 'passed':error <= (1e-7 if arm == 'seeded' else 0.)}
                if arm == 'seeded':
                    frame.to_parquet(out / f'{n}-{seed}-{mode}.parquet')
                    result = evaluate(frame[previous['common_tfs']], meta, mode)
                    reference = previous['results'][f'rustscenic-{mode}-common']
                    record.update(result=result, matching_reference_confusions=sum(
                        a['confusion_matrix'] == b['confusion_matrix'] for a,b in zip(result['folds'], reference['folds'])))
                    record['classifier_converged'] = all(f['status']=='complete' and f['converged'] for f in result['folds'])
                records.append(record)
                write_json(out / 'verification.json', {'arm':arm, 'records':records,
                           'numerical_pass':all(r['passed'] for r in records), 'expected_comparisons':12})


def main(a):
    verify_engine(a.arm)
    a.out.mkdir(parents=True, exist_ok=True)
    write_json(a.out / 'environment.json', environment())
    write_json(a.out / 'phase.json', {'phase':'analysis'})
    if a.kind == 'biology':
        biology(a.arm, a.out, a.threads)
    else:
        edges = pd.read_parquet(ROOT / 'networks' / f'hdma-n817740-seed{a.seed}' / 'rustscenic.parquet')
        scorer = {'baseline':'native', 'unseeded':'native', 'seeded':'seeded', 'reference':'shared'}[a.arm]
        score_file(a.matrix, edges, 'rustscenic', 'signed', scorer, a.out, a.threads, 4096)
    outputs = {p.name:{'sha256':digest(p), 'bytes':p.stat().st_size} for p in a.out.iterdir()
               if p.is_file() and p.suffix in ('.json','.h5','.parquet') and p.name not in ('complete.json','monitor.json','phase.json')}
    write_json(a.out / 'complete.json', {'status':'complete', 'smoke':a.smoke,
               'package':digest(ROOT / 'SHA256SUMS'), 'outputs':outputs})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['biology','atlas'], required=True)
    parser.add_argument('--arm', choices=['baseline','unseeded','seeded','reference'], required=True)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--matrix', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not args.smoke and not os.environ.get('LSB_JOBID'):
        raise RuntimeError('Use start.sh on the login node; scoring requires the compute allocation')
    try:
        main(args)
    except BaseException:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'error.txt').write_text(traceback.format_exc())
        raise
