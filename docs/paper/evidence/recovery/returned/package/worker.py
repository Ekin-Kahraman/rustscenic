"""One fresh process per method, from the same saved input to exported outputs."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import time
import numpy as np
import pandas as pd
from common import ROOT, digest, fingerprint, read_json, write_json
from data_io import load_matrix, normalise
from workflows import Clock, ann, rna_workflow, multiome_workflow


def save_activity(out, name, frame, expected_cells):
    if frame.index.tolist() != expected_cells:
        raise ValueError('Activity cell identities/order changed')
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate activity signatures')
    values = frame.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite activity')
    np.save(out / ('full-' + name + '-activity.npy'), values)
    write_json(out / (name + '-activity-signatures.json'), frame.columns.tolist())
    sample = np.sort(np.random.default_rng(20261028).choice(len(frame), size=min(10000, len(frame)), replace=False))
    sampled = pd.DataFrame(values[sample], index=frame.index[sample], columns=frame.columns)
    sampled.to_parquet(out / ('sampled-' + name + '-activity.parquet'))
    summary = [{'signature': str(col), 'mean': float(values[:, j].mean()), 'std': float(values[:, j].std()),
                'minimum': float(values[:, j].min()), 'maximum': float(values[:, j].max()),
                'nonzero_cells': int(np.count_nonzero(values[:, j]))} for j, col in enumerate(frame.columns)]
    write_json(out / (name + '-activity-summary.json'), summary)


def export_table(out, key, value):
    value.attrs.clear()
    if key == 'motif_evidence' and 'peak_id' in value.columns:
        value.to_parquet(out / 'full-motif-evidence.parquet', index=False)
        compact = value.groupby(['TF', 'peak_id']).agg(max_auc=('auc', 'max'), max_nes=('nes', 'max'),
                                                      supporting_motifs=('motif', 'nunique')).reset_index()
        compact.to_parquet(out / 'motif-evidence.parquet', index=False)
    else:
        value.to_parquet(out / (key.replace('_', '-') + '.parquet'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['atlas', 'bmmc','hdma'], required=True)
    p.add_argument('--method', choices=['rustscenic', 'reference'], required=True)
    p.add_argument('--cells', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--inputs-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mallet', type=Path)
    p.add_argument('--config', type=Path, default=ROOT / 'config.json')
    p.add_argument('--allow-local', action='store_true')
    a = p.parse_args()
    if not os.environ.get('LSB_JOBID') and not a.allow_local:
        raise RuntimeError('Use start.sh; do not run benchmarks on the login node')
    a.out.mkdir(parents=True, exist_ok=True)
    config = read_json(a.config)
    folder = a.inputs_root / a.dataset
    clock = Clock()
    write_json(a.out / 'phase.json', {'phase': 'analysis'})
    began = time.perf_counter()
    x, genes, cells = clock.run('rna_load', lambda: load_matrix(folder / 'rna.h5', a.cells))
    rna = ann(x, genes, cells)
    tfs = (folder / 'tfs.txt').read_text().splitlines()
    assert len(genes) == config['panels'][a.dataset]['genes'] and len(tfs) == 256
    if a.dataset == 'atlas':
        result = rna_workflow(a.method, rna, tfs, ROOT / 'inputs/motifs', config, a.seed, clock)
    else:
        y, peaks, acells = clock.run('atac_counts_load', lambda: load_matrix(folder / 'atac.h5', a.cells))
        assert cells == acells
        atac = ann(y, peaks, cells)
        motif_folder=folder/'motifs'
        if a.allow_local and not motif_folder.exists():
            motif_folder=ROOT/'inputs/motifs'
        result = multiome_workflow(a.method, rna, atac, None, tfs, folder, motif_folder,
                                  config, a.seed, a.out, a.mallet, clock)
    analysis_s = time.perf_counter() - began
    write_json(a.out / 'phase.json', {'phase': 'output'})
    export_began = time.perf_counter()
    for key, value in result.items():
        if isinstance(value, pd.DataFrame) and key not in ['gene_activity', 'region_activity']:
            export_table(a.out, key, value)
        elif key in ['regulons', 'region_regulons']:
            write_json(a.out / (key.replace('_', '-') + '.json'), value)
    if result['grn'].empty or not np.isfinite(result['grn'].importance).all():
        raise ValueError('Empty/nonfinite GRN')
    save_activity(a.out, 'gene', result['gene_activity'], cells)
    if 'region_activity' in result:
        save_activity(a.out, 'region', result['region_activity'], cells)
    write_json(a.out / 'cells.json', cells)
    markers = ['SPI1','PAX5','EBF1','GATA1','TCF7'] if a.dataset == 'bmmc' else ['HNF4A','CRX','NEUROD2','MYOD1','TBX5','SPI1'] if a.dataset == 'hdma' else []
    motif_tfs=set(result.get('motif_evidence',pd.DataFrame()).get('TF',[]))
    grn_tfs=set(result['grn'].get('TF',[]))
    write_json(a.out/'marker-coverage.json', {'predeclared_markers':[
        {'TF':tf,'in_selected_tf_panel':tf in tfs,'has_grn_edges':tf in grn_tfs,
         'has_region_motif_evidence':tf in motif_tfs,'has_final_gene_regulon':tf in result['regulons']}
        for tf in markers], 'selection_changed_after_results':False})
    export_s = time.perf_counter() - export_began
    files = {f.name: {'sha256': digest(f), 'bytes': f.stat().st_size} for f in a.out.iterdir()
             if f.is_file() and f.suffix in ['.parquet', '.npy', '.json'] and f.name != 'phase.json'}
    actual_method = 'rustscenic' if a.method == 'rustscenic' else ('pyscenic' if a.dataset == 'atlas' else 'scenicplus')
    result = {'status': 'complete', 'dataset': a.dataset, 'method': actual_method, 'cells': len(cells),
        'genes': len(genes), 'tfs': len(tfs), 'seed': a.seed, 'threads': config['threads'],
        'host': socket.gethostname(), 'job_id': os.environ.get('LSB_JOBID', ''),
        'analysis_s': analysis_s, 'export_s': export_s, 'input_to_export_s': analysis_s + export_s,
        'stages': clock.stages, 'fit_summary': clock.fit_summary, 'gene_regulons': len(result['regulons']),
        'biological_outputs_nonempty': bool(result['regulons']) and not result['gene_activity'].empty
            and bool(np.any(result['gene_activity'].to_numpy() > 0)),
        'n_eregulons': result.get('n_eregulons'), 'n_default_modules': result.get('n_default_modules'),
        'scope': result['scope'], 'local_test': a.allow_local,
        'input_signature': fingerprint(read_json(folder / 'input.json')), 'config_signature': fingerprint(config),
        'versions': {pkg: importlib.metadata.version(pkg) for pkg in ['rustscenic', 'pyscenic', 'scenicplus', 'pycistarget', 'pycistopic', 'numpy', 'pandas', 'scikit-learn']},
        'memory_orchestration': {'rustscenic_peak_batch':256,'rustscenic_aucell_cell_batch':2048,
                                'native_wheel_unchanged':True,'no_cells_or_features_removed':True},
        'runtime_additions': {pkg: importlib.metadata.version(pkg) for pkg in ['lda','polars-lts-cpu','numexpr']}
                            if not a.allow_local else {'scope':'Local compatibility pilot'},
        'outputs': files,
        'full_activity_portability': 'Full scores retained on HPC with hashes; return archive includes a fixed shared 10000-cell sample, full-network outputs and metrics computed across all cells.'}
    write_json(a.out / 'engine.json', result)
    print(json.dumps({k: result[k] for k in ['method', 'cells', 'analysis_s', 'gene_regulons', 'biological_outputs_nonempty']}), flush=True)


if __name__ == '__main__':
    main()
