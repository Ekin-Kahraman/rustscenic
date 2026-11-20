"""One-command Minerva submission. Load-tests the bundled builds first; always collects complete or partial results.

submit   first launch: checksums, load test, twelve jobs and a collector
retry    after every job has ended: resubmits only the jobs with an arm that never finished fitting, keeping the rest
worker   one job: its arms in order, each under its own time limit
collect  tables, verdict and the return archive from whatever exists
"""
import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from common import ROOT, digest, read_json, write_json, verify_package

WHEELS = {'runtime': 'rustscenic-0.6.1-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
          'sklearn122': 'scikit_learn-1.2.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'}
NATIVE = ('native_cross', 'native_same')
REFERENCE = ('reference_cross', 'reference_original')
HOUR = 3600
THREADS = 16


def job(arms, memory, wall, **limit):
    return dict(arms=arms, memory=memory, wall=wall, limit={'pearson': 1, **limit})


# Per job: fitted arms, MB per slot, LSF wall time and per-arm limits in hours (their sum stays inside the wall time).
# Seed 777 of every dataset also computes the Pearson baseline, which does not depend on the seed.
PLAN = {
    **{f'bmmc-{seed}': job(NATIVE + REFERENCE, 6000, '48:00', native_cross=1, native_same=5, reference_cross=5, reference_original=35)
       for seed in (777, 778, 779)},
    **{f'atlas20000-{seed}': job(NATIVE + REFERENCE, 6000, '36:00', native_cross=1, native_same=4, reference_cross=4, reference_original=25)
       for seed in (777, 778, 779)},
    'atlas100000-777': job(NATIVE + REFERENCE, 6000, '96:00', native_cross=2, native_same=10, reference_cross=8, reference_original=72),
    **{f'atlas100000-{seed}': job(NATIVE, 4000, '14:00', native_cross=2, native_same=10) for seed in (778, 779)},
    'atlas500000-777': job(NATIVE, 4000, '48:00', native_cross=4, native_same=42),
    'atlas1306127-777': job(NATIVE, 6000, '96:00', native_cross=4, native_same=88),
    'hdma817740-777': job(NATIVE, 6000, '72:00', native_cross=4, native_same=64),
}
JOBS = list(PLAN)
QUIET = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}


def arms_of(name):
    """Pearson first (seed 777 only), then the two RustScenic rules in an order that alternates by seed, then pySCENIC."""
    dataset, seed = name.rsplit('-', 1)
    native = NATIVE if int(seed) % 2 else NATIVE[::-1]
    return (('pearson',) if seed == '777' else ()) + native + PLAN[name]['arms'][2:]


def python_path(work, arm=None, emulate=False):
    parts = [work / 'runtime'] + ([work / 'sklearn122'] if arm == 'reference_original' and not emulate else [])
    return os.pathsep.join(str(p) for p in parts)


def install_runtimes(work):
    for name, wheel in WHEELS.items():
        target = work / name
        if target.exists():
            continue
        with zipfile.ZipFile(ROOT / 'resources' / wheel) as archive:
            if not all((target / member).resolve().is_relative_to(target.resolve()) for member in archive.namelist()):
                raise ValueError('Unsafe wheel entry')
            archive.extractall(target)


def preflight(work):
    """Seconds-long load test on this node. Only the RustScenic build and the installed reference are mandatory."""
    def run(kind, path):
        env = {**os.environ, **QUIET, 'RAYON_NUM_THREADS': '2', 'PYTHONPATH': path}
        done = subprocess.run([sys.executable, str(ROOT / 'preflight.py'), kind], capture_output=True, text=True, env=env, timeout=900)
        try:
            return json.loads(done.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            return {'ok': False, 'exit_code': done.returncode, 'error': (done.stdout + done.stderr)[-3000:]}
    result = {'native': run('native', python_path(work)),
              'scikit_learn_installed': run('reference', python_path(work)),
              'scikit_learn_122': run('reference', python_path(work, 'reference_original'))}
    installed, old = result['scikit_learn_installed'], result['scikit_learn_122']
    installed['ok'] = bool(installed.get('ok') and installed.get('scikit_learn') == '1.3.2')
    old['ok'] = bool(old.get('ok') and old.get('scikit_learn') == '1.2.2' and installed['ok']
                     and old['trees']['signal'] >= 3 * installed['trees']['signal'])
    result['reference_original'] = 'scikit-learn 1.2.2' if old['ok'] else 'same-mask monitor on scikit-learn 1.3.2 (fallback)'
    write_json(work / 'preflight.json', result)
    return result


def submit_one(account, name, work, recovery, deps=(), resume=False):
    collector = name == 'collect'
    spec = dict(memory=32000, wall='08:00') if collector else PLAN[name]
    threads = 1 if collector else THREADS
    command = shlex.join(['bash', str(ROOT / 'job.sh'), name, str(recovery), str(work), sys.executable] + (['resume'] if resume else []))
    request = ['bsub', '-P', account, '-q', 'premium', '-n', str(threads), '-W', spec['wall'],
               '-R', f"span[hosts=1] rusage[mem={spec['memory']}]", '-J', 'rs-stop-' + name,
               '-o', str(work / f'lsf/{name}-%J.out'), '-e', str(work / f'lsf/{name}-%J.err')]
    if deps:
        request += ['-w', ' && '.join(f'ended({x})' for x in deps)]
    try:
        run = subprocess.run(request + [command], capture_output=True, text=True)
    except OSError as exc:
        return {'status': 'submission_failed', 'message': str(exc)}
    message = (run.stdout + run.stderr).strip()
    match = re.search(r'Job <([0-9]+)>', message)
    print(name + ': ' + message, flush=True)
    if run.returncode == 0 and match:
        return {'status': 'submitted', 'id': int(match[1]), 'threads': threads, 'memory_per_slot_mb': spec['memory'],
                'walltime': spec['wall'], 'message': message}
    return {'status': 'submission_failed', 'exit_code': run.returncode, 'message': message}


def submit_all(account, recovery, work, names, state, resume=False):
    for name in names:
        state['jobs'][name] = submit_one(account, name, work, recovery, resume=resume)
        write_json(work / 'submission.json', state)
    ids = [state['jobs'][name]['id'] for name in names if 'id' in state['jobs'][name]]
    if not ids:
        collect(work)
        raise RuntimeError('No jobs accepted; return work/SEND_BACK files.')
    state['collector'] = submit_one(account, 'collect', work, recovery, ids)
    write_json(work / 'submission.json', state)
    if state['collector']['status'] != 'submitted':
        print('After the jobs finish, run bash collect.sh', flush=True)
    print(f'Return {work}/SEND_BACK.tar.gz and {work}/SEND_BACK.sha256', flush=True)


def submit(account, recovery, work):
    if not re.fullmatch(r'acc_[A-Za-z0-9_-]+', account):
        raise ValueError('Expected Minerva allocation beginning acc_')
    signature = verify_package()
    (work / 'lsf').mkdir(parents=True, exist_ok=True)
    with (work / '.submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (work / 'submission.json').exists():
            raise RuntimeError('Existing submission. Use bash retry.sh to rerun unfinished jobs, or a fresh RUSTSCENIC_WORK directory.')
        install_runtimes(work)
        checks = preflight(work)
        print('Load test: RustScenic build ' + ('passed' if checks['native']['ok'] else 'FAILED')
              + '; installed reference ' + ('passed' if checks['scikit_learn_installed']['ok'] else 'FAILED')
              + '; original-rule reference uses ' + checks['reference_original'], flush=True)
        if not (checks['native']['ok'] and checks['scikit_learn_installed']['ok']):
            raise RuntimeError(f'Load test failed, nothing was submitted. Please send back {work}/preflight.json')
        for kind in ('atlas', 'hdma'):
            if not (recovery / 'work/prepared' / kind / 'rna.h5').is_file():
                print(f'NOTE: {recovery}/work/prepared/{kind}/rna.h5 is missing. Jobs that need it will say so and stop; '
                      'the others are unaffected. Please mention this when returning results.', flush=True)
        state = {'account': account, 'package_signature': signature, 'recovery': str(recovery), 'jobs': {}, 'retries': []}
        write_json(work / 'submission.json', state)
        submit_all(account, recovery, work, JOBS, state)


def unfinished(work):
    names = []
    for name in JOBS:
        folder = work / 'jobs' / name
        if not all((folder / arm / 'SUMMARY.json').exists() for arm in arms_of(name)):
            names.append(name)
    return names


def retry(account, recovery, work):
    verify_package()
    with (work / '.submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (work / 'submission.json').exists():
            raise RuntimeError('Nothing to retry: this work directory has no earlier submission. Use bash start.sh.')
        state = read_json(work / 'submission.json')
        try:
            running = subprocess.run(['bjobs', '-noheader', '-J', 'rs-stop-*'], capture_output=True, text=True).stdout.strip()
        except OSError:
            running = ''
        if running:
            raise RuntimeError('Jobs from this package are still pending or running. Retry after they have all ended:\n' + running)
        names = unfinished(work)
        if not names:
            print('Every arm is already complete; collecting again.', flush=True)
            return collect(work)
        state.setdefault('retries', []).append({'at_utc': datetime.now(timezone.utc).isoformat(), 'jobs': names})
        print('Resubmitting: ' + ', '.join(names), flush=True)
        submit_all(account, recovery, work, names, state, resume=True)


def check_prepared(recovery, dataset):
    kind = 'atlas' if dataset.startswith('atlas') else 'hdma'
    folder, expected = recovery / 'work/prepared' / kind, read_json(ROOT / f'resources/expected-{kind}.json')['files']
    for name in ('rna.h5', 'tfs.txt'):
        if not (folder / name).is_file():
            raise FileNotFoundError(f'Cached {kind} input is missing: {folder / name}')
        if digest(folder / name) != expected[name]:
            raise ValueError(f'Cached {kind} input differs from the archived benchmark: {folder / name}')


def worker(name, recovery, work, resume=False):
    from monitoring import monitored
    out = work / 'jobs' / name
    out.mkdir(parents=True, exist_ok=resume)
    try:
        verify_package()
        slots = int(os.environ.get('LSB_DJOB_NUMPROC', THREADS))
        if slots != THREADS:
            raise ValueError(f'Expected {THREADS} CPU slots, got {slots}')
        dataset, seed = name.rsplit('-', 1)
        spec = PLAN[name]
        if dataset != 'bmmc':
            check_prepared(recovery, dataset)
        emulate = not read_json(work / 'preflight.json')['scikit_learn_122']['ok']
        arms = {}
        for arm in arms_of(name):
            target = out / arm
            if (target / 'SUMMARY.json').exists():  # the fit finished earlier; a late-stage error is repaired from its saved network
                arms[arm] = 'complete_with_stage_errors' if read_json(target / 'SUMMARY.json').get('stage_errors') else 'complete'
                continue
            if target.exists():
                shutil.rmtree(target)  # an unfinished arm from an earlier attempt restarts from scratch
            command = [sys.executable, str(ROOT / 'fit.py'), dataset, seed, arm, str(target),
                       '--recovery', str(recovery), '--threads', str(THREADS)]
            if arm == 'reference_original' and emulate:
                command.append('--emulate')
            config = {'threads': THREADS, 'sample_interval_seconds': 5, 'memory_limit_mb': int(.9 * THREADS * spec['memory']),
                      'env': {'PYTHONPATH': python_path(work, arm, emulate)}}
            status = monitored(command, target, int(spec['limit'][arm] * HOUR), config)['status']
            if status == 'complete' and read_json(target / 'SUMMARY.json').get('stage_errors'):
                status = 'complete_with_stage_errors'
            arms[arm] = status
            write_json(out / 'arms.json', arms)
        write_json(out / 'arms.json', arms)
        fitted = all(x in ('complete', 'complete_with_stage_errors') for x in arms.values())
        status = 'complete' if all(x == 'complete' for x in arms.values()) else 'complete_with_stage_errors' if fitted else 'partial_or_failed'
        write_json(out / 'status.json', {'status': status})
        return 0 if status == 'complete' else 1
    except Exception:
        error = traceback.format_exc()
        write_json(out / 'status.json', {'status': 'failed', 'error': error})
        print(error, file=sys.stderr)
        return 1


def collect(work):
    work.mkdir(parents=True, exist_ok=True)
    summary_error = None
    try:
        from summarise import summarise
        summarise(work, sum(len(arms_of(name)) for name in JOBS))
    except Exception:
        summary_error = traceback.format_exc()
        write_json(work / 'summary-error.json', {'error': summary_error})
    jobs = read_json(work / 'submission.json').get('jobs', {}) if (work / 'submission.json').exists() else {}
    statuses = {name: read_json(work / f'jobs/{name}/status.json') if (work / f'jobs/{name}/status.json').exists()
                else {'status': 'no_worker_completion', 'submission': jobs.get(name, {})} for name in JOBS}
    complete = all(x.get('status') == 'complete' for x in statuses.values()) and summary_error is None
    write_json(work / 'RUN-STATUS.json', {'collected_at_utc': datetime.now(timezone.utc).isoformat(), 'jobs': statuses,
                                          'status': 'complete' if complete else 'partial_or_failed', 'summary_error': summary_error})
    inventory = [{'path': str(p.relative_to(work)), 'bytes': p.stat().st_size, 'sha256': digest(p)}
                 for p in sorted((work / 'jobs').rglob('*')) if p.is_file()] if (work / 'jobs').exists() else []
    write_json(work / 'OUTPUT-INVENTORY.json', inventory)

    def keep(member):
        return None if member.name.startswith('lsf/collect-') else member
    archive, temporary = work / 'SEND_BACK.tar.gz', work / 'SEND_BACK.tar.gz.part'
    for stale in work.glob('SEND_BACK.tar.gz.part*'):
        stale.unlink()
    with tarfile.open(temporary, 'w:gz', compresslevel=1) as tar:  # the networks inside are already compressed
        for name in ('PROTOCOL.md', 'SHA256SUMS', 'inputs/bmmc/provenance.json'):
            tar.add(ROOT / name, arcname=name)
        for name in ('preflight.json', 'submission.json', 'RUN-STATUS.json', 'OUTPUT-INVENTORY.json', 'jobs', 'lsf', 'summary',
                     'summary-error.json'):
            if (work / name).exists():
                tar.add(work / name, arcname=name, filter=keep)
    temporary.replace(archive)
    (work / 'SEND_BACK.sha256').write_text(digest(archive) + '  SEND_BACK.tar.gz\n')
    if archive.stat().st_size > 850_000_000:
        parts = []
        with archive.open('rb') as stream:
            n = 1
            while chunk := stream.read(850_000_000):
                part = work / f'SEND_BACK.tar.gz.part{n:03d}'
                part.write_bytes(chunk)
                parts.append({'name': part.name, 'bytes': len(chunk), 'sha256': digest(part)})
                n += 1
        write_json(work / 'SEND_BACK.parts.json', {'archive': archive.name, 'sha256': digest(archive), 'parts': parts})
        print(f'The archive is over 850 MB: send the {len(parts)} part files and SEND_BACK.parts.json instead.', flush=True)
    print('Collected ' + str(archive), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='action', required=True)
    for action in ('submit', 'retry'):
        s = sub.add_parser(action)
        s.add_argument('account')
        s.add_argument('recovery', type=Path)
        s.add_argument('work', type=Path)
    s = sub.add_parser('worker')
    s.add_argument('job', choices=JOBS)
    s.add_argument('recovery', type=Path)
    s.add_argument('work', type=Path)
    s.add_argument('--resume', action='store_true')
    sub.add_parser('collect').add_argument('work', type=Path)
    a = p.parse_args()
    if a.action == 'submit':
        submit(a.account, a.recovery.resolve(), a.work.resolve())
    elif a.action == 'retry':
        retry(a.account, a.recovery.resolve(), a.work.resolve())
    elif a.action == 'worker':
        sys.exit(worker(a.job, a.recovery.resolve(), a.work.resolve(), a.resume))
    else:
        collect(a.work.resolve())
