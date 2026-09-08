"""Independent, bounded infra probes. No business pipeline or shared repo writes.

modal run --detach modal_app/infra_probes.py --run-id b00-cpu-01
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import sys

import modal
# Modal uploads this entry as /root/infra_probes.py; source packages live in the image.
if not modal.is_local():
    sys.path.insert(0, "/opt/syncopate-current")
from modal_app.stack_probe import (image, vol, PY, CURRENT_OVERLAY, LOCAL_SOURCE_SHA,
                                  LOCAL_GIT_SHA, OVERLAY_DIRS, OVERLAY_FILES)

app = modal.App('syncopate-infra-probes')


@app.function(image=image, volumes={'/vol': vol}, cpu=2, memory=8192, timeout=900, retries=0)
def b00_cpu(run_id: str):
    from syncopate.infra.probe_run import reserve_run
    from syncopate.train.source_snapshot import archive_source, verify_orchestrator
    root = Path(CURRENT_OVERLAY)
    entry_sha = verify_orchestrator(Path(__file__), root / 'modal_app/infra_probes.py')
    out = reserve_run(Path('/vol/_audit/infra-probes'), 'B00', run_id)
    record = {'experiment': 'infra-probes/B00', 'run_id': run_id, 'git_sha': LOCAL_GIT_SHA,
              'overlay_sha256': LOCAL_SOURCE_SHA, 'entry_sha256': entry_sha,
              'image_id': image.object_id, 'volume': 'syncopate-home', 'evidence_path': str(out),
              'cpu': 2, 'memory_mib': 8192, 'gpu': None, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'lock_sha256': hashlib.sha256((root / 'modal_app/stack/uv.lock').read_bytes()).hexdigest()}
    record['source_archive'] = archive_source(root, OVERLAY_DIRS, OVERLAY_FILES,
                                             expected=LOCAL_SOURCE_SHA, output=out / 'source.tar.gz')
    (out / 'run.json').write_text(json.dumps(record, indent=2))
    vol.commit()
    start = time.monotonic()
    command = [PY, '-m', 'syncopate.infra.length_metric_probe', str(out)]
    try:
        with (out / 'stdout.log').open('w') as log:
            process = subprocess.run(command, cwd=root, env={**os.environ, 'PYTHONPATH': str(root)},
                                     stdout=log, stderr=subprocess.STDOUT, timeout=720)
        record['returncode'] = process.returncode
        record['ok'] = process.returncode == 0 and (out / 'result.json').is_file()
    except subprocess.TimeoutExpired:
        record.update(ok=False, timeout=True)
    finally:
        record.update(command=command, seconds=time.monotonic() - start)
        record['files'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in out.iterdir() if p.is_file() and p.name not in {'run.json', 'source.tar.gz'}}
        (out / 'run.json').write_text(json.dumps(record, indent=2))
        vol.commit()
    return record


def _b01_run(run_id: str, mode: str, cpu_run: str = ""):
    from syncopate.infra.probe_run import reserve_run
    from syncopate.train.source_snapshot import archive_source, verify_orchestrator
    from syncopate.pipeline.process_guard import run_guarded
    root = Path(CURRENT_OVERLAY)
    entry_sha = verify_orchestrator(Path(__file__), root / 'modal_app/infra_probes.py')
    out = reserve_run(Path('/vol/_audit/infra-probes'), 'B01', run_id)
    rec = {'experiment': 'infra-probes/B01', 'run_id': run_id, 'mode': mode,
           'git_sha': LOCAL_GIT_SHA, 'overlay_sha256': LOCAL_SOURCE_SHA, 'entry_sha256': entry_sha,
           'image_id': image.object_id, 'volume': 'syncopate-home', 'evidence_path': str(out),
           'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'lock_sha256': hashlib.sha256((root/'modal_app/stack/uv.lock').read_bytes()).hexdigest()}
    rec['probe_sources'] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
        'modal_app/infra_probes.py','syncopate/infra/probability_probe.py',
        'syncopate/infra/probe_run.py','syncopate/pipeline/process_guard.py','syncopate/train/source_snapshot.py',
        'tests/infra/test_probability_probe.py','tests/infra/test_probe_run.py')}
    rec['source_archive'] = archive_source(root, OVERLAY_DIRS, OVERLAY_FILES,
                                         expected=LOCAL_SOURCE_SHA, output=out/'source.tar.gz')
    env = {**os.environ, 'PYTHONPATH': str(root), 'HF_HUB_OFFLINE': '1' if mode == 'gpu' else '0',
           'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_CACHE_ROOT': str(out/'cache/vllm'),
           'FLASHINFER_WORKSPACE_BASE': str(out/'cache/flashinfer'), 'TORCHINDUCTOR_CACHE_DIR': str(out/'cache/inductor')}
    if mode == 'cpu':
        commands = [[PY, '-m', 'pytest', 'tests/infra/test_probability_probe.py', 'tests/infra/test_probe_run.py', '-q'],
                    [PY, '-m', 'syncopate.infra.probability_probe', 'cpu', str(out)]]
    else:
        # Restrict to a previously successful B01 CPU run; never choose an approximate path.
        import re
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}', cpu_run): raise ValueError('invalid CPU run')
        pre = out.parent/cpu_run
        previous = json.loads((pre/'run.json').read_text())
        assert previous['ok'] and previous['mode'] == 'cpu'
        inputs = (pre/'input.json').read_bytes()
        assert hashlib.sha256(inputs).hexdigest() == previous['files']['input.json']
        assert previous['lock_sha256'] == rec['lock_sha256']
        assert previous['probe_sources'] == rec['probe_sources'], 'probe source differs from CPU-validated files'
        (out/'input.json').write_bytes(inputs)
        rec['cpu_preflight'] = str(pre)
        commands = [[PY, '-m', 'syncopate.infra.probability_probe', 'verify', str(out), str(out/'input.json')],
                    ['nvidia-smi', '--query-gpu=name,uuid,memory.total,memory.used,driver_version', '--format=csv'],
                    [PY, '-c', "import torch; assert torch.cuda.device_count()==1; assert torch.cuda.get_device_capability()==(10,0)"],
                    *[[PY, '-m', 'syncopate.infra.probability_probe', e, str(out), str(out/'input.json')] for e in ['hf','vllm']],
                    [PY, '-m', 'syncopate.infra.probability_probe', 'summary', str(out)]]
    rec['commands'] = commands; rec['ok'] = False; rec['steps'] = []
    (out/'run.json').write_text(json.dumps(rec, indent=2)); vol.commit()
    start = time.monotonic(); limit = 1050 if mode == 'cpu' else 1650
    try:
        for i,cmd in enumerate(commands):
            remaining = limit-(time.monotonic()-start)
            if remaining <= 0: raise TimeoutError('batch deadline exhausted')
            wrapped = ['bash', '-c', 'exec "$@" >"$PROBE_LOG" 2>&1', 'probe', *cmd]
            result = run_guarded(wrapped,cwd=str(root),env={**env,'PROBE_LOG':str(out/f'step-{i}.log')},timeout=remaining)
            rec['steps'].append(result)
            (out/'run.json').write_text(json.dumps(rec, indent=2)); vol.commit()
            if result['rc'] != 0: break
        else: rec['ok'] = True
    finally:
        rec['seconds'] = time.monotonic()-start
        rec['files'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()
                        if p.is_file() and p.name not in {'run.json','source.tar.gz'}}
        (out/'run.json').write_text(json.dumps(rec,indent=2)); vol.commit()
    return rec


@app.function(image=image, volumes={'/vol':vol}, cpu=4, memory=16384, timeout=1200, retries=0)
def b01_cpu(run_id: str):
    return _b01_run(run_id,'cpu')


@app.function(image=image, volumes={'/vol':vol}, gpu='B200', cpu=16, memory=131072, timeout=1800, retries=0)
def b01_gpu(run_id: str, cpu_run: str):
    return _b01_run(run_id,'gpu',cpu_run)


@app.local_entrypoint()
def main(run_id: str, probe: str = "b00-cpu", cpu_run: str = ""):
    if probe == "b00-cpu": result = b00_cpu.remote(run_id)
    elif probe == "b01-cpu": result = b01_cpu.remote(run_id)
    elif probe == "b01-gpu": result = b01_gpu.remote(run_id,cpu_run)
    else: raise ValueError('unknown probe')
    print(json.dumps(result,indent=2))
