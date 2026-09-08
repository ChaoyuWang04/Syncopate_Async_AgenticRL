"""B11: same-allocation native NCCL / PyTorch launch-path comparison."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import modal
if not modal.is_local():
    sys.path.insert(0, '/opt/syncopate-current')
from modal_app.stack_probe import image as base_image, vol, PY, CURRENT_OVERLAY, LOCAL_SOURCE_SHA, LOCAL_GIT_SHA, OVERLAY_DIRS, OVERLAY_FILES

image = base_image.apt_install('openmpi-bin', 'libopenmpi-dev')
app = modal.App('syncopate-b11-baseline')
SOURCES = (
    'modal_app/communication_baseline_batch.py',
    'syncopate/infra/communication_baseline_probe.py',
    'tests/infra/test_communication_baseline_probe.py',
    'syncopate/infra/communication_probe.py',
    'syncopate/infra/hardware_probe.py',
    'syncopate/infra/mechanism_contract.py',
    'syncopate/infra/probe_run.py',
    'syncopate/pipeline/process_guard.py',
    'syncopate/train/source_snapshot.py',
)


def execute(run_id, phase, preflight=''):
    from syncopate.infra.probe_run import reserve_attempt
    from syncopate.infra.mechanism_contract import validate_preflight
    from syncopate.train.source_snapshot import archive_source, verify_orchestrator
    from syncopate.pipeline.process_guard import run_guarded
    if phase not in {'cpu', 'gpu', 'graph', 'direct'}:
        raise ValueError('Unknown phase')
    root = Path(CURRENT_OVERLAY)
    out = reserve_attempt(Path('/vol/_audit/infra-probes'), 'B11', run_id)
    rec = {'experiment':'infra-probes/B11', 'run_id':run_id, 'phase':phase,
           'evidence_path':str(out), 'ok':False, 'steps':[]}
    def persist():
        temp=out/'run.json.tmp'
        temp.write_text(json.dumps(rec,indent=2)); temp.replace(out/'run.json'); vol.commit()
    persist()
    start=time.monotonic()
    env={**os.environ,'PYTHONPATH':str(root),'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    try:
        rec.update(dict(experiment='infra-probes/B11', run_id=run_id, phase=phase,
                   evidence_path=str(out), volume='syncopate-home',
                   modal_task_id=os.environ.get('MODAL_TASK_ID'),
                   git_sha=LOCAL_GIT_SHA, overlay_sha256=LOCAL_SOURCE_SHA,
                   image_id=image.object_id,
                   entry_sha256=verify_orchestrator(Path(__file__), root/'modal_app/communication_baseline_batch.py'),
                   sources={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in SOURCES},
                   lock_sha256=hashlib.sha256((root/'modal_app/stack/uv.lock').read_bytes()).hexdigest(),
                   started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), ok=False, steps=[]))
        rec['source_archive'] = archive_source(root, OVERLAY_DIRS, OVERLAY_FILES, expected=LOCAL_SOURCE_SHA, output=out/'source.tar.gz')
        command = [PY, '-m', 'syncopate.infra.communication_baseline_probe', '--mode', ({'graph':'native-graph','direct':'torch-direct'}.get(phase,phase)), '--out', str(out)]
        if phase == 'cpu':
            commands = [[PY, '-m', 'pytest', 'tests/infra/test_communication_baseline_probe.py', 'tests/infra/test_mechanism_contract.py', '-q', '-ra'], command]
        else:
            pre = Path(preflight)
            if not pre.is_relative_to('/vol/_audit/infra-probes/B11') or not pre.name.startswith('attempt-'):
                raise ValueError('Explicit B11 CPU attempt required')
            prior = json.loads((pre/'run.json').read_text())
            validate_preflight(prior, rec, 'cpu')
            for name, digest in prior['files'].items():
                if name.endswith('.json') and hashlib.sha256((pre/name).read_bytes()).hexdigest() != digest:
                    raise ValueError('CPU result seal mismatch: '+name)
            rec['cpu_preflight'] = str(pre)
            commands = [[PY, '-c', "import torch; assert torch.cuda.device_count()==2; assert all(torch.cuda.get_device_capability(i)==(10,0) and 'B200' in torch.cuda.get_device_name(i) for i in range(2))"], command + ['--preflight', str(pre)]]
        rec['commands'] = commands
        persist()
        for i, cmd in enumerate(commands):
            remaining=(1050 if phase=='cpu' else 1650)-(time.monotonic()-start)
            if remaining <= 0: raise TimeoutError('Phase deadline exhausted')
            result=run_guarded(['bash','-c','exec "$@" >"$PROBE_LOG" 2>&1','probe',*cmd],cwd=str(root),env={**env,'PROBE_LOG':str(out/f'step-{i}.log')},timeout=remaining)
            rec['steps'].append(result); persist()
            if result['rc'] != 0: break
            if phase=='cpu' and i==0:
                log=(out/'step-0.log').read_text()
                if not re.search(r'\b[1-9][0-9]* passed\b',log) or re.search(r'\b[1-9][0-9]* (skipped|failed|xfailed|error)',log):
                    raise RuntimeError('Target CPU tests must pass without skips')
        else:
            name='communication-baseline-cpu.json' if phase=='cpu' else 'communication-baseline-result.json'
            final=json.loads((out/name).read_text())
            if final.get('ok') is not True: raise RuntimeError('Missing successful terminal result')
            rec['validated_results']=[name];rec['ok']=True
    except BaseException as exc:
        rec['exception']={'type':type(exc).__name__,'message':str(exc)}
    finally:
        rec['seconds']=time.monotonic()-start
        rec['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.suffix=='.json' and p.name!='run.json'}
        rec['log_seal_policy']='Logs are shutdown diagnostics; result JSON and artifact manifest hashes are authoritative.'
        persist()
    return rec


@app.function(image=image,volumes={'/vol':vol},cpu=4,memory=16384,timeout=1200,retries=0,single_use_containers=True)
def cpu(run_id):
    return execute(run_id,'cpu')


@app.function(image=image,volumes={'/vol':vol},gpu='B200:2',cpu=16,memory=196608,timeout=1800,retries=0,single_use_containers=True)
def gpu(run_id,preflight,phase='gpu'):
    return execute(run_id,phase,preflight)


@app.local_entrypoint()
def main(run_id:str,phase:str='cpu',preflight:str=''):
    if phase not in {'cpu','gpu','graph','direct'}: raise ValueError('Unknown phase; no resource allocated')
    if phase!='cpu' and not preflight: raise ValueError('CPU preflight required before GPU allocation')
    result=cpu.remote(run_id) if phase=='cpu' else gpu.remote(run_id,preflight,phase)
    print(json.dumps(result,indent=2))
    if not result['ok']: raise RuntimeError('B11 failed; evidence: '+result['evidence_path'])
