"""Bounded B02 FSDP2/LoRA probe. Separate entry keeps B01 CPU/GPU source fixed."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import modal
if not modal.is_local():
    sys.path.insert(0,'/opt/syncopate-current')
from modal_app.stack_probe import (image,vol,PY,CURRENT_OVERLAY,LOCAL_SOURCE_SHA,
                                  LOCAL_GIT_SHA,OVERLAY_DIRS,OVERLAY_FILES)
app=modal.App('syncopate-lora-probe')


def execute(run_id,mode,cpu_run=''):
    from syncopate.infra.probe_run import reserve_run
    from syncopate.train.source_snapshot import archive_source,verify_orchestrator
    from syncopate.pipeline.process_guard import run_guarded
    root=Path(CURRENT_OVERLAY);entry=verify_orchestrator(Path(__file__),root/'modal_app/lora_probe.py')
    out=reserve_run(Path('/vol/_audit/infra-probes'),'B02',run_id)
    record={'experiment':'infra-probes/B02','run_id':run_id,'mode':mode,'volume':'syncopate-home',
            'evidence_path':str(out),'git_sha':LOCAL_GIT_SHA,'overlay_sha256':LOCAL_SOURCE_SHA,
            'entry_sha256':entry,'image_id':image.object_id,
            'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'ok':False,'steps':[]}
    record['probe_sources']={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in [
        'modal_app/lora_probe.py','syncopate/infra/lora_accumulation_probe.py','syncopate/infra/training_probe.py',
        'syncopate/infra/probe_run.py','syncopate/pipeline/process_guard.py','syncopate/train/source_snapshot.py',
        'tests/infra/test_lora_accumulation_probe.py','modal_app/stack/uv.lock']}
    record['source_archive']=archive_source(root,OVERLAY_DIRS,OVERLAY_FILES,expected=LOCAL_SOURCE_SHA,output=out/'source.tar.gz')
    if mode=='cpu':
        commands=[[PY,'-m','pytest','tests/infra/test_lora_accumulation_probe.py','tests/infra/test_probe_run.py','-q'],
                  [PY,'-m','syncopate.infra.lora_accumulation_probe','cpu',str(out)]]
    else:
        assert re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}',cpu_run)
        pre=out.parent/cpu_run;previous=json.loads((pre/'run.json').read_text())
        assert previous['ok'] and previous['mode']=='cpu' and previous['probe_sources']==record['probe_sources']
        raw=(pre/'cpu-result.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==previous['files']['cpu-result.json']
        cpu=json.loads(raw);assert cpu['ok'] and not cpu['cuda_initialized']
        for name,digest in cpu['tiny_files'].items():
            assert Path(name).name==name and hashlib.sha256((pre/'tiny'/name).read_bytes()).hexdigest()==digest
        shutil.copytree(pre/'tiny',out/'tiny')
        record['cpu_preflight']=str(pre)
        commands=[['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.used,driver_version','--format=csv'],
                  [PY,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=2','--max-restarts=0',
                   '--module','syncopate.infra.lora_accumulation_probe','gpu',str(out),str(out/'tiny')]]
    record['commands']=commands
    (out/'run.json').write_text(json.dumps(record,indent=2));vol.commit();start=time.monotonic()
    env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8','HF_HUB_OFFLINE':'1',
         'TORCHINDUCTOR_CACHE_DIR':str(out/'cache/inductor')}
    try:
        for i,command in enumerate(commands):
            remaining=720-(time.monotonic()-start)
            if remaining<=0:raise TimeoutError('batch deadline exhausted')
            result=run_guarded(['bash','-c','exec "$@" >"$PROBE_LOG" 2>&1','probe',*command],cwd=str(root),
                env={**env,'PROBE_LOG':str(out/f'step-{i}.log')},timeout=remaining,stop_grace=100)
            record['steps'].append(result)
            (out/'run.json').write_text(json.dumps(record,indent=2));vol.commit()
            if result['rc']!=0:break
        else:
            if mode=='gpu':
                ranks=[json.loads((out/f'rank-{r}.json').read_text()) for r in range(2)]
                record['ok']=all(r['ok'] and len(r['steps'])==2 for r in ranks)
            else:record['ok']=json.loads((out/'cpu-result.json').read_text())['ok']
    finally:
        record['seconds']=time.monotonic()-start
        record['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()
                         if p.is_file() and p.name not in {'run.json','source.tar.gz'}}
        (out/'run.json').write_text(json.dumps(record,indent=2));vol.commit()
    return record


@app.function(image=image,volumes={'/vol':vol},cpu=4,memory=16384,timeout=900,retries=0)
def cpu(run_id:str):return execute(run_id,'cpu')


@app.function(image=image,volumes={'/vol':vol},gpu='B200:2',cpu=16,memory=65536,timeout=900,retries=0)
def gpu(run_id:str,cpu_run:str):return execute(run_id,'gpu',cpu_run)


@app.local_entrypoint()
def main(run_id:str,mode:str='cpu',cpu_run:str=''):
    if mode=='cpu':result=cpu.remote(run_id)
    elif mode=='gpu':result=gpu.remote(run_id,cpu_run)
    else:raise ValueError('unknown mode')
    print(json.dumps(result,indent=2))
