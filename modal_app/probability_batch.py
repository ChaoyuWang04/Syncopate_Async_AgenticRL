"""Approved three-probe batch; each Modal delivery owns an immutable attempt."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import modal
if not modal.is_local(): sys.path.insert(0,'/opt/syncopate-current')
from modal_app.stack_probe import image,vol,PY,CURRENT_OVERLAY,LOCAL_SOURCE_SHA,LOCAL_GIT_SHA,OVERLAY_DIRS,OVERLAY_FILES
app=modal.App('syncopate-probability-batch')
SOURCES=('syncopate/infra/hardware_probe.py','tests/infra/test_hardware_probe.py',
 'syncopate/infra/residual_probe.py','tests/infra/test_residual_probe.py',
 'syncopate/infra/opd_contract_probe.py','tests/infra/test_opd_contract_probe.py',
 'syncopate/infra/communication_probe.py','tests/infra/test_communication_probe.py',
 'syncopate/infra/splitk_probe.py','tests/infra/test_splitk_probe.py','syncopate/infra/mechanism_contract.py','tests/infra/test_mechanism_contract.py',
 'syncopate/infra/gemm_cause_probe.py','tests/infra/test_gemm_cause_probe.py',
 'syncopate/infra/gdn_fastpath_probe.py','tests/infra/test_gdn_fastpath_probe.py',
 'syncopate/infra/lora_handoff_probe.py','tests/infra/test_lora_handoff_probe.py',
 'modal_app/gdn_fastpath_batch.py','syncopate/infra/gdn_boundary_probe.py','tests/infra/test_gdn_boundary_probe.py','modal_app/probability_batch.py','syncopate/infra/probe_run.py',
 'syncopate/infra/probability_probe.py','syncopate/infra/fsdp_probability_probe.py',
 'syncopate/infra/fp8_probe.py','tests/infra/test_fp8_probe.py',
 'syncopate/infra/batch_layer_probe.py','syncopate/pipeline/process_guard.py',
 'tests/infra/test_probe_run.py','tests/infra/test_probability_probe.py',
 'tests/infra/test_fsdp_probability_probe.py','tests/infra/test_batch_layer_probe.py')


def execute(run_id,phase,preflight='',runtime_image=None,dispatch_file=None):
    from syncopate.infra.probe_run import reserve_attempt
    from syncopate.train.source_snapshot import archive_source,verify_orchestrator
    from syncopate.pipeline.process_guard import run_guarded
    root=Path(CURRENT_OVERLAY)
    cpu_phase=phase=='cpu' or phase.endswith('_cpu')
    selected_image=runtime_image if runtime_image is not None else image
    exp={'cpu':'B01','fsdp':'B01','vllm':'B01','layers':'B03','fp8':'B04','gdn':'B05','gemm_cpu':'B07','gemm':'B07','fast_cpu':'B06','fast':'B06','lora_cpu':'B08','lora':'B08','splitk_cpu':'B07','splitk':'B07','residual_cpu':'B09','residual':'B09','residual_follow_cpu':'B09','residual_follow':'B09','opd_cpu':'B10','communication_cpu':'B11','communication':'B11','communication_focus_cpu':'B11','communication_focus':'B11'}[phase]
    out=reserve_attempt(Path('/vol/_audit/infra-probes'),exp,run_id)
    rec={'experiment':'infra-probes/'+exp,'run_id':run_id,'phase':phase,'evidence_path':str(out),
         'volume':'syncopate-home','modal_task_id':os.environ.get('MODAL_TASK_ID'),'git_sha':LOCAL_GIT_SHA,'overlay_sha256':LOCAL_SOURCE_SHA,
         'image_id':selected_image.object_id,'entry_sha256':verify_orchestrator(Path(__file__),root/'modal_app/probability_batch.py'),
         'sources':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in SOURCES},
         'lock_sha256':hashlib.sha256((root/'modal_app/stack/uv.lock').read_bytes()).hexdigest(),
         'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'ok':False,'steps':[]}
    if dispatch_file is not None:
        rec['dispatch_entry_sha256']=verify_orchestrator(Path(dispatch_file),root/'modal_app/gdn_fastpath_batch.py')
    rec['source_archive']=archive_source(root,OVERLAY_DIRS,OVERLAY_FILES,expected=LOCAL_SOURCE_SHA,output=out/'source.tar.gz')
    env={**os.environ,'PYTHONPATH':str(root),'HF_HUB_OFFLINE':'0' if cpu_phase else '1',
         'VLLM_WORKER_MULTIPROC_METHOD':'spawn','VLLM_CACHE_ROOT':str(out/'cache/vllm'),
         'FLASHINFER_WORKSPACE_BASE':str(out/'cache/flashinfer'),'TORCHINDUCTOR_CACHE_DIR':str(out/'cache/inductor')}
    if cpu_phase:
        commands=[[PY,'-m','pytest',*[p for p in SOURCES if p.startswith('tests/')],'-q','-r','a'],
                  [PY,'-m','syncopate.infra.probability_probe','cpu',str(out)],
                  [PY,'-m','syncopate.infra.fsdp_probability_probe','--cpu','--input',str(out/'input.json'),'--output',str(out)],
                  [PY,'-m','syncopate.infra.fp8_probe','--cpu',str(out)],
                  [PY,'-m','syncopate.infra.gdn_boundary_probe','--cpu','--out',str(out)]]
        if phase in {'residual_cpu','residual_follow_cpu','opd_cpu','communication_cpu','communication_focus_cpu'}:
            module={'residual_cpu':'residual_probe','residual_follow_cpu':'residual_probe','opd_cpu':'opd_contract_probe','communication_cpu':'communication_probe','communication_focus_cpu':'communication_probe'}[phase]
            commands=[[PY,'-m','pytest','tests/infra/test_'+module+'.py','tests/infra/test_mechanism_contract.py','-q','-r','a']]
            if phase in {'communication_cpu','communication_focus_cpu'}: commands[0].insert(-3,'tests/infra/test_hardware_probe.py')
            if phase in {'residual_cpu','residual_follow_cpu'}: commands += [[PY,'-m','syncopate.infra.probability_probe','cpu',str(out)]]
            commands += [[PY,'-m','syncopate.infra.'+module,'--cpu','--out',str(out)]+(['--input',str(out/'input.json')] if phase in {'residual_cpu','residual_follow_cpu'} else [])+(['--followup'] if phase=='residual_follow_cpu' else [])+(['--focused'] if phase=='communication_focus_cpu' else [])]
        elif phase!='cpu':
            module={'gemm_cpu':'gemm_cause_probe','fast_cpu':'gdn_fastpath_probe','lora_cpu':'lora_handoff_probe','splitk_cpu':'splitk_probe'}[phase]
            commands=commands[:2]+[[PY,'-m','syncopate.infra.'+module,'--cpu','--input',str(out/'input.json'),'--out',str(out)]]
            if phase=='lora_cpu': commands=commands[:2]+[[PY,'-m','syncopate.infra.lora_handoff_probe','cpu',str(out),str(out/'input.json')]]
    else:
        pre=Path(preflight)
        assert pre.is_relative_to('/vol/_audit/infra-probes') and pre.name.startswith('attempt-')
        prior=json.loads((pre/'run.json').read_text())
        from syncopate.infra.mechanism_contract import validate_preflight
        expected_cpu=phase+'_cpu' if phase in {'gemm','fast','lora','splitk','residual','residual_follow','communication','communication_focus'} else 'cpu'
        validate_preflight(prior,rec,expected_cpu)
        if phase not in {'communication','communication_focus'}:
            data=(pre/'input.json').read_bytes()
            assert hashlib.sha256(data).hexdigest()==prior['files']['input.json']
            (out/'input.json').write_bytes(data)
        if phase in {'communication','communication_focus'}:
            data=(pre/'communication-cpu.json').read_bytes()
            assert hashlib.sha256(data).hexdigest()==prior['files']['communication-cpu.json']
            (out/'communication-preflight.json').write_bytes(data)
        rec['cpu_preflight']=str(pre)
        commands=[[PY,'-m','syncopate.infra.probability_probe','verify',str(out),str(out/'input.json')],
                  ['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.used,driver_version','--format=csv'],
                  [PY,'-c',"import subprocess,json; r=subprocess.run(['nvidia-smi','topo','-m'],capture_output=True,text=True); print(json.dumps({'available':r.returncode==0,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}))"],
                  [PY,'-c',f"import torch; assert torch.cuda.device_count()=={2 if phase in {'fsdp','communication','communication_focus'} else 1}; assert all(torch.cuda.get_device_capability(i)==(10,0) and 'B200' in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))"]]
        if phase in {'communication','communication_focus'}:
            commands=commands[1:]+[[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','-m','syncopate.infra.communication_probe','--out',str(out),'--preflight',str(out/'communication-preflight.json')]+(['--focused'] if phase=='communication_focus' else [])]
        elif phase in {'residual','residual_follow'}:
            commands += [[PY,'-m','syncopate.infra.residual_probe','--input',str(out/'input.json'),'--out',str(out)]+(['--followup'] if phase=='residual_follow' else [])]
        elif phase in {'gemm','fast','lora','splitk'}:
            module={'gemm':'gemm_cause_probe','fast':'gdn_fastpath_probe','lora':'lora_handoff_probe','splitk':'splitk_probe'}[phase]
            if phase=='lora': commands += [[PY,'-m','syncopate.infra.lora_handoff_probe','gpu',str(out),str(pre)]]
            else: commands += [[PY,'-m','syncopate.infra.'+module,'--input',str(out/'input.json'),'--out',str(out)]]
        elif phase=='fsdp':
            commands += [[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                          '-m','syncopate.infra.fsdp_probability_probe','--input',str(out/'input.json'),'--output',str(out)]]
        elif phase=='gdn':
            commands += [[PY,'-m','syncopate.infra.gdn_boundary_probe','--input',str(out/'input.json'),'--out',str(out)]]
        elif phase=='layers':
            commands += [[PY,'-m','syncopate.infra.batch_layer_probe','--input',str(out/'input.json'),'--out',str(out)]]
        else:
            commands += [[PY,'-m','syncopate.infra.probability_probe',phase,str(out),str(out/'input.json')]]
    rec['commands']=commands
    def persist():
        temp=out/'run.json.tmp';temp.write_text(json.dumps(rec,indent=2));temp.replace(out/'run.json');vol.commit()
    persist();start=time.monotonic();limit=1050 if cpu_phase else 1650
    try:
        for i,cmd in enumerate(commands):
            remaining=limit-(time.monotonic()-start)
            if remaining<=0: raise TimeoutError('phase deadline exhausted')
            result=run_guarded(['bash','-c','exec "$@" >"$PROBE_LOG" 2>&1','probe',*cmd],
                              cwd=str(root),env={**env,'PROBE_LOG':str(out/f'step-{i}.log')},timeout=remaining)
            rec['steps'].append(result);persist()
            if result['rc']!=0: break
            if cpu_phase and i==0:
                log=(out/'step-0.log').read_text()
                import re
                assert re.search(r'\b[1-9][0-9]* passed\b',log) and not re.search(r'\b[1-9][0-9]* (skipped|failed|xfailed|error)',log)
        else:
            required={'cpu':['cpu-result.json','fsdp2-cpu.json','fp8-cpu.json','gdn-cpu.json'],
                      'fsdp':['fsdp2-rank0.json','fsdp2-rank1.json'],
                      'vllm':['vllm.json'],'fp8':['vllm.json'],'layers':['layers-result.json'],'gdn':['gdn-result.json'],
                      'gemm_cpu':['cpu-result.json','gemm-cpu.json'],'gemm':['gemm-result.json'],
                      'fast_cpu':['cpu-result.json','fastpath-cpu.json'],'fast':['fastpath-result.json'],
                      'splitk_cpu':['cpu-result.json','splitk-cpu.json'],'splitk':['splitk-result.json'],'lora_cpu':['cpu-result.json'],'lora':['handoff.json'],
                      'residual_cpu':['cpu-result.json','residual-cpu.json'],'residual':['residual-result.json'],
                      'residual_follow_cpu':['cpu-result.json','residual-cpu.json'],'residual_follow':['residual-result.json'],
                      'opd_cpu':['opd-cpu.json'],'communication_cpu':['communication-cpu.json'],'communication':['communication-result.json'],
                      'communication_focus_cpu':['communication-cpu.json'],'communication_focus':['communication-result.json']}[phase]
            for name in required:
                result=json.loads((out/name).read_text())
                assert result.get('ok') is True, f'missing successful terminal result: {name}'
                if not cpu_phase and phase not in {'gemm','fast','lora','splitk','residual','residual_follow','communication','communication_focus'}: assert len(result['records'])==6, 'incomplete repeated measurements'
            rec['validated_results']=required
            rec['ok']=True
    except BaseException as exc:
        rec['exception']={'type':type(exc).__name__,'message':str(exc)}
        raise
    finally:
        rec['seconds']=time.monotonic()-start
        # Logs may receive shutdown lines later: they are diagnostic, never result seals.
        rec['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()
                      if p.is_file() and p.suffix=='.json' and p.name!='run.json'}
        rec['log_seal_policy']='Logs are unsealed shutdown diagnostics; result JSON SHA is authoritative.'
        persist()
    return rec


@app.function(image=image,volumes={'/vol':vol},cpu=4,memory=16384,timeout=1200,retries=0,single_use_containers=True)
def cpu(run_id,phase='cpu'): return execute(run_id,phase)

@app.function(image=image,volumes={'/vol':vol},gpu='B200:2',cpu=16,memory=196608,timeout=1800,retries=0,single_use_containers=True)
def fsdp(run_id,preflight,phase='fsdp'):
    assert phase in {'fsdp','communication','communication_focus'}
    return execute(run_id,phase,preflight)

@app.function(image=image,volumes={'/vol':vol},gpu='B200',cpu=16,memory=131072,timeout=1800,retries=0,single_use_containers=True)
def single(run_id,phase,preflight):
    assert phase in {'vllm','layers','fp8','gdn','gemm','lora','splitk','residual','residual_follow'}
    return execute(run_id,phase,preflight)

@app.local_entrypoint()
def main(run_id:str,phase:str='cpu',preflight:str=''):
    if phase not in {'cpu','fsdp','vllm','layers','fp8','gdn','gemm_cpu','gemm','lora_cpu','lora','splitk_cpu','splitk','residual_cpu','residual','residual_follow_cpu','residual_follow','opd_cpu','communication_cpu','communication','communication_focus_cpu','communication_focus'}:
        raise ValueError('Unknown phase; rejected before GPU allocation')
    if phase=='cpu' or phase.endswith('_cpu'): result=cpu.remote(run_id,phase)
    elif phase in {'fsdp','communication','communication_focus'}: result=fsdp.remote(run_id,preflight,phase)
    else: result=single.remote(run_id,phase,preflight)
    print(json.dumps(result,indent=2))
    if not result['ok']: raise RuntimeError('Probe failed; exact evidence: '+result['evidence_path'])
