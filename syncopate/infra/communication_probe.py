"""B11: two-B200 default-NCCL size/alignment diagnostic, no training claims."""
from __future__ import annotations
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time
import traceback
from syncopate.infra import hardware_probe as h

OPERATIONS=('all_reduce','all_gather_into_tensor','reduce_scatter_tensor')
MODEL=Path('/vol/models/Qwen3.6-35B-A3B')
REVISION='995ad96eacd98c81ed38be0c5b274b04031597b0'
TIMEOUT=360

def sha(data):return hashlib.sha256(data).hexdigest()

def layout(operation, block_bytes):
    if operation not in OPERATIONS or type(block_bytes) is not int or block_bytes<=0 or block_bytes%2:
        raise ValueError('registered operation and positive BF16-aligned bytes required')
    n=block_bytes//2
    return {'operation':operation,'block_bytes':block_bytes,'dtype':'bfloat16','world_size':2,
            'input_numel':n*(2 if operation=='reduce_scatter_tensor' else 1),
            'output_numel':n*(2 if operation=='all_gather_into_tensor' else 1),
            'algorithm_bytes':block_bytes*(1 if operation=='all_reduce' else 2),
            'block_alignment_mod_16':block_bytes%16}

def bandwidth(operation, block_bytes, seconds):
    h._positive(seconds,'seconds')
    size=layout(operation,block_bytes)['algorithm_bytes']; factor=1 if operation=='all_reduce' else .5
    return {'algorithm_bytes':size,'algorithm_GB_s':size/seconds/1e9,
            'bus_GB_s':size*factor/seconds/1e9,'bus_factor':factor}

def message_matrix(shape, rank):
    if len(shape)!=2 or any(type(v) is not int or v<=0 for v in shape) or rank!=8:
        raise ValueError('positive projection shape and registered LoRA rank 8 required')
    sources=[('o_proj_parameter',2*shape[0]*shape[1]),('o_proj_two_rank_shard',shape[0]*shape[1]),
             ('lora_A',2*rank*shape[1]),('lora_B',2*shape[0]*rank)]
    sizes={}
    for label,size in sources:
        if size%2:raise ValueError('two-rank BF16 shard requires even element count')
        for delta in (-2,0,2):
            sizes.setdefault(size+delta,[]).append({'source':label,'delta_bytes':delta,'source_bytes':size})
    return [{'block_bytes':size,'sources':origins,'workload_frequency':'unmeasured'}
            for size,origins in sorted(sizes.items())]

def model_identity(model):
    config_raw=(model/'config.json').read_bytes(); index_raw=(model/'model.safetensors.index.json').read_bytes()
    config=json.loads(config_raw); cfg=config.get('text_config',config); mapping=json.loads(index_raw)['weight_map']
    full=[i for i,t in enumerate(cfg['layer_types']) if t=='full_attention']
    if not full:raise ValueError('no full attention layer')
    key=f'model.language_model.layers.{full[-1]}.self_attn.o_proj.weight'
    if key not in mapping:raise ValueError('exact parameter absent: '+key)
    shard=model/mapping[key]
    if shard.parent!=model:raise ValueError('invalid shard path')
    with shard.open('rb') as stream:
        length=struct.unpack('<Q',stream.read(8))[0]
        if length>64*1024*1024:raise ValueError('unexpected header size')
        raw=stream.read(length)
    header=json.loads(raw); entry=header[key]; shape=entry['shape']
    if entry['dtype']!='BF16' or shape!=[cfg['hidden_size'],cfg['num_attention_heads']*cfg['head_dim']]:
        raise ValueError('projection shape/dtype disagrees with config')
    meta=model/'.cache/huggingface/download/config.json.metadata'
    revision=meta.read_text().splitlines()[0]
    if revision!=REVISION:raise ValueError('model revision mismatch')
    return {'model_path':str(model),'revision':revision,'config_sha256':sha(config_raw),'index_sha256':sha(index_raw),
            'shard':mapping[key],'header_sha256':sha(raw),'parameter':key,'shape':shape,'dtype':entry['dtype']}

def plan(identity,focused=False):
    if focused and 2*identity['shape'][0]*identity['shape'][1] != 16<<20:
        raise ValueError('focused 16MiB must equal actual selected parameter bytes')
    return {'experiment':'infra-probes/B11','identity':identity,'focused_scope':focused,'messages':focused_messages() if focused else message_matrix(identity['shape'],8),
            'operations':list(OPERATIONS),'warmups_per_window':h.WARMUPS,'windows':h.WINDOWS,
            'repetitions_per_window':h.REPETITIONS,'gpu':'B200:2','process_timeout_seconds':TIMEOUT,
            'distributed_timeout_seconds':60,'independent_allocations':1,'performance_baseline':False,
            'correctness':'every output element exact, finite; corrupt-element negative control must fail',
            'timing':'slowest rank host dispatch+synchronize primary; CUDA event secondary; resets/barriers excluded',
            'nccl_tuning':'defaults; no algorithm/protocol/P2P overrides',
            'screening_threshold':'neighbor latency > 1.2 * aligned and difference > 3 * max window stdev; diagnostic flag only'}

def screen_neighbors(cases,messages):
    lookup={(c['operation'],c['block_bytes']):c['timing']['seconds_per_call_summary'] for c in cases}
    result=[]
    for message in messages:
        for source in message['sources']:
            if source['delta_bytes']==0:continue
            for op in OPERATIONS:
                if (op,message['block_bytes']) not in lookup:continue
                base=lookup[(op,source['source_bytes'])];neighbor=lookup[(op,message['block_bytes'])]
                difference=neighbor['median']-base['median']
                noise=3*max(base['sample_stdev'],neighbor['sample_stdev'])
                result.append({'operation':op,'source':source['source'],'block_bytes':message['block_bytes'],
                    'reference_bytes':source['source_bytes'],'ratio':neighbor['median']/base['median'],
                    'difference_seconds':difference,'noise_threshold_seconds':noise,
                    'flag':neighbor['median']>1.2*base['median'] and difference>noise})
    return result

def aggregate(ranks):
    if len(ranks)!=2 or {r['rank'] for r in ranks}!={0,1} or not all(r.get('ok') is True for r in ranks):
        raise ValueError('exactly two complete successful ranks required')
    a,b=sorted(ranks,key=lambda r:r['rank'])
    if a['plan']!=b['plan'] or len(a['cases'])!=len(b['cases']):raise ValueError('rank plans differ')
    expected=[(op,m['block_bytes']) for m in a['plan']['messages'] for op in OPERATIONS]
    for r in (a,b):
        if [(c['operation'],c['block_bytes']) for c in r['cases']]!=expected:raise ValueError('incomplete matrix')
    cases=[]
    for left,right in zip(a['cases'],b['cases']):
        size=left['block_bytes']; op=left['operation']
        timing=h._aggregate_complete_measurement([left,right],exact_shape=[layout(op,size)['output_numel']])
        seconds=timing['seconds_per_call_summary']['median']
        cases.append({'operation':op,'block_bytes':size,'timing':timing,**bandwidth(op,size,seconds)})
    return {'ok':True,'plan':a['plan'],'cases':cases,'ranks':ranks,'performance_baseline':False,
            'neighbor_screen':screen_neighbors(cases,a['plan']['messages']),
            'independent_allocations':1,'scope':'one-allocation screening; no upstream bug or speedup claim'}

def focused_messages():
    center=16<<20
    return [{'block_bytes':center+d,'sources':[{'source':'o_proj_parameter','delta_bytes':d,'source_bytes':center}],
             'workload_frequency':'unmeasured'} for d in (-2,0,2)]

def focused_schedule():
    sizes=[m['block_bytes'] for m in focused_messages()]
    return [sizes[i:]+sizes[:i] for i in range(3)]

def nvml_topology(uuid_rows):
    result={}
    try:
        import pynvml as n
        n.nvmlInit()
        handles=[n.nvmlDeviceGetHandleByUUID(row['uuid']) for row in uuid_rows]
        def query(name,*args):
            try:return {'ok':True,'value':str(getattr(n,name)(*args))}
            except Exception as exc:return {'ok':False,'error':str(exc)}
        result['common_ancestor']=query('nvmlDeviceGetTopologyCommonAncestor',*handles)
        result['p2p']={str(cap):query('nvmlDeviceGetP2PStatus',*handles,cap) for cap in range(5)}
        result['devices']=[]
        for handle,row in zip(handles,uuid_rows):
            device={'uuid':row['uuid'],'pci':query('nvmlDeviceGetPciInfo',handle),'links':[]}
            for link in range(18):
                device['links'].append({'link':link,'state':query('nvmlDeviceGetNvLinkState',handle,link),
                    'remote_pci':query('nvmlDeviceGetNvLinkRemotePciInfo',handle,link)})
            result['devices'].append(device)
        n.nvmlShutdown()
    except Exception as exc:result['error']=str(exc)
    return result

def focused_measure(torch,dist,rank,out):
    buffers={};records={};traces=[]
    for m in focused_messages():
        for op in OPERATIONS:
            size=m['block_bytes'];key=(op,size)
            c=h.collective_tensors(torch,op,size*2,rank,device=f'cuda:{rank}')
            for field in ('input','source','output','expected'):c[field]=c[field].bfloat16()
            c['layout']=layout(op,size);buffers[key]=c
            records[key]={'rank':rank,'operation':op,'block_bytes':size,'windows':[],'postchecks':[],
                'input_pointer_mod_16':c['input'].data_ptr()%16,'output_pointer_mod_16':c['output'].data_ptr()%16}
    for window,sizes in enumerate(focused_schedule()):
        for size in sizes:
            for op in OPERATIONS:
                c=buffers[(op,size)];record=records[(op,size)]
                for _ in range(h.WARMUPS):
                    h._reset_collective(c);h._barrier(torch,dist,rank);h._call_collective(dist,c);torch.cuda.synchronize()
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                wall=[];cuda=[]
                for _ in range(h.REPETITIONS):
                    h._reset_collective(c);h._barrier(torch,dist,rank);start.record();t=time.perf_counter()
                    h._call_collective(dist,c);end.record();torch.cuda.synchronize()
                    wall.append(time.perf_counter()-t);cuda.append(start.elapsed_time(end)/1000)
                check=h._check_collective(c);h._require_ok(check,'focused post-window')
                record['windows'].append({'window':window,'wall_seconds':wall,'cuda_seconds':cuda})
                record['postchecks'].append(check)
    # CUPTI diagnostics are after all timed rounds and consume no timing samples.
    for key,c in buffers.items():
        h._reset_collective(c);h._barrier(torch,dist,rank)
        op,size=key;path=out/f'trace-rank-{rank}-{op}-{size}.json'
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            h._call_collective(dist,c);torch.cuda.synchronize()
        prof.export_chrome_trace(str(path));check=h._check_collective(c);h._require_ok(check,'profile correctness')
        traces.append({'operation':op,'block_bytes':size,'path':str(path),'check':check,
            'kernels':h.kernel_names_from_trace(json.loads(path.read_text()))})
    return list(records.values()),traces

def validate_nccl_environment(environment,out=None):
    """Distinguish image labels from behavior switches; neither proves loaded NCCL."""
    metadata_keys={'NCCL_VERSION'}
    diagnostic_keys={'NCCL_DEBUG','NCCL_DEBUG_SUBSYS','NCCL_DEBUG_FILE','NCCL_TOPO_DUMP_FILE'}
    for name in ('NCCL_TOPO_DUMP_FILE','NCCL_DEBUG_FILE'):
        if name in environment and (out is None or Path(environment[name]).resolve().parent!=Path(out).resolve()):
            raise ValueError(name+' must target own output directory')
    nccl={k:v for k,v in environment.items() if k.startswith('NCCL_')}
    forbidden={k:v for k,v in nccl.items() if k not in metadata_keys|diagnostic_keys}
    if forbidden:raise ValueError('NCCL tuning environment must be absent: '+repr(forbidden))
    return {'metadata':{k:v for k,v in nccl.items() if k in metadata_keys},
            'diagnostics':{k:v for k,v in nccl.items() if k in diagnostic_keys},
            'metadata_is_runtime_version':False}

def cpu(out,model,focused=False):
    environment=validate_nccl_environment(os.environ,out)
    import torch
    identity=model_identity(model); registered=plan(identity,focused)
    checks=[]
    for op in OPERATIONS:
        cases=[h.collective_tensors(torch,op,12,r,device='cpu') for r in range(2)]
        for c in cases:
            for key in ('input','source','output','expected'):c[key]=c[key].bfloat16()
        if op=='all_gather_into_tensor':actual=[torch.cat([c['input'] for c in cases])]*2
        else:
            total=cases[0]['input']+cases[1]['input']
            actual=[total]*2 if op=='all_reduce' else list(total.chunk(2))
        for value,c in zip(actual,cases):
            check=h.exact_tensor_check(value,c['expected']); assert check['ok']; checks.append(check)
            bad=value.clone();bad[0]+=1;assert not h.exact_tensor_check(bad,c['expected'])['ok']
    import torch.distributed as dist
    if not dist.is_nccl_available():raise ValueError('target CPU environment has no NCCL backend')
    result={'ok':True,'plan':registered,'nccl_environment':environment,'checks':checks,'negative_control':True,'nccl_executed':False,
            'torch':torch.__version__,'cuda_build':torch.version.cuda}
    h.write_json_once(out/'communication-cpu.json',result)
    return result

def gpu(out,model,preflight,focused=False):
    import torch
    import torch.distributed as dist
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=2 or rank!=local:raise ValueError('requires single-node torchrun world 2')
    # Hard exit remains effective even if a collective never returns to Python.
    import threading
    def expired():
        h.write_json_once(out/f'communication-timeout-{rank}.json',{'ok':False,'timeout_seconds':TIMEOUT})
        os._exit(124)
    timer=threading.Timer(TIMEOUT,expired);timer.daemon=True;timer.start()
    registered=plan(model_identity(model),focused)
    prior=json.loads(preflight.read_text())
    if not prior.get('ok') or prior['plan']!=registered:raise ValueError('CPU preregistration missing or identity changed')
    validate_nccl_environment(os.environ,out)
    if focused:
        os.environ.update({'NCCL_DEBUG':'INFO','NCCL_DEBUG_SUBSYS':'INIT,GRAPH,NET,TUNING',
            'NCCL_DEBUG_FILE':str(out/f'nccl-rank-{rank}.log'),'NCCL_TOPO_DUMP_FILE':str(out/f'nccl-topo-{rank}.xml')})
    environment=validate_nccl_environment(os.environ,out)
    torch.cuda.set_device(local);cap=h._gpu_capabilities(torch,rank)
    topology={}
    for name,args in [('topo',['topo','-m']),('links',['nvlink','--status']),('uuids',['--query-gpu=index,uuid,pci.bus_id','--format=csv,noheader'])]:
        command=subprocess.run(['nvidia-smi',*args],capture_output=True,text=True,timeout=15)
        topology[name]={'returncode':command.returncode,'stdout':command.stdout,'stderr':command.stderr}
    for config_path in (Path('/etc/nccl.conf'),Path.home()/'.nccl.conf'):
        if config_path.exists() and any(line.strip() and not line.lstrip().startswith('#') for line in config_path.read_text().splitlines()):
            raise ValueError('nonempty NCCL configuration would change default baseline: '+str(config_path))
    dist.init_process_group('nccl',timeout=timedelta(seconds=60))
    result={'rank':rank,'ok':False,'nccl_environment':environment,'plan':registered,'capabilities':cap,'topology':topology,'cases':[]}
    try:
        if focused:
            result['nvml']=nvml_topology(cap['driver']['uuid_driver_rows'])
            result['cases'],result['diagnostic_trace']=focused_measure(torch,dist,rank,out)
        for message in ([] if focused else registered['messages']):
            size=message['block_bytes']
            for op in OPERATIONS:
                c=h.collective_tensors(torch,op,size*2,rank,device=f'cuda:{rank}')
                for key in ('input','source','output','expected'):c[key]=c[key].bfloat16()
                c['layout']=layout(op,size)
                h._reset_collective(c);h._call_collective(dist,c);torch.cuda.synchronize()
                pre=h._check_collective(c);h._require_ok(pre,'precheck')
                windows,post=h._time_windows(torch,dist,rank,lambda:h._call_collective(dist,c),
                    reset=lambda:h._reset_collective(c),check=lambda:h._check_collective(c))
                result['cases'].append({'rank':rank,'operation':op,'block_bytes':size,'precheck':pre,
                    'input_pointer_mod_16':c['input'].data_ptr()%16,'output_pointer_mod_16':c['output'].data_ptr()%16,
                    'windows':windows,'postchecks':post})
                del c
        if focused:
            result['diagnostic_files']={name:{'path':value,'exists':Path(value).exists(),
                'bytes':Path(value).stat().st_size if Path(value).exists() else None}
                for name,value in environment['diagnostics'].items() if name.endswith('_FILE')}
        result['ok']=True
    except Exception:
        result['error']=traceback.format_exc();raise
    finally:
        h.write_json_once(out/f'communication-rank-{rank}.json',result)
    dist.barrier(device_ids=[rank])
    if rank==0:
        ranks=[json.loads((out/f'communication-rank-{r}.json').read_text()) for r in range(2)]
        h.write_json_once(out/'communication-result.json',aggregate(ranks))
    dist.destroy_process_group();timer.cancel()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cpu',action='store_true');parser.add_argument('--focused',action='store_true')
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--model',type=Path,default=MODEL);parser.add_argument('--preflight',type=Path)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.cpu:cpu(args.out,args.model,args.focused)
    else:gpu(args.out,args.model,args.preflight or args.out/'communication-cpu.json',args.focused)

if __name__=='__main__':main()
