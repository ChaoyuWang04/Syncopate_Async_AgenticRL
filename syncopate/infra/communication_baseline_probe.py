"""B11 bounded native/eager/graph diagnostic. No training or speedup claim."""
from __future__ import annotations
import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import sysconfig
import time
import traceback
from syncopate.infra.communication_probe import validate_nccl_environment

NCCL_TESTS_SHA='b4d5beebca8a76cf01335f724d154b9b9d394d96'
OPS=('all_reduce','all_gather','reduce_scatter')
WINDOWS=3
TARGET_SECONDS=.1
MAX_ITERATIONS=4096


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def native_layout(op,requested_bytes):
    if op not in OPS or type(requested_bytes) is not int or requested_bytes<=0 or requested_bytes%2:
        raise ValueError('registered operation and positive even BF16 bytes required')
    count=requested_bytes//2
    block=count if op=='all_reduce' else (count//2)&~7
    actual=block*2*(1 if op=='all_reduce' else 2)
    if actual==0:raise ValueError('size rounds to zero')
    return dict(operation=op,requested_bytes=requested_bytes,actual_bytes=actual,
                count=block,block_bytes=block*2,dtype='bfloat16',world_size=2,
                compare_mode='in_place' if op=='all_reduce' else 'out_of_place')


def cases():
    result={}
    for op in OPS:
        for delta in (-2,0,2):
            row=native_layout(op,(16<<20)+delta);key=(op,row['actual_bytes'])
            if key not in result:result[key]={**row,'requested_total_bytes':[]}
            result[key]['requested_total_bytes'].append(row['requested_bytes'])
    return list(result.values())


def calibrate_iterations(seconds):
    if not math.isfinite(seconds) or seconds<=0:raise ValueError('positive finite pilot time required')
    return min(MAX_ITERATIONS,max(128,math.ceil(TARGET_SECONDS/seconds)))


def sample_summary(values):
    if len(values)!=WINDOWS or any(not math.isfinite(x) or x<=0 for x in values):
        raise ValueError('three positive finite windows required')
    return {'samples_seconds':values,'median_seconds':statistics.median(values),
            'sample_stdev_seconds':statistics.stdev(values),'max_min_ratio':max(values)/min(values),
            'within_allocation_stable':max(values)/min(values)<=1.1}


def parse_native(text):
    rows=[]
    for line in text.splitlines():
        fields=line.split()
        if not fields or not fields[0].isdigit():continue
        if len(fields)!=13:raise ValueError('unexpected nccl-tests data row: '+line)
        if fields[2]!='bfloat16':raise ValueError('native dtype mismatch')
        row={'actual_bytes':int(fields[0]),'count':int(fields[1]),'dtype':fields[2]}
        for name,start in [('out_of_place',5),('in_place',9)]:
            vals=[float(v) for v in fields[start:start+3]];wrong=int(fields[start+3])
            if any(not math.isfinite(v) or v<=0 for v in vals) or wrong!=0:raise ValueError('native numerical/time check failed')
            row[name]={'seconds':vals[0]/1e6,'algorithm_GB_s':vals[1],'bus_GB_s':vals[2],'wrong':wrong}
        rows.append(row)
    if not rows:raise ValueError('no native nccl-tests measurements')
    return rows


def native_command(build,op,size,iterations,windows):
    if op not in OPS or not 1<=iterations<=MAX_ITERATIONS or not 1<=windows<=WINDOWS:raise ValueError('unregistered native workload')
    return ['mpirun','--allow-run-as-root','--bind-to','none','-np','2',str(Path(build)/(op+'_perf')),
            '-b',str(size),'-e',str(size),'-g','1','-t','1','-d','bfloat16','-o','sum',
            '-n',str(iterations),'-w','20','-m','1','-N',str(windows),'-a','3','-c','1']


def run(command,out,name,env=None,timeout=120):
    start=time.time()
    with (out/(name+'.log')).open('w') as log:
        p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=timeout)
    write(out/(name+'-command.json'),{'argv':command,'returncode':p.returncode,'wall_seconds':time.time()-start})
    if p.returncode:raise RuntimeError(name+' failed; see log')
    return (out/(name+'.log')).read_text()


def nccl_install():
    import torch
    home=Path(sysconfig.get_paths()['purelib'])/'nvidia/nccl'
    lib=home/'lib/libnccl.so.2'
    if not lib.is_file() or not (home/'include/nccl.h').is_file():raise ValueError('exact torch wheel NCCL library/header absent')
    # Dynamic loader inspection binds the library used by torch, independently of NCCL_VERSION image metadata.
    ldd=subprocess.check_output(['ldd',str(Path(torch.__file__).parent/'lib/libtorch_cuda.so')],text=True)
    matches=re.findall(r'libnccl\.so\.2\s+=>\s+(\S+)',ldd)
    if len(matches)!=1 or Path(matches[0]).resolve()!=lib.resolve():raise ValueError('torch ldd NCCL does not match selected wheel')
    return {'home':str(home),'library':str(lib.resolve()),'library_sha256':sha(lib),'ldd_torch':ldd,'torch':torch.__version__}


def execution_environment(environment):
    """PMIx hash avoids fixed-address shared mappings unsupported by this runtime.

    This changes MPI bootstrap metadata storage, not NCCL payload transport.
    Used identically by the CPU MPI health check and both GPU subprocess paths.
    """
    if environment.get('PMIX_MCA_gds','hash')!='hash':
        raise ValueError('registered PMIx gds=hash required')
    return dict(environment,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PMIX_MCA_gds='hash')


def parse_mpi_health(text):
    rows=[dict(zip(('rank','size','sum'),map(int,match))) for match in
          re.findall(r'^MPI_HEALTH rank=(\d+) size=(\d+) sum=(\d+)$',text,re.MULTILINE)]
    expected=[{'rank':0,'size':2,'sum':3},{'rank':1,'size':2,'sum':3}]
    if sorted(rows,key=lambda row:row['rank'])!=expected:raise ValueError('MPI two-rank health check incomplete or wrong')
    return expected


def mpi_health(out,env):
    source=out/'mpi-health.c';binary=out/'mpi-health'
    source.write_text(r'''#include <mpi.h>
#include <stdio.h>
int main(int argc,char **argv) {
  int rank=-1,size=0,value=0,sum=0;
  if(MPI_Init(&argc,&argv)!=MPI_SUCCESS)return 1;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
  value=rank+1;
  if(MPI_Allreduce(&value,&sum,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD)!=MPI_SUCCESS)return 2;
  printf("MPI_HEALTH rank=%d size=%d sum=%d\n",rank,size,sum);fflush(stdout);
  if(MPI_Finalize()!=MPI_SUCCESS)return 3;
  return size==2 && sum==3 ? 0 : 4;
}
''')
    run(['mpicc',str(source),'-O2','-o',str(binary)],out,'mpi-health-build',env,timeout=60)
    text=run(['mpirun','--allow-run-as-root','--bind-to','none','-np','2',str(binary)],out,'mpi-health',env,timeout=60)
    return {'ok':True,'ranks':parse_mpi_health(text),'source_sha256':sha(source),
            'binary_sha256':sha(binary),'PMIX_MCA_gds':env['PMIX_MCA_gds']}


def cpu(out):
    validate_nccl_environment(os.environ,out)
    env=execution_environment(os.environ);health=mpi_health(out,env);api=direct_api_check()
    installed=nccl_install();source=out/'nccl-tests';build=out/'build'
    run(['git','clone','--filter=blob:none','https://github.com/NVIDIA/nccl-tests.git',str(source)],out,'clone',timeout=120)
    run(['git','-C',str(source),'checkout','--detach',NCCL_TESTS_SHA],out,'checkout')
    actual=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if actual!=NCCL_TESTS_SHA:raise ValueError('nccl-tests source SHA mismatch')
    linkhome=out/'nccl-link';(linkhome/'lib').mkdir(parents=True)
    (linkhome/'include').symlink_to(Path(installed['home'])/'include',target_is_directory=True)
    for name in ('libnccl.so','libnccl.so.2'):(linkhome/'lib'/name).symlink_to(installed['library'])
    run(['make','-C',str(source/'src'),'-j4','MPI=1','MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi',
         'CUDA_HOME=/usr/local/cuda','NCCL_HOME='+str(linkhome),'BUILDDIR='+str(build),
         'NVCC_GENCODE=-gencode=arch=compute_100,code=sm_100',*[str(build/(op+'_perf')) for op in OPS]],out,'build',timeout=900)
    binaries={}
    for op in OPS:
        path=build/(op+'_perf')
        if not path.is_file():raise ValueError('missing native binary: '+str(path))
        binaries[op]={'path':str(path),'sha256':sha(path)}
    return {'ok':True,'phase':'cpu','direct_api':api,'mpi_health':health,'nccl':installed,'nccl_tests_sha':actual,'binaries':binaries,'build':str(build),
            'cases':cases(),'windows':WINDOWS,'target_seconds':TARGET_SECONDS,'max_iterations':MAX_ITERATIONS,
            'probe_sha256':sha(__file__),'no_gpu_requested':True}


def verify_preflight(preflight):
    obj=json.loads((preflight/'communication-baseline-cpu.json').read_text())
    if obj.get('ok') is not True or obj['probe_sha256']!=sha(__file__) or obj['cases']!=cases():raise ValueError('preflight/source mismatch')
    if obj.get('mpi_health',{}).get('ok') is not True or obj['mpi_health'].get('PMIX_MCA_gds')!='hash':raise ValueError('MPI health preflight required')
    if sha(obj['nccl']['library'])!=obj['nccl']['library_sha256']:raise ValueError('NCCL library changed')
    for binary in obj['binaries'].values():
        if sha(binary['path'])!=binary['sha256']:raise ValueError('native binary changed')
    return obj


def finish_rank(torch,dist,graphs,out,result):
    """Release graph-owned NCCL resources before destroying their communicator.

    See pytorch/pytorch#115388. Caller transfers the last graph references into
    this list; publishing measurements is separate from successful teardown.
    """
    result={**result,'ok':False,'phase':'teardown_started'}
    path=out/f'rank-{result["rank"]}.json';write(path,result)
    torch.cuda.synchronize()
    graphs.clear()
    dist.destroy_process_group()
    result.update(ok=True,phase='complete');write(path,result)
    return result


def rank(out,preflight):
    import torch
    import torch.distributed as dist
    rank=int(os.environ['LOCAL_RANK']);torch.set_num_threads(1);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=timedelta(seconds=60),device_id=torch.device('cuda',rank))
    def sync():dist.barrier();torch.cuda.synchronize()
    def maxrank(value):
        t=torch.tensor(value,dtype=torch.float64,device='cuda');dist.all_reduce(t,op=dist.ReduceOp.MAX);return t.item()
    if tuple(torch.cuda.nccl.version())!=(2,29,7):raise ValueError('registered NCCL 2.29.7 required')
    samples=[]
    # NCCL work submitted by c10d is ordered back onto the current stream.
    for case in cases():
        op=case['operation'];n=case['count'];device='cuda'
        x=torch.empty(n*(2 if op=='reduce_scatter' else 1),device=device,dtype=torch.bfloat16)
        y=x if op=='all_reduce' else torch.empty(n*(2 if op=='all_gather' else 1),device=device,dtype=torch.bfloat16)
        pattern=(torch.arange(x.numel(),device=device)%8).bfloat16()
        def reset(nonzero):x.copy_(pattern+rank if nonzero else torch.zeros_like(x))
        def call():
            if op=='all_reduce':dist.all_reduce(x)
            elif op=='all_gather':dist.all_gather_into_tensor(y,x)
            else:dist.reduce_scatter_tensor(y,x)
        def expected():
            if op=='all_reduce':return pattern*2+1
            if op=='all_gather':return torch.cat([pattern,pattern+1])
            return (pattern*2+1).chunk(2)[rank]
        def correctness():
            reset(True);sync();call();torch.cuda.synchronize()
            target=expected();ok=torch.equal(y,target) and torch.isfinite(y).all().item()
            if not ok:raise ValueError('nonzero exact correctness failed')
            y.view(-1)[0]+=1
            negative=not torch.equal(y,target)
            if not negative:raise ValueError('corrupt-element negative control failed')
            reset(False);sync()
            return {'all_elements_exact':True,'finite':True,'corrupt_element_rejected':True}
        before=correctness();reset(False)
        for _ in range(20):call()
        sync()
        def batch(fn,iterations):
            sync();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();t=time.perf_counter()
            for _ in range(iterations):fn()
            end.record();torch.cuda.synchronize()
            return {'host_seconds':time.perf_counter()-t,'cuda_seconds':start.elapsed_time(end)/1000}
        pilot=batch(call,32);iterations=calibrate_iterations(maxrank(pilot['host_seconds'])/32)
        graph=torch.cuda.CUDAGraph();capture=torch.cuda.Stream();capture.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture):
            for _ in range(20):call()
        capture.synchronize();sync()
        with torch.cuda.graph(graph,stream=capture):
            for _ in range(iterations):call()
        torch.cuda.current_stream().wait_stream(capture);sync();graph.replay();torch.cuda.synchronize()
        controls=[];windows=[]
        for window in range(WINDOWS):
            order=('eager','graph') if window%2==0 else ('graph','eager')
            measured={}
            for mode in order:
                reset(False);sync()
                measured[mode]=batch(call,iterations) if mode=='eager' else batch(graph.replay,1)
                if not torch.equal(y,torch.zeros_like(y)):raise ValueError('zero fixed-point timing check failed')
            controls.append(batch(lambda:None,iterations))
            windows.append({'window':window,'order':order,**measured})
        after=correctness()
        # A separate one-call graph carries rank-varying values. Replaying the
        # long in-place SUM graph on nonzero input would overflow BF16.
        reset(True);sync();check_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(check_graph):call()
        check_graph.replay();torch.cuda.synchronize()
        if not torch.equal(y,expected()):raise ValueError('nonzero graph exact correctness failed')
        reset(False);sync()
        trace=out/f'trace-rank-{rank}-{op}-{case["actual_bytes"]}.json'
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            call();torch.cuda.synchronize()
        prof.export_chrome_trace(str(trace))
        samples.append({**case,'iterations':iterations,'pilot':pilot,'windows':windows,'zero_work_controls':controls,
                        'before':before,'after':after,'nonzero_graph_exact':True,'diagnostic_trace':str(trace),'pointer_mod_16':{'input':x.data_ptr()%16,'output':y.data_ptr()%16}})
        write(out/f'rank-{rank}.json',{'ok':False,'rank':rank,'cases':samples})
    maps=Path('/proc/self/maps').read_text();loaded=sorted({line.split()[-1] for line in maps.splitlines() if 'libnccl.so' in line})
    if not loaded or any(sha(path)!=preflight['nccl']['library_sha256'] for path in loaded):raise ValueError('runtime NCCL library mismatch')
    result={'ok':True,'rank':rank,'cases':samples,'loaded_nccl':loaded,'nccl_version':torch.cuda.nccl.version(),
            'torch':torch.__version__,'device':str(torch.cuda.get_device_properties(rank)),
            'pid':os.getpid(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'torch_threads':torch.get_num_threads()}
    remaining_graphs=[graph,check_graph]
    del graph,check_graph
    return finish_rank(torch,dist,remaining_graphs,out,result)


def torch_command(out,preflight,direct=False):
    return [sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',
            '-m','syncopate.infra.communication_baseline_probe','--mode','direct-rank' if direct else 'rank',
            '--out',str(out),'--preflight',str(preflight)]


def gpu(out,preflight):
    validate_nccl_environment(os.environ,out);pre=verify_preflight(preflight)
    env=execution_environment(os.environ)
    env['LD_LIBRARY_PATH']=str(Path(pre['nccl']['library']).parent)+':'+env.get('LD_LIBRARY_PATH','')
    topology=run(['nvidia-smi','--query-gpu=uuid,name,pci.bus_id,memory.total','--format=csv'],out,'devices')
    if topology.count('B200')!=2:raise ValueError('exactly two B200 devices required')
    native=[]
    for idx,case in enumerate(cases()):
        op=case['operation'];size=case['actual_bytes']
        ldd=run(['ldd',str(Path(pre['build'])/(op+'_perf'))],out,f'ldd-{idx}',env)
        matches=re.findall(r'libnccl\.so\.2\s+=>\s+(\S+)',ldd)
        if len(matches)!=1 or sha(matches[0])!=pre['nccl']['library_sha256']:raise ValueError('native NCCL mismatch')
        pilot=parse_native(run(native_command(pre['build'],op,size,32,1),out,f'native-pilot-{idx}',env))
        iterations=calibrate_iterations(pilot[0][case['compare_mode']]['seconds'])
        rows=parse_native(run(native_command(pre['build'],op,size,iterations,WINDOWS),out,f'native-{idx}',env))
        if len(rows)!=WINDOWS or any(row['actual_bytes']!=size or row['count']!=case['count'] for row in rows):raise ValueError('native row/count mismatch')
        native.append({**case,'pilot':pilot,'iterations':iterations,'windows':rows})
        write(out/'native-results.json',native)
    run(torch_command(out,preflight),out,'torch',env,timeout=360)
    ranks=[json.loads((out/f'rank-{r}.json').read_text()) for r in (0,1)]
    if not all(r['ok'] is True for r in ranks):raise ValueError('incomplete torch ranks')
    summaries=[]
    for idx,case in enumerate(cases()):
        left,right=[r['cases'][idx] for r in ranks]
        if left['iterations']!=right['iterations']:raise ValueError('rank iterations disagree')
        n=left['iterations'];row={**case,'native':sample_summary([w[case['compare_mode']]['seconds'] for w in native[idx]['windows']])}
        for mode in ('eager','graph'):
            row[mode]={metric:sample_summary([max(left['windows'][w][mode][metric],right['windows'][w][mode][metric])/n for w in range(WINDOWS)]) for metric in ('host_seconds','cuda_seconds')}
        controls=[max(left['zero_work_controls'][w]['host_seconds'],right['zero_work_controls'][w]['host_seconds'])/n for w in range(WINDOWS)]
        row['zero_work']=sample_summary(controls)
        row['control_fraction_of_eager']=row['zero_work']['median_seconds']/row['eager']['host_seconds']['median_seconds']
        row['control_below_five_percent']=row['control_fraction_of_eager']<=.05
        row['eager_graph_paired_host_difference_seconds']=[max(left['windows'][w]['eager']['host_seconds'],right['windows'][w]['eager']['host_seconds'])/n-max(left['windows'][w]['graph']['host_seconds'],right['windows'][w]['graph']['host_seconds'])/n for w in range(WINDOWS)]
        summaries.append(row)
    # TRACE is enabled only after every timed arm has finished.
    diagnostics=[]
    for op in OPS:
        diagnostic_env=dict(env,NCCL_DEBUG='TRACE',NCCL_DEBUG_SUBSYS='INIT,GRAPH,COLL',
                            NCCL_DEBUG_FILE=str(out/(op+'-nccl-%h-%p.log')))
        rows=parse_native(run(native_command(pre['build'],op,16<<20,1,1),out,'diagnostic-'+op,diagnostic_env))
        diagnostics.append({'operation':op,'rows':rows})
    return {'ok':True,'phase':'gpu','summaries':summaries,'diagnostics':diagnostics,'native':native,'ranks':ranks,'devices':topology,
            'source_sha256':sha(__file__),'preflight':str(preflight),'environment':{k:v for k,v in env.items() if k.startswith(('NCCL_','OMP_','MKL_','PMIX_'))},
            'performance_baseline':False,'scope':'one same-pair diagnostic; native runs precede torch; no causal speedup claim',
            'native_timing_caveat':'native in-place SUM is not reset per timed call; correctness is separate. Torch timed SUM uses stable zeros; nonzero pre/post exact checks separate.'}


GRAPH_ARMS=('native_eager','native_graph','native_graph_mixing_off')
GRAPH_ITERATIONS=1024


def native_graph_schedule():
    return [list(GRAPH_ARMS[i:]+GRAPH_ARMS[:i]) for i in range(WINDOWS)]


def native_graph_command(build,op,arm):
    if arm not in GRAPH_ARMS:raise ValueError('registered native graph arm required')
    return native_command(build,op,16<<20,GRAPH_ITERATIONS,1)+['-G','0' if arm=='native_eager' else '1']


def native_graph_environment(environment,arm):
    if arm not in GRAPH_ARMS:raise ValueError('registered native graph arm required')
    validate_nccl_environment(environment)
    env=execution_environment(environment)
    env.update(NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,ENV')
    if arm=='native_graph_mixing_off':env['NCCL_GRAPH_MIXING_SUPPORT']='0'
    return env


def read_native_diagnostics(out,prefix):
    """Read only this exact invocation's two rank logs, never a nearest match."""
    if not re.fullmatch(r'window-[0-2]-(?:native_eager|native_graph|native_graph_mixing_off)-(?:all_reduce|all_gather|reduce_scatter)-nccl-',prefix):
        raise ValueError('unregistered diagnostic prefix')
    paths=sorted(path for path in out.iterdir() if path.name.startswith(prefix) and path.name.endswith('.log'))
    if len(paths)!=2:raise ValueError('exactly two native NCCL rank logs required')
    files=[];texts=[]
    for path in paths:
        if not path.is_file() or path.resolve().parent!=out.resolve():raise ValueError('diagnostic file outside owned output directory')
        text=path.read_text();ranks={int(rank) for rank in re.findall(r'\brank (\d+) nranks 2\b',text)}
        if len(ranks)!=1 or not ranks<={0,1}:raise ValueError('native diagnostic rank identity missing or ambiguous')
        files.append({'path':str(path),'rank':next(iter(ranks)),'sha256':sha(path)});texts.append(text)
    if {file['rank'] for file in files}!={0,1}:raise ValueError('native diagnostics do not cover both ranks')
    return {'files':files,'text':'\n'.join(texts)}


def verify_native_graph_log(log,op,arm,diagnostic_log=""):
    rows=parse_native(log);expected=native_layout(op,16<<20)
    if len(rows)!=1 or rows[0]['actual_bytes']!=expected['actual_bytes'] or rows[0]['count']!=expected['count']:
        raise ValueError('native graph count/row mismatch')
    graph_flag=0 if arm=='native_eager' else 1
    if not re.search(r'graph:\s*'+str(graph_flag)+r'\b',log):raise ValueError('native graph flag not consumed')
    mixing_lines=[line for line in diagnostic_log.splitlines() if 'NCCL_GRAPH_MIXING_SUPPORT' in line]
    if arm=='native_graph_mixing_off' and not any(re.search(r'(?:to |[= ])0\b',line) for line in mixing_lines):
        raise ValueError('NCCL mixing override consumption missing')
    return rows


def native_graph(out,preflight):
    validate_nccl_environment(os.environ,out);pre=verify_preflight(preflight)
    devices=run(['nvidia-smi','--query-gpu=uuid,name,pci.bus_id,memory.total','--format=csv'],out,'devices')
    if devices.count('B200')!=2:raise ValueError('exactly two B200 devices required')
    base=execution_environment(os.environ)
    base['LD_LIBRARY_PATH']=str(Path(pre['nccl']['library']).parent)+':'+base.get('LD_LIBRARY_PATH','')
    for op in OPS:
        ldd=run(['ldd',str(Path(pre['build'])/(op+'_perf'))],out,'ldd-'+op,base)
        matches=re.findall(r'libnccl\.so\.2\s+=>\s+(\S+)',ldd)
        if len(matches)!=1 or sha(matches[0])!=pre['nccl']['library_sha256']:raise ValueError('native NCCL mismatch')
    records=[]
    for window,order in enumerate(native_graph_schedule()):
        for position,arm in enumerate(order):
            env=native_graph_environment(base,arm)
            for op in OPS:
                command=native_graph_command(pre['build'],op,arm)
                prefix=f'window-{window}-{arm}-{op}-nccl-'
                call_env=dict(env,NCCL_DEBUG_FILE=str(out/(prefix+'%h-%p.log')))
                log=run(command,out,f'window-{window}-{arm}-{op}',call_env)
                diagnostics=read_native_diagnostics(out,prefix)
                rows=verify_native_graph_log(log,op,arm,diagnostics['text']);expected=native_layout(op,16<<20)
                graph_flag=0 if arm=='native_eager' else 1
                mixing_lines=[line for line in diagnostics['text'].splitlines() if 'NCCL_GRAPH_MIXING_SUPPORT' in line]
                records.append({**expected,'window':window,'arm':arm,'position':position,'iterations':GRAPH_ITERATIONS,
                                'graph_launches':graph_flag,'rows':rows,'mixing_evidence':mixing_lines,'diagnostic_files':diagnostics['files'],
                                'environment':{k:v for k,v in call_env.items() if k.startswith(('NCCL_','OMP_','MKL_','PMIX_'))}})
                write(out/'native-graph-windows.json',records)
    summaries=[]
    for op in OPS:
        mode=native_layout(op,16<<20)['compare_mode']
        arm_samples={arm:[next(r for r in records if r['operation']==op and r['arm']==arm and r['window']==w)['rows'][0][mode]['seconds'] for w in range(WINDOWS)] for arm in GRAPH_ARMS}
        row={'operation':op,'compare_mode':mode,'actual_bytes':16<<20,
             'arms':{arm:sample_summary(values) for arm,values in arm_samples.items()}}
        row['paired_graph_minus_eager_seconds']=[g-e for g,e in zip(arm_samples['native_graph'],arm_samples['native_eager'])]
        row['paired_mixing_off_minus_default_seconds']=[off-on for off,on in zip(arm_samples['native_graph_mixing_off'],arm_samples['native_graph'])]
        row['graph_over_eager']=row['arms']['native_graph']['median_seconds']/row['arms']['native_eager']['median_seconds']
        row['mixing_off_over_graph']=row['arms']['native_graph_mixing_off']['median_seconds']/row['arms']['native_graph']['median_seconds']
        summaries.append(row)
    return {'ok':True,'phase':'native-graph','cases':records,'summaries':summaries,'devices':devices,
            'preflight':str(preflight),'nccl':pre['nccl'],'binaries':pre['binaries'],'source_sha256':sha(__file__),
            'schedule':native_graph_schedule(),'iterations':GRAPH_ITERATIONS,'performance_baseline':False,
            'scope':'one allocation confirmation of existing NCCL graph-mixing mechanism; no overlapping calls, no project default change'}


def direct_api_check():
    import inspect
    import torch
    import torch.cuda.nccl as nccl
    import cuda.bindings.runtime
    required={'all_reduce':{'inputs','outputs','op','streams','comms'},'init_rank':{'num_ranks','uid','rank'},'unique_id':set()}
    for name,parameters in required.items():
        if set(inspect.signature(getattr(nccl,name)).parameters)!=parameters:raise ValueError('direct API signature differs: '+name)
    if not callable(getattr(torch.cuda.CUDAGraph,'get_graph_data',None)):raise ValueError('graph metadata API unavailable')
    return {'ok':True,'graph_metadata_signature':str(inspect.signature(torch.cuda.CUDAGraph.get_graph_data)),'graph_python_source_sha256':sha(inspect.getfile(torch.cuda.CUDAGraph)),'nccl_python_source_sha256':sha(nccl.__file__),'signatures':{name:str(inspect.signature(getattr(nccl,name))) for name in required},
            'direct_communicator_cleanup':'public capsule destructor is a no-op in pinned 2.13; only process/context exit reclaims it'}


def direct_all_reduce(nccl,tensor,stream,comm):
    nccl.all_reduce([tensor],outputs=[tensor],op=nccl.SUM,streams=[stream],comms=[comm])


def graph_summary(data):
    from collections import Counter
    nodes=data.get('nodes',[])
    types=Counter(str(node['node_type']) for node in nodes)
    if not nodes or not any('kernel' in key.lower() for key in types):raise ValueError('graph has no kernel nodes')
    return {'node_count':len(nodes),'node_types':dict(types),'kernel_names':dict(Counter(node['kernel_name'] for node in nodes if node.get('kernel_name'))),
            'kernel_node_count':sum(value for key,value in types.items() if 'kernel' in key.lower())}


def direct_rank(out,pre):
    import torch
    import torch.distributed as dist
    import torch.cuda.nccl as nccl
    rank=int(os.environ['LOCAL_RANK']);torch.set_num_threads(1);torch.cuda.set_device(rank)
    run(['nvidia-smi','--query-gpu=uuid,driver_version','--format=csv,noheader'],out,f'direct-rank-{rank}-driver')
    dist.init_process_group('gloo',timeout=timedelta(seconds=60))
    pg=dist.new_group([0,1],backend='nccl',timeout=timedelta(seconds=60))
    uid=[nccl.unique_id() if rank==0 else None];dist.broadcast_object_list(uid,src=0)
    direct_comm=nccl.init_rank(2,uid[0],rank)
    x=torch.zeros((16<<20)//2,device='cuda',dtype=torch.bfloat16)
    pattern=(torch.arange(x.numel(),device='cuda')%8).bfloat16();expected=pattern*2+1
    def sync():torch.cuda.synchronize();dist.barrier();torch.cuda.synchronize()
    def call(arm):
        if arm=='process_group':dist.all_reduce(x,group=pg)
        else:direct_all_reduce(nccl,x,torch.cuda.current_stream(),direct_comm)
    def check(arm):
        x.copy_(pattern+rank);sync()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):call(arm)
        g.replay();torch.cuda.synchronize()
        if not torch.equal(x,expected) or not torch.isfinite(x).all().item():raise ValueError('nonzero graph correctness failed')
        x[0]+=1
        if torch.equal(x,expected):raise ValueError('negative control failed')
        del g
        x.zero_();sync()
        return {'nonzero_graph_exact':True,'finite':True,'corrupt_element_rejected':True}
    arms=('process_group','direct');records=[];retained=[]
    for iterations in (128,1024):
        graphs={};checks={};structures={}
        for arm in arms:
            for _ in range(20):call(arm)
            sync();checks[arm]={'before':check(arm)}
            graph=torch.cuda.CUDAGraph(keep_graph=True);graph.enable_debug_mode()
            with torch.cuda.graph(graph):
                for _ in range(iterations):call(arm)
            graph.instantiate();graph.replay();sync()
            try:
                data=graph.get_graph_data();summary=graph_summary(data)
                graph_path=out/f'direct-rank-{rank}-{arm}-{iterations}-graph.json';write(graph_path,data)
                structures[arm]={**summary,'path':str(graph_path),'sha256':sha(graph_path),'metadata_available':True}
            except RuntimeError as error:
                if 'get_graph_data requires' not in str(error):raise
                graph_path=out/f'direct-rank-{rank}-{arm}-{iterations}-graph.dot'
                graph.debug_dump(str(graph_path))
                if not graph_path.is_file() or not graph_path.stat().st_size:raise ValueError('graph DOT missing or empty')
                structures[arm]={'metadata_available':False,'reason':str(error),'path':str(graph_path),'sha256':sha(graph_path)}
            graphs[arm]=graph;retained.append(graph)
        windows=[]
        for window in range(WINDOWS):
            measurements={};order=arms if window%2==0 else arms[::-1]
            for arm in order:
                x.zero_();sync();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();begin=time.perf_counter();graphs[arm].replay();end.record();torch.cuda.synchronize()
                measurements[arm]={'host_seconds':time.perf_counter()-begin,'cuda_seconds':start.elapsed_time(end)/1000}
                if not torch.equal(x,torch.zeros_like(x)):raise ValueError('timed graph zero fixed point failed')
            windows.append({'window':window,'order':order,**measurements})
        for arm in arms:checks[arm]['after']=check(arm)
        # All measurement windows for this N precede its independent replay trace.
        traces={}
        for arm in arms:
            x.zero_();sync();path=out/f'direct-rank-{rank}-{arm}-{iterations}-trace.json'
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                graphs[arm].replay();torch.cuda.synchronize()
            prof.export_chrome_trace(str(path));traces[arm]=str(path)
        records.append({'iterations':iterations,'actual_bytes':16<<20,'dtype':'bfloat16','windows':windows,'checks':checks,'graphs':structures,'traces':traces,
                        'input_output_same_pointer':True,'pointer_mod_16':x.data_ptr()%16})
        write(out/f'rank-{rank}.json',{'ok':False,'phase':'measuring','rank':rank,'cases':records})
        graphs.clear()
    loaded=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libnccl.so' in line})
    if not loaded or any(sha(path)!=pre['nccl']['library_sha256'] for path in loaded):raise ValueError('direct/PG NCCL library mismatch')
    result={'ok':False,'phase':'teardown_started','rank':rank,'cases':records,'loaded_nccl':loaded,'nccl_version':nccl.version(),
            'pid':os.getpid(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
            'direct_cleanup':'no-op public destructor; requires normal process and container exit, not explicit communicator teardown'}
    write(out/f'rank-{rank}.json',result)
    torch.cuda.synchronize();del graph;retained.clear();dist.barrier()
    direct_comm=None
    dist.destroy_process_group(pg);dist.destroy_process_group()
    result.update(ok=True,phase='complete');write(out/f'rank-{rank}.json',result)


def torch_direct(out,preflight):
    validate_nccl_environment(os.environ,out);pre=verify_preflight(preflight);api=direct_api_check()
    env=execution_environment(os.environ)
    env['LD_LIBRARY_PATH']=str(Path(pre['nccl']['library']).parent)+':'+env.get('LD_LIBRARY_PATH','')
    devices=run(['nvidia-smi','--query-gpu=uuid,name,pci.bus_id,memory.total','--format=csv'],out,'devices')
    if devices.count('B200')!=2:raise ValueError('exactly two B200 devices required')
    run(torch_command(out,preflight,direct=True),out,'torch-direct',env,timeout=360)
    ranks=[json.loads((out/f'rank-{rank}.json').read_text()) for rank in (0,1)]
    if not all(r.get('ok') is True and r.get('phase')=='complete' and len(r['cases'])==2 for r in ranks):raise ValueError('direct ranks incomplete')
    summaries=[]
    for idx,iterations in enumerate((128,1024)):
        a,b=[r['cases'][idx] for r in ranks]
        if a['iterations']!=iterations or b['iterations']!=iterations:raise ValueError('direct rank workload mismatch')
        summary={'iterations':iterations,'actual_bytes':16<<20}
        for arm in ('process_group','direct'):
            summary[arm]={metric:sample_summary([max(a['windows'][w][arm][metric],b['windows'][w][arm][metric])/iterations for w in range(WINDOWS)]) for metric in ('host_seconds','cuda_seconds')}
        summary['process_group_over_direct']=summary['process_group']['host_seconds']['median_seconds']/summary['direct']['host_seconds']['median_seconds']
        summaries.append(summary)
    return {'ok':True,'phase':'torch-direct','ranks':ranks,'summaries':summaries,'devices':devices,'api':api,'preflight':str(preflight),
            'source_sha256':sha(__file__),'performance_baseline':False,
            'scope':'one aligned AR, same buffer/context, two communicator wrappers; direct API is only suitable here for single-use processes; container exit required'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['cpu','gpu','rank','native-graph','torch-direct','direct-rank'],required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--preflight',type=Path)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    name='communication-baseline-cpu.json' if args.mode=='cpu' else 'communication-baseline-result.json'
    try:
        if args.mode!='cpu' and args.preflight is None:raise ValueError('--preflight required')
        if args.mode=='rank':rank(args.out,verify_preflight(args.preflight));return
        if args.mode=='direct-rank':direct_rank(args.out,verify_preflight(args.preflight));return
        result=cpu(args.out) if args.mode=='cpu' else (native_graph(args.out,args.preflight) if args.mode=='native-graph' else (torch_direct(args.out,args.preflight) if args.mode=='torch-direct' else gpu(args.out,args.preflight)))
        write(args.out/name,result)
    except Exception:
        if args.mode not in ('rank','direct-rank'):write(args.out/name,{'ok':False,'error':traceback.format_exc()})
        raise

if __name__=='__main__':main()
