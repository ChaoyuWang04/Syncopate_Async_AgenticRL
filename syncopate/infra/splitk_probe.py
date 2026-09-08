"""B07 fixed-M64 cuBLASLt split-K intervention; compile host ABI bridge only."""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from syncopate.infra.gemm_cause_probe import (
    CAPTURE, CAPTURE_SHA, TOKEN_SHA, REVISION, detailed_metric, kernel_index, write)
from syncopate.infra.probability_probe import MODEL, sha_file, verify_model
from syncopate.infra.batch_layer_probe import validate_input

# All CUDA structs/enumerators stay on the compiled C++ side. No guessed Python ABI.
BRIDGE = r'''
#include <cublasLt.h>
#include <cstdint>
#include <cstring>
struct Plan {
  cublasLtHandle_t handle{};
  cublasLtMatmulDesc_t operation{};
  cublasLtMatrixLayout_t a{}, b{}, c{};
  cublasLtMatmulPreference_t pref{};
  cublasLtMatmulAlgo_t algo{};
};
extern "C" void destroy_plan(void* raw) {
  auto p=static_cast<Plan*>(raw); if (!p) return;
  if(p->pref) cublasLtMatmulPreferenceDestroy(p->pref);
  if(p->a) cublasLtMatrixLayoutDestroy(p->a);
  if(p->b) cublasLtMatrixLayoutDestroy(p->b);
  if(p->c) cublasLtMatrixLayoutDestroy(p->c);
  if(p->operation) cublasLtMatmulDescDestroy(p->operation);
  if(p->handle) cublasLtDestroy(p->handle);
  delete p;
}
extern "C" size_t library_version() { return cublasLtGetVersion(); }
extern "C" int header_contract(int* out) {
  out[0]=CUBLASLT_REDUCTION_SCHEME_NONE;
  out[1]=CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK;
  out[2]=CUBLASLT_ALGO_CONFIG_SPLITK_NUM;
  out[3]=CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME;
  out[4]=sizeof(cublasLtMatmulAlgo_t);
  out[5]=CUBLAS_COMPUTE_32F;
  out[6]=CUDA_R_16BF;
  return 0;
}
#define CHECK(call) do { int status=(call); if(status){ destroy_plan(p); return status; } } while(0)
extern "C" int make_plan(int m, int n, int k, int mode, size_t workspace, void** out, int* attrs) {
  auto p=new Plan();
  CHECK(cublasLtCreate(&p->handle));
  CHECK(cublasLtMatmulDescCreate(&p->operation,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t ta=CUBLAS_OP_T, tb=CUBLAS_OP_N;
  CHECK(cublasLtMatmulDescSetAttribute(p->operation,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
  CHECK(cublasLtMatmulDescSetAttribute(p->operation,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
  // D(row M,N) = X(row M,K) * W(row N,K)^T, represented column-major D(N,M).
  CHECK(cublasLtMatrixLayoutCreate(&p->a,CUDA_R_16BF,k,n,k));
  CHECK(cublasLtMatrixLayoutCreate(&p->b,CUDA_R_16BF,k,m,k));
  CHECK(cublasLtMatrixLayoutCreate(&p->c,CUDA_R_16BF,n,m,n));
  CHECK(cublasLtMatmulPreferenceCreate(&p->pref));
  CHECK(cublasLtMatmulPreferenceSetAttribute(p->pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace)));
  if(mode==1) {
    uint32_t mask=CUBLASLT_REDUCTION_SCHEME_NONE;
    CHECK(cublasLtMatmulPreferenceSetAttribute(p->pref,CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,&mask,sizeof(mask)));
  }
  cublasLtMatmulHeuristicResult_t candidates[32]{};
  int count=0;
  CHECK(cublasLtMatmulAlgoGetHeuristic(p->handle,p->operation,p->a,p->b,p->c,p->c,p->pref,32,candidates,&count));
  int selected=-1;
  for(int i=0;i<count;i++) if(candidates[i].state==CUBLAS_STATUS_SUCCESS) { selected=i; break; }
  if(selected<0){destroy_plan(p);return -100;}
  p->algo=candidates[selected].algo;
  cublasLtMatmulAlgoConfigAttributes_t keys[]={CUBLASLT_ALGO_CONFIG_ID,CUBLASLT_ALGO_CONFIG_TILE_ID,
      CUBLASLT_ALGO_CONFIG_SPLITK_NUM,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
      CUBLASLT_ALGO_CONFIG_STAGES_ID,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION};
  for(int i=0;i<7;i++){size_t written=0;CHECK(cublasLtMatmulAlgoConfigGetAttribute(&p->algo,keys[i],attrs+8+i,sizeof(int),&written));
    if(written!=sizeof(int)){destroy_plan(p);return -102;}}
  if(mode==2) {
    uint32_t splits=1, reduction=CUBLASLT_REDUCTION_SCHEME_NONE;
    CHECK(cublasLtMatmulAlgoConfigSetAttribute(&p->algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&splits,sizeof(splits)));
    CHECK(cublasLtMatmulAlgoConfigSetAttribute(&p->algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&reduction,sizeof(reduction)));
  }
  cublasLtMatmulHeuristicResult_t checked{};
  CHECK(cublasLtMatmulAlgoCheck(p->handle,p->operation,p->a,p->b,p->c,p->c,&p->algo,&checked));
  if(checked.state!=CUBLAS_STATUS_SUCCESS || checked.workspaceSize>workspace){destroy_plan(p);return -101;}

  for(int i=0;i<7;i++) {size_t written=0;CHECK(cublasLtMatmulAlgoConfigGetAttribute(&p->algo,keys[i],attrs+i,sizeof(int),&written));
    if(written!=sizeof(int)){destroy_plan(p);return -102;}}
  attrs[7]=count;
  if(mode && (attrs[2]>1 || attrs[3]!=CUBLASLT_REDUCTION_SCHEME_NONE)){destroy_plan(p);return -103;}
  *out=p;return 0;
}
extern "C" int execute_plan(void* raw, const void* w, const void* x, void* y, void* workspace, size_t bytes, void* stream) {
  auto p=static_cast<Plan*>(raw);float alpha=1.f,beta=0.f;
  return cublasLtMatmul(p->handle,p->operation,&alpha,w,p->a,x,p->b,&beta,y,p->c,y,p->c,
                       &p->algo,workspace,bytes,static_cast<cudaStream_t>(stream));
}
'''


def tensor_digest_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def tensor_hash(tensor):
    import torch
    return tensor_digest_bytes(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())


def validate_disabled(attributes, kernels):
    if attributes['split_k'] > 1 or attributes['reduction'] != 0 or not kernels:
        raise ValueError('split-K disabling not proved by algorithm and CUDA trace')
    if any('splitk' in item['name'].lower() for item in kernels):
        raise ValueError('split-K kernel still executed')


def validate_same_algorithm(baseline, original, final):
    keys = ('id', 'tile', 'split_k', 'reduction', 'stages', 'swizzle', 'custom')
    if any(original[k] != baseline[k] for k in keys):
        raise ValueError('reselected heuristic differs from unrestricted algorithm')
    if any(final[k] != original[k] for k in keys if k not in ('split_k', 'reduction')):
        raise ValueError('nonintervened algorithm attribute changed')
    if final['split_k'] != 1 or final['reduction'] != 0:
        raise ValueError('requested split-K intervention not installed')


def causal_status(rows):
    supported = {r['arm']: r for r in rows if r.get('supported')}
    baseline = supported.get('unrestricted')
    edited = supported.get('no_splitk_same_algorithm')
    if edited:
        if not baseline: raise ValueError('same-algorithm arm has no unrestricted baseline')
        validate_same_algorithm(baseline['algorithm'], edited['original_heuristic_algorithm'], edited['algorithm'])
    diagnostic_ok = any(name.startswith('no_splitk') for name in supported)
    attribution_ok = bool(baseline and baseline.get('equals_torch_M64') and edited
                          and baseline['algorithm']['split_k'] > 1)
    return {'diagnostic_ok': diagnostic_ok, 'torch_attribution_ok': attribution_ok,
            'same_algorithm_gate': bool(edited),
            'attribution_scope': 'requires exact Torch M64 baseline, original split-K and unchanged nonintervened algorithm attributes'}


def compile_bridge(output):
    import torch
    candidates = [Path(sys.prefix)/'lib', Path('/usr/local/cuda'), Path('/usr/local/cuda/targets/x86_64-linux'), Path('/usr/local/cuda-13.0/targets/x86_64-linux')]
    import site
    for root in site.getsitepackages():
        candidates += [Path(root)/'nvidia'/name for name in ('cu13','cublas','cuda_runtime','cuda_nvcc','cuda_cccl')]
    includes = sorted({p/'include' for p in candidates if (p/'include').is_dir()})
    headers = [p/'cublasLt.h' for p in includes if (p/'cublasLt.h').is_file()]
    libraries = sorted({p for root in candidates for directory in ('lib','lib64') for p in (root/directory).glob('libcublasLt.so*')})
    if not headers or not libraries: raise RuntimeError('installed cuBLASLt header/library missing')
    # Prefer pip CUDA library, matching torch dependencies, over system toolkit.
    loaded = sorted({Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines()
                     if 'libcublasLt.so' in line})
    if len(loaded) != 1: raise RuntimeError('expected exactly one torch-loaded cuBLASLt library')
    lib = loaded[0]
    header = Path('/usr/local/cuda-13.0/targets/x86_64-linux/include/cublasLt.h')
    if header not in headers: raise RuntimeError('registered CUDA13 header path unavailable')
    if sha_file(header) != '8be43fbf48625b98b8562d7df394e6af0d7ee1c05beeb5333577d5e33182fa3e':
        raise RuntimeError('cuBLASLt header differs from registered CUDA13 header')
    includes = [header.parent] + [p for p in includes if p != header.parent]
    cpp=output/'splitk-bridge.cpp'; so=output/'splitk-bridge.so'
    cpp.write_text(BRIDGE)
    compiler=shutil.which('g++') or shutil.which('c++')
    if not compiler: raise RuntimeError('host C++ compiler missing')
    command=[compiler,'-std=c++17','-shared','-fPIC','-O2',str(cpp),'-o',str(so),
             *[flag for path in includes for flag in ('-I',str(path))],str(lib),f'-Wl,-rpath,{lib.parent}']
    completed=subprocess.run(command,capture_output=True,text=True,timeout=120)
    (output/'splitk-build.log').write_text(completed.stdout+completed.stderr)
    if completed.returncode: raise RuntimeError('host bridge compile failed; see splitk-build.log')
    api=ctypes.CDLL(str(so))
    api.library_version.restype=ctypes.c_size_t
    api.header_contract.argtypes=[ctypes.POINTER(ctypes.c_int)]
    api.make_plan.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_int)]
    api.execute_plan.argtypes=[ctypes.c_void_p]*5+[ctypes.c_size_t,ctypes.c_void_p]
    api.destroy_plan.argtypes=[ctypes.c_void_p]
    values=(ctypes.c_int*7)();api.header_contract(values)
    if values[0]!=0: raise RuntimeError('NONE reduction enum changed')
    metadata={'header':str(header),'header_sha256':sha_file(header),'library':str(lib),
              'library_version':api.library_version(),'header_contract':list(values),
              'bridge_sha256':sha_file(cpp),'compile_command':command,
              'cuda_initialized':torch.cuda.is_initialized(), 'torch_version':torch.__version__,
              'torch_git_version':torch.version.git_version, 'cuda_version':torch.version.cuda}
    write(output/'splitk-api.json',metadata)
    return api,metadata


def run(input_path,output):
    import torch
    from safetensors import safe_open
    from syncopate.infra.gdn_boundary_probe import PROJECTION
    inputs=json.loads(input_path.read_text());validate_input(inputs);verify_model(inputs)
    if inputs['tokens_sha256']!=TOKEN_SHA or inputs['revision']!=REVISION or sha_file(CAPTURE)!=CAPTURE_SHA:
        raise ValueError('B05 witness identity changed')
    api,metadata=compile_bridge(output)
    captures=torch.load(CAPTURE,map_location='cpu',weights_only=True)
    index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    key=PROJECTION+'.weight'
    with safe_open(str(MODEL/index[key]),framework='pt',device='cpu') as file: weight=file.get_tensor(key).cuda()
    x=captures['batch1_input'].cuda().contiguous(); paired=captures['batch2_input'].cuda().contiguous()
    if tuple(weight.shape)!=(2048,4096) or weight.dtype!=torch.bfloat16: raise RuntimeError('weight contract changed')
    with torch.inference_mode():
        single=torch.nn.functional.linear(x,weight)
        batch=torch.nn.functional.linear(paired,weight)[:1]
        reference=torch.nn.functional.linear(x.double(),weight.double()).cpu()
    if not torch.equal(single.cpu(),captures['batch1_output']) or not torch.equal(batch.cpu(),captures['batch2_output'][:1]):
        raise RuntimeError('torch replay differs from B05 witness')
    size=32*1024**2
    workspace=torch.empty(size,dtype=torch.uint8,device='cuda')
    result={'ok':False,'api':metadata,'records':[], 'M':64,'N':2048,'K':4096,'workspace_bytes':size,
            'torch_single_sha256':tensor_hash(single),'torch_batch_target_sha256':tensor_hash(batch),
            'input_sha256':tensor_hash(x),'weight_sha256':tensor_hash(weight),
            'capture_sha256':CAPTURE_SHA,'tokens_sha256':TOKEN_SHA,'revision':REVISION}
    tensors={'torch_single':single.cpu(),'torch_batch_target':batch.cpu(),'reference_fp64':reference}
    from torch.profiler import profile,ProfilerActivity
    for mode,name in enumerate(('unrestricted','no_splitk_preference','no_splitk_same_algorithm')):
        handle=ctypes.c_void_p(); attributes=(ctypes.c_int*15)()
        status=api.make_plan(64,2048,4096,mode,size,ctypes.byref(handle),attributes)
        if status:
            result['records'].append({'arm':name,'supported':False,'status':status})
            write(output/'splitk-partial.json',result);continue
        attrs=dict(zip(('id','tile','split_k','reduction','stages','swizzle','custom','candidate_count'),list(attributes)[:8]))
        base_attrs=dict(zip(('id','tile','split_k','reduction','stages','swizzle','custom'),list(attributes)[8:]))
        if mode == 2:
            baseline = next((r for r in result['records'] if r['arm'] == 'unrestricted' and r.get('supported')), None)
            if baseline is None:
                api.destroy_plan(handle)
                raise RuntimeError('same-algorithm intervention missing unrestricted baseline')
            try:
                validate_same_algorithm(baseline['algorithm'], base_attrs, attrs)
            except Exception:
                api.destroy_plan(handle)
                raise
        y=torch.empty_like(single)
        def execute():
            code=api.execute_plan(handle,weight.data_ptr(),x.data_ptr(),y.data_ptr(),workspace.data_ptr(),size,torch.cuda.current_stream().cuda_stream)
            if code: raise RuntimeError(f'cuBLASLt execute failed: {code}')
        try:
            values=[]
            for repeat in range(3):
                execute();torch.cuda.synchronize();values.append(y.cpu().clone())
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
                execute();torch.cuda.synchronize()
            trace=output/f'splitk-{name}.trace.json';prof.export_chrome_trace(str(trace))
            kernels=kernel_index(json.loads(trace.read_text()))
            if mode:validate_disabled(attrs,kernels)
            if not all(torch.equal(v,values[0]) for v in values) or not torch.equal(y.cpu(),values[0]):
                raise RuntimeError('repeat/profile changes result')
            tensors[name]=values[0]
            result['records'].append({'arm':name,'supported':True,'algorithm':attrs,'original_heuristic_algorithm':base_attrs,'kernels':kernels,
                'trace':str(trace),'trace_sha256':sha_file(trace),'output_sha256':tensor_hash(values[0]),
                'equals_torch_M64':torch.equal(values[0],single.cpu()),'equals_torch_M128':torch.equal(values[0],batch.cpu()),
                'vs_torch_M64':detailed_metric(single,values[0]),'vs_torch_M128':detailed_metric(batch,values[0]),
                'vs_fp64':detailed_metric(reference,values[0]),'vs_fp64_rounded':detailed_metric(reference.bfloat16(),values[0])})
            write(output/'splitk-partial.json',result)
        finally:api.destroy_plan(handle)
    path=output/'splitk-outputs.pt';torch.save(tensors,path)
    result['tensors']={'path':str(path),'sha256':sha_file(path),'bytes':path.stat().st_size}
    result['cross_equal']={a+'__'+b:torch.equal(x,y) for a,x in tensors.items() for b,y in tensors.items() if a<b}
    result.update(causal_status(result['records']))
    result['ok']=result['diagnostic_ok']
    result['scope']='fixed-M64 library intervention; unrestricted must match actual TorchM64 before attributing Torch selection'
    write(output/'splitk-result.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cpu',action='store_true');p.add_argument('--input',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if any(args.out.glob('splitk-*')):raise RuntimeError('fresh output directory required')
    try:
        if args.cpu:
            import torch
            _,metadata=compile_bridge(args.out)
            if torch.cuda.is_initialized():raise RuntimeError('CPU gate initialized CUDA')
            write(args.out/'splitk-cpu.json',{'ok':True,**metadata})
        else:
            if args.input is None:p.error('--input required')
            run(args.input,args.out)
    except Exception as exc:
        write(args.out/'splitk-failure.json',{'ok':False,'error':str(exc),'type':type(exc).__name__});raise


if __name__=='__main__':main()
