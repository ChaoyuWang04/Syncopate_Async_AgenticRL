"""B06: isolated actual GDN layer, native/torch/fast paths, forward+backward."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import statistics
from pathlib import Path
from syncopate.infra.gdn_boundary_probe import B03_SOURCE_SHA, tensor_metric
from syncopate.infra.probability_probe import MODEL, sha_file


PINNED_VERSIONS = {'torch':'2.13.0','transformers':'5.10.4','flash-linear-attention':'0.5.2','causal-conv1d':'1.7.0','vllm':'0.28.0','verl':'0.9.0'}
CAPTURE = Path('/vol/_audit/infra-probes/B05/batch3-gdn-01/attempt-266daa98c57143ab9b6e022642fa48f5/gdn-captures.pt')
CAPTURE_SHA = '24c0face5106870ac0001256d7773f694eb63243b4c478251b40b0e88bffbe23'


def validate_dependencies(d, *, cpu):
    if d.get('versions') != PINNED_VERSIONS:
        raise ValueError('frozen dependency versions changed')
    if d.get('source_sha256',{}).get('modeling') != B03_SOURCE_SHA:
        raise ValueError('frozen model source drift')
    if any(d.get('direct_imports',{}).get(p,{}).get('ok') is not True for p in ('causal_conv1d','fla.ops.gated_delta_rule')):
        raise ValueError('required direct dependency import failed')
    if cpu and d.get('cuda_initialized') is not False:
        raise ValueError('CPU probe initialized CUDA')


def verify_native_capture(layer,x,out):
    import torch
    if sha_file(CAPTURE)!=CAPTURE_SHA:
        raise ValueError('B05 capture identity changed')
    expected=torch.load(CAPTURE,map_location='cpu',weights_only=True)['batch1_output']
    original_conv=layer.causal_conv1d_fn
    try:
        layer.causal_conv1d_fn=None
        with torch.inference_mode(): actual=layer(x).detach().cpu()
    finally:
        layer.causal_conv1d_fn=original_conv
    result={'capture_path':str(CAPTURE),'capture_sha256':CAPTURE_SHA, 'native_vs_B05':tensor_metric(expected,actual)}
    (out/'fastpath-native-identity.json').write_text(json.dumps(result,indent=2))
    if not result['native_vs_B05']['equal']:
        raise RuntimeError('isolated native layer does not exactly reproduce B05; diagnose input/norm/core, do not widen identity threshold')
    return result


def validate_records(rows):
    if [r['arm'] for r in rows] != ['torch', 'native', 'fast']:
        raise ValueError('incomplete arms')
    for r in rows:
        if not r['comparisons'] or not all(c['relative_l2'] is not None and math.isfinite(c['relative_l2']) and c['relative_l2'] <= .03 for c in r['comparisons']):
            raise ValueError('numerics failed')
        if len(r['ms']) != 3 or not all(math.isfinite(x) and x > 0 for x in r['ms']):
            raise ValueError('missing timings')
    if rows[0]['hits'] != {'conv': 0, 'core': 0} or any(rows[2]['hits'][k] < 1 for k in ('conv', 'core')):
        raise ValueError('real kernel hit missing')


def diagnose(out):
    import torch
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m
    from transformers.utils import import_utils as u
    direct_imports={}
    for package in ['causal_conv1d','fla.ops.gated_delta_rule']:
        try:
            __import__(package)
            direct_imports[package]={'ok':True}
        except Exception as exc:
            direct_imports[package]={'ok':False,'error':repr(exc)}
    files = {}
    for name, module in [('modeling', m), ('imports', u)]:
        source = Path(inspect.getfile(module)).read_bytes()
        (out / (name+'.py')).write_bytes(source)
        files[name] = hashlib.sha256(source).hexdigest()
    return dict(direct_imports=direct_imports, source_sha256=files, cuda_available=torch.cuda.is_available(),
                cuda_initialized=torch.cuda.is_initialized(),
                versions={n: importlib.metadata.version(n) for n in PINNED_VERSIONS},
                functions={n: callable(getattr(m,n,None)) for n in ['causal_conv1d_fn','chunk_gated_delta_rule','FusedRMSNormGated']},
                fast_path_available=getattr(m,'is_fast_path_available',None),
                availability_source={n:inspect.getsource(getattr(u,n)) for n in ['is_causal_conv1d_available','is_flash_linear_attention_available']})


def load_layer(inputs):
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m
    config = AutoConfig.from_pretrained(MODEL,local_files_only=True).text_config
    config.dtype = torch.bfloat16
    layer = m.Qwen3_5MoeGatedDeltaNet(config,0).to('cuda',dtype=torch.bfloat16)
    index = json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    prefix = 'model.language_model.layers.0.linear_attn.'
    keys = [prefix+k for k in layer.state_dict()]
    embkey='model.language_model.embed_tokens.weight'
    normkey='model.language_model.layers.0.input_layernorm.weight'
    weights={}
    for key in keys+[embkey,normkey]:
        with safe_open(MODEL/index[key],framework='pt',device='cpu') as f:
            weights[key]=f.get_tensor(key)
    layer.load_state_dict({k:weights[prefix+k] for k in layer.state_dict()},strict=True)
    ids=torch.tensor(inputs['tokens'][:1],dtype=torch.long)
    x=torch.nn.functional.embedding(ids,weights[embkey]).to('cuda')
    norm=m.Qwen3_5MoeRMSNorm(config.hidden_size,eps=config.rms_norm_eps).to('cuda',dtype=torch.bfloat16)
    norm.load_state_dict({'weight':weights[normkey]})
    with torch.no_grad(): x=norm(x).detach()
    return layer,x,m


def measure(layer,x,m,out):
    import torch
    import copy
    original_norm=layer.norm
    fallback_norm=m.Qwen3_5MoeRMSNormGated(layer.head_v_dim,eps=layer.layer_norm_epsilon).to('cuda',dtype=torch.bfloat16)
    fallback_norm.load_state_dict(original_norm.state_dict())
    original_core=layer.chunk_gated_delta_rule
    if not callable(m.causal_conv1d_fn) or not callable(m.chunk_gated_delta_rule):
        raise RuntimeError('full fast dependencies unavailable')
    result={'ok':False,'records':[], 'scope':'actual layer0; 1x64 tokens; no cache; module forward/backward only'}
    baseline=None
    torch.manual_seed(17)
    cotangent=torch.randn_like(x)
    for arm in ['torch','native','fast']:
        layer.norm=copy.deepcopy(fallback_norm if arm=='torch' else original_norm)
        core=m.torch_chunk_gated_delta_rule if arm=='torch' else original_core
        conv=m.causal_conv1d_fn if arm=='fast' else None
        hits={'conv':0,'core':0}
        def counted(fn,key):
            def wrapped(*args,**kwargs):
                hits[key]+=1
                return fn(*args,**kwargs)
            return wrapped
        layer.chunk_gated_delta_rule=counted(core,'core') if core is m.chunk_gated_delta_rule else core
        layer.causal_conv1d_fn=counted(conv,'conv') if conv else None
        def step():
            layer.zero_grad(set_to_none=True)
            xx=x.detach().clone().requires_grad_(True)
            y=layer(xx)
            (y.float()*cotangent.float()).sum().backward()
            return y,xx.grad
        y,g=step()
        tensors={'output':y.detach().clone(),'input_gradient':g.detach().clone()}
        for name,p in layer.named_parameters():
            if p.grad is None: raise RuntimeError('missing gradient '+name)
            tensors['gradient:'+name]=p.grad.detach().clone()
        if baseline is None: baseline=tensors
        if tensors.keys()!=baseline.keys(): raise RuntimeError('gradient keys changed')
        comparisons=[dict(name=k,**tensor_metric(baseline[k],v)) for k,v in tensors.items()]
        observed_hits=dict(hits)
        repeated_y,repeated_g=step()
        noise={'output':tensor_metric(tensors['output'],repeated_y),'input_gradient':tensor_metric(tensors['input_gradient'],repeated_g)}
        layer.chunk_gated_delta_rule=core
        layer.causal_conv1d_fn=conv
        for _ in range(2): step()
        torch.cuda.synchronize()
        ms=[]
        for _ in range(3):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record(); step(); end.record(); end.synchronize()
            ms.append(start.elapsed_time(end))
        row=dict(arm=arm,hits=observed_hits,comparisons=comparisons,ms=ms,
                 core=core.__module__+'.'+core.__name__,norm=type(layer.norm).__name__,noise=noise)
        result['records'].append(row)
        (out/'fastpath-partial.json').write_text(json.dumps(result,indent=2))
    validate_records(result['records'])
    native,fast=result['records'][1:3]
    result['timing_summary']={'native_median_ms':statistics.median(native['ms']), 'fast_median_ms':statistics.median(fast['ms']), 'native_range_ms':[min(native['ms']),max(native['ms'])], 'fast_range_ms':[min(fast['ms']),max(fast['ms'])], 'fast_faster_than_entire_native_range':max(fast['ms'])<min(native['ms']), 'scope':'single allocation screening only'}
    result['ok']=True
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--cpu',action='store_true');p.add_argument('--input');p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    d=diagnose(out);(out/'fastpath-dependencies.json').write_text(json.dumps(d,indent=2))
    if args.cpu:
        d['ok']=False
        try:
            validate_dependencies(d,cpu=True)
            d['ok']=True
        except ValueError as exc:
            d['error']=str(exc)
            raise
        finally:
            (out/'fastpath-cpu.json').write_text(json.dumps(d,indent=2))
        return
    validate_dependencies(d,cpu=False)
    from syncopate.infra.batch_layer_probe import validate_input
    inputs=json.loads(Path(args.input).read_text());validate_input(inputs)
    layer,x,m=load_layer(inputs)
    identity=verify_native_capture(layer,x,out)
    result=measure(layer,x,m,out)
    result['native_identity']=identity
    result['dependencies']=d
    result['input_sha256']=sha_file(Path(args.input))
    result['config_sha256']=sha_file(MODEL/'config.json')
    result['index_sha256']=sha_file(MODEL/'model.safetensors.index.json')
    result['device']=__import__('torch').cuda.get_device_name(0)
    (out/'fastpath-result.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
