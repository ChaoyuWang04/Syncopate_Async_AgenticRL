"""B09: reproduce B07's 322 residual coordinates and intervene at real layer0 boundaries."""
from __future__ import annotations
import argparse
import inspect
import json
from pathlib import Path
from syncopate.infra.gemm_cause_probe import (
    CAPTURE, CAPTURE_SHA, TOKEN_SHA, REVISION, detailed_metric, write, kernel_index)
from syncopate.infra.gdn_boundary_probe import PROJECTION, B03_SOURCE_SHA
from syncopate.infra.batch_layer_probe import validate_input
from syncopate.infra.probability_probe import MODEL, verify_model, sha_file

PREFIX = 'model.language_model.layers.0'
NAMES = ['input_layernorm', 'linear_attn.out_proj', 'linear_attn',
         'post_attention_layernorm', 'mlp.shared_expert.gate_proj',
         'mlp.shared_expert.up_proj', 'mlp.shared_expert.down_proj',
         'mlp.shared_expert', 'mlp.gate', 'mlp.experts',
         'mlp.shared_expert_gate', 'mlp', '']


def tree(value, fn):
    import torch
    if isinstance(value, torch.Tensor): return fn(value)
    if isinstance(value, tuple): return tuple(tree(v, fn) for v in value)
    if isinstance(value, list): return [tree(v, fn) for v in value]
    if isinstance(value, dict): return {k: tree(v, fn) for k,v in value.items()}
    return value


def target(value, tokens=64):
    return tree(value, lambda t: t[:1 if t.ndim >= 3 else tokens].detach().cpu().clone() if t.ndim else t.detach().cpu().clone())


def substitute(value, reference):
    import torch
    if isinstance(value, torch.Tensor):
        if value.ndim != reference.ndim or value.shape[1:] != reference.shape[1:] or len(reference)>len(value):
            raise ValueError('substitution shape mismatch')
        result=value.clone(); result[:len(reference)]=reference.to(value.device)
        if not torch.equal(result[:len(reference)].cpu(),reference.cpu()):raise ValueError('replacement missed')
        if not torch.equal(result[len(reference):],value[len(reference):]):raise ValueError('neighbor altered')
        return result
    if isinstance(value,tuple) and isinstance(reference,tuple) and len(value)==len(reference):
        return tuple(substitute(v,r) for v,r in zip(value,reference))
    raise ValueError('unsupported substitution structure')


def compare(a,b):
    import torch
    if isinstance(a,torch.Tensor):
        if a.shape!=b.shape:return {'equal':False,'changed':max(a.numel(),b.numel()),'shape_mismatch':[list(a.shape),list(b.shape)]}
        return detailed_metric(a,b)
    if isinstance(a,(tuple,list)):
        rows=[compare(x,y) for x,y in zip(a,b)]
        if len(a)!=len(b):raise ValueError('structure mismatch')
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('keys mismatch')
        rows=[compare(a[k],b[k]) for k in a]
    else:return {'equal':a==b,'changed':int(a!=b)}
    return {'equal':all(r['equal'] for r in rows),'changed':sum(r['changed'] for r in rows),'items':rows}


def observe(module,*args,names=NAMES,replacement=None,tokens=64,kwargs=None):
    """Observe native module calls; replacement contains already-sliced target values."""
    modules=dict(module.named_modules()); captured={};handles=[]
    def hook(name):
        def record(mod,argv,kw,value):
            if name in captured:raise RuntimeError('boundary invoked twice: '+name)
            captured[name]={'args':target(argv,tokens),'kwargs':target(kw,tokens),'output':target(value,tokens)}
            if replacement and name==replacement[0]:return substitute(value,replacement[1])
        return record
    try:
        for name in names:handles.append(modules[name].register_forward_hook(hook(name),with_kwargs=True))
        value=module(*args,**(kwargs or {}))
    finally:
        for h in handles:h.remove()
    if set(captured)!=set(names):raise RuntimeError('boundary not hit')
    return value,captured


def profile_call(module,args,kwargs,path):
    import torch
    from torch.profiler import profile,ProfilerActivity
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True) as prof:
        value=module(*args,**kwargs);torch.cuda.synchronize()
    prof.export_chrome_trace(str(path));d=json.loads(path.read_text())
    return value,{'path':str(path),'sha256':sha_file(path),'kernels':kernel_index(d)}


def cpu_check(out):
    import torch
    class Sensitive(torch.nn.Module):
        def forward(self,x):
            y=x.clone()
            if len(x)==2:y[0,0,0]+=1
            return y
    m=Sensitive();x=torch.zeros(2,3,4)
    a,one=observe(m,x[:1],names=[''])
    b,two=observe(m,x,names=[''])
    sham,_=observe(m,x,names=[''],replacement=('',two['']['output']))
    fixed,_=observe(m,x,names=[''],replacement=('',one['']['output']))
    assert compare(one['']['args'],two['']['args'])['equal']
    assert compare(one['']['output'],two['']['output'])['changed']==1
    assert torch.equal(sham,b) and torch.equal(fixed[:1],a) and torch.equal(fixed[1:],b[1:])
    assert not torch.cuda.is_initialized()
    result={'ok':True,'cuda_initialized':False,'positive_control':'batch-only coordinate perturbation detected; same-shape sham unchanged; single-boundary replacement restores target and preserves neighbor'}
    write(out/'residual-cpu.json',result);return result


def expert_linear_diagnosis(layer, layer_inputs, cap, outputs, out):
    """Trace real functional Linear calls, keyed by exact expert weight identity.

    Target expert rows are selected by actual top-k slot/token dispatch order,
    never by assuming target tokens form a contiguous prefix of each expert.
    """
    import torch
    import torch.nn.functional as F
    from contextlib import contextmanager
    expert=layer.mlp.experts
    weight_map={}
    for e in range(expert.num_experts):
        for name,parameter in [('gate_up',expert.gate_up_proj),('down',expert.down_proj)]:
            w=parameter[e]
            weight_map[(w.data_ptr(),tuple(w.shape))]=(e,name)
    saved_inputs={}
    proj=layer.linear_attn.out_proj
    @contextmanager
    def instrument(batch,storage,replacement=None):
        native=F.linear
        route={}
        def pre(mod,args):
            # Native eager dispatch visits slot then token, exactly torch.where order.
            if len(args)!=3:raise RuntimeError('experts signature drift')
            hidden,indices,weights=args
            saved_inputs[batch]=tuple(x.detach().clone() for x in args)
            for e in range(expert.num_experts):
                slots,tokens=torch.where(indices.T==e)
                route[e]=(slots,tokens,tokens<64)
        def wrapped(x,w,bias=None):
            value=native(x,w,bias)
            key=weight_map.get((w.data_ptr(),tuple(w.shape)))
            if key is None:return value
            e,kind=key;label=f'{e}:{kind}';slots,tokens,mask=route[e]
            if x.shape[0]!=len(tokens):raise RuntimeError('non-eager expert layout; cannot assume dispatch rows')
            if label in storage:raise RuntimeError('expert linear repeated')
            storage[label]={'input':x[mask].detach().cpu().clone(),'output':value[mask].detach().cpu().clone(),
                            'all_input':x.detach().cpu().clone(),'all_output':value.detach().cpu().clone(),
                            'target_mask':mask.detach().cpu(),'tokens':tokens[mask].detach().cpu(),'slots':slots[mask].detach().cpu(),
                            'weight_shape':list(w.shape),'M':x.shape[0]}
            if replacement is not None and label in replacement:
                expected=replacement[label]
                edited=value.clone();edited[mask]=expected.to(value.device)
                if not torch.equal(edited[~mask],value[~mask]):raise RuntimeError('expert neighbor changed')
                return edited
            return value
        h=expert.register_forward_pre_hook(pre);F.linear=wrapped
        try:yield
        finally:F.linear=native;h.remove()
    data={}
    for batch in (1,2):
        observed={};hp=proj.register_forward_hook(lambda m,a,v:substitute(v,cap['batch1_output']))
        try:
            with torch.inference_mode(),instrument(batch,observed):value=layer(*layer_inputs[batch][0],**layer_inputs[batch][1])
        finally:hp.remove()
        if not torch.equal(value.cpu(),outputs[batch]):raise RuntimeError('functional observer altered baseline')
        data[batch]=observed
    if not data[1]:raise RuntimeError('native experts not eager functional linear; no expert call captured')
    records=[]
    for label,a in data[1].items():
        if label not in data[2]:raise RuntimeError('target expert missing from batch2')
        b=data[2][label]
        if not torch.equal(a['tokens'],b['tokens']) or not torch.equal(a['slots'],b['slots']):
            records.append({'name':label,'route_identity':False});continue
        if not a['output'].numel():continue
        row={'name':label,'route_identity':True,'M':[a['M'],b['M']],
             'input':compare(a['input'],b['input']),'output':compare(a['output'],b['output'])}
        if row['input']['equal'] and not row['output']['equal']:
            e,kind=label.split(':');w=(expert.gate_up_proj if kind=='gate_up' else expert.down_proj)[int(e)]
            row['profiles']=[]
            for batch,d in [(1,a),(2,b)]:
                x=d['all_input'].to(w.device)
                with torch.inference_mode():value,trace=profile_call(lambda z:F.linear(z,w),(x,),{},out/f'residual-expert-{label.replace(":","-")}-{batch}.trace.json')
                if not torch.equal(value.cpu(),d['all_output']):raise RuntimeError('expert Linear same input replay failed')
                row['profiles'].append({'batch':batch,**trace})
            # One real functional call only, all other experts and paths unchanged.
            observed={};hp=proj.register_forward_hook(lambda m,a,v:substitute(v,cap['batch1_output']))
            try:
                with torch.inference_mode(),instrument(2,observed,{label:a['output']}):value=layer(*layer_inputs[2][0],**layer_inputs[2][1])
            finally:hp.remove()
            row['intervention_layer0']=detailed_metric(outputs[1][:1],value[:1])
            row['neighbor_unchanged']=torch.equal(outputs[2][1:],value[1:].cpu())
        records.append(row)
    # Joint restoration is supplementary; individual counterfactuals above remain separate.
    fixes={r['name']:data[1][r['name']]['output'] for r in records if r.get('input',{}).get('equal') and not r.get('output',{}).get('equal',True)}
    combined=None
    if fixes:
        observed={};hp=proj.register_forward_hook(lambda m,a,v:substitute(v,cap['batch1_output']))
        try:
            with torch.inference_mode(),instrument(2,observed,fixes):value=layer(*layer_inputs[2][0],**layer_inputs[2][1])
        finally:hp.remove()
        combined=detailed_metric(outputs[1][:1],value[:1])
    torch.save(data,out/'residual-expert-captures.pt')
    answer={'ok':True,'records':records,'joint_source_replacement_layer0':combined,'capture_sha256':sha_file(out/'residual-expert-captures.pt'),
            'scope':'native eager expert functional Linear only; exact target dispatch row identities, profiling and individual causal replacement'}
    write(out/'residual-experts.json',answer)
    return answer


def load_official_model(loader=None):
    """One native loading contract shared by initial capture and follow-up."""
    import torch
    if loader is None:
        from transformers import AutoModelForImageTextToText
        loader=AutoModelForImageTextToText.from_pretrained
    return loader(MODEL,local_files_only=True,dtype=torch.bfloat16,
                  attn_implementation='sdpa',device_map='cuda').eval()


def run(input_path,out):
    import torch
    from importlib.metadata import version
    from transformers import AutoModelForImageTextToText
    inputs=json.loads(input_path.read_text());validate_input(inputs)
    if inputs['tokens_sha256']!=TOKEN_SHA or inputs['revision']!=REVISION:raise ValueError('B07 identity drift')
    versions={p:version(p) for p in ('torch','transformers')}
    if versions!={'torch':'2.13.0','transformers':'5.10.4'}:raise ValueError('frozen versions drift')
    verify_model(inputs)
    if sha_file(CAPTURE)!=CAPTURE_SHA:raise ValueError('B05 capture drift')
    cap=torch.load(CAPTURE,map_location='cpu',weights_only=True)
    model=load_official_model()
    modules=dict(model.named_modules());layer=modules[PREFIX];proj=modules[PROJECTION]
    source=Path(inspect.getfile(type(layer)))
    if sha_file(source)!=B03_SOURCE_SHA:raise ValueError('B07 modeling source drift')
    (out/'modeling.py').write_bytes(source.read_bytes())
    result={'ok':False,'versions':versions,'torch_git':torch.version.git_version,'source_sha256':sha_file(source),'capture_sha256':CAPTURE_SHA,'input_sha256':sha_file(input_path),'records':[]}
    layer_inputs={};full={}
    # Retain native layer invocation tensors. Full root provides the exact B07 context.
    for batch in (1,2):
        def pre(mod,args,kw):layer_inputs[batch]=(tree(args,lambda t:t.detach().clone()),tree(kw,lambda t:t.detach().clone()))
        h=layer.register_forward_pre_hook(pre,with_kwargs=True)
        holder={}
        def save(mod,args,value):holder['output']=value.detach().cpu().clone()
        hh=layer.register_forward_hook(save)
        def projection_check(mod,args,value):
            if not torch.equal(value.cpu(),cap[f'batch{batch}_output']):raise RuntimeError('full root B05 projection mismatch')
        hp=proj.register_forward_hook(projection_check)
        ids=torch.tensor(inputs['tokens'][:batch],device='cuda')
        try:
            with torch.inference_mode():model(input_ids=ids,position_ids=torch.tensor(inputs['positions'],device='cuda').unsqueeze(0).expand_as(ids),attention_mask=torch.ones_like(ids),use_cache=False)
        finally:h.remove();hh.remove();hp.remove()
        full[batch]=holder['output']
    # Keep layer only: the root invocation already supplied exact inputs and kwargs.
    natural=detailed_metric(full[1][:1],full[2][:1]);result['natural_layer0']=natural
    if natural['changed']!=4819:raise RuntimeError('B07 natural 4819 gate failed')
    baseline={};outputs={}
    for batch in (1,2):
        args,kw=layer_inputs[batch]
        with torch.inference_mode():
            raw=layer(*args,**kw)
            if not torch.equal(raw.cpu(),full[batch]):raise RuntimeError('isolated native layer differs from full root')
            forced=lambda mod,argv,val:substitute(val,cap['batch1_output'])
            hp=proj.register_forward_hook(forced)
            try:
                plain=layer(*args,**kw)
                value,observed=observe(layer,*args,kwargs=kw)
                sham,_=observe(layer,*args,kwargs=kw,replacement=('mlp',observed['mlp']['output']))
                if not torch.equal(value,plain) or not torch.equal(sham,plain):raise RuntimeError('observer/sham changed layer')
            finally:hp.remove()
        baseline[batch]=observed;outputs[batch]=value.detach().cpu()
    residual=detailed_metric(outputs[1][:1],outputs[2][:1]);result['projection_aligned_layer0']=residual
    write(out/'residual-partial.json',result)
    if residual['changed']!=322:raise RuntimeError('B07 projection-aligned 322 entry gate failed')
    torch.save({'layer_inputs':tree(layer_inputs,lambda t:t.cpu()),'boundaries':baseline,'layer_outputs':outputs},out/'residual-captures.pt')
    result['tensor_artifact']={'path':str(out/'residual-captures.pt'),'sha256':sha_file(out/'residual-captures.pt')}
    for name in NAMES:
        a,b=baseline[1][name],baseline[2][name]
        result['records'].append({'name':name,'input':compare(a['args'],b['args']),'kwargs':compare(a['kwargs'],b['kwargs']),'output':compare(a['output'],b['output'])})
    result['interventions']=[]
    # Every changed boundary receives an independent target-only intervention.
    for row in result['records']:
        name=row['name']
        if not name or not name.startswith('mlp') or row['output']['equal']:continue
        args,kw=layer_inputs[2]
        hp=proj.register_forward_hook(lambda mod,argv,val:substitute(val,cap['batch1_output']))
        try:
            with torch.inference_mode():
                value,_=observe(layer,*args,kwargs=kw,replacement=(name,baseline[1][name]['output']))
                repeat,_=observe(layer,*args,kwargs=kw,replacement=(name,baseline[1][name]['output']))
        finally:hp.remove()
        if not torch.equal(value,repeat):raise RuntimeError('intervention not repeatable')
        result['interventions'].append({'name':name,'layer0_vs_single':detailed_metric(outputs[1][:1],value[:1]),'neighbor_unchanged':torch.equal(value[1:].cpu(),outputs[2][1:])})
        write(out/'residual-partial.json',result)
    # Native modules replay on same original target input; record both actual batch shapes.
    result['operator_replays']=[]
    for row in result['records']:
        name=row['name']
        if not name.startswith('mlp') or row['output']['equal']:continue
        module=dict(layer.named_modules())[name]
        replay=[]
        # Recapture full arguments because target snapshots intentionally omit neighbors.
        for batch in (1,2):
            saved={}
            def before(mod,args,kw):saved.update(args=tree(args,lambda t:t.detach().clone()),kwargs=tree(kw,lambda t:t.detach().clone()))
            h=module.register_forward_pre_hook(before,with_kwargs=True)
            hp=proj.register_forward_hook(lambda mod,argv,val:substitute(val,cap['batch1_output']))
            try:
                with torch.inference_mode():layer(*layer_inputs[batch][0],**layer_inputs[batch][1])
            finally:h.remove();hp.remove()
            with torch.inference_mode():
                plain=module(*saved['args'],**saved['kwargs'])
                value,trace=profile_call(module,saved['args'],saved['kwargs'],out/f'residual-{name or "layer"}-{batch}.trace.json')
            eq=compare(target(value),baseline[batch][name]['output'])['equal']
            if not eq or not compare(target(plain),target(value))['equal']:raise RuntimeError('native operator replay/profile mismatch')
            replay.append({'batch':batch,'capture_equal':eq,**trace})
        result['operator_replays'].append({'name':name,'target_input_equal':row['input']['equal'] and row['kwargs']['equal'],'arms':replay})
        write(out/'residual-partial.json',result)
    result['causal_boundaries']=[r['name'] for r in result['interventions'] if r['layer0_vs_single']['equal'] and r['neighbor_unchanged']]
    result['same_input_new_difference']=[r['name'] for r in result['records'] if r['input']['equal'] and r['kwargs']['equal'] and not r['output']['equal']]
    if not result['same_input_new_difference'] or not result['causal_boundaries']:raise RuntimeError('no same-input source plus causal restoration; diagnosis incomplete')
    if 'mlp.experts' in result['same_input_new_difference']:
        result['expert_diagnosis']=expert_linear_diagnosis(layer,layer_inputs,cap,outputs,out)
    result['diagnostic_ok']=True
    result['mechanism_complete']=any(n in result['same_input_new_difference'] for n in result['causal_boundaries'] if isinstance(dict(layer.named_modules())[n],torch.nn.Linear))
    result['ok']=True
    result['scope']='Real layer0 module localization and native kernel traces. Internal expert functional operations require follow-up if experts is the first differing boundary; ok does not assert a closed-source arithmetic root cause.'
    write(out/'residual-result.json',result);return result


FOLLOW_CAPTURE = Path('/vol/_audit/infra-probes/B09/batch5-residual-gpu-01/attempt-6e1b0d778cb546f78c19f2b4990af010/residual-captures.pt')
FOLLOW_SHA = '3aa32d12357518f1a0f87b8e845239c1d0bfb1c6533d562760707ba5199b39c5'


def joint_fixture():
    import torch
    class Part(torch.nn.Module):
        def __init__(self,coord):super().__init__();self.coord=coord
        def forward(self,x):
            y=x.clone()
            if len(x)==2:y[0,0,self.coord]+=1
            return y
    class Pair(torch.nn.Module):
        def __init__(self):super().__init__();self.gate=Part(0);self.up=Part(1)
        def forward(self,x):return self.gate(x)+self.up(x)
    m=Pair();x=torch.zeros(2,3,4);base=m(x[:1]);natural=m(x)
    hs=[m.gate.register_forward_hook(lambda mod,args,val:substitute(val,x[:1]))]
    try:one=m(x)
    finally:
        for h in hs:h.remove()
    hs=[mod.register_forward_hook(lambda mod,args,val:substitute(val,x[:1])) for mod in (m.gate,m.up)]
    try:joint=m(x)
    finally:
        for h in hs:h.remove()
    assert detailed_metric(base,natural[:1])['changed']==2
    assert detailed_metric(base,one[:1])['changed']==1
    assert torch.equal(base,joint[:1]) and torch.equal(joint[1:],natural[1:])
    return {'ok':True,'natural_changed':2,'single_changed':1,'joint_changed':0}


def exact_coordinates(x,w,single,paired):
    """Only differing coordinates; BF16 values convert exactly to Python fractions."""
    from fractions import Fraction
    import torch
    x,w,single,paired=[v.detach().cpu() for v in (x,w,single,paired)]
    counts={'single_nearer':0,'paired_nearer':0,'equal_distance':0};rows=[]
    for i,j in (single!=paired).nonzero().tolist():
        exact=sum((Fraction(float(a))*Fraction(float(b)) for a,b in zip(x[i].tolist(),w[j].tolist())),Fraction(0))
        a,b=Fraction(float(single[i,j])),Fraction(float(paired[i,j]));da,db=abs(exact-a),abs(exact-b)
        nearer='single_nearer' if da<db else 'paired_nearer' if db<da else 'equal_distance';counts[nearer]+=1
        rows.append({'coordinate':[i,j],'single':float(a),'paired':float(b),'exact_numerator':str(exact.numerator),'exact_denominator':str(exact.denominator),'nearer':nearer})
    return {'coordinates':len(rows),'counts':counts,'rows':rows}


def fixed_splitk(api,x,w,single,paired,out,name):
    import ctypes
    import torch
    from torch.profiler import profile,ProfilerActivity
    from syncopate.infra.splitk_probe import tensor_hash,validate_same_algorithm,validate_disabled
    x=x.contiguous();w=w.contiguous();size=32*1024**2;workspace=torch.empty(size,device='cuda',dtype=torch.uint8)
    m,k=x.shape;n=w.shape[0]
    result={'M':m,'N':n,'K':k,'input_shape':list(x.shape),'input_stride':list(x.stride()),'weight_shape':list(w.shape),'weight_stride':list(w.stride()),'input_sha256':tensor_hash(x),'weight_sha256':tensor_hash(w),'records':[]}
    tensors={'single':single.cpu(),'paired':paired.cpu(),'input':x.cpu(),'weight':w.cpu()}
    reference=torch.nn.functional.linear(x.double(),w.double()).cpu();tensors['reference_fp64']=reference
    result['reference']={'single_vs_fp64':detailed_metric(reference,single),'paired_vs_fp64':detailed_metric(reference,paired),'single_vs_rounded':detailed_metric(reference.bfloat16(),single),'paired_vs_rounded':detailed_metric(reference.bfloat16(),paired)}
    for mode,arm in [(0,'unrestricted'),(2,'no_splitk_same_algorithm')]:
        handle=ctypes.c_void_p();attrs=(ctypes.c_int*15)();status=api.make_plan(m,n,k,mode,size,ctypes.byref(handle),attrs)
        if status:
            result['records'].append({'arm':arm,'supported':False,'status':status});continue
        keys=('id','tile','split_k','reduction','stages','swizzle','custom')
        final=dict(zip(keys,list(attrs)[:7]));original=dict(zip(keys,list(attrs)[8:]))
        try:
            if mode:
                if not result['records'] or not result['records'][0].get('supported'):raise RuntimeError('same-algorithm intervention has no supported baseline')
                validate_same_algorithm(result['records'][0]['algorithm'],original,final)
            y=torch.empty_like(single)
            def execute():
                code=api.execute_plan(handle,w.data_ptr(),x.data_ptr(),y.data_ptr(),workspace.data_ptr(),size,torch.cuda.current_stream().cuda_stream)
                if code:raise RuntimeError(f'cuBLASLt execute {code}')
            copies=[]
            for _ in range(3):execute();torch.cuda.synchronize();copies.append(y.cpu().clone())
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:execute();torch.cuda.synchronize()
            path=out/f'residual-follow-{name}-{arm}.trace.json';prof.export_chrome_trace(str(path));kernels=kernel_index(json.loads(path.read_text()))
            if mode:validate_disabled(final,kernels)
            if not all(torch.equal(v,copies[0]) for v in copies) or not torch.equal(y.cpu(),copies[0]):raise RuntimeError('profile/repeat drift')
            tensors[arm]=copies[0]
            result['records'].append({'arm':arm,'supported':True,'algorithm':final,'original_algorithm':original,'kernels':kernels,'trace':str(path),'trace_sha256':sha_file(path),'equals_single':torch.equal(copies[0],single.cpu()),'equals_paired':torch.equal(copies[0],paired.cpu()),'vs_single':detailed_metric(single,copies[0]),'vs_paired':detailed_metric(paired,copies[0]),'output_sha256':tensor_hash(copies[0])})
        finally:api.destroy_plan(handle)
    result['torch_attribution_ok']=bool(len(result['records'])==2 and result['records'][0].get('equals_single') and result['records'][0].get('algorithm',{}).get('split_k',0)>1 and result['records'][1].get('equals_paired'))
    result['exact_dot']=exact_coordinates(x,w,single,paired)
    path=out/f'residual-follow-{name}.pt';torch.save(tensors,path);result['tensors']={'path':str(path),'sha256':sha_file(path)}
    write(out/f'residual-follow-{name}.json',result);return result


def followup(input_path,out):
    import torch
    from importlib.metadata import version
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m
    from syncopate.infra.splitk_probe import compile_bridge
    inputs=json.loads(input_path.read_text());validate_input(inputs);verify_model(inputs)
    if inputs['tokens_sha256']!=TOKEN_SHA or inputs['revision']!=REVISION:raise ValueError('input drift')
    if {p:version(p) for p in ('torch','transformers')}!={'torch':'2.13.0','transformers':'5.10.4'}:raise ValueError('version drift')
    if sha_file(Path(inspect.getfile(m)))!=B03_SOURCE_SHA or sha_file(FOLLOW_CAPTURE)!=FOLLOW_SHA:raise ValueError('source/capture drift')
    captured=torch.load(FOLLOW_CAPTURE,map_location='cpu',weights_only=True);li=tree(captured['layer_inputs'],lambda t:t.cuda())
    baselines=captured['boundaries'];outputs=captured['layer_outputs']
    model=load_official_model()
    layer=dict(model.named_modules())[PREFIX]
    loader_identity={'parameter_dtypes':{n:str(p.dtype) for n,p in layer.named_parameters()},'experts_implementation':getattr(layer.mlp.experts,'config',None)._experts_implementation if getattr(layer.mlp.experts,'config',None) is not None else getattr(model.config.text_config,'_experts_implementation',None),'loader':'shared load_official_model; same as GPU01'}
    write(out/'residual-follow-loader.json',loader_identity)
    projection=layer.linear_attn.out_proj;split_inputs={}
    # All comparisons retain B07 first-projection alignment.
    def call(batch,arms=()):
        hs=[projection.register_forward_hook(lambda mod,args,value:substitute(value,baselines[1]['linear_attn.out_proj']['output']))]
        for name in arms:
            module=dict(layer.named_modules())[name]
            hs.append(module.register_forward_hook(lambda mod,args,value,n=name:substitute(value,baselines[1][n]['output'])))
        try:
            with torch.inference_mode():value=layer(*li[batch][0],**li[batch][1])
        finally:
            for h in hs:h.remove()
        return value
    for b in (1,2):
        value=call(b)
        if not torch.equal(value.cpu(),outputs[b]):
            write(out/'residual-follow-replay-failure.json',{'batch':b,'difference':detailed_metric(outputs[b],value),'loader_identity':loader_identity})
            raise RuntimeError('official-loaded layer does not reproduce GPU01')
    names=('mlp.shared_expert.gate_proj','mlp.shared_expert.up_proj')
    joint=call(2,names);repeated=call(2,names)
    result={'ok':False,'loader_identity':loader_identity,'capture_sha256':FOLLOW_SHA,'joint_fixture':joint_fixture(),'entry_322':detailed_metric(outputs[1][:1],outputs[2][:1]),'joint_layer0':detailed_metric(outputs[1][:1],joint[:1]),'joint_repeat_equal':torch.equal(joint,repeated),'neighbor_unchanged':torch.equal(joint[1:].cpu(),outputs[2][1:]),'operators':[]}
    if result['entry_322']['changed']!=322 or not result['joint_repeat_equal'] or not result['neighbor_unchanged']:raise RuntimeError('followup identity/intervention gate')
    write(out/'residual-follow-partial.json',result)
    api,metadata=compile_bridge(out);result['api']=metadata
    for name in names:
        module=dict(layer.named_modules())[name];xs={}
        for batch in (1,2):
            h=module.register_forward_pre_hook(lambda mod,args,b=batch:xs.update({b:args[0].detach().clone()}))
            try:call(batch)
            finally:h.remove()
        if not torch.equal(xs[1],xs[2][:64]):raise RuntimeError('actual target Linear inputs differ')
        with torch.inference_mode():single=module(xs[1]);paired=module(xs[2])[:64]
        if not torch.equal(single.cpu(),baselines[1][name]['output']) or not torch.equal(paired.cpu(),baselines[2][name]['output']):raise RuntimeError('followup Linear capture replay failed')
        op=fixed_splitk(api,xs[1],module.weight,single,paired,out,name.rsplit('.',1)[-1]);op['batch2_input_shape']=list(xs[2].shape);op['batch2_input_stride']=list(xs[2].stride());op['name']=name
        result['operators'].append(op);write(out/'residual-follow-partial.json',result)
    result['mechanism_complete']=bool(result['joint_layer0']['equal'] and all(o['torch_attribution_ok'] for o in result['operators']))
    result['ok']=True
    result['scope']='fixed-M per-Linear split-K counterfactual plus real native layer joint intervention; exact references cover stored BF16 dot products, no performance claim'
    write(out/'residual-result.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--cpu',action='store_true');p.add_argument('--followup',action='store_true');p.add_argument('--input',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not a.cpu and a.input is None:p.error('--input required')
    a.out.mkdir(parents=True,exist_ok=True)
    if any(a.out.glob('residual-*')):raise RuntimeError('use fresh output directory')
    try:
        if a.cpu:
            result=cpu_check(a.out)
            if a.followup:
                from syncopate.infra.splitk_probe import compile_bridge
                import torch
                result['joint_fixture']=joint_fixture();_,result['api']=compile_bridge(a.out)
                if torch.cuda.is_initialized():raise RuntimeError('CPU initialized CUDA')
                write(a.out/'residual-cpu.json',result)
        elif a.followup:followup(a.input,a.out)
        else:run(a.input,a.out)
    except Exception as e:
        write(a.out/'residual-failure.json',{'ok':False,'type':type(e).__name__,'error':str(e)});raise

if __name__=='__main__':main()
