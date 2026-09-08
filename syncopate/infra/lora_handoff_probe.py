"""B08: immutable controlled LoRA artifacts through actual vLLM loading and use.

Diagnostic eager TP1 only. No training, network throughput or production claim.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from syncopate.infra.probability_probe import compare_logprobs, sha_file, verify_model, matches_prefill

RANK = 8


def select_target(config, mapping):
    indices=[i for i,t in enumerate(config['layer_types']) if t=='full_attention']
    if not indices:raise ValueError('no full attention layer')
    key=f'model.language_model.layers.{indices[-1]}.self_attn.o_proj.weight'
    if key not in mapping:raise ValueError('exact supported projection absent: '+key)
    return key


def assess_cycle(rows):
    required={'base','A','B','A_reload','base_restored'}
    if set(rows)!=required or any(len(v)!=2 for v in rows.values()):
        raise ValueError('missing phase or repeat')
    noise=max(compare_logprobs(v[0],v[1])['max_abs'] for v in rows.values())
    delta=compare_logprobs(rows['A'][0],rows['B'][0])['max_abs']
    effect=compare_logprobs(rows['A'][0],rows['base'][0])['max_abs']
    recovered=compare_logprobs(rows['A'][0],rows['A_reload'][0])['max_abs']
    restored=compare_logprobs(rows['base'][0],rows['base_restored'][0])['max_abs']
    return {'ok':noise<=1e-5 and delta>max(1e-4,10*noise) and effect>max(1e-4,10*noise)
            and recovered<=1e-5 and restored<=1e-5,'repeat_max':noise,'AB_max':delta,
            'base_A_max':effect,'A_reload_max':recovered,'base_restored_max':restored}


def cpu(output, input_path):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file, load_file
    from vllm.config import LoRAConfig
    from vllm.lora.request import LoRARequest
    from vllm.lora.peft_helper import PEFTHelper
    import vllm
    from importlib.metadata import version
    assert version('vllm')=='0.28.0'
    package=Path(vllm.__file__).parent
    source_files=['lora/worker_manager.py','lora/model_manager.py','lora/layers/base_linear.py',
                  'model_executor/models/qwen3_5.py','v1/worker/gpu_worker.py','entrypoints/llm.py','v1/engine/core.py','v1/engine/llm_engine.py','v1/engine/core_client.py']
    source_files += ['model_executor/layers/fused_moe/oracle/unquantized.py','lora/layers/fused_moe.py']
    source_identity={n:sha_file(package/n) for n in source_files}
    assert 'self._adapter_manager.activate_adapter(lora_request.lora_int_id)' in (package/'lora/worker_manager.py').read_text()
    assert 'self.model.lora_manager = self' in (package/'lora/model_manager.py').read_text()
    # Compose real installed target matching with the exact failing selector branch.
    from types import SimpleNamespace
    from vllm.lora.model_manager import LoRAModelManager
    manager=SimpleNamespace(supported_lora_modules=['o_proj','experts'],packed_modules_mapping={},
                            lora_config=SimpleNamespace(target_modules=['o_proj']))
    assert not LoRAModelManager._match_target_modules(manager,'language_model.model.layers.0.mlp.experts')
    manager.lora_config.target_modules=None
    assert LoRAModelManager._match_target_modules(manager,'language_model.model.layers.0.mlp.experts')
    oracle=(package/'model_executor/layers/fused_moe/oracle/unquantized.py').read_text()
    assert oracle.index('if moe_config.is_lora_enabled:') < oracle.index('runner_backend = moe_config.moe_backend')
    assert 'fused_experts.set_lora_context(lora_context)' in (package/'lora/layers/fused_moe.py').read_text()
    from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA
    wrapper=object.__new__(FusedMoE3DWithLoRA)
    torch.nn.Module.__init__(wrapper)
    wrapper.moe_config=SimpleNamespace(hidden_dim=16,num_local_experts=2,intermediate_size_per_partition=8)
    wrapper.device=torch.device('cpu');wrapper.fully_sharded=False;wrapper.tp_size=1
    wrapper.enable_moe_shared_loras=False;wrapper._w13_slices=1
    tiny_config=SimpleNamespace(max_lora_rank=RANK,lora_dtype=torch.bfloat16)
    wrapper._create_lora_a_weights(1,tiny_config);wrapper._create_lora_b_weights(1,tiny_config)
    for attr in ['w13_lora_a_stacked','w13_lora_b_stacked','w2_lora_a_stacked','w2_lora_b_stacked']:
        buffers=getattr(wrapper,attr)
        assert isinstance(buffers,tuple) and buffers
        assert all(t.ndim==4 and t.shape[0]==1 and torch.count_nonzero(t[0]).item()==0 for t in buffers)


    from vllm.v1.serial_utils import MsgpackEncoder, MsgpackDecoder
    from vllm.v1.engine.core import EngineCore, EngineCoreProc
    os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION']='1'
    assert MsgpackEncoder().encode(inspect_runtime)
    # Actual wire round-trip: generic RPC loses nested Struct type information;
    # the dedicated add_lora utility restores its top-level typed argument.
    probe_request=LoRARequest('wire-positive',1,'/tmp/b08-controlled-wire-only')
    core=object.__new__(EngineCore)
    generic=MsgpackDecoder().decode(MsgpackEncoder().encode(('add_lora',None,(probe_request,),{})))
    generic_args=EngineCoreProc._convert_msgspec_args(core.collective_rpc,generic)
    assert isinstance(generic_args[2][0],list)
    typed=MsgpackDecoder().decode(MsgpackEncoder().encode((probe_request,)))
    typed_args=EngineCoreProc._convert_msgspec_args(core.add_lora,typed)
    assert isinstance(typed_args[0],LoRARequest) and typed_args[0]==probe_request

    inputs=json.loads(Path(input_path).read_text());verify_model(inputs)
    assert len(inputs['tokens'][0])==64
    (output/'input.json').write_bytes(Path(input_path).read_bytes())
    cfg=inputs['config']['text_config']
    mapping=json.loads((Path(inputs['model_path'])/'model.safetensors.index.json').read_text())['weight_map']
    target=select_target(cfg,mapping)
    with safe_open(str(Path(inputs['model_path'])/mapping[target]),framework='pt',device='cpu') as f:
        shape=f.get_slice(target).get_shape()
    assert len(shape)==2
    start=time.monotonic();adapters={}
    gen=torch.Generator().manual_seed(808)
    a=(torch.randn(RANK,shape[1],generator=gen)*.05).to(torch.bfloat16)
    b=(torch.randn(shape[0],RANK,generator=gen)*.05).to(torch.bfloat16)
    for label,sign in [('A',1),('B',-1)]:
        dest=output/'sender'/label;dest.mkdir(parents=True)
        config={'base_model_name_or_path':inputs['model'],'peft_type':'LORA','task_type':'CAUSAL_LM',
                'r':RANK,'lora_alpha':RANK,'lora_dropout':0.,'bias':'none','target_modules':['o_proj'],
                'inference_mode':True,'use_rslora':False,'use_dora':False}
        (dest/'adapter_config.json').write_text(json.dumps(config,indent=2)+'\n')
        prefix='base_model.model.'+target.removesuffix('.weight')
        tensors={prefix+'.lora_A.weight':a,prefix+'.lora_B.weight':(b*sign).contiguous()}
        save_file(tensors,str(dest/'adapter_model.safetensors'))
        loaded=load_file(str(dest/'adapter_model.safetensors'))
        assert all(torch.equal(tensors[k],loaded[k]) and torch.count_nonzero(loaded[k])>0 for k in tensors)
        helper=PEFTHelper.from_local_dir(str(dest),256)
        helper.validate_legal(LoRAConfig(max_lora_rank=RANK,max_loras=1,lora_dtype=torch.bfloat16))
        assert LoRARequest(label,1,str(dest)).lora_int_id==1
        from vllm.lora.lora_model import LoRAModel
        from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration
        mapper=Qwen3_5MoeForConditionalGeneration.hf_to_vllm_mapper.get_unstacked_mapper()
        # CPU-only image has no NVIDIA driver. Upstream cached PIN_MEMORY=True
        # requests CUDA host allocation even for device='cpu'. Disable only that
        # allocation flag for this fixture; mapping/loading code is unchanged.
        import vllm.lora.lora_model as loader_module
        from unittest.mock import patch
        prior_pin_memory=loader_module.PIN_MEMORY
        with patch.object(loader_module,'PIN_MEMORY',False):
            actual=LoRAModel.from_local_checkpoint(str(dest),{'o_proj'},helper,lora_model_id=1,
                device='cpu',dtype=torch.bfloat16,model_vocab_size=cfg['vocab_size'],weights_mapper=mapper)
        assert loader_module.PIN_MEMORY is prior_pin_memory
        expected_name='language_model.model.'+target.removeprefix('model.language_model.').removesuffix('.weight')
        assert list(actual.loras)==[expected_name],list(actual.loras)
        assert torch.equal(actual.loras[expected_name].lora_a,a)
        assert torch.equal(actual.loras[expected_name].lora_b,b*sign)

        adapters[label]={'files':{p.name:sha_file(p) for p in dest.iterdir()},'version':label,
                         'shape':shape,'target':target,'rank':RANK,'alpha':RANK}
    assert adapters['A']['files']['adapter_model.safetensors']!=adapters['B']['files']['adapter_model.safetensors']
    result={'ok':True,'adapters':adapters,'target':target,'pack_seconds':time.monotonic()-start,
            'input_sha256':sha_file(output/'input.json'),'cuda_initialized':torch.cuda.is_initialized(),
            'installed_sources':source_identity,'vllm_version':version('vllm'),
            'wire_contract':{'generic_rpc_nested_struct_becomes_list':True,'typed_utility_restores_LoRARequest':True},
            'cpu_loader_allocation':{'PIN_MEMORY_scoped_false':True,'restored':True,'gpu_path_unmodified':True}}
    assert not result['cuda_initialized']
    (output/'cpu-result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def inspect_runtime(model, adapter_path, adapter_id, target, trace_path=None):
    """Read actual registered weights and GPU slot, then attach observation only."""
    import torch
    from safetensors.torch import load_file
    import os
    manager=model.lora_manager
    registered=manager.list_adapters()
    assert adapter_id in registered
    adapter=manager.get_adapter(adapter_id)
    suffix=target.removeprefix('model.language_model.').removesuffix('.weight')
    names=[n for n in adapter.loras if n.endswith('.'+suffix)]
    assert len(names)==1,(names,list(adapter.loras))
    name=names[0];weights=adapter.loras[name]
    assert list(adapter.loras)==[name], 'unexpected loaded adapter target'
    from vllm.lora.layers.fused_moe import FusedMoEWithLoRA
    expert_inventory=[]
    for module_name,module in manager.modules.items():
        if isinstance(module,FusedMoEWithLoRA):
            assert module._moe_kernel.fused_experts._lora_context is not None
            buffers=[t for attr in ['w13_lora_a_stacked','w13_lora_b_stacked','w2_lora_a_stacked','w2_lora_b_stacked'] for t in getattr(module,attr)]
            assert all(torch.count_nonzero(t[manager.lora_index_to_id.index(adapter_id)]).item()==0 for t in buffers)
            expert_inventory.append({'name':module_name,'class':type(module).__name__,'context_set':True,'active_slot_zero':True})
    assert expert_inventory, 'official default must install MoE LoRA wrappers'
    source=load_file(str(Path(adapter_path)/'adapter_model.safetensors'))
    prefix='base_model.model.'+target.removesuffix('.weight')
    a=source[prefix+'.lora_A.weight'];b=source[prefix+'.lora_B.weight']
    assert torch.equal(weights.lora_a.cpu(),a) and torch.equal(weights.lora_b.cpu(),b)
    slot=manager.lora_index_to_id.index(adapter_id)
    layer=manager.modules[name]
    actual_a=layer.lora_a_stacked[0][slot].squeeze(0)
    actual_b=layer.lora_b_stacked[0][slot].squeeze(0)
    assert torch.equal(actual_a[:a.shape[0],:a.shape[1]].cpu(),a)
    assert torch.equal(actual_b[:b.shape[0],:b.shape[1]].cpu(),b)
    if trace_path and not hasattr(layer,'_b08_hook'):
        def observe(module,args,result):
            value=result[0] if isinstance(result,tuple) else result
            with open(trace_path,'a') as stream:
                stream.write(json.dumps({'pid':os.getpid(),'module':name,'slots':manager.lora_index_to_id,
                                         'shape':list(value.shape),'finite':bool(torch.isfinite(value).all())})+'\n')
        layer._b08_hook=layer.register_forward_hook(observe)
    return {'pid':os.getpid(),'registered':sorted(registered),'module':name,'slot':slot,
            'cpu_weights_exact':True,'gpu_weights_exact':True,'layer_class':type(layer).__name__,
            'expert_wrappers':expert_inventory}


def gpu(output, preflight):
    os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION']='1'
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    preflight=Path(preflight);prepared=json.loads((preflight/'cpu-result.json').read_text())
    assert prepared['ok'] and not prepared['cuda_initialized']
    assert sha_file(preflight/'input.json')==prepared['input_sha256']
    inputs=json.loads((preflight/'input.json').read_text());verify_model(inputs)
    rows={};record={'ok':False,'phases':[],'pack_seconds':prepared['pack_seconds']}
    (output/'input.json').write_bytes((preflight/'input.json').read_bytes())
    target=prepared['target'];trace=str(output/'projection.jsonl');root_trace=str(output/'root.jsonl')
    try:
        start=time.monotonic()
        for label,item in prepared['adapters'].items():
            source=preflight/'sender'/label
            assert all(sha_file(source/n)==h for n,h in item['files'].items())
            shutil.copytree(source,output/'receiver'/label)
            assert all(sha_file(output/'receiver'/label/n)==h for n,h in item['files'].items())
        record['local_filesystem_send_seconds']=time.monotonic()-start
        llm=LLM(model=inputs['model_path'],dtype='bfloat16',tensor_parallel_size=1,max_model_len=256,
                max_num_seqs=1,max_num_batched_tokens=256,gpu_memory_utilization=.70,
                enable_prefix_caching=False,seed=137,limit_mm_per_prompt={'image':0,'video':0},
                enable_lora=True,max_loras=1,max_cpu_loras=1,max_lora_rank=RANK,
                enforce_eager=True)
        def install_root(model):
            def observe(module,args,kwargs):
                ids=kwargs.get('input_ids',args[0] if args else None)
                pos=kwargs.get('positions',args[1] if len(args)>1 else None)
                if ids is not None and pos is not None:
                    with open(root_trace,'a') as f:f.write(json.dumps({'tokens':ids.cpu().tolist(),'positions':pos.cpu().tolist()})+'\n')
            model.register_forward_pre_hook(observe,with_kwargs=True)
            return type(model).__name__
        record['model_classes']=llm.apply_model(install_root)
        params=SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=1,seed=137)
        for phase,label in [('base',None),('A','A'),('B','B'),('A_reload','A'),('base_restored',None)]:
            phase_record={'phase':phase,'version':label,'adapter_id':1 if label else None}
            start=time.monotonic()
            if phase in {'B','A_reload','base_restored'}:
                removed=llm.llm_engine.remove_lora(1)
                assert removed is True
                remaining=llm.llm_engine.list_loras()
                assert 1 not in remaining
                phase_record['removed']=removed
            phase_record['unload_seconds']=time.monotonic()-start
            request=LoRARequest(label,1,str(output/'receiver'/label)) if label else None
            start=time.monotonic()
            if request:
                loaded=llm.llm_engine.add_lora(request)
                assert loaded is True
                phase_record['load_ack']=loaded
                phase_record['artifact_sha256']=prepared['adapters'][label]['files']['adapter_model.safetensors']
                # add_lora activates the adapter on the actual worker in v0.28.
                from functools import partial
                phase_record['receiver']=llm.apply_model(partial(inspect_runtime,adapter_path=request.lora_path,
                    adapter_id=1,target=target,trace_path=trace))
            phase_record['load_and_identity_seconds']=time.monotonic()-start
            rows[phase]=[]
            for repeat in range(2):
                before=len(Path(root_trace).read_text().splitlines()) if Path(root_trace).exists() else 0
                projection_before=len(Path(trace).read_text().splitlines()) if Path(trace).exists() else 0
                start=time.monotonic()
                result=llm.generate([{'prompt_token_ids':inputs['tokens'][0]}],params,lora_request=request,use_tqdm=False)[0]
                phase_record.setdefault('wait_including_compute_seconds',[]).append(time.monotonic()-start)
                assert result.prompt_token_ids==inputs['tokens'][0]
                traces=[json.loads(s) for s in Path(root_trace).read_text().splitlines()[before:]]
                assert any(matches_prefill(t,inputs['tokens'][:1]) for t in traces)
                if request:
                    observations=[json.loads(s) for s in Path(trace).read_text().splitlines()[projection_before:]]
                    assert observations and all(t['finite'] and 1 in t['slots'] for t in observations)
                logp=[v[t].logprob for v,t in zip(result.prompt_logprobs[1:],result.prompt_token_ids[1:],strict=True)]
                compare_logprobs(logp,logp);rows[phase].append(logp)
                phase_record.setdefault('generated_token_ids',[]).append(list(result.outputs[0].token_ids))
            record['phases'].append(phase_record);record['logprobs']=rows
            (output/'handoff-partial.json').write_text(json.dumps(record,indent=2)+'\n')
        record['assessment']=assess_cycle(rows)
        record['ok']=record['assessment']['ok']
        assert record['ok'],'identity passed but A/B/A output control failed; no handoff acceptance'
    except Exception:
        record['error']=traceback.format_exc()
    (output/'handoff.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    mode,out,source=sys.argv[1:4];out=Path(out);out.mkdir(parents=True,exist_ok=True)
    result=cpu(out,Path(source)) if mode=='cpu' else gpu(out,Path(source))
    sys.exit(0 if result['ok'] else 1)
