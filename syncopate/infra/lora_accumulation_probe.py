"""B02: official verl SFT + FSDP2, q/v LoRA, unequal token counts and tail window."""
from __future__ import annotations
import copy
from datetime import timedelta
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import traceback


def strict_gradients(reference, actual):
    import torch
    from syncopate.infra.training_probe import compare_gradients
    result=compare_gradients(reference,actual)
    coordinates={n:bool(torch.allclose(reference[n],actual[n],atol=1e-6,rtol=1e-4))
                 for n in reference.keys() & actual.keys()}
    result['coordinate_allclose']=coordinates
    result['ok']=result['ok'] and bool(coordinates) and all(coordinates.values())
    return result


def trainable_optimizer(state, trainable, all_names):
    names=[n for group in state['param_groups'] for n in group['params']]
    if len(names)!=len(set(names)) or not trainable<=set(names)<=all_names or not set(state['states'])<=trainable:
        raise ValueError('invalid optimizer coverage or state on frozen parameters')
    projected=copy.deepcopy(state)
    for group in projected['param_groups']:
        group['params']=[n for n in group['params'] if n in trainable]
    projected['param_groups']=[group for group in projected['param_groups'] if group['params']]
    return projected


def windows():
    result=[]
    for step,(lengths,counts) in enumerate([([7,11,9,13],[3,5,4,7]),([6,10],[2,6])]):
        rows=[]
        for i,(length,count) in enumerate(zip(lengths,counts)):
            rows.append({'row_id':i,'input_ids':[3+(i*23+p*7+step*17)%125 for p in range(length)],
                         'position_ids':list(range(length)), 'loss_mask':[int(p>=length-count) for p in range(length)]})
        result.append(rows)
    return result


def batch(rows, *, denominator=None, dp_size=1):
    import torch
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    values={key:torch.nested.as_nested_tensor([torch.tensor(r[key],dtype=torch.float32 if key=='loss_mask' else torch.long)
                                             for r in rows],layout=torch.jagged)
            for key in ['input_ids','position_ids','loss_mask']}
    values.update(row_id=torch.tensor([r['row_id'] for r in rows]),temperature=torch.ones(len(rows)))
    td=TensorDict(values,batch_size=[len(rows)])
    tu.assign_non_tensor(td,use_dynamic_bsz=False,micro_batch_size_per_gpu=1,use_remove_padding=False,
                        use_fused_kernels=False,return_model_output=False,dp_size=dp_size,
                        batch_num_tokens=denominator or sum(sum(r['loss_mask']) for r in rows))
    return td


def lora(model):
    from peft import LoraConfig,TaskType,get_peft_model
    return get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=4,lora_alpha=8,
                                         target_modules=['q_proj','v_proj'],lora_dropout=0,bias='none'))


def full_loss(model, rows):
    import torch
    import torch.nn.functional as F
    device=next(model.parameters()).device;width=max(len(r['input_ids']) for r in rows)
    ids=torch.tensor([r['input_ids']+[0]*(width-len(r['input_ids'])) for r in rows],device=device)
    attn=torch.tensor([[1]*len(r['input_ids'])+[0]*(width-len(r['input_ids'])) for r in rows],device=device)
    mask=torch.tensor([r['loss_mask']+[0]*(width-len(r['input_ids'])) for r in rows],device=device)
    pos=torch.arange(width,device=device).unsqueeze(0).expand_as(ids)
    logits=model(input_ids=ids,attention_mask=attn,position_ids=pos,use_cache=False).logits
    per_token=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='none').reshape(ids.shape[0],-1)
    return (per_token*mask[:,1:]).sum()/mask.sum()


def engine_configs(tiny):
    from syncopate.infra.training_probe import build_engine_configs
    configs=build_engine_configs(str(tiny),False)
    from dataclasses import replace
    configs['model_config']=replace(configs['model_config'],lora_rank=4,lora_alpha=8,
                                    target_modules=['q_proj','v_proj'])
    return configs


def cpu(out):
    import torch
    from syncopate.infra.training_probe import build_tiny_model,capture_gradients,compare_gradients
    from verl.workers.utils.losses import sft_loss
    from verl.utils import tensordict_utils as tu
    torch.set_num_threads(2)
    torch.random.default_generator.manual_seed(137)
    assert not torch.cuda.is_initialized()
    assert not strict_gradients({'x':torch.tensor([1000.,0.])},{'x':torch.tensor([1000.,.001])})['ok']
    tiny=out/'tiny';build_tiny_model().save_pretrained(tiny)
    cfg=engine_configs(tiny)
    assert cfg['model_config'].lora_rank==4 and cfg['model_config'].target_modules==['q_proj','v_proj']
    base=build_tiny_model();model=lora(base).train()
    initial=copy.deepcopy(model.state_dict());checks=[]
    for rows in windows():
        model.load_state_dict(initial);model.zero_grad(set_to_none=True)
        loss=full_loss(model,rows);loss.backward();reference=capture_gradients(model)
        total=sum(sum(r['loss_mask']) for r in rows)
        metadata=batch(rows,denominator=total)
        assert tu.get_non_tensor_data(metadata,'batch_num_tokens',None)==total
        assert tu.get_non_tensor_data(metadata,'dp_size',None)==1
        model.zero_grad(set_to_none=True)
        for row in rows:
            ids=torch.tensor([row['input_ids']]);logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),
                position_ids=torch.arange(ids.shape[1]).unsqueeze(0),use_cache=False).logits
            labels=ids.roll(-1,-1)
            lp=logits.float().log_softmax(-1).gather(-1,labels.unsqueeze(-1)).squeeze(-1)
            nested=torch.nested.as_nested_tensor([lp[0]],layout=torch.jagged)
            actual,_=sft_loss(None,{'log_probs':nested},batch([row],denominator=total))
            actual.backward()
        check=strict_gradients(reference,capture_gradients(model));assert check['ok'],check
        model.zero_grad(set_to_none=True)
        for row in rows: (full_loss(model,[row])/len(rows)).backward()
        wrong=strict_gradients(reference,capture_gradients(model))
        assert not wrong['ok'], 'negative control failed to detect micro-mean weighting'
        checks.append({'tokens':total,'correct':check,'wrong_micro_mean':wrong})
    model.save_pretrained(out/'adapter')
    from peft import PeftModel
    restored=PeftModel.from_pretrained(build_tiny_model(),out/'adapter',is_trainable=True)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    sources={inspect.getfile(sft_loss):hashlib.sha256(Path(inspect.getfile(sft_loss)).read_bytes()).hexdigest()}
    result={'ok':True,'checks':checks,'adapter_reload_exact':True,'cuda_initialized':torch.cuda.is_initialized(),
            'sources':sources,'tiny_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tiny.iterdir()},
            'lora_trainable':[n for n,p in model.named_parameters() if p.requires_grad]}
    assert not result['cuda_initialized']
    return result


def worker(out,tiny):
    import torch
    import torch.distributed as dist
    from transformers import Qwen3_5MoeForCausalLM
    from verl.workers.engine import EngineRegistry
    from verl.workers.utils.losses import sft_loss
    from verl.utils import tensordict_utils as tu
    from syncopate.infra.training_probe import (build_engine_configs,_capture_model,_capture_optimizer,
        capture_gradients,compare_gradients,compare_exact_trees,same_state_optimizer_reference,
        compare_optimizer_transition,_compare_parameters,_collective_require,_cross_rank_check)
    rank=int(os.environ['RANK']);result={'rank':rank,'ok':False,'steps':[]}
    try:
        assert int(os.environ['WORLD_SIZE'])==2 and torch.cuda.device_count()==2
        torch.cuda.set_device(rank);torch.set_num_threads(1);torch.manual_seed(137)
        assert torch.cuda.get_device_capability()==(10,0)
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        dist.init_process_group('nccl',timeout=timedelta(seconds=90),device_id=torch.device('cuda',rank))
        configs=engine_configs(tiny)
        engine=EngineRegistry.new(model_type='language_model',backend='fsdp2',**configs);engine.initialize()
        assert engine._is_lora and engine.get_data_parallel_size()==2
        reference=lora(Qwen3_5MoeForCausalLM.from_pretrained(tiny,dtype=torch.float32,
            attn_implementation='eager',experts_implementation='eager')).to(torch.device('cuda',rank)).train()
        root_calls=[]
        expected_roots=[r for rows in windows() for r in rows[rank*(len(rows)//2):(rank+1)*(len(rows)//2)]]
        def observe_root(module,args,kwargs):
            expected=expected_roots[len(root_calls)]
            ids=kwargs['input_ids'].detach().cpu();pos=kwargs['position_ids'].detach().cpu()
            length=len(expected['input_ids']);width=ids.shape[-1]
            assert ids.numel()==width and width>=length
            assert ids.reshape(-1)[:length].tolist()==expected['input_ids']
            assert not ids.reshape(-1)[length:].any()
            assert all(row[:length].tolist()==expected['position_ids'] for row in pos.reshape(-1,width))
            root_calls.append({'shape':list(ids.shape),'tokens':ids.tolist(),'positions':pos.tolist(),'input_equal':True})
        handle=engine.module.register_forward_pre_hook(observe_root,with_kwargs=True)
        frozen_names={n for n,p in engine.module.named_parameters() if not p.requires_grad}
        names={n for n,p in engine.module.named_parameters() if p.requires_grad}
        assert names and all('lora_' in n for n in names)
        result['trainable_names']=sorted(names)
        result['device']={'name':torch.cuda.get_device_name(),'uuid':str(torch.cuda.get_device_properties(rank).uuid)}
        for step,rows in enumerate(windows(),1):
            local=rows[rank*(len(rows)//2):(rank+1)*(len(rows)//2)]
            total=sum(sum(r['loss_mask']) for r in rows)
            before=_capture_model(engine.module);opt_full_before=_capture_optimizer(engine.module,engine.optimizer)
            opt_before=trainable_optimizer(opt_full_before,names,set(before))
            reference.load_state_dict(before,strict=True);reference.zero_grad(set_to_none=True)
            assert compare_exact_trees(before,_capture_model(reference))['ok']
            loss=full_loss(reference,rows);loss.backward();expected_grad=capture_gradients(reference)
            callbacks=[]
            def observed_loss(*,model_output,data,dp_group):
                tokens=tu.get_non_tensor_data(data,'batch_num_tokens',None)
                dp=tu.get_non_tensor_data(data,'dp_size',None)
                assert tokens==total and dp==2 and len(data)==1
                callbacks.append({'global_tokens':tokens,'dp_size':dp,'row_id':int(data['row_id'].item())})
                return sft_loss(None,model_output,data,dp_group)
            with engine.train_mode(zero_grad_on_exit=False):
                engine.optimizer_zero_grad()
                engine.forward_backward_batch(batch(local),observed_loss,False)
                gradients=capture_gradients(engine.module)
                grad_check=strict_gradients(expected_grad,gradients)
                nonzero=sum(g.double().square().sum().item() for g in gradients.values())>0
                parameter_before={n:before[n] for n in names}
                grouped=[n for g in opt_before['param_groups'] for n in g['params']]
                assert set(grouped)==names
                pre={'step':step,'gradient_check':grad_check,'nonzero':nonzero,'callbacks':callbacks,
                     'reference_loss':loss.item(),'actual_micro_count':len(callbacks)}
                (out/f'rank-{rank}-step-{step}-pre.json').write_text(json.dumps(pre,indent=2))
                torch.save({'reference_gradients':expected_grad,'actual_gradients':gradients,'before':before,
                            'optimizer_before':opt_before},out/f'rank-{rank}-step-{step}.pt')
                _collective_require(grad_check['ok'] and nonzero and len(callbacks)==len(local), 'gradient or micro gate failed')
                same=same_state_optimizer_reference(parameter_before,opt_before,gradients,step=step)
                independent=same_state_optimizer_reference(parameter_before,opt_before,expected_grad,step=step)
                engine.optimizer_step();engine.lr_scheduler_step()
            after=_capture_model(engine.module);actual={n:after[n] for n in names}
            opt_full_after=_capture_optimizer(engine.module,engine.optimizer)
            optimizer=trainable_optimizer(opt_full_after,names,set(after))
            assert compare_exact_trees(opt_full_before['param_groups'],opt_full_after['param_groups'])['ok']
            parameter_check=_compare_parameters(same['parameters'],actual)
            independent_check=_compare_parameters(independent['parameters'],actual)
            moments=compare_optimizer_transition(opt_before,gradients,same['optimizer'],optimizer)
            frozen=all(torch.equal(before[n],after[n]) for n in frozen_names)
            changed=any(not torch.equal(before[n],after[n]) for n in names)
            rank_check=_cross_rank_check({'weights':after,'optimizer':optimizer},engine)
            record={**pre,'parameter_check':parameter_check,'independent_gradient_update':independent_check,
                    'moments':moments,'frozen_unchanged':frozen,'adapter_changed':changed,'rank_check':rank_check,
                    'optimizer_full_groups':opt_full_after['param_groups'],
                    'optimizer_steps':{n:float(v['step']) for n,v in optimizer['states'].items()}}
            result['steps'].append(record)
            _collective_require(parameter_check['ok'] and independent_check['ok'] and moments['ok'] and frozen and changed
                                and rank_check['ok'] and all(v==step for v in record['optimizer_steps'].values()),'AdamW or frozen/rank gate failed')
        handle.remove();result['root_calls']=root_calls
        assert len(root_calls)==3
        result['ok']=True
    except Exception:
        result['error']=traceback.format_exc()
    finally:
        (out/f'rank-{rank}.json').write_text(json.dumps(result,indent=2))
        if dist.is_initialized():dist.destroy_process_group()
    return result


if __name__=='__main__':
    mode=sys.argv[1];out=Path(sys.argv[2])
    if mode=='cpu':
        result=cpu(out);(out/'cpu-result.json').write_text(json.dumps(result,indent=2)+'\n')
    else: result=worker(out,Path(sys.argv[3]))
    sys.exit(0 if result['ok'] else 1)
