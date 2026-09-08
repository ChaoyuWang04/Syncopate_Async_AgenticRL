"""B01 fixed-token cross-engine probe; no model updates or text round-tripping."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import sys

MODEL = Path('/vol/models/Qwen3.6-35B-A3B')
MODEL_ID = 'Qwen/Qwen3.6-35B-A3B'


def matches_prefill(trace, rows):
    ids=[t for row in rows for t in row]
    positions=[i for row in rows for i in range(len(row))]
    actual=trace['positions']
    return trace['tokens']==ids and (actual==positions or actual==[positions]*3)


def compare_logprobs(a, b):
    if not a or len(a) != len(b) or not all(math.isfinite(x) for x in [*a, *b]):
        raise ValueError('unaligned, empty or nonfinite probabilities')
    differences = sorted(abs(x-y) for x,y in zip(a,b))
    return {'max_abs': max(differences), 'mean_abs': sum(differences)/len(differences),
            'p95_abs': differences[math.ceil(.95*len(differences))-1], 'tokens': len(differences)}


def validate_tokens(rows, vocab_size):
    if not rows or any(len(row)<2 or any(type(t) is not int or not 0<=t<vocab_size for t in row) for row in rows):
        raise ValueError('invalid raw tokens')


def validate_shards(index, actual):
    expected=set(index['weight_map'].values())
    if not expected or expected != set(actual):
        raise ValueError('missing or extra checkpoint shards')


def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def cpu(output):
    # apply_model transfers only this probe's own callable over its private engine IPC.
    os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION']='1'
    from vllm.v1.serial_utils import MsgpackEncoder
    assert MsgpackEncoder().encode(lambda: 'probe-owned-function')
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer, AutoConfig
    from importlib.metadata import version
    metadata=MODEL/'.cache/huggingface/download/config.json.metadata'
    if not metadata.is_file(): raise RuntimeError('missing downloaded revision metadata')
    revision=metadata.read_text().splitlines()[0]
    info=HfApi().model_info(MODEL_ID,revision=revision,files_metadata=True)
    assert info.sha==revision
    for name in ['config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json']:
        if not (MODEL/name).is_file(): raise RuntimeError('missing metadata: '+name)
    required=[p for p in MODEL.iterdir() if p.suffix=='.safetensors' or p.name in {
        'config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json'}]
    validate_shards(json.loads((MODEL/'model.safetensors.index.json').read_text()),
                    [p.name for p in required if p.suffix=='.safetensors'])
    siblings={s.rfilename:s for s in info.siblings}
    manifests=[]
    for p in sorted(required):
        s=siblings[p.name];digest=sha_file(p)
        if s.lfs: assert digest==s.lfs.sha256,(p.name,'LFS SHA mismatch')
        else:
            data=p.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            assert blob==s.blob_id,(p.name,'Git blob mismatch')
        manifests.append({'path':str(p),'bytes':p.stat().st_size,'sha256':digest})
    config=AutoConfig.from_pretrained(MODEL,local_files_only=True)
    assert config.model_type=='qwen3_5_moe'
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    texts=['A training service collects model outputs and updates its weights. ',
           'An inference engine batches requests and returns token probabilities. ']
    rows=[tokenizer.encode(t*20,add_special_tokens=False)[:64] for t in texts]
    validate_tokens(rows,config.text_config.vocab_size)
    assert len(rows[0])==len(rows[1])==64 and rows[0]!=rows[1]
    payload={'model':MODEL_ID,'revision':revision,'model_path':str(MODEL), 'manifest':manifests,
             'tokens':rows,'positions':list(range(64)),'attention_mask':[1]*64,
             'config':config.to_dict(),'versions':{p:version(p) for p in ['torch','transformers','vllm','verl','peft']}}
    payload['tokens_sha256']=hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    (output/'input.json').write_text(json.dumps(payload,indent=2)+'\n')
    return {'ok':True,'revision':revision,'weights_bytes':sum(p['bytes'] for p in manifests),
            'tokens_sha256':payload['tokens_sha256'],'versions':payload['versions']}


def verify_model(inputs):
    if inputs['model_path'] != str(MODEL) or inputs['model'] != MODEL_ID:
        raise ValueError('model path or repository changed')
    validate_shards(json.loads((MODEL/'model.safetensors.index.json').read_text()),
                    [p.name for p in MODEL.glob('*.safetensors')])
    for item in inputs['manifest']:
        path=Path(item['path'])
        if path.parent != MODEL or path.stat().st_size != item['bytes'] or sha_file(path) != item['sha256']:
            raise ValueError('model bytes changed since CPU check: '+str(path))


def hf(output,inputs):
    import torch
    from transformers import AutoModelForImageTextToText
    model=AutoModelForImageTextToText.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,
                                                     attn_implementation='sdpa',device_map='cuda').eval()
    assert not any('lora_' in n for n,_ in model.named_parameters())
    records=[]
    with torch.inference_mode():
        for batch in [inputs['tokens'][:1]]*3+[inputs['tokens']]*3:
            ids=torch.tensor(batch,device='cuda')
            positions=torch.arange(ids.shape[1],device='cuda').unsqueeze(0).expand_as(ids)
            logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=positions,use_cache=False).logits
            lp=logits[0,:-1].float().log_softmax(-1).gather(-1,ids[0,1:,None]).squeeze(-1)
            assert torch.isfinite(lp).all()
            records.append({'batch_size':len(batch),'logprobs':lp.cpu().tolist()})
    result={'ok':True,'engine':'hf','class':type(model).__name__,'config':model.config.to_dict(),'records':records,
            'peak_allocated':torch.cuda.max_memory_allocated()}
    (output/'hf.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def vllm(output,inputs, *, fp8=False):
    os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION']='1'
    from vllm import LLM, SamplingParams
    from syncopate.infra.fp8_probe import quantization_kwargs, inspect_model
    quant_kwargs=quantization_kwargs() if fp8 else {'kv_cache_dtype':'auto'}
    llm=LLM(model=str(MODEL),dtype='bfloat16',**quant_kwargs,tensor_parallel_size=1,max_model_len=256,
            max_num_seqs=2,max_num_batched_tokens=512,gpu_memory_utilization=.70,
            enable_prefix_caching=False,seed=137,limit_mm_per_prompt={'image':0,'video':0})
    trace_path=str(output/'vllm-root.jsonl')
    def install_trace(model):
        def observe(module,args,kwargs):
            ids=kwargs.get('input_ids',args[0] if args else None)
            pos=kwargs.get('positions',args[1] if len(args)>1 else None)
            if ids is not None and pos is not None:
                with open(trace_path,'a') as stream:
                    stream.write(json.dumps({'tokens':ids.detach().cpu().tolist(),
                                             'positions':pos.detach().cpu().tolist()})+'\n')
        model.register_forward_pre_hook(observe,with_kwargs=True)
        return type(model).__name__
    model_classes=llm.apply_model(install_trace)
    inventory=llm.apply_model(inspect_model) if fp8 else []
    (output/'quantization-inventory.json').write_text(json.dumps(inventory,indent=2))
    params=SamplingParams(temperature=1,top_p=1,top_k=-1,max_tokens=1,prompt_logprobs=1,seed=137)
    records=[]
    for batch in [inputs['tokens'][:1]]*3+[inputs['tokens']]*3:
        before=len(Path(trace_path).read_text().splitlines()) if Path(trace_path).exists() else 0
        llm.sleep(level=0)
        try:
            llm.enqueue([{'prompt_token_ids':r} for r in batch],params,use_tqdm=False)
        finally:
            llm.wake_up(tags=['scheduling'])
        outputs=llm.wait_for_completion(use_tqdm=False)
        traces=[json.loads(line) for line in Path(trace_path).read_text().splitlines()[before:]]
        actual_prefill=any(matches_prefill(t,batch) for t in traces)
        target=outputs[0]
        assert target.prompt_token_ids==inputs['tokens'][0]
        lp=[d[t].logprob for d,t in zip(target.prompt_logprobs[1:],target.prompt_token_ids[1:],strict=True)]
        compare_logprobs(lp,lp)
        records.append({'batch_size':len(batch),'logprobs':lp,'observed_same_batch':actual_prefill})
        (output/'vllm-partial.json').write_text(json.dumps({'records':records},indent=2)+'\n')
        assert actual_prefill, 'requested batch was not observed together at model root'
    result={'ok':True,'engine':'vllm','quantization':quant_kwargs,'inventory':inventory,'model_classes':model_classes,'records':records}
    (output/'vllm.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def summarize(output):
    records={e:json.loads((output/(e+'.json')).read_text())['records'] for e in ['hf','vllm']}
    metrics={}
    for e,rows in records.items():
        metrics[e+'_repeat_single']=max(compare_logprobs(rows[i]['logprobs'],rows[j]['logprobs'])['max_abs'] for i in range(3) for j in range(i+1,3))
        metrics[e+'_repeat_batch']=max(compare_logprobs(rows[i]['logprobs'],rows[j]['logprobs'])['max_abs'] for i in range(3,6) for j in range(i+1,6))
        metrics[e+'_batch']=compare_logprobs(rows[0]['logprobs'],rows[3]['logprobs'])
    for i,name in [(0,'single'),(3,'batch')]:
        metrics['cross_'+name]=compare_logprobs(records['hf'][i]['logprobs'],records['vllm'][i]['logprobs'])
    return {'ok':True,'metrics':metrics,'scope':'fixed prompt tokens, BF16 forward only; no speed conclusion'}


if __name__=='__main__':
    mode=sys.argv[1];out=Path(sys.argv[2])
    if mode=='cpu': result=cpu(out)
    elif mode=='summary': result=summarize(out)
    elif mode=='verify':
        inputs=json.loads(Path(sys.argv[3]).read_text());verify_model(inputs);result={'ok':True,'manifest_verified':True}
    else:
        inputs=json.loads(Path(sys.argv[3]).read_text())
        result=vllm(out,inputs,fp8=True) if mode=='fp8' else {'hf':hf,'vllm':vllm}[mode](out,inputs)
    (out/(mode+'-result.json')).write_text(json.dumps(result,indent=2)+'\n')
