#!/usr/bin/env python3
"""E26 · PrefixGrouper 的**净收益**（不是纸面上界）。

⚠️ 三臂而不是两臂 —— 因为能跑通打包的后端是 SDPA，而生产今天用的是 FA2：
    A 生产现状   逐条 ×G · flash_attention_2
    B 同后端对照 逐条 ×G · SDPA（右下对齐）      ← 用来拆开"打包的收益"和"换后端的代价"
    C 打包       PrefixGrouper · SDPA
  真正该报的数是 **A → C**（换掉整条路之后端到端快多少），B 只是用来解释成因。
"""
import argparse, os, statistics as st, time, torch, torch.nn.functional as F

def brcausal(module,q,k,v,am,*a,scaling=None,dropout=0.0,**kw):
    Lq,Lk=q.shape[-2],k.shape[-2]
    i=torch.arange(Lq,device=q.device).view(-1,1); j=torch.arange(Lk,device=q.device).view(1,-1)
    o=F.scaled_dot_product_attention(q,k,v,attn_mask=(j<=i+(Lk-Lq)),scale=scaling,enable_gqa=True)
    return o.transpose(1,2).contiguous(), None

def build(impl, lora=True, gc=True):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    m=AutoModelForCausalLM.from_pretrained(MP,dtype=torch.bfloat16,
        attn_implementation=impl,trust_remote_code=True).to("cuda")
    if lora:
        m=get_peft_model(m,LoraConfig(r=32,lora_alpha=64,lora_dropout=0.0,bias="none",
            target_modules="all-linear",task_type="CAUSAL_LM"))
    m.config.use_cache=False
    if gc:
        m.enable_input_require_grads()
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    m.train(); return m

def bench(fn,label,it):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); ts=[]
    for i in range(it+1):
        torch.cuda.synchronize(); t=time.time(); fn(); torch.cuda.synchronize()
        if i: ts.append(time.time()-t)
    s=st.median(ts); g=torch.cuda.max_memory_allocated()/2**30
    print(f"  {label:34s} {s:7.3f} s   峰值 {g:5.2f} GB"); return s,g

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--prefix-len",type=int,default=4196); ap.add_argument("--resp-len",type=int,default=654)
    ap.add_argument("--group",type=int,default=8); ap.add_argument("--iters",type=int,default=2)
    ap.add_argument("--arm",required=True,choices=["A","B","C","D"])
    a=ap.parse_args()
    os.environ["SYNCOPATE_PREFIX_GROUPER"]="1"
    from transformers import AutoConfig
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["brcausal"]=brcausal
    from syncopate.train import verl_patches as vp
    from verl.models.transformers import monkey_patch as mp
    from verl.trainer.ppo import prefix_grouper_utils as pgu
    vp._patch_prefix_grouper(); mp.apply_prefix_grouper_patch()
    ALL_ATTENTION_FUNCTIONS["brcausal"]=mp._create_prefix_grouper_wrapper(brcausal)
    global MP; MP="models/Qwen3-4B-sft-v13-e1"
    P,R,G=a.prefix_len,a.resp_len,a.group
    V=AutoConfig.from_pretrained(MP,trust_remote_code=True).vocab_size
    torch.manual_seed(0)
    pre=torch.randint(1,V,(1,P),device="cuda").expand(G,-1).contiguous()
    resp=torch.randint(1,V,(G,R),device="cuda")
    impl={"A":"flash_attention_2","B":"brcausal","C":"brcausal","D":"flash_attention_2"}[a.arm]
    m=build(impl)
    if a.arm in ("A","B"):
        def run():
            for i in range(G):
                ids=torch.cat([pre[i:i+1],resp[i:i+1]],1)
                lg=m(input_ids=ids,attention_mask=torch.ones_like(ids)).logits
                lp=torch.log_softmax(lg[:,P-1:P+R-1].float(),-1).gather(-1,resp[i:i+1].unsqueeze(-1))
                (-lp.mean()/G).backward()
        lab={"A":"A 生产现状（逐条 · FA2）","B":"B 同后端对照（逐条 · SDPA）"}[a.arm]
    else:
        mb=dict(prompts=pre,responses=resp,response_mask=torch.ones(G,R,dtype=torch.long,device="cuda"),
                attention_mask=torch.ones(G,P+R,dtype=torch.long,device="cuda"),pad_token_id=0)
        def run():
            _,lp=pgu.forward_micro_batch_with_prefix_grouper(micro_batch=mb,model=m,temperature=1.0,
                calculate_entropy=False,device_name="cuda",param_dtype=torch.bfloat16)
            (-lp.float().mean()).backward()
        lab="C/D 打包（PrefixGrouper · "+impl+"）"
    try:
        s,g=bench(run,lab,a.iters); print(f"MEASURE {a.arm} {s:.4f} {g:.3f}")
    except torch.cuda.OutOfMemoryError:
        print(f"MEASURE {a.arm} OOM")
