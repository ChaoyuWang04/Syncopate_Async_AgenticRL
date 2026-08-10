"""T4: SFT train.parquet token 构成（真实 Qwen3 tokenizer, transformers 5.x）。只读。"""
import pandas as pd, numpy as np
from transformers import AutoTokenizer
R="reference/industrial_posttrain_training_release"
tok=AutoTokenizer.from_pretrained("/home/samwang/code/projects/models/Qwen3-0.6B")
tr=pd.read_parquet(f"{R}/data/sft/stage5/train.parquet")
def py(o):
    if isinstance(o,np.ndarray): return [py(x) for x in o.tolist()]
    if isinstance(o,dict): return {k:py(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [py(x) for x in o]
    if isinstance(o,np.generic): return o.item()
    return o
tools=py(tr.tools.iloc[0])
def enc(m):
    o=tok.apply_chat_template(m,tools=tools,tokenize=True,add_generation_prompt=False,enable_thinking=False)
    return len(o["input_ids"])
rows=[]
for r in tr.itertuples():
    msgs=py(r.messages)
    for m in msgs:
        if not m.get("tool_calls"): m.pop("tool_calls",None)
    cum=[0]+[enc(msgs[:i]) for i in range(1,len(msgs)+1)]
    total=cum[-1]; final=cum[-1]-cum[-2]
    assist=sum(cum[i+1]-cum[i] for i,m in enumerate(msgs) if m["role"]=="assistant")
    rows.append(dict(total=total,assistant=assist,final=final,n_msg=len(msgs)))
d=pd.DataFrame(rows)
print("=== [T4] SFT train 121 条 token 统计 (Qwen3 tokenizer, enable_thinking=False) ===")
print(d.describe(percentiles=[.5,.9,.99]).round(1).to_string())
print(f"\n合计: total={d.total.sum()}  assistant(≈有loss)={d.assistant.sum()}  终答段={d.final.sum()}")
print(f"assistant 占全序列           : {d.assistant.sum()/d.total.sum():.1%}")
print(f"★ 终答段 / 全部有 loss token  : {d.final.sum()/d.assistant.sum():.1%}")
q=d.final/d.assistant
print(f"逐条终答占比 中位数 {q.median():.1%} | P10 {q.quantile(.1):.1%} | P90 {q.quantile(.9):.1%}")
print(f"\n最长样本 total={d.total.max()}  (max_length=12288 → 截断风险: {'有' if d.total.max()>12288 else '无'})")
d.to_csv("_audit/t4_tokens.csv",index=False)
