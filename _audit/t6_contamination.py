import json, collections, os, hashlib, re
R="reference/industrial_posttrain_training_release"
B=f"{R}/data/batches/stage5_full"
pools={}
for m in ["sft","eval","rl"]:
    d=json.load(open(f"{R}/data/batches/{m}/manifest.json"))
    pools[m]={e["case_id"] for e in d["entries"]}
sft_only=pools["sft"]; ev=pools["eval"]; rl=pools["rl"]
train_side = sft_only|rl          # SFT ⊂ RL, 训练侧全集
print(f"训练侧(SFT∪RL)={len(train_side)}  EVAL={len(ev)}  case_id 交集={len(train_side&ev)}")

fields=["order_id","customer_id","customer_message","expected_policy_id","market"]
vals={f:{ "train":collections.Counter(), "eval":collections.Counter()} for f in fields}
norm=lambda s: re.sub(r"\s+","",s or "")
for cid in sorted(train_side|ev):
    p=f"{B}/cases/{cid}.json"
    if not os.path.exists(p): continue
    c=json.load(open(p))
    side="eval" if cid in ev else "train"
    for f in fields:
        v=c.get(f)
        if v is None: continue
        if f=="customer_message": v=norm(v)
        vals[f][side][v]+=1

print("\n=== 同一值同时出现在 EVAL 与训练侧的数量（模糊污染直接证据）===")
for f in fields:
    tr,evv=vals[f]["train"],vals[f]["eval"]
    shared=set(tr)&set(evv)
    n_ev_cases=sum(evv[k] for k in shared)
    print(f"{f:22s} distinct train={len(tr):5d} eval={len(evv):5d} | 共享值={len(shared):4d} | 受影响 EVAL case 数={n_ev_cases:4d}/{len(ev)}")

# 展示 customer_message 逐字重合的例子
tr,evv=vals["customer_message"]["train"],vals["customer_message"]["eval"]
sh=sorted(set(tr)&set(evv))
print(f"\ncustomer_message 逐字重合样例（共 {len(sh)} 个不同文本）：")
for s in sh[:5]:
    print(f"  train x{tr[s]} / eval x{evv[s]}  «{s[:60]}»")
