import pandas as pd, json, collections, hashlib, glob, os
R="reference/industrial_posttrain_training_release"
tr=pd.read_parquet(f"{R}/data/sft/stage5/train.parquet")
al=pd.concat([tr,pd.read_parquet(f"{R}/data/sft/stage5/val.parquet")])

print("### T5 结构签名去重（实体遮蔽后：intent + 工具调用名序列）")
def skel(row):
    seq=[]
    for m in row.messages:
        if m["role"]=="assistant" and m.get("tool_calls") is not None and len(m.get("tool_calls"))>0:
            for tc in m["tool_calls"]: seq.append(tc["function"]["name"])
    return (row.primary_intent, tuple(seq))
sig=[skel(r) for r in al.itertuples()]
c=collections.Counter(sig)
print("样本数:",len(sig),"  distinct 签名:",len(c))
print("重复组数(>1):",sum(1 for v in c.values() if v>1), " 涉及样本:",sum(v for v in c.values() if v>1))
print("最大重复条数:",max(c.values()))
print("Top5 重复签名:")
for k,v in c.most_common(5): print(f"  x{v}  {k[0]} | {'->'.join(k[1])[:120]}")
# 更严格：仅工具序列（跨 intent）
c2=collections.Counter(t for _,t in sig)
print("仅工具序列 distinct:",len(c2),"最大重复:",max(c2.values()))
# 完全相同的 messages（逐字）
mh=[hashlib.sha256(json.dumps(list(m),ensure_ascii=False,sort_keys=True,default=str).encode()).hexdigest() for m in al.messages]
print("逐字相同 messages 的重复组:",sum(1 for v in collections.Counter(mh).values() if v>1))

print("\n### T6 case 是否携带来源工单/订单字段")
cid=al.case_id.tolist()[0]
case=json.load(open(f"{R}/data/batches/sft/cases/{cid}.json"))
print("case top keys:",list(case.keys()))
print(json.dumps({k:v for k,v in case.items() if not isinstance(v,(list,dict))},ensure_ascii=False)[:800])
