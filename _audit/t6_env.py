import json, collections, os, hashlib
R="reference/industrial_posttrain_training_release"; B=f"{R}/data/batches/stage5_full"
pools={m:{e["case_id"] for e in json.load(open(f"{R}/data/batches/{m}/manifest.json"))["entries"]} for m in ["sft","eval","rl"]}
ev=pools["eval"]; train=pools["sft"]|pools["rl"]
byorder=collections.defaultdict(lambda: {"train":[], "eval":[]})
for cid in sorted(train|ev):
    p=f"{B}/cases/{cid}.json"
    if not os.path.exists(p): continue
    c=json.load(open(p)); oid=c.get("order_id")
    if oid: byorder[oid]["eval" if cid in ev else "train"].append(cid)
shared={k:v for k,v in byorder.items() if v["train"] and v["eval"]}
print("共享 order_id 数:",len(shared))
e0=json.load(open(f"{B}/env_snapshots/{sorted(ev)[0]}.env.json"))
print("env_snapshot top keys:",list(e0.keys()))
h=lambda x: hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()[:12]
def find_order(snap,oid):
    out=[]
    def walk(o):
        if isinstance(o,dict):
            if o.get("order_id")==oid: out.append(o)
            for v in o.values(): walk(v)
        elif isinstance(o,list):
            for v in o: walk(v)
    walk(snap); return out
same=diff=miss=0; examples=[]
for oid,v in list(shared.items()):
    eo=find_order(json.load(open(f"{B}/env_snapshots/{v['eval'][0]}.env.json")),oid)
    to=find_order(json.load(open(f"{B}/env_snapshots/{v['train'][0]}.env.json")),oid)
    if not eo or not to: miss+=1; continue
    if h(eo[0])==h(to[0]):
        same+=1
        if len(examples)<3: examples.append((oid,v['eval'][0],v['train'][0]))
    else: diff+=1
print(f"同 order_id 的订单记录：完全相同={same}  不同={diff}  取不到={miss}")
for oid,e,t in examples: print(f"  {oid}: EVAL {e}  ==  TRAIN {t}")
