import json, collections, re
R="reference/industrial_posttrain_training_release"
pools={m:{e["case_id"] for e in json.load(open(f"{R}/data/batches/{m}/manifest.json"))["entries"]} for m in ["sft","eval","rl"]}
ev,train=pools["eval"],pools["sft"]|pools["rl"]
fam=lambda c: re.sub(r"_[a-z]\d+$","",c)   # 去掉尾部 _b01 / _f09 之类
fe=collections.Counter(fam(c) for c in ev); ft=collections.Counter(fam(c) for c in train)
shared=set(fe)&set(ft)
print(f"EVAL 家族数={len(fe)}  训练侧家族数={len(ft)}  共享家族={len(shared)}")
print(f"落在共享家族里的 EVAL case = {sum(fe[k] for k in shared)}/{len(ev)} "
      f"({sum(fe[k] for k in shared)/len(ev):.1%})")
print(f"EVAL 独占家族里的 case = {sum(fe[k] for k in set(fe)-shared)}")
print("\n样例（同家族，EVAL vs 训练侧）:")
for k in sorted(shared)[:4]:
    e=sorted(c for c in ev if fam(c)==k)[:2]; t=sorted(c for c in train if fam(c)==k)[:2]
    print(f"  {k}\n     EVAL: {e}\n     TRAIN:{t}")
