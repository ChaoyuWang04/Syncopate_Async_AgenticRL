import pandas as pd, json, collections, hashlib
R="reference/industrial_posttrain_training_release"
tr=pd.read_parquet(f"{R}/data/sft/stage5/train.parquet")
va=pd.read_parquet(f"{R}/data/sft/stage5/val.parquet")
al=pd.concat([tr,va]); 
print("=== rows: train/val/all ===", len(tr), len(va), len(al))
print("\n=== [T4] difficulty (train) ===");print(tr.difficulty.value_counts(dropna=False).to_string())
print("\n=== [T4] primary_intent (train) ===");print(tr.primary_intent.value_counts(dropna=False).to_string())
print("\n=== [T4] routing_bucket (train) ===");print(tr.routing_bucket.value_counts(dropna=False).to_string())
print("\n=== [T4] gold_reward (train) ===");print(tr.gold_reward.describe().to_string())
print("uniq gold_reward:",sorted(set(tr.gold_reward)))
print("\n=== [T4] enable_thinking values ===",set(al.enable_thinking))
# tools column identity
th=[hashlib.sha256(json.dumps(list(t),ensure_ascii=False,sort_keys=True,default=str).encode()).hexdigest()[:16] for t in al.tools]
print("=== [T4] distinct tools-column hashes ===",len(set(th)), set(th))
print("n tools per row:",set(len(t) for t in al.tools))
# assistant turn counts
def stats(msgs):
    roles=[m["role"] for m in msgs]
    n_assist=sum(1 for r in roles if r=="assistant")
    n_tool=sum(1 for r in roles if r=="tool")
    return n_assist,n_tool,len(roles)
S=[stats(m) for m in tr.messages]
print("\n=== [T4] assistant msgs per sample (train) ===")
print(pd.Series([s[0] for s in S]).value_counts().sort_index().to_string())
print("=== tool msgs per sample ===")
print(pd.Series([s[1] for s in S]).value_counts().sort_index().to_string())
print("=== tool-call rounds (=assistant-1) distribution 1/2/3+ ===")
rounds=pd.Series([s[0]-1 for s in S])
print(pd.Series(pd.cut(rounds,[-1,0,1,2,100],labels=["0","1","2","3+"])).value_counts().sort_index().to_string())
# intent x difficulty
print("\n=== [T3] intent x difficulty (train) ===")
print(pd.crosstab(tr.primary_intent,tr.difficulty).to_string())
