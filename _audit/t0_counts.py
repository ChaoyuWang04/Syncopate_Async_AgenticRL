import json, pandas as pd, os
R="reference/industrial_posttrain_training_release"
for p in ["data/sft/stage5/train.parquet","data/sft/stage5/val.parquet",
          "data/rl/stage5/train.parquet","data/rl/stage5/val.parquet",
          "data/rl/eval/train.parquet","data/rl/eval/val.parquet"]:
    df=pd.read_parquet(os.path.join(R,p))
    print(f"{p:40s} rows={len(df):6d} cols={list(df.columns)}")
print()
for p in ["data/sft/stage5/all.jsonl","data/sft/stage5/train.jsonl","data/sft/stage5/val.jsonl",
          "data/rl/stage5/all.jsonl","data/rl/stage5/train.jsonl","data/rl/stage5/val.jsonl",
          "data/rl/eval/all.jsonl","data/rl/eval/train.jsonl","data/rl/eval/val.jsonl"]:
    n=sum(1 for _ in open(os.path.join(R,p)))
    print(f"{p:40s} lines={n}")
print()
for m in ["sft","stage5_full","eval","rl"]:
    d=json.load(open(f"{R}/data/batches/{m}/manifest.json"))
    print(m, "count=",d.get("count"), "pool=",d.get("pool"), "manifest_id=",d.get("manifest_id"),
          "source=",d.get("source"), "version=",d.get("version"), "verifier_version=",d.get("verifier_version"),
          "n_entries=",len(d.get("entries",[])), "include_extended=",d.get("include_extended"))
