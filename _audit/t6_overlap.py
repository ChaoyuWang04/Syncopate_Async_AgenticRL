import json
R="reference/industrial_posttrain_training_release"
pools={}
for m in ["sft","eval","rl","stage5_full"]:
    d=json.load(open(f"{R}/data/batches/{m}/manifest.json"))
    e=d["entries"]
    print(m, "entry keys:", list(e[0].keys()) if e else None)
    pools[m]=set(x.get("case_id") or x.get("id") for x in e)
print()
for a in ["sft","eval","rl"]:
    for b in ["sft","eval","rl"]:
        if a<b:
            print(f"{a} ∩ {b} = {len(pools[a]&pools[b])}")
print()
full=pools["stage5_full"]
union=pools["sft"]|pools["eval"]|pools["rl"]
print("|full|",len(full),"|sft∪eval∪rl|",len(union))
print("in full not in union:",len(full-union))
print("in union not in full:",len(union-full))
print("sample entry sft:", json.dumps(json.load(open(f"{R}/data/batches/sft/manifest.json"))["entries"][0],ensure_ascii=False)[:600])
