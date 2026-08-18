import glob, torch, gc
from safetensors import safe_open
KEYS=["model.layers.0.self_attn.q_proj.weight","model.layers.20.self_attn.v_proj.weight",
      "model.layers.35.mlp.down_proj.weight","model.layers.10.self_attn.o_proj.weight"]
def hw(d,key):
    for f in sorted(glob.glob(d+"/*.safetensors")):
        with safe_open(f,framework="pt") as fh:
            if key in fh.keys(): return fh.get_tensor(key)
    return None
print(f"{'层':<45}{'dtype':<10}{'RL−SFT ‖Δ‖':>14}{'不同元素':>12}{'最大|Δ|':>12}{'SFT−裸基座 ‖Δ‖':>16}")
for k in KEYS:
    b=hw("models/Qwen3-4B-sft-v13-e1",k); m=hw("models/Qwen3-4B-rl-v13-s110",k); r=hw("models/Qwen3-4B",k)
    if b is None or m is None: print(k,"缺"); continue
    d=(m.float()-b.float()); d2=(b.float()-r.float()) if r is not None else torch.zeros(1)
    print(f"{k:<45}{str(m.dtype):<10}{d.norm():>14.6f}{int((d!=0).sum()):>12}{d.abs().max():>12.3e}{d2.norm():>16.4f}")
print("\n--- 合并产物里的 lora_adapter/ ---")
for f in sorted(glob.glob("models/Qwen3-4B-rl-v13-s110/lora_adapter/*")):
    print("  ", f)
for f in sorted(glob.glob("models/Qwen3-4B-rl-v13-s110/lora_adapter/*.safetensors")):
    with safe_open(f,framework="pt") as fh:
        ks=sorted(fh.keys()); print("  张量数",len(ks))
        for k in ks[:4]:
            t=fh.get_tensor(k); print(f"    {k}  {tuple(t.shape)}  ‖·‖={t.float().norm():.6f}")
