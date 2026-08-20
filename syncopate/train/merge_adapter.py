"""把 SFT 的 LoRA 合并进基座权重，产出 RL 的起点模型。

★★★ 为什么要合并，而不是用 verl 的 `lora_adapter_path`

verl 确实能从已有 adapter 继续训（`transformer_impl.py:_build_lora_module`，
`PeftModel.from_pretrained(..., is_trainable=True)`）。但它同时决定了 reference policy：

    ref_in_actor = lora_rank > 0 or lora_adapter_path is not None

用 LoRA 时 verl 算 reference 的办法是**把 adapter 关掉**——关掉之后得到的是**基座**。
于是 KL 项会一直把模型往「没学过业务的那个基座」拉，把 SFT 学到的东西往回拽。
手册 §24③ 说的正是这件事：**reference 必须指向 SFT ckpt，不是 base。**

合并之后：

    新基座 = SFT 模型   ⇒ RL 的起点对了
    关掉 adapter = SFT  ⇒ reference 也对了

一次合并同时解决两件事，而且不依赖 verl 那条 adapter 加载路径的细节行为。

代价是磁盘上多一份 8GB 的模型。相对于「跑两小时 RL 结果起点是错的」，这个代价可以忽略。

    python -m syncopate.train.merge_adapter \
        --base models/Qwen3-4B --adapter checkpoints/sft/v3_ctrl/epoch1 \
        --out models/Qwen3-4B-sft-e1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]



def _assert_merge_landed(base_path: str, adapter_path: str, merged_model, max_resid: float = 0.5) -> None:
    """判据：合并后的权重 − 基座权重 ≈ adapter 的 ΔW_eff，**按全部被适配层加权**。

    量两件事（**必须分开看**）：
        保真残差 = ‖kept − Δ‖ / ‖Δ‖      ← 判据。方向对不对
        幅度比   = ‖kept‖ / ‖Δ‖          ← 参考。只看它会得出错误的安心结论

    ⚠️ 判据口径改过一次（2026-08-19，v13r2 首次开火时抓到的）：
    第一版只探**一个**层（safetensors 排序里的第一个 lora_A），而那恰好是全模型
    252 层里残差**最差**的一层（layers.0.mlp.down_proj，Δ 只占 0.21% ⇒ 残差 0.58）；
    同一份 adapter 的**全局加权残差是 0.307**，比被接受的先例（老 v13-e1 标定 0.36，
    实测任务分代价 −0.025，E24）还好。⇒ 拿最差层代表整体 = 尺子错了不是数据错了
    （同「位移口径错 33 倍」那次）。门槛 0.5 **不动**，改的是被量的对象：
    全局加权残差判决，最差 5 层照打出来供人看。
    ⚠️ 合并后仍要用任务级尺子兜底：merged 审计 vs adapter 审计配对（E24 协议）。
    """
    from safetensors import safe_open

    cfg = json.loads(Path(adapter_path, "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    files = sorted(Path(adapter_path).glob("*.safetensors"))
    if not files:
        print("[合并] ⚠️ adapter 不是 safetensors，跳过合并校验")
        return
    ab: dict[str, torch.Tensor] = {}
    for f in files:
        with safe_open(f, framework="pt") as fh:
            for k in fh.keys():
                if "lora_A" in k or "lora_B" in k:
                    ab[k] = fh.get_tensor(k)
    stems = sorted({k.split(".lora_A")[0] for k in ab if ".lora_A" in k})
    if not stems:
        print("[合并] ⚠️ adapter 里没有 lora_A，跳过合并校验")
        return
    sd = merged_model.state_dict()
    base_idx: dict[str, Path] = {}
    for f in sorted(Path(base_path).glob("*.safetensors")):
        with safe_open(f, framework="pt") as fh:
            for k in fh.keys():
                base_idx[k] = f

    per_layer: list[tuple[float, float, str]] = []   # (resid, ‖Δ‖, name)
    err2 = d2 = kept2 = 0.0
    for stem in stems:
        name = stem.replace("base_model.model.", "") + ".weight"
        if name not in sd or name not in base_idx:
            continue
        A = ab[stem + ".lora_A.weight"].float()
        B = ab[stem + ".lora_B.weight"].float()
        delta = scale * (B @ A)
        with safe_open(base_idx[name], framework="pt") as fh:
            base_w = fh.get_tensor(name).float()
        kept = sd[name].float().cpu() - base_w
        if kept.norm().item() == 0.0:
            raise SystemExit(f"🔴 合并后 {name} 与基座**逐位相同** ⇒ 增量根本没进权重。")
        dn = delta.norm().item()
        resid = ((kept - delta).norm() / delta.norm()).item()
        per_layer.append((resid, dn, name))
        err2 += (resid ** 2) * (dn ** 2)
        d2 += dn ** 2
        kept2 += kept.norm().item() ** 2

    global_resid = (err2 / d2) ** 0.5
    mag = (kept2 / d2) ** 0.5
    worst = sorted(per_layer, reverse=True)[:5]
    print(f"[合并] 校验 {len(per_layer)} 个被适配层: 全局加权残差 {global_resid:.3f} · "
          f"幅度比 {mag:.2f} · 超 {max_resid} 的层 {sum(1 for r, _, _ in per_layer if r > max_resid)} 个")
    print("       最差 5 层: " + " · ".join(f"{n.split('.weight')[0].split('model.')[-1]} {r:.2f}"
                                            for r, _, n in worst))
    if global_resid > max_resid:
        raise SystemExit(
            f"🔴 全局加权保真残差 {global_resid:.3f} > {max_resid} ⇒ 增量太小，"
            "被 bf16 存储的舍入噪声淹没了。\n"
            "   **不要合并这一级的增量**，保持 adapter 形态；\n"
            "   （RL 一轮的增量典型是 0.05% 量级，残差 ~0.87 —— 见 docs/syncopate/18 §3.3）"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把 LoRA 合并进基座")
    parser.add_argument("--base", default="models/Qwen3-4B")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_path = str((ROOT / args.base).resolve())
    adapter_path = str((ROOT / args.adapter).resolve())
    out_dir = ROOT / args.out

    print(f"[合并] 基座 {base_path}")
    print(f"       adapter {adapter_path}")
    # 用 bf16 加载并合并：RL 侧 FSDP 也是 bf16 持参，保持一致，
    # 免得合并时用 fp32、训练时降到 bf16 又引入一次舍入差异
    model = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    # ★★ 2026-08-18：合并之后必须验一遍「增量真的落进权重了」。
    # 实测过的坑：**损失来自存储精度，不是累加精度** —— 在 fp32 里相加再存 bf16 一样丢。
    #   Δ 占基座 0.42%（SFT 一遍）⇒ 保真残差 0.36，可用
    #   Δ 占基座 0.056%（RL 一轮）⇒ 保真残差 0.87，**合并等于毁掉它**
    # ⇒ 这个断言不是防打字错，是防「小增量被 bf16 静默吃掉」。
    _assert_merge_landed(base_path, adapter_path, model)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_path).save_pretrained(out_dir)
    # 留个来源记录：合并出来的模型看起来和基座一模一样，不写清楚来源，
    # 下一个窗口没法分辨 models/ 下哪个是哪个
    (out_dir / "syncopate_provenance.json").write_text(json.dumps({
        "base": args.base, "adapter": args.adapter,
        "note": "SFT LoRA 已合并进权重。RL 在此之上挂新 LoRA，"
                "关掉 adapter 得到的 reference 就是 SFT 本身（手册 §24③）。",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
