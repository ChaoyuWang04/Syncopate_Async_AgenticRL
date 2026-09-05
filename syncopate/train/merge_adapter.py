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
from syncopate.core.model_paths import TEST_TOKENIZER, STUDENT_MODEL, TEACHER_MODEL

ROOT = Path(__file__).resolve().parents[2]


def _resolve_weight_name(adapter_stem: str, available_names: set[str]) -> str:
    """把 PEFT 的 adapter 名字唯一映射到一套真实权重名。

    PEFT 会隐藏某些模型包装层。例如 Qwen3.5 的 adapter 使用
    ``model.layers.*``，而下载到磁盘的基座 shard 使用
    ``model.language_model.layers.*``。内存模型和磁盘基座要分别调用本函数；
    这里只接受唯一的精确后缀匹配；
    匹配不到或命中多个都硬报错，不能猜。
    """
    name = adapter_stem
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    expected = name + ".weight"
    if expected in available_names:
        return expected

    parts = expected.split(".")
    # 至少保留「层号/模块/weight」三个片段，避免用过短后缀误配。
    for drop in range(1, max(1, len(parts) - 2)):
        suffix = ".".join(parts[drop:])
        matches = sorted(
            candidate for candidate in available_names
            if candidate == suffix or candidate.endswith("." + suffix)
        )
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise SystemExit(
                f"🔴 adapter 权重 {adapter_stem} 的后缀 {suffix!r} "
                f"匹配到 {len(matches)} 个基座权重，拒绝猜测: {matches[:5]}"
            )
    raise SystemExit(
        f"🔴 adapter 权重 {adapter_stem} 在合并模型与基座中找不到唯一对应项"
    )


def _converted_base_key_map(
    base_names: set[str], model_names: set[str], conversions: list[object]
) -> dict[str, list[str]]:
    """按 Transformers 加载器同一条转换链建立「内存 key → 磁盘 key」表。"""
    result: dict[str, list[str]] = {}
    for source_name in sorted(base_names):
        target_name = source_name
        for conversion in conversions:
            target_name, _ = conversion.rename_source_key(target_name)
        if target_name in model_names:
            result.setdefault(target_name, []).append(source_name)
    return result


def _official_base_key_map(merged_model, base_names: set[str]) -> dict[str, list[str]] | None:
    """Transformers 5 有官方 checkpoint 转换链；旧本地栈没有时返回 None。"""
    try:
        from transformers.conversion_mapping import get_model_conversion_mapping
    except ImportError:
        return None
    conversions = get_model_conversion_mapping(merged_model)
    return _converted_base_key_map(base_names, set(merged_model.state_dict()), conversions)



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

    merged_names = set(sd)
    base_names = set(base_idx)
    official_base_map = _official_base_key_map(merged_model, base_names)
    per_layer: list[tuple[float, float, str]] = []   # (resid, ‖Δ‖, name)
    mapped_names: set[tuple[str, str]] = set()
    used_merged: set[str] = set()
    used_base: set[str] = set()
    zero_delta = 0
    err2 = d2 = kept2 = 0.0
    for stem in stems:
        a_key = stem + ".lora_A.weight"
        b_key = stem + ".lora_B.weight"
        if a_key not in ab or b_key not in ab:
            raise SystemExit(f"🔴 adapter 的 A/B 权重不成对: {stem}")
        merged_name = _resolve_weight_name(stem, merged_names)
        if official_base_map is None:
            # 本机旧 Transformers 只用于便宜单测；生产新栈必须走上面的官方转换链。
            base_name = _resolve_weight_name(stem, base_names)
        else:
            base_candidates = official_base_map.get(merged_name, [])
            if len(base_candidates) != 1:
                raise SystemExit(
                    f"🔴 Transformers 官方映射对 {merged_name} 得到 "
                    f"{len(base_candidates)} 个磁盘权重，拒绝猜测: {base_candidates[:5]}"
                )
            base_name = base_candidates[0]
        if merged_name in used_merged or base_name in used_base:
            raise SystemExit(
                f"🔴 多个 adapter 层映射到同一权重: merged={merged_name}, base={base_name}"
            )
        used_merged.add(merged_name)
        used_base.add(base_name)
        mapped_names.add((merged_name, base_name))
        A = ab[a_key].float()
        B = ab[b_key].float()
        delta = scale * (B @ A)
        dn = delta.norm().item()
        if dn == 0.0:
            zero_delta += 1
            continue
        with safe_open(base_idx[base_name], framework="pt") as fh:
            base_w = fh.get_tensor(base_name).float()
        merged_w = sd[merged_name].float().cpu()
        if merged_w.shape != base_w.shape or merged_w.shape != delta.shape:
            raise SystemExit(
                f"🔴 合并校验形状不一致: adapter={tuple(delta.shape)}, "
                f"merged[{merged_name}]={tuple(merged_w.shape)}, "
                f"base[{base_name}]={tuple(base_w.shape)}"
            )
        kept = merged_w - base_w
        resid = ((kept - delta).norm() / dn).item()
        per_layer.append((resid, dn, merged_name))
        err2 += (resid ** 2) * (dn ** 2)
        d2 += dn ** 2
        kept2 += kept.norm().item() ** 2

    if not per_layer or d2 == 0.0:
        raise SystemExit(
            "🔴 adapter 没有任何非零的有效 ΔW，无法证明合并落地"
        )
    global_resid = (err2 / d2) ** 0.5
    mag = (kept2 / d2) ** 0.5
    worst = sorted(per_layer, reverse=True)[:5]
    mapping_source = "Transformers 官方 checkpoint 映射" if official_base_map is not None else "唯一后缀回退"
    print(f"[合并] {mapping_source}：唯一映射 {len(mapped_names)}/{len(stems)} 层；"
          f"校验 {len(per_layer)} 个非零 ΔW（零 ΔW {zero_delta}）: "
          f"全局加权残差 {global_resid:.3f} · "
          f"幅度比 {mag:.2f} · 超 {max_resid} 的层 {sum(1 for r, _, _ in per_layer if r > max_resid)} 个")
    print("       最差 5 层: " + " · ".join(f"{n.split('.weight')[0].split('model.')[-1]} {r:.2f}"
                                            for r, _, n in worst))
    if global_resid > max_resid:
        raise SystemExit(
            f"🔴 全局加权保真残差 {global_resid:.3f} > {max_resid} ⇒ 增量太小，"
            "被 bf16 存储的舍入噪声淹没了。\n"
            "   **不要合并这一级的增量**，保持 adapter 形态；\n"
            "   （这一历史量级只用于解释保护逻辑，见 "
            "docs/archive/syncopate/pre-consolidation-v16/18-pipeline-assumption-probes.md §3.3）"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把 LoRA 合并进基座")
    parser.add_argument("--base", default=STUDENT_MODEL)
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
