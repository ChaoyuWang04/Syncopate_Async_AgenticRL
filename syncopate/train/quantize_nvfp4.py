#!/usr/bin/env python3
"""E19-c 第四臂 · 把 SFT 模型离线量化成 NVFP4（W4A4，16 元素块 E4M3 缩放 + 张量级 FP32）。

★ 为什么要离线：vLLM 的 FP8 可以在线动态量化，NVFP4 不行——W4A4 的激活缩放需要
  校准统计，产物是 compressed-tensors 格式的 checkpoint，vLLM 直接 serve 它。
★ 校准数据用**我们自己的负载**（data/rl/v13 的真实 prompt 过 chat 模板），不用开源
  聊天语料——激活分布要贴真实 serving 流量（4k 长题面），别校准到别人的分布上。
lm_head 不量化（惯例 + 它的误差直接进 logprob）。
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MP = "models/Qwen3-4B-sft-v13r2-e1"
OUT = "models/Qwen3-4B-sft-v13r2-e1-NVFP4"  # --scheme NVFP4A16 时自动改后缀


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--scheme", default="NVFP4", choices=["NVFP4", "NVFP4A16"])
    a = ap.parse_args()
    global OUT
    if a.scheme == "NVFP4A16":
        OUT = OUT + "A16"

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True)
    df = pd.read_parquet("data/rl/v13/train.parquet").head(a.samples)
    texts = [tok.apply_chat_template(list(p), tokenize=False, add_generation_prompt=True)
             for p in df["prompt"]]
    ds = Dataset.from_dict({"text": texts})
    print(f"[calib] {len(texts)} 条真实负载 prompt（首条 {len(texts[0])} 字符）")

    model = AutoModelForCausalLM.from_pretrained(
        MP, dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda:0")

    oneshot(
        model=model,
        dataset=ds,
        recipe=QuantizationModifier(targets="Linear", scheme=a.scheme, ignore=["lm_head"]),
        max_seq_length=a.max_len,
        num_calibration_samples=len(texts),
        output_dir=OUT,
    )
    tok.save_pretrained(OUT)
    print(f"[done] NVFP4 checkpoint -> {OUT}")
    # 判据行：产物必须真的是压缩格式（config 里带 quantization_config）
    import json
    cfg = json.loads((Path(OUT) / "config.json").read_text())
    qc = cfg.get("quantization_config", {})
    print(f"[verify] quantization_config.format = {qc.get('format', '缺失!')} · "
          f"scheme keys = {list(qc.get('config_groups', {}).keys()) or '缺失!'}")


if __name__ == "__main__":
    main()
