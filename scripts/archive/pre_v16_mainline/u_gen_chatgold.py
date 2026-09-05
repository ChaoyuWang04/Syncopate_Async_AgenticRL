#!/usr/bin/env python
"""U 路 P2 · 闲聊 gold 生成（离线蒸馏第一步：教师自由文本 → 契约模板装壳）。

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/u_gen_chatgold.py

150 条新闲聊 prompt（与 P1 训练集/两考场逐字去重），底座 Qwen3-4B 以"简短友好的
投放助手"身份生成自由文本回复，程序装进契约 JSON（summary 取首句截断，reply=全文）。
质量闸：20≤len≤400 字符 · 无 role 标记泄漏 · 无 JSON 嵌套；不合格丢弃并计数。
产物 data/u_route/chat_gold.jsonl：{prompt, reply, gold_json}
"""

from __future__ import annotations

import json
import random
import sys

import torch
from syncopate.core.model_paths import TEST_TOKENIZER, STUDENT_MODEL, TEACHER_MODEL

sys.path.insert(0, ".")

rng = random.Random(140828)

TOPICS = ["投放圈的黑话", "第一次独立操盘的感觉", "怎么向非行业朋友解释你的工作",
          "数据看板一片红的早晨", "素材审核被拒的心情", "和设计同学沟通素材",
          "凌晨盯竞价的经历", "行业前辈给过你什么建议", "怎么快速进入工作状态",
          "投放和运营谁更累", "预算被砍时怎么调整心态", "最近学到的新东西",
          "远程办公的体验", "开会太多怎么办", "怎么记住那么多指标",
          "给新手推荐入门资料", "职业倦怠怎么破", "跨时区投放的作息"]
OPENERS = ["聊聊", "说说", "你怎么看", "有什么想法", "分享一下"]
SMALLTALK = ["今天天气不错", "刚喝了杯咖啡精神多了", "周五啦", "下雨天没干劲",
             "刚跑完步回来", "午休回来了", "今天地铁好挤", "终于把周报写完了",
             "刚开完季度复盘", "新同事今天入职"]
QUESTIONS = ["你叫什么名字来着", "你是人还是程序", "你几岁了", "你会做梦吗",
             "你喜欢什么游戏", "你觉得自己聪明吗", "你有性格吗", "你怎么学习新知识",
             "你最自豪的能力是什么", "你害怕被关掉吗"]


def build_prompts() -> list[str]:
    out = []
    for t in TOPICS:
        for o in rng.sample(OPENERS, 3):
            out.append(f"{o}{t}吧")
    out += SMALLTALK * 2
    out += QUESTIONS * 3
    rng.shuffle(out)
    # 与既有集合去重
    seen = set()
    for f in ("talk_exam.jsonl", "context_exam.jsonl", "p1_prompts.jsonl"):
        for x in open(f"data/u_route/{f}"):
            for t in json.loads(x)["turns"]:
                seen.add(t)
    out = [p for p in dict.fromkeys(out) if p not in seen]
    return out[:150]


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    prompts = build_prompts()
    print(f"闲聊 prompt {len(prompts)} 条")
    tok = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()

    SYS = ("你是一个手游买量投放团队的 AI 助手，正在和运营同事闲聊。"
           "用简短、自然、友好的中文口吻回答，不超过三句话，不要列条目，不要提工具或系统。")
    out, dropped = [], 0
    for i in range(0, len(prompts), 8):
        batch = prompts[i:i + 8]
        msgs = [tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": p}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False)
            for p in batch]
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        enc = tok(msgs, return_tensors="pt", padding=True).to(0)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=180, do_sample=True,
                               temperature=0.8, top_p=0.95,
                               pad_token_id=tok.pad_token_id)
        for j, p in enumerate(batch):
            rep = tok.decode(o[j][enc.input_ids.shape[1]:],
                             skip_special_tokens=True).strip()
            bad = (not (20 <= len(rep) <= 400) or "<|" in rep or "{" in rep
                   or "assistant" in rep.lower())
            if bad:
                dropped += 1
                continue
            summary = rep.split("。")[0][:40]
            gold = json.dumps({"behavior": "answer",
                               "answer": {"summary": summary, "reply": rep}},
                              ensure_ascii=False)
            out.append({"prompt": p, "reply": rep, "gold_json": gold})
        print(f"  {len(out)} 条已收（丢 {dropped}）", flush=True)
    with open("data/u_route/chat_gold.jsonl", "w") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"✅ chat_gold {len(out)} 条（质量闸丢弃 {dropped}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
