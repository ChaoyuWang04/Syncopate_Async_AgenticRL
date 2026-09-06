"""CPU 上量生成形状；不改变采样、预算或评分，也不输出业务原文。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Sequence


def generation_is_incomplete(*, finish_reason=None, stop_reason=None, discarded_tokens=0) -> bool:
    """训练、评测和 Runtime 共用；completed 不是“答案完整”的证明。"""
    return discarded_tokens > 0 or finish_reason == "length" or stop_reason == "length"


def repeated_span(tokens: Sequence, max_period: int = 16) -> dict[str, int]:
    """最长连续周期段。O(序列长度 × 周期数)，避免坏长串触发平方耗时。"""
    best = {"span_tokens": 0, "period_tokens": 0, "copies": 0}
    for period in range(1, min(max_period, len(tokens) // 2) + 1):
        matched = 0
        for i in range(period, len(tokens)):
            matched = matched + 1 if tokens[i] == tokens[i - period] else 0
            copies = (matched + period) // period
            span = copies * period
            if copies >= 2 and span > best["span_tokens"]:
                best = {"span_tokens": span, "period_tokens": period, "copies": copies}
    return best


def quantiles(values: Sequence[int]) -> dict:
    """分位数使用 nearest-rank；空集合显式返回 null。"""
    ordered = sorted(values)
    result = {"count": len(ordered), "min": min(ordered) if ordered else None}
    for label, q in (("p50", .5), ("p90", .9), ("p99", .99), ("max", 1)):
        result[label] = ordered[max(0, math.ceil(q * len(ordered)) - 1)] if ordered else None
    return result


def generation_shape(text: str, token_ids: Sequence[int], *, implicit_think_open: bool) -> dict:
    from syncopate.core.parsing_v15 import strip_thinking

    _, had_thinking, thinking = strip_thinking(text, implicit_open=implicit_think_open)
    opened = text.count("<think>") + int(implicit_think_open and not text.lstrip().startswith("<think>"))
    closed = text.count("</think>")
    lexical = re.findall(r"[A-Za-z_]+|\d+(?:\.\d+)?|[\u4e00-\u9fff]", text.lower())
    return {
        "generated_tokens": len(token_ids),
        "think_unclosed": opened > closed,
        "think_open_count": opened,
        "think_close_count": closed,
        "thinking_chars": len(thinking),
        "had_thinking": had_thinking,
        "repeat": repeated_span(token_ids),
        "lexical_repeat": repeated_span(lexical),
    }


def assistant_spans(ids: list[int], prompt_length: int, *, header: list[int], end_id: int):
    """首轮从已存的 prompt_length 开始；后续轮从真实 ChatML assistant 头之后开始。"""
    start = prompt_length
    while start < len(ids):
        try:
            end = ids.index(end_id, start)
        except ValueError:
            yield start, len(ids), False
            return
        yield start, end, True
        start = next((i + len(header) for i in range(end + 1, len(ids) - len(header) + 1)
                      if ids[i:i + len(header)] == header), len(ids))


def summarize_turns(turns: list[dict]) -> dict:
    return {
        "turn_tokens": quantiles([x["generated_tokens"] for x in turns]),
        "thinking_tokens": quantiles([x["thinking_tokens"] for x in turns]),
        "unclosed_turns": sum(x["think_unclosed"] for x in turns),
        "missing_turn_end": sum(not x["has_turn_end"] for x in turns),
        "longest_repeat": max((x["repeat"]["span_tokens"] for x in turns), default=0),
        "longest_lexical_repeat": max((x["lexical_repeat"]["span_tokens"] for x in turns), default=0),
        "longest_turns": sorted(turns, key=lambda x: x["generated_tokens"], reverse=True)[:8],
    }


def audit_sft(path: Path, tokenizer) -> dict:
    import pyarrow.parquet as pq
    from syncopate.core.parsing_v15 import strip_thinking

    rows = pq.read_table(path).to_pylist()
    header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    assert len(end_ids) == 1, "ChatML 结束标记必须是一个 token"
    turns, prompt_lengths, mask_segments = [], [], []
    buckets = Counter()
    for row_index, row in enumerate(rows):
        ids, mask = list(row["input_ids"]), list(row["loss_mask"])
        assert len(ids) == len(mask), f"row {row_index}: mask 长度错误"
        prompt_length = int(row["prompt_length"])
        assert 0 < prompt_length < len(ids), f"row {row_index}: prompt_length 错误"
        assert sum(mask) > 0, f"row {row_index}: 零监督"
        prompt_lengths.append(prompt_length)
        bucket = row.get("bucket", "unknown")
        buckets[bucket] += 1
        cid = hashlib.sha256(str(row["case_id"]).encode()).hexdigest()[:12]
        for turn, (start, end, has_end) in enumerate(assistant_spans(
                ids, prompt_length, header=header, end_id=end_ids[0])):
            text = tokenizer.decode(ids[start:end], skip_special_tokens=False)
            implicit = turn == 0 and tokenizer.decode(ids[:start]).endswith("<think>\n")
            shape = generation_shape(text, ids[start:end], implicit_think_open=implicit)
            _, _, thinking = strip_thinking(text, implicit_open=implicit)
            turns.append({"case_hash": cid, "row_index": row_index, "turn": turn + 1,
                          "bucket": bucket, "has_turn_end": has_end,
                          "thinking_tokens": len(tokenizer.encode(thinking, add_special_tokens=False)),
                          "supervised_tokens": sum(mask[start:end]), **shape})
        length = 0
        for value in mask + [0]:
            if value:
                length += 1
            elif length:
                mask_segments.append(length)
                length = 0
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(rows), "buckets": dict(buckets), "prompt_tokens": quantiles(prompt_lengths),
            "supervised_spans": quantiles(mask_segments), **summarize_turns(turns)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"不覆盖已有证据：{args.out}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    result = {"tokenizer": args.tokenizer,
              "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
              "sft": audit_sft(args.sft, tokenizer)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
