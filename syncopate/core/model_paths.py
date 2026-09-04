"""模型/分词器路径的**唯一来源**（2026-09-03，裁定⑫⑬）。

以前 `models/Qwen3-0.6B` / `models/Qwen3-4B` 散在 30 个文件里当默认值；换代 = 30 处改。现在换代只改这里或设环境变量。
Modal 上 `/vol/repo/models -> /vol/models`（stack_probe._sync_repo 建软链），本机 `models/` 里放分词器即可。
"""
from __future__ import annotations

import os

# 测试与建库分词用（小、快、与学生同一词表/同一模板族）
TEST_TOKENIZER = os.environ.get("SYNCOPATE_TEST_TOKENIZER", "models/Qwen3.5-0.8B")
# 学生：最新小 MoE（EP/GDN/MTP 全在场）
STUDENT_MODEL = os.environ.get("SYNCOPATE_STUDENT_MODEL", "models/Qwen3.6-35B-A3B")
# 教师：人话 + 思考同一个（裁定⑬：只要装得下就用大的）
TEACHER_MODEL = os.environ.get("SYNCOPATE_TEACHER_MODEL", "models/Qwen3.8-27B")

# 已退役（只作历史读数对照，不许再当默认）：Qwen3-0.6B · Qwen3-4B · Qwen3.5-4B · Qwen3.5-9B · Qwen3.5-27B


def build_tokenizer_path() -> str:
    """建库 / 出厂体检 / 画廊 / 预算表用的分词器（唯一定义）：学生权重在（Modal）用学生自己的；本机没权重 ⇒ 同词表的 TEST_TOKENIZER。
    09-05：此前三个脚本各写一份，其中画廊/预算表拿**退役的 Qwen3-4B** 存不存在当判据（本机存在 ⇒ 选到不存在的 35B ⇒ 崩）。"""
    return STUDENT_MODEL if os.path.exists(os.path.join(STUDENT_MODEL, "tokenizer.json")) else TEST_TOKENIZER
