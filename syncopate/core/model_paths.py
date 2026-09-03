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
