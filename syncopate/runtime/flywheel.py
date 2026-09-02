"""K10 · 回流飞轮（课件 CH10）：**执行完成 ≠ 问题解决**——两把尺子有时间差，信号要结构化落库、归因、成题、导出。

三句限制（课件 §9）：runtime 数据不直接训练 · 手动导出 · 不过门禁不上线。
四步顺序：先有信号 → 再有归因 → 再有题目 → 最后才有优化（先建平台 = 工具空转）。
`/auto-apply` 这类接口**永远不实现**（最强的权限控制是能力根本不存在）。

⚠️ 与 26 号管线的交界：本模块只把产物落到 `data/feedback_exports/<batch>/`（K 线目录），
   **不写**训练线的数据/考卷文件，不改任何生成器；吸入由 26 线的脚本决定（verl-22 09-02 已问）。
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

# ---- 归因词表（label=症状 / reason_code=病因，必须分列：同一症状不同病因走向完全不同的修法）----
# label **不另造**（verl-22 09-02）：= verifier 的 cap 名（CAPS.names()，12 个）∪ 行为（contract 的
#   TERMINAL_SIGNALS + tool_call/answer）∪ 多轮六族前缀（26 §1.2-S2 ①–⑥）∪ good。
FAMILY_PREFIXES = ("F1_object", "F2_progress", "F3_revision", "F4_commitment", "F5_time", "F6_meta")
_LABEL_VOCAB: frozenset[str] | None = None


def label_vocab() -> frozenset[str]:
    global _LABEL_VOCAB
    if _LABEL_VOCAB is None:
        from syncopate.domains.adcampaign import build_domain
        build_domain()                                   # 注册 caps
        from syncopate.core.verifier_engine import CAPS
        from syncopate.core.contract import TERMINAL_SIGNALS
        caps = set(CAPS.names())
        behaviors = {f"behavior:{str(b).split('.')[-1]}" for b in (*TERMINAL_SIGNALS, "tool_call", "answer")}
        _LABEL_VOCAB = frozenset({"good"} | caps | behaviors | set(FAMILY_PREFIXES))
    return _LABEL_VOCAB


def label_ok(label: str | None) -> bool:
    if label is None:
        return True
    if label in label_vocab():
        return True
    return any(label.startswith(p + ":") for p in FAMILY_PREFIXES)      # 六族细分：F4_commitment:defer_then_enough
REASON_CODES = frozenset({
    "model_reasoning",            # 模型判断错（真负样本）
    "tool_output_wrong",          # 工具返回本身错/脏
    "external_system_error",      # 环境/外部系统错：模型没错（08-20 四课 · 行为异常先查输入）
    "environment_data_missing",   # 租户数据为空/缺（seed 那课）
    "prompt_contract_gap",        # 契约/prompt 没给它该有的位置（answer_fields 那课）
    "policy_gap",                 # 政策/规则本身缺
    "user_input_ambiguous",       # 用户问法歧义（该 clarify 的场景）
    "infra_timeout",              # 超时/限流/断连
    "unknown",
})
# 这些病因下的负样本**不许进负样本池**：模型没错，喂进 RL 是在惩罚做对了的模型（loss 曲线看不出来）
EXTERNAL_REASONS = frozenset({"external_system_error", "environment_data_missing", "infra_timeout", "tool_output_wrong"})
SECRET_KEY_RE = re.compile(r"(token|password|secret|authorization|api_key|apikey|idempotency_key|resume_token)", re.I)


def registry_version() -> str:
    """工具注册表版本 = 全部 spec（名/参数/kind）的稳定哈希；改任何工具描述都会变。"""
    from syncopate.domains.adcampaign import build_domain
    reg = build_domain().registry
    payload = json.dumps([(n, reg.get(n).kind, reg.get(n).parameters, reg.get(n).description)
                          for n in reg.names()], sort_keys=True, ensure_ascii=False, default=str)
    return "reg_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# ---- 抽候选：四路 SQL（"你抽不到的 case 永远不会被优化"）----
CANDIDATE_ROUTES = ("failed", "negative_feedback", "business_outcome", "high_value_success")


async def extract_candidates(db: Any, *, org_id: str | None = None, window_hours: int = 24,
                             limit: int = 50) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    async with db.tx() as conn:
        out["failed"] = [dict(r) for r in await conn.fetch("""
            SELECT org_id, run_id, 'failed' AS route FROM agent_runs
             WHERE status='failed' AND created_at > now() - make_interval(hours => $1)
               AND ($2::text IS NULL OR org_id=$2) ORDER BY created_at DESC LIMIT $3""", window_hours, org_id, limit)]
        out["negative_feedback"] = [dict(r) for r in await conn.fetch("""
            SELECT DISTINCT ON (f.org_id, f.run_id) f.org_id, f.run_id, 'negative_feedback' AS route
              FROM feedback_items f
             WHERE f.created_at > now() - make_interval(hours => $1) AND ($2::text IS NULL OR f.org_id=$2)
               AND f.rating = -1
               AND NOT EXISTS (SELECT 1 FROM feedback_items g WHERE g.org_id=f.org_id AND g.run_id=f.run_id
                                  AND g.created_at > f.created_at AND g.rating <> -1)   -- 后条推翻前条
             ORDER BY f.org_id, f.run_id, f.created_at DESC LIMIT $3""", window_hours, org_id, limit)]
        # 业务结果通道：**跑对了但业务结果坏** —— run succeeded 而 D7 回收结果为负 / 或写调用最终 failed
        out["business_outcome"] = [dict(r) for r in await conn.fetch("""
            SELECT DISTINCT r.org_id, r.run_id, 'business_outcome' AS route
              FROM agent_runs r
              LEFT JOIN approval_cases a ON a.org_id=r.org_id AND a.run_id=r.run_id
              LEFT JOIN tool_calls t ON t.org_id=r.org_id AND t.run_id=r.run_id AND t.side_effect
             WHERE r.status='succeeded' AND r.created_at > now() - make_interval(hours => $1)
               AND ($2::text IS NULL OR r.org_id=$2)
               AND ((a.outcome_result IS NOT NULL AND (a.outcome_result->>'ok') = 'false')
                    OR t.status IN ('failed','response_lost'))
             LIMIT $3""", window_hours, org_id, limit)]
        out["high_value_success"] = [dict(r) for r in await conn.fetch("""
            SELECT DISTINCT r.org_id, r.run_id, 'high_value_success' AS route
              FROM agent_runs r JOIN tool_calls t ON t.org_id=r.org_id AND t.run_id=r.run_id
             WHERE r.status='succeeded' AND t.side_effect AND t.status='succeeded'
               AND r.created_at > now() - make_interval(hours => $1) AND ($2::text IS NULL OR r.org_id=$2)
             LIMIT $3""", window_hours, org_id, limit)]
    return out


# ---- trace → case 四步：拉齐八表 → 深拷贝 → 密钥类删除 → 人工定 expected ----


def _scrub(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: ("<removed>" if SECRET_KEY_RE.search(str(k)) else _scrub(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub(x) for x in obj]
    return obj


def trace_digest(trace: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(trace, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


async def build_case_from_run(db: Any, *, org_id: str, run_id: str) -> dict[str, Any]:
    """原始 trace 逐字节不被加工过程修改（前后哈希一致）；产物是**深拷贝**后脱敏的**考卷 v4 题形**
    （verl-22 09-02 定）：id/level=FB/turns/prior/judge/note/meta。`note`（expected）只能人签。
    prior 形状 = agent_runs 行（user/status/error/result），`u_exam_run.seed_prior` 直接吃。"""
    from syncopate.runtime.db import trace as _trace
    raw = await _trace(db, org_id=org_id, run_id=run_id)
    if raw is None:
        raise LookupError(run_id)
    before = trace_digest(raw)
    work = _scrub(copy.deepcopy(raw))
    run = work["run"]
    ann = None
    prior_rows: list[dict[str, Any]] = []
    async with db.tx() as conn:
        a = await conn.fetchrow("SELECT label, reason_code, expected_json, notes, annotator FROM run_annotations "
                                "WHERE org_id=$1 AND run_id=$2 ORDER BY created_at DESC LIMIT 1", org_id, run_id)
        ann = dict(a) if a else None
        if run.get("conversation_id"):
            prior_rows = [dict(r) for r in await conn.fetch(
                "SELECT user_message AS \"user\", status, error, result FROM agent_runs "
                "WHERE org_id=$1 AND conversation_id=$2 AND created_at < "
                "(SELECT created_at FROM agent_runs WHERE org_id=$1 AND run_id=$3) "
                "AND status IN ('succeeded','cancelled') ORDER BY created_at", org_id, run["conversation_id"], run_id)]
    case = {
        "id": None,                                       # 导出时编 FB_<batch>_<n>
        "level": "FB",
        "turns": [run.get("user_message") or ""],
        "prior": [{"user": p["user"], "status": p["status"], "error": p.get("error"),
                   "result": _scrub(copy.deepcopy(p.get("result") or {}))} for p in prior_rows],
        "judge": {"type": None},                          # 由人从 u_exam_judge_v4 现有判类里选；空 = 报告项
        "note": (ann or {}).get("expected_json") or (ann or {}).get("notes"),   # ⛔ 只能人签
        "meta": {
            "source_run_id": run_id, "org": org_id, "status": run.get("status"),
            "contract": run.get("contract_version"), "prompt_version": run.get("prompt_version"),
            "model": run.get("model_version"),
            "registry": next((t.get("registry_version") for t in work["tool_calls"] if t.get("registry_version")), None),
            "label": (ann or {}).get("label"), "reason_code": (ann or {}).get("reason_code"),
            "annotator": (ann or {}).get("annotator"),
            "trajectory": [{"tool": t["tool"], "status": t.get("status")} for t in work["tool_calls"]],
            "trace_digest": before,
        },
    }
    assert trace_digest(raw) == before, "加工过程改动了原始 trace"
    return case


# ---- 导出：出局（OR）先于准入（AND），默认拒绝 ----


def admission_verdict(case: dict[str, Any]) -> tuple[bool, str]:
    meta = case.get("meta", {})
    label, reason = meta.get("label"), meta.get("reason_code")
    negative = label not in (None, "good")
    # 出局清单（任一命中即出局）
    if case.get("note") in (None, "", {}):
        return False, "no_expected"                                     # expected 只能人签
    if negative and not reason:
        return False, "negative_without_reason_code"
    if negative and reason in EXTERNAL_REASONS:
        return False, f"external_reason:{reason}"                       # 模型没错的负样本不进负样本池
    if label and not label_ok(label):
        return False, "unknown_label"
    if reason is not None and reason not in REASON_CODES:
        return False, "unknown_reason_code"
    blob = json.dumps(case, ensure_ascii=False, default=str)
    if re.search(r"(dev-token-[a-z]+|Bearer [A-Za-z0-9._-]{8,}|syncopate-dev)", blob):
        return False, "secret_residue"
    # 准入清单（全部满足）
    if not case.get("turns") or not case["turns"][0] or not meta.get("contract"):
        return False, "missing_turns_or_contract_version"
    return True, "ok"


async def export_batch(db: Any, cases: list[dict[str, Any]], *, dataset_version: str, created_by: str,
                       root: str | os.PathLike[str] = "data/feedback_exports") -> dict[str, Any]:
    batch_id = f"fb_{time.strftime('%Y%m%d_%H%M%S')}_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:6]}"
    out_dir = Path(root) / batch_id
    out_dir.mkdir(parents=True, exist_ok=True)
    admitted, rejected = [], []
    for c in cases:
        ok, why = admission_verdict(c)
        (admitted if ok else rejected).append({**c, "_verdict": why})
    with open(out_dir / "cases.jsonl", "w", encoding="utf-8") as f:
        for n, c in enumerate(admitted, 1):
            c["id"] = f"FB_{batch_id}_{n}"
            f.write(json.dumps({k: v for k, v in c.items() if not k.startswith("_")}, ensure_ascii=False, default=str) + "\n")
    manifest = {"batch_id": batch_id, "dataset_version": dataset_version, "n_cases": len(admitted),
                "n_rejected": len(rejected),
                "rejected": [{"source_run_id": c["meta"]["source_run_id"], "why": c["_verdict"]} for c in rejected],
                "source_run_ids": [c["meta"]["source_run_id"] for c in admitted], "created_by": created_by,
                "destination": str(out_dir), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    async with db.tx() as conn:
        await conn.execute(
            "INSERT INTO training_exports (batch_id, dataset_version, n_cases, n_rejected, destination, manifest, created_by) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7)", batch_id, dataset_version, len(admitted), len(rejected), str(out_dir),
            manifest, created_by)
    print(f"[export] batch={batch_id} admitted={len(admitted)} rejected={len(rejected)} → {out_dir}", flush=True)
    return manifest
