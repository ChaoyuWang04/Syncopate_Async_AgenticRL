"""K9-4 · 结构化日志：一行一个 JSON（课件 CH9 §9 十一字段的裁剪版）。

判据行（`[worker] …` 这类人读的行）保留——它们是机制生效的证据（infra 守则①）；
**错误路径**一律走这里：run_id / org_id / step / tool / error_code / latency_ms / request_id 可检索。
⛔ 密钥 / token / 完整 prompt 不进日志：字段名命中 SECRET_KEYS 一律打码（CI 正则守着）。
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any

SECRET_KEYS = ("token", "password", "secret", "authorization", "api_key", "apikey", "requirepass")
MAX_FIELD_CHARS = 500


def _redact(key: str, value: Any) -> Any:
    k = key.lower()
    if any(s in k for s in SECRET_KEYS):
        return "***"
    if key in ("prompt", "messages", "request_json"):
        return f"<{key} omitted>"
    if isinstance(value, str) and len(value) > MAX_FIELD_CHARS:
        return value[:MAX_FIELD_CHARS] + "…"
    return value


def log_event(component: str, event: str, *, level: str = "info", **fields: Any) -> dict[str, Any]:
    rec = {"ts": time.time(), "level": level, "component": component, "event": event}
    for k, v in fields.items():
        rec[k] = _redact(k, v)
    sys.stderr.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    sys.stderr.flush()
    return rec
