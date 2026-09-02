"""0004 · K6 Tool Runtime（课件 CH6，2026-09-02）：

  tool_calls.error_json   失败分诊三字段 {code, message, retryable}（+ alert）：结构化，不再只有一段文本
  tool_calls.blocked_by   被哪道闸拦下（unknown_tool / validation / permission / tier_d / release_gate /
                          daily_cost_cap / cancel_requested / max_steps）——"拦下也落库"
  幂等唯一索引            改按五态：只覆盖 status <> 'skipped_duplicate' 的行（课件待查#4 定死；
                          重复行用 duplicate_of 引用首次调用）
"""
from __future__ import annotations

from alembic import op

revision = "0004_k6_tool_calls"
down_revision = "0003_k3_outbox"
branch_labels = None
depends_on = None

UP = r"""
ALTER TABLE tool_calls
    ADD COLUMN IF NOT EXISTS error_json JSONB,
    ADD COLUMN IF NOT EXISTS blocked_by TEXT;
DROP INDEX IF EXISTS tool_calls_external_idem_uniq;
CREATE UNIQUE INDEX tool_calls_external_idem_uniq
    ON tool_calls (org_id, external_idempotency_key)
    WHERE external_idempotency_key IS NOT NULL AND status <> 'skipped_duplicate';
CREATE INDEX IF NOT EXISTS tool_calls_open_side_effects
    ON tool_calls (created_at) WHERE side_effect AND status IN ('running','response_lost');
"""

DOWN = r"""
DROP INDEX IF EXISTS tool_calls_open_side_effects;
DROP INDEX IF EXISTS tool_calls_external_idem_uniq;
CREATE UNIQUE INDEX tool_calls_external_idem_uniq
    ON tool_calls (org_id, external_idempotency_key)
    WHERE external_idempotency_key IS NOT NULL AND replayed_from IS NULL;
ALTER TABLE tool_calls DROP COLUMN IF EXISTS blocked_by, DROP COLUMN IF EXISTS error_json;
"""


def upgrade() -> None:
    op.get_bind().exec_driver_sql(UP)


def downgrade() -> None:
    op.get_bind().exec_driver_sql(DOWN)
