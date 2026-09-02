"""0006 · K9 生产硬化（课件 CH9 §6/§4）：run 级预算四字段 + org 日预算表 + usage 记账粒度改"每次模型调用一行"。

  agent_runs.max_model_calls / max_tokens / max_duration_s   run 级预算（步数上限已在 ActionGate）
  agent_runs.budget_exceeded_at                              超限转 waiting_for_user 的留痕（"还有救就转 waiting 不判死"）
  org_budgets                                                org 日 token 预算两档（近限告警 / 超限拒新建）
  usage_records.call_index                                   语义改为 (attempts-1)*1000 + 第几次模型调用（记录粒度决定控制粒度）
"""
from __future__ import annotations

from alembic import op

revision = "0006_k9_budgets"
down_revision = "0005_k8_platform_ledger"
branch_labels = None
depends_on = None

UP = r"""
ALTER TABLE agent_runs
    ADD COLUMN IF NOT EXISTS max_model_calls    INTEGER NOT NULL DEFAULT 40,
    ADD COLUMN IF NOT EXISTS max_tokens         BIGINT  NOT NULL DEFAULT 400000,
    ADD COLUMN IF NOT EXISTS max_duration_s     INTEGER NOT NULL DEFAULT 900,
    ADD COLUMN IF NOT EXISTS budget_exceeded_at TIMESTAMPTZ;
CREATE TABLE IF NOT EXISTS org_budgets (
    org_id             TEXT   PRIMARY KEY,
    daily_tokens       BIGINT NOT NULL DEFAULT 2000000,
    daily_cost_micros  BIGINT NOT NULL DEFAULT 10000000,
    warn_ratio         REAL   NOT NULL DEFAULT 0.8 CHECK (warn_ratio > 0 AND warn_ratio < 1),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE usage_records ADD COLUMN IF NOT EXISTS model TEXT;
ALTER TABLE usage_records ADD COLUMN IF NOT EXISTS usage_source TEXT NOT NULL DEFAULT 'measured'
    CHECK (usage_source IN ('measured','estimated'));
"""

DOWN = r"""
ALTER TABLE usage_records DROP COLUMN IF EXISTS usage_source, DROP COLUMN IF EXISTS model;
DROP TABLE IF EXISTS org_budgets;
ALTER TABLE agent_runs DROP COLUMN IF EXISTS budget_exceeded_at, DROP COLUMN IF EXISTS max_duration_s,
                       DROP COLUMN IF EXISTS max_tokens, DROP COLUMN IF EXISTS max_model_calls;
"""


def upgrade() -> None:
    op.get_bind().exec_driver_sql(UP)


def downgrade() -> None:
    op.get_bind().exec_driver_sql(DOWN)
