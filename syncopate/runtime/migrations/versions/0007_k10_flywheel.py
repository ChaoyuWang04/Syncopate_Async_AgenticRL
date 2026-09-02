"""0007 · K10 回流飞轮（课件 CH10）：两张回流表 + 导出留痕 + 版本归因列。

  feedback_items     绑 run_id 的结构化反馈（同 run 多行、后条可推翻前条）；label=症状 / reason_code=病因分列
  run_annotations    人工标注（reason_code 必填；expected 只能人签）
  training_exports   导出留痕（dataset_version + 条数 + 去向 + manifest）
  agent_runs         contract_version / prompt_version / model_version（run 级）
  tool_calls         registry_version（tool call 级）——没有版本切片，"错误率涨了"永远拆不成可排除的假设
"""
from __future__ import annotations

from alembic import op

revision = "0007_k10_flywheel"
down_revision = "0006_k9_budgets"
branch_labels = None
depends_on = None

UP = r"""
CREATE TABLE IF NOT EXISTS feedback_items (
    id          BIGSERIAL   PRIMARY KEY,
    org_id      TEXT        NOT NULL,
    run_id      TEXT        NOT NULL,
    rating      SMALLINT    NOT NULL CHECK (rating IN (-1, 0, 1)),
    label       TEXT,
    reason_code TEXT,
    comment     TEXT,
    created_by  TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS feedback_by_run ON feedback_items (org_id, run_id, created_at);

CREATE TABLE IF NOT EXISTS run_annotations (
    id            BIGSERIAL   PRIMARY KEY,
    org_id        TEXT        NOT NULL,
    run_id        TEXT        NOT NULL,
    label         TEXT        NOT NULL,
    reason_code   TEXT        NOT NULL,
    expected_json JSONB,
    notes         TEXT,
    annotator     TEXT        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS annotations_by_run ON run_annotations (org_id, run_id, created_at);

CREATE TABLE IF NOT EXISTS training_exports (
    id              BIGSERIAL   PRIMARY KEY,
    batch_id        TEXT        NOT NULL UNIQUE,
    dataset_version TEXT        NOT NULL,
    n_cases         INTEGER     NOT NULL,
    n_rejected      INTEGER     NOT NULL DEFAULT 0,
    destination     TEXT        NOT NULL,
    manifest        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_by      TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE agent_runs
    ADD COLUMN IF NOT EXISTS contract_version TEXT,
    ADD COLUMN IF NOT EXISTS prompt_version   TEXT,
    ADD COLUMN IF NOT EXISTS model_version    TEXT;
ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS registry_version TEXT;
"""

DOWN = r"""
ALTER TABLE tool_calls DROP COLUMN IF EXISTS registry_version;
ALTER TABLE agent_runs DROP COLUMN IF EXISTS model_version, DROP COLUMN IF EXISTS prompt_version,
                       DROP COLUMN IF EXISTS contract_version;
DROP TABLE IF EXISTS training_exports;
DROP TABLE IF EXISTS run_annotations;
DROP TABLE IF EXISTS feedback_items;
"""


def upgrade() -> None:
    op.get_bind().exec_driver_sql(UP)


def downgrade() -> None:
    op.get_bind().exec_driver_sql(DOWN)
