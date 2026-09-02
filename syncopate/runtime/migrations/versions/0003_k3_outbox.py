"""0003 · K3 队列地基（课件 CH3 §6/§9/§13，2026-09-02）。

  outbox_jobs       "要投递"变成一行数据，与 run 同事务提交（Outbox，H21）；
                    索引 (status, next_attempt_at) 不是可选优化（H24）；
                    status/attempts 是**投递**状态与投递次数，不是执行的（H22/H25）
  dead_letter_jobs  病历不是垃圾桶（§9.2）：payload/attempts/error 全留，可人工 reprocess；
                    source 区分投递层死信（outbox）与执行层死信（worker）
  notify_outbox     dispatcher 门铃：INSERT 即 pg_notify——nudge 是优化，扫表是正确性（CH3 待查#4）
"""
from __future__ import annotations

from alembic import op

revision = "0003_k3_outbox"
down_revision = "0002_k2_foundation"
branch_labels = None
depends_on = None

UP = r"""
CREATE TABLE IF NOT EXISTS outbox_jobs (
    id              BIGSERIAL   PRIMARY KEY,
    org_id          TEXT        NOT NULL,
    job_type        TEXT        NOT NULL,
    payload         JSONB       NOT NULL DEFAULT '{}'::jsonb,   -- 只放 run_id/org_id（H23）
    queue           TEXT        NOT NULL DEFAULT 'interactive',
    status          TEXT        NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','dispatched','failed')),
    attempts        INTEGER     NOT NULL DEFAULT 0,             -- dispatcher 投递次数（max 10）
    max_attempts    INTEGER     NOT NULL DEFAULT 10,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_error      JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    dispatched_at   TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS outbox_jobs_pending ON outbox_jobs (status, next_attempt_at);
CREATE INDEX IF NOT EXISTS outbox_jobs_by_run  ON outbox_jobs ((payload->>'run_id'));

CREATE TABLE IF NOT EXISTS dead_letter_jobs (
    id              BIGSERIAL   PRIMARY KEY,
    org_id          TEXT        NOT NULL,
    source          TEXT        NOT NULL CHECK (source IN ('outbox','worker')),
    original_job_id BIGINT,
    job_type        TEXT        NOT NULL,
    payload         JSONB       NOT NULL DEFAULT '{}'::jsonb,
    attempts        INTEGER     NOT NULL DEFAULT 0,
    error           JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    reprocessed_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS dead_letter_open ON dead_letter_jobs (org_id, created_at) WHERE reprocessed_at IS NULL;

CREATE OR REPLACE FUNCTION notify_outbox() RETURNS trigger AS $$
BEGIN
  PERFORM pg_notify('outbox_jobs', NEW.id::text);
  RETURN NULL;
END; $$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS trg_notify_outbox ON outbox_jobs;
CREATE TRIGGER trg_notify_outbox AFTER INSERT ON outbox_jobs
  FOR EACH ROW EXECUTE FUNCTION notify_outbox();
"""

DOWN = r"""
DROP TRIGGER IF EXISTS trg_notify_outbox ON outbox_jobs;
DROP FUNCTION IF EXISTS notify_outbox();
DROP TABLE IF EXISTS dead_letter_jobs;
DROP TABLE IF EXISTS outbox_jobs;
"""


def upgrade() -> None:
    op.get_bind().exec_driver_sql(UP)


def downgrade() -> None:
    op.get_bind().exec_driver_sql(DOWN)
