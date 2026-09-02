"""0002 · K2 地基：课件 CH2 承诺过但 schema 里没有的列/约束/触发器一次建齐（2026-09-02）。

对应 27 §4 K2-1/K2-2/K2-4/K2-5 与 29 组 B 的 🔶/❌ 行：
  agent_runs   attempt→attempts（三个 attempts 分账口径，H25/H28）· run_type（H06 分发校验的地基）
               · input_hash（幂等第二把锁，H11）· cancel_requested_at / resume_token（H12）
               · last_seq（seq 领号器，H13）· version（CAS 守卫，K4/K8）· parent_run_id / rerun_reason（K4-5）
  run_events   删与 UNIQUE(org_id,run_id,seq) 完全重复的索引（H16）
  agent_runs   updated_at 触发器（H14：DEFAULT now() 只在 INSERT 生效一次）· 列表页索引 (org,status,created_at)
  tool_calls   五态 status（K6 消费）· ended_at（两阶段写入）· side_effect + CHECK(有副作用 ⇒ 必带幂等键，H19）
               · duplicate_of（skipped_duplicate 行引用首次调用）
  usage_records call_index + 部分唯一索引（H15 账单翻倍）
"""
from __future__ import annotations

from alembic import op

revision = "0002_k2_foundation"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None

UP = r"""
-- ---- agent_runs ------------------------------------------------------------
ALTER TABLE agent_runs RENAME COLUMN attempt TO attempts;
ALTER TABLE agent_runs
    ADD COLUMN run_type            TEXT        NOT NULL DEFAULT 'chat',
    ADD COLUMN input_hash          TEXT,
    ADD COLUMN cancel_requested_at TIMESTAMPTZ,
    ADD COLUMN resume_token        TEXT,
    ADD COLUMN last_seq            INTEGER     NOT NULL DEFAULT 0,
    ADD COLUMN version             INTEGER     NOT NULL DEFAULT 0,
    ADD COLUMN parent_run_id       TEXT,
    ADD COLUMN rerun_reason        TEXT;
-- 存量 run 的 last_seq 追平已有事件，否则领号器会从 1 重新发号撞 UNIQUE
UPDATE agent_runs r
   SET last_seq = COALESCE((SELECT max(e.seq) FROM run_events e
                             WHERE e.org_id = r.org_id AND e.run_id = r.run_id), 0);
CREATE INDEX IF NOT EXISTS agent_runs_list ON agent_runs (org_id, status, created_at);

-- ---- updated_at 触发器（H14）--------------------------------------------------
CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS trigger AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END; $$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS trg_agent_runs_touch ON agent_runs;
CREATE TRIGGER trg_agent_runs_touch BEFORE UPDATE ON agent_runs
  FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

-- ---- run_events：与 UNIQUE 完全重复的索引（H16）----------------------------------
DROP INDEX IF EXISTS run_events_replay;

-- ---- tool_calls 五态 + 两阶段 + 副作用约束 ------------------------------------------
ALTER TABLE tool_calls
    ADD COLUMN status       TEXT NOT NULL DEFAULT 'running'
               CHECK (status IN ('running','succeeded','failed','skipped_duplicate','response_lost')),
    ADD COLUMN ended_at     TIMESTAMPTZ,
    ADD COLUMN side_effect  BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN duplicate_of BIGINT REFERENCES tool_calls (id);
UPDATE tool_calls
   SET status = CASE WHEN replayed_from IS NOT NULL THEN 'skipped_duplicate'
                     WHEN ok IS TRUE  THEN 'succeeded'
                     WHEN ok IS FALSE THEN 'failed'
                     ELSE 'running' END,
       duplicate_of = replayed_from,
       ended_at = CASE WHEN ok IS NOT NULL THEN created_at ELSE NULL END;
ALTER TABLE tool_calls
    ADD CONSTRAINT tool_calls_side_effect_needs_key
    CHECK (NOT side_effect OR external_idempotency_key IS NOT NULL);

-- ---- usage_records 账单防重（H15）：粒度 = 每 run 每次记账一行 ------------------------
ALTER TABLE usage_records ADD COLUMN call_index INTEGER NOT NULL DEFAULT 0;
UPDATE usage_records u SET call_index = s.rn - 1
  FROM (SELECT id, row_number() OVER (PARTITION BY org_id, run_id ORDER BY id) AS rn
          FROM usage_records WHERE run_id IS NOT NULL) s
 WHERE u.id = s.id;
CREATE UNIQUE INDEX IF NOT EXISTS usage_records_once
    ON usage_records (org_id, run_id, call_index) WHERE run_id IS NOT NULL;
"""

DOWN = r"""
DROP INDEX IF EXISTS usage_records_once;
ALTER TABLE usage_records DROP COLUMN IF EXISTS call_index;
ALTER TABLE tool_calls DROP CONSTRAINT IF EXISTS tool_calls_side_effect_needs_key;
ALTER TABLE tool_calls DROP COLUMN IF EXISTS duplicate_of, DROP COLUMN IF EXISTS side_effect,
                       DROP COLUMN IF EXISTS ended_at, DROP COLUMN IF EXISTS status;
CREATE INDEX IF NOT EXISTS run_events_replay ON run_events (org_id, run_id, seq);
DROP TRIGGER IF EXISTS trg_agent_runs_touch ON agent_runs;
DROP FUNCTION IF EXISTS touch_updated_at();
DROP INDEX IF EXISTS agent_runs_list;
ALTER TABLE agent_runs DROP COLUMN IF EXISTS rerun_reason, DROP COLUMN IF EXISTS parent_run_id,
                       DROP COLUMN IF EXISTS version, DROP COLUMN IF EXISTS last_seq,
                       DROP COLUMN IF EXISTS resume_token, DROP COLUMN IF EXISTS cancel_requested_at,
                       DROP COLUMN IF EXISTS input_hash, DROP COLUMN IF EXISTS run_type;
ALTER TABLE agent_runs RENAME COLUMN attempts TO attempt;
"""


def upgrade() -> None:
    op.get_bind().exec_driver_sql(UP)


def downgrade() -> None:
    op.get_bind().exec_driver_sql(DOWN)
