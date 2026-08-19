-- M9 · Runtime 的 8+1 张表（设计文档 §37）
--
-- ★ 这个文件是**真相来源**。数据库本身是派生产物 —— `scripts/pg_bootstrap.sh`
--   一条命令从这里重建。PGDATA 在 `/workspace/pgdata/16/syncopate`
--   （2026-08-17 换机器后：/workspace 是本地 XFS、权限位生效；旧机器的 mfs 才放不进）。
--
-- ★ 幂等：全部 `IF NOT EXISTS`，可以反复应用。
--
-- 8 张沿用 Harness 课的骨架，**本业务特有的列才是重点**：
--   tool_calls.external_idempotency_key   三层幂等的第三层，唯一被外部系统认的那层
--   audit_logs.param_source               参数来自用户还是工具返回 —— 防注入
--   agent_runs.automation_tier            自动化四档（设计文档 §3）
--   agent_steps.data_maturity_at_step     归因延迟是第一性约束，要记"决策时数据多熟"
--
-- 第 9 张 approval_cases 的最后三列是**数据飞轮的物理接口**（§37）：
--   modified_params / outcome_checked_at / outcome_result
--   没有它们飞轮转不起来。

-- ---------------------------------------------------------------------------
-- 枚举：写成 CHECK 而不是 PG enum type —— enum 加值要 ALTER TYPE，
-- 迁移期间会锁表；CHECK 改起来是一条 ALTER TABLE，且值域在 schema 里一眼可见。
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS agent_runs (
    id                BIGSERIAL PRIMARY KEY,
    run_id            TEXT        NOT NULL,
    org_id            TEXT        NOT NULL,
    -- ★ 请求级幂等：同一个 org 内 Idempotency-Key 唯一。见 §38 第一层。
    idempotency_key   TEXT,
    status            TEXT        NOT NULL DEFAULT 'queued'
                      CHECK (status IN ('queued','running','waiting_for_user',
                                        'succeeded','failed','cancelled')),
    -- ---- 本业务特有 ----
    intent            TEXT,                       -- I01/I07/I09… 设计文档 §7
    automation_tier   TEXT        CHECK (automation_tier IN ('A','B','C','D')),
    requires_approval BOOLEAN     NOT NULL DEFAULT FALSE,
    -- ---- 任务级幂等：状态机 + lease。见 §38 第二层 ----
    lease_owner       TEXT,
    lease_expires_at  TIMESTAMPTZ,
    attempt           INTEGER     NOT NULL DEFAULT 0,
    user_message      TEXT,
    -- ★ 端到端延迟的两个锚点（§19 要求「按意图分」的 P50/P95/P99）。
    -- 不能用 created_at→updated_at 代替：那里面混着排队时间，
    -- 而"用户等了多久"和"我们跑了多久"是两个不同的问题，混在一起两个都答不了。
    started_at        TIMESTAMPTZ,
    ended_at          TIMESTAMPTZ,
    result            JSONB,
    error             TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, run_id)
);

-- ★★ 请求级幂等的物理保证。PARTIAL INDEX：只对非空的 key 生效，
-- 这样"不带 Idempotency-Key 的请求"不会互相冲突。
CREATE UNIQUE INDEX IF NOT EXISTS agent_runs_idem_uniq
    ON agent_runs (org_id, idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS agent_runs_claimable
    ON agent_runs (status, lease_expires_at);

-- 流水（追加式，永不 UPDATE）—— SSE 断线补发靠 seq 定位
CREATE TABLE IF NOT EXISTS run_events (
    id         BIGSERIAL PRIMARY KEY,
    run_id     TEXT        NOT NULL,
    org_id     TEXT        NOT NULL,
    seq        INTEGER     NOT NULL,
    kind       TEXT        NOT NULL,
    payload    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- ★ 作用域必须是 (org_id, run_id)，不是 run_id。
    -- run_id 只在 org 内唯一（见 agent_runs 的 UNIQUE），两个 org 用同名 run_id 时
    -- 按 run_id 分配 seq 会被对方顶高 ⇒ **SSE 断线补发定位到错的位置**。
    UNIQUE (org_id, run_id, seq)
);
CREATE INDEX IF NOT EXISTS run_events_replay ON run_events (org_id, run_id, seq);

CREATE TABLE IF NOT EXISTS agent_steps (
    id         BIGSERIAL PRIMARY KEY,
    run_id     TEXT        NOT NULL,
    org_id     TEXT        NOT NULL,
    step       INTEGER     NOT NULL,
    phase      TEXT,
    -- ★ 本业务特有：决策时数据成熟到什么程度（归因延迟是第一性约束，设计文档 §0.3）
    data_maturity_at_step TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at   TIMESTAMPTZ,
    -- ★ 同 run_events：作用域必须带 org_id。run_id 只在 org 内唯一。
    UNIQUE (org_id, run_id, step)
);

CREATE TABLE IF NOT EXISTS model_calls (
    id            BIGSERIAL PRIMARY KEY,
    run_id        TEXT        NOT NULL,
    org_id        TEXT        NOT NULL,
    step          INTEGER,
    model         TEXT,
    prompt_tokens INTEGER     NOT NULL DEFAULT 0,
    output_tokens INTEGER     NOT NULL DEFAULT 0,
    latency_ms    INTEGER,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id         BIGSERIAL PRIMARY KEY,
    run_id     TEXT        NOT NULL,
    org_id     TEXT        NOT NULL,
    step       INTEGER,
    tool       TEXT        NOT NULL,
    arguments  JSONB       NOT NULL DEFAULT '{}'::jsonb,
    ok         BOOLEAN,
    result     JSONB,
    error      TEXT,
    -- ★★★ 三层幂等的第三层，**唯一被外部系统认的那层**（§38）。
    --
    -- 实查过的事实：Meta Marketing API **本身没有幂等机制**。沙盒里我们把它建成
    -- "平台支持"，是为了让模型学会**传这个键**；真实接入时这层保证**由我们兑现** ——
    -- 就是这张表 + 下面那个唯一索引。缺了它，重试一次就是多花一次钱。
    external_idempotency_key TEXT,
    -- 同一个幂等键第二次调用时，这里记它命中了哪一行（用于返回原结果而不是重放）
    replayed_from BIGINT REFERENCES tool_calls (id),
    -- ⚠️ 这一列**曾经是 TEXT**（而 model_calls.latency_ms 是 INTEGER）。
    -- 一直没往里写值所以没暴露，但 §19 的「工具调用 P95 < 2s」要按它算 ——
    -- 类型错了那条门槛就量不了。2026-08-17 改正并接上写入。
    latency_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ★★ 工具级幂等的物理保证：同一个 org 的同一个外部幂等键，**全局只能成功一次**。
-- 不按 run_id 分组 —— 跨 run 重试同一个动作同样是重复扣款。
CREATE UNIQUE INDEX IF NOT EXISTS tool_calls_external_idem_uniq
    ON tool_calls (org_id, external_idempotency_key)
    WHERE external_idempotency_key IS NOT NULL AND replayed_from IS NULL;

CREATE TABLE IF NOT EXISTS checkpoints (
    id         BIGSERIAL PRIMARY KEY,
    run_id     TEXT        NOT NULL,
    org_id     TEXT        NOT NULL,
    step       INTEGER     NOT NULL,
    state      JSONB       NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, run_id, step)
);

CREATE TABLE IF NOT EXISTS usage_records (
    id          BIGSERIAL PRIMARY KEY,
    org_id      TEXT        NOT NULL,
    run_id      TEXT,
    day         DATE        NOT NULL DEFAULT CURRENT_DATE,
    tokens_in   BIGINT      NOT NULL DEFAULT 0,
    tokens_out  BIGINT      NOT NULL DEFAULT 0,
    cost_micros BIGINT      NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS usage_by_org_day ON usage_records (org_id, day);

CREATE TABLE IF NOT EXISTS audit_logs (
    id         BIGSERIAL PRIMARY KEY,
    run_id     TEXT,
    org_id     TEXT        NOT NULL,
    action     TEXT        NOT NULL,
    object_key TEXT,
    -- ★★★ 本业务特有：这个参数是**用户给的**还是**工具返回里读出来的**。
    --
    -- 防注入的关键证据：设计文档 §27.2「假设模型已被策反」。工具返回是 data 不是
    -- instruction —— 一个从工具返回里长出来的写参数，和用户明确要求的写参数，
    -- 事后追责时是完全不同的两件事。不记这一列，注入发生了也查不出来。
    param_source TEXT CHECK (param_source IN ('user','tool_result','system','model')),
    detail     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS audit_by_org ON audit_logs (org_id, created_at DESC);

-- ---------------------------------------------------------------------------
-- ★ 第 9 张：审批网关（§37）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS approval_cases (
    id              BIGSERIAL PRIMARY KEY,
    case_ref        TEXT        NOT NULL,
    run_id          TEXT        NOT NULL,
    org_id          TEXT        NOT NULL,
    action_type     TEXT        NOT NULL,       -- campaign.create / scale_budget / ...
    proposed_params JSONB       NOT NULL DEFAULT '{}'::jsonb,
    rationale       TEXT,
    evidence        JSONB       NOT NULL DEFAULT '{}'::jsonb,   -- ★ agent 给出的证据
    -- ★ 为什么停下来。六个降级触发器之一，见 runtime/gateway.py。
    -- 触发条件**不能靠 token 概率** —— LLM 编造时往往最自信（§39）。
    trigger_reason  TEXT,
    status          TEXT        NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','approved','rejected','modified','expired')),
    reviewer_id     TEXT,
    reviewed_at     TIMESTAMPTZ,
    -- ★★★ 下面三列是**数据飞轮的物理接口**（§37）。没有它们飞轮转不起来。
    modified_params    JSONB,          -- 人改了什么 → 回路 2（天级）的燃料
    outcome_checked_at TIMESTAMPTZ,    -- D7 后回填
    outcome_result     JSONB,          -- 真实回收结果 → 回路 3（周级），唯一的真值
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, case_ref)
);
CREATE INDEX IF NOT EXISTS approval_pending ON approval_cases (org_id, status, created_at);
-- 飞轮回路 3 的扫描入口：已批准但还没回填 D7 结果的
CREATE INDEX IF NOT EXISTS approval_awaiting_outcome
    ON approval_cases (outcome_checked_at) WHERE status IN ('approved','modified');

-- ---------------------------------------------------------------------------
-- ★ RAG 语料（M9 之后新增，设计见 docs/syncopate/12-rag-runtime-design.md）
--
-- 在这两张表之前，项目里**根本没有"语料"这个东西** —— 政策条款和复盘结论
-- 由 WorldBuilder 每条 case 现造、只活在内存里、跑完就没了。
--
-- ★ scope 是多租户边界：平台政策（Meta 的规矩）是全局的，内部 SOP 是 org 私有的。
--   查询一律 `WHERE scope='global' OR scope=$org` —— 越权在 SQL 里挡，同 §36 的纪律。
--
-- ★ 生效期与取代关系**存原始事实**，`expired` / `superseded_by` 是**查询时算**的：
--   取代关系从新条款反查（新版本声明 supersedes=<旧 id>），不在旧条款上回写 ——
--   语料是增量追加的，回写等于每加一版都要改历史。
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS policy_clauses (
    id           BIGSERIAL PRIMARY KEY,
    scope        TEXT        NOT NULL DEFAULT 'global',   -- 'global' | <org_id>
    clause_id    TEXT        NOT NULL,
    title        TEXT        NOT NULL,
    body         TEXT        NOT NULL,
    section_path TEXT        NOT NULL DEFAULT '',
    platform     TEXT,
    region       TEXT,
    valid_from   TIMESTAMPTZ,
    valid_to     TIMESTAMPTZ,                             -- NULL = 长期有效
    version      TEXT        NOT NULL DEFAULT 'v1',
    supersedes   TEXT,                                    -- ★ 指向被它取代的旧 clause_id
    source_doc   TEXT        NOT NULL DEFAULT '',
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (scope, clause_id)
);
CREATE INDEX IF NOT EXISTS policy_by_scope  ON policy_clauses (scope);
-- 反查取代关系用（§4）
CREATE INDEX IF NOT EXISTS policy_supersedes ON policy_clauses (scope, supersedes)
    WHERE supersedes IS NOT NULL;

CREATE TABLE IF NOT EXISTS insights (
    id            BIGSERIAL PRIMARY KEY,
    scope         TEXT        NOT NULL DEFAULT 'global',
    claim_id      TEXT        NOT NULL,
    claim         TEXT        NOT NULL,
    scope_json    JSONB       NOT NULL DEFAULT '{}'::jsonb,  -- region / product / period
    evidence      TEXT        NOT NULL DEFAULT '',
    confidence    TEXT        NOT NULL DEFAULT 'medium'
                  CHECK (confidence IN ('low','medium','high')),
    source_doc    TEXT        NOT NULL DEFAULT '',
    -- ★ status 是 M12 飞轮的物理接口：跑出相反结论时把旧条目标 refuted 而不是删掉，
    --   **你需要知道"我们曾经这么以为"**。
    status        TEXT        NOT NULL DEFAULT 'active'
                  CHECK (status IN ('active','superseded','refuted')),
    superseded_by TEXT,
    recorded_at   TIMESTAMPTZ,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (scope, claim_id)
);
CREATE INDEX IF NOT EXISTS insights_by_scope ON insights (scope, status);

-- ══════════════════════════════════════════════════════════════════════════
-- B-2 · 记忆库与安全线表（2026-08-19）
-- ══════════════════════════════════════════════════════════════════════════

-- 记忆库。四个 lane：
--   episodic  历史投放动作   ★ **系统维护，agent 不可写**（硬边界，等价于 403）
--   semantic  素材与受众属性
--   business  优化干预效果
--   risk      风控标记      ★ 写它之前必须先调 risk.check_account
CREATE TABLE IF NOT EXISTS memory_records (
    id            BIGSERIAL PRIMARY KEY,
    org_id        TEXT        NOT NULL,
    record_id     TEXT        NOT NULL,
    lane          TEXT        NOT NULL
                  CHECK (lane IN ('episodic','semantic','business','risk')),
    subject       JSONB       NOT NULL DEFAULT '{}'::jsonb,   -- account/campaign/creative/region
    content       TEXT        NOT NULL,
    confidence    NUMERIC(3,2) NOT NULL,
    evidence_refs JSONB       NOT NULL DEFAULT '[]'::jsonb,
    -- ★ TTL：`memory.search` 要自动剔除已过期的（沙盒描述里的原话）。
    --   ⚠️ 但 `memory.read` **不剔除** —— 它明写「不校验这条记忆现在还成不成立」。
    --     两个工具对同一条记录的可见性刻意不同，别"顺手统一"。
    expires_at    TIMESTAMPTZ,
    -- ★ 作废是**标记**不是删除：「你需要知道『我们曾经这么以为』」。
    invalidated_at TIMESTAMPTZ,
    invalidate_reason TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, record_id)
);
CREATE INDEX IF NOT EXISTS memory_by_lane ON memory_records (org_id, lane);

-- 记忆写入**提案**。★ 沙盒描述：「不会立即入库，需经审核」
-- ⇒ 提案和记录是两张表。写进 memory_records 的唯一路径是审核通过，
--   不是这张表的 INSERT —— 否则"需经审核"就只是一句话。
CREATE TABLE IF NOT EXISTS memory_proposals (
    id            BIGSERIAL PRIMARY KEY,
    org_id        TEXT        NOT NULL,
    run_id        TEXT        NOT NULL,
    kind          TEXT        NOT NULL
                  CHECK (kind IN ('write','invalidate','conflict_resolve')),
    payload       JSONB       NOT NULL,
    status        TEXT        NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending','approved','rejected')),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS memory_proposals_by_run ON memory_proposals (org_id, run_id);

-- 投放安全线（内部每周更新）。
-- ⚠️⚠️ **表里不存 `expired` 这一列，也不许查询时算出来。**
--   `axes.py` 的纪律：「工具不替模型判断过没过期，只如实返回 valid_to。
--   真实世界里没人会在返回里塞一个 expired: true。模型必须自己拿它和今天比。」
--   ⇒ 加一个 expired 字段 = 把这道判断从模型手里拿走，且与训练侧不一致。
CREATE TABLE IF NOT EXISTS safety_lines (
    id            BIGSERIAL PRIMARY KEY,
    org_id        TEXT        NOT NULL,
    product_id    TEXT        NOT NULL,
    region        TEXT        NOT NULL,
    cpi_d7_max    NUMERIC(10,2),
    roas_d7_min   NUMERIC(10,4),
    retention_d1_min NUMERIC(10,4),
    daily_budget_max INTEGER,
    valid_from    DATE        NOT NULL,
    valid_to      DATE        NOT NULL,
    UNIQUE (org_id, product_id, region, valid_from)
);
CREATE INDEX IF NOT EXISTS safety_lines_lookup ON safety_lines (org_id, product_id, region);

-- ══════════════════════════════════════════════════════════════════════════
-- B-2 第五批 · 数据源表（2026-08-19）
-- ══════════════════════════════════════════════════════════════════════════

-- 行业基准。⚠️ **不是决策依据**（决定能不能扩量要用 safety_lines 的内部安全线）。
CREATE TABLE IF NOT EXISTS industry_baselines (
    id          BIGSERIAL PRIMARY KEY,
    platform    TEXT NOT NULL,
    game_genre  TEXT NOT NULL,
    metric      TEXT NOT NULL,
    p25         NUMERIC(12,4),
    p50         NUMERIC(12,4),
    p75         NUMERIC(12,4),
    sample_size INTEGER,
    UNIQUE (platform, game_genre, metric)
);

-- 时令活动。只给背景，不判断该不该投。
CREATE TABLE IF NOT EXISTS seasonal_events (
    id           BIGSERIAL PRIMARY KEY,
    region       TEXT NOT NULL,
    event        TEXT NOT NULL,
    event_date   DATE NOT NULL,
    lift_factor  NUMERIC(6,2),
    creative_tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    UNIQUE (region, event, event_date)
);

-- 账户级预算调整规则。⚠️ 只给规则，**不做风控判断**（那在 account_risk）。
CREATE TABLE IF NOT EXISTS budget_rules (
    org_id            TEXT PRIMARY KEY,
    max_increase_pct  NUMERIC(6,2) NOT NULL DEFAULT 20,
    approval_threshold INTEGER      NOT NULL DEFAULT 100000,
    risk_check_required BOOLEAN     NOT NULL DEFAULT true,
    monthly_cap       INTEGER
);

-- 账户风控。⚠️ 只看账户状态，**不判断金额合不合政策**。
CREATE TABLE IF NOT EXISTS account_risk (
    org_id      TEXT NOT NULL,
    account_id  TEXT NOT NULL,
    flags       JSONB NOT NULL DEFAULT '[]'::jsonb,
    state       TEXT  NOT NULL DEFAULT 'normal'
                CHECK (state IN ('normal','restricted','frozen')),
    allow_increase BOOLEAN NOT NULL DEFAULT true,
    PRIMARY KEY (org_id, account_id)
);

-- 打法库。⚠️ `anomaly_type` 必须是 detect_anomalies 真的会返回的那些。
CREATE TABLE IF NOT EXISTS playbooks (
    anomaly_type TEXT PRIMARY KEY,
    steps        JSONB NOT NULL DEFAULT '[]'::jsonb,
    cautions     JSONB NOT NULL DEFAULT '[]'::jsonb
);

-- 素材 feature 的地域 lift。★★ **必须逐地域存**：
-- 同一个 feature 在不同地域可能**符号相反**，混在一起算会得出一个两头都不对的数。
CREATE TABLE IF NOT EXISTS feature_lifts (
    id          BIGSERIAL PRIMARY KEY,
    feature     TEXT NOT NULL,
    region      TEXT NOT NULL,
    product_id  TEXT,
    lift        NUMERIC(8,4) NOT NULL,
    ci_low      NUMERIC(8,4),
    ci_high     NUMERIC(8,4),
    n_treatment INTEGER,
    n_control   INTEGER,
    UNIQUE (feature, region, product_id)
);

-- 地域表现。⚠️ 素材条数少的地域，数字本身就不可信 ⇒ `asset_count` **必须返回**，
-- 否则模型没法判断"这个地域不行"和"这个地域样本太少"的区别。
CREATE TABLE IF NOT EXISTS geo_performance (
    id          BIGSERIAL PRIMARY KEY,
    product_id  TEXT NOT NULL,
    region      TEXT NOT NULL,
    roas_d7     NUMERIC(8,4),
    cpi_d7      NUMERIC(10,2),
    asset_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE (product_id, region)
);
