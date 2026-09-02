"""0005 · K8 平台去重账本（课件 CH8 §11：副作用的真相在下游，本地只知道"我用哪个 key 发过什么"）。

我们的"下游"是自家假平台（D-2：不接真 Meta）⇒ 课件里最不可控的"下游认不认 key"在这里退化成
查自己的表：platform_ledger(idempotency_key → response)。FakeAdPlatform 写穿到这里，对账任务按键回查。
"""
from __future__ import annotations

from alembic import op

revision = "0005_k8_platform_ledger"
down_revision = "0004_k6_tool_calls"
branch_labels = None
depends_on = None

UP = r"""
CREATE TABLE IF NOT EXISTS platform_ledger (
    idempotency_key TEXT        PRIMARY KEY,
    tool            TEXT        NOT NULL,
    response        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

DOWN = r"""
DROP TABLE IF EXISTS platform_ledger;
"""


def upgrade() -> None:
    op.get_bind().exec_driver_sql(UP)


def downgrade() -> None:
    op.get_bind().exec_driver_sql(DOWN)
