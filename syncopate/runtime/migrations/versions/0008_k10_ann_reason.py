"""0008 · run_annotations.reason_code：只有**负样本**必填（label<>'good'）；good 不需要病因。约束进 CHECK，不靠 API 记得。"""
from __future__ import annotations

from alembic import op

revision = "0008_k10_ann_reason"
down_revision = "0007_k10_flywheel"
branch_labels = None
depends_on = None

UP = r"""
ALTER TABLE run_annotations ALTER COLUMN reason_code DROP NOT NULL;
ALTER TABLE run_annotations ADD CONSTRAINT run_annotations_negative_needs_reason
    CHECK (label = 'good' OR reason_code IS NOT NULL);
"""

DOWN = r"""
ALTER TABLE run_annotations DROP CONSTRAINT IF EXISTS run_annotations_negative_needs_reason;
ALTER TABLE run_annotations ALTER COLUMN reason_code SET NOT NULL;
"""


def upgrade() -> None:
    op.get_bind().exec_driver_sql(UP)


def downgrade() -> None:
    op.get_bind().exec_driver_sql(DOWN)
