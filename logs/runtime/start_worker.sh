#!/bin/bash
# B-4 测试期的 worker 启动器（临时件，正式化走 supervisor / 09 §0）
cd /workspace/Syncopate_Async_AgenticRL
set -a; . /workspace/.env; set +a
source .venv/bin/activate
export SYNCOPATE_DECIDER_URL=http://127.0.0.1:8100
# ★ 常驻 worker 只消费 org_demo（真人租户）；org_acme/globex 归测试套件
exec python -m syncopate.runtime.worker --org-id org_demo
