"""F-1 · 会话门面（chatbox 壳的载体，22 §I-1）。

★ 会话只是组织方式：run 的幂等/审批/事件语义一律不动 —— 这里主要钉三件事：
  ① 越权（别人的会话读不到、发不进）② automation_tier 在新入口**必填**
  ③ 历史回放顺序 = 提交顺序。
"""

from __future__ import annotations

import asyncio
import uuid

import httpx
import pytest

from syncopate.runtime.api import create_app
from syncopate.runtime.db import Database

ACME = {"Authorization": "Bearer dev-token-acme"}
GLOBEX = {"Authorization": "Bearer dev-token-globex"}


def _pg_available() -> bool:
    async def probe() -> bool:
        db = Database()
        try:
            await db.connect(max_size=2)
            await db.close()
            return True
        except Exception:
            return False
    return asyncio.run(probe())


pytestmark = pytest.mark.skipif(
    not _pg_available(), reason="需要 PostgreSQL：bash scripts/pg_bootstrap.sh")


class Client:
    """同 test_api.Client：ASGITransport 直连，lifespan 包最外层（那边有全文注释）。"""

    def __init__(self, app):
        self.app = app
        self.loop = asyncio.new_event_loop()
        self._lifespan = app.router.lifespan_context(app)
        self.loop.run_until_complete(self._lifespan.__aenter__())
        self._http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t")

    def _call(self, method: str, url: str, **kw):
        return self.loop.run_until_complete(self._http.request(method, url, **kw))

    def get(self, url, **kw):  return self._call("GET", url, **kw)
    def post(self, url, **kw): return self._call("POST", url, **kw)

    def close(self) -> None:
        self.loop.run_until_complete(self._http.aclose())
        self.loop.run_until_complete(self._lifespan.__aexit__(None, None, None))
        self.loop.close()


@pytest.fixture()
def client():
    c = Client(create_app())
    yield c
    c.close()


def _conv(client, headers=ACME, title="测试会话") -> str:
    r = client.post("/conversations", json={"title": title}, headers=headers)
    assert r.status_code == 201
    return r.json()["conversation_id"]


def _msg(text: str, tier: str = "C") -> dict:
    return {"user_message": text, "intent": "I01", "automation_tier": tier}


# ── 越权：会话是新资源，防线必须和 run 同级 ────────────────────────────────

def test_no_token_is_rejected(client) -> None:
    assert client.post("/conversations", json={}).status_code == 401


def test_foreign_conversation_is_indistinguishable_from_missing(client) -> None:
    """★ 别人的会话和不存在的会话必须是**同一个** 404 —— 否则这个接口
    就成了探测别人 conversation_id 的工具（同 run 的那条纪律）。"""
    cid = _conv(client, headers=ACME)
    r1 = client.get(f"/conversations/{cid}/messages", headers=GLOBEX)
    r2 = client.get(f"/conversations/conv_{uuid.uuid4().hex[:12]}/messages",
                    headers=GLOBEX)
    assert r1.status_code == r2.status_code == 404
    assert r1.json() == r2.json()
    # 发消息同理
    r3 = client.post(f"/conversations/{cid}/messages", json=_msg("x"), headers=GLOBEX)
    assert r3.status_code == 404


def test_listing_only_shows_own_conversations(client) -> None:
    cid = _conv(client, headers=ACME, title=f"acme-{uuid.uuid4().hex[:6]}")
    ids = [c["conversation_id"] for c in
           client.get("/conversations", headers=GLOBEX).json()]
    assert cid not in ids


# ── automation_tier 必填（09 §4.6.4 的缺口在新入口关掉）─────────────────────

def test_message_does_not_require_a_declared_tier(client) -> None:
    """★ 2026-08-20 改判：档位**不再要调用方填**（`22 §I` 后续，Chaoyu）。

    09 §4.6.4 的「automation_tier 应当必填」换了个解法关闭：不是逼人填，
    而是让它有一个不依赖任何人填写的来源 —— `tier_policy.derive_tier`
    从「动作是什么」推导。给了也只能往严了拉（more_cautious）。
    """
    cid = _conv(client)
    r = client.post(f"/conversations/{cid}/messages",
                    json={"user_message": "查一下指标", "intent": "I01"},
                    headers=ACME)
    assert r.status_code == 201


def test_message_rejects_invalid_tier(client) -> None:
    cid = _conv(client)
    r = client.post(f"/conversations/{cid}/messages", json=_msg("x", tier="E"),
                    headers=ACME)
    assert r.status_code == 422


# ── 消息 = run：语义原样 ───────────────────────────────────────────────────

def test_message_creates_a_queued_run_bound_to_conversation(client) -> None:
    cid = _conv(client)
    r = client.post(f"/conversations/{cid}/messages", json=_msg("查 CMP_1 指标"),
                    headers=ACME)
    assert r.status_code == 201
    run = r.json()
    assert run["status"] == "queued" and run["automation_tier"] == "C"
    msgs = client.get(f"/conversations/{cid}/messages", headers=ACME).json()
    assert [m["run_id"] for m in msgs] == [run["run_id"]]


def test_history_replays_in_submission_order(client) -> None:
    cid = _conv(client)
    rids = [client.post(f"/conversations/{cid}/messages", json=_msg(f"第 {i} 条"),
                        headers=ACME).json()["run_id"] for i in range(3)]
    msgs = client.get(f"/conversations/{cid}/messages", headers=ACME).json()
    assert [m["run_id"] for m in msgs] == rids
    assert [m["user_message"] for m in msgs] == ["第 0 条", "第 1 条", "第 2 条"]


def test_message_idempotency_key_still_works(client) -> None:
    """★ 会话入口不许削弱 run 的请求级幂等。"""
    cid = _conv(client)
    key = f"k_{uuid.uuid4().hex[:12]}"
    r1 = client.post(f"/conversations/{cid}/messages", json=_msg("同一条"),
                     headers={**ACME, "Idempotency-Key": key})
    r2 = client.post(f"/conversations/{cid}/messages", json=_msg("同一条"),
                     headers={**ACME, "Idempotency-Key": key})
    assert r1.json()["run_id"] == r2.json()["run_id"]
    assert r1.json()["created"] is True and r2.json()["created"] is False
    msgs = client.get(f"/conversations/{cid}/messages", headers=ACME).json()
    assert len(msgs) == 1              # 历史里也只有一条，不是两条


def test_conversation_list_reports_run_count(client) -> None:
    cid = _conv(client, title=f"计数-{uuid.uuid4().hex[:6]}")
    client.post(f"/conversations/{cid}/messages", json=_msg("a"), headers=ACME)
    client.post(f"/conversations/{cid}/messages", json=_msg("b"), headers=ACME)
    convs = {c["conversation_id"]: c for c in
             client.get("/conversations", headers=ACME).json()}
    assert convs[cid]["runs"] == 2
