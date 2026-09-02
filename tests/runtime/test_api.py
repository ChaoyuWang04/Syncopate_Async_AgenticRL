"""M9.3 · API 层的验收测试：越权打不穿 + 幂等在 HTTP 层也成立。

★ 这一份最重要的不是"接口能用"，是**越权测试打不穿**。
多租户系统里，一个能读到别人数据的接口比一个挂掉的接口危险得多 ——
后者会被立刻发现，前者可能几个月无人知晓。
"""

from __future__ import annotations

import asyncio
import uuid

import httpx
import pytest

from syncopate.runtime.api import create_app
from syncopate.runtime.db import Database

ACME = {"Authorization": "Bearer dev-token-acme"}

def _key() -> str:
    """★ 每次唯一。数据库**跨 pytest 运行持久**，写死的幂等键第二轮就已经存在了 ——
    表现是"第一次调用 created=False"，看起来像幂等坏了，其实是测试没隔离。"""
    return f"k_{uuid.uuid4().hex[:12]}"

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
    """同步外壳，内部用 httpx.ASGITransport 直连 ASGI 应用。

    ⚠️ **不用 `fastapi.testclient.TestClient`**：本环境里它（连最小 app 都）挂死，
    卡在自己的 portal 线程上。ASGITransport 不起线程，直接 await 应用。

    ★ 附带好处：startup 事件和请求跑在**同一个事件循环**里 ⇒
    asyncpg 的池不会踩「池绑定在创建它的循环上」那个坑
    （见 test_idempotency.with_db 的说明）。

    ⚠️ lifespan 必须包在**最外层**：startup 里才建连接池，进得比请求晚一步的话
    第一个请求就撞 "连接池没建"。

    ★ 一个测试共用**一个事件循环 + 一份 lifespan**：每次调用都重进 lifespan 的话，
    连接池会被反复建/关，而且第二次 startup 会看到上一次留下的引用。
    """

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


# --------------------------------------------------------------------------
# 越权：这一组是 M9.3 的主验收
# --------------------------------------------------------------------------


def test_no_token_is_rejected(client) -> None:
    assert client.post("/runs", json={"user_message": "x"}).status_code == 401


def test_bad_token_is_rejected(client) -> None:
    r = client.post("/runs", json={"user_message": "x"},
                    headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_org_id_in_body_is_ignored(client) -> None:
    """★★★ 前端传 org_id 必须**完全无效**。

    这里传一个别的 org，然后用那个 org 的 token 去读 —— 读不到才算通过。
    注意 Pydantic 模型里根本没有 org_id 字段，所以它会被直接丢弃；
    这条测试守的是"将来有人手滑把字段加回去"。
    """
    r = client.post("/runs", json={"user_message": "x", "org_id": "org_globex"},
                    headers=ACME)
    assert r.status_code == 201
    run_id = r.json()["run_id"]

    assert client.get(f"/runs/{run_id}", headers=GLOBEX).status_code == 404, \
        "请求体里的 org_id 生效了 —— 越权"
    assert client.get(f"/runs/{run_id}", headers=ACME).status_code == 200


def test_cannot_read_another_orgs_run(client) -> None:
    """★★ 知道 run_id 也读不到别人的。"""
    run_id = client.post("/runs", json={"user_message": "acme 的活"},
                         headers=ACME).json()["run_id"]
    assert client.get(f"/runs/{run_id}", headers=GLOBEX).status_code == 404


def test_other_org_and_nonexistent_are_indistinguishable(client) -> None:
    """★ 别人的 run 和不存在的 run 必须返回**同一个** 404。

    区分开（403 vs 404）就成了一个探测别人 run_id 是否存在的接口。
    """
    mine = client.post("/runs", json={"user_message": "x"}, headers=ACME).json()["run_id"]
    theirs = client.get(f"/runs/{mine}", headers=GLOBEX)
    ghost = client.get("/runs/run_completely_made_up", headers=GLOBEX)
    assert theirs.status_code == ghost.status_code == 404
    assert theirs.json() == ghost.json(), "两种 404 的响应体不同 ⇒ 可以据此探测"


# --------------------------------------------------------------------------
# 幂等在 HTTP 层
# --------------------------------------------------------------------------


def test_same_idempotency_key_returns_the_same_run(client) -> None:
    h = {**ACME, "Idempotency-Key": _key()}
    a = client.post("/runs", json={"user_message": "加预算"}, headers=h)
    b = client.post("/runs", json={"user_message": "加预算"}, headers=h)
    assert a.status_code == b.status_code == 201
    assert a.json()["run_id"] == b.json()["run_id"]
    assert a.json()["created"] is True and b.json()["created"] is False


def test_idempotent_replay_is_not_an_error(client) -> None:
    """★ 幂等命中返回 201 而不是 409。

    报错会让客户端以为出事了，从而**再重试一次** —— 那正好是我们要避免的。
    """
    h = {**ACME, "Idempotency-Key": _key()}
    client.post("/runs", json={"user_message": "x"}, headers=h)
    assert client.post("/runs", json={"user_message": "x"}, headers=h).status_code == 201


def test_idempotency_key_is_scoped_to_org(client) -> None:
    """两个 org 用同一个 key 不能互相挡。"""
    key = {"Idempotency-Key": _key()}
    a = client.post("/runs", json={"user_message": "x"}, headers={**ACME, **key})
    b = client.post("/runs", json={"user_message": "x"}, headers={**GLOBEX, **key})
    assert a.json()["created"] and b.json()["created"]
    assert a.json()["run_id"] != b.json()["run_id"]


# --------------------------------------------------------------------------
# 白名单与健康检查
# --------------------------------------------------------------------------


def test_response_never_leaks_internal_fields(client) -> None:
    """★ response_model 是白名单：内部字段不能因为"顺手 return 了整行"漏出去。"""
    body = client.post("/runs", json={"user_message": "x"}, headers=ACME).json()
    for leaked in ("lease_owner", "lease_expires_at", "attempts", "org_id",
                   "idempotency_key", "user_message", "error"):
        assert leaked not in body, f"内部字段 {leaked} 漏出去了"


def test_healthz_reports_instead_of_crashing(client) -> None:
    """健康检查的作用是**如实报告**，不是自己也挂掉 —— 压测场景②要靠它。"""
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] in ("ok", "degraded")


def test_validation_rejects_empty_message(client) -> None:
    assert client.post("/runs", json={"user_message": ""}, headers=ACME).status_code == 422


def test_invalid_automation_tier_is_rejected(client) -> None:
    r = client.post("/runs", json={"user_message": "x", "automation_tier": "Z"}, headers=ACME)
    assert r.status_code == 422
