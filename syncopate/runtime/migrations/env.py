"""Alembic 环境：同步 psycopg 引擎只给迁移用；业务代码仍是 asyncpg 裸 SQL（不引 ORM）。

DSN 来源优先级：测试注入 `config.set_main_option("syncopate.dsn", …)` > 环境变量
SYNCOPATE_PG_DSN > db.py 的默认值。三者是同一条连接串的不同注入口，不是三份配置。
"""
from __future__ import annotations

import os

from alembic import context
from sqlalchemy import create_engine, pool

config = context.config
_DEFAULT = "postgresql://syncopate:syncopate@127.0.0.1:5432/syncopate"


def _url() -> str:
    dsn = (config.get_main_option("syncopate.dsn")
           or os.environ.get("SYNCOPATE_PG_DSN") or _DEFAULT)
    # asyncpg 风格的 postgresql:// 转成 SQLAlchemy+psycopg 方言
    return dsn.replace("postgresql://", "postgresql+psycopg://", 1) \
        if dsn.startswith("postgresql://") else dsn


def run_migrations_offline() -> None:
    context.configure(url=_url(), literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        # 每个迁移一个事务：单个迁移失败整段回滚，已过的迁移不受影响；
        # 需要 CONCURRENTLY 的迁移自己用 autocommit_block（28 P-06）。
        context.configure(connection=connection, transaction_per_migration=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
