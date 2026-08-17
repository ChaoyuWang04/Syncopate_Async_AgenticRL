#!/usr/bin/env bash
# M9 · PostgreSQL 引导：把数据库做成**一条命令可重建的派生产物**。
#
#   bash scripts/pg_bootstrap.sh          # 幂等：已在跑就什么都不做
#   bash scripts/pg_bootstrap.sh --reset  # 推倒重来（会删数据）
#
# ============================================================================
# ★★★ 为什么 PGDATA 不在 /workspace —— 一条实测出来的物理限制
# ============================================================================
#
#   mfs 网络盘        chmod 700 → 仍是 777（FUSE，不支持权限位）
#   容器能力          !cap_sys_admin ⇒ 不能 mount，loop 挂 ext4 镜像这条路也堵死
#   PostgreSQL 要求   PGDATA 必须 0700 或 0750，否则**拒绝启动**（硬编码检查）
#
# ⇒ PGDATA 在物理上放不进 /workspace。所以拆成两件事：
#
#   PG 二进制/库   /workspace/tools/postgres/root   ← 工具，必须持久（bootstrap.sh 的约定）
#   deb 离线包     /workspace/tools/postgres/debs   ← 断网也能重装
#   PGDATA        本地盘（容器重启即丢）             ← 物理限制，无解
#   schema/迁移    **本仓库**                        ← 真相来源
#
# ⇒ **数据库是派生产物，不是真相来源。** 重启后丢的只是可重建的东西，
#   跑一次本脚本就回来了。⚠️ 真要长期保存业务数据，得用支持权限位的卷
#   或托管 PG —— 那是部署问题，不是开发期问题。
# ============================================================================

set -euo pipefail

PG_HOME="${PG_HOME:-/workspace/tools/postgres/root/usr/lib/postgresql/16}"
PG_SHARE="${PG_SHARE:-/workspace/tools/postgres/root/usr/share/postgresql/16}"
PG_LIB="${PG_LIB:-/workspace/tools/postgres/root/usr/lib/x86_64-linux-gnu}"
# `dpkg -x` 解出来的 libpq.so.5 不在 ldconfig 的搜索路径里（旧机器上 libpq5 是
# apt 装进系统的，所以一直没暴露）。⇒ 所有 PG 二进制都要自带库路径。
export LD_LIBRARY_PATH="$PG_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PGDATA="${PGDATA:-/var/lib/postgresql/16/syncopate}"
PGPORT="${PGPORT:-5432}"
DB_NAME="${DB_NAME:-syncopate}"
DB_USER="${DB_USER:-syncopate}"
DB_PASS="${DB_PASS:-syncopate}"
# 日志跟着 PGDATA 走：写死 /var/lib/... 的话，一旦 PGDATA 被换到别处（比如持久卷），
# 那个目录不会被建出来，pg_ctl -l 直接失败。
LOGFILE="${LOGFILE:-$(dirname "$PGDATA")/pg.log}"

step() { echo; echo "=== $* ==="; }
die()  { echo "❌ $*" >&2; exit 1; }

[[ -x "$PG_HOME/bin/postgres" ]] || die "找不到 $PG_HOME/bin/postgres
  持久卷上的 PG 没了？重装：
    mkdir -p /workspace/tools/postgres/root
    for d in /workspace/tools/postgres/debs/*.deb; do dpkg -x \"\$d\" /workspace/tools/postgres/root/; done"

# ---- postgres 系统用户（2026-08-17 补）----
# `dpkg -x` 只解包、**不跑 maintainer 脚本**，所以从 debs 重装出来的 PG 没有
# `postgres` 这个用户 —— 而下面每一步都是 `su postgres -c`。旧机器上它是当初
# `apt install` 建的，于是这个缺口在"一条命令重建"里藏了下来，只有干净机器才暴露。
if ! id postgres &>/dev/null; then
  step "创建 postgres 系统用户"
  groupadd -r postgres 2>/dev/null || true
  useradd -r -g postgres -d /var/lib/postgresql -s /bin/bash postgres \
    || die "创建 postgres 用户失败"
fi

if [[ "${1:-}" == "--reset" ]]; then
  step "重置：停服务、删数据"
  su postgres -c "LD_LIBRARY_PATH=$PG_LIB $PG_HOME/bin/pg_ctl -D $PGDATA stop -m immediate" 2>/dev/null || true
  rm -rf "$PGDATA"
fi

# ---- initdb（PGDATA 必须 0700，所以只能在本地盘）----
if [[ ! -s "$PGDATA/PG_VERSION" ]]; then
  step "initdb -> $PGDATA"
  mkdir -p "$PGDATA" /var/run/postgresql
  chown postgres:postgres "$PGDATA" /var/run/postgresql
  chmod 700 "$PGDATA"
  [[ "$(stat -c %a "$PGDATA")" == "700" ]] || die "PGDATA 权限设不成 0700 —— 是不是又放到网络盘上了？"
  su postgres -c "LD_LIBRARY_PATH=$PG_LIB PGSHAREDIR=$PG_SHARE $PG_HOME/bin/initdb -D $PGDATA -A trust -E UTF8 --locale=C" >/dev/null
fi

# ---- 启动（幂等）----
if ! "$PG_HOME/bin/pg_isready" -p "$PGPORT" -q 2>/dev/null; then
  step "启动 PostgreSQL"
  # ⚠️ `mkdir -p "$PGDATA"` 只 chown 了最里层那一级，日志写在**父目录**里 ⇒ 也要给权限，
  # 否则 pg_ctl -l 报 "Permission denied"，而错误信息里只字不提是父目录的事。
  mkdir -p "$(dirname "$LOGFILE")"
  chown postgres:postgres "$(dirname "$LOGFILE")" /var/run/postgresql 2>/dev/null || true
  su postgres -c "LD_LIBRARY_PATH=$PG_LIB PGSHAREDIR=$PG_SHARE $PG_HOME/bin/pg_ctl -D $PGDATA -o '-p $PGPORT' -l $LOGFILE start" >/dev/null
  for _ in $(seq 1 20); do
    "$PG_HOME/bin/pg_isready" -p "$PGPORT" -q 2>/dev/null && break
    sleep 0.5
  done
fi
"$PG_HOME/bin/pg_isready" -p "$PGPORT" || die "起不来，看 $LOGFILE"

# ---- 角色与库（幂等）----
psql_su() { su postgres -c "LD_LIBRARY_PATH=$PG_LIB $PG_HOME/bin/psql -p $PGPORT -tAc \"$1\""; }
if [[ "$(psql_su "select 1 from pg_roles where rolname='$DB_USER'")" != "1" ]]; then
  step "建角色 $DB_USER"
  psql_su "create user $DB_USER with password '$DB_PASS' createdb" >/dev/null
fi
if [[ "$(psql_su "select 1 from pg_database where datname='$DB_NAME'")" != "1" ]]; then
  step "建库 $DB_NAME"
  su postgres -c "LD_LIBRARY_PATH=$PG_LIB $PG_HOME/bin/createdb -p $PGPORT -O $DB_USER $DB_NAME"
fi

# ---- schema（真相来源在仓库里，见 syncopate/runtime/schema.sql）----
SCHEMA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/syncopate/runtime/schema.sql"
if [[ -f "$SCHEMA" ]]; then
  step "应用 schema"
  PGPASSWORD="$DB_PASS" "$PG_HOME/bin/psql" -h 127.0.0.1 -p "$PGPORT" -U "$DB_USER" -d "$DB_NAME" \
    -v ON_ERROR_STOP=1 -q -f "$SCHEMA"
  n=$(PGPASSWORD="$DB_PASS" "$PG_HOME/bin/psql" -h 127.0.0.1 -p "$PGPORT" -U "$DB_USER" -d "$DB_NAME" \
      -tAc "select count(*) from information_schema.tables where table_schema='public'")
  echo "    public schema 下 $n 张表"
fi

echo
echo "✅ PostgreSQL 就绪  postgresql://$DB_USER:$DB_PASS@127.0.0.1:$PGPORT/$DB_NAME"
