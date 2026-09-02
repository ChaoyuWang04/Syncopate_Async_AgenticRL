#!/usr/bin/env bash
# K3 · Redis 引导（2026-09-02，§16-1 裁定 Celery + Redis 后新建）：和 pg_bootstrap 同一纪律——
# **Redis 是派生产物，不是事实来源**（事实在 PostgreSQL 的 outbox/agent_runs 里，课件 CH3 §6/附录A §5.2）。
#
#   bash scripts/redis_bootstrap.sh          # 幂等：已在跑就什么都不做
#   bash scripts/redis_bootstrap.sh --reset  # 停服务、删 AOF/RDB
#
# 配置里每一项都对应 28 号坑表的一条（R-01…），别"顺手"改回默认：
#   bind 127.0.0.1 + requirepass      R-03  密码泄漏 = 能往队列投毒、绕过 API 全部入口闸（课件 H85）
#   appendonly yes / everysec         R-01  默认 RDB 快照会丢最近几分钟的消息；AOF 把窗口压到 1s
#   maxmemory-policy noeviction       R-02  任何 LRU/LFU 策略都会**静默淘汰队列消息**——broker 只能 noeviction
#   maxmemory 1gb                     R-02  不设上限 = 主机 OOM；设了上限 + noeviction = 写入报错（可见，正确）
#   databases 16 + 约定 db 编号       R-07  broker=0 · 限流=1 · 信号量=2 · 缓存=3（四职分库，FLUSHDB 不互伤）
set -euo pipefail

REDIS_HOME="${REDIS_HOME:-$HOME/Downloads/ENTER/envs/syncopate-infra}"
REDIS_DIR="${REDIS_DIR:-$HOME/.local/share/syncopate/redis}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASS="${REDIS_PASS:-syncopate-dev}"     # 开发默认；生产从 secret 注入，⛔ 不进日志/URL
CONF="$REDIS_DIR/redis.conf"

step() { echo; echo "=== $* ==="; }
die()  { echo "❌ $*" >&2; exit 1; }
[[ -x "$REDIS_HOME/bin/redis-server" ]] || die "找不到 $REDIS_HOME/bin/redis-server（conda create -n syncopate-infra -c conda-forge redis-server）"
RCLI="$REDIS_HOME/bin/redis-cli -p $REDIS_PORT -a $REDIS_PASS --no-auth-warning"

if [[ "${1:-}" == "--reset" ]]; then
  step "重置：停服务、删持久化文件"
  $RCLI shutdown nosave 2>/dev/null || true
  rm -rf "$REDIS_DIR"
fi

mkdir -p "$REDIS_DIR"
cat > "$CONF" <<CONF
bind 127.0.0.1
port $REDIS_PORT
protected-mode yes
requirepass $REDIS_PASS
daemonize yes
pidfile $REDIS_DIR/redis.pid
logfile $REDIS_DIR/redis.log
dir $REDIS_DIR
databases 16
# R-01 持久化：AOF 每秒 fsync；RDB 保留作二级快照
appendonly yes
appendfsync everysec
save 900 1
# R-02 内存：上限 + 不淘汰（broker 的唯一正确策略）
maxmemory 1gb
maxmemory-policy noeviction
CONF

if ! $RCLI ping 2>/dev/null | grep -q PONG; then
  step "启动 Redis"
  "$REDIS_HOME/bin/redis-server" "$CONF"
  for _ in $(seq 1 20); do $RCLI ping 2>/dev/null | grep -q PONG && break; sleep 0.3; done
fi
$RCLI ping | grep -q PONG || die "起不来，看 $REDIS_DIR/redis.log"

# 判据行：三项配置必须是我们写的值（28 号 R-01/R-02/R-03 的常驻检查）
for kv in "appendonly yes" "maxmemory-policy noeviction" "protected-mode yes"; do
  k=${kv%% *}; want=${kv#* }
  got=$($RCLI config get "$k" | sed -n 2p)
  [[ "$got" == "$want" ]] || die "[redis-config] $k=$got，应为 $want"
done
echo "[redis-config] appendonly=yes maxmemory-policy=noeviction protected-mode=yes  ✅"
echo "✅ Redis 就绪  redis://:<REDIS_PASS>@127.0.0.1:$REDIS_PORT/0   (db0 broker · db1 ratelimit · db2 semaphore · db3 cache)"
