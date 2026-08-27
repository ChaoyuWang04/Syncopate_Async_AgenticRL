# RDMA 实操学习项目

> **目标**：从零成本起步，建立对 RDMA 的**可验证的**工程理解，最终能读懂并有能力改动 NCCL / NVSHMEM / DeepEP 这一层的通信代码。
>
> **面向**：项目负责人（你）+ 执行 agent（跑在 5090 服务器上）
>
> **核心原则**：**每一步都必须产出一个数字或一条报错**。"我理解了"不是验收条件。

---

## 0 · 这份文档怎么用

### 人 / agent 分工

| 角色 | 负责 |
|---|---|
| **你** | 读 §3 建立心智模型 · 每个阶段结束时**亲自看数据并判断是否通过** · 决定是否进入下一级 |
| **agent** | 按 §4 起的任务卡执行 · 写代码 · 跑测量 · **原样记录失败，不许静默修复** |

### 三条硬规则（写给 agent）

1. **不许跳步。** 每个任务卡有 `前置` 字段，前置未通过就停下并报告。
2. **失败是产出，不是障碍。** 遇到报错先**完整记录**（errno / `ibv_wc.status` / 完整输出），写进坑登记册，再讨论修不修。§4.5 整个阶段就是**故意制造失败**。
3. **不许用"应该"描述结果。** 只写实测值。没测到就写"未测到"。

---

## 1 · 总目标与能力分层

### 最终能回答的问题

学完这个项目，下面每个问题你都应该能用**自己跑出来的数字**回答：

| # | 问题 | 在哪一级解决 |
|---|---|---|
| 1 | 单边 RDMA WRITE 时，对端进程到底做了什么？ | L0 |
| 2 | 内存"注册"具体注册了什么？不注册会怎样？ | L0 |
| 3 | QP 状态机的每一步在协商什么？漏一个字段的症状是什么？ | L0 |
| 4 | 同样传 1GB，TCP / 双边 RDMA / 单边 RDMA 的 CPU 开销差多少？ | L0 → L2 |
| 5 | 零拷贝的收益在什么 message size 以上才显现？ | L2 |
| 6 | GPUDirect RDMA 省掉的那次拷贝，实际值多少延迟？ | L3 |
| 7 | 权重从 trainer 推到 rollout 引擎，走哪条路最快？ | L0 原型 → L3 验证 |

### ★ 一个必须先说清的口径边界

$$\boxed{\textbf{Soft-RoCE 能验证「语义」，不能验证「硬件卸载」}}$$

| 结论 | Soft-RoCE 能证明吗 |
|---|---|
| "单边 WRITE 时接收端**应用进程**零参与" | ✅ 能 —— 进程 CPU 时间可测 |
| "单边 WRITE 时接收端 **CPU** 零参与" | ❌ **不能** —— 软件实现下内核在收包，系统 CPU 不为零 |
| "零拷贝省掉了内存拷贝" | ❌ 不能 —— rxe 内部仍走内核网络栈 |
| "verbs API 的语义和真硬件一致" | ✅ 能 |

⚠️ **这条口径必须写进你所有的实验报告和简历里。** 说"我在 Soft-RoCE 上验证了单边操作接收端 CPU 为零"是**错的**，会被当场问倒。正确说法是"验证了接收端应用进程不参与"。

---

## 2 · 分级路线图

| 级 | 硬件要求 | 成本 | 能学到 | 优先级 |
|---|---|---|---|---|
| **L0** | **1 台 Linux 裸机**（你的 5090 服务器） | **¥0** | verbs 全套语义、QP 状态机、单双边差异、故障模式 | ★★★ **先做这个** |
| **L1** | 同上（用 netns 模拟两台）或两台云 CPU 机 | ¥0 – 几元 | 真实跨机建链、GID/MTU 协商、丢包行为 | ★★ |
| **L2** | 2× 二手 ConnectX-5 + DAC 直连 | ¥1500–3000 一次性 | **真零拷贝**、亚微秒延迟、inline/batching 调优 | ★ |
| **L3** | 多节点 GPU + IB（RunPod/Lambda 按小时） | ¥100–500/次 | GPUDirect RDMA、NCCL 拓扑、NVSHMEM | ☆ 验证用 |

**L0 占整个项目价值的 60–70%，且完全免费。先把 L0 做透。**

### 前置条件检查

L0 开始前必须满足：

- [ ] 一台可以 `sudo modprobe` 的 **Linux 裸机**（⚠️ 容器/RunPod pod 通常不行）
- [ ] 内核 ≥ 4.8（`rdma_rxe` 从 4.8 进主线）
- [ ] 能装包：`rdma-core` `ibverbs-utils` `perftest` `libibverbs-dev`
- [ ] 会写 C（verbs 是 C API；有 Python 绑定但**不要用**，会绕过你该学的东西）

---

## 3 · 心智模型速览（10 分钟，人读）

### 3.1 一句话

> **RDMA = 你把家里储物柜的钥匙和柜号给对方，对方自己去取东西，你可能在睡觉。**

普通 TCP 是寄快递：两边的人都得动。RDMA 单边操作是给钥匙：只有一边动。

### 3.2 六个对象及其关系

```
ibv_context   打开一个设备（网卡）
     │
     ├── ibv_pd          保护域 —— "同一个 PD 里的东西才能互相认"
     │      │
     │      ├── ibv_mr   内存区域 —— pin 住 + 把地址翻译表灌进网卡
     │      │              产出 lkey（本地用）和 rkey（给对端用）
     │      │
     │      └── ibv_qp   队列对 = SQ(发送队列) + RQ(接收队列)
     │
     └── ibv_cq          完成队列 —— 操作做完了往这里丢一条记录
```

**关键理解**：`ibv_mr` 和 `ibv_qp` 必须属于**同一个 PD**，否则网卡拒绝服务。PD 是硬件层面的隔离边界。

### 3.3 两类操作，本质不同

| | 操作 | 对端参与吗 | 对端有 completion 吗 |
|---|---|---|---|
| **双边** | `SEND` / `RECV` | ✅ 必须提前 `post_recv` | ✅ 有 |
| **单边** | `RDMA_WRITE` / `RDMA_READ` | ❌ **完全不知道** | ❌ **没有** |
| 折中 | `RDMA_WRITE_WITH_IMM` | 需要 `post_recv` | ✅ 有（带 32bit 立即数） |

★ **这张表是整个 L0 的核心。** 纯 `RDMA_WRITE` 之后对端 `poll_cq` 会**永远等下去**——这不是 bug，是设计。想通知对端必须用 `WITH_IMM` 或者让对端轮询内存里的 flag。

### 3.4 建链要交换什么（带外，用普通 TCP socket 换）

```c
struct conn_info {
    uint32_t qp_num;      // 对端 QP 编号
    uint32_t psn;         // 起始包序号 —— 我的 sq_psn 必须等于对端的 rq_psn
    uint32_t rkey;        // 远端访问我的 MR 需要的钥匙
    uint64_t addr;        // 我的 MR 起始地址
    union ibv_gid gid;    // RoCE 必须用 GID（IB 才用 LID）
    uint16_t lid;         // IB 用；RoCE 下为 0
};
// ⚠️ 网络传输前必须处理字节序（htonl/htobe64）
```

### 3.5 QP 状态机（最容易卡死的地方）

```
RESET ──modify_qp──> INIT ──modify_qp──> RTR ──modify_qp──> RTS
                      │                   │                  │
              设端口/权限          设对端信息/MTU/PSN      设超时/重试
```

每一步需要的属性掩码**必须精确**，少一位就 `EINVAL`：

| 转换 | 必须的 attr_mask |
|---|---|
| RESET→INIT | `IBV_QP_STATE` `IBV_QP_PKEY_INDEX` `IBV_QP_PORT` `IBV_QP_ACCESS_FLAGS` |
| INIT→RTR | `IBV_QP_STATE` `IBV_QP_AV` `IBV_QP_PATH_MTU` `IBV_QP_DEST_QPN` `IBV_QP_RQ_PSN` `IBV_QP_MAX_DEST_RD_ATOMIC` `IBV_QP_MIN_RNR_TIMER` |
| RTR→RTS | `IBV_QP_STATE` `IBV_QP_TIMEOUT` `IBV_QP_RETRY_CNT` `IBV_QP_RNR_RETRY` `IBV_QP_SQ_PSN` `IBV_QP_MAX_QP_RD_ATOMIC` |

⚠️ **RoCE 专属陷阱**（Soft-RoCE 也算 RoCE）：INIT→RTR 时的 `ah_attr` 必须设
```c
attr.ah_attr.is_global = 1;              // RoCE 没有 LID，必须走 GRH
attr.ah_attr.grh.dgid = remote_gid;
attr.ah_attr.grh.sgid_index = <本地 GID index>;
attr.ah_attr.grh.hop_limit = 1;          // 设 0 会静默丢包
```

---

# 4 · L0 阶段 ★★★ 零成本 · 最高优先级

**总目标**：在一台机器上把 verbs 的全部核心语义跑通、测出来、并**故意打破八次**。

**预计工时**：15–25 小时（分 7 个任务卡）

**总交付物**：一个 `rdma-lab/` 目录 + 一份 `L0-report.md`

---

## L0-A · 环境搭建与基线

**前置**：无
**预计**：1–2 h

### 要做的事

```bash
# 1. 装包
sudo apt update
sudo apt install -y rdma-core ibverbs-utils libibverbs-dev perftest build-essential

# 2. 加载 Soft-RoCE
sudo modprobe rdma_rxe

# 3. 在真实网卡上创建 rxe 设备（把 eth0 换成 `ip a` 里的实际网卡名）
sudo rdma link add rxe0 type rxe netdev eth0

# 4. 验证
rdma link show              # 期望看到 rxe0/1 state ACTIVE physical_state LINK_UP
ibv_devices                 # 期望列出 rxe0
ibv_devinfo -d rxe0         # 期望 state: PORT_ACTIVE

# 5. 查 GID 表（后面建链必需）
ibv_devinfo -d rxe0 -v | grep -A2 GID
# 或直接看 sysfs
for i in 0 1 2 3; do
  echo -n "gid[$i] type="; cat /sys/class/infiniband/rxe0/ports/1/gid_attrs/types/$i 2>/dev/null
  echo -n "  gid="; cat /sys/class/infiniband/rxe0/ports/1/gids/$i 2>/dev/null
done
```

⚠️ **如果 `rdma link add` 挂在 `lo` 上不工作**（已知不稳定），务必挂在真实网卡上，然后用本机 IP 自连。

### 基线测量

```bash
# 终端 1
ib_write_bw -d rxe0 -x <你的RoCEv2 GID index>
# 终端 2
ib_write_bw -d rxe0 -x <同上> <本机IP>

# 同样跑一遍延迟
ib_write_lat -d rxe0 -x <idx>
ib_send_bw   -d rxe0 -x <idx>
ib_read_bw   -d rxe0 -x <idx>
```

### ✅ 验收指标（全部必须满足）

| # | 指标 | 判定 |
|---|---|---|
| A1 | `ibv_devinfo` 输出中 `state` | 必须是 `PORT_ACTIVE (4)` |
| A2 | 找到至少一个 type 为 `RoCE v2` 的 GID index | 记录下 index 号，后面全用它 |
| A3 | `ib_write_bw` 成功完成并输出带宽表 | 无报错退出，记录 65536B 时的 BW (Gb/s) |
| A4 | `ib_write_lat` 输出 p50/p99 延迟 | 记录 2B 和 65536B 两个 size 的 t_avg (µs) |
| A5 | 四个 perftest 工具（write_bw/write_lat/send_bw/read_bw）全部跑通 | 4/4 |

**交付**：`rdma-lab/L0A-baseline.md`，含完整环境信息（内核版本、网卡型号、GID index）+ 上述四张测量表。

> **注意**：Soft-RoCE 的绝对性能很难看（带宽可能只有几 Gb/s，延迟几十 µs），**这是正常的**。这些数字的用途是**做后续对比的基线**，不是用来评价 RDMA 的。

---

## L0-B · 对象模型与生命周期

**前置**：L0-A 通过
**预计**：2–3 h

### 要做的事

写一个 `obj_lifecycle.c`：按顺序创建 context → PD → MR → CQ → QP，每步打印返回的指针/句柄和关键字段，然后**按正确顺序**销毁。

然后写一个 `obj_lifecycle_bad.c`：**故意用错误顺序销毁**（比如先 dealloc PD 再 dereg MR）。

### ✅ 验收指标

| # | 指标 | 判定 |
|---|---|---|
| B1 | 正常版本创建 5 类对象并全部成功销毁 | 所有 `ibv_*` 返回值非 NULL / 返回 0 |
| B2 | 打印出 MR 的 `lkey` 和 `rkey` | 两个值都非零，且**记录它们是否相等** |
| B3 | 注册 1 GB MR，测量 `ibv_reg_mr` 耗时 | 记录毫秒数；与注册 1 MB 对比，画出 size→耗时曲线（至少 5 个点） |
| B4 | 错误销毁顺序的实际症状 | 记录：是返回 `EBUSY`？还是段错误？还是静默成功？ |
| B5 | 用 `/proc/<pid>/status` 记录注册前后的 `VmLck` | 证明内存确实被 pin 住了，记录增量 |

★ **B3 和 B5 是这张卡的价值所在。** 它们把"注册很贵"和"pin 住内存"从概念变成数字。B3 的曲线直接解释了为什么生产代码要**预先建 MR 池而不是每次传输前注册**。

**交付**：`rdma-lab/L0B-objects.md` + 两个 `.c` 文件 + size→注册耗时 的数据表。

---

## L0-C · 最小双边通信（SEND/RECV）

**前置**：L0-B 通过
**预计**：3–4 h

### 要做的事

写 `rc_send.c`：一个程序，既能当 server 也能当 client（`-s` / 目标IP 区分），完成：

1. 用普通 TCP socket 交换 `struct conn_info`
2. QP 走完 RESET→INIT→RTR→RTS
3. server 先 `ibv_post_recv`，client `ibv_post_send`
4. 两端各自 `ibv_poll_cq` 等完成
5. server 校验收到的数据

**数据内容**：用可校验的 pattern（如 `buf[i] = (uint8_t)(i * 31 + 7)`），不要用全零。

### ✅ 验收指标

| # | 指标 | 判定 |
|---|---|---|
| C1 | 传输 1 MB，接收端 `memcmp` 校验 | 必须完全一致，0 字节差异 |
| C2 | 两端 `ibv_wc.status` | 都必须是 `IBV_WC_SUCCESS (0)` |
| C3 | 记录 `ibv_wc.opcode` | 发送端 `IBV_WC_SEND(0)`，接收端 `IBV_WC_RECV(128)` |
| C4 | 循环传 1000 次，统计端到端延迟 | 输出 p50 / p99 / max，单位 µs |
| C5 | 测量**接收端进程**的 CPU 时间 | 用 `getrusage(RUSAGE_SELF)`，记录 `utime+stime` (ms) |
| C6 | 传输 size 扫描：1KB / 64KB / 1MB / 16MB / 256MB | 每个 size 记录带宽和延迟 |

**交付**：`rdma-lab/L0C-send-recv/` + `rc_send.c` + 测量表。

---

## L0-D · 最小单边通信（RDMA WRITE）★★

**前置**：L0-C 通过
**预计**：3–4 h

### 要做的事

写 `rc_write.c`。**关键差异**：

- 接收端（被写方）在建链后**完全不 post 任何 WR，也不 poll_cq**，只是 `sleep` 然后检查内存
- MR 的 access flags 必须是 `IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE`
- 发送端 `wr.opcode = IBV_WR_RDMA_WRITE`，填 `wr.wr.rdma.remote_addr` 和 `wr.wr.rdma.rkey`

然后写第二个版本 `rc_write_imm.c`，用 `IBV_WR_RDMA_WRITE_WITH_IMM`，接收端 `post_recv` 后能收到 completion。

### ✅ 验收指标 ★ 这是 L0 最重要的一张卡

| # | 指标 | 判定 |
|---|---|---|
| D1 | 单边 WRITE 后接收端内存内容正确 | `memcmp` 0 差异 |
| D2 | **接收端进程的 `poll_cq` 调用次数** | 必须 = **0**（代码里根本没这行） |
| D3 | **接收端进程 CPU 时间：单边 vs 双边** | 计算比值 `CPU(单边) / CPU(L0-C 双边)`，**期望 < 0.1** |
| D4 | 纯 `RDMA_WRITE` 时接收端故意 `poll_cq(timeout=5s)` | 记录：**必须超时且返回 0 条 completion** |
| D5 | `WRITE_WITH_IMM` 版本接收端能拿到 completion | `wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM(130)`，且 `wc.imm_data` 等于发送端设的值 |
| D6 | 系统级 CPU 对比（`/proc/stat` 或 `mpstat`） | ⚠️ **记录并说明为什么系统 CPU 不为零**——这是 Soft-RoCE 的软件实现在收包 |

$$\boxed{\textbf{D3 和 D6 放在一起，就是本项目最有价值的一个结论}}$$

D3 证明了 API 语义（应用零参与），D6 诚实地标出了 Soft-RoCE 的能力边界（硬件卸载没验证到）。**这个"能证明什么 / 不能证明什么"的自觉，是面试里最难伪造的信号。**

**交付**：`rdma-lab/L0D-write/` + 两个 `.c` + 一张四列对比表（双边/单边 × 发送端CPU/接收端CPU）。

---

## L0-E · 故障注入实验 ★★★

**前置**：L0-D 通过
**预计**：4–5 h
**这是全项目最高性价比的一张卡。**

### 要做的事

从 L0-D 的正确版本出发，**每次只破坏一件事**，记录**实际观察到的**症状。

⚠️ **单变量纪律**：每个实验从干净的正确版本改一处，跑完立刻改回来。不要叠加。

### 注入清单（预测 vs 实测）

下表的"预测"列是**待证伪的假设**，不是答案。**实测与预测不符的地方，才是这张卡真正的产出。**

| ID | 注入什么 | 预测症状 | 实测（agent 填） |
|---|---|---|---|
| **RD01** | MR 只设 `REMOTE_WRITE`，不设 `LOCAL_WRITE` | `ibv_reg_mr` 返回 NULL，`errno=EINVAL` | |
| **RD02** | 两端 `path_mtu` 设不一样 | 建链成功但传输 hang / completion error | |
| **RD03** | 发送端 `sq_psn` ≠ 接收端 `rq_psn` | `IBV_WC_RETRY_EXC_ERR (12)` | |
| **RD04** | RoCE 下 `ah_attr.is_global = 0` | INIT→RTR 失败或包发不出去 | |
| **RD05** | `grh.hop_limit = 0` | 静默丢包，最终超时 | |
| **RD06** | 用 `lkey` 冒充 `rkey` 发过去 | `IBV_WC_REM_ACCESS_ERR (10)` | |
| **RD07** | 纯 `RDMA_WRITE`，接收端 `poll_cq` | 永远收不到（无 completion） | |
| **RD08** | 发 `SEND` 但接收端没 `post_recv` | `IBV_WC_RNR_RETRY_EXC_ERR (13)` | |
| **RD09** | QP 从 INIT 直接跳到 RTS | `ibv_modify_qp` 返回 `EINVAL` | |
| **RD10** | 传一个**没注册过**的地址 | `IBV_WC_LOC_PROT_ERR (4)` | |
| **RD11** | MR 和 QP 分属两个不同 PD | 建链或传输失败 | |
| **RD12** | `remote_addr` 越过 MR 边界 | `IBV_WC_REM_ACCESS_ERR` | |

### ✅ 验收指标

| # | 指标 | 判定 |
|---|---|---|
| E1 | 12 项全部复现并记录 | 12/12，每项有完整输出（errno 或 wc.status 数值 + 字符串） |
| E2 | **预测错误的项数** | **≥ 1 项即为优秀**——说明你在学真东西而不是背答案 |
| E3 | 区分"立即报错" vs "静默 hang" | 每项标注属于哪类，并统计比例 |
| E4 | 每项给出"生产代码里怎么防" | 一句话，例如 RD02 → "建链后校验双方协商的 MTU 是否一致" |

★★ **E3 的统计结果是核心洞察**：数一数有多少 bug 是**静默的**。RDMA 编程最痛的地方就是**大量错误不报错，只是不动**——这和你在 verl 那边遇到的"坑 57 静默不一致"是同一类问题。

**交付**：`rdma-lab/L0E-faults/` + 12 个 patch/脚本 + `RD-registry.md`（坑登记册，编号 RD01– 起）。

---

## L0-F · 三通道对比实验

**前置**：L0-E 通过
**预计**：3–4 h

### 要做的事

实现三个功能等价的"把 N 字节从 A 传到 B"的通道，接口统一：

```c
typedef struct {
    const char *name;
    int  (*init)(config_t *cfg);
    int  (*transfer)(void *buf, size_t len);   // 阻塞直到对端拿到数据
    void (*teardown)(void);
} channel_t;

// 三个实现
channel_t ch_tcp;          // 普通 socket + write/read
channel_t ch_rdma_send;    // SEND/RECV
channel_t ch_rdma_write;   // RDMA_WRITE_WITH_IMM
```

对每个通道 × 每个 size 测：延迟(p50/p99)、带宽、**发送端进程 CPU**、**接收端进程 CPU**。

### ✅ 验收指标

| # | 指标 | 判定 |
|---|---|---|
| F1 | 三通道 × 6 个 size（4KB→1GB）全部跑通 | 18/18 组合，数据完整 |
| F2 | 每组重复 ≥ 20 次，报 p50/p99 | 不许只报单次 |
| F3 | 输出一张 4 指标 × 18 组合的完整表格 | CSV + markdown 双份 |
| F4 | **接收端 CPU：ch_rdma_write / ch_tcp 的比值** | 记录该比值随 size 的变化曲线 |
| F5 | 找出三通道的**交叉点** | 在哪个 size 以下 TCP 反而更快？给出数字 |
| F6 | 数据一致性校验 | 每次传输后 `memcmp`，18/18 全部 0 差异 |

★ **F5 是最容易被忽略但最有价值的一问。** RDMA 有固定的 WQE 提交开销，小消息下未必赢。**知道"什么时候不该用 RDMA"比知道"RDMA 快"重要得多。**

**交付**：`rdma-lab/L0F-compare/` + `results.csv` + `L0F-report.md`（含至少 2 张图）

---

## L0-G · 权重同步原型 ⭐ 对接你的研究线

**前置**：L0-F 通过
**预计**：4–6 h

### 背景

这是本项目**唯一直接产出简历素材**的一卡。Syncopate 的核心工程问题是：**训练引擎把新权重推给推理引擎，有多快、期间训练能不能继续**。

### 要做的事

模拟一个"trainer → rollout engine"的权重推送通道：

**场景设定**
- 权重总量可配置，默认 **256 MB**（Soft-RoCE 下够快，方便迭代；跑通后再试 2 GB）
- 分片：模拟按 layer 切成 N 个 chunk（默认 32 个）
- trainer 侧在推送**期间要持续做假计算**（一个 busy loop 或矩阵乘），模拟训练继续

**三种推送策略**
1. **阻塞式**：`transfer()` 完了才继续算
2. **双边流水**：SEND/RECV，分 chunk，边算边发
3. **单边流水**：预先注册好整块 MR，`RDMA_WRITE` 各 chunk，最后一个用 `WITH_IMM` 通知

**测量**
- 端到端权重同步延迟
- **trainer 在同步期间损失了多少计算吞吐**（对比无同步时的假计算 ops/s）
- rollout 侧从"开始"到"可用"的时间

### ✅ 验收指标

| # | 指标 | 判定 |
|---|---|---|
| G1 | 三种策略传 256 MB，接收端权重校验 | 3/3 校验通过（逐 chunk checksum） |
| G2 | 记录三种策略的端到端同步延迟 | 三个数字 + 相对阻塞式的加速比 |
| G3 | **计算吞吐损失率** = 1 − (同步期间 ops/s) / (基线 ops/s) | 三种策略各给一个百分比 |
| G4 | 单边流水的吞吐损失率 | **期望显著低于阻塞式**；给出实测比值 |
| G5 | chunk 数量扫描（8/16/32/64/128） | 找出同步延迟最低的 chunk 数，说明为什么存在最优值 |
| G6 | MR 注册策略对比 | "每次注册" vs "预注册复用" 的端到端延迟差，用 L0-B 的注册耗时曲线解释 |

★★ **G5 和 G6 是能讲成故事的两个点**：
- G5 存在最优值，是因为 chunk 太少则重叠不足、太多则 WQE 提交开销主导——**这是一个可以画出 U 型曲线的真实权衡**
- G6 直接把 L0-B 的微观测量（注册很贵）连到了宏观决策（必须预注册），**是一条完整的"从测量到设计"的链条**

**交付**：`rdma-lab/L0G-weightsync/` + `L0G-report.md` + U 型曲线图

### ⚠️ 结论口径

写报告时必须标注：这是 **Soft-RoCE 上的相对比较**，绝对数字不代表真实硬件。有效结论是**策略之间的相对优劣和趋势**，不是绝对性能。

---

## L0 阶段总验收

全部满足才算 L0 通过：

- [ ] A1–A5、B1–B5、C1–C6、D1–D6、E1–E4、F1–F6、G1–G6 **全部有实测数据**
- [ ] `RD-registry.md` 至少 12 条，每条有实测症状（不是预测）
- [ ] 至少 **1 处预测被实测推翻**，并写清楚为什么
- [ ] 能不看文档口述 QP 四状态各协商什么
- [ ] `L0-report.md` 汇总，含"能证明什么 / 不能证明什么"的口径声明

---

# 5 · L1 阶段 · 跨机（仍然零成本）

**前置**：L0 全部通过
**预计**：4–6 h
**成本**：¥0（用 network namespace）或几元（两台最小云主机）

## 要做的事

### 方案 A：单机双 netns（推荐，零成本）

```bash
# ⚠️ 必须在没有任何 rdma 设备时执行
sudo rdma system set netns exclusive

sudo ip netns add ns1
sudo ip netns add ns2
sudo ip link add veth1 type veth peer name veth2
sudo ip link set veth1 netns ns1
sudo ip link set veth2 netns ns2

sudo ip netns exec ns1 ip addr add 10.10.0.1/24 dev veth1
sudo ip netns exec ns1 ip link set veth1 up
sudo ip netns exec ns1 ip link set lo up
sudo ip netns exec ns2 ip addr add 10.10.0.2/24 dev veth2
sudo ip netns exec ns2 ip link set veth2 up
sudo ip netns exec ns2 ip link set lo up

sudo ip netns exec ns1 rdma link add rxe1 type rxe netdev veth1
sudo ip netns exec ns2 rdma link add rxe2 type rxe netdev veth2

# 验证
sudo ip netns exec ns1 rdma link show
```

⚠️ 这一步在不同内核版本行为差异较大。**如果失败，完整记录错误并转方案 B**，不要硬调。

### 方案 B：两台最小规格云 CPU 主机

同 VPC，各自 `modprobe rdma_rxe`，用内网 IP 互连。成本几毛钱/小时。

### 新增实验

1. **GID 选择实验**：故意选错 GID index（选 RoCE v1 或 IPv6 的），记录症状
2. **MTU 协商**：两端设不同 MTU（复现 RD02 在真跨机下的表现是否不同）
3. **丢包行为**：用 `tc netem` 注入 1% / 5% 丢包，观察 RC 类型 QP 的重传
4. **连接断开**：中途 kill 一端，另一端的 completion 状态是什么

## ✅ 验收指标

| # | 指标 | 判定 |
|---|---|---|
| L1-1 | 两个 rxe 设备跨 netns / 跨机建链成功 | `ib_write_bw` 跑通 |
| L1-2 | L0-F 的三通道对比在跨机环境重跑 | 18/18，**与单机结果对比，标出差异 > 20% 的项** |
| L1-3 | GID 选错的症状 | 记录具体错误（建链失败？还是通了但慢？） |
| L1-4 | 丢包 1% / 5% 下的带宽衰减 | 两个百分比数字 |
| L1-5 | 单端 kill 后另一端的 `wc.status` | 记录具体错误码 |

**交付**：`rdma-lab/L1-crossnode/` + `L1-report.md`

---

# 6 · L2 阶段 · 真硬件（¥1500–3000 一次性）

**前置**：L0 + L1 通过，且你确认要继续投入
**预计**：10–15 h（不含到货等待）

## 采购清单

| 项 | 建议 | 备注 |
|---|---|---|
| 网卡 ×2 | **ConnectX-5**（MCX515A/MCX516A 系列，100GbE） | 驱动支持好 |
| 备选 | ConnectX-4 Lx（25GbE） | 更便宜，够学 |
| ⚠️ 避免 | **ConnectX-3 及以下** | `mlx4` 在新版驱动栈已弃用 |
| 线 ×1 | DAC 直连线（对应速率） | **不需要交换机**，两卡背靠背 |
| 主机 ×2 | 需要 PCIe x8 以上插槽 | ⚠️ Mac 不行，需要第二台 x86 |

★ **省事技巧**：如果买到 VPI 卡，**直接切 InfiniBand 模式**，在一端跑 `opensm` 当子网管理器。这样可以完全绕过 RoCE 的 PFC/ECN 无损网络配置——两节点背靠背，IB 模式比 RoCE 好调得多。

## 新增实验

1. **重跑全部 L0 测量**，得到真硬件基线
2. **真零拷贝验证**：这次可以测**系统级 CPU**，验证 L0-D6 那条"不能证明"的项
3. **inline send**：小消息用 `IBV_SEND_INLINE`，测延迟改善
4. **doorbell batching**：一次 `post_send` 挂多个 WR 链表 vs 逐个提交
5. **CQ moderation / 忙轮询 vs 事件通知**：`ibv_get_cq_event` vs busy poll 的延迟/CPU 权衡
6. **多 QP 并行**：1/2/4/8 个 QP 的聚合带宽

## ✅ 验收指标

| # | 指标 | 目标值（100GbE 参考） |
|---|---|---|
| L2-1 | `ib_write_bw` 大消息带宽 | **≥ 线速的 90%**（100G 卡约 ≥ 88 Gb/s） |
| L2-2 | `ib_write_lat` 2 字节延迟 | **< 3 µs**（典型 1–2 µs） |
| L2-3 | ★ **单边 WRITE 时接收端系统级 CPU** | **接近 0%** ← 这条 Soft-RoCE 上做不到，是 L2 的核心价值 |
| L2-4 | inline send 对小消息（≤256B）延迟的改善 | 给出百分比 |
| L2-5 | batching 对小消息吞吐（Mops/s）的改善 | 给出倍数 |
| L2-6 | 忙轮询 vs 事件通知的延迟差 / CPU 差 | 两个数字，说明各自适用场景 |
| L2-7 | L0-F 的交叉点在真硬件上是否移动 | 与 L0 结果对比，给出新的交叉点 size |

★★ **L2-3 是把 L0 那条"不能证明"的项补上的时刻。** 到这里你才真正验证了 RDMA 的硬件卸载。这个"先诚实标出边界、再补齐"的过程本身就是一个好的面试叙事。

**交付**：`rdma-lab/L2-hardware/` + `L2-report.md` + 与 L0/L1 的三方对比表

---

# 7 · L3 阶段 · 集群与 GPU（按小时租）

**前置**：L2 通过（或跳过 L2 直接来，但会缺少调优直觉）
**预计**：单次 6–10 h，建议分 2–3 次
**成本**：每次 ¥100–500

## 平台选择

| 平台 | 是否标准 verbs | 备注 |
|---|---|---|
| **RunPod Instant Clusters** | ✅ IB | `ibstat \| grep "State: Active"` 确认 |
| **Lambda / Nebius / CoreWeave** | ✅ IB | 多节点 IB 是强项 |
| **Azure HB/ND 系列** | ✅ 真 InfiniBand | 无 GPU 也可以，便宜 |
| ⚠️ **AWS EFA** | ❌ | 用 **libfabric 不是 ibverbs**，SRD 协议，学标准 verbs 别选 |

## 实验清单

1. **GPUDirect RDMA 配置**：`dma-buf` 路径（推荐）或 `nvidia-peermem`（传统）
2. **有无 GPUDirect 的对比**：显存→显存跨节点，走/不走主机内存中转
3. **`nccl-tests` 全套**：`all_reduce_perf` / `all_gather_perf` / `reduce_scatter_perf`
4. **NCCL 环境变量影响**：`NCCL_NET_GDR_LEVEL` / `NCCL_IB_HCA` / `NCCL_ALGO` 各自的效果
5. **读 NCCL 源码对照**：`src/transport/net_ib.cc` —— 你现在应该能逐行看懂
6. **L0-G 权重同步原型移植到真集群**

## ✅ 验收指标

| # | 指标 | 判定 |
|---|---|---|
| L3-1 | 确认 GPUDirect 生效 | `NCCL_DEBUG=INFO` 日志中出现 `[GDRDMA]` 或等价标识 |
| L3-2 | 显存→显存跨节点带宽：开/关 GDR | 两个数字 + 提升百分比 |
| L3-3 | 首次 pin 显存进 BAR 的耗时 | 实测毫秒数，与 §3 说的"可达毫秒级"对照 |
| L3-4 | `all_reduce_perf` 在 2/4/8 节点的 busbw | 画出扩展性曲线 |
| L3-5 | 关掉 GDR（`NCCL_NET_GDR_LEVEL=0`）的 all-reduce 性能损失 | 百分比 |
| L3-6 | 能指出 `net_ib.cc` 里对应 L0 中每个 verbs 调用的位置 | 列出至少 6 处映射（`ibv_reg_mr` 在哪、QP 状态机在哪…） |
| L3-7 | L0-G 三策略在真集群重跑 | 结论是否与 Soft-RoCE 一致？**不一致的地方是最有价值的发现** |

★ **L3-6 是这一级的智力核心。** 从"我写过 300 行 verbs"到"我能在 NCCL 里找到对应代码"，是从玩具到生产的跨越。

**交付**：`rdma-lab/L3-cluster/` + `L3-report.md` + NCCL 源码映射表

---

# 8 · 坑登记册模板

沿用你既有的编号体系，RDMA 线用 **RD** 前缀，与 agentic-rl(1–65) / harness(H) / Claude Code(CC) / OpenClaw(OC) **分开维护**。

```markdown
## RD-NN · <一句话标题>

**级别**：L0 / L1 / L2 / L3
**类型**：立即报错 / 静默 hang / 静默错误结果 / 性能悄悄退化
**触发条件**：
**实测症状**：（完整贴 errno / wc.status 数值+字符串 / 日志）
**预测是否正确**：✅ / ❌（❌ 时说明差在哪）
**根因**：
**生产代码怎么防**：
**关联**：（与其他登记册的坑有无同构关系）
```

⚠️ **"类型"字段最重要**。RDMA 的坑里静默类占比很高，统计这个比例本身就是一个结论。

---

# 9 · 学习资料

## 必读（按顺序）

| # | 资料 | 用途 | 阶段 |
|---|---|---|---|
| 1 | **RDMAmojo**（Dotan Barak 博客）`rdmamojo.com` | **verbs 的事实标准教程**，每个 API 逐个讲透 | L0 全程 |
| 2 | **`rdma-core` 仓库** `github.com/linux-rdma/rdma-core` | `libibverbs/examples/rc_pingpong.c` 是最好的起点范例 | L0-C |
| 3 | **`perftest` 源码** `github.com/linux-rdma/perftest` | `ib_write_bw` 本身就是参考实现，看它怎么做 batching 和测量 | L0-F, L2 |
| 4 | **RDMA Aware Networks Programming User Manual**（NVIDIA/Mellanox） | 官方手册，QP 状态机的属性表在这里 | L0-B/C |
| 5 | `man ibv_*` | 每个 API 的权威签名和错误码 | 随时 |

## 论文（理解"为什么"）

| 论文 | 核心贡献 | 为什么读 |
|---|---|---|
| **Design Guidelines for High Performance RDMA Systems**（Kalia et al., USENIX ATC'16） | RDMA 系统设计的经典准则 | **最该读的一篇**；解释 inline / doorbell batching / 何时该用单边 |
| **Using RDMA Efficiently for Key-Value Services**（HERD, SIGCOMM'14） | 单边不总是更好 | 打破"单边一定快"的直觉 |
| **FaSST**（OSDI'16） | 用双边 datagram 反而更快的场景 | 反直觉，值得读 |
| **Mooncake: KVCache-centric Disaggregated Architecture**（FAST'25） | KV cache 跨节点搬运 | ⭐ 直接对应你关心的 PD 分离 |

## 源码（L3 阶段对照读）

| 项目 | 关键文件 | 看什么 |
|---|---|---|
| **NCCL** | `src/transport/net_ib.cc` | 生产级的 verbs 封装 |
| **NVSHMEM** | 文档 + IBGDA 部分 | GPU 发起通信怎么做的 |
| **DeepEP** | 通信路径 | MoE dispatch 的细粒度重叠 |
| **UCX** | `src/uct/ib/` | 更通用的抽象层 |

## 工具速查

```bash
ibv_devices / ibv_devinfo -v      # 设备与端口信息
rdma link show / rdma res show    # 链路与资源
ib_write_bw / ib_write_lat        # 单边性能
ib_send_bw  / ib_send_lat         # 双边性能
ib_read_bw  / ib_read_lat         # 单边读
perfquery                          # 端口计数器（错误/重传统计）
ibstat / ibstatus                  # 端口状态
```

---

# 10 · 给 agent 的执行规则

```
1. 严格按 L0-A → L0-G 顺序执行。每张卡的「前置」未全部通过，停下并报告。

2. 每张任务卡结束时，必须产出：
   a) 代码文件（放在指定目录）
   b) 一份 markdown 报告，含**实测数值表格**
   c) 若有失败，写进 RD-registry.md
   不满足三项之一，该卡不算完成。

3. 禁止行为：
   - 禁止跳过验收指标里的任何一项。做不到就写「未完成 + 原因」。
   - 禁止用「应该」「预计」「大约」描述实测结果。
   - 禁止静默修复报错。所有报错先完整记录（errno 数值 + strerror + 完整 stderr），
     再询问是否修复。
   - 禁止在 L0-E 里一次改多个变量。

4. 测量纪律：
   - 每个数字至少重复 20 次，报 p50/p99，不许报单次。
   - 每次测量前记录：内核版本、CPU 型号、当前负载（uptime）。
   - 性能对比必须交替进行（A/B/A/B…），不能先跑完 20 次 A 再跑 B。

5. 代码约定：
   - C 语言，`gcc -O2 -Wall -Wextra`，链接 `-libverbs -lpthread`
   - 所有 verbs 调用必须检查返回值，失败时打印 `errno` 和 `strerror(errno)`
   - 所有 `poll_cq` 必须检查 `wc.status`，非 SUCCESS 时打印
     `ibv_wc_status_str(wc.status)` 和数值
   - 单文件，不要拆成一堆小文件；每个任务卡一个 `.c`

6. 报告先输出到 chat 让人过目，确认后再写文件。
```

---

# 附录 A · 目录结构

```
rdma-lab/
├── README.md                    # 进度看板：每张卡的通过/未通过状态
├── RD-registry.md               # 坑登记册（跨阶段累积）
├── env.md                       # 环境快照：内核/网卡/驱动/GID index
├── L0A-baseline/
│   ├── setup.sh
│   └── L0A-baseline.md
├── L0B-objects/
│   ├── obj_lifecycle.c
│   ├── obj_lifecycle_bad.c
│   └── L0B-objects.md
├── L0C-send-recv/
│   ├── rc_send.c
│   └── L0C-report.md
├── L0D-write/
│   ├── rc_write.c
│   ├── rc_write_imm.c
│   └── L0D-report.md
├── L0E-faults/
│   ├── faults/RD01.patch … RD12.patch
│   ├── run_all_faults.sh
│   └── L0E-report.md
├── L0F-compare/
│   ├── channels.c
│   ├── bench.c
│   ├── results.csv
│   └── L0F-report.md
├── L0G-weightsync/
│   ├── weight_sync.c
│   ├── results.csv
│   └── L0G-report.md
├── L1-crossnode/
├── L2-hardware/
└── L3-cluster/
```

---

# 附录 B · 环境检查脚本

L0-A 开始前先跑这个。**任何一项 FAIL 就不要往下走。**

```bash
#!/usr/bin/env bash
# rdma-lab/check_env.sh
set -u
pass=0; fail=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "✅ $1"; pass=$((pass+1));
        else echo "❌ $1"; fail=$((fail+1)); fi }

echo "=== 基础环境 ==="
echo "kernel : $(uname -r)"
echo "distro : $(. /etc/os-release && echo $PRETTY_NAME)"

chk "内核 >= 4.8"            '[ "$(uname -r | cut -d. -f1)" -ge 5 ] || \
                              { [ "$(uname -r|cut -d. -f1)" -eq 4 ] && \
                                [ "$(uname -r|cut -d. -f2)" -ge 8 ]; }'
chk "有 sudo 权限"           'sudo -n true'
chk "非容器环境"             '[ ! -f /.dockerenv ]'
chk "rdma_rxe 模块可用"      'modinfo rdma_rxe'
chk "ibv_devices 存在"       'command -v ibv_devices'
chk "perftest 已装"          'command -v ib_write_bw'
chk "libibverbs 开发头文件"  '[ -f /usr/include/infiniband/verbs.h ]'
chk "gcc 可用"               'command -v gcc'

echo
echo "=== 网卡 ==="
ip -o link show | awk -F': ' '$2!="lo"{print "  "$2}'

echo
echo "结果: PASS=$pass FAIL=$fail"
[ "$fail" -eq 0 ] || { echo "⚠️  有 FAIL 项，先解决再开始 L0-A"; exit 1; }
```

⚠️ 如果"非容器环境"这一项 FAIL，说明你在容器里——**换到 5090 服务器的裸机上跑**，容器里 `modprobe` 通常没权限。

---

# 附录 C · 与现有研究线的接口

这个项目不是孤立的。做完后应该回填到这些地方：

| L0-G 的产出 | 接到哪 |
|---|---|
| 三种权重推送策略的吞吐损失率 | **Syncopate**：异步架构里 trainer→rollout 的权重同步开销 |
| chunk 数的 U 型曲线 | 通信-计算重叠的通用权衡，可迁移到 MoE 场景 |
| MR 预注册 vs 每次注册 | 解释了 DeepEP 为什么要预分配 buffer |

| L3 的产出 | 接到哪 |
|---|---|
| NCCL 源码映射表 | 面试时"读过 NCCL"的具体证据 |
| GDR 开/关的性能差 | 你 MoE-DeepEP 那条线的底层解释 |
| PD 分离的 KV cache 搬运 | 简历项目二（Agent + Harness runtime）的可选延伸 |

---

*文档版本：v1.0 · 2026-08-27*
*后续修订应记录在 README.md 的变更日志里。*
