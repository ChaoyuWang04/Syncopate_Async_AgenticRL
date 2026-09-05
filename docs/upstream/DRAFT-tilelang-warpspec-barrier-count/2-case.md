# tilelang · warp 特化流水的释放 barrier 计数与生产者分区不同步 ⇒ 数据竞态

```
状态     DRAFT —— 待上游考据（负责该包的 Codex：复现最小化 + 上游代码定位 + 按 tilelang issue 模板成稿）
目标仓库  tile-ai/tilelang（0.1.13 复现）
性质     正确性 bug（静默、尺寸相关才显形）
原始发现  ../../archive/infra_exp/legacy-4x5090/E30-tilelang-nvfp4-gemm.md §4（2026-08-27）
复现载体  syncopate/train/tilelang_mxfp8.py（本仓库）——把 SFA_sh/SFB_sh 两个 T.copy 删掉即触发
```

## 现象（全部实测）

带 TMA 的 `T.Pipelined` kernel（256 线程声明）被 warp 特化为 512 线程（生产 256/消费 256），
消费者用 `mbarrier[2/3]`（init 计数 **256**）向生产者释放 stage。
当循环体里的**非 TMA smem 拷贝全部移除**后，TileLang 把生产者分区缩到 **128** 线程
（`if (threadIdx.x < 128)`，总线程 384），**但 mbarrier[2/3] 的 init 仍是 256**，
而消费者 384 线程全部 arrive ⇒ 相位错乱 ⇒ 生产者提前覆写在读 stage。

## 关键证据

- 生成源对照：有 smem 拷贝时 `if (tx<256)` + `init(256)` 自洽；删掉后 `if (tx<128)` + `init(256)` 失配（`get_kernel_source()` 可直接看）；
- 错误形状 = 竞态签名：4096³ 时 0.12% 元素错、107/1024 block 无空间规律散布；**2048³ 及以下对拍全过**（侥幸）——报 issue 时这点要写：小尺寸测试防不住它；
- 规避法已验证：保留任意一条消费者可见的 smem 拷贝让分区回到 256/256 即消失。

## 给上游同事的任务

① 最小复现（建议从本仓库 kernel 裁）：一个 T.Pipelined + TMA copy + call_extern 的玩具 kernel，删/留一条 smem 拷贝两态对比 `get_kernel_source()` 的 init 计数；
② 上游定位：warp 特化 pass 里分区宽度与 barrier arrive-count 的推导处（怀疑分区宽度按"软件拷贝线程需求"推、计数按"声明线程数"推，两处各算各的）；
③ 顺带一提（可并入同 issue 或另开 docs issue）：warp 特化下 `T.call_extern` 设备函数拿到的 `threadIdx.x` 是 256..511，逻辑线程号需 `&255`——文档没写，首撞必 nan。

## 补充证据（08-27 晚）：缺陷代价已定价

裸 CUDA 复刻同一 kernel（TMA+warp 特化+64×64 大块，`scripts/infra/mxf8_gemm_limit_tma.cu`）
实测 **627 TFLOPS**，tilelang 版被寄存器机制限制在 543 ⇒ 该缺陷/限制的性能代价 ≈15%，
且 627 即消费卡寄存器包络顶点（E30 §10）——PR/issue 里可作为"修复收益上界"引用。
