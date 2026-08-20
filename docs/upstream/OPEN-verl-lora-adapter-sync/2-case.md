# Case · verl disaggregated 下 LoRA adapter 从不同步（E22）

```
状态      OPEN —— 已提交，等 CI / review（2026-08-20）
issue/PR  #7495 · #7496（标题 [rollout, fully_async] fix: ... 已过 verl 的 check_pr_title）
分支      ChaoyuWang04/verl → fix/disaggregated-lora-adapter-sync @ bba0c45（基于 main 9326156，DCO 已签）
待办      ⬜ 飞书群申请 CI（话术见 3-submission.md §③，PR=7496 / ISSUE=7495）
          ⬜ PR 正文重贴一次：补回 "### What does this PR do?" 标题 + 补全被截断的半句
             + 去掉硬折行（3-submission.md 已是重排版）
验证      测试 修前 3 failed（红在行为断言）→ 修后 7 passed · pre-commit 14 钩子全过
          源码树实测 3 次同步全 adapter（252 MiB）、基座 0 次、kl 贴地板（6.4e-05 / 2.9e-04）
```

⚠️ **定位要点**：包①（`fsdp_size=1`）被维护者以 **"rare case"** 驳回。本条相反 ——
`lora.merge=False` 是**默认值**、任何 disaggregated recipe 都用非 `naive` 后端 ⇒
**触发它不需要任何非常规配置**，这句话已放在 issue/PR 最前面。

## 修法为什么是这个形状（提交时被问到就答这个）

```
零 wire 改动     不新增序列化任何东西 ⇒ named_tensors 的后端（nccl/nixl/mooncake/kimi）全覆盖
自描述判别       adapter 推送 100% 是 lora_ 张量、基座推送 0 个 ⇒ peek 第一个名字即可，两侧零协调
照抄上游语义     base_sync_done 初始化 = "dummy" not in load_format（与 colocate 同款）
                 ⇒ 真实权重加载时第一次同步就只推 adapter，8.4 GB 基座一次都不用推
三类不受影响     全参 / merge=True / dummy 首推基座 —— 载荷里没有 lora_ ⇒ 走今天的原路
delta_sharded    peek 守在 wire_format=="named_tensors" 上 ⇒ delta 引擎自管的路不碰
```

★ **上游自己留了面包屑**：`engine_workers.py` 里那句注释
*"base_sync_done is unused in merge-only mode but **kept for Phase 2 adapter path**"*
—— 他们规划过这个第二阶段，只是没写。⇒ PR 的定位是**完成你们自己规划的 Phase 2**。

发现来源 [`../../infra_exp/E22-lora-never-synced.md`](../../infra_exp/E22-lora-never-synced.md) ·
提交件 [`3-submission.md`](3-submission.md) · 补丁与测试见本目录

---

---

> 🆕 **2026-08-19 · 提交前调查完成，四条**：
> ① **断点在 main 上活着，两处都验过**：`engine_workers.py:756` 仍是 `get_per_tensor_param()`
>    不传参 + `peft_config` 丢进 `_`；`CheckpointEngineWorker.update_weights` 签名仍没有
>    `peft_config`（该函数刚被 delta-sync 系列改过、加了 `wire_format` —— 唯独没加这个）。
> ② **空白确认**：`base_sync_done`(26) / `collect_lora_params`(9) / `TensorLoRARequest`(13)
>    的 issue+PR 逐条看过 —— **没有人报过这条路、没有修它的 PR**。近期的 adapter-sync 修复
>    （[#7287](https://github.com/verl-project/verl/pull/7287) / #7413 / #3907）**全在 colocate 侧**。
> ③ ★ 上游留了面包屑：`engine_workers.py:614` 注释 *"base_sync_done is unused in merge-only
>    mode but **kept for Phase 2 adapter path**"* —— 他们规划过这个第二阶段，没写。
>    ⇒ PR 定位：**完成你们自己规划的 Phase 2**。
> ④ [#7290](https://github.com/verl-project/verl/issues/7290) 确认 `peft_config` 的 base 契约
>    是 **dict**（vLLM 原样塞进 `TensorLoRARequest`）—— 背书我们 rollout 侧重建 dict 的形态。
>    另 #7436 显示 LoRA 区域有指定 codeowner（HollowMan6），PR 知道 tag 谁。

> 🆕🆕 **2026-08-18 检索到的情报 —— 论点要按它改写：**
> [verl#2048 「[Async VLLM] LoRA support?」](https://github.com/verl-project/verl/issues/2048) 明写
> 「LoRA 只支持同步的 `vLLMRollout`，async worker 会**抛出错误**」，**issue 关成 `not planned`**。
> ⇒ 因此本文的论点**不是**「你们有个 bug」，而是：
> **「一个已知不支持的组合，失败方式从『明确报错』退化成了『静默推一份冻结基座』。」**
> 静默损失的是使用者两个月的实验；报错只损失一次启动。
> 同族：[verl#3654](https://github.com/verl-project/verl/issues/3654) · [verl#3882](https://github.com/volcengine/verl/issues/3882)

## 0 · 一句话

**在 disaggregated 训练（`fully_async` / `one_step_off`，即 `checkpoint_engine.backend != "naive"`）
下使用 LoRA 且 `model.lora.merge=False`（默认值）时，每一次权重同步推给 rollout 的都是
**未经修改的冻结基座**，LoRA adapter 一个字节都不会被推过去
⇒ 生成数据的策略**永远停在起点**，训练学到的东西从不参与 rollout，
而训练照常跑完、loss/reward/grad_norm/ESS 全部正常、没有任何告警。**

---

## 1 · 触发条件（RL + LoRA 的默认组合就会撞上）

```yaml
actor_rollout_ref.model.lora_rank: 32           # 用 LoRA
actor_rollout_ref.model.lora.merge: false       # ← **默认值**
actor_rollout_ref.rollout.checkpoint_engine.backend: nccl   # ← 非 naive ⇒ disaggregated
# 即：trainer 与 rollout 分卡的异步训练（fully_async / one_step_off）
```

⇒ **"异步 RL + LoRA"这个组合本身就是触发条件**，而这正是 LoRA 在 RL 里最常见的用法：
模型单卡放得下、只训 adapter、trainer 与 rollout 分卡以提高吞吐。

⚠️ `colocate`（`backend: naive`）**不受影响** —— 那条路径调了两次
`get_per_tensor_param()`，先推基座、再推 adapter。**同一个配置，换个模式就是对的。**

---

## 2 · 代码路径（verl 0.8.0）

### 2.1 disaggregated 那条路：只调一次，且用的是默认参数

```python
# verl/workers/engine_workers.py:695-700  ActorRolloutRefWorker.update_weights
effective_mode = mode if mode != "auto" else self.config.rollout.checkpoint_engine.backend

# 0. send_weights only for async training with disaggregated trainer and rollout
if effective_mode != "naive":
    per_tensor_param, _ = self.actor.engine.get_per_tensor_param()   # ← **不传任何参数**
    await self.checkpoint_engine.send_weights(per_tensor_param, global_steps=global_steps)
    return                                                            # ← **直接返回，没有第二次**
```

⚠️ 注意第二个返回值 `peft_config` 在这里被**丢弃**（`, _ =`）——
而它正是"这是个 LoRA 模型"的信号。

### 2.2 对照：naive 那条路调了两次，是对的

```python
# verl/workers/engine_workers.py:711-731
per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
    layered_summon=self.layered_summon, base_sync_done=True)          # ← adapter

do_lora_base_sync = False
if not self.peft_merge and peft_config is not None:
    do_lora_base_sync = not self.base_sync_done

if do_lora_base_sync:                                                  # 第一次：先推基座
    per_tensor_param_base, peft_config = self.actor.engine.get_per_tensor_param(
        layered_summon=self.layered_summon, base_sync_done=False)
    await self.rollout.update_weights(per_tensor_param_base, peft_config=peft_config,
                                      base_sync_done=False, global_steps=global_steps)

await self.rollout.update_weights(per_tensor_param, peft_config=peft_config,   # 第二次：推 adapter
                                  base_sync_done=True, global_steps=global_steps)
```

### 2.3 默认参数把它送进了"只收集基座"的分支

```python
# verl/workers/engine/fsdp/transformer_impl.py:794
def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **kwargs):
    merge_lora = self.model_config.lora.get("merge", False)           # 默认 False
    if hasattr(peft_model, "peft_config"):
        if not merge_lora:
            params = collect_lora_params(module=self.module, layered_summon=layered_summon,
                                         base_sync_done=base_sync_done)   # ← False
            if not base_sync_done:
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}

# verl/utils/fsdp_utils.py:700-713  collect_lora_params，base_sync_done=False 分支
model = peft_model.base_model.model
for name, param in model.state_dict().items():
    if any(x in name for x in ["_flat_param", "lora_"]):     # ← **显式跳过所有 LoRA 张量**
        continue
    name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
    lora_params[name] = param.detach().cpu()
```

⇒ 收集到的是**冻结基座**（LoRA 被跳过），
随后 `replace_lora_wrapper`（`fsdp_utils.py:749`）又把名字改回 `...q_proj.base_layer.weight`
—— 好让 vLLM 那个已启用 LoRA 的包装层能收下基座权重。

**`base_sync_done=False` 的语义本来是「vLLM 还没有基座，先把基座推过去」**，
它设计上就应该跟着一次 `base_sync_done=True`（推 adapter）。
**disaggregated 那条路只有前半句。**

---

## 3 · 证据

### 3.1 分支行为（离线，小模型，不依赖 Ray / 不起训练）

`scripts/probe_weight_sync_payload.py`：构造一个 2 层的 Qwen3 + LoRA(r=32, 7 个 target module)，
FSDP 包好，直接调 verl 自己的 `collect_lora_params`：

| 分支 | 张量个数 | 总字节 | 含 `lora_` 的 |
|---|---|---|---|
| **`base_sync_done=False`（disaggregated 实际走的）** | 25 | 74.8 MiB | **0** |
| `base_sync_done=True`（colocate 会再调一次） | 28 | 0.2 MiB | **28** |

### 3.2 真实训练里的实测（4 次独立短跑，每跑 2 次同步 × 3 个 trainer rank）

探针挂在 `NCCLCheckpointEngine.send_weights` 的入口，**流式穿过、不改内存行为**：

```
[sync-payload] 本次同步推出去：399 个张量 / 8,414.1 MiB / 其中 lora_ 0 个
               （首个非 lora 张量名：model.embed_tokens.weight）
[sync-payload] 盯住的层 model.layers.0.self_attn.q_proj.base_layer.weight  ‖W‖=75.377708
```

**判据（"两个东西应当相同"型，不设阈值）**：

```
推出去的  model.layers.0.self_attn.q_proj.base_layer.weight   ‖W‖ = 75.377708
磁盘上起点模型的同一层（safetensors 直接读）                    ‖W‖ = 75.377708
                                                               ↑ **逐位相同**
且两次同步之间完全一致（冻结基座本来就不会变）
```

⇒ **推的就是起点模型，不是训练中的模型。**

补充旁证：

```
8,414.1 MiB ≈ Qwen3-4B 基座 bf16 的完整大小（而 LoRA r=32 只有 ~132 MiB）
vLLM 侧以 `--enable_lora` 启动、`PunicaWrapperGPU` 已初始化
⇒ **adapter 槽位是有的，只是从来没被填过。**
```

### 3.3 ✅ 对照实验：`lora.merge=True` 确实能修好（同一条探针，同一份配置）

| | `lora.merge=False`（默认） | **`lora.merge=True`** |
|---|---|---|
| 推出去的张量名 | `...q_proj.**base_layer**.weight`（`replace_lora_wrapper` 加的后缀） | `...q_proj.weight`（合并后的正常 HF 名） |
| 含 `lora_` 的张量 | 0 | 0（**这次是对的** —— 增量已合并进基座，本就不该有独立的 adapter 张量） |
| 盯住层 `‖W‖` | **75.377708**，四跑六次同步**全部相同** | **75.397400 → 75.397392**，**随训练变化** |
| 与磁盘起点模型比 | **逐位相同** ⇒ 推的是冻结基座 | 相对差 2.6e-4 ⇒ **增量在里面** |
| 载荷 | 399 张量 / 8,414.1 MiB | 399 张量 / 8,414.1 MiB（**一样大**） |

⇒ **两条结论**：
1. `model.lora.merge=True` 是当前可用的解 —— **而它的传输代价和坏掉的那条完全一样**
   （都推 8.4 GB）。也就是说，**用户本来就在付这笔钱，只是没拿到货。**
2. 「推出去的权重**逐次不变**」本身就是一条极强的判据：
   正常训练中的策略不可能每次同步都逐位相同。

---

## 4 · 后果

```
配置意图    异步 RL：trainer 学 → 每 k 步把新策略推给 rollout → rollout 用新策略采样
实际发生    rollout **永远用起点策略采样**；trainer 学到的东西从不进入数据生成回路
⇒ 整个 RL 回路是断开的：策略梯度算的是"当前策略"，而数据来自"起点策略"，
  且这个偏离**随训练单调增大**
```

⚠️ **它在任何指标上都看不出来**：loss 会降、reward 会动、grad_norm 正常、熵正常、
`rollout_is_eff_sample_size` 有值、没有任何 warning。

**反而是那些"看起来像陈旧度"的现象在替它顶包**（我们踩了两个月）：

| 现象 | 我们当时的解释 | 真实解释 |
|---|---|---|
| 固定 `sync_every`，`rollout_corr/kl` 单调涨 36× | 陈旧度累积 | π_old 恒为 π₀ ⇒ 量的是**累计位移** |
| ESS 沿「lr × 步数」这条轴重合，与 `sync_every` 无关 | 巧合 | ESS 是**位移**的函数，本来就与陈旧度无关 |
| `staleness_threshold` 0.1→0.5，陈旧轨迹 6×，ESS 纹丝不动 | 阈值不敏感 | **这个旋钮没接到任何东西** |
| 权重同步耗时"与数据量无关"（8 GB 与 132 MB 同耗时） | 固定开销主导 | **一直都是 8 GB** |

⇒ **顺带还有一笔吞吐账**：每次同步推 8.4 GB 全量基座，而需要推的 adapter 只有 ~132 MiB
（**64×**），且推过去的内容**每次都一样**。

---

## 5 · 建议的修法（按侵入性从小到大）

**① 让 disaggregated 路径与 naive 路径行为一致**（首选）

```python
if effective_mode != "naive":
    per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
        base_sync_done=self.base_sync_done)          # 首次 False（推基座），之后 True（推 adapter）
    await self.checkpoint_engine.send_weights(per_tensor_param, peft_config=peft_config,
                                              global_steps=global_steps)
    self.base_sync_done = True
    return
```
✅✅ **🆕 我们已经把这条修法实现出来并跑通了**（2026-08-18，60 步 / 0 错误）：

```
判据①  vLLM 引擎里    list_loras() 从 [] 变成 **[123]**（= VLLM_LORA_INT_ID）
判据②  载荷           第 1 次 399 张量 / 8,414 MiB（基座）→ 之后 **504 张量 / 252 MiB（全是 lora_）**
判据③  rollout_corr/kl 每次同步后**回落到数值地板**（末两次 0.00032 / 0.00034）
                       对照未修时同位置 0.00344 —— **相差 10×**
附带    param_sync     **6.25 s → 0.974 s（6.4×）**，占一步从 ~6.5% 掉到 **0.8%**
```

**实现只需要两处各补一段**（我们写成了 monkey patch，没改 verl 源码）：
① trainer 侧记住"基座推过了没有"，首次 `base_sync_done=False`、之后 `True`；
② rollout 侧把 `peft_config` + `base_sync_done` 传给 `server_adapter.update_weights`。
`peft_config` 可以在 rollout 侧用同源的 `model_config` **就地重建**
（`PEFTHelper.from_dict` 只要 `r / lora_alpha / target_modules`），**不必跨进程传 PEFT 对象**。

⇒ ★ **所以这条修法是低风险、可立即落地的**，而且**顺带把每次同步的载荷从 8.4 GB 降到 252 MB**。

**🆕 可行性（我们查实的接线图）：两端的能力都在，缺的只是中间传参。**

```
trainer 侧   get_per_tensor_param(base_sync_done=True) ⇒ 直接吐 LoRA 张量 + peft_config   ✅ 有
🔴 断点      CheckpointEngineWorker.update_weights(global_steps)   ← 签名里没有 peft_config
             （base.py:323，只 receive_weights() 后原样交给 server_adapter）
rollout 侧   vllm_rollout.update_weights(..., **kwargs) ⇒ update_weights_from_ipc(peft_config=…)
             ⇒ _update_weights ⇒ **TensorLoRARequest(lora_tensors=weights) + add_lora**    ✅ 有
             （vllm_rollout/utils.py:262 —— **能直接从张量装 LoRA，不需要文件路径**，
               colocate 每次同步都在用它）
```

⇒ **所以这不是一个架构性的缺口，是一段没接上的管子。**
⇒ 附带收益也很大：载荷会从 **8,414 MiB 掉到 ~132 MiB（64×）** ——
目前每次同步都在搬一份**完全不变的 8.4 GB 冻结基座**。

**② 如果这条路径设计上就要求 `lora.merge=True`，那就应该硬失败**

✅ **已实测 `lora.merge=True` 能修好（§3.3）**，所以这条修法是成立的。
`update_weights` 的 docstring 里确实写着
「LoRA handling: when `model.lora.merge=True` (peft_merge), LoRA is merged into base weights before sync」
—— 若这是**要求**而非可选项，则：

```python
if effective_mode != "naive" and peft_config is not None and not self.peft_merge:
    raise ValueError(
        "disaggregated 模式下使用 LoRA 必须设 model.lora.merge=True，"
        "否则只会同步冻结基座、adapter 永远不会到达 rollout。")
```
★ **静默推一份没有 adapter 的基座，是最坏的处理方式** ——
它让整个 RL 回路断开，而所有指标都正常。

**③ 防御性判据（建议无论如何都加）**：同步之后比一比两边

```python
# 权重同步完成后，取同一层在 trainer 侧与 rollout 侧的权重，比范数；
# 相对差应 < 1e-3。若 LoRA 已启用而两边恒等于起点模型 ⇒ 直接失败。
# 成本：每次同步两个标量。收益：这一类"推了但推的是旧的/错的"再也活不过第一次同步。
```

★ 我们更希望有 ③ —— ① 只修了这一种形状，而"**推过去的东西 ≠ 手上的东西**"
这一类失败在任何后端（nccl / nixl / mooncake / kimi）下都应该是硬失败。

---

## 6 · ⚠️ 我们还没做的（提交前要么补做、要么写明）

1. ~~`lora.merge=True` 是不是就能修好，正在验证中~~ ⇒ ✅ **已验证，见 §3.3。**
2. **只测了 `backend: nccl`** —— `nixl` / `mooncake` / `kimi` 未测（但它们共用
   `engine_workers.py:698` 那条分支，**预期同样受影响**，属于[推断，未验证]）。
3. **只测了 FSDP1（`strategy: fsdp`）** —— fsdp2 / megatron 的
   `get_per_tensor_param` 是各自的实现，未验。
4. **只测了 `fully_async`**；`one_step_off` 走同一分支，未单独实测。
5. 只在 **verl 0.8.0 / torch 2.9.0 / vLLM（`--enable_lora`）/ 3 trainer + 1 rollout** 上测过。
6. 没有验证"若 rollout 侧从未收到过 adapter，vLLM 的 LoRA 槽位处于什么状态"
   （我们只证明了 adapter 没被推过去，没去读 vLLM 内部状态）。

---

## 7 · 提交清单（真要提的时候照做）

- [x] 复核 verl 主干 ——**未改**（2026-08-19，两处断点都在；见顶部情报块①）
- [x] ★ 源码版修法在 verl 源码树里实测（2026-08-19，monkeypatch 关闭）：5 步 / 3 次同步
      **全部 adapter 推送（252 MiB，基座 0 次）**、rollout 侧自描述判别命中、kl 贴地板、
      三 rank ckpt 504/504 逐位相同。产物 `logs/e22_verl_fix_20260819.log`，
      PR 版 patch + mock 测试（3/3 过）在本目录
- [ ] 补上 §6-1 的验证结果（`lora.merge=True` 是否修好）
- [ ] 附 §3.1 的离线分支复现（**两个分支的对照表是最短的说明**）
- [ ] 附 §3.2 的真实跑探针输出（**"‖W‖ 与磁盘起点逐位相同"这条判据是核心**）
- [ ] 附 §4 的后果表（**"看起来像陈旧度的现象在替它顶包"这一段要突出** ——
      它说明这条 bug 会把使用者的注意力引向完全错误的方向）
- [ ] 提出 §5-③ 的防御性判据（比只修分支更有价值）
- [ ] 与同族两份互相引用（`OPEN-verl-fsdp-size-1/analysis.md` /
      `OPEN-verl-fsdp-size-1/pytorch-background.md`）—— **同一天、同一形状：
      配置意图正确，静默走进错误分支，所有指标正常**
