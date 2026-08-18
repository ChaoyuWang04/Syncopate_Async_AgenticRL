# Syncopate · 18 — 管线前提探针审计（E21 之后的同族排查）

> 建于 **2026-08-18**，起因是 [`../infra_exp/E21-ddp-not-syncing.md`](../infra_exp/E21-ddp-not-syncing.md)。
>
> **E21 教给我们的不是"FSDP 有个坑"，而是一个方法**：
> > 那条 bug 不是查出来的，是**一句顺手写下的前提断言炸出来的** ——
> > 「DDP 下各 rank 的 LoRA 应该相同」。
> > ⇒ **凡是"我假设 X 成立"的地方，把它写成断言/探针，成本几乎为零。**
>
> 本文把这个方法**系统地施加到主线管线上**：逐段列出「我们一直假设成立、但从没验过」的前提，
> 能当场验的当场验。**已跑 8 条，抓到 3 个真问题。**
>
> ⚠️ 全部在**已有产物**上跑，**没占一张卡**（batch4 当时还在跑）。

---

## 0 · 三十秒读懂

```
🔴 P1  「合并后的 RL 模型」根本没合并 —— models/Qwen3-4B-rl-v13-s110 的主权重
       与 SFT 模型**逐位相同**（4 层抽查，0 个元素不同）。RL 学到的东西只在 lora_adapter/ 里。
       ⇒ M7-b 的评测是对的（它显式传了 --adapter），
         但**下一轮 RL 若拿它当起点，会静默丢掉整轮 RL** —— launch_rl 没有加载 adapter 的入口。

🔴 P2  那个 lora_adapter **就是 rank_0 一份**（q_proj ‖ΔW_eff‖ 0.041528 vs rank0 0.041530，
       rank1 0.046643 / rank2 0.043280）⇒ **E21 的 1/3 通过 merger 交付到了我们评测的模型上**，
       这是那条 bug 影响主线产物的**确切路径**。

🟠 P3  prune_rl_ckpts.py 用 `next(glob(...))` **非确定性地**留一份 LoRA、删掉其余
       ⇒ global_step_5..25 的 rank1/2 **已永久丢失**，事后无法重建正确的 DDP 平均。
       只有未瘦身的 global_step_27 还留着三份。

✅ 另外 5 条验过没问题（评测分片合并、审计配对、SFT 桶泄漏、GRPO 组结构、logprob 覆盖），
   结果与"合格"的边界写在 §2。
```

---

## 1 · 方法：把「前提」变成「探针」

一条可用的探针要同时满足三点，否则它会变成又一个「看起来在量、其实量错对象」的判据：

```
① 前提能写成一句「A 与 B 应当相同 / 某集合应当完整 / 某量应当恰好为 X」
② 它在**已有产物**上就能验（不必重跑训练）
③ **它有能力失败** —— 先在一个已知会违反的输入上确认它会红
```

★ 第三点是本次真正救命的：P1 的探针同时打印了「SFT 模型 − 裸基座」的差
（0.17–0.37，明显非零）⇒ **证明探针能检出真实差异**，
所以「RL 模型 − SFT 模型 = 0」不是探针坏了。
⇒ 这就是 `blank-thresholds-are-not-passes` ⑤「判据为空时，先怀疑解析器」的正向做法。

---

## 2 · 已跑的 8 条探针

| # | 前提（我们一直假设成立） | 结果 |
|---|---|---|
| **P1** | 「merge 之后的模型 = SFT + RL 的 LoRA」 | 🔴 **不成立**，见 §3 |
| **P2** | 「ckpt 的三个 rank 是同一份副本，取哪个都行」 | 🔴 **不成立**（E21），且已交付到产物，见 §4 |
| **P3** | 「瘦身只是省空间，不丢信息」 | 🟠 **不成立**，见 §5 |
| **P4** | 「每次更新看到 6 条**不同**的题 × 8 采样」 | 🟡 107/110 步成立，3 步只有 4–5 条不同题（§6） |
| **P5** | 「4 片并行评测合并后 = 冻结 EVAL 全集，不漏不重」 | ✅ 343/343，重复 0、漏 0、多 0（三份审计都查了） |
| **P6** | 「配对比较真的按 case_id 配对」 | ✅ `compare.py:93` 取 `set(a) & set(b)` 交集，三份审计 case 集完全一致 |
| **P7** | 「SFT 的 val 桶与 train 桶不重叠」 | ✅ case 级交集 = 0 …… ⚠️ **但模板家族 100% 重合**（§7） |
| **P8** | 「rollout 的 logprob 全部来自引擎，没有占位值」 | ✅ `logprob_coverage` 全样本 = 1.0000 |

---

## 3 · 🔴 P1 ·「合并后的 RL 模型」根本没有把 RL 合并进去

**探针**（`/tmp/probe_merged2.py`，CPU、秒级）：抽 4 个被适配的层，逐元素比。

```
层                                        RL−SFT ‖Δ‖   不同元素   最大|Δ|   SFT−裸基座 ‖Δ‖
model.layers.0.self_attn.q_proj.weight      0.000000        0     0.0e+00        0.3641
model.layers.20.self_attn.v_proj.weight     0.000000        0     0.0e+00        0.1709
model.layers.35.mlp.down_proj.weight        0.000000        0     0.0e+00        0.3695
model.layers.10.self_attn.o_proj.weight     0.000000        0     0.0e+00        0.3162
                                            ↑ 逐位相同              ↑ 探针能检出真实差异
```

**这不是 verl 的 bug，是我们文档的读法错了。** LoRA 训练里基座是冻结的，
`verl.model_merger` 对 LoRA 的产出**本来就是**「未改的基座 + 旁边一个 `lora_adapter/`」
（`base_model_merger.save_lora_adapter`）。`08 §4.3` 写的「产出里同时有完整模型和 `lora_adapter/`」
**字面是对的**，但「完整模型」被读成了「已合并」——`16 §6` 就直接写了「（已合并）」。

### 3.1 后果一：M7-b 的评测**是对的**（虚惊）

审计文件里记着完整的 label：

```
_audit/v13_rl_s110.json → "models/Qwen3-4B-sft-v13-e1 + models/Qwen3-4B-rl-v13-s110/lora_adapter [vllm]"
```

⇒ 显式传了 `--adapter`，vLLM 侧 `enable_lora=True`（分片日志里 `PunicaWrapperGPU` 已初始化）
⇒ **评的是「SFT 基座 + RL 的 LoRA」，组合正确。**
★ 这一条要归功于 `eval_parallel.sh` 那个 `MODEL="${MODEL:?}"` 必填改造 ——
**上一次踩的坑（基座默认成裸模型）留下的那道闸，这次挡住了另一个方向的同类错误。**

### 3.2 后果二：🔴 **下一轮 RL 的起点会静默丢掉整轮 RL**

`08 §4.2` 写着：

> ★ **RL 起点必须是 merge 后的模型**：`launch_rl` **没有加载 adapter 的入口**，
> 而且 verl 用 LoRA 时 reference = 关掉 adapter = 基座，合并之后 reference 才等于 SFT。

⇒ 谁按这句话把 `models/Qwen3-4B-rl-v13-s110` 当第二轮 RL 的起点，
**实际拿到的是 SFT 模型**，第一轮 RL 的成果全部丢失，**而且不会有任何报错**
（形状对、能加载、能训、指标正常）。

⇒ **这是第七形态的教科书复刻**：*一个默认值/路径看起来正常，实际指向了另一件事，而且不报错*。

### 3.3 修法（两件，都很小）

```
① 补一个真正的合并入口：把 lora_adapter 用 peft merge_and_unload 折进主权重，
   另存 models/<name>-merged/。⚠️ 注意 bf16：ΔW 的相对量级只有 5.5e-4，
   而 bf16 的相对分辨率约 3.9e-3 ⇒ **必须在 fp32 下相加再转回 bf16**，否则大部分元素会被舍回原值。
   ★ 这一条本身就要配一条断言：合并后 ‖W_merged − W_base‖ 必须 > 0。
② launch_rl 起手加一条前提检查：若 --model 指向的目录里存在 lora_adapter/，
   **直接报错**并提示「这不是合并后的模型」。给错宁可报错。
```

---

## 4 · 🔴 P2 · 那个 adapter 就是 rank_0 一份 —— E21 交付到主线产物的确切路径

```
lora_adapter 的 layers.0.self_attn.q_proj  ‖ΔW_eff‖ = 0.041528
ckpt global_step_27  rank0 = 0.041530   rank1 = 0.046643   rank2 = 0.043280
                        ↑ 就是它（差值是 safetensors 存 bf16 的舍入）
```

**为什么 merger 会只取 rank_0**：`fsdp_model_merger._merge_by_placement` 对
`placement.is_replicate()` 直接 `return tensors[0]`。DDP 下每 rank 是完整副本，
在 E21 之前这是**正确且高效**的做法。E21 之后它变成了「静默丢掉 2/3」。

⇒ **E21 §4 那句「最终被保存/被推给 vLLM 的那一份只学到 1/3」，在产物层面得到了确证。**
⇒ 因此 **M7-b 的全部评测数字（逐题 85 涨/70 跌、写桶 +0.057、`unauthorized_write` 91→128）
都是 rank_0 那 1/3 的结果。** 结论的**方向**未必错，但**幅度全部要在 E21 修好后重测。**

---

## 5 · 🟠 P3 · 瘦身脚本非确定性地留一份、删掉其余

```python
# scripts/prune_rl_ckpts.py:47
full = next(actor.glob("model_world_size_*_rank_*.pt"), None)   # ← 不排序，取文件系统先给的那个
...
full.unlink()
```

实测 `global_step_5/actor/` 里已经只剩 `model_lora_only.pt`，三个 rank 的全量文件都没了。

⇒ 两条后果：
1. **`global_step_5..25` 的 rank1/2 永久丢失** ⇒ 事后无法重建"本该有的 DDP 平均"。
   **只有未瘦身的 `global_step_27` 还留着三份**（E21 的静态证据也是从 `b16_ref_off_60/global_step_15` 取的）。
2. `next(glob)` **不排序** ⇒ 留下来的是哪一个 rank **没有记录**。
   在 E21 之前这无所谓（"反正都一样"），现在它意味着**我们不知道那份 LoRA 是谁**。

⇒ **修法**：`sorted(...)[0]` + 把保留的 rank 写进 `lora_train_meta.json`；
在 E21 修好之前，**暂停瘦身**（27 GB/个，但信息一旦删掉就回不来了）。

---

## 6 · 🟡 P4 · GRPO 组结构：3/110 步里同一条题出现了两三组

```
每步的组数分布   {6: 107,  5: 2,  4: 1}
组大小分布       {8: 653,  16: 2,  24: 1}
异常步           13.jsonl(4 组: 8/8/8/24) · 31.jsonl(5 组: 8×4/16) · 38.jsonl(同)
```

⇒ 动态分池**有放回地**抽到了同一条 case。那一步实际只看了 4–5 条不同的题。
⚠️ **这本身不一定是 bug**（verl 按 `uid` 分组，两次抽样会得到两个独立的 8 人组，
组内归一化仍然正确）—— 但它**没有被任何判据看着**。
⇒ **动作**：给 `Pool` 加一条断言「同一批次内不重复抽同一 case」，或显式接受并记一行日志。
成本：几行。

---

## 7 · ✅ P5–P8 的边界（合格，但有两条要写下来）

- **P5 评测分片合并**：`v13_sft_e1 / v13_sft_e2 / v13_rl_s110` 三份审计
  **各 343 行 / 343 唯一 case / 重复 0 / 漏 0 / 多 0**，与 `data/splits/v13/eval_cases.json` 完全一致。
  ⇒ `merge_eval_shards --expect` 那道闸是有效的。
  ⚠️ **顺带一处文档过期**：`05-handoff §1` 写 v13 三桶是「EVAL 278 / SFT 511 / RL 881」，
  实测 `eval_cases.json` 是 **343**（`16-m7b` 写的 343 才对）。**引用桶大小请以 splits 为准。**
- **P7 SFT 桶泄漏**：case 级交集 = **0**（L1/L2 门禁有效）。
  ⚠️ **但 val 的 21 个模板家族 100% 出现在 train 里**，而全库只有 160 种句式
  ⇒ **`val_loss` 在这份数据上基本不含信息**（v13 e2 降到 **0.0110**，v11 当年是 0.111）。
  这不是泄漏 bug，是**尺子失效** ⇒ 选 ckpt 一律按决策位熵 + 有梯度格子，别看 val_loss（`14 §1-②`）。

---

## 8 · ⬜ 还没跑的探针（按「能不能抓到 E21 那种问题」排）

> 每条都写了**前提 / 判据 / 成本**。前四条我认为值得优先做。

| # | 前提（从没验过） | 判据 | 成本 |
|---|---|---|---|
| **Q1** | 🔴 **权重同步之后，vLLM 里的权重 == trainer 里的权重** | 同步后各取同一层的范数，两边比；相对差应 < 1e-3。<br>★ **这是 E21 的同族**：中间隔着 CheckpointEngine 的 bucket 推送，我们只验过"没 OOM" | 训练里加 10 行探针 |
| **Q2** | 🔴 **评测走 vLLM LoRA，训练走合并权重，两条路径等价** | 同一 prompt、同一 ckpt，`base+peft adapter` 的 logprob vs `合并权重` 的 logprob，逐 token 比。<br>★ 若不等价，**所有 RL 评测都系统性偏**，而且和 E20 的 `log_ppl_diff` 地板混在一起分不开 | 1 卡 · 10 分钟 |
| **Q3** | 🔴 **E20 的 `log_ppl_diff` 增长来自陈旧度** | **按 rank 分别打 `log_ppl_diff`**。E21 下 rank1/2 的 π_old 已经偏离被推给 vLLM 的 rank_0 ⇒ 若 rank0 ≈ 地板、rank1/2 明显更大，**E20 §3「229/230 来自陈旧度」要重写** | 1 行 print |
| **Q4** | 🟠 **失败注入对同一 case 的 8 次 rollout 是确定性的** | 同组 8 条轨迹，注入的失败类型必须一致（`05 §8-13`：GRPO 下随机注入会污染 advantage）。现在**没有任何守卫** | dump 里加字段 |
| Q5 | 🟠 SFT 与 RL 看到的 token 序列同构（整段渲染 vs 增量拼接） | 取同一 case，SFT parquet 的 `input_ids` 与 `run_rollout` 回放 gold 的拼接结果逐 token 比 | CPU · 30 分钟 |
| Q6 | 🟠 `response_mask=1` 只落在模型生成的 token 上，工具返回一律 0 | 从 `segments` 重建 mask，与训练用的 mask 逐位比 | CPU |
| Q7 | 🟠 RL parquet 的 prompt 与冻结 EVAL 渲染同源、且无截断 | 逐 case 比对渲染结果；`prompt_truncated_tokens` 必须恒为 0 | CPU |
| Q8 | 🟡 `--rollout-n 8` 真的产生 8 条**不同**的轨迹 | 已有旁证（组内 std 0.258、79% 有 ≥2 种工具序列），但没有断言 | 几行 |
| Q9 | 🟡 优化器状态在断点续跑后被正确恢复 | 存 → 载 → 逐张量比 | CPU |

---

## 9 · 建议**常驻**的五条断言（成本几乎为零，但能自己喊）

E21 的教训是「顺手写下的一条断言抓到了最大的 bug」。下面五条建议直接写进代码，
而不是写进文档：

```
A1  launch_rl 启动时：若 --model 目录里有 lora_adapter/ ⇒ 报错（§3.2）
A2  merge 产出后：assert ‖W_merged − W_base‖ > 0                      （§3.3）
A3  任何"读一个 rank 就代表全部"的地方：先比两个 rank，不同就报错
    —— 现存三处：rl_ckpt_to_adapter.py(已有✅) / rl_ckpt_drift.py(❌) / prune_rl_ckpts.py(❌)
A4  Pool 每批次：assert 没有重复 case                                  （§6）
A5  eval 审计写盘时：assert case 集合 == split 的 eval_cases          （已有 --expect ✅，
    但只查了条数，没查集合；改成集合比对）
```

★ 挑选原则（也是 E21 那条断言之所以有效的原因）：
**断言要写在「两个东西应当相同」的地方，而不是写在「这个数应该在某个范围里」的地方。**
前者非黑即白、不需要阈值、不会因为基线漂移而失效；
后者就是本项目一直在踩的「空门槛 / 门槛太宽」。

---

## 10 · 对已有结论的影响

| 结论 | 影响 |
|---|---|
| `16-m7b` 的全部评测数字 | 组合是对的（§3.1），但那份 LoRA 是 **rank_0 的 1/3**（§4）⇒ **幅度要重测** |
| `16 §6` 写的「models/…-rl-v13-s110/（已合并）」 | 🔴 **错的**，要改；否则下一轮 RL 会静默丢掉整轮（§3.2） |
| `17-rl-learning-blocked` §3 的位移算术 | 位移是 rank_0 一份的位移 ⇒ 「lr 占 10×、次数占 1.9×」的**排序**不变，**倍数要重算** |
| `05-handoff §1` 的 v13 三桶数字 | 过期（343 不是 278），以 `data/splits/v13` 为准 |
| 历史 ckpt 能否事后修复 | ❌ `global_step_5..25` 的 rank1/2 已删（§5），只有 `global_step_27` 完整 |
