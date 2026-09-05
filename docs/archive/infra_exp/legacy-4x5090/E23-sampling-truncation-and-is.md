# E23 · 采样截尾与重要性采样：**训练侧绝不能对齐到 `top_p=0.95/top_k=20`**

> 建于 2026-08-18　起因：主线 2026-08-18 交接时点名要 infra 定的问题（原信已删，⛔ 往来只走根目录 `MAINLINE-INFRA.md`）。
> 状态：✅ **已用源码实证回答**（不是讨论出来的）。⬜ 建议的验证探针见 §5。

---

## 0 · 主线问的问题与我的回答

> **主线问**：要不要把训练侧的采样对齐到 `top_p=0.95 / top_k=20`（评测现在用的那套）？
> 顾虑是「截尾之后**实际采样的分布** ≠ **算 logprob 的分布**」会污染 TIS/ESS。

**答：顾虑成立，而且比预想的更具体 —— 但方向要反过来。**

```
🔴 不要把训练对齐到评测（top_p=0.95/top_k=20）
✅ 把**评测**对齐到训练（top_p=1.0 / top_k=-1）
```

理由：**verl 的两侧本来就用着两个不同的分布，只是在 `top_p=1.0` 下它们恰好重合。**
一旦开启截尾，重合就没了，而**没有任何一层会报错**。

---

## 1 · 实证（两处源码，互为另一半）

### 1.1 rollout 侧报的 logprob 是**截尾之后**的（这一半是对的）

```yaml
# verl/trainer/config/rollout/rollout.yaml:85
logprobs_mode: processed_logprobs
```
而 vLLM 的定义（`vllm/config/model.py:216-223`，默认值其实是 `raw_logprobs`）：

> Raw means the values **before** applying any logit processors.
> **Processed** means the values **after** applying all processors, **including temperature and top_k/top_p**.

⇒ **verl 主动改成了 `processed`** —— 这对重要性采样来说是**正确**的选择：
π_old 应当是**真实的行为策略**，而行为策略就是截尾后的那个分布。

### 1.2 trainer 侧算 logprob **不截尾**（这一半是缺的）

```python
# verl/utils/torch_functional.py:672
def post_process_logits(input_ids, logits, temperature, top_k, top_p):
    if temperature != 1.0:
        logits = logits.div_(temperature)
    # TODO: add them back
    # if top_k is not None and top_k > 0:
    #     logits = TopKLogitsWarper(top_k=top_k)(input_ids, logits)
    # if top_p is not None and top_p < 1.0 and top_p > 0.0:
    #     logits = TopPLogitsWarper(top_p=top_p)(input_ids, logits)
    return logits
```

⇒ **两个截尾器被注释掉了，还留着 `TODO: add them back`。**
⇒ trainer 侧的 π_new 是**完整 softmax**（只除温度）。

★ 这两行注释本身就是证据：**上游知道这里缺一块，并且知道自己没补。**

---

## 2 · 后果：开启截尾会引入一个**系统性**的 IS 偏差

记截尾保留的概率质量为 `Z ≤ 1`（`top_p=0.95` ⇒ `Z ≳ 0.95`；`top_k=20` 还会再砍）：

```
rollout 报的   π_old = p_old / Z          （截尾后的分布，processed_logprobs）
trainer 算的   π_new = p_new              （完整 softmax，没截尾）
IS 比值        π_new/π_old = **Z · (p_new/p_old)**
```

⇒ **每个 token 的比值被系统性地乘上 `Z`**（约 0.95–0.99），而这**与策略好坏无关**。

| 口径 | 影响 |
|---|---|
| **token 级 IS** | 每 token 系统性偏低 1–5% ⇒ 直接把 `chi2_token` / ESS 顶偏，**而且看起来像"分布失配"** |
| **序列级 IS** | `Z^L`，我们的 `L ≈ 694` ⇒ `0.97^694 ≈ e^-21`。**指数级崩塌，且与 E20 §2 那条机制混在一起分不开** |

⇒ ⛔ **这会制造第四个"看起来像陈旧度"的现象** —— 我们刚为前三个付了两个月的代价（[`E22 §4`](E22-lora-never-synced.md)）。

### 2.1 ★ 为什么现在没事（也是这条推理的验算）

我们当前跑的是 `temperature=1.0 / top_p=1.0 / top_k=-1`
⇒ 没有任何 logit processor 生效 ⇒ **`processed` 与 `raw` 逐位相同** ⇒ `Z = 1` ⇒ 无偏差。

★ **这与实测对上了**：主线量到的首步 `log_ppl_diff` / `kl` 地板是 **3.4e-4**，
纯粹是 vLLM↔FSDP 的数值失配量级。**若两侧分布不同族，这个地板不可能这么小。**
⇒ 这条不是推理，是**已有数据的交叉验证**。

---

## 3 · 所以该往哪个方向对齐

| 方案 | 训练侧 | 评测侧 | 判断 |
|---|---|---|---|
| **A：训练对齐评测**（主线原议） | 改成 0.95/20 | 不动 | 🔴 **否决** —— 引入 §2 的系统性 IS 偏差，且没有任何一层会报错 |
| **B：评测对齐训练**（推荐） | 不动（1.0/-1） | 改成 1.0/-1 | ✅ **不碰 TIS 一根汗毛**，而且直接兑现主线 §10.3-① 的诉求：**评测终于在量训练的那个策略** |
| **C：两侧都截尾** | 打开 verl 那两行 | 不动 | 🟡 理论上最干净，但那是**上游标了 TODO 的改动**，要自己维护一个 warper 补丁并验数值；**除非产品上必须部署 0.95/20，否则不值得** |

**⇒ 推荐 B。** 附带一条：`top_p=1.0` 下评测会**更容易**产生格式违规
—— 但那正是**训练时真实发生的事**，评测本来就该量它（主线 §10.3-① 说的就是这个）。
⇒ 主线那个 `multi_tool_per_step_cap` 18.8% vs 0% 的落差，**方案 B 会让评测侧也显形**，
这不是坏事，是尺子终于对上了。

### 3.1 ⚠️ 唯一会改变这个建议的条件

**如果 M9 runtime 上线时必须用 `top_p=0.95/top_k=20`**，那么"训练的策略"与"部署的策略"
就必须是同一个 ⇒ 只能走方案 C（两侧都截尾），不能走 B。
⇒ 我查了 `syncopate/runtime/`：**那一层没有钉死任何 LLM 采样参数**
（`top_k` 的出现全在 `retrieval.py`，是检索的 top-k，不是采样）。
⇒ **[待主线确认]** 部署侧是否有硬约束。**没有的话就选 B。**

★ 这也是「**两个入口各写各的常量**」的第四例（前三例：长度预算 3584/1536 vs 5120/2048、
eval-vLLM 对齐 eval-HF、采样参数）。⇒ **建议采样参数也合并成一份**
（照 `syncopate/train/rollout_budget.py` 的做法），**由训练侧定义、评测侧引用**。

---

## 4 · 顺带否掉主线 §10.2 的一个残留推理

主线 §11.5 已自查出 §10.2「只剩采样分布这一处差别」不成立（两个混淆变量）。**同意，并补第三个**：

```
③ 评测的 RL 模型走 vLLM 的 **LoRA 适配路径**（base + adapter），
   而训练 rollout 走的是**合并权重**（而且因 E22 那份权重根本没更新过）
   ⇒ 两条推理路径的数值行为从没验证过等价（主线 19 §5.2 的 Q2 / 本线 R0-b）
```

⇒ ⇒ **18.8% vs 0% 这个落差目前有四个候选解释**，采样只是其中之一。
⇒ 但 §10.4 那个**决策**不依赖于查清落差 —— B 方案无论落差的成因是什么都成立。

---

## 5 · ⬜ 建议的验证探针（便宜，但不必挡住决策）

```
同一份冻结 prompt、同一个 ckpt、同一条 rollout 路径，只改 top_p：
    臂① top_p=1.0  ｜ 臂② top_p=0.95, top_k=20
判据：比 `chi2_token` 与 `rollout_is_eff_sample_size`
      若臂②系统性更差且差值随 response_length 增长 ⇒ §2 的偏差已实测坐实
成本：两次短跑（各 ~35 min），可并进 R0-a 那一批
```

⚠️ **它不该挡住决策** —— 因为方案 B 的成立不依赖这个探针（B 根本不改训练侧）。
这个探针的价值是**把 §2 从推理变成实测**，以及在选 C 时提供数值依据。
