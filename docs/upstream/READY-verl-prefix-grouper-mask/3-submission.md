# 提交件 · issue + PR + 两条评论（可直接粘贴）

> ⚠️ **提交前还要做**：建分支、把掩码修复与死开关警告落成 diff、按 verl 约定写测试并实跑负例。
> 本文只是正文；代码尚未备好（与包②不同）。

> 状态：**按"掩码 bug 主打"定位重写完成（2026-08-19），等 Chaoyu 过目后提交。**
> 中文分析与考古 → [`analysis.md`](analysis.md)。零 GPU 复现 → [`repro_prefix_grouper_wiring.py`](repro_prefix_grouper_wiring.py)。
> **提交顺序**：① issue（掩码 bug 为主、no-op 现状为辅）→ ② 小 PR 填 `Fixes #<n>`
> （掩码修复 + 死开关警告，刻意不做接线）→ ③ **三条评论**：#7202（作者本人，他的复活版仍带此 bug，
> 最高优先）· #6689 / #6401（MAGI 方向）。
> ⚠️ 刻意**不**提"重新接线"的 PR —— #7202 已为此被关（维护者转向 MAGI），别撞同一堵墙；
> 我们自己的接线走本地补丁，验证数据回头作为评论补充。

---

## 1 · Issue

**Title:**

```
[trainer, fsdp] prefix_grouper_utils packs the model input with the gradient
mask — multi-turn tool-observation tokens are silently dropped (latent today,
bites any re-enablement; + use_prefix_grouper has been a silent no-op since #6067)
```

**Body:**

````markdown
## TL;DR

Two related facts about `actor.use_prefix_grouper`, one known and one new:

1. (known, context) The flag has been a **silent no-op** on the engine-based
   FSDP path since the worker-to-engine refactor #6067 — documented in #7202,
   which was closed in favor of the MAGI-attention direction (#6689). Still
   true on current `main`: `apply_monkey_patch()`'s single call site
   (`workers/engine/fsdp/transformer_impl.py`) does not forward the flag, and
   `forward_micro_batch_with_prefix_grouper()` has zero call sites. The only
   thing the flag still does is switch `_balance_batch` to group-level
   balancing.
2. **(new, the point of this issue)** `verl/trainer/ppo/prefix_grouper_utils.py`
   builds the packed input from the **wrong mask**: it passes `response_mask` —
   a *gradient* mask — where PrefixGrouper expects an *existence* mask. In
   multi-turn / tool-calling rollouts these differ, and **every
   tool-observation token is silently dropped from the model input**. The
   shapes still line up, so nothing errors: the model is simply trained on a
   conversation with the tool results removed.

   This is latent today only because of (1). It will bite the moment *any*
   shared-prefix path is re-enabled — and concretely: **#7202 (the closed
   revival PR) still carries it.** Its diff to `prefix_grouper_utils.py` only
   converts nested tensors and unwraps UIDs; the
   `suffix_mask=response_mask` line is untouched, so re-opening that PR as-is
   would ship the token-dropping behavior. The prefix-tree work in
   #6689/#6401 inherits the same question wherever packed inputs are built
   from multi-turn rollouts. Filing it now so the trap is on record before any
   wiring comes back.

## The mask bug, concretely

PrefixGrouper uses `suffix_mask` to decide **which tokens exist**:

```python
# prefix_grouper/__init__.py
suffix_lens = suffix_mask.sum(dim=1)      # length of each suffix suffix_mask.nonzero(as_tuple=False)       # which positions get gathered into the packed input
```

verl passes `response_mask`:

```python
# verl/trainer/ppo/prefix_grouper_utils.py  (identical on 0.8.0 and main)
prefix_grouper = PrefixGrouper.from_ungrouped_masks( prefix_mask=prefix_ids.ne(pad_token_id), suffix_mask=response_mask,            # <-- gradient mask, not existence mask ...) concat_input_ids = prefix_grouper.concat_input(prefix_ids, prefix_mask, responses, response_mask)
```

In single-turn RLVR the two masks coincide, so nothing is visibly wrong. In
multi-turn they do not — and this is verl's own convention, not a downstream
customization:

```python
# verl/experimental/agent_loop/tool_agent_loop.py
agent_data.response_mask += [1] * len(response_ids)   # model-generated  -> gradient agent_data.response_mask += [0] * len(response_ids)   # tool observation -> no gradient
```

Minimal demonstration (16 tokens, CPU-only; full script attached):

```
packed with existence mask : [[1,2,3,4, 10,11,12,13,14,15, 20,21,22,23,24,25]] packed with response_mask  : [[1,2,3,4, 10,11,      14,15, 20,21,      24,25]] tokens dropped from input  : [12,13,22,23]   <-- the tool observations
```

The failure mode is the worst kind: no crash, no shape mismatch, no warning —
just a model that can no longer see what its tools returned, while every
metric keeps looking plausible.

## Suggested actions (PR to follow for 1+2)

1. **Fix the mask semantics in `prefix_grouper_utils.py` now**, while the code
   is dormant: pack with the response *existence* mask
   (`attention_mask[:, prompt_len:]`), keep `response_mask` strictly for the
   loss. Small diff, independent of which attention backend eventually wins.
2. **Make the inert flag loud**: `use_prefix_grouper=True` currently changes
   batch balancing while silently skipping the optimization it names. A
   `warnings.warn` in `ActorConfig.__post_init__` until re-enablement lands
   spares users the "enabled it, measured no speedup, concluded it's useless"
   trap (#7202 describes exactly this state).
3. For the prefix-tree direction (#6689 / #6401): the same convention question
   applies wherever packed inputs are built from multi-turn rollouts —
   existence must come from the attention mask, never from the loss mask. Happy
   to help test on a multi-turn agentic workload (group size 8, ~4.2k-token
   shared prefixes, tool loops) — this is our production shape.

## Reproduction

`python repro_prefix_grouper_wiring.py` (attached; zero GPU — static scan for
the two disconnections + the 16-token mask round-trip above). All three checks
reproduce on verl 0.8.0 and the utils/call-site code is unchanged on `main`.

## Environment

verl 0.8.0 (+ `main` re-checked 2026-08-19) · prefix-grouper 0.0.1.post1 ·
torch 2.9 · FSDP engine · multi-turn tool-calling workload via tool_agent_loop
````

---

## 2 · PR（小而无争议：掩码修复 + 死开关警告，**不含接线**）

**Title:**

```
[trainer] fix: pack prefix-grouped input with the existence mask, not the
gradient mask; warn while use_prefix_grouper is inert
```

**Diff ①（掩码语义，`verl/trainer/ppo/prefix_grouper_utils.py`）：**

```diff
@@ def build_pg_from_micro_batch(
     prompts = micro_batch["prompts"]
     responses = micro_batch["responses"]
     response_mask = micro_batch["response_mask"]
     uids = micro_batch["uid"]
+    # `response_mask` marks which response tokens receive gradient
+    # (model-generated tokens), NOT which tokens exist: multi-turn rollouts
+    # zero it on tool-observation tokens (tool_agent_loop.py), and those
+    # tokens must still be part of the model input. Pack with the existence
+    # mask; `response_mask` stays loss-only.
+    if "attention_mask" in micro_batch:
+        response_exist_mask = micro_batch["attention_mask"][:, prompts.size(1) :].bool()
+    else:
+        # fallback; only correct when pad_token_id cannot appear inside a response
+        response_exist_mask = responses.ne(pad_token_id)
@@
     prefix_grouper = PrefixGrouper.from_ungrouped_masks(
         prefix_mask=prefix_mask,
-        suffix_mask=response_mask,
+        suffix_mask=response_exist_mask,
         group_sizes=group_sizes,
@@
-    concat_input_ids = prefix_grouper.concat_input(prefix_ids, prefix_mask, responses, response_mask)
+    concat_input_ids = prefix_grouper.concat_input(prefix_ids, prefix_mask, responses, response_exist_mask)
```

（`forward_micro_batch_with_prefix_grouper` 已经把 `response_mask` 单独传给损失 （`completion_mask=response_mask`）——语义分离之后它恰好就是对的，无需改。）

**Diff ②（死开关警告，`verl/workers/config/actor.py::ActorConfig.__post_init__`）：**

```diff
     def __post_init__(self):
         """config validation logics go here"""
+        if self.use_prefix_grouper:
+            warnings.warn(
+                "actor.use_prefix_grouper currently has NO effect on the engine-based "
+                "FSDP path: the shared-prefix forward has been disconnected since the "
+                "worker-to-engine refactor (#6067); only group-aware batch balancing "
+                "remains active. See #<issue> for details; re-enablement is discussed "
+                "in #7202 (closed) and #6689.",
+                stacklevel=2,
+            )
```

**Body:**

````markdown
Fixes #<issue-number>.

Two small, independent corrections around `use_prefix_grouper`; deliberately
**no re-wiring** here (that discussion lives in #7202 / #6689):

1. `prefix_grouper_utils.py` packed the grouped model input with
   `response_mask`, which is a *gradient* mask — in multi-turn tool-calling
   rollouts (verl's own `tool_agent_loop` convention) it is 0 on
   tool-observation tokens, so those tokens were silently dropped from the
   model input. Pack with the response *existence* mask instead;
   `response_mask` stays loss-only. Fixing it while the code is dormant means
   whichever re-enablement lands (#7202-style revival or the prefix-tree work)
   does not inherit the trap. Single-turn behavior is unchanged (the two masks
   coincide there).
2. `use_prefix_grouper=True` still switches `_balance_batch` to group-level
   balancing while silently skipping the optimization it names. Warn until the
   forward path is reconnected, so "enabled but inert" can no longer be
   mistaken for "PrefixGrouper gives no speedup".

Test: 16-token CPU round-trip (attached in the issue) — packed input with the
existence mask retains tool-observation tokens; with the old code they are
dropped. Plus a config test asserting the warning fires.
````

---

## 3 · 递给 MAGI 方向的两条评论

**贴在 [#6689](https://github.com/verl-project/verl/pull/6689)（评论）：**

````markdown
One input from a multi-turn agentic workload (tool loops, ~4.2k-token shared
prefixes, group size 8), since trie construction here consumes rollout tokens
directly: the existing `prefix_grouper_utils.py` packs the grouped input with
`response_mask`, which in multi-turn rollouts is a *gradient* mask
(`tool_agent_loop` zeroes it on tool observations) — so tool-observation
tokens get silently dropped from the packed input while all shapes still line
up. Filed with a CPU-only repro in #<issue>. Worth a check that the packing /
leaf-segment masks in this PR are derived from the attention mask rather than
the loss mask — same trap, and it is invisible to shape checks. Happy to test
this PR on our workload once the FSDP path is covered.
````

**贴在 [#7202](https://github.com/verl-project/verl/pull/7202)（已关闭的复活 PR，作者 supercharleszhu）：**

````markdown
Thanks for writing this up — the "silent no-op since #6067" diagnosis matched
what we hit independently on a multi-turn agentic workload, and we converged on
the same design as your model-forward patch (split at hidden states, then
response-only `FusedLinearForPPO`) before finding this PR. Two notes in case
this gets revived:

1. `build_pg_from_micro_batch` still passes `response_mask` as PrefixGrouper's
   `suffix_mask`. That is a *gradient* mask, and PrefixGrouper treats it as an
   *existence* mask — in multi-turn rollouts (`tool_agent_loop` zeroes it on
   tool observations) every tool-observation token gets dropped from the packed
   model input, silently, with all shapes still lining up. Packing should use
   `attention_mask[:, prompt_len:]`, with `response_mask` kept loss-only.
   CPU repro + details in #<issue>.
2. Your comment about flattening outside the custom autograd Function (so the
   hidden-state gradient survives) saved us a debugging round — thank you.

We are running the equivalent wiring locally on FSDP1 + multi-turn tooling
(group size 8, ~4.2k-token shared prefix). Happy to report equivalence
(`log_probs` bitwise) and throughput numbers here if that is useful evidence
for re-opening.
````

**贴在 [#6401](https://github.com/verl-project/verl/issues/6401)（RFC 评论）：**

````markdown
+1 on the RFC — multi-turn is exactly where shared-prefix pays off most (our
agentic workload: 87% of tokens are shared prefix at group size 8). One
convention worth pinning in the design: packed-input construction must take
token *existence* from the attention mask, never from `response_mask` (which
is a gradient mask and zeroes tool observations in `tool_agent_loop`). The
dormant PrefixGrouper utils currently get this wrong — CPU repro in #<issue> —
and it is the silent kind of wrong: shapes line up, metrics look fine, the
model just never sees its tool results.
````

---

## 4 · 提交注意事项

```
⚠️ 顺序：issue 先发拿编号 → PR 填 Fixes → 两条评论各自带 #<issue> 链接
⚠️ PR 刻意不含接线 —— 不要被 review 带偏去"顺便修好它"；那是 #7202 的坟场
⚠️ 我们自己的接线（本地补丁 + response-only LM-head 投影 + logprob 逐位判据）
   完成后，验证数据以评论形式补进 issue —— 它是"FSDP 侧有真实需求"的证据
⚠️ 与包①②③ 同批提交时互相引用一句（第四例同形状：config accepted, wire missing）
⚠️ tag：supercharleszhu（#7202，盟友；他的 PR 仍带掩码 bug —— 评论稿见 §3 第一条）·
   arvyanh（#6689/#6401）；wuxibin89 会自己看到
⚠️ 三条评论里都别提"我们有更好的方案" —— #7202 的隐状态切分与我们的设计同构，
   正确说法是"独立收敛 + 我们补上你缺的那块（掩码）"（见 SYNC 文档更正②）
```
