# 提交件 · issue + PR + 三条评论（可直接粘贴）

> 状态：**正文成稿 + 分支/测试已备（2026-08-21）**，等 Chaoyu 提交。
> 分支：`/workspace/_upstream/verl` → `fix/prefix-grouper-pack-with-existence-mask @ 03b9a91`（基于上游 main `4905d0c`）。
> 复现脚本 [`repro_prefix_grouper_wiring.py`](repro_prefix_grouper_wiring.py) · 背景与考据 [`2-case.md`](2-case.md)。

## 0 · 提交顺序（照做，别跳）

```
① issue  用 Bug report **表单**开（不是整篇贴）：https://github.com/verl-project/verl/issues/new/choose
         System Info 要在**干净的上游 main** 上跑 `python scripts/diagnose.py`（在我们分支上跑会打出上游没有的 commit）
② PR     正文填 `Fixes #<issue>`；标题先本地验：
         PR_TITLE='<标题>' python3 tests/special_sanity/check_pr_title.py
③ 评论   三条，各自带 #<issue> 链接：#7202（作者本人，最高优先）· #6689 · #6401
④ CI     PR 开完**不会自动跑**，去飞书群申请（包② 实测：开 PR 到起跑隔了 1h40m）
```

⚠️ **PR 刻意只修「打包用哪个掩码」+ 加一句告警，不做重新接线** —— #7202 正是为接线被关的（维护者转向 MAGI #6689），别撞同一堵墙。
⚠️ **注意力后端那条（下面的 E）只写进 issue，不进 PR** —— 它要动 `_create_prefix_grouper_wrapper` 的行为，属于接线范畴。
⛔ **散文段落一段就是一行，不许硬折行**（GitHub 的 issue/PR 正文把单换行渲染成 `<br>`；包② 吃过亏）。

---

## 1 · Issue（Bug report 表单，逐栏填）

**Title**（一行）：

```
[trainer] PrefixGrouper packs the model input from the wrong masks - multi-turn tool tokens and prompt padding both land in the wrong place (latent today, inherited by any re-enablement)
```

### System Info

> `python scripts/diagnose.py`，在**干净的上游 main `4905d0c`** 上跑（2026-08-21）。
> 已删掉两行只暴露我们本地 checkout 路径 / 容器主机名的内容。

```
----------Python Info----------
Version      : 3.12.14
Compiler     : Clang 22.1.3 
Build        : ('main', 'Aug 14 2026 15:34:45')
Arch         : ('64bit', 'ELF')
------------Pip Info-----------
No corresponding pip install for current python.
vllm	     : 0.12.0
sglang	     : not found.
ray	     : 2.57.0
torch	     : 2.9.0
----------verl Info-----------
Version      : 0.10.0.dev
Commit Hash  : 4905d0cf4ebc7297231e15efa4cf837163efca45
----------Platform Info----------
Platform     : Linux-6.8.0-85-generic-x86_64-with-glibc2.39
system       : Linux
release      : 6.8.0-85-generic
version      : #85~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep 19 16:18:59 UTC 2
----------Environment----------
CUDA Runtime : 12.8
CUDA Compiler : Cuda compilation tools, release 12.8, V12.8.93
----------System Info----------
CPU Memory	: 503.42 GB
GPU Count	: 4
GPU 1	Type    : NVIDIA GeForce RTX 5090
GPU 1	Memory  : 31.84 GB
GPU 2	Type    : NVIDIA GeForce RTX 5090
GPU 2	Memory  : 31.84 GB
GPU 3	Type    : NVIDIA GeForce RTX 5090
GPU 3	Memory  : 31.84 GB
GPU 4	Type    : NVIDIA GeForce RTX 5090
GPU 4	Memory  : 31.84 GB
```

### Information

```
[ ] The official example scripts
[x] My own modified scripts
```

### Tasks

```
[ ] An officially supported task in the `examples` folder
[x] My own task or dataset (give details below)
```

### Reproduction

````markdown
`actor.use_prefix_grouper` has been a silent no-op on the engine-based FSDP path since the worker-to-engine refactor (#6067). That part is already known and documented in #7202, which was closed in favour of the MAGI-attention direction (#6689). This report is about what is sitting *underneath* that no-op: `verl/trainer/ppo/prefix_grouper_utils.py` builds the packed model input from the wrong masks, and the attention wrapper cannot be correct on three of the five backends it declares support for. None of it bites today because the path is dormant. All of it is inherited by whichever re-enablement lands - concretely, the closed revival PR #7202 still carries the first one: its diff to `prefix_grouper_utils.py` only converts nested tensors and unwraps UIDs, and leaves `suffix_mask=response_mask` untouched.

Everything below reproduces on `main` (re-checked today) and on 0.8.0 with a single CPU-only script, no GPU and no model download:

```bash
python repro_prefix_grouper_wiring.py     # attached below
```

```
verl 0.8.0  at /workspace/Syncopate_Async_AgenticRL/.venv/lib/python3.12/site-packages/verl

[PASS] A. apply_monkey_patch() is never called with use_prefix_grouper => the attention patch never applies
       verl/workers/engine/fsdp/transformer_impl.py:292  kwargs=['model', 'use_remove_padding', 'ulysses_sp_size', 'use_fused_kernels', 'fused_kernels_backend']
[PASS] B. forward_micro_batch_with_prefix_grouper() has zero call sites => the shared-prefix forward is dead code
       (no call sites found)
[PASS] C. passing response_mask (gradient mask) silently drops tool-observation tokens from the packed model input
       packed with existence mask : [[1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25]]
       packed with response_mask  : [[1, 2, 3, 4, 10, 11, 14, 15, 20, 21, 24, 25]]
       tokens dropped from input  : [12, 13, 22, 23]   <-- the tool observations
[PASS] D. the prefix mask is built by guessing the pad id (default 0), not from the attention mask => prompt padding is packed into the shared prefix
       prefix_grouper_utils.py : micro_batch.get("pad_token_id", 0)      <- plain dict get
       engine/fsdp/...         : get_non_tensor_data(..., default=0)     <- the accessor that
                                                                            reaches non-tensor data
       "pad_token_id": is only ever written in: ['verl/trainer/ppo/ray_trainer.py', 'verl/trainer/sft_trainer.py', 'verl/trainer/sft_trainer_ray.py', 'verl/workers/config/model.py']
                       (meta_info / model config - not the training TensorDict)
       
       left-padded prompt [[151643, 151643, 7, 8, 9]] (pad id 151643):
         prefix_ids.ne(pad_token_id=0) -> 5 "real" tokens   <- the padding is packed
         attention_mask                -> 3 real tokens
[PASS] E. on the sdpa/eager/flex_attention backends the suffix sub-call (q_len=R, k_len=P+R) cannot be correct
       verl declares supported: ['eager', 'flash_attention_2', 'flash_attention_3', 'flex_attention', 'sdpa']
       PrefixGrouper hands attn_func a 2-D padding mask (2, 7) (all True here - it carries no causality)
         -> HF sdpa raises: RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 2
       dropping the mask does not save it either - torch aligns is_causal TOP-LEFT:
         |sdpa(is_causal=True) - top-left|     = 0.000e+00
         |sdpa(is_causal=True) - bottom-right| = 9.136e-01
         keys visible per response token: top-left [1, 2, 3] vs correct [5, 6, 7]
         (top-left: a response token sees only the first t+1 PROMPT tokens, never its own prefix tail)

All checks reproduced.
```

**C - the packed input is built from the gradient mask.** PrefixGrouper uses `suffix_mask` to decide which tokens *exist*: `suffix_lens = suffix_mask.sum(dim=1)` and `suffix_mask.nonzero(...)` in `prefix_grouper/__init__.py`. verl passes `response_mask`, which is a *gradient* mask. In single-turn RLVR the two coincide, so nothing looks wrong. In multi-turn they do not, and that is verl's own convention rather than a downstream customization:

```python
# verl/experimental/agent_loop/tool_agent_loop.py
agent_data.response_mask += [1] * len(response_ids)   # model-generated  -> gradient
agent_data.response_mask += [0] * len(response_ids)   # tool observation -> no gradient
```

The shapes still line up, so nothing errors. The model is simply trained on a conversation with the tool results deleted.

**D - the prompt half is recovered by guessing the pad id.** `prefix_mask = prefix_ids.ne(pad_token_id)`, where `pad_token_id` comes from `micro_batch.get("pad_token_id", 0)`. That key is only ever written into `meta_info` and the model config - never into the training batch - and every other pad-id read on the engine path goes through `get_non_tensor_data(...)` instead. So the default 0 is what this code actually gets, and for any model whose pad id is not 0 the whole left padding of the prompt is packed into the shared prefix. Both halves have the same one-line answer: the attention mask already says which tokens exist.

**E - three of the five declared backends cannot be correct.** `_create_prefix_grouper_wrapper` forwards PrefixGrouper's mask to the HF attention function as `attention_mask`, but the two conventions do not agree. PrefixGrouper's `suffix_attn_mask` is a 2-D padding mask over `prefix + suffix` and carries no causality; the suffix sub-call has `q_len = R` while `k_len = P + R`. Feeding it to `sdpa_attention_forward` raises a broadcast error, and dropping it does not help either, because `torch.scaled_dot_product_attention(is_causal=True)` aligns the causal window **top-left** when `q_len != k_len` - so each response token would attend to the first t+1 *prompt* tokens and never to its own prefix tail. Only flash-attention's `causal=True` uses the bottom-right alignment this needs, yet `_PREFIX_GROUPER_SUPPORTED_ATTENTIONS` lists `sdpa`, `eager` and `flex_attention` as supported as well.

For context on the shape of workload this matters for: ours is multi-turn tool-calling, group size 8, roughly 4.2k-token shared prompts, where 87% of trainer tokens are shared prefix. We wired the path locally to measure it, which is how C, D and E surfaced - each one first appeared as a plausible but wrong explanation of something else, and none of them announced itself.
````

### Expected behavior

````markdown
1. `build_pg_from_micro_batch` should take token *existence* from `attention_mask` for both halves of the pack, and leave `response_mask` for the loss. Single-turn behaviour is unchanged, because there the two masks coincide.
2. `use_prefix_grouper=True` should say out loud that it no longer reaches the forward. Right now it silently changes `_balance_batch` to group-level balancing and skips the optimization it names, so "enabled it, measured no speedup, concluded PrefixGrouper is not worth it" reads like a clean experiment while being an artifact of the wiring.
3. `_PREFIX_GROUPER_SUPPORTED_ATTENTIONS` should not claim `sdpa` / `eager` / `flex_attention` until the suffix sub-call gets a mask those backends can act on. Whichever way this is resolved, it is a decision about the re-enablement design rather than a small fix, so I have deliberately left it out of the PR.
4. For the prefix-tree direction (#6689 / #6401): the same convention question applies wherever packed inputs are built from multi-turn rollouts - existence must come from the attention mask, never from the loss mask, and never from a pad-id guess.

I have a PR up for 1 and 2 (the parts that are small and uncontroversial), with a CPU regression test that is red on the current code and green after. Happy to test any re-enablement on the workload above; equivalence on our side is checked as bitwise-identical `log_probs` in fp32 against a noise-floor control, not as throughput.
````

---

## 2 · PR

**Title**（一行，已过 `check_pr_title.py`）：

```
[trainer] fix: build the prefix-grouped input from the attention mask
```

**Body**（按 PULL_REQUEST_TEMPLATE 六节）：

````markdown
### What does this PR do?

Fixes #<issue-number>.

`prefix_grouper_utils.build_pg_from_micro_batch` builds the packed model input from two masks that do not mean "does this token exist", so real tokens are dropped from the input and padding is packed into the shared prefix - silently, with every shape still lining up. This PR takes both from `attention_mask` instead, and adds a warning so that `use_prefix_grouper=True` stops looking like it does something.

The code being fixed is dormant: the flag has not reached the forward since #6067, as documented in #7202. That is exactly why this is worth doing now rather than later - whichever re-enablement lands inherits it, and #7202's own diff to this file leaves `suffix_mask=response_mask` untouched. This PR deliberately does **not** re-wire anything; that discussion belongs in #7202 / #6689.

### Checklist Before Starting

- [x] Search for similar PRs: [`prefix_grouper`](https://github.com/verl-project/verl/search?q=repo%3Averl-project%2Fverl+prefix_grouper&type=issues) (6 results: #4368 merged then regressed by #6067, #7202 closed, #6401 RFC, #6689 draft, #7292, #6439) and [`suffix_mask`](https://github.com/verl-project/verl/search?q=repo%3Averl-project%2Fverl+suffix_mask&type=issues) (0 results).
- [x] Format the PR title as `[{modules}] {type}: {description}`

### Test

`tests/trainer/ppo/test_prefix_grouper_utils_on_cpu.py`, four cases, asserting on the packed token ids rather than on which mask object was passed. Against the current code, with the same `pad_token_id=0` its only caller resolves:

```
FAILED test_tool_observation_tokens_stay_in_the_packed_input
        AssertionError: tool-observation tokens dropped from the model input: [102, 103, 202, 203]
FAILED test_prompt_padding_is_not_packed_into_the_shared_prefix
        AssertionError: the 3 pad positions were packed as prefix
        assert 7 == 4  where 7 = Info(prefix_len=7, suffix_lens=[6, 6]).prefix_len
FAILED test_returned_mask_is_the_existence_mask
PASSED test_single_turn_packing_is_unchanged
3 failed, 1 passed
```

After: `4 passed`. `test_single_turn_packing_is_unchanged` passes both before and after on purpose - it is the guard that this does not move single-turn behaviour.

`tests/workers/config/test_actor_config_on_cpu.py` gains one case for the warning (red before: `enabling the flag must warn; got []`; green after; and no warning when the flag is off).

One caveat worth stating plainly: `prefix-grouper` is not a verl dependency and is not in `uv.lock`, so the mask test is guarded with `pytest.importorskip` and will **skip** in CI rather than run. The numbers above are from a local run with the package installed. The config-warning test needs nothing extra and does run in CI.

`pre-commit run --files ...` passes all 14 hooks.

### API and Usage Example

No API change. `build_pg_from_micro_batch`'s `pad_token_id` parameter is no longer read; it keeps its place in the signature with a default so existing callers - including #7202's revival - do not break.

```python
# the mask contract, now explicit in the code:
#   attention_mask -> which tokens exist  -> what gets packed
#   response_mask  -> which tokens learn  -> loss only
prefix_mask = attention_mask[:, :prompt_len]
suffix_mask = attention_mask[:, prompt_len:]
```

### Design & Code Changes

- `verl/trainer/ppo/prefix_grouper_utils.py`: derive both `prefix_mask` and the suffix existence mask from `micro_batch["attention_mask"]`; return the existence mask instead of `response_mask`, because the caller feeds it to `PrefixGrouper.convert_padding()`, which has to see the same layout the pack was built from; drop the now-unused `pad_token_id` plumbing.
- `verl/workers/config/actor.py`: warn in `ActorConfig.__post_init__` when `use_prefix_grouper` is set, pointing at #7202 / #6689.

Not in this PR, described in the issue: `_PREFIX_GROUPER_SUPPORTED_ATTENTIONS` lists `sdpa` / `eager` / `flex_attention`, and the suffix sub-call cannot be correct on any of them (`q_len != k_len` with a 2-D padding mask; torch aligns `is_causal` top-left). That is a re-enablement design decision, not a small fix.

### Checklist Before Submitting

- [x] Read the Contribute Guide.
- [x] Apply pre-commit checks.
- [ ] Add / Update the documentation. — not applicable, the flag is undocumented beyond its config comment
- [x] Add unit test(s). See the caveat about `importorskip` above.
- [ ] Once your PR is ready for CI, send a message in the `ci-request` channel. — ⬜ 提交后去飞书群
- [ ] recipe submodule — not applicable
````

---

## 3 · 三条评论

**贴在 [#7202](https://github.com/verl-project/verl/pull/7202)（已关闭的复活 PR，作者 supercharleszhu）—— 最高优先，他是天然盟友：**

````markdown
Thanks for writing this up - the "silent no-op since #6067" diagnosis matched what we hit independently on a multi-turn agentic workload, and we converged on the same design as your model-forward patch (split at hidden states, then response-only `FusedLinearForPPO`) before finding this PR. Three notes in case this gets revived, all filed with a CPU-only repro in #<issue>:

1. `build_pg_from_micro_batch` still passes `response_mask` as PrefixGrouper's `suffix_mask`. That is a *gradient* mask and PrefixGrouper reads it as an *existence* mask, so in multi-turn rollouts (`tool_agent_loop` zeroes it on tool observations) every tool-observation token is dropped from the packed model input, silently, with all shapes lining up. Your diff to this file converts nested tensors and unwraps UIDs but leaves that line untouched, so reviving as-is would ship it.
2. The prompt half of the same pack uses `prefix_ids.ne(pad_token_id)`, and the caller resolves that id with `micro_batch.get("pad_token_id", 0)` - a key that never reaches the training batch. For any model whose pad id is not 0, the prompt padding gets packed into the shared prefix.
3. `_create_prefix_grouper_wrapper` hands PrefixGrouper's 2-D padding mask to the HF attention function as `attention_mask`. On `sdpa` that raises, and dropping the mask does not save it either, because torch aligns `is_causal` top-left when `q_len != k_len`. Only flash-attention's `causal=True` is bottom-right aligned, so of the five backends in `_PREFIX_GROUPER_SUPPORTED_ATTENTIONS` only the two flash ones can be right.

Your comment about flattening outside the custom autograd Function (so the hidden-state gradient survives) saved us a debugging round - thank you. We are running the equivalent wiring locally on FSDP + multi-turn tooling (group size 8, ~4.2k-token shared prompt) and can report equivalence (bitwise `log_probs` in fp32, against a noise-floor control) and throughput here if that is useful evidence for re-opening.
````

**贴在 [#6689](https://github.com/verl-project/verl/pull/6689)（MAGI 方向）：**

````markdown
One input from a multi-turn agentic workload (tool loops, ~4.2k-token shared prompts, group size 8), since trie construction here consumes rollout tokens directly. The existing `prefix_grouper_utils.py` builds its packed input from `response_mask`, which in multi-turn rollouts is a *gradient* mask (`tool_agent_loop` zeroes it on tool observations), so tool-observation tokens are silently dropped while all shapes still line up; the prompt half of the same pack is recovered by comparing against a `pad_token_id` that defaults to 0. Filed with a CPU-only repro in #<issue>. Worth checking that the packing and leaf-segment masks here come from the attention mask rather than the loss mask - same trap, and it is invisible to shape checks. Happy to test this PR on our workload once the FSDP path is covered.
````

**贴在 [#6401](https://github.com/verl-project/verl/issues/6401)（RFC）：**

````markdown
+1 on the RFC - multi-turn is exactly where shared-prefix pays off most; on our agentic workload 87% of trainer tokens are shared prefix at group size 8. One convention worth pinning in the design: packed-input construction must take token *existence* from the attention mask - never from `response_mask` (a gradient mask, zeroed on tool observations by `tool_agent_loop`) and never from a pad-id comparison (the dormant utils default that id to 0). The current PrefixGrouper utils get both wrong - CPU repro in #<issue> - and it is the silent kind of wrong: shapes line up, metrics look fine, the model just never sees its tool results.
````

---

## 4 · 提交注意事项

```
⚠️ 顺序：issue 先发拿编号 → PR 正文把 `Fixes #<issue-number>` 填上 → 三条评论各带 #<issue>
⚠️ 正文里 `#<issue>` / `#<issue-number>` 共 6 处占位符，粘贴前全部替换
⚠️ 粘完回读一遍渲染结果：包② 粘贴时丢了首个小标题和半句话，肉眼没看出来
⚠️ PR 刻意不含接线，也不含 E（后端支持列表）—— 不要被 review 带偏去「顺便修好它」，那是 #7202 的坟场
⚠️ tag：supercharleszhu（#7202，盟友）· arvyanh（#6689/#6401）；wuxibin89 会自己看到
⚠️ 三条评论里都别提「我们有更好的方案」—— #7202 的隐状态切分与我们的设计同构，
   正确说法是「独立收敛 + 我们补上你缺的那几块」
```
