#!/usr/bin/env python3
"""Zero-GPU repro: `actor.use_prefix_grouper=True` is a no-op in verl 0.8.0,
and the mask it would use is the wrong one for multi-turn agent loops.

Run:  python repro_prefix_grouper_wiring.py
Needs: verl installed; `pip install prefix_grouper` only for check C.

Five independent checks, each printed with PASS/FAIL.

Wiring (context - already reported in #7202):

  A. `apply_monkey_patch()` has exactly one call site in verl, and it does NOT
     forward `use_prefix_grouper` -> `apply_prefix_grouper_patch()` never runs.
  B. `forward_micro_batch_with_prefix_grouper()` has zero call sites
     -> the shared-prefix forward path is never entered.

Latent correctness bugs that any re-enablement inherits (the point of this repro):

  C. `suffix_mask` means "does this token exist", but verl passes `response_mask`,
     which in multi-turn agent loops is 0 for tool-observation tokens
     -> those tokens are dropped from the packed model input.
  D. the prefix half of the same pack is built from `prefix_ids.ne(pad_token_id)`,
     where `pad_token_id` comes from a plain dict `.get(..., 0)` - unlike every
     other pad-id read on the engine path, which goes through
     `get_non_tensor_data` -> for any model whose pad id is not 0, the prompt
     padding is packed into the shared prefix.
  E. the wrapper hands PrefixGrouper's 2-D padding mask straight to the HF
     attention interface, where `attention_mask` means something else. On the
     `sdpa`/`eager`/`flex_attention` backends - all three declared supported -
     the suffix sub-call (q_len=R, k_len=P+R) cannot be correct.
"""
from __future__ import annotations
import ast, pathlib, sys

import verl

VERL = pathlib.Path(verl.__file__).parent
FAILED = []


def _call_sites(func_name: str):
    """Return [(path, lineno, [kwarg names])] for every call of func_name in verl."""
    out = []
    for py in VERL.rglob("*.py"):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                name = getattr(f, "id", None) or getattr(f, "attr", None)
                if name == func_name:
                    out.append((py.relative_to(VERL.parent), node.lineno,
                                [k.arg for k in node.keywords if k.arg]))
    return out


def check(label: str, ok: bool, detail: str = ""):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    if detail:
        print("\n".join("       " + l for l in detail.strip().split("\n")))
    if not ok:
        FAILED.append(label)


print(f"verl {getattr(verl, '__version__', '?')}  at {VERL}\n")

# ── A ────────────────────────────────────────────────────────────────────────
sites = _call_sites("apply_monkey_patch")
detail = "\n".join(f"{p}:{n}  kwargs={k}" for p, n, k in sites)
forwards = [s for s in sites if "use_prefix_grouper" in s[2]]
check("A. apply_monkey_patch() is never called with use_prefix_grouper "
      "=> the attention patch never applies",
      len(sites) >= 1 and not forwards, detail)

# ── B ────────────────────────────────────────────────────────────────────────
sites_b = _call_sites("forward_micro_batch_with_prefix_grouper")
check("B. forward_micro_batch_with_prefix_grouper() has zero call sites "
      "=> the shared-prefix forward is dead code",
      len(sites_b) == 0,
      "\n".join(f"{p}:{n}" for p, n, _ in sites_b) or "(no call sites found)")

# ── C ────────────────────────────────────────────────────────────────────────
try:
    import torch
    from prefix_grouper import PrefixGrouper
except ImportError as e:
    print(f"[SKIP] C. needs torch + prefix_grouper ({e})")
else:
    # One group, two samples. Response layout mirrors a tool-calling agent loop:
    #   [model tokens][tool observation][model tokens]
    # attention_mask (existence) is all ones; response_mask (gradient) zeroes the tool part.
    prefix_ids = torch.tensor([[1, 2, 3, 4]])            # shared prompt, 4 real tokens
    prefix_mask = torch.ones_like(prefix_ids)
    responses = torch.tensor([[10, 11, 12, 13, 14, 15],
                              [20, 21, 22, 23, 24, 25]])
    existence_mask = torch.ones_like(responses)          # every response token is real
    gradient_mask = torch.tensor([[1, 1, 0, 0, 1, 1],    # positions 2,3 = tool observation
                                  [1, 1, 0, 0, 1, 1]])

    def packed_len(suffix_mask):
        pg = PrefixGrouper.from_ungrouped_masks(
            prefix_mask=prefix_mask, suffix_mask=suffix_mask, group_sizes=[2])
        return pg.concat_input(prefix_ids, prefix_mask, responses, suffix_mask)

    got_existence = packed_len(existence_mask)
    got_gradient = packed_len(gradient_mask)          # what verl passes today
    dropped = sorted(set(responses.flatten().tolist()) - set(got_gradient.flatten().tolist()))
    check("C. passing response_mask (gradient mask) silently drops tool-observation "
          "tokens from the packed model input",
          len(dropped) > 0,
          f"packed with existence mask : {got_existence.tolist()}\n"
          f"packed with response_mask  : {got_gradient.tolist()}\n"
          f"tokens dropped from input  : {dropped}   <-- the tool observations")


# ── D ────────────────────────────────────────────────────────────────────────
UTILS = VERL / "trainer" / "ppo" / "prefix_grouper_utils.py"
ENGINE = VERL / "workers" / "engine" / "fsdp" / "transformer_impl.py"
utils_src = UTILS.read_text(encoding="utf-8", errors="ignore")
engine_src = ENGINE.read_text(encoding="utf-8", errors="ignore")
plain_get = 'micro_batch.get("pad_token_id", 0)' in utils_src
proper_get = 'get_non_tensor_data(data=micro_batch, key="pad_token_id"' in engine_src
# where does "pad_token_id" ever get written?
writers = sorted({str(py.relative_to(VERL.parent))
                  for py in VERL.rglob("*.py")
                  if '"pad_token_id":' in py.read_text(encoding="utf-8", errors="ignore")})

detail_d = (f'prefix_grouper_utils.py : micro_batch.get("pad_token_id", 0)      <- plain dict get\n'
            f'engine/fsdp/...         : get_non_tensor_data(..., default=0)     <- the accessor that\n'
            f'                                                                     reaches non-tensor data\n'
            f'"pad_token_id": is only ever written in: {writers}\n'
            f'                (meta_info / model config - not the training TensorDict)')

if "torch" in sys.modules:
    import torch
    PAD = 151643                                   # Qwen3's real pad id
    prompt = torch.tensor([[PAD, PAD, 7, 8, 9]])   # left-padded: 2 pad + 3 real tokens
    guessed = prompt.ne(0).sum().item()            # what the code computes today
    truth = prompt.ne(PAD).sum().item()            # what the attention mask says
    detail_d += (f'\n\nleft-padded prompt {prompt.tolist()} (pad id {PAD}):\n'
                 f'  prefix_ids.ne(pad_token_id=0) -> {guessed} "real" tokens   <- the padding is packed\n'
                 f'  attention_mask                -> {truth} real tokens')
    d_ok = plain_get and proper_get and guessed != truth
else:
    d_ok = plain_get and proper_get

check("D. the prefix mask is built by guessing the pad id (default 0), not from "
      "the attention mask => prompt padding is packed into the shared prefix",
      d_ok, detail_d)

# ── E ────────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn.functional as F
    from prefix_grouper import PrefixGrouper
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from verl.models.transformers.monkey_patch import _PREFIX_GROUPER_SUPPORTED_ATTENTIONS
except ImportError as e:
    print(f"[SKIP] E. needs torch + prefix_grouper + transformers ({e})")
else:
    G, H, P, R, Dh = 2, 1, 4, 3, 2
    pg = PrefixGrouper.from_ungrouped_masks(
        prefix_mask=torch.ones(1, P, dtype=torch.bool),
        suffix_mask=torch.ones(G, R, dtype=torch.bool),
        group_sizes=[G], padding_mode="right", device="cpu")

    class _Mod(torch.nn.Module):
        is_causal = True

    torch.manual_seed(0)
    q = torch.randn(G, H, R, Dh, dtype=torch.float64)
    k = torch.randn(G, H, P + R, Dh, dtype=torch.float64)
    v = torch.randn(G, H, P + R, Dh, dtype=torch.float64)

    # E1: the mask PrefixGrouper hands to attn_func is a 2-D padding mask.
    raised = ""
    try:
        sdpa_attention_forward(_Mod(), q, k, v, pg.suffix_attn_mask)
    except Exception as exc:
        raised = f"{type(exc).__name__}: {exc}"

    # E2: dropping the mask does not save sdpa - torch aligns is_causal top-left.
    def aligned(offset):
        i = torch.arange(R).view(-1, 1)
        j = torch.arange(P + R).view(1, -1)
        return j <= i + offset

    auto = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    top_left = F.scaled_dot_product_attention(q, k, v, attn_mask=aligned(0))
    bottom_right = F.scaled_dot_product_attention(q, k, v, attn_mask=aligned(P))

    check("E. on the sdpa/eager/flex_attention backends the suffix sub-call "
          "(q_len=R, k_len=P+R) cannot be correct",
          bool(raised) and torch.allclose(auto, top_left) and not torch.allclose(auto, bottom_right),
          f"verl declares supported: {sorted(_PREFIX_GROUPER_SUPPORTED_ATTENTIONS)}\n"
          f"PrefixGrouper hands attn_func a 2-D padding mask "
          f"{tuple(pg.suffix_attn_mask.shape)} (all True here - it carries no causality)\n"
          f"  -> HF sdpa raises: {raised}\n"
          f"dropping the mask does not save it either - torch aligns is_causal TOP-LEFT:\n"
          f"  |sdpa(is_causal=True) - top-left|     = {(auto - top_left).abs().max().item():.3e}\n"
          f"  |sdpa(is_causal=True) - bottom-right| = {(auto - bottom_right).abs().max().item():.3e}\n"
          f"  keys visible per response token: top-left {aligned(0).sum(1).tolist()} "
          f"vs correct {aligned(P).sum(1).tolist()}\n"
          f"  (top-left: a response token sees only the first t+1 PROMPT tokens, "
          f"never its own prefix tail)")

print()
if FAILED:
    print(f"{len(FAILED)} check(s) did NOT reproduce: {FAILED}")
    sys.exit(1)
print("All checks reproduced.")
