#!/usr/bin/env python3
"""Zero-GPU repro: `actor.use_prefix_grouper=True` is a no-op in verl 0.8.0,
and the mask it would use is the wrong one for multi-turn agent loops.

Run:  python repro_prefix_grouper_wiring.py
Needs: verl installed; `pip install prefix_grouper` only for check C.

Three independent checks, each printed with PASS/FAIL:

  A. `apply_monkey_patch()` has exactly one call site in verl, and it does NOT
     forward `use_prefix_grouper` -> `apply_prefix_grouper_patch()` never runs.
  B. `forward_micro_batch_with_prefix_grouper()` has zero call sites
     -> the shared-prefix forward path is never entered.
  C. `suffix_mask` means "does this token exist", but verl passes `response_mask`,
     which in multi-turn agent loops is 0 for tool-observation tokens
     -> those tokens would be dropped from the model input once A and B are fixed.
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

print()
if FAILED:
    print(f"{len(FAILED)} check(s) did NOT reproduce: {FAILED}")
    sys.exit(1)
print("All checks reproduced.")
