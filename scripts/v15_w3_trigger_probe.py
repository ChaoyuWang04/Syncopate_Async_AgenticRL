#!/usr/bin/env python
"""v15 · W3② —— 触发可学性探针：用**题面文本**能不能预测「该行是不是难例」（`26 §W3` 门槛② ≥80%）。

    .venv/bin/python scripts/v15_w3_trigger_probe.py

题面预测不出来 = 模型也学不出来（难例是隐藏的模板族标签，不是题面特征）。
两个口径都报：⒜ 族级（BUD/DIA/FAIL/RAG/SCALE vs 其它族）⒝ **族内**（同族里进了 CoT 池的 vs 没进的）
——⒝ 才是"该想"的真判据；再报 ⒞ = ⒝ 加上 W3② 显性化后的题面。
分类器：哈希字符 1–3gram + numpy 逻辑回归，5 折交叉验证（不引入 sklearn 依赖）。
读数落盘 _audit/v15_w3/trigger_probe.json。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from u_build_v15_cot import explicit_hard_prompt  # noqa: E402
from syncopate.pipeline.split import DEFAULT_BATCH_DIR, DEFAULT_SPLIT_DIR, DEFAULT_SFT_DIR, DEFAULT_RL_DIR

D = 2 ** 12


def feats(text: str) -> np.ndarray:
    v = np.zeros(D, dtype=np.float32)
    for n in (1, 2, 3):
        for i in range(len(text) - n + 1):
            v[hash(text[i:i + n]) % D] += 1.0
    nrm = np.linalg.norm(v)
    return v / nrm if nrm else v


def cv_acc(X: np.ndarray, y: np.ndarray, folds: int = 5, epochs: int = 20, lr: float = 0.5) -> float:
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    accs = []
    for k in range(folds):
        te = idx[k::folds]; tr = np.setdiff1d(idx, te)
        w = np.zeros(X.shape[1], dtype=np.float32); b = 0.0
        pos = y[tr].mean()
        for _ in range(epochs):
            z = X[tr] @ w + b
            p = 1 / (1 + np.exp(-z))
            # 类别加权（难例是少数类）
            wt = np.where(y[tr] == 1, 0.5 / max(pos, 1e-3), 0.5 / max(1 - pos, 1e-3))
            g = (p - y[tr]) * wt
            w -= lr * (X[tr].T @ g) / len(tr); b -= lr * g.mean()
        pred = (X[te] @ w + b > 0).astype(int)
        # 平衡准确率（正负各占一半权重），否则"全判简单"就 80%
        tp = ((pred == 1) & (y[te] == 1)).sum() / max(1, (y[te] == 1).sum())
        tn = ((pred == 0) & (y[te] == 0)).sum() / max(1, (y[te] == 0).sum())
        accs.append((tp + tn) / 2)
    return float(np.mean(accs))


def main() -> int:
    from syncopate.pipeline.split import load_bundles
    bundles = load_bundles(Path(DEFAULT_BATCH_DIR))
    pool = {r["case_id"].replace("_COT15", "") for r in json.load(open("data/u_route/v15_cot_rows.json"))}
    hard_fams = {c.split("_")[0] for c in pool}
    ids = [c for c, b in bundles.items() if b.gold]
    msg = {c: bundles[c].case.user_message for c in ids}
    # ⒜ 族级
    ya = np.array([int(c.split("_")[0] in hard_fams) for c in ids])
    Xa = np.stack([feats(msg[c]) for c in ids])
    acc_a = cv_acc(Xa, ya)
    # ⒝ 族内：同族里进池 vs 没进池
    fam_ids = [c for c in ids if c.split("_")[0] in hard_fams]
    yb = np.array([int(c in pool) for c in fam_ids])
    Xb = np.stack([feats(msg[c]) for c in fam_ids])
    acc_b = cv_acc(Xb, yb)
    # ⒞ 族内 + 显性化（池内题面加多步诊断问法）
    Xc = np.stack([feats(explicit_hard_prompt(msg[c], c) if c in pool else msg[c]) for c in fam_ids])
    acc_c = cv_acc(Xc, yb)
    out = {"n_cases": len(ids), "hard_families": sorted(hard_fams), "pool": len(pool),
           "family_level_balanced_acc": round(acc_a, 3),
           "within_family_balanced_acc": round(acc_b, 3),
           "within_family_after_explicit_prompt": round(acc_c, 3),
           "threshold": 0.80, "pool_by_family": dict(Counter(c.split("_")[0] for c in pool))}
    Path("_audit/v15_w3").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("_audit/v15_w3/trigger_probe.json", "w"), ensure_ascii=False, indent=1)
    print(f"[trigger-probe] 题面→难例 平衡准确率（5 折）：族级 {acc_a:.1%} · 族内 {acc_b:.1%} · 族内+显性化 {acc_c:.1%}（门槛 ≥80%）")
    print(f"  难例族 {sorted(hard_fams)} · 池 {len(pool)} / 族内 {len(fam_ids)} · 全量 {len(ids)}")
    print("  读法：族内低 = 难例是隐藏标签、题面看不出该想；显性化后过线 = 触发特征变得可学")
    return 0 if acc_c >= 0.80 else 2


if __name__ == "__main__":
    raise SystemExit(main())
