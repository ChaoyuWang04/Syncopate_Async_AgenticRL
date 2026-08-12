"""按模板给每个 case 裁剪工具菜单（`Case.tool_menu`）。

★ 为什么做这件事

prompt 里 **78.7% 是工具说明书**（实测：最长的一条 4889 token 里，24 个工具的
schema 占 3846，system 规则书 915，真正的任务只有 116）。而 `max_model_len =
max_prompt_length + max_response_length` 直接决定 vLLM 的显存预分配 ——
5120+2048=7168 让单卡 colocate 在 `wake_up` 时必挂（连挂四次，见 `launch_rl.py` 顶部）。

生成侧的实测需求只有 1335 token（模型自己生成 358 + 工具返回 977），所以
**压 response 没用，杠杆全在 prompt 的那 3846**。

★ 和老师那套的分歧（`reference/.../agent/runtime.py:236`）

    # 线上一致性：rollout 暴露生产注册表的完整工具集；case.allowed_tools/allowed_write_tools
    # 只属于 verifier/routing，不参与 prompt 裁剪。

他**故意不裁剪**，理由是线上暴露全量注册表，训练时裁剪会训推不一致。
我们不同：Syncopate 是 L1→L7 的垂类闭环，编排器知道当前处在哪一段，
**部署时本来就是按阶段给菜单**，所以按阶段训练才是一致的那一侧。
这是产品形态的选择，不是谁对谁错 —— 但它意味着**我们必须自己保住「选工具」的难度**。

★ 菜单怎么定：三部分，都来自实测，不拍脑袋

    gold 实际用到的工具        —— 保证任务可解（菜单必须是 gold 的超集）
  ∪ SFT 模型实测调用过的工具   —— 保证不会出现「模型想调的工具不在菜单里」
  ∪ CORE（出现在 ≥4 个模板并集里的 8 个工具）—— 保底难度

第二项**自带干扰项**：SFT 模型比 gold 多调 3–5 个工具（BUD 里会调
`campaign.list`、`system.wait`），这些留在菜单里，「从一堆里选对的那个」这个
能力就还在被训练，不需要另外掺假工具。

★★ CORE 是第一版 dry-run 打脸后加的，两个洞：

  1. CLAR / REJ 的 gold **一个工具都不调**，并集是空集 ⇒ 按「空就给 None」处理
     就是全量 24 个，最坏 prompt 还是 4889，**显存一点没省**；
     而如果真给空菜单，等于直接告诉模型「这题不用调工具」，把题做没了。
  2. HIGH 的并集只有 1 个工具 ⇒ 选工具的难度归零。

  ⇒ 每个模板都并上 CORE，最小菜单 8 个（CLAR/FRESH/HIGH/REJ），最大 13 个（FAIL）。
  **菜单里必须有该选和不该选的，"裁剪"才不等于"送答案"。**

效果（实测）：最坏情况 prompt 4889 → 3388（-31%），
配 `--max-prompt-length 3584 --max-response-length 1536` ⇒ max_model_len 5120，
和 2026-08-11 跑通 50 步的 4608 同一量级。

⚠️ 只改 `Case.tool_menu`，**case_id / gold / env / verifier 一律不动** ——
所以三桶切分不受影响，冻结 EVAL 仍然是同一批 case_id。
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按模板裁剪 case 的工具菜单")
    parser.add_argument("--batch", default="data/batches/v8")
    parser.add_argument("--sft-audit", default="_audit/v8_sft_epoch1.json",
                        help="SFT 模型的评测结果，用它的 tool_seqs 取「模型实际会调的工具」")
    parser.add_argument("--core-min", type=int, default=4,
                        help="出现在 ≥N 个模板并集里的工具进 CORE，每个模板都并上它（保底难度）")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from syncopate.pipeline.split import load_bundles

    batch_dir = ROOT / args.batch
    bundles = load_bundles(batch_dir)

    # ---- 并集 1：gold 用到的 ----
    by_template: dict[str, set[str]] = collections.defaultdict(set)
    for cid, b in bundles.items():
        by_template[cid.split("_")[0]] |= {a["tool"] for a in b.gold.actions if a.get("tool")}

    # ---- 并集 2：SFT 模型实测调用过的 ----
    audit = json.loads((ROOT / args.sft_audit).read_text(encoding="utf-8"))
    for row in audit["rows"]:
        tmpl = row["case_id"].split("_")[0]
        for seq in row.get("tool_seqs", []):
            by_template[tmpl] |= set(seq) if isinstance(seq, list) else {seq}

    # ---- ★★ 补上「cap 监视的动作工具」：护栏要训得出来，诱惑必须在菜单里 ----
    #
    # 踩到的实例：GEO 的 gold 只开审批单、从不建站，于是 campaign.create 不在并集里 ⇒
    # **模型根本没有"不打招呼就建站"这个选项** ⇒ unconfirmed_irreversible_cap 和
    # cross_region_generalization_cap 两条 cap 永远不可能命中，M4 的核心教学点全废。
    #
    # 这是「工具不在菜单里，那道题就不存在」的又一次现形（和 real_person|JP 变成
    # 0 条素材同源）。⇒ 一般化的规矩：**本 case 启用了哪条 cap，就要把那条 cap
    # 监视的动作工具放进菜单**，否则那条 cap 是死的。
    CAP_WATCHED_TOOLS = {
        "unconfirmed_irreversible_cap": ["campaign.create", "campaign.scale_budget"],
        "cross_region_generalization_cap": ["campaign.create", "campaign.scale_budget"],
        "risk_blocked_write_cap": ["campaign.update_budget"],
        "budget_over_limit_cap": ["campaign.update_budget"],
        "duplicate_write_cap": ["campaign.update_budget"],
    }
    for cid, bundle in bundles.items():
        for cap in (bundle.verifier.active_caps or []):
            by_template[cid.split("_")[0]] |= set(CAP_WATCHED_TOOLS.get(cap, ()))

    # ---- CORE：出现在 ≥CORE_MIN 个模板并集里的工具，每个模板都并上 ----
    freq: collections.Counter = collections.Counter()
    for tools in by_template.values():
        freq.update(tools)
    core = {t for t, c in freq.items() if c >= args.core_min}
    print(f"CORE（出现在 ≥{args.core_min} 个模板）{len(core)} 个: {', '.join(sorted(core))}\n")

    menus = {tmpl: sorted(tools | core) for tmpl, tools in by_template.items()}

    changed = 0
    print(f"{'模板':<7}{'菜单':>5}   工具")
    for tmpl in sorted(menus):
        print(f"{tmpl:<7}{len(menus[tmpl]):>5}   {', '.join(menus[tmpl])}")

    for cid, b in bundles.items():
        new = menus[cid.split("_")[0]]
        assert new, f"{cid} 的菜单为空 —— 空菜单等于把题做没了"
        if b.case.tool_menu != new:
            b.case.tool_menu = new
            changed += 1
            if not args.dry_run:
                b.write(batch_dir)

    verb = "将改写" if args.dry_run else "已改写"
    print(f"\n{verb} {changed}/{len(bundles)} 条 case 的 tool_menu"
          f"{'（dry-run，没落盘）' if args.dry_run else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
