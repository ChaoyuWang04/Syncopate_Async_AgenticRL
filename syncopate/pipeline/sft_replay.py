"""SFT 样本 = 用同一个 rollout 循环回放 gold。

★ 为什么不用 `apply_chat_template(整段对话)` 构造 SFT 数据

Qwen3 的 chat template 对 assistant 轮的处理是不对称的：**只给最后一个 assistant 轮
加空的 `<think>\\n\\n</think>\\n\\n`，历史轮不加**（历史推理会被剥掉）。
而增量拼接时每一轮都是「当前最后一轮」，所以每轮都会带上这个前缀。

于是「整段渲染」和「增量拼接」**天生逐 token 不相等**，无论 enable_thinking 设什么。
这不是参数问题，是结构问题。

老师包踩的就是这个坑的变体（sft-truth-report T10）：SFT 侧硬编码
enable_thinking=False、RL 侧从不传，两阶段分布不一致且**没有任何报错**。

我们的解法是**只保留一条代码路径**：SFT 数据由 `run_rollout` 回放 gold 产出，
和 RL 跑出来的序列同构是构造保证的，不需要靠测试去碰运气验证。
代价是不能直接用 verl 的 MultiTurnSFTDataset（它自己做 chat template），
所以我们产出**预分词**的样本，配一个最小 SFT 训练脚本——单卡 LoRA 场景下这更简单也更可控。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from syncopate.core.contract import IS_V15
from syncopate.core.parsing import render_final_answer, render_tool_call
from syncopate.core.parsing_v15 import render_report, render_signal
from syncopate.train.rollout_budget import ENABLE_THINKING
from syncopate.core.schemas import CaseBundle
from syncopate.core.tool_registry import ToolRegistry
from syncopate.train.rollout_loop import RolloutConfig, run_rollout


def gold_script(bundle: CaseBundle, behavior: str | None = None,
                thinking: dict[int, str] | None = None) -> list[str]:
    """把 gold 轨迹翻译成「模型每一步该输出什么文本」。

    这个函数是 SFT 和测试共用的——保证「SFT 教的格式」和「RL 解析的格式」
    是同一个东西，不会各自漂移。

    ★ behavior 必须来自 `bundle.verifier.expected_behavior`。

    早期版本这里写的是 `behavior: str = "tool_call"` 且调用方从不传值，
    结果 **clarify / reject 类 case 的监督目标里写的是 `"behavior": "tool_call"`**
    ——我们在教模型输出错误的标签。

    症状极具迷惑性：分组 val_loss 降到 **0.0000**（它确实完美学会了那个错误目标），
    但自回归生成时 behavior 恒为 tool_call，behavior_mismatch 命中率 100%。
    当时误判成"token 失衡把边界能力挤没了"，做了加权采样——那只是让它把错的学得更牢。

    教训：**loss 降到 0 只说明学到了标签，不说明标签是对的。**
    tests/train 里有一条测试专门比对监督目标和 expected_behavior。
    """
    assert bundle.gold is not None, f"{bundle.case_id} 没有 gold"
    resolved = behavior or bundle.verifier.expected_behavior
    steps = [render_tool_call(a["tool"], a.get("arguments", {})) for a in bundle.gold.actions]
    if IS_V15:
        steps.extend(_v15_tail(bundle, resolved))
    else:
        steps.append(render_final_answer(resolved, bundle.gold.final_answer))
    if ENABLE_THINKING:
        steps = attach_think(steps, thinking or {})
    return steps


def _machine_fields(bundle: CaseBundle) -> dict:
    """判分器真正会核对的字段（= 必填字段里非「只查存在」的那些）。

    ★ 这些字段**一律走 session.report**，不管本轮的行为是什么 —— 包括 defer/clarify/
      reject。理由（R1 门槛⑤ 实测逼出来的）：
        · 信令 schema 装不下它们。`defer` 要报 `data_maturity`，但
          `session.defer{reason, recheck_after_days}` 里没有这一格；
          而 R0 双臂数据是按现 schema 冻结的，改 schema 就得重建 R0。
        · 硬做字段名映射（missing_fields→missing_field、reason_code→reject_reason）
          等于在契约里再加一层翻译表 —— 多一处会漂的副本。
      ⇒ **分工写死：session.* 管编排语义（挂起/等补充/终止），session.report 管判分字段。**
      两者在 `recheck_after_days` 上有意重叠：编排要它挂复查、判分要它核数值，
      各取各的，不是副本。
    """
    # ⛔ `summary` 在 v15 是**已废除的字段**（`25 §3.1`）——它不能因为出现在某些 case 的
    #   required_answer_fields 里就混进 session.report。08-30 实案：L1 桶 150 行的 report
    #   里带着 "summary"，门槛③（summary 出现 =0）当场红。
    #   ⇒ 契约级的废除要写在**取数的地方**，不能指望每个数据构造器记得不传。
    # ⛔ 08-30 第二例（考场炸出来的，比 summary 贵得多）：`reply` 也被当成机器字段
    #   （它没有 value_source ⇒ 过不了 "any" 那道筛子），于是 L1 那 150 行变成
    #   「report 里写一句人话 → 下一步原样抄一遍」。⇒ 判据升级成 `PROSE_FIELDS` 家族。
    from syncopate.core.contract import PROSE_FIELDS
    fa = dict(bundle.gold.final_answer or {})
    return {f.key: fa[f.key] for f in bundle.verifier.required_answer_fields
            if f.key in fa and f.value_source != "any" and f.key not in PROSE_FIELDS}


# ── 信令自由文本的句式池（`25 §7㉘`）────────────────────────────────────────
#
# ⛔ 此前三条信令的 explanation / question / reason 都是**常量模板**：全库只有
#   3 句拒绝话、1 句追问话。08-30 考场实测的后果不是"话说得单调"，而是
#   **模型把那句话背下来当成了退路** —— 一旦当前局面不像训练里的样子，
#   它就吐那句 out_of_scope 了事（50 道概念题 41 道被拒）。
#   ⇒ 常量模板会长成一个"万能出口"。句式池按 case_id 定选，可复现。
_REJECT_LINES = {
    "unauthorized": [
        "这个动作需要更高的操作权限，我这边做不了。",
        "我没有执行这一步的权限，得由有授权的同事来操作。",
        "超出我能动的范围了，这类操作要走授权流程。",
        "这属于授权外的动作，我不能替你执行。",
    ],
    "out_of_scope": [
        "这个不在投放这块的范围里，我帮不上。",
        "这超出了我能处理的事情，换个投放相关的问题我再看看。",
        "这件事和广告投放没关系，我这边没法处理。",
        "我只管投放这一摊，这个问题我答不了。",
    ],
    "policy": [
        "这么做会踩到平台政策，不能执行。",
        "平台政策不允许这样操作，我这边过不了。",
        "这和现行的广告政策冲突，做不了。",
        "按政策这条路是封死的，得换个做法。",
    ],
}
_CLARIFY_LINES = [
    "还差一个 {mf}，你给我之后我马上接着查。",
    "得知道 {mf} 才能往下走，能补一下吗？",
    "麻烦给个 {mf}，不然这一步定不下来。",
    "缺 {mf} 这个信息，补上我就继续。",
]
_DEFER_LINES = [
    "现在这个数还在动，等它稳下来再下结论。",
    "观测窗口还不够，这时候判断容易判反。",
    "数据还没收敛，再等等更靠谱。",
    "现在下结论对错全看运气，先等数据稳。",
]


def _pick(pool: list[str], case_id: str) -> str:
    """按 case_id 定选 —— 同一条 case 每次重建拿到同一句（重建要可复现）。"""
    return pool[sum(map(ord, case_id)) % len(pool)]


def _v15_tail(bundle: CaseBundle, behavior: str) -> list[str]:
    """v15 终答段：机器字段 → session.report（单独一步）；行为 → 信令调用或纯文本。

    ⚠️ report 之所以**单独一步**（而不是和信令/收尾话挤一步）：解析器把
    「信令 + 别的工具同一步」判成混合形态（`25 §6③`），把「有 tool_call」读成轨迹继续。
    代价（多一步）已在判分侧抵消 —— `trajectory.num_business_steps` 排除 session.*
    （R1 门槛⑤ 实测：不排除的话 120/120 条 gold 的 efficiency 全变）。
    """
    fa = dict(bundle.gold.final_answer or {})
    machine = _machine_fields(bundle)
    head = [render_report(machine)] if machine else []
    if behavior == "defer":
        return head + [render_signal("session.defer", {
            "reason": str(fa.get("defer_reason")
                          or _pick(_DEFER_LINES, bundle.case.case_id)),
            "recheck_after_days": int(fa.get("recheck_after_days") or 5)})]
    if behavior == "clarify":
        mf = fa.get("missing_field") or "campaign_id"
        # ★ 09-02（W2⑦ WIN 族）：gold 给了具体追问句（clarify_question）就用它——
        #   「最早那条记录已出窗、请再说一次」这种问法不能靠 {mf} 模板凑
        return head + [render_signal(
            "session.clarify",
            {"question": str(fa.get("clarify_question")
                             or _pick(_CLARIFY_LINES, bundle.case.case_id).format(mf=mf)),
             "missing_fields": [mf]})]
    if behavior == "reject":
        rr = {"unauthorized": "unauthorized", "policy": "policy"}.get(
            fa.get("reject_reason"), "out_of_scope")
        expl = _pick(_REJECT_LINES[rr], bundle.case.case_id)
        return head + [render_signal("session.reject",
                                     {"reason_code": rr, "explanation": expl})]
    # tool_call / answer：机器字段已在 head 的 report 里，这里只剩一句人话。
    # ★ 人话字段（value_source=="any"，实测 60/4100 全是 CHAT 的 reply）**不进 report** ——
    #   否则等于逼模型把同一句人话在机器通道里再抄一遍（「summary 污染」同族）。
    return head + [str(fa.get("reply") or _prose_from_fields(fa))]


# 机器字段 → 一句人话（v15 契约要求终答是自然语言，而 v13 的 gold 只有机器字段）
_FIELD_CN = {
    "conclusion": "结论", "lift": "提升", "region": "地区", "sample_size": "样本量",
    "reason": "原因", "feature": "特征", "excluded": "已排除", "recommendation": "建议",
    "spend_7d": "近 7 天消耗", "installs_7d": "近 7 天安装", "roas_d7": "7 日 ROAS",
    "cpi": "CPI", "ctr": "点击率", "frequency": "频次", "new_budget": "新预算",
    "approved_budget": "核准预算", "review_status": "审核状态", "asset_id": "素材 ID",
    "data_maturity": "数据成熟度", "missing_field": "缺少字段",
    "conflict_record_ids": "冲突记录", "recheck_after_days": "建议复查天数",
}
_CONCLUSION_CN = {
    "positive": "有正向效果", "negative": "有负向效果",
    "no_significant_effect": "没有显著效果", "insufficient_evidence": "证据不足",
}


def _prose_from_fields(fa: dict) -> str:
    """把 gold 的机器字段渲染成一句人话。

    ⛔ 2026-08-30（缺陷㉖，Chaoyu 裁定修）：这里原本是一句**常量兜底**
      「已经按上面的结果处理完了。」——而 v13 压舱石那 419 行的 gold **本来就没有 reply**
      （v14 时代终答是 JSON 壳，机器字段就是答案）⇒ **41.8% 的训练行终答变成同一句空话**。
      后果实测：L2 从 78% 掉到 14%（判据要"读数在场"）、L3 归零（只查不做还说"处理完了"
      = 空头支票的训练版）。**每一条都合法、不报错。**
    ★ 一般化：换契约时「旧契约里不存在的那个字段」要有**真实来源**，不能用常量兜底。
    ★ 顺带兑现 S1 挖出的那条：非判分字段（excluded/feature/region/reason…）本来只有 14%
      活在人话里，现在它们**全部**被说出来 —— 答案不再变笼统。
    """
    if not fa:
        return "已按上面的结果处理完成。"
    parts: list[str] = []
    for k, v in fa.items():
        if k in ("summary", "reply") or v is None or v == "":
            continue
        name = _FIELD_CN.get(k, k)
        if k == "conclusion":
            parts.append(_CONCLUSION_CN.get(str(v), str(v)))
        elif isinstance(v, (list, tuple)):
            if not v:
                continue
            parts.append(f"{name}{'、'.join(str(x) for x in v)}")
        elif isinstance(v, bool):
            parts.append(f"{name}{'是' if v else '否'}")
        else:
            parts.append(f"{name} {v}")
    if not parts:
        return "已按上面的结果处理完成。"
    return "、".join(parts) + "。"


EMPTY_THINK = "<think>\n\n</think>\n\n"


def attach_think(steps: list[str], thinking: dict[int, str]) -> list[str]:
    """★ think-on 下**每个** assistant 轮都要显式写出 think 段（`25 §3.2` 修法 B）。

    ⚠️ 只做 A（切 think-on）不做 B 的后果是实测过的：监督段直接以 <tool_call> 开头、
    一个 think 块都不出现 ⇒ 变成**主动训练"永不思考"**，比 think-off 更糟。

    `thinking` = {步号: 推理文本}；没给的步填**显式空块**（= 教"这步不用想"）。
    空块与非空块的比例就是 N3「按需思考」的旋钮。
    """
    out = []
    for i, body in enumerate(steps):
        content = (thinking.get(i) or "").strip()
        prefix = f"<think>\n{content}\n</think>\n\n" if content else EMPTY_THINK
        out.append(prefix + body)
    return out


class _ScriptedEngine:
    """按剧本吐 token 的假引擎。回放 gold 时代替真模型。"""

    def __init__(self, tokenizer: Any, script: list[str]) -> None:
        self.tokenizer = tokenizer
        self.script = list(script)

    async def __call__(self, prompt_ids: list[int], sampling_params: dict[str, Any]) -> list[int]:
        if not self.script:
            return []
        return self.tokenizer.encode(self.script.pop(0), add_special_tokens=False)


@dataclass
class SFTSample:
    """一条预分词的 SFT 样本。

    loss_mask 直接复用 rollout 的 response_mask：1=模型该学会生成的 token，
    0=环境插入的工具返回。prompt 段全部为 0（不监督 system/user）。
    """

    case_id: str
    input_ids: list[int]
    loss_mask: list[int]
    prompt_length: int

    @property
    def total_length(self) -> int:
        return len(self.input_ids)

    @property
    def supervised_tokens(self) -> int:
        return sum(self.loss_mask)


async def build_sft_sample(
    bundle: CaseBundle,
    *,
    tokenizer: Any,
    registry: ToolRegistry,
    config: RolloutConfig | None = None,
    thinking: dict[int, str] | None = None,
) -> SFTSample:
    """回放 gold，产出一条 SFT 样本。

    副作用是顺带验证了 gold 走得通——工具报错会体现在 observation 里，
    进而污染后续 token。所以构造 SFT 数据这一步本身就是一次 gold 健全性检查。
    """
    # ★ 默认预算也必须**从契约派生**，不能落在 RolloutConfig 的魔数 8 上。
    #   ⛔ 2026-08-30：v15 的 report 多占一步，用满 max_steps 的 case（SIG_LOW_001）
    #     在默认配置下直接被截断——而 P3-1 那次（131/503 条）的修法只修了 build_dataset
    #     的调用方，**这里的默认值还是 8**。判据写在发生点，不靠调用方记得传参。
    from syncopate.train.rollout_budget import assistant_turn_budget
    config = config or RolloutConfig(
        max_assistant_turns=assistant_turn_budget(bundle.case.max_steps))
    output = await run_rollout(
        bundle, registry=registry, tokenizer=tokenizer,
        generate=_ScriptedEngine(tokenizer, gold_script(bundle, thinking=thinking)),
        config=config,
        rollout_id="gold", run_id="sft",
    )
    # ★ gold 回放**不许截断**——被截掉的一定是轨迹结尾（终答那段），
    #   而那正是最该学的。v13 实测 131/503 条因轮数上限用了默认 8（< case.max_steps）
    #   被无声掐断，最终结论从没进过训练数据。判据写在发生点，不靠调用方记得检查。
    if output.trajectory.truncated:
        raise ValueError(
            f"{bundle.case_id}: gold 回放被截断（原因 {output.trajectory.truncation_reason}，"
            f"需要 {len(bundle.gold.actions) + 1} 个 assistant 轮，"
            f"上限 {config.max_assistant_turns}）——"
            "SFT 样本必须是完整轨迹；轮数上限应取 case.max_steps（见 build_dataset）"
        )
    # ★ 09-02（Chaoyu 裁定，26 §4.1 改写）：**空 think 块不监督**。
    #   修法 B 让每个 assistant 轮显式带 think 段是为了**位置对齐**（think-on 模板下模型每轮都会先想），
    #   但 949 行版 4049 个块里 3959 个空块全在拿梯度教「输出空思考」——R5 全场只有 1 条非空思考，
    #   根因在此。空块留在序列里（结构在场），loss_mask 置 0（不从空块学任何东西）；
    #   非空 think（教师选中 gold 动作的那些）照常监督。简单题该不该想，交给模型自己/RL 的 reward。
    ids = output.prompt_ids + output.response_ids
    mask = [0] * len(output.prompt_ids) + output.response_mask
    n_masked = _mask_empty_think(tokenizer, ids, mask, start=len(output.prompt_ids))
    return SFTSample(
        case_id=bundle.case_id,
        input_ids=ids,
        loss_mask=mask,
        prompt_length=len(output.prompt_ids),
    )


_EMPTY_THINK_IDS: dict[int, list[int]] = {}


def _mask_empty_think(tokenizer: Any, ids: list[int], mask: list[int], *, start: int) -> int:
    """把 response 区里每个 EMPTY_THINK 的 token 段 loss_mask 置 0；返回置零的块数。
    ⚠️ 按 token 序列匹配（不是按字符），与 attach_think 产出的字面量同源；
       tokenizer 若把 "</think>\n\n" 与后续文本合并分词，会匹配不到 ⇒ 用测试守着（test_rollout_loop）。"""
    key = id(tokenizer)
    pat = _EMPTY_THINK_IDS.get(key)
    if pat is None:
        pat = _EMPTY_THINK_IDS[key] = tokenizer.encode(EMPTY_THINK, add_special_tokens=False)
    n, L, i = 0, len(pat), start
    while i <= len(ids) - L:
        if ids[i:i + L] == pat and any(mask[i:i + L]):
            for j in range(i, i + L):
                mask[j] = 0
            n += 1
            i += L
        else:
            i += 1
    return n
