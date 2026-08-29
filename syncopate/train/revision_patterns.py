"""revision（自我修正）标记正则库 —— v14.5 S3 产出（24 §4-P2）。

来源：Congliu R1 中文 110k 流式采样 30000 条 think 的频率筛选（≥30 次才收）；
用途：CoT 承诺闸的修正样本优先标记 + P3 aha 观测器（first_success_events）。
本地 59 条 8B think 命中率 0/59。外部数据零入库（模式 B）。
"""

REVISION_PATTERNS = {
    'recheck': r'再(检查|确认|核对|验证)一?下?',
    'let_me_again': r'让我再',
    'went_wrong': r'(算|想|理解)错了',
    'wait_stop': r'等等[，,、]',
    'not_right': r'不对[，,。]',
    'maybe_wrong': r'(可能|似乎|好像)(不对|有问题|错了)',
    'redo': r'重新(想|算|看|考虑|梳理|检查)',
    'hold_on': r'等一下',
    'reconsider': r'重新考虑',
    'actually': r'其实(不|并不|应该)',
    'correct_it': r'修正一?下?',
    'switch': r'换(个|一个|种)(思路|角度|方法)',
    'oh_no': r'哦[，,]?\\s*不',
    'backtrack': r'回(过头|头)来?(看|想)',
}


def has_revision(text: str) -> bool:
    import re
    return any(re.search(p, text) for p in REVISION_PATTERNS.values())
