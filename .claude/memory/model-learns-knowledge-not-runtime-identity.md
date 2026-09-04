---
name: model-learns-knowledge-not-runtime-identity
description: ★Chaoyu 09-02 立的界限：模型脑子里装知识与策略，不装运行态身份/动态清单；这类信息由 runtime 注入；数据必须逐条渲染给人看才能抓到
metadata:
  type: feedback
---

**模型该学的是知识和策略（什么时候调什么工具、该怎么想），不是运行态的动态信息。**
account_id 这类身份来自登录态，和 run_id 同类：不进 prompt、不进工具 schema、不进 gold，
由沙盒/收口按当前租户注入，**模型填了也覆盖**（安全）。campaign 清单几万条每天在变，
不进提示词，靠 campaign.list 自己翻。落地：`contract.RUNTIME_INJECTED_PARAMS` 一处定义。

**Why：** 09-02 Chaoyu 逐条看训练数据画廊（`scripts/v16_data_gallery.py`）抓到四条我自己的
脚本判据没抓到的问题：闲聊行空 think 块有梯度、压舱行列字段清单、context 塞 7 条 campaign 清单
（08-20 演示补丁被当产品形状）、WIN 行 8 轮全进 prompt 而 gold 说"看不到"（=教撒谎）。
脚本只能抓被告知要抓的东西；**数据必须逐条渲染成人能读的样子**，元信息行就是判据。

**How to apply：**
1. 任何新进 prompt 的字段先问：这是知识/用户输入/环境事实（时间），还是运行态身份/动态清单？后者一律注入。
2. 每次建库后出画廊（每桶抽样、⟦⟧ 标监督 token、空块有梯度/折叠历史/字段清单/纯日期四项标记）给 Chaoyu 过目。
3. 原话「这一版就是最终版，改好为止」——发现设计错误不许拖到下一版。
相关：[[train-data-must-match-production-shape]] [[v15-contract-refactor]]
