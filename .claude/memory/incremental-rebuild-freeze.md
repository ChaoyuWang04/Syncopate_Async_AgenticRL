---
name: incremental-rebuild-freeze
description: 增量重建数据时，有全局统计参与的产物会顺手改掉不相关的部分——必须冻结旧的，只算新的
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 957ae9f2-2820-4a54-ab6d-75be32051e25
  modified: 2026-08-14T13:09:51.809Z
---

**2026-08-14（M8）**：只想加两个模板，重算工具菜单却**动了 1030/1370 条存量 case 的 prompt**。

**Why**：菜单 = gold 并集 ∪ **SFT 审计里模型实际调过的工具** ∪ **CORE（出现在 ≥4 个模板 = 全局统计）**。
后两项都不是本地的 —— 换一份 `--sft-audit`、或者只是**多了两个模板**，
都可能让某个工具（实测是 `metrics.get_freshness`）掉出 CORE，
从而改掉一堆和本次改动无关的模板的输入分布 ⇒ **它们的历史基线全部作废**。

⚠️ 我当时的判断是「`tool_menu` 是逐 case 存的，所以只有用得上新工具的 case 会变」。
**错在：逐 case 存 ≠ 逐 case 算。**

**How to apply**：
1. 增量里程碑用 `scripts/set_tool_menus.py --freeze-from <上一版 batch>`：
   存量模板**逐字节沿用**旧菜单，只给新模板现算。
   修完实测 1370 条输入逐字节相同、0 条变化。
2. **"哪些基线仍可比"要是构造保证的，不是事后 diff 出来的。**
3. 一般化：**凡是有全局统计参与的产物（CORE、词表、阈值、分位数、归一化常数），
   增量重建时都要先问「我这次改动会不会顺手改掉不相关的部分」。**
   问法是跑一次逐字节比对，而不是推理。

相关：[[project-mechanism-not-wired]] [[feedback-measure-dont-infer]]
