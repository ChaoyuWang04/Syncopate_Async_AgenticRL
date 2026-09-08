# 2026-09-08 · 真实训练资源只读盘点

Volume：`syncopate-home`。用Modal文件API递归列`/models`、`/data`、`/hf_cache`，只回读小型配置、权重索引、缓存revision和dataset_info。没有GPU申请、模型下载或删除。精确UTC/配置SHA/索引SHA/分片大小见[机读索引](20260908-resources.json)。

| 模型目录（/vol/models/下） | 当前用途 | 索引分片数 / 非空文件数 | 已列权重文件逻辑大小 |
|---|---|---|---|
| Qwen3.6-35B-A3B | 主MoE | 26 / 26 | 71.90 GB |
| Qwen3.8-27B | 主dense | 18 / 18 | 55.56 GB |
| Qwen3.5-27B | 库存，不纳入主模型变量 | 11 / 11 | 55.56 GB |
| Qwen3.5-9B | 库存，不纳入主模型变量 | 4 / 4 | 19.31 GB |
| Qwen3.5-4B | 库存，不纳入主模型变量 | 2 / 2 | 9.32 GB |
| Qwen3.5-0.8B | 库存，不纳入主模型变量 | 1 / 1 | 1.75 GB |

固定优先复用前两项：MoE的text_config为`qwen3_5_moe_text`、256专家/top-k8；dense为`qwen3_5_text`。两者都含linear_attention/full_attention，不能称为纯attention MoE/dense单因素对照。缓存revision分别为`995ad96eacd98c81ed38be0c5b274b04031597b0`和`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`，与各config的下载metadata一致。

**本次只证明索引引用的分片存在且非空，不能证明每个权重字节完整、所有分片同revision或当前框架能加载。** 开跑前绑定/复核完整身份与模型支持，优先修缺失项，不重新下载整模。其余模型只是库存，不增加第三个主模型，也未核对其他任务引用，不清理。

/data已列内容属于旧v16；HF datasets缓存有四份匿名parquet，元信息不足以确认公开来源。此范围内未识别出新的function-calling/长程公开数据，不能说整个账户没有；未扫描其他Volume或读取所有原始轨迹。旧业务不恢复，公开数据在场景准备时检查许可/revision/形状，再在Modal CPU下载必要subset。

GB是十进制逻辑文件字节，不是Volume物理占用或账单；目录size不是递归总大小。
