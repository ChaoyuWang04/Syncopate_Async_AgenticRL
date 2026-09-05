# Scripts archive

这里保存已经退出当前 v16 主线的历史脚本。归档只改变位置，不删除内容。

- `pre_v16_mainline/`：已经被 v16 数据、训练和考场入口替代的旧主线脚本。
- `legacy_4x5090/`：绑定旧 v11/v13 数据、旧模型路径或本地 4×5090 环境的批处理与实验脚本。

当前代码、测试和运行手册不应从这里调用程序。需要复盘旧实验时，可以按原文件名在对应子目录查找，并结合 git 历史还原当时环境。

原先仍被 v16 复用的 `u_*` 组件已脱离脚本区，分别收入
`syncopate/evaluation/`、`syncopate/pipeline/` 和 `syncopate/pipeline/materials/`；归档内剩余的
`u_*` 只是 pre-v16 历史实现。
