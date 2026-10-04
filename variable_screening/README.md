# Form I 资格审查、Lasso 筛选与单份插补交接

## 当前冻结版本

`results_form1_m0_lasso_0p5/` 是 M0 参考基线 v2 的唯一上游目录。1995–2000 伤年成年患者按 ID 固定种子 `20261003` 分为 80% 训练与 20% 内部验证，2001–2004 伤年患者为时间留出评价，20 年节点封存。第一阶段从 658 个 Form I 原始字段建立 125 个候选字段，并排除 `AASAC6RL`、`AASAL3RL`、`AASAS1RR`。四域各用固定 α=0.5 的 Lasso；筛选只使用训练患者，其患者级五折及 20 次子抽样均在折/子样本内拟合连续变量 MICE。分类缺失保留 `__MISSING__`，稀有类别映射 `__OTHER__`。

原筛选生成五份训练集拟合的 MICE 完成数据，稳定性频率的分母为 20 次子抽样 × 5 份插补。用户随后决定**沿用已有结果**：`select_existing_draw.py` 只用内部验证中四域 RMSE 的等权平均，在已保存的五份中选出第 3 份；不重新拟合 MICE、不重选变量。四域入选字段数依次为 19、15、13、8，去重后 25 个。最终目录只有一份 `prepared_baseline.csv`、一份 `model_ready_long.csv` 和训练集拟合的 `mice_imputer.joblib`；manifest 同时记载原筛选 `screening_mice_imputations=5` 和最终 `mice_imputations=1`、`selected_mice_draw_index=2`（从零开始）。`draw_comparison_internal.csv` 与 `draw_selection_decision.json` 保留选份依据。

`clinical_pair_review.py` 根据官方字典及所引临床研究，逐对审查 164 对同类型入选字段，生成 `variable_pair_review.csv` 和 `variable_relation_table.csv`。分布为 L3 15 对、L4 101 对、无已知关系 48 对；没有 L1、L2 或待核查。这里的 `frozen` 是项目对字段定义和关系规则的冻结，`variable_pair_review_signedoff_2026_10_04.csv` 的文件名不代表独立临床专家已签字。具体判定见 [`PAIR_REVIEW_GUIDE.md`](./PAIR_REVIEW_GUIDE.md)。关系标签只供后续 M2 提候选，不进入 M0 公式；无已知关系不等于统计独立。

## 从已有五份产物复现当前版本

当前项目保留最终单份结果；以下命令说明其生成链。要从零重建，先准备空的输出目录，再按历史配置运行上游五份筛选与四域五份 M0，随后从五份中选一份。日常下游工作直接使用最终目录，不重新运行 MICE。

```powershell
python -m variable_screening.pipeline NSCISC_Public_Data_2021.zip variable_screening/rebuild_five_draw
python -m m0.mixed_effects_m0_template variable_screening/rebuild_five_draw m0/rebuild_five_draw
python -m variable_screening.select_existing_draw variable_screening/rebuild_five_draw m0/rebuild_five_draw variable_screening/results_form1_m0_lasso_0p5
python -m variable_screening.clinical_pair_review variable_screening/results_form1_m0_lasso_0p5
python -m m0.mixed_effects_m0_template variable_screening/results_form1_m0_lasso_0p5 m0/results_from_lasso_0p5
python -m variable_screening.temporal_validation NSCISC_Public_Data_2021.zip variable_screening/results_form1_m0_lasso_0p5 m0/results_from_lasso_0p5 m0/temporal_validation_lasso_0p5_2001_2004
```

`screening_config.yaml` 保留原五份筛选配置，用于完整重现历史筛选；最终单份的选择和哈希在 `feature_manifest.json` 中。当前最终目录由已有五份产物提取，所以这些命令的前两步本轮没有重跑。`ASex` 男/女指示作为连续插补辅助，不按性别拆分拟合。观测连续值保持原值，缺失插补值限制在训练观察和字典范围内。时间评价仅用已训练处理器变换 Form I，使用保存的第 3 份随机种子，不用测试组拟合插补器或选份。

## 代码边界与输出

- `pipeline.py`、`eligibility.py`、`screening.py`、`preprocess.py` 构成原筛选层：读取原始 ZIP、审查候选资格、清理特殊码、患者级 Lasso、折内 MICE，输出模型就绪数据和哈希。Form II 只提供开发结局，不作为出院输入。
- `select_existing_draw.py` 读取已有五份数据及已拟合 M0 参数，只用内部验证结果选一份并复制数据、插补器和清单；不会重新拟合 MICE。
- `clinical_pair_review.py` 按明确的字典定义和来源给当前同类型配对定级，更新关系表、计数、manifest 状态及哈希；不会依据相关系数自动合并字段。
- `temporal_validation.py` 在独立入口加载最终处理器和 M0 固定效应，对 2001–2004 伤年患者只做变换、预测与真实有效 CHART 评分。

当前上游只有训练/内部验证的 1、5、10、15 年模型网格。缺少真实 CHART 的行不参与误差；死亡后及死亡同年先后不明者不进入评价。四域有效行分别保留，不要求四域四节点完整。不同年代的已观察均值和缺失率见项目索引中的审计目录；均值差异本身不构成单位换算依据。内部验证被用于挑选第 3 份，因此最终内部 RMSE 具有选择乐观性；时间组只按既定模型评分，是同一数据库内的伤年留出评价。
