# 项目内容索引

| 内容 | 本地路径 | GitHub |
|---|---|---|
| 项目背景与分析边界 | [`AGENTS.md`](./AGENTS.md) | — |
| 当前研究决策与冻结记录 | [`decisions/README.md`](./decisions/README.md)、[`decisions/2026-10-04-selected-draw-freeze.md`](./decisions/2026-10-04-selected-draw-freeze.md) | [GitHub 决策记录](https://github.com/82gw4867bh-boop/shape-sharing/tree/master/decisions) |
| 变量筛选与变量关系映射方案 | [`variable_screening/variable_screening_framework.md`](./variable_screening/variable_screening_framework.md) | [GitHub 方案](https://github.com/82gw4867bh-boop/shape-sharing/blob/master/variable_screening/variable_screening_framework.md) |
| Form I 上游骨架说明、代码边界、关系审查与配置 | [`variable_screening/README.md`](./variable_screening/README.md)、[`variable_screening/PAIR_REVIEW_GUIDE.md`](./variable_screening/PAIR_REVIEW_GUIDE.md)、[`variable_screening/pipeline.py`](./variable_screening/pipeline.py)、[`variable_screening/screening_config.yaml`](./variable_screening/screening_config.yaml) | [GitHub 上游说明](https://github.com/82gw4867bh-boop/shape-sharing/blob/master/variable_screening/README.md) · [筛选代码](https://github.com/82gw4867bh-boop/shape-sharing/blob/master/variable_screening/pipeline.py) |
| 变量资格审查使用的官方字典文本 | [`variable_screening/reference/`](./variable_screening/reference/) | — |
| M0 四域分别拟合骨架说明与代码 | [`m0/M0_README.md`](./m0/M0_README.md)、[`m0/mixed_effects_m0_template.py`](./m0/mixed_effects_m0_template.py) | [GitHub M0 说明](https://github.com/82gw4867bh-boop/shape-sharing/blob/master/m0/M0_README.md) · [M0 代码](https://github.com/82gw4867bh-boop/shape-sharing/blob/master/m0/mixed_effects_m0_template.py) |
| 原始 2021 公开版数据与随包官方 PDF | [`NSCISC_Public_Data_2021.zip`](./NSCISC_Public_Data_2021.zip) | — |

当前正式 M0 参考基线 v2 的输入与结果位于 [`variable_screening/results_form1_m0_lasso_0p5/`](./variable_screening/results_form1_m0_lasso_0p5/) 和 [`m0/results_from_lasso_0p5/`](./m0/results_from_lasso_0p5/)。原筛选使用训练集五份 MICE；最终从已保存结果中按内部验证四域等权 RMSE 选第 3 份作唯一 M0 输入，未重拟合 MICE。25 个入选字段的 164 对关系已逐对作定义和临床来源审查，manifest 状态为 `frozen`；独立临床专家尚未签核。

正式 M0 基线的 2001–2004 年时间外部评价位于 [`m0/temporal_validation_lasso_0p5_2001_2004/validation_report.md`](./m0/temporal_validation_lasso_0p5_2001_2004/validation_report.md)，逐节点汇总在 `metrics_by_node.csv`。处理器与模型只在 1995–2000 训练患者中拟合，时间组仅用于变换、预测和与真实结局评分；它是同一数据库的伤年留出评价，不等于独立数据库验证。20 年节点仍封存。GitHub 版本不包含逐患者预测明细。

LASSO 0.5 旧入模 Form I 字段的患者级缺失审计、跨年代同字段均值对照和插补方案草案见 [`variable_screening/missingness_audit_lasso_0p5_2026_10_04/report.md`](./variable_screening/missingness_audit_lasso_0p5_2026_10_04/report.md)，含训练、内部验证及 2001–2004 时间组的逐字段、逐性别、逐伤年 CSV。此审计不读取随访结局；其写作时尚未实施插补，现已由下文新运行更新，但规则仍未冻结。

三项康复入院运动评分的缺失机制、Form II 随访对照及左右合并审查见 [`variable_screening/missingness_audit_lasso_0p5_2026_10_04/rehab_motor_followup_audit.md`](./variable_screening/missingness_audit_lasso_0p5_2026_10_04/rehab_motor_followup_audit.md)。Form II 仅用于解释性审计，不进入出院预测插补；报告也核对了出院时运动总分的可用性。

基线选择、单份插补、关系冻结和重跑结果见 [`decisions/2026-10-04-selected-draw-freeze.md`](./decisions/2026-10-04-selected-draw-freeze.md)。复现步骤见 [`variable_screening/README.md`](./variable_screening/README.md) 与 [`m0/M0_README.md`](./m0/M0_README.md)。

当前清单 25 字段的配对缺失率、逐性别分布和跨年代均值表见 [`variable_screening/missingness_audit_lasso_0p5_mice_exclude_rehab3_2026_10_04/`](./variable_screening/missingness_audit_lasso_0p5_mice_exclude_rehab3_2026_10_04/)；变量清单来自原五份筛选，选定单份只改变最终 M0 输入。
