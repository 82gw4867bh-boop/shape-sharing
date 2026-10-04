# M0：四域分别拟合的普通混合效应基准

当前 `results_from_lasso_0p5/` 为正式参考基线 v2。上游原 Lasso α=0.5 筛选基于训练患者的五份 MICE；最终从已保存的五份中按**内部验证四域等权平均 RMSE**选择第 3 份，作为唯一 M0 输入。此阶段没有重新拟合 MICE，也没有更改入选的 25 个原始字段。选择记录在上游 `draw_comparison_internal.csv` 和 `draw_selection_decision.json`。单份插补的内部验证性能因用于选份而有乐观性。

M0 只读取 [`../variable_screening/results_form1_m0_lasso_0p5/`](../variable_screening/results_form1_m0_lasso_0p5/) 中的 manifest、数据矩阵和逐域清单，验证哈希、划分、列、结局有效性后，按四域分别拟合 `observed ~ t + S_d 主效应 + (1|UniID)`；`t=(BYear-1)/5`，1 年为参考点。每域只拟合**一**个随机截距模型，预测用固定效应，不用新患者未来 CHART 估计随机效应。四域不求和，不拟合共享函数、领域效应、互作或随机斜率。

```powershell
python -m m0.mixed_effects_m0_template variable_screening/results_form1_m0_lasso_0p5 m0/results_from_lasso_0p5
python -m variable_screening.temporal_validation NSCISC_Public_Data_2021.zip variable_screening/results_form1_m0_lasso_0p5 m0/results_from_lasso_0p5 m0/temporal_validation_lasso_0p5_2001_2004
```

`summary_*.txt` 和 `run_metadata.json` 记录每域参数、收敛、随机截距和残差方差；`predictions.csv`、`metrics.csv`、`predicted_changes.csv` 记录固定效应预测、逐节点误差与节点间变化。单次 MICE 没有跨插补 Rubin 合并；`fixed_effects_se` 是该次混合模型条件标准误，未涵盖插补、筛选和选份不确定性。`imputation_prediction_sd` 在单份预测中无定义，输出为空。时间留出组只用原训练集插补器的第 3 份随机种子和已拟合固定效应，报告见 [`temporal_validation_lasso_0p5_2001_2004/validation_report.md`](./temporal_validation_lasso_0p5_2001_2004/validation_report.md)。

训练、内部验证及时间评价均只用伤后 1、5、10、15 年真实有效 CHART。缺失结局不填作真值；死亡后和死亡同年先后不明者不参与误差。20 年节点继续封存。当前 164 对字段关系表已完成定义审查并冻结，作为后续 M2 提出候选关系的资料，M0 不使用这些标签。详见 [`../variable_screening/PAIR_REVIEW_GUIDE.md`](../variable_screening/PAIR_REVIEW_GUIDE.md)。
