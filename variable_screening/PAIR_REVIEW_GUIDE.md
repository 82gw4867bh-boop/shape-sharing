# 25 个入选 Form I 字段的临床关系审查与冻结

2026-10-04 按当前最终 M0 入选字段的并集，对 17 个连续、8 个分类字段分别做无序同类型配对，共 **164 对**。`results_form1_m0_lasso_0p5/variable_pair_review.csv` 已全部审查：L3 15 对、L4 101 对、无已知关系 48 对；L1/L2/待核查均为 0。`variable_relation_table.csv` 只包含 L3/L4 的 116 对。逐行 `evidence` 写出两字段的官方字典页码、定义、判定理由及需要时的研究链接。`variable_empirical_associations.csv` 仅为训练集描述，不决定关系等级。

| 等级 | 当前判定原则 |
|---|---|
| L1 | 同一信息的重复字段；若出现需先在资格阶段去重。当前未发现。 |
| L2 | 两字段可由明确公式直接转换，或有适合本数据的验证换算。当前未发现。 |
| L3 | 同一或相近构念而不能由单字段互换，如相邻 ASIA 同侧肌群、同一 FIM 子领域活动、FIM 运动总分与其中单项。总分由 13 项合成，单项不能单独反推总分。 |
| L4 | 临床有关但构念不同，如上/下肢肌力、神经肌力与 FIM 任务表现、照护时间与出院功能、性别与膀胱管理方式。 |
| 无已知关系 | 已核对定义和可用来源，未找到足以预设 L1–L4 的直接关系；不代表统计独立或因果无关。 |

官方定义来自 NSCISC 2021 公开数据字典（`reference/Data_Dictionary_Public_AR2021.txt`，项目 ZIP 随包）；[NSCISC 官方数据库页](https://sites.uab.edu/nscisc/database/)给出字典与表单入口。[ASIA 官方 ISNCSCI 资源](https://asia-spinalinjury.org/isncsci-resources/)说明分节段运动检查。[SCI 队列的性别与泌尿管理研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC7614058/)、[教育与劳动力参与研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC10070183/)及[保险/种族差异研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC9754933/)用于限定少数 L4 社会/护理关联。来源只支持构念关系，不能推出本队列的效应大小。

`clinical_pair_review.py` 保存判定规则及逐行证据，`variable_pair_review_signedoff_2026_10_04.csv` 是机器可复现的冻结审查副本；**没有独立临床专家签核**。manifest 的 `analysis_status=frozen` 表示当前字段、编码、单份插补选择及关系规则作为项目 M0/M2 比较基准固定；它不声称 M0 预测已获独立数据库验证。后续发现字典错误或新证据，应写明原因另建版本，不覆盖当前判定。关系表不自动转换单位、不删除 L3/L4 字段、不改变 M0 公式。均值不同不能直接视为单位不同。
