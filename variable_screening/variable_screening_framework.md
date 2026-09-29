# 纵向多任务 Symbolic Regression 项目：变量筛选与变量关系映射实施方案

## 1. 目的与适用范围

本文档用于指导纵向多任务 Symbolic Regression（SR）项目中的变量筛选流程。

本方案只处理以下问题：

1. 如何从原始数据库中筛选出“有资格进入建模流程”的变量；
2. 如何建立允许存在相关变量的候选池；
3. 如何在候选池内部建立变量字典与变量关系表；
4. 如何使用机器学习筛选方法，从候选池进一步筛出真正进入 SR 的变量；
5. 如何处理数学可转换变量、临床意义相似变量及多重共线性；
6. 如何将第二层筛选后的变量交给不同预测任务的 SR 模型。

本文档**不负责**：

- repeated cross-validation；
- bootstrap；
- 外部验证；
- 最终训练/验证划分；
- 模型性能评估；
- 稳定性选择；
- shape sharing 本身的具体数学算法。

这些内容属于后续训练、验证和模型结构发现阶段。

---

# 2. 总体原则

整个变量筛选流程采用“两阶段筛选”。

## 第一阶段：Eligibility Screening

目标：

> 判断一个变量是否“有资格”进入候选池。

这一阶段主要依据：

- 数据质量；
- 时间逻辑；
- 信息泄漏风险；
- 临床合理性；
- 可获得性；
- 数据定义一致性。

这一阶段**不因为变量之间相关性高而简单删除变量**。

输出：

> Candidate Variable Pool（候选变量池）

例如：

```text
原始数据库变量
      ↓
Eligibility Screening
      ↓
70 个候选变量
```

这 70 个变量允许存在：

- 高相关变量；
- 同一临床构念的不同测量方式；
- 数学相关变量；
- 不同历史版本的量表；
- 可能互为代理变量的指标。

---

## 第二阶段：Predictive Screening

目标：

> 从候选池中筛出真正进入某一个预测任务 SR 模型的变量。

推荐使用：

- Elastic Net；
- 或其他带正则化的机器学习方法；
- 若需要严格考虑纵向重复测量结构，可进一步使用 penalized mixed-effects / longitudinal-aware regularization。

输出：

```text
Candidate Pool
例如 70 variables
      ↓
Predictive Screening
      ↓
Task-specific selected variables
      ↓
进入 SR
```

注意：

> 第二阶段允许不同预测任务得到不同的变量子集。

例如：

```text
Task A: 18 variables
Task B: 15 variables
Task C: 21 variables
Task D: 17 variables
```

不要求四个任务提前共享相同变量集合。

---

# 3. 第一阶段：Eligibility Screening

## 3.1 第一阶段的核心问题

对每个原始变量，只问：

> 这个变量是否有资格参与本研究？

而不是：

> 这个变量当前是否与 outcome 显著相关？

第一阶段不建议使用：

- 单因素 P 值；
- 单因素相关性；
- 与 outcome 的单变量显著性；
- LASSO；
- SR；
- 最终模型表现。

避免把模型结果提前带入变量资格判断。

---

# 4. 第一阶段的建议筛选规则

## 4.1 删除身份识别和管理字段

例如：

- patient ID；
- record ID；
- 数据库内部流水号；
- site ID（若无分析意义）；
- 数据录入编号；
- 数据管理字段；
- 文件版本号；
- 审核状态码。

这些变量通常不具有预测学或临床解释意义。

---

## 4.2 删除明显 outcome leakage 变量

任何只有在预测时间点之后才能获得，或直接编码结局的信息，都不应进入候选池。

例如：

- 最终出院状态；
- 最终死亡结果；
- 明确由未来事件产生的指标；
- 后续随访才能获得、但模型预测时间点尚不可见的信息；
- 由最终 outcome 直接推导出来的变量。

原则：

```text
Predictor information time
必须 ≤ prediction time
```

---

# 5. 时间结构检查

对于纵向研究，所有变量必须明确其时间属性。

建议至少分为：

## 5.1 Baseline variables

例如：

- 性别；
- 损伤原因；
- 初始损伤严重度；
- 初始神经功能；
- 基线年龄；
- 既往病史。

---

## 5.2 Time-varying variables

例如：

- BMI；
- 功能评分；
- 心理量表；
- 社会参与指标；
- 当前就业状态；
- 当前生活满意度。

这些变量可能在多个随访节点重复记录。

---

## 5.3 Derived variables

例如：

```text
BMI = Weight / Height²
```

此类变量应明确标注为 derived variable。

---

## 5.4 Historical-version variables

例如不同年代使用：

- FIM；
- SCI-FI；
- SCIM；
- PHQ 不同版本；
- SWLS 不同版本。

这类变量不能直接视为普通独立变量，需要进入后续 harmonization / relation mapping。

---

# 6. 数据质量筛选

第一阶段应至少检查：

## 6.1 缺失率

不建议机械采用一个固定缺失率阈值。

应同时考虑：

- 总体缺失率；
- 不同随访节点缺失率；
- 不同 subgroup 缺失率；
- 该变量是否属于核心临床变量；
- 是否存在合理插补方式；
- 是否只在少数年份收集。

建议输出：

| Variable | Overall missing | T1 | T2 | T3 | T4 | Keep? |
|---|---:|---:|---:|---:|---:|---|

---

## 6.2 低方差 / 无变化变量

例如：

- 99.8% 患者均为同一个值；
- 某变量在当前 cohort 内几乎不发生变化。

这种变量即使理论上有意义，也通常难以提供预测信息。

---

## 6.3 极低频分类水平

例如：

```text
Category A = 98%
Category B = 1%
Category C = 1%
```

可考虑：

- 合并；
- 重新编码；
- 删除极少数水平；
- 保留为二分类。

---

## 6.4 编码一致性

检查：

- 不同年代编码是否变化；
- 相同数字是否在不同版本代表不同含义；
- “unknown / not applicable / missing” 是否被混入正常数值；
- 不同随访节点变量定义是否改变。

---

# 7. 第一阶段不应因为相关性直接删除变量

这是本项目的核心原则之一。

例如候选变量中存在：

```text
BMI
Weight
Height
```

即使三者高度相关，也不应仅因为相关性而在 Eligibility Screening 中强行删除。

原因：

本项目后续需要研究：

- 不同任务是否选择不同代理变量；
- 不同变量是否属于同一个 construct；
- 是否存在数学可转换关系；
- 是否可以在不同任务之间发现 family-level shared relationship。

因此：

> 高相关变量允许共同存在于 Candidate Pool。

---

# 8. 但需要删除“纯重复变量”

需要区分：

## 8.1 高相关

例如：

```text
Weight
BMI
```

不是完全重复。

可以同时进入候选池。

---

## 8.2 完全重复 / 确定性重复编码

例如：

```text
Variable_A = Variable_B
```

或者只是：

```text
0/1 coding
与
No/Yes coding
```

完全表达同一字段。

这种应在第一阶段去重。

原则：

> 保留不同的信息表达；
> 删除完全重复的数据编码。

---

# 9. Candidate Variable Pool

完成 Eligibility Screening 后，形成冻结的候选池。

例如：

```text
Candidate Pool = 70 variables
```

Candidate Pool 是本研究中：

> 所有有资格被后续算法选择的变量全集。

后续原则上不应因为模型结果不好，而临时从候选池之外增加变量。

如果确实需要增加：

- 应有明确数据或方法学理由；
- 应记录为 protocol amendment；
- 应重新运行相关变量预处理流程。

---

# 10. Candidate Pool 冻结

建议在第二阶段筛选之前冻结以下内容：

```text
candidate_variables_v1.csv
```

包含：

- variable name；
- original name；
- label；
- domain；
- construct；
- data type；
- time structure；
- missingness；
- availability；
- inclusion reason；
- exclusion flag；
- notes。

---

# 11. Variable Dictionary

对候选池建立“一变量一行”的变量字典。

建议格式：

| Variable | Label | Domain | Family | Construct | Type | Time-varying | Derived | Availability | Notes |
|---|---|---|---|---|---|---|---|---|---|

示例：

| Variable | Domain | Family | Construct | Type |
|---|---|---|---|---|
| BMI | anthropometric | body composition | adiposity/body size | continuous |
| Weight | anthropometric | body composition | body size | continuous |
| FIM | function | functional independence | independence | score |
| SCI-FI | function | functional independence | independence | score |
| PHQ-9 | mental health | depression | depressive burden | score |

---

# 12. Variable Relation Table

变量关系表在 Candidate Pool 内建立。

它的目标不是为了提前限制 SR，而是用于：

1. 标记潜在可互变变量；
2. 标记变量之间的数学关系；
3. 标记 validated crosswalk；
4. 标记同一 clinical construct；
5. 为后续跨任务 shared-variable discovery 提供依据；
6. 解释不同任务选出不同变量时，是否可能属于同一 underlying mechanism。

---

# 13. Relation Table 不是全变量两两排列

如果候选池有 70 个变量：

```text
70 × 69 / 2 = 2415
```

不需要人工建立 2415 行。

只有存在明确关系的变量对才进入关系表。

---

# 14. 建议的变量关系等级

## L1：Identical / Duplicate Representation

完全相同的信息。

示例：

```text
同一个变量的不同字段名
同一个二分类变量的不同编码版本
```

通常此类应在第一阶段已经去重。

---

## L2：Deterministic / Validated Transformation

存在明确的数学转换或经过验证的转换关系。

例如：

```text
BMI ↔ Weight + Height
```

或者：

```text
FIM ↔ SCI-FI
```

如果存在 validated crosswalk。

---

## L3：Same Clinical Construct

变量不完全可数学互换，但主要测量同一临床构念。

例如：

```text
PHQ-9
PHQ-8
PHQ-2
```

均属于 depressive symptom burden。

---

## L4：Clinically Related Domain

变量临床意义相关，但不能直接视为同一个 construct 或直接替换。

例如：

```text
不同 body-composition 指标
不同功能维度
相关但不同的神经功能指标
```

L4 主要用于生成候选共享关系，而不是直接宣布变量等价。

---

# 15. Relation Table 建议格式

| Variable A | Variable B | Family | Strength | Basis | Transform available | Evidence |
|---|---|---|---|---|---|---|
| BMI | Weight+Height | body composition | L2 | mathematical | yes | BMI formula |
| FIM | SCI-FI | functional independence | L2 | validated crosswalk | yes | published crosswalk |
| PHQ-9 | PHQ-8 | depression | L3 | same construct | partial | psychometric relation |

---

# 16. 为什么关系表要在第二层筛选之前建立

这是为了避免：

> 看完模型结果以后，再人为解释两个不同变量“其实差不多”。

预先建立 Relation Table，可以降低：

- post-hoc cherry-picking；
-人为强行凑共享结构；
- 结果导向式变量解释。

原则：

```text
Relation Table must be defined
before task-specific SR results are observed.
```

---

# 17. 多重共线性的处理原则

本方案不要求在 Candidate Pool 中彻底消除高相关变量。

需要区分：

## Candidate Pool level

允许：

```text
BMI
Weight
Height
```

共同存在。

---

## Predictive Screening level

由第二层正则化方法处理：

- 冗余；
- 高相关；
- 无贡献变量；
- 变量数量缩减。

---

## Final SR level

只接收第二层筛选后的变量。

因此：

> 高相关变量可以存在于候选池；
> 不意味着它们最终一定会一起进入 SR。

---

# 18. 第二阶段：Predictive Screening

第二阶段的目标：

> 从 Candidate Pool 中筛出对某个预测任务具有实际预测价值的变量。

这一阶段才开始使用 outcome 信息。

---

# 19. 不推荐的第二阶段方法

不建议采用：

```text
单因素回归
→ P < 0.05
→ 进入模型
```

原因：

- 会漏掉联合模型中有价值的变量；
- 对高度相关变量不稳定；
- 容易产生 selection bias；
- 不适合后续非线性 SR。

---

# 20. LASSO 的定位

LASSO 可以做：

- shrinkage；
- variable selection；
- 高维变量缩减。

但在高度相关变量中：

```text
X1 ≈ X2
```

LASSO 可能：

```text
Run A → X1
Run B → X2
```

因此它适合作为基础方法，但不是本项目最优默认选择。

---

# 21. 推荐优先考虑 Elastic Net

Elastic Net 综合：

```text
L1 penalty
+
L2 penalty
```

优势：

- 能产生变量筛选；
- 对高度相关变量通常比纯 LASSO 更稳定；
- 对变量组更友好；
- 适合 Candidate Pool 中保留相关变量的设计。

因此建议：

> 将 Elastic Net 作为第二阶段的主要基线筛选方法。

---

# 22. 如果严格考虑纵向结构

如果第二阶段使用的是纵向重复测量数据，需要注意：

```text
同一 patient 的多次随访
并不是相互独立的样本。
```

因此严格版本可以考虑：

```text
Penalized Mixed-Effects Model
```

或：

```text
Longitudinal-aware regularized model
```

其基本目标是：

```text
Outcome
~ fixed effects
+ patient-level random effect
```

同时对 fixed effects 做正则化选择。

---

# 23. 第二层筛选是否必须用 Mixed-Effects LASSO/Elastic Net

不一定。

建议将其分成两个实现级别。

## Version A：简化实现

```text
Elastic Net
```

用于快速缩减特征。

前提：

- 预处理合理；
- patient-level independence 在实现中被谨慎处理；
- 主要目的只是 feature screening，而不是最终推断。

---

## Version B：正式实现

```text
Penalized mixed-effects / longitudinal regularized screening
```

优先考虑纵向相关结构。

推荐在正式主分析中使用这一类方案。

---

# 24. 四个预测任务分别筛选

假设有四个预测任务：

```text
Task A
Task B
Task C
Task D
```

共同使用同一个 Candidate Pool：

```text
70 variables
```

但分别进行第二阶段筛选：

```text
Task A:
70 → screening → 18

Task B:
70 → screening → 15

Task C:
70 → screening → 21

Task D:
70 → screening → 17
```

不要求四个任务筛出相同的变量。

这一步应允许：

> task-specific variable selection。

---

# 25. 不在第二阶段强行要求 family-level exclusivity

当前方案不采用：

```text
同一 family 最多只能选一个变量
```

作为硬性限制。

原因：

1. 第二层正则化本身会抑制大量冗余变量；
2. 不希望人为过度干预变量选择；
3. 某些相关变量可能仍包含不同信息；
4. 需要让不同任务自由选择不同 proxy representation；
5. 后续可通过 Relation Table 判断这些不同变量是否属于同一 family。

---

# 26. 但完全数学重复变量仍应提前处理

注意：

```text
不设置 family exclusivity
```

不意味着：

```text
所有确定性重复变量都可以同时保留。
```

例如：

```text
BMI
Weight / Height²
```

如果两者完全相同，则没有必要同时作为独立候选变量。

因此：

> deterministic duplicate 应在第一阶段去重；
> high correlation 可以保留；
> same construct 可以保留。

---

# 27. 第二阶段输出

每个任务输出一个变量列表：

```text
task_A_selected_variables.csv
task_B_selected_variables.csv
task_C_selected_variables.csv
task_D_selected_variables.csv
```

建议至少包含：

| Variable | Selected | Coefficient | Standardized coefficient | Family | Notes |
|---|---:|---:|---:|---|---|

如果所用方法有不同输出，则按方法调整。

---

# 28. 进入 SR 前的最终检查

SR 输入变量应满足：

- 来自 Candidate Pool；
- 已通过第二阶段 predictive screening；
- 数据类型适合 SR；
- 缺失已处理；
- 分类变量已正确编码；
- 连续变量已按统一策略标准化；
- 时间变量处理方案明确；
- 变量定义跨 subgroup 一致。

---

# 29. SR 阶段原则

进入 SR 之后：

> 不再额外施加基于 family 的硬性变量限制。

即：

```text
Selected variables
      ↓
Free SR search
```

SR 可以自由发现：

- 主效应；
- nonlinear effect；
- 两变量 interaction；
- 三变量 interaction；
- 与时间的非线性关系。

---

# 30. Relation Table 在 SR 后的用途

SR 输出以后，再使用第一阶段建立好的 Relation Table。

假设：

```text
Task A:
Age × BMI

Task B:
Age × Weight

Task C:
Age × BMI

Task D:
Age × Weight
```

通过 Relation Table：

```text
BMI
Weight
↓
Body composition family
```

因此可以发现：

```text
Age × Body composition
```

可能是一个潜在的 shared interaction。

---

# 31. Relation Table 只产生“共享候选”，不直接证明共享

变量映射只能说明：

> 四个任务中的变量可能属于相同或相关构念。

不能直接证明：

```text
interaction mechanism is shared
```

因此后续还必须进行：

```text
variable equivalence
      ↓
candidate shared interaction
      ↓
shape comparison
      ↓
shared / partially shared / task-specific
```

---

# 32. 数学互变变量的使用规则

例如：

```text
BMI
Weight
Height
```

如果不同任务分别选择：

```text
A → BMI
B → Weight
C → BMI
D → Weight
```

不应简单宣称：

```text
BMI = Weight
```

而应表述为：

```text
这些变量属于同一 body-composition family，
存在已知数学或生理联系，
因此进入候选 shared interaction analysis。
```

只有在存在明确数学映射时，才能做正式转换。

---

# 33. 临床意义互变变量的使用规则

例如两个变量：

- 测量相近临床构念；
- 但不存在数学转换。

可以进入：

```text
same clinical construct
```

或：

```text
clinically related family
```

但不能直接视为数值等价。

后续应通过：

- shape similarity；
- clinical interpretation；
- sensitivity analysis；
- model fit；

判断是否真正共享。

---

# 34. Candidate Pool 之外的变量是否可以重新进入

默认规则：

> 不允许因为 SR 结果或模型性能而临时从 Candidate Pool 外拉回变量。

可以重新进入的情况应属于预先允许的特殊情况，例如：

1. 数据清洗后发现该变量原本被错误排除；
2. 原始变量定义理解错误；
3. 新发现 validated crosswalk；
4. 数据版本更新；
5. protocol amendment。

所有此类情况必须记录。

---

# 35. 推荐的项目文件结构

```text
variable_screening/
│
├── raw_variable_inventory.csv
│
├── eligibility_rules.md
│
├── eligibility_screening_results.csv
│
├── candidate_variable_pool.csv
│
├── variable_dictionary.csv
│
├── variable_relation_table.csv
│
├── screening_config.yaml
│
├── task_A_selected_variables.csv
│
├── task_B_selected_variables.csv
│
├── task_C_selected_variables.csv
│
└── task_D_selected_variables.csv
```

---

# 36. Candidate Variable Pool 推荐字段

```text
variable_name
original_variable_name
label
domain
family
construct
variable_type
time_varying
baseline_or_followup
derived_variable
historical_version
availability_start
availability_end
missing_rate
low_variance_flag
leakage_flag
eligible
inclusion_reason
exclusion_reason
notes
```

---

# 37. Relation Table 推荐字段

```text
variable_A
variable_B
family_A
family_B
shared_family
equivalence_level
basis
mathematical_transform
validated_crosswalk
clinical_rationale
evidence_reference
notes
```

---

# 38. 第二阶段配置建议

建议将第二阶段筛选参数写入独立配置文件：

```yaml
screening_method: elastic_net

standardize_numeric: true

categorical_encoding: one_hot

include_time: true

include_group: true

handle_missing_before_screening: true

task_specific_screening: true
```

正式项目可根据最终方法修改。

---

# 39. 不属于变量筛选阶段的内容

以下内容暂不放入本模块：

- repeated CV；
- bootstrap；
- seed repetition；
- test set；
- external validation；
- model calibration；
- confidence interval；
- selection frequency；
- stability selection；
- shape-sharing threshold；
- final symbolic-function evaluation。

这些内容在训练与验证模块单独规划。

---

# 40. 完整流程图

```text
RAW DATABASE
      │
      ▼
────────────────────────────
Stage 1: Eligibility Screening
────────────────────────────
      │
      ├── remove IDs / admin fields
      ├── remove leakage
      ├── check time validity
      ├── check missingness
      ├── check low variance
      ├── harmonize coding
      ├── remove exact duplicates
      └── retain clinically plausible correlated variables
      │
      ▼
Candidate Variable Pool
例如 70 variables
      │
      ├───────────────┐
      │               │
      ▼               ▼
Variable          Relation
Dictionary        Table
      │               │
      │               ├── L1 identical
      │               ├── L2 transform/crosswalk
      │               ├── L3 same construct
      │               └── L4 related domain
      │
      ▼
────────────────────────────
Stage 2: Predictive Screening
────────────────────────────
      │
      ├── Elastic Net
      │       or
      └── penalized longitudinal /
          mixed-effects model
      │
      ▼
Task-specific selected variables
      │
      ▼
────────────────────────────
Symbolic Regression
────────────────────────────
      │
      ├── main effects
      ├── nonlinear effects
      ├── interactions
      └── time-related structures
      │
      ▼
SR Results from 4 Tasks
      │
      ▼
Query frozen Relation Table
      │
      ▼
Candidate Shared Interactions
      │
      ▼
Shape Sharing Analysis
```

---

# 41. 核心原则总结

## 原则 1

第一层筛的是：

> 有没有资格进入研究。

不是：

> 有没有预测价值。

---

## 原则 2

Candidate Pool 中：

> 允许存在高度相关变量。

---

## 原则 3

完全重复、确定性重复编码：

> 第一阶段去除。

---

## 原则 4

数学相关、临床相关但不完全重复的变量：

> 可以同时留在 Candidate Pool。

---

## 原则 5

Relation Table：

> 在第二层模型筛选以及 SR 之前建立并冻结。

---

## 原则 6

第二层使用机器学习/正则化方法：

> 筛选真正进入 SR 的变量。

推荐优先：

```text
Elastic Net
```

或：

```text
Penalized mixed-effects / longitudinal regularization
```

---

## 原则 7

四个任务：

> 使用同一个 Candidate Pool，
> 但分别进行第二层筛选。

---

## 原则 8

进入 SR 以后：

> 不再人为规定同一 family 最多只能出现一个变量。

---

## 原则 9

SR 后利用 Relation Table：

> 识别“变量名称不同，但可能属于同一 underlying construct”的跨任务关系。

---

## 原则 10

变量关系相似：

> 只能产生 candidate sharing。

最终是否真正共享：

> 由后续 shape-sharing analysis 决定。

---

# 42. 推荐的最终一句话方法描述

可将整个变量筛选策略概括为：

> A two-stage feature selection strategy was used. First, a broad eligibility-based candidate pool was defined using prespecified rules regarding temporal validity, data quality, leakage prevention and clinical plausibility, while retaining potentially correlated but non-duplicate representations. A predefined variable dictionary and equivalence map were then constructed within this candidate pool. Second, task-specific regularized predictive screening was applied to reduce the candidate set before symbolic regression. The subsequent SR search remained unconstrained by variable-family membership, while the prespecified equivalence map was used only after model discovery to identify candidate cross-task shared relationships for downstream shape-sharing analysis.
