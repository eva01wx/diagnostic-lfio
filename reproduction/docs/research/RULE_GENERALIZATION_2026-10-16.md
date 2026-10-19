# 显式规则泛化：实现、结果与使用

已实现第二种泛化：对已学到的规则删除条件、合并规则和消除冗余。
它独立于原来的候选规则搜索；原 2026-10-15 冻结算法和实验源码未改写。

## 可用实现

- `src/rule_generalization.py`：数值阈值 DNF、布尔简化、全域等价验证、条件删除、
  区间包围合并、累计危害约束和单独校准。
- `src/lfio_generalized.py`：`GeneralizedLFIO` 可直接训练并输出泛化后规则，
  提供 `predict`、`get_rules`、`explain` 和包含证书的 `export`。
- `run_rule_generalization.py`：新划分完整实验；`generate_rule_generalization.py`：
  证书、路径、校准与预测重放，生成表格和图。

默认 `mode='exact'`。每个原始特征的条件化为开下界、闭上界；证明整个程序的
覆盖区间相同后才接受压缩。证明资源耗尽为 unknown，不作为成功。保证是对原代理
的全域预测等价，不是对教师正确性、因果机制或全局最短描述的证明。

`mode='calibrated'` 允许扩大覆盖，需另一个校准分区。路径只在拟合数据上生成；
相对原模型累计核算危害。保持相同头类别，使新增错误只可能发生在教师的默认类别。
默认两类平衡误差容差 0.02、每条路径 delta=0.05，候选数 K 的单侧二项上界用
delta/K 分配。不通过时保留全域等价压缩模型。

概率解释要求原始学习/路径与校准样本独立，并要求类内 iid；原始行不重叠是必要的
工程检查，并不自动证明这些抽样假设。保证不覆盖分布漂移、单类测试折的宏观评分，
也不是整项研究所有模型同时 95% 有效，更不是每个有限测试折绝不超过两百分点。

```python
from src.lfio_generalized import GeneralizedLFIO

# y_* 是同一个已训练教师的输出；四个分区须预先隔离。
model = GeneralizedLFIO(
    literal_budget=8, mode="calibrated", epsilon=0.02, delta=0.05,
)
model.fit(
    X_fit, y_fit, X_selection, y_selection,
    X_calibration=X_calibration, y_calibration=y_calibration,
    feature_names=feature_names,
)
rules = model.get_rules()
prediction = model.predict(X_test)
trace = model.explain(X_test[0])
record = model.export()
```

仅需等价压缩时用 `mode="exact"`，不传校准数据。对已有规则可直接使用
`RuleProgram.from_lfio(export, n_features)` 和 `RuleGeneralizer`，无需重训教师。
`explain` 是可执行预测路径，不是因果解释。

## 完整对照结果

新增种子 7001--7003；同六数据集、两教师、两噪声；三个基模型、三个预算。
共 72 教师设置、648 基模型/预算组合、3,888 结果行。训练、选择、校准、测试约为
60/10/10/20，原始重复行分组隔离。此为已知数据集的新划分扩展，不作为新增外部数据。
所有策略共享相同原始基模型，近似策略共享同一拟合专用路径，校准选择先落盘再测试。

| 泛化策略 | 平均逐模型条件减少 | 测试宏观保真度变化（百分点） | 减少条件的组合 |
| --- | ---: | ---: | ---: |
| 布尔吸收／互补合并 | 0.00% | 0 | 0/648 |
| 数值语义等价压缩 | 0.12% | 0 | 7/648 |
| 仅拟合样本约束 | 18.94% | -0.259 | 365/648 |
| 经验校准门槛 | 16.26% | -0.181 | 324/648 |
| 同时上界校准 | **4.78%** | **-0.017** | **103/648** |

同时上界策略条件减少的描述性数据集簇区间为 [0.12%, 10.26%]；保真度变化区间为
[-0.058, +0.012] 个百分点。它选择 99 个非等价泛化模型，其他情况回退到等价阶段。
经验门槛与仅拟合约束分别有 36、43 个双类测试组合损失超过两百分点；同时上界策略
观察到 0 个。该观测不等于所有未来样本都受到确定性约束，也不证明保真度非劣。

等价压缩的改善有限，说明已有短规则中可直接消除的冗余较少。主要压缩来自受控扩大覆盖，
不能将两阶段合并宣传成“无损减少 4.78%”。经典算子与 Clopper--Pearson 界不是新发明；
当前可核验贡献是数值语义验证、可重放的显式泛化及其取舍证据。
没有据此声称更高的人类理解度、结构稳定性或普遍优于其他规则学习器。

## 复现

```bash
python -m pytest -q tests/test_rule_generalization.py tests/test_lfio_generalized.py
python generate_rule_generalization.py
```

新训练必须在单独目录保留原记录，先运行 `run_rule_generalization.py --prepare-only`，
再运行 `run_rule_generalization.py --workers 16`。源、配置和数据在首次运行前锁定；
不要改写 `results/rule_generalization_20261016` 的已完成结果来容纳新策略。

表格、区间、图、逐模型指标及审计位于 `rule_generalization_2026-10-16/`。
审计重放 3,888 个测试预测向量、1,232 个等价步骤和 544 个近似步骤，重新计算校准界。
