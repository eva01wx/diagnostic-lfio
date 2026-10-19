# 背景约束、成本选择与实例查询：实现及结果

算法源代码与实验复现材料公开于 [https://github.com/eva01wx/diagnostic-lfio/](https://github.com/eva01wx/diagnostic-lfio/)。

## 实际新增的贡献

1. **合法补全下的认证**：支持有理数差值约束、等式、顺序和单变量界；
   精确处理严格不等式。将已确认类别、确有歧义、证明未完成、无合法补全分别报告。
2. **不展开全局公式的查询**：固定保留值后，对剩余 DNF 检查两类补全，
   缓存相同剩余问题和精确有理数见证。存在工作限额时返回 UNKNOWN。
3. **覆盖率约束下的成本选择**：在指定选择集认证覆盖率和候选预算内，
   比较五种贪心策略，明确衡量隐藏多少活动特征成本。算法不保证最优子集。

逻辑恒等式和差值约束算法是既有基础；这里的贡献是数值语义、失败状态、
成本任务及可回放实现的组合。已加入 Wang/Brewster DATA 2024、Doherty/Szalas
AI 2024、Liberatore 2024/2025、DivKC 2026 和必要解释复杂性的文献对照。
DATA 2024 的图重要性启发任务影响评价，本实现没有把图评分当作形式证书。

## API

```python
import numpy as np
from src.contextual_forgetting import ContextualForgetter, DifferenceBound as B
from src.rule_generalization import Atom, RuleProgram
from src.forgetting_selection import select_features

source = RuleProgram(((Atom(0, 0., True),),), head=1, n_features=2)
# x0 = x1；None 表示常数零。隐藏 x0，只读取 x1。
model = ContextualForgetter().fit(source, [0], [B(0, 1), B(1, 0)])
assert model.predict([[np.nan, -1.], [np.nan, 1.]]).tolist() == [0, 1]
certificate = model.query([np.nan, 1.])
assert model.verify_query([np.nan, 1.], certificate)
# predicate: x_left - x_right <= bound；strict=True 改为 <。
infeasible = ContextualForgetter().fit(source, [0],
    [B(0, 1), B(1, 0), B(1, None, 1)])
assert infeasible.query([np.nan, 2.])['prediction'] == -2

# 成本选择实验使用无背景约束的查询。选择集与测试集分开。
selection = np.array([[-1., -1.], [1., 1.]])
choice = select_features(source, selection, costs=[4., 1.], targets=(.8,))
```

`prediction` 的 0/1 是类别，-1 为弃权，-2 为输入无合法补全。
两侧状态均 SAT 时才称为真实歧义，UNKNOWN 不等同于 UNSAT。
导出的是可重建的查询配置；认证必须调用 `verify_query` 重新验证。
隐藏列可为 NaN，也可使用 `retained_only=True` 只传保留列。

## 实验结果

- 复用六数据集的 **648 个 Original 代理**，没有重新训练，也没有增加独立数据集。
- 五策略、两种透明成本设定、三个覆盖率要求，共 **19,440** 条策略记录。
- 异质成本、选择集覆盖率要求 80%：CostAware 隐藏 **27.87%** 的活动特征成本，
  Coverage 隐藏 **25.90%**；配对提升 **1.97 个百分点**，描述性六数据集区间
  **[0.93, 3.10]**。测试认证覆盖率分别为 **89.91%** 和 **89.99%**。
- **22.2%** 的 CostAware 结果选择不隐藏任何活动特征；均值包含这些结果。
  **9.7%** 的拟合在测试集未达到 80% 要求；选择集约束不是部署覆盖率保证。
- 对旧版 **7,507** 个 Original 投影执行实例查询，旧版其中 **376** 个全局编译回退；
  实例查询本轮没有未完成投影，新增 **51,892** 次测试行认证，丢失 0 次。
  重复查询次数不构成独立样本量。
- 32 个刻意选择的大表示案例中，全局 L/U 有 **3,837–5,260** 个条件；
  Python 分配峰值中位数由 **1,528.7 KiB** 降至 **10.8 KiB**。
  这不是总进程内存测量，也不是代表性速度基准。
- 真实等式约束的合成正例将 0/5 认证恢复到 5/5；错误等式在真实域上造成 2/5 冲突，
  无合法补全对照被单独拒绝。另有 **900** 次随机约束查询与独立精确消元匹配。

完整数字见 `contextual_forgetting_2026-10-16/numbers.json`，选择记录和查询比较见
`policies.csv` / `queries.csv`。所有 **18,984** 个候选子集都重算选择覆盖率，
并在一行上完成独立阈值单元检查；旧投影另在前三个测试行独立检查。
原始完整输入上的已赋值结果没有来源代理冲突。

## 复现

使用已有 Python 3.12 环境及 `requirements-major-revision.txt`；新增模块无需新依赖。

```bash
python -m pytest -q tests/test_contextual_forgetting.py tests/test_feature_forgetting.py tests/test_rule_generalization.py tests/test_lfio_generalized.py
python generate_contextual_forgetting.py --workers 12
```

新执行应在单独副本中使用 `run_contextual_forgetting.py --prepare-only`，随后
`run_contextual_forgetting.py --workers 12`、`benchmark_contextual_forgetting.py`，
最后执行生成器。新记录位于 `results/contextual_forgetting_20261016/`。
源文件冻结方案见 `CONTEXTUAL_FORGETTING_PROTOCOL_2026-10-16.md`；单独计时的说明见
`CONTEXTUAL_BENCHMARK_NOTE_2026-10-16.md`。CPU 重复三次且关闭内存跟踪；分配峰值
保留原始跟踪测量，二者不混用。

目前仍缺少真实采集成本、经过数据字典核实的真实约束任务，以及人的理解实验。
证书始终针对给定代理与明确的补全域；背景知识错误时不能保证真实域上的结论。
