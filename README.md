# Meridian OLS 线性对比推断组件

对数值特征矩阵做经典 OLS 拟合，并对系数的线性对比（linear contrast）做统计推断。
组件能正确处理重复编码、常量列与精确线性依赖：只有当一个对比在所有产生相同
拟合值的系数解上取值一致（即**可估**，estimable）时才报告推断结果，否则返回
`"not_estimable"`，不附带估计值或区间。模型整体秩亏不影响可估对比的推断。

## 安装与依赖

- Python 3.10+，仅依赖 NumPy（Student t 分布由内置的 `_student_t` 模块纯 Python 实现，
  无需 SciPy）。
- 将仓库根目录加入 `PYTHONPATH` 后直接 `import meridian_ols`。

## API

仅两个操作：

### `fit(X, y) -> OLSModel`

- `X`：形状 `(n, p)` 的 float64 特征矩阵，无缺失、无无穷值；可含常量列与精确线性依赖。
- `y`：长度 `n` 的 float64 响应向量。
- 自动添加截距；通过 SVD 得到有效秩 `rank`，残差自由度为 `n - rank`
  （**不是** `n - p - 1`），残差方差为 `RSS / (n - rank)`。
- 要求 `n > rank` 且残差方差为正，否则抛出 `ValueError`。

### `infer(model, C) -> list[ContrastResult]`

- `C`：形状 `(k, p + 1)` 的对比矩阵，每行 `[截距权重, 特征1权重, ..., 特征p权重]`，
  始终使用**原始特征单位**（与 `fit` 输入的列一一对应）。
- 同一批 `C` 可混合可估与不可估对比，逐行返回 `ContrastResult`：
  - 可估：`status="estimable"`，并给出 `estimate`、`std_error`、
    双侧 `p_value`（检验对比等于零）与 95% 置信区间 `[ci_lower, ci_upper]`
    （基于 `n - rank` 自由度的 Student t 分布，同方差正态误差的经典 OLS 定义）。
  - 不可估：`status="not_estimable"`，其余字段均为 `None`。
- 全零对比或列数不匹配的 `C` 抛出 `ValueError`。

## 示例

```python
import numpy as np
from meridian_ols import fit, infer

rng = np.random.default_rng(0)
x1 = rng.normal(size=60)
x2 = rng.normal(size=60)
X = np.column_stack([x1, 2.0 * x1, x2, np.ones(60)])  # 重复编码 + 常量列
y = 0.5 + 1.0 * x1 + 2.0 * x2 + rng.normal(scale=0.2, size=60)

model = fit(X, y)
print(model.rank, model.df_residual)  # 3, 57（不是 60 - 5）

results = infer(model, np.array([
    [0, 1, 0, 0, 0],   # x1 斜率：不可估
    [0, 1, 2, 0, 0],   # x1 斜率 + 2*重复列斜率：可估，≈ 1.0
    [1, 0, 0, 0, 1],   # 截距 + 常量列：可估，≈ 0.5
]))
for r in results:
    print(r.status, r.estimate, r.ci_lower, r.ci_upper)
```

## 运行测试

```bash
python3 -m unittest discover -s tests -v
```

## 实现说明

- 拟合使用 SVD 的最小范数最小二乘解作为代表系数；可估对比的取值与解的选取无关。
- 可估性判定：对比向量 `c` 与设计矩阵零空间正交（即 `c` 落在行空间内），
  等价于 `c` 在所有产生相同拟合值的解上取值一致。
- 可估对比的方差为 `σ² c'(Z'Z)⁺c`（`Z` 为增广设计矩阵，`⁺` 为 Moore–Penrose 伪逆），
  通过 SVD 因子直接计算，不显式求逆。
