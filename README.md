# task26100587

## meridian_ols：OLS 线性对比推断

对数值特征矩阵自动添加截距拟合 OLS，并对系数线性对比做经典
同方差正态误差推断。秩亏（重复编码、常量列、精确线性依赖）时，
只有可估对比才返回统计量，不可估对比返回 `estimable=False` 且
不附带任何估计值或区间。

### 使用

```python
import numpy as np
from meridian_ols import fit, infer

X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0]])
y = np.array([2.1, 4.2, 5.9, 8.1, 10.2])

model = fit(X, y)            # 自动添加截距
print(model.rank, model.df_residual)

C = np.array([
    [0.0, 1.0, 0.0],         # 第一个斜率：不可估（与第二列共线）
    [0.0, 1.0, 2.0],         # beta1 + 2*beta2：可估
])
for r in infer(model, C):
    if r.estimable:
        print(r.estimate, r.std_error, r.p_value, (r.ci_lower, r.ci_upper))
    else:
        print("not_estimable")
```

### 约定

- `C` 每行一个对比，第一项是截距权重，其余按原始输入列顺序。
- 残差方差 = RSS / (n - 有效秩)；区间与 p 值使用该自由度的
  Student t 分布，置信水平 95%。
- 仅依赖 NumPy（SciPy 可选但未使用）；t 分布由内置不完全 beta
  函数实现。

### 测试

```bash
python3 -m pytest tests/ -q
```
