"""OLS 模型拟合：自动添加截距，处理秩亏设计矩阵。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OLSModel:
    """拟合后的 OLS 模型（内部表示，系数顺序为 [截距, 原始特征...]）。"""

    coefficients: np.ndarray  # 形状 (p+1,)，最小范数最小二乘解
    rank: int  # 设计矩阵有效秩
    df_residual: int  # 残差自由度 = n - rank
    sigma2: float  # 残差方差估计 RSS / df_residual
    n_features: int  # 原始输入列数 p
    _v_t: np.ndarray  # SVD 的 V^T，前 rank 行为行空间基，其余为零空间基
    _singular_values: np.ndarray  # 非零奇异值（长度 rank）


def fit(X: np.ndarray, y: np.ndarray) -> OLSModel:
    """对数值特征矩阵 X 自动添加截距并拟合 OLS，返回模型。

    通过 SVD 得到有效秩，使用最小范数最小二乘解作为代表系数；
    秩亏时该解仍产生唯一拟合值，可估对比的取值与解的选取无关。
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X 必须是二维数组")
    n, p = X.shape
    if y.ndim != 1 or y.shape[0] != n:
        raise ValueError("y 必须是长度等于 X 行数的一维数组")

    design = np.column_stack([np.ones(n), X])
    u, s, v_t = np.linalg.svd(design, full_matrices=False)
    tol = np.finfo(np.float64).eps * max(design.shape) * (s[0] if s.size else 0.0)
    rank = int(np.sum(s > tol))
    df_residual = n - rank
    if df_residual <= 0:
        raise ValueError("样本数必须大于有效秩（残差自由度必须为正）")

    # 最小范数解：beta = V_r S_r^{-1} U_r^T y
    s_r = s[:rank]
    coefficients = (v_t[:rank].T / s_r) @ (u[:, :rank].T @ y)
    fitted = design @ coefficients
    rss = float(np.sum((y - fitted) ** 2))
    sigma2 = rss / df_residual
    # 浮点下精确拟合的 RSS 未必恰为 0，按 y 的量级做相对判定
    y_scale = float(np.sum(y**2)) / n
    if sigma2 <= 1e-12 * max(y_scale, np.finfo(np.float64).tiny):
        raise ValueError("残差方差必须为正（y 不能被设计矩阵精确拟合）")

    return OLSModel(
        coefficients=coefficients,
        rank=rank,
        df_residual=df_residual,
        sigma2=sigma2,
        n_features=p,
        _v_t=v_t,
        _singular_values=s_r,
    )
