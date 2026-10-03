"""线性对比推断：可估性判定、估计、标准误、p 值与置信区间。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._student_t import t_cdf, t_ppf
from .model import OLSModel

NOT_ESTIMABLE = "not_estimable"
ESTIMABLE = "estimable"


@dataclass(frozen=True)
class ContrastResult:
    """单个线性对比的推断结果。不可估时除 status 外均为 None。"""

    status: str  # "estimable" 或 "not_estimable"
    estimate: Optional[float] = None
    std_error: Optional[float] = None
    p_value: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


def _is_estimable(c: np.ndarray, model: OLSModel) -> bool:
    """c 可估当且仅当 c 与设计矩阵零空间正交（即 c 落在行空间内）。

    等价于：c 在所有产生相同拟合值的系数解上取值一致。
    """
    null_basis = model._v_t[model.rank:]
    if null_basis.size == 0:
        return True
    leakage = np.abs(null_basis @ c)
    scale = np.linalg.norm(c)
    tol = 1e-8 * max(1.0, scale)
    return bool(np.all(leakage <= tol))


def infer(model: OLSModel, C: np.ndarray) -> list[ContrastResult]:
    """对 C 的每一行做线性对比推断。

    C 的每行定义为 [截距权重, 特征1权重, ..., 特征p权重]。
    同一批对比中可估与不可估可以混合：可估行返回估计值、标准误、
    双侧 p 值与 95% 置信区间；不可估行仅返回 status="not_estimable"。
    """
    C = np.asarray(C, dtype=np.float64)
    if C.ndim == 1:
        C = C.reshape(1, -1)
    if C.ndim != 2 or C.shape[1] != model.n_features + 1:
        raise ValueError("C 的列数必须等于 1（截距）+ 原始特征数")

    df = model.df_residual
    t_crit = t_ppf(0.975, df)
    # (Z'Z)^+ 的作用：对行空间向量 c，c'(Z'Z)^+c = ||c V_r S_r^{-1}||^2
    v_scaled = model._v_t[: model.rank] / model._singular_values[:, None]

    results: list[ContrastResult] = []
    for c in C:
        norm = np.linalg.norm(c)
        if norm == 0.0:
            raise ValueError("不支持全零对比")
        if not _is_estimable(c, model):
            results.append(ContrastResult(status=NOT_ESTIMABLE))
            continue
        estimate = float(c @ model.coefficients)
        proj = v_scaled @ c  # S_r^{-1} V_r^T c
        var = model.sigma2 * float(proj @ proj)
        std_error = float(np.sqrt(var))
        t_stat = estimate / std_error
        p_value = 2.0 * (1.0 - t_cdf(abs(t_stat), df))
        margin = t_crit * std_error
        results.append(
            ContrastResult(
                status=ESTIMABLE,
                estimate=estimate,
                std_error=std_error,
                p_value=float(p_value),
                ci_lower=estimate - margin,
                ci_upper=estimate + margin,
            )
        )
    return results
