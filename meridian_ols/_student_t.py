"""Student t 分布的 CDF 与分位数（纯 NumPy 实现）。

通过正则化不完全 Beta 函数 I_x(a, b) 计算 t 分布 CDF，
连分式算法采用 Numerical Recipes 的 betacf。分位数用二分法反解。
"""
from __future__ import annotations

import math


def _betacf(a: float, b: float, x: float) -> float:
    """不完全 Beta 函数连分式部分（Lentz 算法）。"""
    max_iter = 200
    eps = 3.0e-14
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta 函数 I_x(a, b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: int) -> float:
    """Student t 分布的累积分布函数 P(T <= t)，自由度 df。"""
    if df < 1:
        raise ValueError("df 必须 >= 1")
    x = df / (df + t * t)
    ib = _betai(df / 2.0, 0.5, x)
    if t >= 0.0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


def t_ppf(p: float, df: int) -> float:
    """Student t 分布的分位数函数，即 t_cdf 的反函数（二分法）。"""
    if not 0.0 < p < 1.0:
        raise ValueError("p 必须在 (0, 1) 内")
    if df < 1:
        raise ValueError("df 必须 >= 1")
    lo, hi = -1.0, 1.0
    while t_cdf(lo, df) > p:
        lo *= 2.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
