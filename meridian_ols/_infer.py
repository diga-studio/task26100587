"""Linear-contrast inference for fitted OLS models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ._model import OLSModel
from ._tdist import t_ppf, t_sf_two_sided

_ESTIMABILITY_RTOL = 1e-8


@dataclass(frozen=True)
class ContrastResult:
    """Inference result for one linear contrast.

    Attributes:
        estimable: Whether the contrast is estimable, i.e. takes the same
            value under every coefficient solution that yields the same
            fitted values.
        estimate: Point estimate of the contrast, or None if not estimable.
        std_error: Standard error of the estimate, or None.
        p_value: Two-sided p-value for H0: contrast == 0, or None.
        ci_lower: Lower bound of the 95% confidence interval, or None.
        ci_upper: Upper bound of the 95% confidence interval, or None.
    """

    estimable: bool
    estimate: Optional[float] = None
    std_error: Optional[float] = None
    p_value: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


def infer(model: OLSModel, C: np.ndarray) -> List[ContrastResult]:
    """Evaluate linear contrasts against a fitted model.

    Each row of C defines one contrast over [intercept, *features]:
    the first entry weights the intercept, the remaining entries weight
    the original input columns in their original units.

    Args:
        model: A model returned by ``fit``.
        C: Contrast matrix of shape (n_contrasts, 1 + n_features).

    Returns:
        One ContrastResult per row of C, in order. Non-estimable rows
        return only ``estimable=False`` with no estimate or interval.
    """
    C = np.asarray(C, dtype=np.float64)
    if C.ndim == 1:
        C = C.reshape(1, -1)
    if C.ndim != 2 or C.shape[1] != model.n_features + 1:
        raise ValueError(
            "C must have shape (n_contrasts, 1 + n_features); "
            f"expected {model.n_features + 1} columns, got "
            f"{C.shape[1] if C.ndim == 2 else C.ndim - 1}"
        )

    V_r = model._row_space_basis
    df = model.df_residual
    t_crit = t_ppf(0.975, df)

    results: List[ContrastResult] = []
    for c in C:
        norm_c = float(np.linalg.norm(c))
        if norm_c == 0.0:
            raise ValueError("zero contrast row is not allowed")

        residual = c - V_r.T @ (V_r @ c)
        if float(np.linalg.norm(residual)) > _ESTIMABILITY_RTOL * norm_c:
            results.append(ContrastResult(estimable=False))
            continue

        estimate = float(c @ model.coefficients)
        var = model.sigma2 * float(c @ model._cov_unscaled @ c)
        std_error = float(np.sqrt(var))
        t_stat = estimate / std_error
        p_value = t_sf_two_sided(t_stat, df)
        margin = t_crit * std_error
        results.append(
            ContrastResult(
                estimable=True,
                estimate=estimate,
                std_error=std_error,
                p_value=p_value,
                ci_lower=estimate - margin,
                ci_upper=estimate + margin,
            )
        )
    return results
