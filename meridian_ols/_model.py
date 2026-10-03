"""OLS model representation and fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OLSModel:
    """Fitted OLS model on the design matrix [1, X].

    Attributes:
        coefficients: A minimum-norm least-squares solution of length
            ``1 + n_features`` (intercept first). Not unique when the
            design is rank deficient; use ``infer`` for conclusions.
        rank: Effective (numerical) rank of the design matrix.
        df_residual: Residual degrees of freedom, ``n_samples - rank``.
        sigma2: Residual variance estimate, RSS / df_residual.
        n_features: Number of original input columns (excluding intercept).
    """

    coefficients: np.ndarray
    rank: int
    df_residual: int
    sigma2: float
    n_features: int
    _design: np.ndarray
    _cov_unscaled: np.ndarray
    _row_space_basis: np.ndarray


def fit(X: np.ndarray, y: np.ndarray) -> OLSModel:
    """Fit OLS with an automatic intercept.

    Args:
        X: Numeric feature matrix of shape (n_samples, n_features).
        y: Response vector of shape (n_samples,).

    Returns:
        The fitted OLSModel.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D matrix")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be a 1-D vector with len(y) == X.shape[0]")

    n_samples, n_features = X.shape
    design = np.column_stack([np.ones(n_samples), X])

    U, s, Vt = np.linalg.svd(design, full_matrices=False)
    tol = s.max() * max(design.shape) * np.finfo(np.float64).eps
    rank = int(np.count_nonzero(s > tol))
    if n_samples <= rank:
        raise ValueError("need n_samples > effective rank for inference")

    V_r = Vt[:rank]
    s_r = s[:rank]
    coefficients = V_r.T @ ((U[:, :rank].T @ y) / s_r)

    residuals = y - design @ coefficients
    df_residual = n_samples - rank
    sigma2 = float(residuals @ residuals) / df_residual
    if not sigma2 > 0.0:
        raise ValueError("residual variance must be positive")

    cov_unscaled = (V_r.T / s_r**2) @ V_r

    return OLSModel(
        coefficients=coefficients,
        rank=rank,
        df_residual=df_residual,
        sigma2=sigma2,
        n_features=n_features,
        _design=design,
        _cov_unscaled=cov_unscaled,
        _row_space_basis=V_r,
    )
