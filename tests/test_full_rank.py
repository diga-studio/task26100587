"""Full-rank inference checked against an independent reference."""
import numpy as np
import pytest

from meridian_ols import fit, infer
from meridian_ols._tdist import t_ppf, t_sf_two_sided


@pytest.fixture()
def full_rank_data():
    rng = np.random.default_rng(7)
    n, p = 50, 3
    X = rng.normal(size=(n, p))
    beta = np.array([1.0, -2.0, 0.5])
    y = 0.3 + X @ beta + rng.normal(scale=0.5, size=n)
    return X, y, n, p


def test_fit_recovers_coefficients(full_rank_data):
    X, y, n, p = full_rank_data
    model = fit(X, y)
    assert model.rank == p + 1
    assert model.df_residual == n - p - 1
    assert model.coefficients == pytest.approx(
        np.array([0.3, 1.0, -2.0, 0.5]), abs=0.3
    )


def test_inference_matches_closed_form(full_rank_data):
    X, y, n, p = full_rank_data
    model = fit(X, y)
    C = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -1.0],
        [2.0, 1.0, 1.0, 1.0],
    ])
    results = infer(model, C)

    A = np.column_stack([np.ones(n), X])
    beta_ref, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta_ref
    df = n - A.shape[1]
    sigma2 = resid @ resid / df
    cov = sigma2 * np.linalg.inv(A.T @ A)

    assert model.df_residual == df
    assert model.sigma2 == pytest.approx(sigma2, rel=1e-10)

    t_crit = t_ppf(0.975, df)
    for c, r in zip(C, results):
        assert r.estimable
        est = c @ beta_ref
        se = np.sqrt(c @ cov @ c)
        assert r.estimate == pytest.approx(est, rel=1e-10)
        assert r.std_error == pytest.approx(se, rel=1e-10)
        assert r.p_value == pytest.approx(
            t_sf_two_sided(est / se, df), rel=1e-8
        )
        assert r.ci_lower == pytest.approx(est - t_crit * se, rel=1e-10)
        assert r.ci_upper == pytest.approx(est + t_crit * se, rel=1e-10)


def test_single_row_vector_input(full_rank_data):
    X, y, _, _ = full_rank_data
    model = fit(X, y)
    (r,) = infer(model, np.array([0.0, 1.0, 0.0, 0.0]))
    assert r.estimable
