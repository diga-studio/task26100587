"""Required scenario: exact collinearity plus a constant column."""

import numpy as np
import pytest

from meridian_ols import fit, infer


@pytest.fixture()
def collinear_data():
    rng = np.random.default_rng(42)
    n = 40
    x1 = rng.normal(size=n)
    # Second feature is exactly twice the first; third column is all ones.
    X = np.column_stack([x1, 2.0 * x1, np.ones(n)])
    y = 0.7 + 3.0 * x1 + rng.normal(scale=0.3, size=n)
    return X, y, n


def test_effective_rank_and_residual_df(collinear_data):
    X, y, n = collinear_data
    model = fit(X, y)
    # Design [1, x1, 2*x1, 1] has only 2 independent directions.
    assert model.rank == 2
    # Residual df uses the effective rank, not the input column count.
    assert model.df_residual == n - 2


def test_individual_slopes_not_estimable(collinear_data):
    X, y, _ = collinear_data
    model = fit(X, y)
    C = np.array([
        [0.0, 1.0, 0.0, 0.0],  # slope of feature 1 alone
        [0.0, 0.0, 1.0, 0.0],  # slope of feature 2 alone
    ])
    results = infer(model, C)
    assert [r.estimable for r in results] == [False, False]
    for r in results:
        assert r.estimate is None
        assert r.std_error is None
        assert r.p_value is None
        assert r.ci_lower is None
        assert r.ci_upper is None


def test_combined_slope_estimable(collinear_data):
    X, y, _ = collinear_data
    model = fit(X, y)
    # beta1 + 2*beta2 is estimable; the data-generating value is 3.0.
    C = np.array([[0.0, 1.0, 2.0, 0.0]])
    (r,) = infer(model, C)
    assert r.estimable
    assert r.estimate == pytest.approx(3.0, abs=0.2)
    assert r.std_error > 0.0
    assert 0.0 <= r.p_value <= 1.0
    assert r.ci_lower < r.estimate < r.ci_upper
    # 95% CI half-width equals t_{0.975, df} * se with df = n - rank.
    from meridian_ols._tdist import t_ppf

    half_width = t_ppf(0.975, model.df_residual) * r.std_error
    assert r.ci_upper - r.estimate == pytest.approx(half_width, rel=1e-10)


def test_mixed_batch(collinear_data):
    X, y, _ = collinear_data
    model = fit(X, y)
    C = np.array([
        [0.0, 1.0, 0.0, 0.0],  # not estimable
        [0.0, 1.0, 2.0, 0.0],  # estimable
        [1.0, 0.0, 0.0, 1.0],  # intercept + constant column: estimable
    ])
    results = infer(model, C)
    assert [r.estimable for r in results] == [False, True, True]
    assert results[1].estimate is not None
    assert results[2].estimate is not None
