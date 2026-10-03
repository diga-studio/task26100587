"""Edge behaviour: constants, duplicates, validation errors."""

import numpy as np
import pytest

from meridian_ols import fit, infer


def test_constant_column_absorbed_into_intercept():
    rng = np.random.default_rng(3)
    n = 25
    x = rng.normal(size=n)
    X = np.column_stack([x, np.ones(n)])
    y = 2.0 - x + rng.normal(scale=0.2, size=n)
    model = fit(X, y)
    assert model.rank == 2
    assert model.df_residual == n - 2
    # Slope of x is estimable; intercept alone is not (constant column).
    r_slope, r_intercept = infer(
        model, np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    )
    assert r_slope.estimable
    assert r_slope.estimate == pytest.approx(-1.0, abs=0.2)
    assert not r_intercept.estimable
    # Intercept + constant column weight is estimable again.
    (r_sum,) = infer(model, np.array([[1.0, 0.0, 1.0]]))
    assert r_sum.estimable


def test_duplicated_column_split_is_not_estimable():
    rng = np.random.default_rng(11)
    n = 30
    x = rng.normal(size=n)
    X = np.column_stack([x, x])
    y = 1.0 + 2.0 * x + rng.normal(scale=0.1, size=n)
    model = fit(X, y)
    r1, r2, r_sum = infer(
        model,
        np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]),
    )
    assert not r1.estimable
    assert not r2.estimable
    assert r_sum.estimable
    assert r_sum.estimate == pytest.approx(2.0, abs=0.1)


def test_rank_deficiency_does_not_block_estimable_contrasts():
    rng = np.random.default_rng(5)
    n = 60
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([x1, x2, x1 + x2])  # exact dependence
    y = 0.5 + 1.5 * x1 - 0.5 * x2 + rng.normal(scale=0.4, size=n)
    model = fit(X, y)
    assert model.rank == 3
    # Any contrast with c1 + c3 coefficient pattern in the row space works;
    # e.g. (0, 1, 0, 1) picks beta1 + beta3 which is identified.
    (r,) = infer(model, np.array([[0.0, 1.0, 0.0, 1.0]]))
    assert r.estimable


def test_zero_contrast_rejected():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 2))
    y = rng.normal(size=10)
    model = fit(X, y)
    with pytest.raises(ValueError, match="zero contrast"):
        infer(model, np.array([[0.0, 0.0, 0.0]]))


def test_wrong_contrast_width_rejected():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 2))
    y = rng.normal(size=10)
    model = fit(X, y)
    with pytest.raises(ValueError):
        infer(model, np.array([[1.0, 0.0]]))


def test_fit_rejects_insufficient_samples():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="effective rank"):
        fit(X, y)
