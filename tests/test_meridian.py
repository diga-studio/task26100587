"""Meridian OLS 线性对比推断组件的自动化测试。"""
import math
import unittest

import numpy as np

from meridian_ols import fit, infer
from meridian_ols._student_t import t_cdf, t_ppf


def make_rank_deficient_data(n=60, seed=42):
    """第二个特征恰为第一个的两倍，另有一列全为一。"""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([x1, 2.0 * x1, x2, np.ones(n)])
    # 真实模型：y = 0.5 + 1.0*x1 + 2.0*x2 + 噪声
    y = 0.5 + 1.0 * x1 + 2.0 * x2 + rng.normal(scale=0.2, size=n)
    return X, y


class TestStudentT(unittest.TestCase):
    def test_cdf_known_values(self):
        self.assertAlmostEqual(t_cdf(0.0, 5), 0.5, places=12)
        self.assertAlmostEqual(t_cdf(2.2281388519649385, 10), 0.975, places=10)
        self.assertAlmostEqual(t_cdf(-2.2281388519649385, 10), 0.025, places=10)

    def test_ppf_inverts_cdf(self):
        for df in (1, 3, 10, 100):
            for p in (0.025, 0.5, 0.975):
                self.assertAlmostEqual(t_cdf(t_ppf(p, df), df), p, places=9)


class TestFit(unittest.TestCase):
    def test_full_rank_recovers_coefficients(self):
        rng = np.random.default_rng(1)
        n = 200
        X = rng.normal(size=(n, 3))
        beta = np.array([1.5, -2.0, 0.7])
        y = 3.0 + X @ beta + rng.normal(scale=0.5, size=n)
        m = fit(X, y)
        self.assertEqual(m.rank, 4)
        self.assertEqual(m.df_residual, n - 4)
        np.testing.assert_allclose(m.coefficients, [3.0, 1.5, -2.0, 0.7], atol=0.2)

    def test_residual_df_uses_effective_rank_not_column_count(self):
        X, y = make_rank_deficient_data(n=60)
        m = fit(X, y)
        # 设计矩阵共 5 列（截距 + 4 特征），但常量列与截距共线、
        # 第 2 特征是第 1 个的两倍，有效秩 = 3（截距、x1、x2）
        self.assertEqual(m.rank, 3)
        self.assertEqual(m.df_residual, 60 - 3)

    def test_rejects_nonpositive_residual_df(self):
        # n = 3，设计矩阵满秩 3，残差自由度为 0
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]])
        y = np.array([1.0, 2.0, 4.0])
        with self.assertRaises(ValueError):
            fit(X, y)

    def test_rejects_zero_residual_variance(self):
        # y 被设计矩阵精确拟合，残差方差为 0
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([3.0, 5.0, 7.0, 9.0])
        with self.assertRaises(ValueError):
            fit(X, y)


class TestInfer(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_rank_deficient_data(n=60)
        self.model = fit(self.X, self.y)

    def test_duplicated_slopes_not_estimable_but_combination_is(self):
        # 列顺序：[x1, 2*x1, x2, 常量列]；C 每行：[截距, x1, 2x1, x2, 常量]
        results = infer(
            self.model,
            np.array(
                [
                    [0, 1, 0, 0, 0],  # x1 斜率：不可估
                    [0, 0, 1, 0, 0],  # 重复列斜率：不可估
                    [0, 1, 2, 0, 0],  # x1 斜率 + 2*重复列斜率：可估
                ]
            ),
        )
        self.assertEqual(results[0].status, "not_estimable")
        self.assertIsNone(results[0].estimate)
        self.assertIsNone(results[0].ci_lower)
        self.assertEqual(results[1].status, "not_estimable")
        self.assertIsNone(results[1].estimate)

        combo = results[2]
        self.assertEqual(combo.status, "estimable")
        # x1 对 y 的总效应 = 1.0（真实生成系数）
        self.assertAlmostEqual(combo.estimate, 1.0, delta=0.1)
        self.assertGreater(combo.std_error, 0.0)
        self.assertLess(combo.ci_lower, combo.estimate)
        self.assertGreater(combo.ci_upper, combo.estimate)
        self.assertTrue(0.0 <= combo.p_value <= 1.0)

    def test_estimable_contrast_invariant_across_solutions(self):
        # 可估对比在任意产生相同拟合值的解上取值一致：
        # 零空间方向 (0, 2, -1, 0, 0) 与 (1, 0, 0, 0, -1) 上加任意倍数，
        # 可估对比 (0, 1, 2, 0, 0) 的值不变。
        c = np.array([0, 1, 2, 0, 0])
        est = infer(self.model, c)[0].estimate
        beta = self.model.coefficients
        for null_dir in (
            np.array([0, 2.0, -1.0, 0, 0]),
            np.array([-1.0, 0, 0, 0, 1.0]),
        ):
            for k in (-3.0, 0.7, 10.0):
                self.assertAlmostEqual(c @ (beta + k * null_dir), est, places=6)

    def test_mixed_batch_and_intercept_contrast(self):
        results = infer(
            self.model,
            np.array(
                [
                    [1, 0, 0, 0, 1],  # 截距 + 常量列：可估
                    [0, 0, 0, 1, 0],  # x2 斜率：可估
                    [0, 1, 0, 0, 0],  # x1 斜率：不可估
                ]
            ),
        )
        self.assertEqual([r.status for r in results],
                         ["estimable", "estimable", "not_estimable"])
        # 截距 + 常量列 ≈ 0.5（真实截距），x2 斜率 ≈ 2.0
        self.assertAlmostEqual(results[0].estimate, 0.5, delta=0.15)
        self.assertAlmostEqual(results[1].estimate, 2.0, delta=0.1)

    def test_full_rank_all_contrasts_estimable(self):
        rng = np.random.default_rng(7)
        n = 100
        X = rng.normal(size=(n, 2))
        y = 1.0 + X @ np.array([2.0, -1.0]) + rng.normal(scale=0.3, size=n)
        m = fit(X, y)
        C = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, -1]])
        results = infer(m, C)
        self.assertTrue(all(r.status == "estimable" for r in results))
        self.assertAlmostEqual(results[0].estimate, 1.0, delta=0.15)
        self.assertAlmostEqual(results[1].estimate, 2.0, delta=0.15)
        self.assertAlmostEqual(results[2].estimate, -1.0, delta=0.15)
        self.assertAlmostEqual(results[3].estimate, 3.0, delta=0.25)

    def test_confidence_interval_and_pvalue_consistency(self):
        # 95% 区间不包含 0 当且仅当双侧 p 值 < 0.05
        rng = np.random.default_rng(3)
        n = 80
        X = rng.normal(size=(n, 1))
        y = 0.3 * X[:, 0] + rng.normal(scale=1.0, size=n)
        m = fit(X, y)
        r = infer(m, np.array([[0, 1]]))[0]
        covers_zero = r.ci_lower <= 0.0 <= r.ci_upper
        self.assertEqual(covers_zero, r.p_value >= 0.05)
        # 区间半宽 = t_0.975(df) * se
        half_width = t_ppf(0.975, m.df_residual) * r.std_error
        self.assertAlmostEqual(r.ci_upper - r.estimate, half_width, places=10)

    def test_zero_contrast_rejected(self):
        with self.assertRaises(ValueError):
            infer(self.model, np.array([[0, 0, 0, 0, 0]]))

    def test_wrong_contrast_width_rejected(self):
        with self.assertRaises(ValueError):
            infer(self.model, np.array([[1, 0, 0]]))


if __name__ == "__main__":
    unittest.main()
