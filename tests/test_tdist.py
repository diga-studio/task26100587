"""Student t helpers against published reference values."""

import pytest

from meridian_ols._tdist import t_ppf, t_sf_two_sided


@pytest.mark.parametrize(
    "df,expected",
    [
        (1, 12.706204736432095),
        (2, 4.302652729691275),
        (10, 2.2281388519649385),
        (16, 2.1199052992210112),
        (30, 2.0422724563012373),
        (100, 1.9839715184497334),
    ],
)
def test_t_ppf_975(df, expected):
    assert t_ppf(0.975, df) == pytest.approx(expected, rel=1e-9)


def test_t_sf_two_sided_known_points():
    assert t_sf_two_sided(0.0, 5) == pytest.approx(1.0)
    assert t_sf_two_sided(2.2281388519649385, 10) == pytest.approx(0.05, abs=1e-9)
    assert t_sf_two_sided(2.093024054408263, 19) == pytest.approx(0.05, abs=1e-9)


def test_t_ppf_symmetry():
    assert t_ppf(0.025, 12) == pytest.approx(-t_ppf(0.975, 12), rel=1e-12)
