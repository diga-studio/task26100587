"""Meridian OLS linear-contrast inference.

Public API:
    fit(X, y) -> OLSModel
    infer(model, C) -> list[ContrastResult]
"""

from ._infer import ContrastResult, infer
from ._model import OLSModel, fit

__all__ = ["ContrastResult", "OLSModel", "fit", "infer"]
