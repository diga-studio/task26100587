"""Meridian OLS 线性对比推断组件。"""
from .model import OLSModel, fit
from .inference import ContrastResult, infer

__all__ = ["OLSModel", "fit", "ContrastResult", "infer"]
