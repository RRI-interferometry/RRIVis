"""Typed visualization failures owned by the canonical result renderers."""

from __future__ import annotations


class ResultPlotContractError(RuntimeError):
    """A plot request violates the canonical renderer input contract."""


class ResultBrowserError(RuntimeError):
    """Browser presentation failed after plot files were already published."""


__all__ = [
    "ResultBrowserError",
    "ResultPlotContractError",
]
