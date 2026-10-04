"""Portfolio concentration and contributions to estimated risk."""

import numpy as np
import pandas as pd

from .core import Analysis


def risk_summary(analysis: Analysis) -> pd.DataFrame:
    """Return concentration and diversification measures from the training model.

    Effective holdings equal 1 / sum(weight ** 2). The diversification ratio
    divides the weighted sum of asset volatilities by the portfolio volatility.
    A zero portfolio variance gives an undefined diversification ratio.
    """
    labels = analysis.mean_returns.index
    covariance = analysis.covariance.loc[labels, labels].to_numpy(dtype=float)
    asset_volatility = np.sqrt(np.maximum(np.diag(covariance), 0))
    rows = {}
    for name, portfolio in analysis.portfolios.items():
        weights = portfolio.weights.loc[labels].to_numpy(dtype=float)
        variance = float(weights @ covariance @ weights)
        rows[name] = {
            "max_weight": float(weights.max()),
            "effective_holdings": float(1 / (weights @ weights)),
            "diversification_ratio": (
                float(weights @ asset_volatility / np.sqrt(variance))
                if variance > 0 else float("nan")
            ),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("portfolio")


def risk_contributions(analysis: Analysis) -> pd.DataFrame:
    """Return each asset's share of portfolio variance from the training model.

    Shares equal w_i * (covariance @ w)_i / portfolio variance and sum to one.
    Shares can be negative. Zero portfolio variance gives undefined shares.
    """
    labels = analysis.mean_returns.index
    covariance = analysis.covariance.loc[labels, labels].to_numpy(dtype=float)
    columns = {}
    for name, portfolio in analysis.portfolios.items():
        weights = portfolio.weights.loc[labels].to_numpy(dtype=float)
        contributions = weights * (covariance @ weights)
        variance = float(contributions.sum())
        columns[name] = contributions / variance if variance > 0 else np.full(len(labels), np.nan)
    return pd.DataFrame(columns, index=labels)
