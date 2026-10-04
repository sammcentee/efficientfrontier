"""Shared annualization and realized portfolio performance measures."""

from numbers import Real

import numpy as np
import pandas as pd


def validate_periods_per_year(value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError("periods_per_year must be a finite positive number.")
    return float(value)


def performance_metrics(equity: pd.Series, risk_free_rate: float, periods_per_year: float) -> dict[str, float]:
    """Measure a positive equity path that includes its initial value.

    Sortino uses the annual arithmetic excess return divided by annual downside
    deviation. Its period target is risk_free_rate / periods_per_year, and its
    denominator includes all observations. Calmar uses CAGR / absolute drawdown.
    """
    returns = equity.pct_change(fill_method=None).iloc[1:]
    growth = float(equity.iloc[-1] / equity.iloc[0])
    cagr = float(growth ** (periods_per_year / len(returns)) - 1)
    volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year))
    excess = float(returns.mean() * periods_per_year - risk_free_rate)
    shortfall = np.minimum(returns.to_numpy() - risk_free_rate / periods_per_year, 0)
    downside = float(np.sqrt(np.mean(shortfall ** 2) * periods_per_year))
    drawdown = float((equity / equity.cummax() - 1).min())
    return {
        "total_return": growth - 1,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": excess / volatility if volatility > 1e-12 else float("nan"),
        "max_drawdown": drawdown,
        "sortino": excess / downside if downside > 0 else float("nan"),
        "calmar": cagr / abs(drawdown) if drawdown < 0 else float("nan"),
    }
