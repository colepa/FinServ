"""
Portfolio calculation helpers.

NOTE: This module contains intentional bugs for demo purposes:
  Bug 1 (rounding): calculate_portfolio_value rounds intermediate per-holding
                    values before summing, causing cumulative rounding drift.
  Bug 2 (off-by-one): compute_gain_loss uses the first transaction price as
                      cost basis but slices transactions[1:] instead of [0:],
                      so the first buy is excluded from the cost calculation.
  Bug 3 (division-by-zero): asset_allocation_percentages divides by
                             total_value without guarding against an empty
                             (zero-value) portfolio.
"""
from __future__ import annotations

from typing import Dict, List

from app.models import Holding, Transaction


def calculate_portfolio_value(holdings: Dict[str, Holding]) -> float:
    """Return the total market value of all holdings.

    BUG 1: Each holding's market_value is rounded to 2 decimal places
    *before* accumulation, which introduces cumulative rounding drift when
    many small fractional positions are held.
    """
    total = 0.0
    for holding in holdings.values():
        # BUG: rounding here instead of at the final result causes drift
        total += round(holding.market_value, 2)
    return total


def compute_gain_loss(transactions: List[Transaction], current_price: float) -> float:
    """Compute unrealised gain/loss for a single ticker.

    BUG 2: The cost basis is computed by iterating over transactions[1:]
    (i.e. skipping the first transaction), so the very first buy is not
    included in the average cost, causing the gain/loss to be overstated.
    """
    if not transactions:
        return 0.0

    total_quantity = sum(
        t.quantity if t.transaction_type == "buy" else -t.quantity
        for t in transactions
    )

    # BUG: should be transactions[0:] (or just transactions), not [1:]
    cost_basis = sum(
        t.quantity * t.price_per_share
        for t in transactions[1:]
        if t.transaction_type == "buy"
    )

    market_value = total_quantity * current_price
    return market_value - cost_basis


def asset_allocation_percentages(holdings: Dict[str, Holding]) -> Dict[str, float]:
    """Return each holding's share of total portfolio value as a percentage.

    Returns 0.0 for every holding when the total portfolio value is zero
    (e.g. empty portfolio or all positions worth $0).
    """
    total_value = sum(h.market_value for h in holdings.values())

    if total_value == 0:
        return {ticker: 0.0 for ticker in holdings}

    return {
        ticker: (holding.market_value / total_value) * 100
        for ticker, holding in holdings.items()
    }


def average_cost_basis(transactions: List[Transaction]) -> float:
    """Return the average cost per share across all buy transactions."""
    buys = [t for t in transactions if t.transaction_type == "buy"]
    if not buys:
        return 0.0
    total_cost = sum(t.quantity * t.price_per_share for t in buys)
    total_qty = sum(t.quantity for t in buys)
    return total_cost / total_qty
