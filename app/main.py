"""FastAPI application entry-point for the FinServ portfolio tracker."""
from __future__ import annotations

import uuid
from typing import Dict

from fastapi import FastAPI, HTTPException

from app import data
from app.calculations import asset_allocation_percentages, average_cost_basis, calculate_portfolio_value
from app.models import (
    Holding,
    Portfolio,
    PortfolioCreate,
    PortfolioSummary,
    PriceUpdate,
    Transaction,
    TransactionCreate,
)

app = FastAPI(title="FinServ Portfolio Tracker", version="0.1.0")


# ---------------------------------------------------------------------------
# Portfolio endpoints
# ---------------------------------------------------------------------------


@app.post("/portfolios", response_model=Portfolio, status_code=201)
def create_portfolio(payload: PortfolioCreate) -> Portfolio:
    """Create a new portfolio."""
    portfolio_id = str(uuid.uuid4())
    portfolio = Portfolio(
        id=portfolio_id,
        name=payload.name,
        owner=payload.owner,
        cash_balance=payload.cash_balance,
    )
    data.portfolios[portfolio_id] = portfolio
    return portfolio


@app.get("/portfolios/{portfolio_id}", response_model=PortfolioSummary)
def get_portfolio(portfolio_id: str, prices: Dict[str, float] | None = None) -> PortfolioSummary:
    """
    Retrieve a portfolio by ID.

    Optionally accepts a JSON body with a ``prices`` mapping (ticker -> price)
    to revalue holdings at current market prices.  When omitted, the stored
    ``current_price`` on each holding is used.
    """
    portfolio = _get_or_404(portfolio_id)

    holdings = portfolio.holdings
    total_market_value = calculate_portfolio_value(holdings)
    total_cost_basis = round(sum(h.average_cost * h.quantity for h in holdings.values()), 2)
    total_gain_loss = total_market_value - total_cost_basis

    return PortfolioSummary(
        id=portfolio.id,
        name=portfolio.name,
        owner=portfolio.owner,
        total_market_value=total_market_value,
        total_cost_basis=total_cost_basis,
        total_gain_loss=total_gain_loss,
        cash_balance=portfolio.cash_balance,
        holdings=holdings,
    )


@app.post("/portfolios/{portfolio_id}/transactions", response_model=Transaction, status_code=201)
def add_transaction(portfolio_id: str, payload: TransactionCreate) -> Transaction:
    """Record a buy or sell transaction and update the portfolio's holdings."""
    portfolio = _get_or_404(portfolio_id)

    transaction = Transaction(
        id=str(uuid.uuid4()),
        portfolio_id=portfolio_id,
        ticker=payload.ticker.upper(),
        transaction_type=payload.transaction_type,
        quantity=payload.quantity,
        price_per_share=payload.price_per_share,
    )
    portfolio.transactions.append(transaction)

    ticker = transaction.ticker
    tx_type = transaction.transaction_type.value

    if ticker not in portfolio.holdings:
        if tx_type == "sell":
            raise HTTPException(status_code=400, detail=f"No holding found for ticker {ticker}")
        portfolio.holdings[ticker] = Holding(
            ticker=ticker,
            quantity=0.0,
            average_cost=0.0,
            current_price=payload.price_per_share,
            market_value=0.0,
            gain_loss=0.0,
            allocation_pct=0.0,
        )

    holding = portfolio.holdings[ticker]
    ticker_txns = [t for t in portfolio.transactions if t.ticker == ticker]

    if tx_type == "buy":
        new_total_qty = holding.quantity + payload.quantity
        new_avg_cost = average_cost_basis(ticker_txns)
        holding.quantity = new_total_qty
        holding.average_cost = new_avg_cost
        holding.current_price = payload.price_per_share
        portfolio.cash_balance -= payload.quantity * payload.price_per_share
    else:
        if payload.quantity > holding.quantity:
            raise HTTPException(status_code=400, detail="Insufficient shares to sell")
        holding.quantity -= payload.quantity
        holding.current_price = payload.price_per_share
        portfolio.cash_balance += payload.quantity * payload.price_per_share

    holding.market_value = holding.quantity * holding.current_price

    # Refresh gain/loss for this holding
    holding.gain_loss = holding.market_value - (holding.average_cost * holding.quantity)

    # Refresh allocation percentages across all holdings
    try:
        alloc = asset_allocation_percentages(portfolio.holdings)
        for t, pct in alloc.items():
            portfolio.holdings[t].allocation_pct = pct
    except ZeroDivisionError:
        for t in portfolio.holdings:
            portfolio.holdings[t].allocation_pct = 0.0

    return transaction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_or_404(portfolio_id: str) -> Portfolio:
    portfolio = data.portfolios.get(portfolio_id)
    if portfolio is None:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return portfolio
