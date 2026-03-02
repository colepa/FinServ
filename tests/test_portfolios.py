"""Tests for the FinServ portfolio tracker API.

These tests cover happy-path behaviour and basic validation.
The buggy code paths in calculations.py (rounding drift, off-by-one
cost basis, division-by-zero on empty portfolio) are intentionally
not covered here so the automated triage system can discover them.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import data
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_store():
    """Reset the in-memory store before each test."""
    data.portfolios.clear()
    yield
    data.portfolios.clear()


# ---------------------------------------------------------------------------
# POST /portfolios
# ---------------------------------------------------------------------------


def test_create_portfolio_returns_201():
    resp = client.post("/portfolios", json={"name": "Retirement", "owner": "alice"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "Retirement"
    assert body["owner"] == "alice"
    assert "id" in body


def test_create_portfolio_with_cash_balance():
    resp = client.post(
        "/portfolios",
        json={"name": "Growth Fund", "owner": "bob", "cash_balance": 5000.0},
    )
    assert resp.status_code == 201
    assert resp.json()["cash_balance"] == 5000.0


def test_create_portfolio_negative_cash_rejected():
    resp = client.post(
        "/portfolios",
        json={"name": "Bad Fund", "owner": "eve", "cash_balance": -100.0},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /portfolios/{id}
# ---------------------------------------------------------------------------


def test_get_portfolio_not_found():
    resp = client.get("/portfolios/nonexistent-id")
    assert resp.status_code == 404


def test_get_portfolio_after_create():
    create_resp = client.post("/portfolios", json={"name": "Tech", "owner": "carol"})
    pid = create_resp.json()["id"]

    resp = client.get(f"/portfolios/{pid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == pid
    assert body["name"] == "Tech"
    assert body["holdings"] == {}


# ---------------------------------------------------------------------------
# POST /portfolios/{id}/transactions
# ---------------------------------------------------------------------------


def test_add_buy_transaction():
    pid = client.post("/portfolios", json={"name": "P1", "owner": "dan"}).json()["id"]

    resp = client.post(
        f"/portfolios/{pid}/transactions",
        json={
            "ticker": "aapl",
            "transaction_type": "buy",
            "quantity": 10,
            "price_per_share": 150.0,
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["ticker"] == "AAPL"
    assert body["transaction_type"] == "buy"
    assert body["quantity"] == 10


def test_buy_updates_holdings():
    pid = client.post("/portfolios", json={"name": "P2", "owner": "dan"}).json()["id"]
    client.post(
        f"/portfolios/{pid}/transactions",
        json={"ticker": "MSFT", "transaction_type": "buy", "quantity": 5, "price_per_share": 300.0},
    )

    portfolio_resp = client.get(f"/portfolios/{pid}")
    holdings = portfolio_resp.json()["holdings"]
    assert "MSFT" in holdings
    assert holdings["MSFT"]["quantity"] == 5


def test_sell_reduces_holding():
    pid = client.post("/portfolios", json={"name": "P3", "owner": "erin"}).json()["id"]
    client.post(
        f"/portfolios/{pid}/transactions",
        json={"ticker": "GOOG", "transaction_type": "buy", "quantity": 8, "price_per_share": 100.0},
    )
    resp = client.post(
        f"/portfolios/{pid}/transactions",
        json={"ticker": "GOOG", "transaction_type": "sell", "quantity": 3, "price_per_share": 110.0},
    )
    assert resp.status_code == 201

    holdings = client.get(f"/portfolios/{pid}").json()["holdings"]
    assert holdings["GOOG"]["quantity"] == 5


def test_sell_more_than_owned_rejected():
    pid = client.post("/portfolios", json={"name": "P4", "owner": "frank"}).json()["id"]
    client.post(
        f"/portfolios/{pid}/transactions",
        json={"ticker": "TSLA", "transaction_type": "buy", "quantity": 2, "price_per_share": 200.0},
    )
    resp = client.post(
        f"/portfolios/{pid}/transactions",
        json={"ticker": "TSLA", "transaction_type": "sell", "quantity": 5, "price_per_share": 220.0},
    )
    assert resp.status_code == 400


def test_transaction_on_missing_portfolio():
    resp = client.post(
        "/portfolios/bad-id/transactions",
        json={"ticker": "X", "transaction_type": "buy", "quantity": 1, "price_per_share": 10.0},
    )
    assert resp.status_code == 404


def test_sell_unknown_ticker_rejected():
    pid = client.post("/portfolios", json={"name": "P5", "owner": "grace"}).json()["id"]
    resp = client.post(
        f"/portfolios/{pid}/transactions",
        json={"ticker": "UNKNOWN", "transaction_type": "sell", "quantity": 1, "price_per_share": 50.0},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Floating-point rounding drift regression tests
# ---------------------------------------------------------------------------


def test_no_rounding_drift_after_many_small_transactions():
    """Adding many small fractional transactions must not accumulate
    floating-point noise in the total market value."""
    pid = client.post(
        "/portfolios", json={"name": "Drift Test", "owner": "tester"}
    ).json()["id"]

    # 30 small buys with awkward fractional values
    trades = [
        ("AAPL", 0.33, 10.33),
        ("AAPL", 0.17, 5.71),
        ("GOOG", 0.29, 12.07),
        ("GOOG", 0.41, 7.89),
        ("MSFT", 0.13, 15.43),
        ("MSFT", 0.37, 9.11),
        ("AAPL", 0.11, 11.17),
        ("GOOG", 0.23, 8.53),
        ("MSFT", 0.19, 14.29),
        ("AAPL", 0.07, 6.67),
        ("GOOG", 0.31, 10.99),
        ("MSFT", 0.43, 7.31),
        ("AAPL", 0.09, 13.37),
        ("GOOG", 0.27, 9.41),
        ("MSFT", 0.21, 11.83),
        ("AAPL", 0.39, 8.17),
        ("GOOG", 0.47, 6.43),
        ("MSFT", 0.53, 12.61),
        ("AAPL", 0.61, 7.79),
        ("GOOG", 0.03, 14.53),
        ("MSFT", 0.67, 5.39),
        ("AAPL", 0.71, 10.91),
        ("GOOG", 0.59, 8.27),
        ("MSFT", 0.83, 6.73),
        ("AAPL", 0.97, 9.59),
        ("GOOG", 0.89, 11.41),
        ("MSFT", 0.79, 7.97),
        ("AAPL", 0.02, 13.03),
        ("GOOG", 0.14, 10.67),
        ("MSFT", 0.06, 8.89),
    ]

    for ticker, qty, price in trades:
        resp = client.post(
            f"/portfolios/{pid}/transactions",
            json={
                "ticker": ticker,
                "transaction_type": "buy",
                "quantity": qty,
                "price_per_share": price,
            },
        )
        assert resp.status_code == 201

    summary = client.get(f"/portfolios/{pid}").json()
    total_mv = summary["total_market_value"]

    # The total must have at most 2 decimal places (no floating-point noise)
    assert total_mv == round(total_mv, 2), (
        f"total_market_value has floating-point noise: {total_mv!r}"
    )


def test_calculate_portfolio_value_no_intermediate_rounding():
    """Unit-level check: calculate_portfolio_value should sum raw values
    and round only at the end."""
    from app.calculations import calculate_portfolio_value
    from app.models import Holding

    # Construct holdings whose individual market_values would each round
    # differently if rounded before summation.
    holdings = {
        f"T{i}": Holding(
            ticker=f"T{i}",
            quantity=0.33,
            average_cost=10.0,
            current_price=10.33,
            market_value=0.33 * 10.33,  # 3.4089 — rounds to 3.41
            gain_loss=0.0,
            allocation_pct=0.0,
        )
        for i in range(50)
    }

    result = calculate_portfolio_value(holdings)

    # Expected: round(50 * 0.33 * 10.33, 2)
    expected = round(50 * 0.33 * 10.33, 2)
    assert result == expected, f"Expected {expected}, got {result}"
