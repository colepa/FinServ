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
# Floating-point drift regression
# ---------------------------------------------------------------------------


def test_no_floating_point_drift_after_many_transactions():
    """Adding many small fractional transactions must not accumulate rounding noise.

    Regression test for the bug where intermediate per-holding rounding caused
    cumulative drift in the total portfolio market value.
    """
    pid = client.post("/portfolios", json={"name": "Drift Test", "owner": "tester"}).json()["id"]

    # Add 50 small fractional buy transactions across several tickers
    small_trades = [
        ("AAPL", 0.33, 10.33),
        ("MSFT", 0.17, 5.71),
        ("GOOG", 0.29, 7.13),
        ("TSLA", 0.41, 3.97),
        ("AMZN", 0.11, 12.59),
    ]

    expected_market_value = 0.0
    for i in range(50):
        ticker, qty, price = small_trades[i % len(small_trades)]
        client.post(
            f"/portfolios/{pid}/transactions",
            json={
                "ticker": ticker,
                "transaction_type": "buy",
                "quantity": qty,
                "price_per_share": price,
            },
        )
        expected_market_value += qty * price

    resp = client.get(f"/portfolios/{pid}")
    assert resp.status_code == 200
    total_market_value = resp.json()["total_market_value"]

    # The total should match the expected value when both are rounded to 2 dp
    assert total_market_value == round(expected_market_value, 2), (
        f"Floating point drift detected: got {total_market_value}, "
        f"expected {round(expected_market_value, 2)}"
    )


def test_calculate_portfolio_value_no_intermediate_rounding():
    """Directly test that calculate_portfolio_value sums before rounding."""
    from app.calculations import calculate_portfolio_value
    from app.models import Holding

    # Craft holdings whose individual market_values would cause drift if
    # rounded before summing: 3 × round(3.335, 2) = 3 × 3.34 = 10.02
    # but round(3 × 3.335, 2) = round(10.005, 2) = 10.0 (banker's rounding)
    # or 10.01 depending on rounding. The key is they must differ.
    holdings = {
        f"T{i}": Holding(
            ticker=f"T{i}",
            quantity=1.0,
            average_cost=0.0,
            current_price=3.335,
            market_value=3.335,
            gain_loss=0.0,
            allocation_pct=0.0,
        )
        for i in range(3)
    }

    result = calculate_portfolio_value(holdings)
    # Should be round(3 * 3.335, 2) = round(10.005, 2) = 10.01
    assert result == round(3 * 3.335, 2)
