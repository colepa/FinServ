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


def test_many_small_transactions_no_rounding_drift():
    """Adding many small fractional transactions should not accumulate
    floating-point rounding artifacts in the total market value."""
    pid = client.post(
        "/portfolios", json={"name": "Drift Test", "owner": "tester"}
    ).json()["id"]

    # A set of small fractional transactions designed to expose rounding drift
    transactions = [
        {"ticker": "AAA", "transaction_type": "buy", "quantity": 0.33, "price_per_share": 10.33},
        {"ticker": "BBB", "transaction_type": "buy", "quantity": 0.17, "price_per_share": 5.71},
        {"ticker": "CCC", "transaction_type": "buy", "quantity": 0.91, "price_per_share": 3.14},
        {"ticker": "AAA", "transaction_type": "buy", "quantity": 0.47, "price_per_share": 10.50},
        {"ticker": "BBB", "transaction_type": "buy", "quantity": 0.83, "price_per_share": 5.99},
        {"ticker": "CCC", "transaction_type": "buy", "quantity": 0.29, "price_per_share": 3.07},
        {"ticker": "DDD", "transaction_type": "buy", "quantity": 0.61, "price_per_share": 7.77},
        {"ticker": "AAA", "transaction_type": "buy", "quantity": 0.13, "price_per_share": 10.11},
        {"ticker": "BBB", "transaction_type": "buy", "quantity": 0.53, "price_per_share": 6.03},
        {"ticker": "DDD", "transaction_type": "buy", "quantity": 0.37, "price_per_share": 7.89},
    ]

    # Add many rounds of similar transactions (30 total)
    for _ in range(3):
        for tx in transactions:
            resp = client.post(f"/portfolios/{pid}/transactions", json=tx)
            assert resp.status_code == 201

    summary = client.get(f"/portfolios/{pid}").json()
    total_mv = summary["total_market_value"]

    # The total should be a clean number with at most 2 decimal places
    # (i.e. no floating-point noise like 4312.004999999)
    assert total_mv == round(total_mv, 2), (
        f"Total market value {total_mv} has floating-point rounding artifacts"
    )


def test_calculate_portfolio_value_no_intermediate_rounding():
    """Unit test: calculate_portfolio_value should sum raw values then round,
    not round each holding individually."""
    from app.calculations import calculate_portfolio_value
    from app.models import Holding

    # Create holdings whose individual market_values would round differently
    # if rounded before summing vs. after summing.
    # e.g. 3 holdings each with market_value = 1.005
    # round(1.005,2) = 1.0 (or 1.01 depending on banker's rounding)
    # but sum = 3.015, round(3.015,2) = 3.01 or 3.02
    holdings = {
        f"T{i}": Holding(
            ticker=f"T{i}",
            quantity=0.33,
            average_cost=10.0,
            current_price=10.33,
            market_value=0.33 * 10.33,  # 3.4089
            gain_loss=0.0,
            allocation_pct=0.0,
        )
        for i in range(50)
    }

    result = calculate_portfolio_value(holdings)
    # Should be the sum of all raw values, rounded once at the end
    raw_sum = sum(0.33 * 10.33 for _ in range(50))
    expected = round(raw_sum, 2)
    assert result == expected, (
        f"Expected {expected}, got {result}. "
        "Intermediate rounding is still being applied."
    )
