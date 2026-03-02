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
# Floating-point accumulation regression tests (#8)
# ---------------------------------------------------------------------------


def test_portfolio_value_no_floating_point_drift_many_trades():
    """After many small fractional transactions the total market value must
    not accumulate floating-point noise (issue #8)."""
    pid = client.post(
        "/portfolios", json={"name": "FP Test", "owner": "tester"}
    ).json()["id"]

    # Add 50 small fractional buys that are prone to fp drift
    small_trades = [
        ("AAPL", 0.33, 10.33),
        ("MSFT", 0.17, 5.71),
        ("GOOG", 0.29, 12.07),
        ("TSLA", 0.41, 7.89),
        ("AMZN", 0.11, 3.33),
    ]

    for i in range(50):
        ticker, qty, price = small_trades[i % len(small_trades)]
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
    total = summary["total_market_value"]

    # The value must be a clean 2-decimal-place number (no trailing noise)
    assert total == round(total, 2), (
        f"Floating-point drift detected: total_market_value={total!r}"
    )


def test_calculate_portfolio_value_rounds_at_end():
    """Unit test: calculate_portfolio_value should sum raw values then round."""
    from app.calculations import calculate_portfolio_value
    from app.models import Holding

    # Construct holdings whose individual market_values would each round
    # differently if rounded before summing.
    holdings = {
        "A": Holding(
            ticker="A", quantity=1.0, average_cost=0.0,
            current_price=1.005, market_value=1.005,
            gain_loss=0.0, allocation_pct=0.0,
        ),
        "B": Holding(
            ticker="B", quantity=1.0, average_cost=0.0,
            current_price=1.005, market_value=1.005,
            gain_loss=0.0, allocation_pct=0.0,
        ),
        "C": Holding(
            ticker="C", quantity=1.0, average_cost=0.0,
            current_price=1.005, market_value=1.005,
            gain_loss=0.0, allocation_pct=0.0,
        ),
    }

    result = calculate_portfolio_value(holdings)

    # 3 * 1.005 = 3.015 → round(3.015, 2) == 3.01 or 3.02 depending on
    # banker's rounding, but crucially NOT 3 * round(1.005, 2) which would
    # give 3 * 1.0 = 3.0 or 3 * 1.01 = 3.03.
    # The key assertion: the result equals round(sum_of_raw_values, 2).
    expected = round(1.005 + 1.005 + 1.005, 2)
    assert result == expected
