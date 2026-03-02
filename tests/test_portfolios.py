"""Tests for the FinServ portfolio tracker API.

These tests cover happy-path behaviour and basic validation.
The buggy code paths in calculations.py (rounding drift, off-by-one
cost basis) are intentionally not covered here so the automated
triage system can discover them.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import data
from app.calculations import asset_allocation_percentages
from app.main import app
from app.models import Holding

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
# asset_allocation_percentages – division-by-zero fix
# ---------------------------------------------------------------------------


def test_allocation_empty_holdings():
    """Empty holdings dict should return an empty dict, not raise."""
    result = asset_allocation_percentages({})
    assert result == {}


def test_allocation_zero_value_holdings():
    """Holdings with zero market value should all get 0.0%."""
    holdings = {
        "AAPL": Holding(
            ticker="AAPL", quantity=0, average_cost=0,
            current_price=0, market_value=0, gain_loss=0, allocation_pct=0,
        ),
        "MSFT": Holding(
            ticker="MSFT", quantity=0, average_cost=0,
            current_price=0, market_value=0, gain_loss=0, allocation_pct=0,
        ),
    }
    result = asset_allocation_percentages(holdings)
    assert result == {"AAPL": 0.0, "MSFT": 0.0}


def test_allocation_normal_holdings():
    """Normal holdings should return correct percentages."""
    holdings = {
        "AAPL": Holding(
            ticker="AAPL", quantity=10, average_cost=100,
            current_price=150, market_value=1500, gain_loss=500, allocation_pct=0,
        ),
        "MSFT": Holding(
            ticker="MSFT", quantity=5, average_cost=200,
            current_price=300, market_value=1500, gain_loss=500, allocation_pct=0,
        ),
    }
    result = asset_allocation_percentages(holdings)
    assert result["AAPL"] == pytest.approx(50.0)
    assert result["MSFT"] == pytest.approx(50.0)


def test_buy_then_sell_all_no_crash():
    """Selling all shares should not crash due to zero-value allocation."""
    pid = client.post("/portfolios", json={"name": "P6", "owner": "hank"}).json()["id"]
    client.post(
        f"/portfolios/{pid}/transactions",
        json={"ticker": "TSLA", "transaction_type": "buy", "quantity": 5, "price_per_share": 200.0},
    )
    resp = client.post(
        f"/portfolios/{pid}/transactions",
        json={"ticker": "TSLA", "transaction_type": "sell", "quantity": 5, "price_per_share": 200.0},
    )
    assert resp.status_code == 201
    holdings = client.get(f"/portfolios/{pid}").json()["holdings"]
    assert holdings["TSLA"]["allocation_pct"] == 0.0
