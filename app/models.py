from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TransactionType(str, Enum):
    buy = "buy"
    sell = "sell"


class Transaction(BaseModel):
    id: str
    portfolio_id: str
    ticker: str
    transaction_type: TransactionType
    quantity: float = Field(..., gt=0)
    price_per_share: float = Field(..., gt=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TransactionCreate(BaseModel):
    ticker: str
    transaction_type: TransactionType
    quantity: float = Field(..., gt=0)
    price_per_share: float = Field(..., gt=0)


class Holding(BaseModel):
    ticker: str
    quantity: float
    average_cost: float
    current_price: float
    market_value: float
    gain_loss: float
    allocation_pct: float


class Portfolio(BaseModel):
    id: str
    name: str
    owner: str
    holdings: Dict[str, Holding] = Field(default_factory=dict)
    transactions: List[Transaction] = Field(default_factory=list)
    cash_balance: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PortfolioCreate(BaseModel):
    name: str
    owner: str
    cash_balance: float = Field(default=0.0, ge=0)


class PortfolioSummary(BaseModel):
    id: str
    name: str
    owner: str
    total_market_value: float
    total_cost_basis: float
    total_gain_loss: float
    cash_balance: float
    holdings: Dict[str, Holding]


class PriceUpdate(BaseModel):
    """Map of ticker -> current price used when fetching portfolio summary."""
    prices: Dict[str, float]
