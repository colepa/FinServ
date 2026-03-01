# FinServ — Portfolio Tracker API

A small but realistic **FastAPI** application that models a personal investment
portfolio tracker.  It is used to demo an automated issue-triage and fix system.

---

## Features

| Endpoint | Description |
|---|---|
| `POST /portfolios` | Create a new portfolio |
| `GET /portfolios/{id}` | Retrieve portfolio summary (holdings, gain/loss, allocation) |
| `POST /portfolios/{id}/transactions` | Record a buy or sell transaction |

Holdings, transactions, and cash balances are kept in an **in-memory store**
(no database required).

---

## Project layout

```
app/
  main.py          – FastAPI application and route handlers
  models.py        – Pydantic models (Portfolio, Transaction, Holding, …)
  calculations.py  – Portfolio value, gain/loss, and allocation helpers
  data.py          – In-memory data store
tests/
  test_portfolios.py
requirements.txt
```

---

## Quick start

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Interactive API docs are available at <http://localhost:8000/docs>.

---

## Running tests

```bash
pytest -v
```

---

## Known issues (intentional — for triage demo)

Three subtle bugs live in `app/calculations.py`:

1. **Rounding drift** — `calculate_portfolio_value` rounds each holding's
   market value *before* accumulation instead of rounding the final sum,
   causing cumulative drift with many fractional positions.

2. **Off-by-one in cost basis** — `compute_gain_loss` slices
   `transactions[1:]` instead of `transactions[0:]`, silently dropping the
   first buy from the average-cost calculation and overstating gain/loss.

3. **Division-by-zero on empty portfolio** — `asset_allocation_percentages`
   divides by `total_value` without guarding against a zero-value (empty)
   portfolio, raising `ZeroDivisionError` at runtime.

These bugs are **intentional**.  Do not fix them here — the automated triage
system is responsible for detecting and patching them.

