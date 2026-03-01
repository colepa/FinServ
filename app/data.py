"""In-memory data store for portfolios."""
from typing import Dict

from app.models import Portfolio

# Simple dict acting as our "database"
portfolios: Dict[str, Portfolio] = {}
