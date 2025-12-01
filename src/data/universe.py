"""
Stock universe management module.
"""

from typing import List
from .downloader import get_sp500_tickers


class UniverseManager:
    """Manages stock universe selection."""

    def __init__(self, universe_type: str = "spy_only"):
        """
        Initialize universe manager.

        Args:
            universe_type: Type of universe ("spy_only", "top_10_sp500", "top_50_sp500")
        """
        self.universe_type = universe_type

    def get_tickers(self) -> List[str]:
        """Get list of tickers for the universe."""
        if self.universe_type == "spy_only":
            return ["SPY"]
        elif self.universe_type == "top_10_sp500":
            return get_sp500_tickers(top_n=10)
        elif self.universe_type == "top_50_sp500":
            return get_sp500_tickers(top_n=50)
        else:
            raise ValueError(f"Unknown universe type: {self.universe_type}")
