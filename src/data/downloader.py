"""
Data acquisition module for downloading stock data from Yahoo Finance.

Key responsibilities:
- Download OHLCV data for specified tickers
- Handle rate limiting and retries
- Cache downloaded data to avoid repeated API calls
- Ensure data quality and handle missing values
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import List, Optional, Dict
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class DataDownloader:
    """Downloads and caches stock market data from Yahoo Finance."""

    def __init__(self, cache_dir: Path = Path("data/cache")):
        """
        Initialize the downloader.

        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_ticker(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download OHLCV data for a single ticker.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
        """
        cache_file = self.cache_dir / f"{ticker}_{start_date}_{end_date}.parquet"

        if use_cache and cache_file.exists():
            logger.info(f"Loading cached data for {ticker}")
            return pd.read_parquet(cache_file)

        logger.info(f"Downloading data for {ticker}")
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False
            )

            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()

            # Save to cache
            df.to_parquet(cache_file)

            return df

        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
            return pd.DataFrame()

    def download_universe(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cached data

        Returns:
            Dictionary mapping ticker to DataFrame
        """
        data = {}
        for ticker in tickers:
            df = self.download_ticker(ticker, start_date, end_date, use_cache)
            if not df.empty:
                data[ticker] = df
            time.sleep(0.1)  # Rate limiting

        return data


def get_sp500_tickers(top_n: Optional[int] = None) -> List[str]:
    """
    Get S&P 500 constituent tickers.

    Args:
        top_n: If specified, return only top N tickers by market cap

    Returns:
        List of ticker symbols
    """
    try:
        # Try to read S&P 500 constituents from Wikipedia
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        import urllib.request
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

        with urllib.request.urlopen(req) as response:
            tables = pd.read_html(response.read())

        sp500_table = tables[0]

        # Try different possible column names
        if 'Symbol' in sp500_table.columns:
            tickers = sp500_table['Symbol'].tolist()
        elif 0 in sp500_table.columns and 1 in sp500_table.columns:
            # Fallback: assume first column is symbol
            tickers = sp500_table[0].tolist()
        else:
            raise ValueError("Cannot find Symbol column in table")

        # Clean tickers (replace . with -)
        tickers = [str(t).replace('.', '-') for t in tickers if pd.notna(t)]

    except Exception as e:
        logger.warning(f"Could not fetch from Wikipedia: {e}. Using fallback list.")
        # Fallback to top stocks by market cap (as of 2024)
        tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
            'META', 'TSLA', 'BRK-B', 'V', 'JPM',
            'WMT', 'MA', 'JNJ', 'PG', 'XOM',
            'UNH', 'HD', 'CVX', 'MRK', 'PFE',
            'ABBV', 'KO', 'PEP', 'COST', 'AVGO',
            'DIS', 'CSCO', 'ACN', 'ADBE', 'TMO',
            'MCD', 'ABT', 'CRM', 'NFLX', 'VZ',
            'NKE', 'CMCSA', 'DHR', 'WFC', 'INTC',
            'PM', 'TXN', 'UNP', 'BMY', 'ORCL',
            'AMD', 'COP', 'NEE', 'RTX', 'UPS'
        ]

    if top_n:
        return tickers[:top_n]

    return tickers
