"""
Data download script.

Usage:
    python scripts/download_data.py --config config/config.yaml
"""

import argparse
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.downloader import DataDownloader
from data.universe import UniverseManager
from utils.logging import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Get universe
    universe = UniverseManager(config['data']['universe'])
    tickers = universe.get_tickers()
    logger.info(f"Universe: {config['data']['universe']}")
    logger.info(f"Tickers: {tickers}")

    # Download data
    downloader = DataDownloader(cache_dir=Path("data/cache"))

    logger.info(f"Downloading data from {config['data']['start_date']} to {config['data']['end_date']}")

    # Download main tickers
    main_data = downloader.download_universe(
        tickers,
        config['data']['start_date'],
        config['data']['end_date'],
        use_cache=not args.force
    )

    # Download market context data
    market_data = downloader.download_universe(
        config['data']['market_tickers'],
        config['data']['start_date'],
        config['data']['end_date'],
        use_cache=not args.force
    )

    logger.info(f"Downloaded {len(main_data)} main tickers")
    logger.info(f"Downloaded {len(market_data)} market tickers")
    logger.info("Data download complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--force", action="store_true", help="Force re-download (ignore cache)")
    args = parser.parse_args()
    main(args)
