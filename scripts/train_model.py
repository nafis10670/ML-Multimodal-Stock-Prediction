"""
Model training script.

Usage:
    python scripts/train_model.py --model ridge --config config/config.yaml
"""

import argparse
import yaml
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.downloader import DataDownloader
from data.universe import UniverseManager
from features.feature_pipeline import FeaturePipeline
from validation.walk_forward import WalkForwardValidator, ExpandingWindowScaler
from models.baselines.naive import NaivePredictor, MomentumPredictor
from models.baselines.linear import RidgePredictor, LassoPredictor
from models.tree_models.gradient_boosting import XGBoostPredictor, LightGBMPredictor
from metrics.prediction import compute_all_metrics
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
    main_ticker = tickers[0]  # Use first ticker for now

    logger.info(f"Training on: {main_ticker}")

    # Download data
    downloader = DataDownloader(cache_dir=Path("data/cache"))
    data = downloader.download_ticker(
        main_ticker,
        config['data']['start_date'],
        config['data']['end_date']
    )

    if data.empty:
        logger.error(f"No data for {main_ticker}")
        return

    # Download market data
    market_data = downloader.download_universe(
        config['data']['market_tickers'],
        config['data']['start_date'],
        config['data']['end_date']
    )

    # Compute features
    logger.info("Computing features...")
    pipeline = FeaturePipeline(
        rolling_windows=config['features']['rolling_windows'],
        return_lags=config['features']['return_lags']
    )
    features = pipeline.create_features(data, market_data)

    # Create target
    horizon = config['data']['horizon']
    if config['data']['target'] == 'log_return':
        target = features['log_return'].shift(-horizon)
    else:
        target = np.sign(features['log_return'].shift(-horizon))

    # Drop NaN rows
    valid_idx = features.dropna().index.intersection(target.dropna().index)
    features = features.loc[valid_idx]
    target = target.loc[valid_idx]

    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Target shape: {target.shape}")

    # Drop target from features
    feature_cols = [col for col in features.columns if col not in ['log_return', 'simple_return']]
    X = features[feature_cols]

    # Setup validation
    logger.info("Setting up walk-forward validation...")
    validator = WalkForwardValidator(
        train_years=config['validation']['train_years'],
        val_years=config['validation']['val_years'],
        test_years=config['validation']['test_years'],
        step_months=config['validation']['step_months'],
        embargo_days=config['validation']['embargo_days']
    )

    splits = validator.generate_splits(
        X.index.min(),
        X.index.max()
    )

    logger.info(f"Generated {len(splits)} splits")

    # Select model
    if args.model == 'naive':
        model = NaivePredictor()
    elif args.model == 'momentum':
        model = MomentumPredictor()
    elif args.model == 'ridge':
        model = RidgePredictor()
    elif args.model == 'lasso':
        model = LassoPredictor()
    elif args.model == 'xgboost':
        model = XGBoostPredictor()
    elif args.model == 'lightgbm':
        model = LightGBMPredictor()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Train and evaluate
    all_predictions = []
    all_actuals = []

    for i, split in enumerate(splits):
        logger.info(f"Processing split {i+1}/{len(splits)}: {split}")

        # Get split data
        X_train, y_train, X_val, y_val, X_test, y_test = validator.split_data(
            X, target, split
        )

        if len(X_train) == 0 or len(X_test) == 0:
            logger.warning(f"Empty split, skipping")
            continue

        # Scale features
        scaler = ExpandingWindowScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert to numpy for models
        X_train_np = X_train_scaled.values
        y_train_np = y_train.values
        X_test_np = X_test_scaled.values

        # Train model
        logger.info(f"Training {args.model}...")
        model.fit(X_train_np, y_train_np)

        # Predict on test
        predictions = model.predict(X_test_np)

        # Store results
        all_predictions.extend(predictions)
        all_actuals.extend(y_test.values)

    # Compute final metrics
    predictions = np.array(all_predictions)
    actuals = np.array(all_actuals)

    metrics = compute_all_metrics(actuals, predictions)

    logger.info("\n" + "="*50)
    logger.info(f"Final Metrics for {args.model}:")
    logger.info("="*50)
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")
    logger.info("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ridge", choices=[
        'naive', 'momentum', 'ridge', 'lasso', 'xgboost', 'lightgbm'
    ])
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    main(args)
