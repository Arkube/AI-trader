#!/usr/bin/env python3
"""
Automated Training Pipeline with Optuna
────────────────────────────────────────
Trains all models (macro, micro, strategy, RL) with hyperparameter optimization,
walk-forward validation, backtest verification, and versioned model registry.

Usage:
  python scripts/train_all_models.py                    # Full pipeline
  python scripts/train_all_models.py --macro-only       # Only macro model
  python scripts/train_all_models.py --strategy-only    # Only strategy models
  python scripts/train_all_models.py --rl-only          # Only RL agent
  python scripts/train_all_models.py --optuna-trials 50 # Custom Optuna trials
  python scripts/train_all_models.py --no-backtest      # Skip backtest verification
  python scripts/train_all_models.py --fresh            # Ignore existing models, retrain from scratch
"""

import os
import sys
import argparse
import json
import time
import shutil
import warnings
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import optuna
import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger
from database.db import read_sql, get_engine
from features.indicators import compute_all_macro_indicators
from features.micro_features import compute_micro_features
from models.train_model import MacroModelTrainer, MicroModelTrainer
from models.strategy_models import train_strategy_model, train_all_strategy_models, StrategyPredictor
from models.rl_exit_agent import RLExitAgent
from models.dqn_exit_agent import DQNExitAgent
from backtest.tick_replay_backtest import replay_day, get_available_days
from config.settings import (
    FEATURE_COLUMNS_MACRO, FEATURE_COLUMNS_MICRO,
    MACRO_MODEL_PATH, MICRO_MODEL_PATH,
    MODEL_DIR
)

logger = get_logger("train_pipeline")

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_OPTUNA_TRIALS = 30
DEFAULT_N_SPLITS = 5
BACKTEST_VERIFICATION_DAYS = 5  # Days to backtest for verification
MIN_BACKTEST_TRADES = 10        # Minimum trades for valid backtest
MIN_BACKTEST_WIN_RATE = 0.45    # Minimum win rate for model acceptance
MIN_BACKTEST_PROFIT_FACTOR = 1.1 # Minimum profit factor

MODEL_REGISTRY_PATH = Path(MODEL_DIR) / "registry.json"
BACKUP_DIR = Path(MODEL_DIR) / "backups"


# ═══════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════

class ModelRegistry:
    """Manages model versions, metadata, and lineage."""
    
    def __init__(self, registry_path: Path = MODEL_REGISTRY_PATH):
        self.registry_path = registry_path
        self.registry = self._load()
    
    def _load(self) -> Dict:
        if self.registry_path.exists():
            try:
                return json.loads(self.registry_path.read_text())
            except Exception:
                pass
        return {"models": {}, "history": []}
    
    def _save(self):
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(self.registry, indent=2, default=str))
    
    def register_model(
        self,
        model_name: str,
        model_path: str,
        metrics: Dict,
        hyperparams: Dict,
        training_info: Dict
    ):
        """Register a new model version."""
        version = len(self.registry["models"].get(model_name, [])) + 1
        timestamp = datetime.now().isoformat()
        
        entry = {
            "version": version,
            "path": model_path,
            "timestamp": timestamp,
            "metrics": metrics,
            "hyperparams": hyperparams,
            "training_info": training_info,
            "status": "active"
        }
        
        if model_name not in self.registry["models"]:
            self.registry["models"][model_name] = []
        
        # Deactivate previous versions
        for v in self.registry["models"][model_name]:
            v["status"] = "archived"
        
        self.registry["models"][model_name].append(entry)
        self.registry["history"].append({
            "model": model_name,
            "version": version,
            "action": "registered",
            "timestamp": timestamp
        })
        self._save()
        logger.info(f"Registered {model_name} v{version}")
        return version
    
    def get_latest(self, model_name: str) -> Optional[Dict]:
        models = self.registry["models"].get(model_name, [])
        active = [m for m in models if m["status"] == "active"]
        return active[0] if active else None
    
    def get_best_by_metric(self, model_name: str, metric: str, higher_better: bool = True) -> Optional[Dict]:
        models = self.registry["models"].get(model_name, [])
        if not models:
            return None
        valid = [m for m in models if metric in m.get("metrics", {})]
        if not valid:
            return None
        return max(valid, key=lambda x: x["metrics"][metric]) if higher_better else min(valid, key=lambda x: x["metrics"][metric])


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_training_data(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    symbol: str = "NIFTY-I"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare training data from database."""
    
    if end_date is None:
        end_date = date.today() - timedelta(days=1)
    if start_date is None:
        start_date = end_date - timedelta(days=180)  # 6 months default
    
    logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")
    
    # Load minute candles
    candles = read_sql(
        "SELECT timestamp, symbol, open, high, low, close, volume, vwap, oi "
        "FROM minute_candles WHERE symbol = :sym AND timestamp::date BETWEEN :start AND :end "
        "ORDER BY timestamp",
        {"sym": symbol, "start": str(start_date), "end": str(end_date)}
    )
    
    if candles.empty:
        raise ValueError(f"No candle data for {symbol} in date range")
    
    candles['timestamp'] = pd.to_datetime(candles['timestamp'])
    candles = candles.sort_values('timestamp').reset_index(drop=True)
    
    # Load tick data for micro features
    ticks = read_sql(
        "SELECT timestamp, symbol, price, volume, oi, bid_price, ask_price "
        "FROM tick_data WHERE symbol = :sym AND timestamp::date BETWEEN :start AND :end "
        "ORDER BY timestamp",
        {"sym": symbol, "start": str(start_date), "end": str(end_date)}
    )
    
    if not ticks.empty:
        ticks['timestamp'] = pd.to_datetime(ticks['timestamp'])
        ticks = ticks.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Loaded {len(candles)} candles, {len(ticks)} ticks")
    return candles, ticks


def prepare_macro_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Compute macro features and generate labels."""
    logger.info("Computing macro indicators...")
    features = compute_all_macro_indicators(candles)
    
    # Generate labels: forward return over 15 bars > 0.1%
    from models.train_model import generate_macro_labels
    features = generate_macro_labels(features, forward_periods=15, threshold=0.001)
    
    # Drop NaN
    features = features.dropna(subset=FEATURE_COLUMNS_MACRO + ['label'])
    logger.info(f"Macro features: {len(features)} samples, {features['label'].mean():.2%} positive")
    return features


def prepare_micro_features(ticks: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    """Compute micro features from tick data."""
    if ticks.empty:
        logger.warning("No tick data for micro features")
        return pd.DataFrame()
    
    logger.info("Computing micro features...")
    features = compute_micro_features(ticks, candles)
    features = features.dropna(subset=FEATURE_COLUMNS_MICRO + ['label'])
    logger.info(f"Micro features: {len(features)} samples")
    return features


# ═══════════════════════════════════════════════════════════════════════
# OPTUNA OBJECTIVE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def macro_objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> float:
    """Optuna objective for macro model."""
    from xgboost import XGBClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': 0
    }
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        aucs.append(auc)
    
    return np.mean(aucs)


def micro_objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, n_splits: int = 3) -> float:
    """Optuna objective for micro model."""
    from xgboost import XGBClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': 0
    }
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        aucs.append(auc)
    
    return np.mean(aucs)


# ═══════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def train_macro_model(
    candles: pd.DataFrame,
    n_trials: int = DEFAULT_OPTUNA_TRIALS,
    n_splits: int = DEFAULT_N_SPLITS,
    fresh: bool = False
) -> Tuple[Any, Dict, Dict]:
    """Train macro model with Optuna optimization."""
    
    model_path = Path(MACRO_MODEL_PATH)
    if model_path.exists() and not fresh:
        logger.info("Macro model exists, skipping (use --fresh to retrain)")
        return None, {}, {}
    
    logger.info("=" * 60)
    logger.info("TRAINING MACRO MODEL")
    logger.info("=" * 60)
    
    # Prepare data
    features = prepare_macro_features(candles)
    X = features[FEATURE_COLUMNS_MACRO]
    y = features['label']
    
    # Optuna optimization
    logger.info(f"Running Optuna optimization ({n_trials} trials)...")
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: macro_objective(t, X, y, n_splits), n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    best_auc = study.best_value
    logger.info(f"Best AUC: {best_auc:.4f}, Params: {best_params}")
    
    # Train final model on all data
    from xgboost import XGBClassifier
    final_params = {**best_params, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'n_jobs': -1, 'random_state': 42, 'verbosity': 0}
    model = XGBClassifier(**final_params)
    model.fit(X, y)
    
    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Saved macro model to {model_path}")
    
    # Training info
    training_info = {
        'n_samples': len(X),
        'positive_rate': float(y.mean()),
        'n_features': len(FEATURE_COLUMNS_MACRO),
        'optuna_trials': n_trials,
        'cv_splits': n_splits,
        'best_cv_auc': best_auc
    }
    
    return model, best_params, training_info


def train_micro_model(
    ticks: pd.DataFrame,
    candles: pd.DataFrame,
    n_trials: int = DEFAULT_OPTUNA_TRIALS,
    n_splits: int = 3,
    fresh: bool = False
) -> Tuple[Any, Dict, Dict]:
    """Train micro model with Optuna optimization."""
    
    model_path = Path(MICRO_MODEL_PATH)
    if model_path.exists() and not fresh:
        logger.info("Micro model exists, skipping (use --fresh to retrain)")
        return None, {}, {}
    
    if ticks.empty:
        logger.warning("No tick data, skipping micro model")
        return None, {}, {}
    
    logger.info("=" * 60)
    logger.info("TRAINING MICRO MODEL")
    logger.info("=" * 60)
    
    features = prepare_micro_features(ticks, candles)
    if features.empty:
        logger.warning("No micro features generated")
        return None, {}, {}
    
    X = features[FEATURE_COLUMNS_MICRO]
    y = features['label']
    
    logger.info(f"Running Optuna optimization ({n_trials} trials)...")
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: micro_objective(t, X, y, n_splits), n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    best_auc = study.best_value
    logger.info(f"Best AUC: {best_auc:.4f}, Params: {best_params}")
    
    from xgboost import XGBClassifier
    final_params = {**best_params, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'n_jobs': -1, 'random_state': 42, 'verbosity': 0}
    model = XGBClassifier(**final_params)
    model.fit(X, y)
    
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Saved micro model to {model_path}")
    
    training_info = {
        'n_samples': len(X),
        'positive_rate': float(y.mean()),
        'n_features': len(FEATURE_COLUMNS_MICRO),
        'optuna_trials': n_trials,
        'cv_splits': n_splits,
        'best_cv_auc': best_auc
    }
    
    return model, best_params, training_info


def train_strategy_models(
    candles: pd.DataFrame,
    fresh: bool = False
) -> Dict[str, Dict]:
    """Train per-strategy models using the strategy_models module."""
    
    logger.info("=" * 60)
    logger.info("TRAINING STRATEGY MODELS")
    logger.info("=" * 60)
    
    # First compute macro features on candles
    logger.info("Computing macro features for strategy training...")
    features_df = prepare_macro_features(candles)
    
    if features_df.empty:
        logger.warning("No macro features generated")
        return {}
    
    # Use the strategy_models module to train all strategy models
    from models.strategy_models import train_all_strategy_models
    
    results = train_all_strategy_models(features_df)
    
    # Convert to expected format
    formatted_results = {}
    for strategy_name, metrics in results.items():
        formatted_results[strategy_name] = {
            'metrics': metrics,
            'model_path': str(Path(MODEL_DIR) / "strategy" / f"{strategy_name}_model.pkl")
        }
        logger.info(f"Trained {strategy_name}: {metrics}")
    
    return formatted_results


def train_rl_agent(
    n_epochs: int = 50,
    fresh: bool = False
) -> Tuple[Any, Dict]:
    """Train RL exit agent on journey data."""
    
    model_path = Path(MODEL_DIR) / "rl_exit_agent.pkl"
    if model_path.exists() and not fresh:
        logger.info("RL agent exists, skipping (use --fresh to retrain)")
        return None, {}
    
    logger.info("=" * 60)
    logger.info("TRAINING RL EXIT AGENT")
    logger.info("=" * 60)
    
    agent = RLExitAgent()
    
    # Load journey data
    journey_files = list(Path("backtest_results").glob("journeys_*_risk.json"))
    all_journeys = []
    
    for jf in journey_files:
        try:
            with open(jf) as f:
                journeys = json.load(f)
                all_journeys.extend(journeys)
        except Exception as e:
            logger.warning(f"Failed to load {jf}: {e}")
    
    if not all_journeys:
        logger.warning("No journey data found for RL training")
        return None, {}
    
    logger.info(f"Training RL agent on {len(all_journeys)} journeys for {n_epochs} epochs...")
    
    # Train
    agent.train(all_journeys, n_epochs=n_epochs)
    
    # Save
    agent.save(str(model_path))
    logger.info(f"Saved RL agent to {model_path}")
    
    training_info = {
        'n_journeys': len(all_journeys),
        'n_epochs': n_epochs,
        'q_table_size': len(agent.q_table) if hasattr(agent, 'q_table') else 0
    }
    
    return agent, training_info


def train_dqn_agent(
    n_episodes: int = 1000,
    fresh: bool = False
) -> Tuple[Any, Dict]:
    """Train DQN exit agent."""
    
    model_path = Path(MODEL_DIR) / "dqn_exit_agent.pt"
    if model_path.exists() and not fresh:
        logger.info("DQN agent exists, skipping (use --fresh to retrain)")
        return None, {}
    
    logger.info("=" * 60)
    logger.info("TRAINING DQN EXIT AGENT")
    logger.info("=" * 60)
    
    agent = DQNExitAgent()
    
    # Load journey data
    journey_files = list(Path("backtest_results").glob("journeys_*_risk.json"))
    all_journeys = []
    
    for jf in journey_files:
        try:
            with open(jf) as f:
                journeys = json.load(f)
                all_journeys.extend(journeys)
        except Exception as e:
            logger.warning(f"Failed to load {jf}: {e}")
    
    if not all_journeys:
        logger.warning("No journey data found for DQN training")
        return None, {}
    
    logger.info(f"Training DQN agent on {len(all_journeys)} journeys for {n_episodes} episodes...")
    
    # Train
    agent.train(all_journeys, n_episodes=n_episodes)
    
    # Save
    agent.save(str(model_path))
    logger.info(f"Saved DQN agent to {model_path}")
    
    training_info = {
        'n_journeys': len(all_journeys),
        'n_episodes': n_episodes,
        'epsilon': agent.epsilon if hasattr(agent, 'epsilon') else 0
    }
    
    return agent, training_info


# ═══════════════════════════════════════════════════════════════════════
# BACKTEST VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

def verify_models_via_backtest(
    risk_profiles: List[str] = ["high", "medium", "low"],
    n_days: int = BACKTEST_VERIFICATION_DAYS
) -> Dict[str, Dict]:
    """Run quick backtest to verify trained models."""
    
    logger.info("=" * 60)
    logger.info("BACKTEST VERIFICATION")
    logger.info("=" * 60)
    
    results = {}
    available_days = get_available_days()
    
    if not available_days:
        logger.warning("No days available for backtest verification")
        return results
    
    # Use most recent days
    test_days = available_days[-n_days:]
    logger.info(f"Verifying on {len(test_days)} days: {test_days}")
    
    for risk in risk_profiles:
        logger.info(f"Verifying {risk} risk model...")
        
        try:
            # Run backtest for this risk profile
            all_trades = []
            for test_day in test_days:
                day_trades = replay_day(
                    replay_date=test_day,
                    risk_profile=risk,
                    verbose=False
                )
                all_trades.extend(day_trades)
            
            if not all_trades:
                logger.warning(f"No trades generated for {risk} risk")
                results[risk] = {"trades": 0, "win_rate": 0, "profit_factor": 0, "passed": False}
                continue
            
            # Calculate metrics
            wins = sum(1 for t in all_trades if t.get('pnl', 0) > 0)
            losses = sum(1 for t in all_trades if t.get('pnl', 0) <= 0)
            total_pnl = sum(t.get('pnl', 0) for t in all_trades)
            gross_profit = sum(t.get('pnl', 0) for t in all_trades if t.get('pnl', 0) > 0)
            gross_loss = abs(sum(t.get('pnl', 0) for t in all_trades if t.get('pnl', 0) < 0))
            
            win_rate = wins / len(all_trades) if all_trades else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            passed = (
                len(all_trades) >= MIN_BACKTEST_TRADES and
                win_rate >= MIN_BACKTEST_WIN_RATE and
                profit_factor >= MIN_BACKTEST_PROFIT_FACTOR
            )
            
            results[risk] = {
                "trades": len(all_trades),
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "profit_factor": profit_factor,
                "passed": passed
            }
            
            status = "PASSED" if passed else "FAILED"
            logger.info(f"  {risk}: {len(all_trades)} trades, WR={win_rate:.2%}, PF={profit_factor:.2f} [{status}]")
            
        except Exception as e:
            logger.error(f"Backtest verification failed for {risk}: {e}")
            results[risk] = {"error": str(e), "passed": False}
    
    return results


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def backup_existing_models():
    """Backup existing models before training."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / timestamp
    backup_path.mkdir(parents=True, exist_ok=True)
    
    model_files = [
        MACRO_MODEL_PATH,
        MICRO_MODEL_PATH,
        str(Path(MODEL_DIR) / "rl_exit_agent.pkl"),
        str(Path(MODEL_DIR) / "dqn_exit_agent.pt"),
    ]
    
    for mf in model_files:
        src = Path(mf)
        if src.exists():
            dst = backup_path / src.name
            shutil.copy2(src, dst)
            logger.info(f"Backed up {src.name} to {dst}")
    
    # Backup strategy models
    strategy_dir = Path(MODEL_DIR) / "strategy"
    if strategy_dir.exists():
        for sm in strategy_dir.glob("*.pkl"):
            dst = backup_path / sm.name
            shutil.copy2(sm, dst)
    
    logger.info(f"Backed up models to {backup_path}")
    return backup_path


def main():
    parser = argparse.ArgumentParser(description="Automated Training Pipeline")
    parser.add_argument("--macro-only", action="store_true", help="Train only macro model")
    parser.add_argument("--micro-only", action="store_true", help="Train only micro model")
    parser.add_argument("--strategy-only", action="store_true", help="Train only strategy models")
    parser.add_argument("--rl-only", action="store_true", help="Train only RL agents")
    parser.add_argument("--optuna-trials", type=int, default=DEFAULT_OPTUNA_TRIALS, help="Optuna trials per model")
    parser.add_argument("--cv-splits", type=int, default=DEFAULT_N_SPLITS, help="CV splits for walk-forward")
    parser.add_argument("--rl-epochs", type=int, default=50, help="RL training epochs")
    parser.add_argument("--dqn-episodes", type=int, default=1000, help="DQN training episodes")
    parser.add_argument("--no-backtest", action="store_true", help="Skip backtest verification")
    parser.add_argument("--fresh", action="store_true", help="Retrain all models from scratch")
    parser.add_argument("--no-backup", action="store_true", help="Skip model backup")
    parser.add_argument("--start-date", type=str, help="Training start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="Training end date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("AUTOMATED TRAINING PIPELINE STARTED")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else None
    
    # Backup existing models
    if not args.no_backup:
        backup_existing_models()
    
    # Initialize registry
    registry = ModelRegistry()
    
    # Load data
    try:
        candles, ticks = load_training_data(start_date, end_date)
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        return 1
    
    # Load strategy data
    strategy_data = prepare_strategy_data()
    
    # Track results
    all_results = {}
    
    # ═══════════════════════════════════════════════════════════════
    # TRAIN MACRO MODEL
    # ══════════════════════════════════════════════════════════════
    if not args.micro_only and not args.strategy_only and not args.rl_only:
        model, params, info = train_macro_model(
            candles, args.optuna_trials, args.cv_splits, args.fresh
        )
        if model:
            registry.register_model("macro", MACRO_MODEL_PATH, 
                                  {"cv_auc": info.get("best_cv_auc", 0)}, params, info)
            all_results["macro"] = info
    
    # ═══════════════════════════════════════════════════════════════
    # TRAIN MICRO MODEL
    # ══════════════════════════════════════════════════════════════
    if not args.macro_only and not args.strategy_only and not args.rl_only:
        model, params, info = train_micro_model(
            ticks, candles, args.optuna_trials, 3, args.fresh
        )
        if model:
            registry.register_model("micro", MICRO_MODEL_PATH,
                                  {"cv_auc": info.get("best_cv_auc", 0)}, params, info)
            all_results["micro"] = info
    
    # ══════════════════════════════════════════════════════════════
    # TRAIN STRATEGY MODELS
    # ══════════════════════════════════════════════════════════════
    if not args.macro_only and not args.micro_only and not args.rl_only:
        strategy_results = train_strategy_models(candles, args.fresh)
        for strategy_name, info in strategy_results.items():
            model_path = info.get('model_path', '')
            metrics = info.get('metrics', {})
            registry.register_model(f"strategy_{strategy_name}", model_path,
                                  metrics, {}, info)
            all_results[f"strategy_{strategy_name}"] = info
    
    # ══════════════════════════════════════════════════════════════
    # TRAIN RL AGENTS
    # ══════════════════════════════════════════════════════════════
    if not args.macro_only and not args.micro_only and not args.strategy_only:
        agent, info = train_rl_agent(args.rl_epochs, args.fresh)
        if agent:
            registry.register_model("rl_exit_agent", str(Path(MODEL_DIR) / "rl_exit_agent.pkl"),
                                  {}, {}, info)
            all_results["rl_exit_agent"] = info
        
        agent, info = train_dqn_agent(args.dqn_episodes, args.fresh)
        if agent:
            registry.register_model("dqn_exit_agent", str(Path(MODEL_DIR) / "dqn_exit_agent.pt"),
                                  {}, {}, info)
            all_results["dqn_exit_agent"] = info
    
    # ══════════════════════════════════════════════════════════════
    # BACKTEST VERIFICATION
    # ══════════════════════════════════════════════════════════════
    if not args.no_backtest:
        verification = verify_models_via_backtest()
        all_results["backtest_verification"] = verification
        
        # Log summary
        logger.info("=" * 60)
        logger.info("BACKTEST VERIFICATION SUMMARY")
        logger.info("=" * 60)
        for risk, metrics in verification.items():
            if "error" in metrics:
                logger.info(f"  {risk}: ERROR - {metrics['error']}")
            else:
                status = "✅ PASSED" if metrics.get("passed") else "❌ FAILED"
                logger.info(f"  {risk}: {metrics['trades']} trades, WR={metrics['win_rate']:.2%}, PF={metrics['profit_factor']:.2f} {status}")
    
    # ══════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    logger.info(f"Models trained: {len([k for k in all_results if not k.startswith('backtest')])}")
    
    for model_name, info in all_results.items():
        if model_name == "backtest_verification":
            continue
        logger.info(f"  {model_name}: {info}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())