"""
Market Prediction Engine Configuration
Centralized configuration management for the autonomous self-healing system.
"""
import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for ML models"""
    model_type: str = "ensemble"  # ensemble, lstm, gradient_boosting
    training_window_days: int = 90
    validation_split: float = 0.2
    retraining_threshold: float = 0.85  # Accuracy threshold for retraining
    ensemble_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                "lstm": 0.4,
                "gradient_boosting": 0.35,
                "statistical": 0.25
            }


@dataclass
class DataConfig:
    """Configuration for data collection and preprocessing"""
    symbols: List[str] = None
    timeframes: List[str] = None
    features: List[str] = None
    max_retries: int = 3
    retry_delay_seconds: int = 5
    cache_ttl_minutes: int = 15
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        if self.timeframes is None:
            self.timeframes = ["1h", "4h", "1d"]
        if self.features is None:
            self.features = [
                "open", "high", "low", "close", "volume",
                "rsi_14", "macd", "bollinger_upper", "bollinger_lower",
                "volume_ma_20", "price_change_24h"
            ]


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection and self-healing"""
    z_score_threshold: float = 3.0
    prediction_error_threshold: float = 0.15  # 15% error threshold
    consecutive_failures_before_reset: int = 5
    health_check_interval_minutes: int = 30
    auto_recovery_enabled: bool = True


class MarketEngineConfig:
    """Main configuration manager for the prediction engine"""
    
    def __init__(self, config_path: str = "config/market_engine_config.json"):
        self.config_path = config_path
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        self.anomaly_config = AnomalyConfig()
        self.firebase_config = self._load_firebase_config()
        self._ensure_config_directory()
        
        logger.info("Market Engine Configuration initialized")
    
    def _ensure_config_directory(self) -> None:
        """Ensure config directory exists"""
        config_dir = os.path.dirname(self.config_path)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)
            logger.info(f"Created config directory: {config_dir}")
    
    def _load_firebase_config(self) -> Dict[str, Any]:
        """Load Firebase configuration from environment or file"""
        firebase_config = {
            "project_id": os.getenv("FIREBASE_PROJECT_ID", "market-prediction-engine"),
            "database_url": os.getenv("FIREBASE_D