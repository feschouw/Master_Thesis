"""
Temporal Fusion Transformer (TFT) model wrapper with fixed prediction handling.
"""

from typing import Tuple
import numpy as np
from darts import TimeSeries
from darts.models import TFTModel
from darts.metrics import mse

class TFTModelWrapper:
    def __init__(
        self,
        config: dict,
        target_variable: str,
        device: str = 'mps',
        precision: str = '32-true'
    ):
        self.config = config
        self.target_variable = target_variable
        self.model = TFTModel(
            model_name=f"TFT_{target_variable}",
            input_chunk_length=config['input_chunk_length'],
            output_chunk_length=config['output_chunk_length'],
            hidden_size=config['hidden_size'],
            lstm_layers=config['lstm_layers'],
            num_attention_heads=config['num_attention_heads'],
            dropout=config['dropout'],
            batch_size=config['batch_size'],
            n_epochs=config['n_epochs'],
            add_relative_index=True,
            force_reset=True,
            pl_trainer_kwargs={
                'accelerator': device,
                'precision': precision
            }
        )

    def prepare_data(
        self, 
        target: TimeSeries,
        covariates: TimeSeries,
        train_test_split: float
    ) -> Tuple[TimeSeries, TimeSeries, TimeSeries, TimeSeries]:
        """Split data into training and testing sets."""
        split_point = int(train_test_split * len(target))
        
        train_target = target[:split_point]
        test_target = target[split_point:]
        
        # For training, we only need covariates up to the split point
        train_covariates = covariates[:split_point]
        
        # For testing, we need covariates from start through the entire prediction horizon
        test_covariates = covariates[:split_point + len(test_target)]
        
        return train_target, test_target, train_covariates, test_covariates

    def train(self, train_target: TimeSeries, train_covariates: TimeSeries):
        """Train the TFT model."""
        self.model.fit(
            train_target,
            past_covariates=train_covariates,
            future_covariates=train_covariates
        )

    def predict(
        self,
        n: int,
        train_target: TimeSeries,
        covariates: TimeSeries,
        num_samples: int = 100
    ) -> TimeSeries:
        """Generate predictions using the trained model."""
        predictions = self.model.predict(
            n=n,
            series=train_target,
            past_covariates=covariates,
            future_covariates=covariates,
            num_samples=num_samples,
            show_warnings=False
        )
        return predictions

    def calculate_prediction_intervals(
        self,
        predictions: TimeSeries,
        test_target: TimeSeries
    ) -> Tuple[TimeSeries, TimeSeries, TimeSeries, float, float]:
        """Calculate prediction intervals and scores."""
        lower_pred = predictions.quantile_timeseries(0.1)
        median_pred = predictions.quantile_timeseries(0.5)
        upper_pred = predictions.quantile_timeseries(0.9)
        
        # Calculate scores using aligned data
        interval_score = self._calculate_interval_score(
            test_target[:len(predictions)],
            lower_pred,
            upper_pred
        )
        point_score = mse(test_target[:len(predictions)], median_pred)
        
        return lower_pred, median_pred, upper_pred, interval_score, point_score

    def _calculate_interval_score(
        self,
        y_true: TimeSeries,
        y_pred_lower: TimeSeries,
        y_pred_upper: TimeSeries,
        alpha: float = 0.2
    ) -> float:
        """Calculate the interval score."""
        y_true_values = y_true.values()
        lower_values = y_pred_lower.values()
        upper_values = y_pred_upper.values()
        
        interval_width = upper_values - lower_values
        under_predictions = np.maximum(0, lower_values - y_true_values)
        over_predictions = np.maximum(0, y_true_values - upper_values)
        
        return float(np.mean(
            interval_width + 2/alpha * (under_predictions + over_predictions)
        ))