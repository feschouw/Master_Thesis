"""
Analyzer class for Permutation Feature Importance in causal discovery.
"""

from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from darts import TimeSeries
import networkx as nx
from dataclasses import dataclass
from ..visualization.plot_metrics import MetricsPlotter

@dataclass
class PFIResult:
    """Data class for storing PFI analysis results."""
    target: str
    real_interval_score: float
    real_point_score: float
    shuffle_interval_scores: Dict[str, float]
    shuffle_point_scores: Dict[str, float]

class PFIAnalyzer:
    """Analyzer class for Permutation Feature Importance in causal discovery."""
    
    def __init__(self, model_wrapper):
        """
        Initialize PFI analyzer.
        
        Args:
            model_wrapper: Instance of TFTModelWrapper for predictions
        """
        self.model_wrapper = model_wrapper

    def shuffle_covariate(
        self,
        covariates: TimeSeries,
        column_name: str
    ) -> TimeSeries:
        """
        Shuffle values of a specific covariate while preserving the time index.
        
        Args:
            covariates: Original covariate TimeSeries
            column_name: Name of column to shuffle
            
        Returns:
            TimeSeries with shuffled column
        """
        shuffled_covariates = covariates.pd_dataframe().copy()
        shuffled_values = shuffled_covariates[column_name].values.copy()
        np.random.shuffle(shuffled_values)
        shuffled_covariates[column_name] = shuffled_values
        return TimeSeries.from_dataframe(shuffled_covariates)

    def analyze_target(
        self,
        target_variable: str,
        df: pd.DataFrame,
        covariate_variables: List[str],
        train_test_split: float,
        num_samples: int,
        metrics_plotter: MetricsPlotter
    ) -> PFIResult:
        """
        Perform PFI analysis for a single target variable.
        """
        # Create shifted version of target
        df_shifted = df.copy()
        shifted_target_name = f"{target_variable}_shifted"
        df_shifted[shifted_target_name] = df_shifted[target_variable].shift(1)
        df_shifted.fillna(method='bfill', inplace=True)

        # Prepare data
        target = TimeSeries.from_series(df_shifted[target_variable].astype(np.float32))
        all_covariates = covariate_variables + [shifted_target_name]
        covariates = TimeSeries.from_dataframe(df_shifted[all_covariates].astype(np.float32))

        # Split data
        train_target, test_target, train_covariates, test_covariates = \
            self.model_wrapper.prepare_data(target, covariates, train_test_split)

        # Train model
        self.model_wrapper.train(train_target, train_covariates)

        # Get baseline predictions
        predictions = self.model_wrapper.predict(
            len(test_target), train_target, test_covariates, num_samples
        )

        # Calculate baseline scores
        _, _, _, real_interval_score, real_point_score = \
            self.model_wrapper.calculate_prediction_intervals(predictions, test_target)
            
        # Plot and save predictions
        metrics_plotter.plot_predictions(
            test_target,
            predictions,
            real_point_score,
            target_variable
        )

        # Analyze each covariate through shuffling
        shuffle_interval_scores = {}
        shuffle_point_scores = {}
        
        for covariate in covariate_variables + [target_variable]:
            # Use target_variable instead of shifted name for results
            key_name = target_variable if covariate == shifted_target_name else covariate
            
            # Shuffle the appropriate column
            shuffle_col = shifted_target_name if covariate == target_variable else covariate
            shuffled_covariates = self.shuffle_covariate(test_covariates, shuffle_col)

            # Get predictions with shuffled data
            shuffled_predictions = self.model_wrapper.predict(
                len(test_target), train_target, shuffled_covariates, num_samples
            )
            
            # Calculate scores with shuffled data
            _, _, _, shuffled_interval_score, shuffled_point_score = \
                self.model_wrapper.calculate_prediction_intervals(
                    shuffled_predictions, test_target
                )
            
            shuffle_interval_scores[key_name] = shuffled_interval_score
            shuffle_point_scores[key_name] = shuffled_point_score

        return PFIResult(
            target=target_variable,
            real_interval_score=real_interval_score,
            real_point_score=real_point_score,
            shuffle_interval_scores=shuffle_interval_scores,
            shuffle_point_scores=shuffle_point_scores
        )

    def run_analysis(
        self,
        df: pd.DataFrame,
        train_test_split: float,
        num_samples: int,
        metrics_plotter: MetricsPlotter
    ) -> List[PFIResult]:
        """
        Run PFI analysis for all variables in the dataset.
        """
        variables = df.columns.tolist()
        results = []

        for target_variable in variables:
            print(f"\nAnalyzing target variable: {target_variable}")
            covariate_variables = [var for var in variables if var != target_variable]
            
            result = self.analyze_target(
                target_variable,
                df,
                covariate_variables,
                train_test_split,
                num_samples,
                metrics_plotter
            )
            results.append(result)

        return results

    def prepare_matrices(
        self,
        results: List[PFIResult]
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Prepare PFI ratio matrices from analysis results.
        """
        variables = [result.target for result in results]
        n_vars = len(variables)
        
        interval_scores_before = np.array([r.real_interval_score for r in results])
        point_scores_before = np.array([r.real_point_score for r in results])
        
        interval_scores_after = np.full((n_vars, n_vars), np.nan)
        point_scores_after = np.full((n_vars, n_vars), np.nan)

        for i, result in enumerate(results):
            for j, var in enumerate(variables):
                if var in result.shuffle_interval_scores:
                    interval_scores_after[j, i] = result.shuffle_interval_scores[var]
                    point_scores_after[j, i] = result.shuffle_point_scores[var]

        # Calculate PFI ratios
        interval_pfi_ratios = interval_scores_before[np.newaxis, :] / interval_scores_after
        point_pfi_ratios = point_scores_before[np.newaxis, :] / point_scores_after
        
        return variables, interval_pfi_ratios, point_pfi_ratios

    def create_causal_graph(
        self,
        variables: List[str],
        pfi_ratios: np.ndarray,
        threshold: float
    ) -> nx.DiGraph:
        """
        Create a causal graph based on PFI ratios.
        """
        G = nx.DiGraph()
        G.add_nodes_from(variables)
        
        for i, source in enumerate(variables):
            for j, target in enumerate(variables):
                if pfi_ratios[i, j] < threshold:
                    G.add_edge(source, target)
        
        return G

    def evaluate_graph(
        self,
        G: nx.DiGraph,
        ground_truth_adj: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate the discovered causal graph against ground truth.
        """
        variables = list(G.nodes())
        n = len(variables)
        discovered_adj = np.zeros((n, n))
        
        for i, source in enumerate(variables):
            for j, target in enumerate(variables):
                if G.has_edge(source, target):
                    discovered_adj[i, j] = 1
        
        ground_truth_np = ground_truth_adj.values
        
        # Calculate metrics
        true_positives = np.sum((discovered_adj == 1) & (ground_truth_np == 1))
        false_positives = np.sum((discovered_adj == 1) & (ground_truth_np == 0))
        false_negatives = np.sum((discovered_adj == 0) & (ground_truth_np == 1))
        
        precision = true_positives / (true_positives + false_positives) \
            if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) \
            if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) \
            if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }