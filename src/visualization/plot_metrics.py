"""
Module for plotting various metrics and evaluation results.
"""

from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from darts import TimeSeries
from pathlib import Path
import pandas as pd
from dataclasses import dataclass

# Define PFIResult locally to avoid circular imports
@dataclass
class PFIResult:
    """Data class for storing PFI analysis results."""
    target: str
    real_interval_score: float
    real_point_score: float
    shuffle_interval_scores: Dict[str, float]
    shuffle_point_scores: Dict[str, float]

class MetricsPlotter:
    """Class for plotting various metrics and evaluation results."""
    
    def __init__(self, output_dir: str):
        """
        Initialize the MetricsPlotter.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_plot(self, filename: str) -> None:
        """
        Save the current plot to the output directory.
        
        Args:
            filename: Name of the file to save
        """
        plt.savefig(self.output_dir / filename, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_predictions(
        self,
        test_target: TimeSeries,
        predictions: TimeSeries,
        point_estimate_score: float,
        target_variable: str
    ) -> None:
        """
        Plot model predictions against actual values.
        """
        plt.figure(figsize=(12, 6))
        test_target.plot(label="Actual")
        predictions.plot(label="Forecast")
        median_pred = predictions.quantile_timeseries(0.5)
        median_pred.plot(label="Forecast (quantile 0.5)")
        plt.title(f"TFT Forecast vs Actual for {target_variable}\nMSE: {point_estimate_score:.4f}")
        plt.legend()
        plt.tight_layout()
        self.save_plot(f"{target_variable}_predictions.png")

    def plot_interval_scores(self, results: List[PFIResult]) -> None:
        """
        Plot interval scores before and after permutation.
        """
        variables = [result.target for result in results]
        n_vars = len(variables)
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.rainbow(np.linspace(0, 1, n_vars))
        
        for i, result in enumerate(results):
            target = result.target
            score_before = result.real_interval_score
            color = colors[i]
            
            plt.scatter(1, score_before, color=color, s=100, zorder=5)
            plt.text(0.9, score_before, target, ha='right', va='center', fontweight='bold')
            
            for shuffled_var, score_after in result.shuffle_interval_scores.items():
                plt.plot([1, 2], [score_before, score_after], '-', color=color, alpha=0.5)
                plt.scatter(2, score_after, color=color, s=100, zorder=5)
                plt.text(2.1, score_after, f"{shuffled_var} → {target}", ha='left', va='center')
        
        plt.xlim(0.5, 2.5)
        plt.xticks([1, 2], ['Before permutation', 'After permutation'])
        plt.ylabel('Interval Score')
        plt.title('Interval scores before and after permutation')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        self.save_plot("interval_scores.png")

    def plot_point_scores(self, results: List[PFIResult]) -> None:
        """
        Plot point estimate scores before and after permutation.
        """
        variables = [result.target for result in results]
        n_vars = len(variables)
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.rainbow(np.linspace(0, 1, n_vars))
        
        for i, result in enumerate(results):
            target = result.target
            score_before = result.real_point_score
            color = colors[i]
            
            plt.scatter(1, score_before, color=color, s=100, zorder=5)
            plt.text(0.9, score_before, target, ha='right', va='center', fontweight='bold')
            
            for shuffled_var, score_after in result.shuffle_point_scores.items():
                plt.plot([1, 2], [score_before, score_after], '-', color=color, alpha=0.5)
                plt.scatter(2, score_after, color=color, s=100, zorder=5)
                plt.text(2.1, score_after, f"{shuffled_var} → {target}", ha='left', va='center')
        
        plt.xlim(0.5, 2.5)
        plt.xticks([1, 2], ['Before permutation', 'After permutation'])
        plt.ylabel('Point Estimate Score (MSE)')
        plt.title('Point estimate scores before and after permutation')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        self.save_plot("point_estimate_scores.png")

    def plot_pfi_ratios(
        self,
        variables: List[str],
        pfi_ratios: np.ndarray,
        score_type: str,
    ) -> None:
        """
        Plot PFI ratios heatmap.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            pfi_ratios,
            annot=True,
            fmt='.3f',
            cmap='YlGnBu',
            ax=ax,
            vmin=0,
            vmax=2,
            xticklabels=variables,
            yticklabels=variables
        )
        ax.set_title(
            f'PFI ratio between variables ({score_type})\n'
            'Lower values indicate stronger causal relationships',
            fontsize=14
        )
        ax.set_xlabel('Target Variable (Effect)', fontsize=12)
        ax.set_ylabel('Shuffled Variable (Cause)', fontsize=12)
        ax.set_xticklabels(variables, rotation=45, ha='right', fontsize=10, fontweight='bold')
        ax.set_yticklabels(variables, rotation=0, fontsize=10, fontweight='bold')
        plt.tight_layout()
        self.save_plot(f'pfi_ratios_{score_type.lower().replace(" ", "_")}.png')
