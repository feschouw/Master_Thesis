#!/usr/bin/env python3
"""
Main script to run the causal discovery experiments.
"""

import sys
from pathlib import Path
import logging
import argparse
from datetime import datetime

# project_root = Path(__file__).resolve().parents[1]
# sys.path.append(str(project_root))

from src.config.default_config import ExperimentConfig
from src.data.data_loader import DataLoader
from src.models.tft_model import TFTModelWrapper
from src.analysis.pfi_analysis import PFIAnalyzer
from src.visualization.plot_metrics import MetricsPlotter
from src.visualization.plot_graphs import GraphPlotter

def setup_logging(output_dir):
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'experiment.log'),
            logging.StreamHandler()
        ]
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run causal discovery experiment')
    parser.add_argument('--config', type=str, default='src/config/default_config.py',
                      help='Path to configuration file')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'],
                      help='Device to run the model on')
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_args()
    
    # Load configuration
    config = ExperimentConfig()
    if args.device:
        config.DEVICE = args.device
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f'output/results_{config.DATASET_NAME}_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    logging.info(f"Starting experiment with config: {config.get_full_config()}")
    
    try:
        # Initialize components
        data_loader = DataLoader(project_root / 'data')
        metrics_plotter = MetricsPlotter(output_dir)
        graph_plotter = GraphPlotter(output_dir)
        
        # Load data
        logging.info("Loading data...")
        df = data_loader.load_data(config.DATASET_NAME)
        ground_truth = data_loader.load_ground_truth(config.DATASET_GROUNDTRUTH_NAME)
        logging.info(f"Dataset shape: {df.shape}")
        
        # Initialize model and analyzer
        model_wrapper = TFTModelWrapper(
            config=config.MODEL_CONFIG,
            target_variable="temp",
            device=config.DEVICE
        )
        
        pfi_analyzer = PFIAnalyzer(model_wrapper)
        
        # Run analysis
        logging.info("Running PFI analysis...")
        results = pfi_analyzer.run_analysis(
            df=df,
            train_test_split=config.TRAIN_TEST_SPLIT,
            num_samples=config.NUM_SAMPLES,
            metrics_plotter=metrics_plotter
        )
        
        # Generate visualizations
        logging.info("Preparing visualization matrices...")
        variables, interval_pfi_ratios, point_pfi_ratios = pfi_analyzer.prepare_matrices(results)
        
        # Plot metrics and graphs
        metrics_plotter.plot_interval_scores(results)
        metrics_plotter.plot_point_scores(results)
        metrics_plotter.plot_pfi_ratios(variables, interval_pfi_ratios, "Interval Score")
        metrics_plotter.plot_pfi_ratios(variables, point_pfi_ratios, "Point Score")
        graph_plotter.plot_ground_truth_graph(ground_truth)
        
        # Evaluate with configured threshold
        threshold = config.THRESHOLD
        logging.info(f"Evaluating threshold: {threshold}")
        
        # Create and evaluate graphs
        interval_graph = pfi_analyzer.create_causal_graph(
            variables, interval_pfi_ratios, threshold
        )
        point_graph = pfi_analyzer.create_causal_graph(
            variables, point_pfi_ratios, threshold
        )
        
        # Plot results
        graph_plotter.plot_adjacency_matrix(
            variables, interval_pfi_ratios, "Interval Score", threshold
        )
        graph_plotter.plot_adjacency_matrix(
            variables, point_pfi_ratios, "Point Score", threshold
        )
        graph_plotter.plot_causal_graph(
            interval_graph, "Interval Score", threshold, interval_pfi_ratios
        )
        graph_plotter.plot_causal_graph(
            point_graph, "Point Score", threshold, point_pfi_ratios
        )
        
        # Evaluate against ground truth
        interval_metrics = pfi_analyzer.evaluate_graph(interval_graph, ground_truth)
        point_metrics = pfi_analyzer.evaluate_graph(point_graph, ground_truth)
        
        # Log results
        logging.info("\nInterval Score Metrics:")
        for metric, value in interval_metrics.items():
            logging.info(f"{metric}: {value:.4f}")
        
        logging.info("\nPoint Score Metrics:")
        for metric, value in point_metrics.items():
            logging.info(f"{metric}: {value:.4f}")
        
        logging.info(f"Experiment completed successfully. Results saved in: {output_dir}")
        
    except Exception as e:
        logging.error(f"Error during experiment execution: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()