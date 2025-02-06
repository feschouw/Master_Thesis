#!/usr/bin/env python3
"""
Main script for running causal discovery experiments with TFT.
"""

import argparse
from pathlib import Path
from datetime import datetime

from src.config.default_config import ExperimentConfig
from src.data.data_loader import DataLoader
from src.models.tft_model import TFTModelWrapper
from src.analysis.pfi_analysis import PFIAnalyzer
from src.visualization.plot_metrics import MetricsPlotter
from src.visualization.plot_graphs import GraphPlotter

def parse_args():
    parser = argparse.ArgumentParser(description='Run causal discovery experiments with TFT')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Directory containing dataset files')
    parser.add_argument('--dataset', type=str, default='harvard_diamond_data_0',
                      help='Name of dataset to use')
    parser.add_argument('--ground_truth', type=str, default='harvard_diamond_data_ground_truth',
                      help='Name of ground truth adjacency matrix file')
    
    # TFT Model architecture arguments
    parser.add_argument('--input_chunk_length', type=int, default=30,
                      help='Input sequence length in time steps')
    parser.add_argument('--output_chunk_length', type=int, default=14,
                      help='Output sequence length in time steps')
    parser.add_argument('--hidden_size', type=int, default=64,
                      help='Hidden layer size')
    parser.add_argument('--lstm_layers', type=int, default=2,
                      help='Number of LSTM layers')
    parser.add_argument('--attention_heads', type=int, default=2,
                      help='Number of attention heads')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--dropout', type=float, default=0.25,
                      help='Dropout rate')
    parser.add_argument('--device', type=str, default='mps',
                      choices=['cpu', 'cuda', 'mps'],
                      help='Device to run model on')
    parser.add_argument('--train_split', type=float, default=0.8,
                      help='Train/test split ratio')
    parser.add_argument('--num_samples', type=int, default=100,
                      help='Number of samples for prediction')
    
    # Analysis arguments
    parser.add_argument('--threshold', type=float, default=0.8,
                      help='Threshold for causal validation')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output',
                      help='Directory to save results')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f'results_{args.dataset}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration and update with command line args
    config = ExperimentConfig()
    config.DATASET_NAME = args.dataset
    config.DATASET_GROUNDTRUTH_NAME = args.ground_truth
    config.NUM_SAMPLES = args.num_samples
    config.THRESHOLD = args.threshold
    config.TRAIN_TEST_SPLIT = args.train_split
    config.DEVICE = args.device
    config.MODEL_CONFIG.update({
        'input_chunk_length': args.input_chunk_length,
        'output_chunk_length': args.output_chunk_length,
        'hidden_size': args.hidden_size,
        'lstm_layers': args.lstm_layers,
        'num_attention_heads': args.attention_heads,
        'batch_size': args.batch_size,
        'dropout': args.dropout,
        'n_epochs': args.epochs
    })
    
    print(f"Starting experiment with config: {config.get_full_config()}")
    
    try:
        # Initialize components
        data_loader = DataLoader(args.data_dir)
        metrics_plotter = MetricsPlotter(output_dir)
        graph_plotter = GraphPlotter(output_dir)
        
        # Load data
        print("Loading data...")
        df = data_loader.load_data(config.DATASET_NAME)
        ground_truth = data_loader.load_ground_truth(config.DATASET_GROUNDTRUTH_NAME)
        print(f"Dataset shape: {df.shape}")
        
        # Initialize model and analyzer
        model_wrapper = TFTModelWrapper(
            config=config.MODEL_CONFIG,
            target_variable="temp",
            device=config.DEVICE
        )
        pfi_analyzer = PFIAnalyzer(model_wrapper)
        
        # Run analysis
        print("Running PFI analysis...")
        results = pfi_analyzer.run_analysis(
            df=df,
            train_test_split=config.TRAIN_TEST_SPLIT,
            num_samples=config.NUM_SAMPLES,
            metrics_plotter=metrics_plotter
        )
        
        # Generate visualizations
        print("Preparing visualization matrices...")
        variables, interval_pfi_ratios, point_pfi_ratios = pfi_analyzer.prepare_matrices(results)
        
        # Plot results
        metrics_plotter.plot_interval_scores(results)
        metrics_plotter.plot_point_scores(results)
        metrics_plotter.plot_pfi_ratios(variables, interval_pfi_ratios, "Interval Score")
        metrics_plotter.plot_pfi_ratios(variables, point_pfi_ratios, "Point Score")
        graph_plotter.plot_ground_truth_graph(ground_truth)
        
        # Create and evaluate graphs
        print(f"Evaluating threshold: {config.THRESHOLD}")
        interval_graph = pfi_analyzer.create_causal_graph(
            variables, interval_pfi_ratios, config.THRESHOLD
        )
        point_graph = pfi_analyzer.create_causal_graph(
            variables, point_pfi_ratios, config.THRESHOLD
        )
        
        # Plot final graphs
        graph_plotter.plot_adjacency_matrix(
            variables, interval_pfi_ratios, "Interval Score", config.THRESHOLD
        )
        graph_plotter.plot_adjacency_matrix(
            variables, point_pfi_ratios, "Point Score", config.THRESHOLD
        )
        graph_plotter.plot_causal_graph(
            interval_graph, "Interval Score", config.THRESHOLD, interval_pfi_ratios
        )
        graph_plotter.plot_causal_graph(
            point_graph, "Point Score", config.THRESHOLD, point_pfi_ratios
        )
        
        # Final evaluation
        interval_metrics = pfi_analyzer.evaluate_graph(interval_graph, ground_truth)
        point_metrics = pfi_analyzer.evaluate_graph(point_graph, ground_truth)
        
        print("\nInterval Score Metrics:")
        for metric, value in interval_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print("\nPoint Score Metrics:")
        for metric, value in point_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print(f"\nExperiment completed successfully. Results saved in: {output_dir}")
        
    except Exception as e:
        print(f"Error during experiment execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()