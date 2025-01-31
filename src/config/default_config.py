"""
Default configuration for the causal discovery experiments.
"""

class ExperimentConfig:
    # Dataset configuration
    DATASET_NAME = "harvard_diamond_data_0"
    DATASET_GROUNDTRUTH_NAME = "harvard_diamond_data_ground_truth"
    NUM_SAMPLES = 100
    THRESHOLD = 0.8
    TRAIN_TEST_SPLIT = 0.8

    # Model configuration
    MODEL_CONFIG = {
        'input_chunk_length': 30,
        'output_chunk_length': 14,
        'hidden_size': 16,
        'lstm_layers': 2,
        'num_attention_heads': 2,
        'batch_size': 64,
        'dropout': 0.25,
        'n_epochs': 200
    }

    # Training configuration
    DEVICE = 'mps'  #  can go to 'cuda' for NVIDIA GPUs
    PRECISION = '32-true'

    def get_full_config(self):
        """Combine all configurations into a single dictionary."""
        return {
            'dataset_name': self.DATASET_NAME,
            'dataset_groundtruth_name': self.DATASET_GROUNDTRUTH_NAME,
            'num_samples': self.NUM_SAMPLES,
            'threshold': self.THRESHOLD,
            'train_test_split': self.TRAIN_TEST_SPLIT,
            **self.MODEL_CONFIG
        }
