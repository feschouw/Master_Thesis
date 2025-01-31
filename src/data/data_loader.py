"""
Data loading and preprocessing utilities.
"""

import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)

    def load_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Load and preprocess the time series data.
        """
        df = pd.read_csv(self.data_dir / f"{dataset_name}.csv")
        df['timestamp'] = pd.to_datetime(df['time_index'], unit='D', origin='2023-01-01')
        df.set_index('timestamp', inplace=True)
        df = df.drop('time_index', axis=1, errors='ignore')
        return df

    def load_ground_truth(self, dataset_groundtruth_name: str) -> pd.DataFrame:
        """
        Load ground truth adjacency matrix.
        """
        ground_truth_adj = pd.read_csv(
            self.data_dir / f"{dataset_groundtruth_name}.csv", 
            index_col=0
        )
        return ground_truth_adj

    def prepare_data_for_training(self, df: pd.DataFrame, target_variable: str):
        """
        Prepare data for TFT model training by creating shifted target variable.
        """
        df_shifted = df.copy()
        shifted_target_name = f"{target_variable}_shifted"
        df_shifted[shifted_target_name] = df_shifted[target_variable].shift(1)
        df_shifted.fillna(method='bfill', inplace=True)
        
        covariate_variables = [var for var in df.columns if var != target_variable]
        
        return df_shifted, covariate_variables, shifted_target_name