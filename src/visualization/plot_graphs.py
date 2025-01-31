"""
Module for plotting causal graphs and adjacency matrices.
"""

from typing import List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
import pandas as pd

class GraphPlotter:
    """Class for plotting causal graphs and related visualizations."""
    
    def __init__(self, output_dir: str):
        """
        Initialize the GraphPlotter.
        
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

    def plot_adjacency_matrix(
        self,
        variables: List[str],
        pfi_ratios: np.ndarray,
        score_type: str,
        threshold: float
    ) -> None:
        """
        Plot adjacency matrix based on PFI ratios.
        
        Args:
            variables: List of variable names
            pfi_ratios: Matrix of PFI ratios
            score_type: Type of score (interval or point)
            threshold: Threshold for edge creation
        """
        adjacency_matrix = (pfi_ratios < threshold).astype(int)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            adjacency_matrix,
            annot=True,
            fmt='d',
            cmap='binary',
            ax=ax,
            vmin=0,
            vmax=1,
            xticklabels=variables,
            yticklabels=variables
        )
        ax.set_title(
            f'Final adjacency matrix based on {score_type} PFI\n'
            f'Threshold < {threshold}',
            fontsize=14
        )
        ax.set_xlabel('Target Variable (Effect)', fontsize=12)
        ax.set_ylabel('Source Variable (Cause)', fontsize=12)
        ax.set_xticklabels(variables, rotation=45, ha='right', fontsize=10, fontweight='bold')
        ax.set_yticklabels(variables, rotation=0, fontsize=10, fontweight='bold')
        plt.tight_layout()
        self.save_plot(f'adjacency_matrix_{score_type.lower().replace(" ", "_")}.png')

    def plot_causal_graph(
        self,
        G: nx.DiGraph,
        score_type: str,
        threshold: float,
        pfi_ratios: np.ndarray = None
    ) -> None:
        """
        Plot causal graph visualization.
        
        Args:
            G: NetworkX DiGraph object
            score_type: Type of score used
            threshold: Threshold used for edge creation
            pfi_ratios: Optional matrix of PFI ratios for edge labels
        """
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color='lightgreen',
            node_size=3000
        )
        
        # Draw regular edges
        regular_edges = [(u, v) for (u, v) in G.edges() if u != v]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=regular_edges,
            edge_color='black',
            arrows=True,
            arrowsize=20
        )
        
        # Draw self-loops
        self_loops = [(u, u) for u in G.nodes() if G.has_edge(u, u)]
        if self_loops:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=self_loops,
                edge_color='black',
                arrows=True,
                arrowsize=20,
                connectionstyle='arc3, rad=0.3'
            )
        
        # Add node labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        # Add edge labels if PFI ratios are provided
        if pfi_ratios is not None:
            variables = list(G.nodes())
            edge_labels = {
                (u, v): f'{pfi_ratios[variables.index(u), variables.index(v)]:.2f}'
                for (u, v) in G.edges()
            }
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title(
            f'Causal Graph based on {score_type} PFI\n'
            f'Edges where PFI ratio < {threshold}',
            fontsize=16
        )
        plt.axis('off')
        plt.tight_layout()
        self.save_plot(f'causal_graph_{score_type.lower().replace(" ", "_")}.png')
        
    def plot_ground_truth_graph(self, ground_truth_adj: pd.DataFrame) -> None:
        """
        Plot ground truth causal graph.
        
        Args:
            ground_truth_adj: Ground truth adjacency matrix as DataFrame
        """
        G_true = nx.DiGraph(ground_truth_adj.values)
        variables = ground_truth_adj.index.tolist()
        nx.relabel_nodes(G_true, dict(enumerate(variables)), copy=False)
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G_true, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G_true,
            pos,
            node_color='lightgreen',
            node_size=3000
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G_true,
            pos,
            edge_color='black',
            arrows=True,
            arrowsize=20
        )
        
        # Add labels
        nx.draw_networkx_labels(
            G_true,
            pos,
            font_size=10,
            font_weight='bold'
        )
        
        plt.title('Ground Truth Causal Graph', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        self.save_plot('ground_truth_graph.png')