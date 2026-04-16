"""
Results Visualization Module
Visualization for OGLCM-AKBO Clustering Results

Features:
    1. Feature distribution box plots (feature distribution across clusters)
    2. PCA scatter plots (2D projection visualization)
    3. Cluster centers radar chart (shows each cluster center's feature values)
    4. Bayesian optimization history plots (shows the optimization process)

Authors: Cuka
Date: 2026-03-17
Version: 2.0 (supports visualization after feature selection)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# Configure matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ClusteringVisualizer:
    """
    Cluster Results Visualizer.

    Attributes:
        depth:        ndarray, depth data (N,)
        features:     ndarray, feature data (N x D)
        labels:       ndarray, cluster labels (N,)
        feature_names: list, feature name list
        n_clusters:   int,    number of clusters
        colors:       ndarray, cluster color mapping
    """

    def __init__(self, depth, features, labels, feature_names=None):
        """
        Initialize the visualizer.

        Parameters:
            depth:        ndarray, depth data (N,)
            features:     ndarray, feature data (N x D)
            labels:       ndarray, cluster labels (N,)
            feature_names: list, feature name list (optional)
        """
        self.depth  = depth
        self.features = features
        self.labels = labels
        self.feature_names = feature_names if feature_names else [
            f'Feature {i}' for i in range(features.shape[1])
        ]
        self.n_clusters = len(np.unique(labels))

        # Use tab10 colormap (supports up to 10 clusters)
        self.colors = plt.cm.tab10(np.linspace(0, 1, self.n_clusters))

    def plot_feature_distribution(self, save_path='results/figures/feature_distribution.png'):
        """
        Plot feature distribution box plots (optimized version).

        Displays:
            - Distribution of each feature across different clusters (box plot)
            - Optimized: larger figure, clearer labels, better colors
        """
        print("\n[Visualization] Plotting feature distribution box plots...")

        df = pd.DataFrame(self.features, columns=self.feature_names)
        df['Cluster'] = self.labels

        n_features = len(self.feature_names)

        # Create a larger figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # Use a more professional color scheme
        palette = sns.color_palette("husl", self.n_clusters)

        for idx in range(n_features):
            ax = axes[idx]
            feat_name = self.feature_names[idx]

            # Prepare data
            data_to_plot = []
            cluster_labels = []
            for i in range(self.n_clusters):
                cluster_data = df[df['Cluster'] == i][feat_name].values
                data_to_plot.append(cluster_data)
                cluster_labels.append(f'Cluster {i}\n(n={len(cluster_data)})')

            # Draw box plot (modern style)
            bp = ax.boxplot(
                data_to_plot,
                labels=cluster_labels,
                patch_artist=True,
                widths=0.6,
                showfliers=False,      # hide outliers
                medianprops=dict(color='black', linewidth=2),  # bold median line
                boxprops=dict(linewidth=1.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5)
            )

            # Set colors (professional palette)
            for patch, color in zip(bp['boxes'], palette):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Add grid
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.set_ylabel('Normalized Value', fontsize=11)
            ax.set_title(feat_name, fontsize=12, fontweight='bold', pad=10)
            ax.tick_params(axis='x', labelsize=9)

        # Hide extra subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Feature Distribution by Cluster (Box Plot)', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      Saved: {save_path}")
        plt.close()

    def plot_pca_scatter(self, save_path='results/figures/pca_scatter.png'):
        """
        Plot PCA dimensionality reduction scatter plot.

        Projects high-dimensional features onto a 2D plane to show clustering separation.
        """
        print("\n[Visualization] Plotting PCA scatter plot...")

        # PCA dimensionality reduction
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(self.features)

        # Explained variance
        explained_var = pca.explained_variance_ratio_
        print(f"      PCA explained variance: PC1={explained_var[0]*100:.1f}%, PC2={explained_var[1]*100:.1f}%")

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot by cluster
        for i in range(self.n_clusters):
            mask = self.labels == i
            ax.scatter(
                features_pca[mask, 0],
                features_pca[mask, 1],
                c=[self.colors[i]],
                label=f'Cluster {i}',
                alpha=0.6,
                s=30,
                edgecolors='k',
                linewidth=0.5
            )

        ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontsize=12)
        ax.set_title('PCA Projection of Clusters', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      Saved: {save_path}")
        plt.close()

    def plot_cluster_centers(self, save_path='results/figures/cluster_centers.png'):
        """
        Plot cluster centers radar chart.

        Shows the normalized value of each cluster center across features.
        """
        print("\n[Visualization] Plotting cluster centers radar chart...")

        # Compute cluster centers
        centers = np.zeros((self.n_clusters, self.features.shape[1]))
        for i in range(self.n_clusters):
            centers[i] = self.features[self.labels == i].mean(axis=0)

        # Normalize to [0, 1]
        centers_norm = (centers - centers.min()) / (centers.max() - centers.min() + 1e-10)

        # Select up to 8 features
        n_features = min(8, len(self.feature_names))
        feature_indices = np.linspace(
            0,
            len(self.feature_names) - 1,
            n_features,
            dtype=int
        )
        features_selected = [self.feature_names[i] for i in feature_indices]
        centers_selected  = centers_norm[:, feature_indices]

        # Compute radar chart angles
        angles = np.linspace(0, 2 * np.pi, len(features_selected), endpoint=False).tolist()
        angles += angles[:1]   # close the loop

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

        # Plot radar chart for each cluster
        for i in range(self.n_clusters):
            values = centers_selected[i].tolist()
            values += values[:1]   # close the loop
            ax.plot(
                angles,
                values,
                'o-',
                linewidth=2,
                label=f'Cluster {i}',
                color=self.colors[i]
            )
            ax.fill(angles, values, alpha=0.15, color=self.colors[i])

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f[:15] for f in features_selected], fontsize=10)
        ax.set_title('Cluster Centers (Radar Chart)', fontsize=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      Saved: {save_path}")
        plt.close()

    def plot_optimization_history(self, history, save_path='results/figures/optimization_history.png'):
        """
        Plot Bayesian optimization history.

        Shows:
            - Left: UIndex vs. iteration
            - Right: K value selection history
        """
        print("\n[Visualization] Plotting optimization history...")

        if not history:
            print("      No optimization history to plot")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Extract data (compatible with both uppercase and lowercase keys)
        iteration_nums = list(range(1, len(history) + 1))
        uindex_values  = [h.get('uindex', h.get('UIndex', 0)) for h in history]
        k_values        = [h['K'] for h in history]

        # -----------------------------------------------------------------
        # Left: UIndex vs. iteration
        # -----------------------------------------------------------------
        ax = axes[0]
        ax.plot(
            iteration_nums,
            uindex_values,
            'bo-',
            linewidth=2,
            markersize=8
        )
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('UIndex', fontsize=12)
        ax.set_title('Bayesian Optimization History', fontsize=14)
        ax.grid(True, alpha=0.3)

        # -----------------------------------------------------------------
        # Right: K value selection history
        # -----------------------------------------------------------------
        ax = axes[1]
        scatter = ax.scatter(
            iteration_nums,
            k_values,
            c=uindex_values,
            cmap='viridis',
            s=100,
            alpha=0.6,
            edgecolors='k'
        )
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('K Value', fontsize=12)
        ax.set_title('K Value Selection History', fontsize=14)
        plt.colorbar(scatter, label='UIndex')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      Saved: {save_path}")
        plt.close()

    def plot_all(self, history=None):
        """
        Plot all visualizations.

        Parameters:
            history: list, Bayesian optimization history (optional)
        """
        self.plot_feature_distribution()
        self.plot_pca_scatter()
        self.plot_cluster_centers()

        if history:
            self.plot_optimization_history(history)

        print("\n[OK] All visualizations generated!")


def visualize_results(depth, features, labels, feature_names=None, history=None):
    """
    Convenience function to visualize clustering results.

    Parameters:
        depth:        ndarray, depth data
        features:     ndarray, feature data
        labels:       ndarray, cluster labels
        feature_names: list, feature name list
        history:      list, optimization history

    Returns:
        visualizer: ClusteringVisualizer object
    """
    visualizer = ClusteringVisualizer(depth, features, labels, feature_names)
    visualizer.plot_all(history)

    return visualizer


if __name__ == "__main__":
    # Test
    from data_loader import load_and_preprocess
    from akbo_clustering import akbo_clustering

    file_path = r"C:\Users\Maple\.openclaw\workspace\OGLCM-AKBO\TZ1H_texture_logging.csv"
    depth, features, preprocessor, report = load_and_preprocess(file_path)

    labels, clusterer = akbo_clustering(
        features,
        k_range=(2, 6),
        n_init=3,
        max_iter=5
    )

    visualize_results(
        depth,
        features,
        labels,
        preprocessor.feature_columns,
        clusterer.optimization_history
    )
