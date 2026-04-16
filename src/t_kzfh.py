"""
Test Script for KMeansZFH
============================
Test the grid-based K-means++ implementation with:
    1. Random N x M data generation (M = 4 features, N = random)
    2. Multiple K values test
    3. Visualization using PCA for dimensionality reduction

Run: python t_kzfh.py
"""

import sys
import os

# Add src/ to path so we can import kmeans_zfh as a module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
import csv as csv_module

warnings.filterwarnings('ignore')

# Import our custom KMeansZFH
from kmeans_zfh import KMeansZFH, pairwise_distances

# ============================================================================
# Configuration
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Data generation parameters
N_SAMPLES = 800       # Number of data points
N_FEATURES = 4        # Dimensionality (M)
N_CLUSTERS_TRUE = 5    # True number of clusters in synthetic data

# Test configurations
TEST_CONFIGS = [
    {
        'name': 'grid_corner_seed42',
        'n_clusters': 5,
        'grid_divisions': 5,
        'first_center_method': 'grid_corner',
        'random_seed': 42,
    },
    {
        'name': 'grid_corner_seed7',
        'n_clusters': 5,
        'grid_divisions': 5,
        'first_center_method': 'grid_corner',
        'random_seed': 7,
    },
    {
        'name': 'variance_max',
        'n_clusters': 5,
        'grid_divisions': 5,
        'first_center_method': 'variance_max',
        'random_seed': None,
    },
    {
        'name': 'index_first',
        'n_clusters': 5,
        'grid_divisions': 5,
        'first_center_method': 'index_first',
        'random_seed': None,
    },
]

# ============================================================================
# Data Generation
# ============================================================================

def generate_synthetic_data(n_samples, n_features, n_clusters, seed=42):
    """
    Generate synthetic clustered data.

    Strategy:
        - Create n_clusters cluster centers randomly in [2, 8]^n_features
        - Each cluster has different covariance structure
          (some elongated, some spherical, some correlated)
        - Uneven sample allocation per cluster (more realistic)

    Parameters
    ----------
    n_samples : int
    n_features : int
    n_clusters : int
    seed : int

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    true_labels : ndarray of shape (n_samples,)
    centers_true : ndarray of shape (n_clusters, n_features)
    """
    rng = np.random.default_rng(seed)

    # Cluster centers in [2, 8] range (well separated)
    centers = []
    for _ in range(n_clusters):
        center = rng.uniform(2.0, 8.0, size=n_features)
        centers.append(center)
    centers_true = np.array(centers)

    # Allocate samples per cluster (uneven, Dirichlet distribution)
    proportions = rng.dirichlet(np.ones(n_clusters) * 1.5)
    counts = (proportions * n_samples).astype(int)
    diff = n_samples - counts.sum()
    counts[0] += diff  # fix rounding error

    X_list = []
    labels_list = []

    for k in range(n_clusters):
        n_k = counts[k]
        center = centers_true[k]

        # Different covariance per cluster type
        if k % 3 == 0:
            # Elongated cluster (high variance in first dimension)
            cov = rng.uniform(0.5, 1.5, size=n_features)
            cov[0] *= 3.0  # extra elongated
            cov = np.diag(cov)
        elif k % 3 == 1:
            # Spherical cluster
            scale = rng.uniform(0.8, 1.5)
            cov = np.eye(n_features) * scale
        else:
            # Correlated cluster (non-diagonal covariance via Cholesky)
            L = rng.standard_normal((n_features, n_features))
            cov = L @ L.T / n_features

        # Generate multivariate normal data
        x_k = rng.multivariate_normal(center, cov, size=n_k)
        X_list.append(x_k)
        labels_list.append(np.full(n_k, k, dtype=int))

    X = np.vstack(X_list)
    true_labels = np.concatenate(labels_list)

    # Shuffle
    shuffle_idx = rng.permutation(n_samples)
    X = X[shuffle_idx]
    true_labels = true_labels[shuffle_idx]

    return X, true_labels, centers_true


# ============================================================================
# PCA Utilities (no sklearn needed)
# ============================================================================

def pca_transform(X, n_components=2):
    """
    PCA dimensionality reduction — pure NumPy, no sklearn.

    Steps:
        1. Center the data
        2. Compute covariance matrix
        3. Eigen decomposition
        4. Project onto top n_components eigenvectors

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    n_components : int

    Returns
    -------
    X_proj : ndarray of shape (n_samples, n_components)
    components : ndarray of shape (n_components, n_features)
    explained_variance_ratio : ndarray of shape (n_components,)
    """
    # Center
    X_centered = X - X.mean(axis=0)

    # Covariance matrix
    n = len(X)
    cov = (X_centered.T @ X_centered) / (n - 1)

    # Eigen decomposition (eigh is faster for symmetric matrices)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Top n_components
    components = eigenvectors[:, :n_components].T  # shape: (n_comp, n_feat)
    X_proj = X_centered @ components.T

    # Explained variance ratio
    total_var = eigenvalues.sum()
    explained_ratio = eigenvalues[:n_components] / total_var

    return X_proj, components, explained_ratio


# ============================================================================
# Visualization
# ============================================================================

def plot_clustering_results(X, labels, centers, X_proj, explained_ratio,
                             config_name, n_features, save_path=None):
    """
    Plot clustering results in a 4-panel figure.

    Panels:
        1. PCA scatter plot with cluster colors
        2. Cluster size distribution (bar chart)
        3. Cluster center heatmap (parallel coordinates style)
        4. Feature distribution by cluster (box plot)
    """
    n_clusters = len(centers)
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap('tab10', n_clusters)
    colors = [cmap(i) for i in range(n_clusters)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f'KMeansZFH — {config_name}\n'
        f'{X.shape[0]} samples, {X.shape[1]} features, K={n_clusters}',
        fontsize=14, fontweight='bold'
    )

    # ---------------------------------------------------------------
    # Panel 1: PCA scatter plot
    # ---------------------------------------------------------------
    ax1 = axes[0, 0]
    for k in unique_labels:
        mask = labels == k
        ax1.scatter(
            X_proj[mask, 0], X_proj[mask, 1],
            c=[colors[k]], label=f'Cluster {k} (n={mask.sum()})',
            alpha=0.6, s=30, edgecolors='white', linewidths=0.3
        )

    # Project and plot cluster centers
    centers_proj = pca_transform(centers, n_components=2)[0]
    ax1.scatter(
        centers_proj[:, 0], centers_proj[:, 1],
        c='black', marker='X', s=200,
        edgecolors='white', linewidths=2, zorder=10,
        label='Centroids'
    )

    ax1.set_xlabel(f'PC1 ({explained_ratio[0]*100:.1f}% var)', fontsize=11)
    ax1.set_ylabel(f'PC2 ({explained_ratio[1]*100:.1f}% var)', fontsize=11)
    ax1.set_title('PCA Scatter Plot', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')

    # ---------------------------------------------------------------
    # Panel 2: Cluster size distribution
    # ---------------------------------------------------------------
    ax2 = axes[0, 1]
    sizes = [np.sum(labels == k) for k in unique_labels]
    bars = ax2.bar(
        [f'C{k}' for k in unique_labels], sizes,
        color=[colors[k] for k in unique_labels],
        edgecolor='black', linewidth=0.5
    )
    for bar, size in zip(bars, sizes):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f'{size}\n({size/len(labels)*100:.1f}%)',
            ha='center', va='bottom', fontsize=9
        )

    ax2.set_xlabel('Cluster', fontsize=11)
    ax2.set_ylabel('Number of Samples', fontsize=11)
    ax2.set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(sizes) * 1.25)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_facecolor('#f8f9fa')

    # ---------------------------------------------------------------
    # Panel 3: Cluster center heatmap (parallel coordinates)
    # ---------------------------------------------------------------
    ax3 = axes[1, 0]
    centers_norm = (centers - centers.min(axis=0)) / (
        centers.max(axis=0) - centers.min(axis=0) + 1e-8
    )

    for k in unique_labels:
        ax3.plot(
            range(n_features), centers_norm[k],
            marker='o', markersize=8, linewidth=2,
            color=colors[k], label=f'Cluster {k}', alpha=0.8
        )

    ax3.set_xticks(range(n_features))
    ax3.set_xticklabels([f'Feature {i}' for i in range(n_features)], fontsize=10)
    ax3.set_ylabel('Normalized Feature Value', fontsize=11)
    ax3.set_title('Cluster Centers (Parallel Coordinates)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#f8f9fa')

    # ---------------------------------------------------------------
    # Panel 4: Feature distribution by cluster (box plot)
    # ---------------------------------------------------------------
    ax4 = axes[1, 1]
    positions = []
    data_by_cluster = []
    tick_positions = []
    current_pos = 0

    for k in unique_labels:
        mask = labels == k
        for d in range(n_features):
            data_by_cluster.append(X[mask, d])
            positions.append(current_pos)
            current_pos += 1
        tick_positions.append(
            (positions[len(data_by_cluster) - n_features] + current_pos - 1) / 2
        )
        current_pos += 1  # gap between clusters

    bp = ax4.boxplot(
        data_by_cluster, positions=positions,
        patch_artist=True, widths=0.6, showfliers=False
    )

    # Color the boxes by cluster
    box_idx = 0
    for k in unique_labels:
        for _ in range(n_features):
            bp['boxes'][box_idx].set_facecolor(colors[k])
            bp['boxes'][box_idx].set_alpha(0.7)
            bp['boxes'][box_idx].set_edgecolor('black')
            box_idx += 1

    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels([f'Cluster {k}' for k in unique_labels], fontsize=10)
    ax4.set_ylabel('Feature Value', fontsize=11)
    ax4.set_title('Feature Distribution by Cluster (Box Plot)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_facecolor('#f8f9fa')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"  Figure saved: {save_path}")

    plt.close(fig)


def plot_comparison(X_list, results_list, config_names, save_path=None):
    """
    Compare multiple KMeansZFH configurations side by side (PCA only).
    """
    n_configs = len(X_list)
    n_cols = min(3, n_configs)
    n_rows = (n_configs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    fig.suptitle(
        'KMeansZFH — Multiple Configurations Comparison',
        fontsize=14, fontweight='bold'
    )

    if n_configs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    cmap = plt.cm.get_cmap('tab10', 10)
    colors = [cmap(i) for i in range(10)]

    for i, (X, result, name) in enumerate(zip(X_list, results_list, config_names)):
        ax = axes[i]
        labels = result['labels']
        centers = result['centers']
        unique_labels = np.unique(labels)

        X_proj = pca_transform(X, n_components=2)[0]

        for k in unique_labels:
            mask = labels == k
            ax.scatter(
                X_proj[mask, 0], X_proj[mask, 1],
                c=[colors[k % 10]], label=f'C{k} (n={mask.sum()})',
                alpha=0.55, s=20
            )

        centers_proj = pca_transform(centers, n_components=2)[0]
        ax.scatter(
            centers_proj[:, 0], centers_proj[:, 1],
            c='black', marker='X', s=150,
            edgecolors='white', linewidths=1.5, zorder=10
        )

        ax.set_title(
            f'{name}\nInertia={result["inertia"]:.2f}, Iter={result["n_iter"]}',
            fontsize=9, fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')

    # Hide unused subplots
    for j in range(n_configs, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Comparison figure saved: {save_path}")

    plt.close(fig)


# ============================================================================
# Main Test Runner
# ============================================================================

def run_single_test(X, config, verbose=1):
    """Run a single KMeansZFH test and return structured results."""
    model = KMeansZFH(
        n_clusters=config['n_clusters'],
        grid_divisions=config['grid_divisions'],
        first_center_method=config['first_center_method'],
        random_seed=config.get('random_seed'),
        n_init=config.get('n_init', 1),
        max_iter=config.get('max_iter', 300),
        tol=config.get('tol', 1e-4),
        verbose=verbose,
    )

    model.fit(X)

    return {
        'model': model,
        'labels': model.labels_,
        'centers': model.cluster_centers_,
        'inertia': model.inertia_,
        'n_iter': model.n_iter_,
        'history': model.history_,
        'config': config,
    }


def main():
    print("=" * 70)
    print("KMeansZFH Test Suite")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Generate synthetic data
    # ------------------------------------------------------------------
    print(f"\n[Step 1] Generating synthetic data...")
    print(f"         N={N_SAMPLES}, M={N_FEATURES}, K={N_CLUSTERS_TRUE}")

    X, true_labels, centers_true = generate_synthetic_data(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_clusters=N_CLUSTERS_TRUE,
        seed=RANDOM_SEED
    )

    print(f"  Data shape    : {X.shape}")
    print(f"  True centers   :\n{centers_true.round(3)}")
    print(f"  True label dist: {np.bincount(true_labels).tolist()}")

    # ------------------------------------------------------------------
    # 2. Output directory
    # ------------------------------------------------------------------
    output_dir = os.path.join(
        os.path.dirname(__file__), '..', 'results', 'kzfh_tests'
    )
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Run all configurations
    # ------------------------------------------------------------------
    print(f"\n[Step 2] Running {len(TEST_CONFIGS)} configurations...")
    results = []
    for cfg in TEST_CONFIGS:
        print(f"\n  --- Config: {cfg['name']} ---")
        result = run_single_test(X, cfg, verbose=1)
        results.append(result)
        print(f"  Done: K={cfg['n_clusters']}, inertia={result['inertia']:.4f}, "
              f"iter={result['n_iter']}")

    # ------------------------------------------------------------------
    # 4. PCA for all configs
    # ------------------------------------------------------------------
    X_proj_all, _, explained_ratio = pca_transform(X, n_components=2)

    # ------------------------------------------------------------------
    # 5. Visualize each configuration
    # ------------------------------------------------------------------
    print(f"\n[Step 3] Generating individual figures...")
    for i, (result, cfg) in enumerate(zip(results, TEST_CONFIGS)):
        safe_name = cfg['name'].replace(' ', '_').replace('(', '').replace(')', '')
        save_path = os.path.join(output_dir, f'config_{i+1}_{safe_name}.png')

        plot_clustering_results(
            X, result['labels'], result['centers'],
            X_proj_all, explained_ratio,
            config_name=cfg['name'],
            n_features=N_FEATURES,
            save_path=save_path
        )

    # ------------------------------------------------------------------
    # 6. Side-by-side comparison
    # ------------------------------------------------------------------
    print(f"\n[Step 4] Generating comparison figure...")
    comparison_path = os.path.join(output_dir, 'comparison_all_configs.png')
    plot_comparison(
        [X] * len(results),
        [{'labels': r['labels'], 'centers': r['centers'],
          'inertia': r['inertia'], 'n_iter': r['n_iter']}
         for r in results],
        [cfg['name'] for cfg in TEST_CONFIGS],
        save_path=comparison_path
    )

    # ------------------------------------------------------------------
    # 7. Summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Summary Table")
    print(f"{'='*70}")
    print(f"{'Config':<35} {'K':>3} {'Inertia':>12} {'Iter':>5} {'MaxShift':>12}")
    print(f"{'-'*70}")

    for result, cfg in zip(results, TEST_CONFIGS):
        max_shift = 0.0
        if result['history']:
            shifts = [h['centroid_shift'] for h in result['history']
                      if 'centroid_shift' in h]
            max_shift = max(shifts) if shifts else 0.0

        print(
            f"{cfg['name']:<35} {cfg['n_clusters']:>3} "
            f"{result['inertia']:>12.4f} {result['n_iter']:>5} "
            f"{max_shift:>12.6f}"
        )

    # ------------------------------------------------------------------
    # 8. Save clustering results to CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(output_dir, 'clustering_results_config1.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv_module.writer(f)
        writer.writerow(['sample_idx', 'CLUSTER_LABEL'] +
                        [f'FEAT_{i}' for i in range(N_FEATURES)])
        for i in range(len(X)):
            writer.writerow([i, results[0]['labels'][i]] + X[i].tolist())
    print(f"\n  CSV saved: {csv_path}")

    print(f"\n{'='*70}")
    print(f"Test complete! Figures saved to: {output_dir}")
    print(f"{'='*70}")

    return results, TEST_CONFIGS


if __name__ == '__main__':
    results, configs = main()
