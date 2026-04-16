"""
OGLCM-AKBO Unsupervised Clustering Main Program

Main Program for OGLCM-AKBO Unsupervised Clustering

Automatic shale microfacies characterization method based on
Orientational Gray-Level Co-occurrence Matrix (OGLCM) and
Auto-Kmeans with Bayesian Optimization (AKBO)

Input:
    - TZ1H_texture_logging.csv: Texture feature data (28 OGLCM features)

Output:
    - results/clustering_results.csv: Clustering results
    - results/clustering_metrics.json: Evaluation metrics
    - results/figures/*.png: Visualization figures
    - results/test_report.md: Test report

Authors: Doctor (Fuhao Zhang) & Cuka
Date: 2026-03-17
Version: 2.0 (Added feature selection, using 4 important features)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_and_preprocess_manual
from src.akbo_clustering import AKBOClusterer
from src.visualization import ClusteringVisualizer


def main():
    """
    Main function

    Pipeline:
        1. Data loading and preprocessing
        2. Feature selection (using 4 important texture features)
        3. AKBO clustering optimization
        4. Results visualization
        5. Save results
    """

    print("="*80)
    print("OGLCM-AKBO Unsupervised Clustering Algorithm")
    print("Based on Orientational Gray-Level Co-occurrence Matrix and Bayesian Optimization Auto-Kmeans")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ==================== Step 1: Data Loading and Preprocessing ====================
    print("="*80)
    print("Step 1: Data Loading and Preprocessing")
    print("="*80)

    data_file = r"C:\Users\Maple\.openclaw\workspace\OGLCM-AKBO\TZ1H_texture_logging.csv"

    # Preselected 4 optimal features
    selected_features = [
        'CON_SUB_DYNA',
        'DIS_SUB_DYNA',
        'HOM_SUB_DYNA',
        'ENG_SUB_DYNA'
    ]

    # Load data and preprocess (auto-select specified features)
    features, preprocessor, quality_report = load_and_preprocess_manual(
        data_file,
        feature_columns=selected_features
    )

    # Read depth data
    import pandas as pd
    df = pd.read_csv(data_file)
    depth = df['DEPTH'].values

    print(f"\n[OK] Data loading complete")
    print(f"  - Number of samples: {len(depth)}")
    print(f"  - Number of features: {features.shape[1]}")
    print(f"  - Depth range: {depth.min():.2f} - {depth.max():.2f} m")

    # ==================== Step 2: Confirm Selected Features ====================
    print("\n" + "="*80)
    print("Step 2: Confirm Selected Features")
    print("="*80)

    # Features already selected during data preprocessing
    selected_feature_names = preprocessor.selected_columns
    selected_indices = list(range(len(selected_feature_names)))
    features_selected = features  # Already selected features

    print(f"\nUsing preset {len(selected_indices)} optimal features:")
    for i, name in enumerate(selected_feature_names):
        print(f"  {i+1}. {name}")

    # ==================== Step 3: AKBO Clustering Optimization ====================
    print("\n" + "="*80)
    print("Step 3: AKBO Clustering Optimization")
    print("="*80)

    # Create clusterer
    clusterer = AKBOClusterer(
        k_range=(5, 20),    # K value search range
        n_init=8,           # Number of initial sampling points
        max_iter=30,        # Maximum number of iterations
        n_patience=8,      # Convergence patience count
        tol=1e-4,           # Convergence threshold
        random_state=42     # Random seed
    )

    # Execute optimization (using specified feature indices)
    optimal_k = clusterer.optimize(
        features,
        feature_names=selected_feature_names,
        selected_indices=selected_indices,
    )
    labels = clusterer.fit(features_selected)

    # ==================== Step 4: Results Visualization ====================
    print("\n" + "="*80)
    print("Step 4: Results Visualization")
    print("="*80)

    visualizer = ClusteringVisualizer(
        depth=depth,
        features=features_selected,  # Use selected features for visualization
        labels=labels,
        feature_names=selected_feature_names
    )

    # Plot all figures
    visualizer.plot_all(history=clusterer.optimization_history)

    # ==================== Step 5: Save Results ====================
    print("\n" + "="*80)
    print("Step 5: Save Results")
    print("="*80)

    # 1. Save clustering results
    results_df = pd.DataFrame({
        'DEPTH': depth,
        'CLUSTER_LABEL': labels
    })

    # Add clustering probability (based on GMM posterior probability)
    probs = clusterer.get_cluster_probs(features_selected)
    for i in range(probs.shape[1]):
        results_df[f'CLUSTER_{i}_PROB'] = probs[:, i]

    results_file = 'results/clustering_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"[OK] Clustering results saved: {results_file}")

    # 2. Generate test report (with detailed iteration history)
    generate_test_report(
        optimal_k=optimal_k,
        n_samples=len(depth),
        n_features=len(selected_indices),
        selected_features=selected_feature_names,
        depth_range=[float(depth.min()), float(depth.max())],
        best_metrics=clusterer.best_metrics,
        cluster_distribution={f'cluster_{i}': int(np.sum(labels == i)) for i in range(optimal_k)},
        optimization_history=clusterer.optimization_history,
        results_file=results_file
    )

    # ==================== Complete ====================
    print("\n" + "="*80)
    print("[OK] All tasks completed!")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutput files:")
    print(f"  1. Clustering results: {results_file}")
    print(f"  2. Visualization figures: results/figures/")
    print(f"  3. Test report: results/test_report.md (with detailed iteration history)")

    return results_df, clusterer


def generate_test_report(optimal_k, n_samples, n_features, selected_features,
                         depth_range, best_metrics, cluster_distribution,
                         optimization_history, results_file):
    """
    Generate test report (with detailed iteration history)

    Parameters:
        optimal_k: int, optimal K value
        n_samples: int, number of samples
        n_features: int, number of features
        selected_features: list, feature names
        depth_range: list, depth range
        best_metrics: dict, best metrics
        cluster_distribution: dict, clustering distribution
        optimization_history: list, optimization history
        results_file: str, results file path
    """

    # Quality evaluation
    sil = best_metrics['si']
    if sil > 0.7:
        quality_rating = "Excellent - Clustering structure is clear with good separation"
    elif sil > 0.5:
        quality_rating = "Good - Clustering structure is reasonable, acceptable"
    elif sil > 0.25:
        quality_rating = "Fair - Clustering structure is weak, parameter adjustment recommended"
    else:
        quality_rating = "Poor - Clustering structure is not obvious, need to reconsider"

    from datetime import datetime
    timestamp = datetime.now().isoformat()

    report = f"""# OGLCM-AKBO Clustering Test Report

**Generated at:** {timestamp}

---

## Data Overview

| Item | Value |
|------|-----|
| **Number of samples** | {n_samples} |
| **Number of features** | {n_features} |
| **Depth range** | {depth_range[0]:.2f} - {depth_range[1]:.2f} m |
| **Optimal number of clusters (K)** | {optimal_k} |

---

## Feature Selection

**Selected important features ({len(selected_features)} total):**

| # | Feature Name | Geological Significance |
|---|--------|----------|
| 1 | {selected_features[0]} | Contrast_sub-region_dynamic - Reflects local texture variation intensity |
| 2 | {selected_features[1]} | Dissimilarity_sub-region_dynamic - Reflects gray-level difference of sub-regions |
| 3 | {selected_features[2]} | Homogeneity_sub-region_dynamic - Reflects texture uniformity of sub-regions |
| 4 | {selected_features[3]} | Entropy_sub-region_dynamic - Reflects texture complexity |

**Feature selection basis:** Based on Random Forest feature importance analysis (completed during data preprocessing)

---

## Clustering Quality Evaluation

| Metric | Value | Description | Evaluation |
|------|-----|------|------|
| **UIndex** | {best_metrics['uindex']:.4f} | Composite index (higher is better) | {'Excellent' if best_metrics['uindex'] > 0.5 else 'Needs improvement'} |
| **Silhouette Index (SI)** | {best_metrics['si']:.4f} | >0.5 indicates reasonable clustering | {'Good' if best_metrics['si'] > 0.5 else 'Fair'} |
| **DBI** | {best_metrics['dbi']:.4f} | <1.0 is good | {'Good' if best_metrics['dbi'] < 1.0 else 'Needs improvement'} |
| **DVI** | {best_metrics['dvi']:.4f} | Higher is better | {'Good' if best_metrics['dvi'] > 0.5 else 'Low'} |

### Clustering Quality Evaluation:

**{quality_rating}**

---

## Clustering Distribution

| Cluster Label | Number of Samples | Percentage (%) | Geological Interpretation (TBD) |
|---------|--------|----------|-------------|
"""

    for i in range(optimal_k):
        count = cluster_distribution[f'cluster_{i}']
        percentage = count / n_samples * 100
        report += f"| Cluster {i} | {count} | {percentage:.1f}% | TBD |\n"

    report += f"""
---

## Bayesian Optimization History (Detailed Iteration Records)

**Total iterations:** {len(optimization_history)} (including initial sampling and Bayesian optimization)

### Complete Iteration History

| Iteration | K Value | UIndex | SI | DBI | DVI | Improved |
|------|-----|--------|----|----|----|------|
"""

    for record in optimization_history:
        iteration = record['iteration']
        k = record['K']
        uindex = record['UIndex']
        si = record['SI']
        dbi = record['DBI']
        dvi = record['DVI']
        improved = 'Yes' if record['improved'] else 'No'
        report += f"| {iteration} | {k} | {uindex:.4f} | {si:.4f} | {dbi:.4f} | {dvi:.4f} | {improved} |\n"

    report += f"""
---

## Output Files

1. **Clustering results:** {results_file}
   - DEPTH: Depth
   - CLUSTER_LABEL: Cluster label
   - CLUSTER_X_PROB: Probability of belonging to each cluster (based on GMM posterior probability)

2. **Visualization figures:** results/figures/
   - depth_profile.png: Depth profile plot
   - feature_distribution.png: Feature distribution box plot
   - pca_scatter.png: PCA dimensionality reduction scatter plot
   - cluster_centers.png: Cluster centers radar chart
   - correlation_heatmap.png: Feature correlation heatmap

3. **Test report:** results/test_report.md (this document)

---

## Algorithm Parameters

| Parameter | Value |
|------|-----|
| K value search range | [5, 20] |
| Number of initial sampling points | 5 |
| Maximum iterations | 30 |
| Convergence patience count | 5 |
| Convergence threshold | 1e-4 |

*Report automatically generated by OGLCM-AKBO algorithm*
"""

    # Save report
    report_file = 'results/test_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"[OK] Test report saved: {report_file}")


if __name__ == "__main__":
    # Ensure results directory exists
    os.makedirs('results/figures', exist_ok=True)

    # Run main program
    results, clusterer = main()
