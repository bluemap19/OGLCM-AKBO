"""
AKBO Core Clustering Algorithm Module
Auto-Kmeans with Bayesian Optimization Clustering

Features:
    1. Composite clustering quality metric UIndex (SI + DBI + DVI)
    2. Gaussian Process surrogate model
    3. Bayesian optimization loop (with correct EI acquisition function)
    4. K-means++ initialization
    5. GMM-based cluster probability computation

Note:
    Feature selection has been completed during data preprocessing.
    The 4 preset optimal features are used:
    ['CON_SUB_DYNA', 'DIS_SUB_DYNA', 'HOM_SUB_DYNA', 'ENG_SUB_DYNA']
    These features were determined by prior Random Forest feature importance analysis.

Authors: Cuka & Doctor (Fuhao Zhang)
Date: 2026-03-17
Version: 2.1 (Removed Random Forest feature selection; using preset features)
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist, pdist
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Clustering Quality Metric Functions
# ============================================================================

def compute_dunn_index(X, labels, use_approximation=True):
    """
    Compute the Dunn Index (DVI) - measures compactness and separation of clusters.

    Dunn Index = min(inter-cluster distance) / max(intra-cluster diameter)
    Higher values indicate better clustering quality.

    Parameters:
        X: ndarray, feature data (N x D)
            N: number of samples, D: feature dimensionality
        labels: ndarray, cluster labels (N,)
            cluster label for each sample (0, 1, ..., K-1)
        use_approximation: bool, default True
            Whether to use the approximate algorithm (2 x max distance to centroid)
            True:  O(N) complexity, fast
            False: O(N^2) complexity, exact

    Returns:
        dvi: float, Dunn Index value
            - dvi > 1.0: Excellent
            - dvi > 0.5: Good
            - dvi < 0.5: Acceptable

    Notes:
        - Returns infinity when intra-cluster diameter is 0
        - Returns 0.0 when number of clusters < 2
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Edge case: fewer than 2 clusters
    if n_clusters < 2:
        return 0.0

    # -------------------------------------------------------------------------
    # Step 1: Compute maximum intra-cluster diameter
    # -------------------------------------------------------------------------
    max_diameter = 0.0
    centers = []

    for label in unique_labels:
        cluster_points = X[labels == label]

        # Compute cluster centroid
        center = cluster_points.mean(axis=0)
        centers.append(center)

        # Compute cluster diameter
        if len(cluster_points) > 1:
            if use_approximation:
                # Approximate: diameter ≈ 2 × max radius (distance to centroid)
                # Complexity: O(N), suitable for large datasets
                distances = np.linalg.norm(cluster_points - center, axis=1)
                diameter = 2 * np.max(distances)
            else:
                # Exact: maximum pairwise distance
                # Complexity: O(N^2), suitable for small datasets
                distances = pdist(cluster_points)
                diameter = np.max(distances) if len(distances) > 0 else 0.0

            max_diameter = max(max_diameter, diameter)

    # Edge case: all cluster diameters are 0
    if max_diameter == 0:
        return float('inf')

    # -------------------------------------------------------------------------
    # Step 2: Compute minimum inter-cluster distance (using centroid distances)
    # -------------------------------------------------------------------------
    centers = np.array(centers)
    center_distances = cdist(centers, centers, metric='euclidean')

    # Ignore diagonal (self-to-self distances)
    np.fill_diagonal(center_distances, np.inf)
    min_distance = np.min(center_distances)

    # -------------------------------------------------------------------------
    # Step 3: Compute Dunn Index
    # -------------------------------------------------------------------------
    dvi = min_distance / max_diameter

    return dvi


def compute_uindex(X, labels):
    """
    Compute the composite clustering quality metric UIndex.

    UIndex integrates three classical metrics using the formula proposed by Doctor:
        UIndex(K) = 1 / (0.1/SI(K) + DBI(K)/1.0 + 0.01/DVI(K))

    Where:
        - SI  (Silhouette Index):      measures per-sample clustering quality [-1, 1], higher is better
        - DBI (Davies-Bouldin Index):  inter-cluster separation [0, inf), lower is better
        - DVI (Dunn Index):            cluster compactness [0, inf), higher is better

    Formula design rationale:
        - Larger SI  -> smaller 0.1/SI  -> smaller denominator -> larger UIndex  (correct)
        - Smaller DBI -> smaller DBI/1.0 -> smaller denominator -> larger UIndex (correct)
        - Larger DVI  -> smaller 0.01/DVI -> smaller denominator -> larger UIndex (correct)

    Parameters:
        X: ndarray, feature data (N x D)
        labels: ndarray, cluster labels (N,)

    Returns:
        uindex: float, composite metric value (higher is better)
        metrics: dict, individual metric values
            {
                'si':     float,   # Silhouette Index
                'dbi':    float,   # Davies-Bouldin Index
                'dvi':    float,   # Dunn Index
                'uindex': float    # Composite index
            }

    Quality thresholds:
        - UIndex > 0.5: Excellent
        - UIndex > 0.2: Good
        - UIndex < 0.2: Needs improvement
    """
    n_samples = len(labels)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Edge case: single sample or single cluster
    if n_samples < 2 or n_clusters < 2:
        return 0.0, {'si': 0.0, 'dbi': float('inf'), 'dvi': 0.0, 'uindex': 0.0}

    # -------------------------------------------------------------------------
    # Step 1: Compute Silhouette Index (SI)
    # -------------------------------------------------------------------------
    si = silhouette_score(X, labels, metric='euclidean')

    # -------------------------------------------------------------------------
    # Step 2: Compute Davies-Bouldin Index (DBI)
    # -------------------------------------------------------------------------
    dbi = davies_bouldin_score(X, labels)

    # -------------------------------------------------------------------------
    # Step 3: Compute Dunn Index (DVI)
    # -------------------------------------------------------------------------
    dvi = compute_dunn_index(X, labels, use_approximation=True)

    # Handle infinity and zero edge cases
    if np.isinf(dvi) or dvi == 0:
        dvi = 10.0      # assign a large value
    if np.isinf(dbi):
        dbi = 0.0001    # avoid division by zero
    if si <= 0:
        si = 0.0001     # avoid division by zero

    # -------------------------------------------------------------------------
    # Step 4: Compute UIndex (using Doctor's formula)
    # -------------------------------------------------------------------------
    # UIndex(K) = 1 / (0.1/SI(K) + DBI(K)/1.0 + 0.01/DVI(K))
    uindex = 1.0 / (0.1/si + dbi/1.0 + 0.01/dvi)

    metrics = {
        'uindex': float(uindex),
        'si':     float(si),
        'dbi':    float(dbi),
        'dvi':    float(dvi)
    }

    return uindex, metrics


# ============================================================================
# AKBOClusterer - Auto-Kmeans with Bayesian Optimization Clusterer
# ============================================================================

class AKBOClusterer:
    """
    AKBO Clusterer (Auto-Kmeans with Bayesian Optimization)

    Core idea:
        Formulate K selection as a black-box optimization problem and use
        Bayesian Optimization to automatically find the optimal K value.

    Optimization objective:
        max K in [Kmin, Kmax]  UIndex(K)

    Attributes:
        k_range:              tuple,   K search range (K_min, K_max)
        n_init:               int,     number of initial sampling points
        max_iter:             int,     maximum Bayesian optimization iterations
        n_patience:           int,     convergence patience (early stop after N non-improving steps)
        tol:                  float,   convergence threshold
        random_state:         int,     random seed
        best_k:               int,     optimal K value
        best_labels:          ndarray, optimal cluster labels
        best_model:           KMeans,  optimal K-means model
        best_metrics:         dict,    optimal clustering metrics
        optimization_history: list,    per-iteration optimization records
        selected_features:    list,    selected feature indices
        gmm_model:            GaussianMixture, GMM model for probability computation

    Note:
        Feature selection has been completed during data preprocessing.
        The 4 preset optimal features are:
        ['CON_SUB_DYNA', 'DIS_SUB_DYNA', 'HOM_SUB_DYNA', 'ENG_SUB_DYNA']
        These were determined by prior Random Forest feature importance analysis.

    Example:
        >>> clusterer = AKBOClusterer(k_range=(2, 10), max_iter=30)
        >>> optimal_k = clusterer.optimize(X, feature_names, selected_indices)
        >>> labels = clusterer.fit(X)
    """

    def __init__(self, k_range=(2, 10), n_init=5, max_iter=30,
                 n_patience=5, tol=1e-4, random_state=42):
        """
        Initialize the AKBO clusterer.

        Parameters:
            k_range:      tuple, default (2, 10)
                          K search range [K_min, K_max)
            n_init:       int, default 5
                          number of initial sampling points (randomly chosen K values)
            max_iter:     int, default 30
                          maximum Bayesian optimization iterations
            n_patience:   int, default 5
                          convergence patience (early stop after N consecutive non-improving steps)
            tol:          float, default 1e-4
                          convergence threshold (improvement < tol is treated as no improvement)
            random_state: int, default 42
                          random seed (ensures reproducibility)
        """
        self.k_range = k_range
        self.n_init = n_init
        self.max_iter = max_iter
        self.n_patience = n_patience
        self.tol = tol
        self.random_state = random_state

        # Optimization results
        self.best_k = None
        self.best_labels = None
        self.best_model = None
        self.best_metrics = None
        self.optimization_history = []   # detailed per-iteration records

        # Feature selection (completed during data preprocessing)
        self.selected_features = None

        # GMM model (for probability computation)
        self.gmm_model = None

    def _expected_improvement(self, X, gp, best_f):
        """
        Compute the Expected Improvement (EI) acquisition function.

        EI formula:
            EI(x) = E[max(0, f(x) - f_best)]
                  = (mu - f_best) * Phi(Z) + sigma * phi(Z)

        Where:
            - mu:    GP predicted mean
            - sigma: GP predicted standard deviation
            - Z    = (mu - f_best) / sigma
            - Phi:   standard normal CDF
            - phi:   standard normal PDF

        Parameters:
            X:      ndarray, candidate K values (N x 1)
            gp:     GaussianProcessRegressor, trained GP model
            best_f: float, current best UIndex value

        Returns:
            ei: ndarray, EI value for each candidate K (N,)
        """
        # Get GP predictions (mean and standard deviation)
        mu, sigma = gp.predict(X.reshape(-1, 1), return_std=True)

        # Compute EI
        with np.errstate(divide='ignore'):
            imp = mu - best_f    # improvement
            Z   = imp / sigma    # standardized improvement

            # EI = (mu - f_best) * Phi(Z) + sigma * phi(Z)
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

            # Handle zero standard deviation
            ei[sigma == 0.0] = 0.0

        return ei

    def _initialize_sampling(self, X):
        """
        Initial sampling: randomly select K values and evaluate UIndex.

        Strategy:
            - Randomly select n_init distinct K values from the search range
            - Run K-means++ clustering for each K
            - Compute UIndex as the evaluation metric

        Parameters:
            X: ndarray, standardized feature data (N x D)

        Returns:
            K_values:      list, initial K values
            uindex_values: list, corresponding UIndex values
            models:        dict, K-means model dictionary {K: model}
        """
        print(f"\n[AKBO] Initial sampling ({self.n_init} points)...")

        K_values = []
        uindex_values = []
        models = {}

        # Set random seed for reproducibility
        np.random.seed(self.random_state)

        # Randomly select K values (without replacement)
        candidate_K = np.random.choice(
            range(self.k_range[0], self.k_range[1]),
            size=min(self.n_init, self.k_range[1] - self.k_range[0]),
            replace=False
        )

        for i, k in enumerate(candidate_K):
            print(f"      Sample {i+1}/{self.n_init}: K={k}", end=' ')

            # K-means++ clustering
            # init='k-means++': smart centroid initialization for faster convergence
            # n_init=10: run 10 times and select the best result
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=10,
                random_state=self.random_state,
                max_iter=300
            )
            labels = kmeans.fit_predict(X)

            # Compute UIndex
            uindex, metrics = compute_uindex(X, labels)

            K_values.append(k)
            uindex_values.append(uindex)
            models[k] = kmeans

            print(f"-> UIndex={uindex:.4f} (SI={metrics['si']:.3f})")

        return K_values, uindex_values, models

    def _bayesian_optimization(self, X, K_values, uindex_values, models):
        """
        Bayesian optimization loop.

        Workflow:
            1. Train a Gaussian Process (GP) surrogate model on existing data
            2. Use the EI acquisition function to select the next candidate K
            3. Run K-means clustering and compute UIndex
            4. Update the GP model and dataset
            5. Check convergence condition

        Parameters:
            X:             ndarray, feature data (N x D)
            K_values:      list, evaluated K values
            uindex_values: list, corresponding UIndex values
            models:        dict, trained K-means model dictionary

        Returns:
            K_values:      list, updated K values
            uindex_values: list, updated UIndex values
            models:        dict, updated model dictionary
        """
        print(f"\n[AKBO] Bayesian optimization (max {self.max_iter} iterations)...")

        # Current best UIndex
        best_uindex = max(uindex_values)
        patience_counter = 0

        # -------------------------------------------------------------------------
        # Initialize Gaussian Process surrogate model
        # -------------------------------------------------------------------------
        # Kernel: RBF (Radial Basis Function) + ConstantKernel
        # RBF: captures smooth relationships between K values
        # ConstantKernel: scales signal variance
        kernel = C(1.0) * RBF(length_scale=1.0)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            random_state=self.random_state
        )

        # -------------------------------------------------------------------------
        # Bayesian optimization loop
        # -------------------------------------------------------------------------
        for iteration in range(self.max_iter):

            # Prepare training data (K values as input, UIndex as output)
            X_gp = np.array(K_values).reshape(-1, 1)
            y_gp = np.array(uindex_values)

            # Train GP model
            gp.fit(X_gp, y_gp)

            # -------------------------------------------------------------------------
            # Select next K value using EI acquisition function
            # -------------------------------------------------------------------------
            # Candidate K values: all unevaluated integers in the search range
            candidate_K = np.array([
                k for k in range(self.k_range[0], self.k_range[1])
                if k not in K_values
            ])

            if len(candidate_K) == 0:
                print(f"      Iteration {iteration+1}/{self.max_iter}: all K values evaluated")
                break

            # Compute EI for each candidate K
            ei_values = self._expected_improvement(candidate_K, gp, best_uindex)

            # Select K with maximum EI
            next_idx = np.argmax(ei_values)
            next_k   = candidate_K[next_idx]

            # -------------------------------------------------------------------------
            # Run K-means clustering and compute UIndex
            # -------------------------------------------------------------------------
            print(f"      Iteration {iteration+1}/{self.max_iter}: K={next_k}", end=' ')

            kmeans = KMeans(
                n_clusters=next_k,
                init='k-means++',
                n_init=10,
                random_state=self.random_state,
                max_iter=300
            )
            labels = kmeans.fit_predict(X)

            uindex, metrics = compute_uindex(X, labels)

            # -------------------------------------------------------------------------
            # Record iteration history (detailed metrics)
            # -------------------------------------------------------------------------
            iteration_record = {
                'iteration': iteration + 1,
                'K':         next_k,
                'UIndex':    uindex,
                'SI':        metrics['si'],
                'DBI':       metrics['dbi'],
                'DVI':       metrics['dvi'],
                'improved':  uindex > best_uindex
            }
            self.optimization_history.append(iteration_record)

            # -------------------------------------------------------------------------
            # Update dataset
            # -------------------------------------------------------------------------
            K_values.append(next_k)
            uindex_values.append(uindex)
            models[next_k] = kmeans

            print(f"-> UIndex={uindex:.4f} (SI={metrics['si']:.3f}, DBI={metrics['dbi']:.3f}, DVI={metrics['dvi']:.3f})")

            # -------------------------------------------------------------------------
            # Check for improvement
            # -------------------------------------------------------------------------
            if uindex > best_uindex:
                improvement  = uindex - best_uindex
                best_uindex  = uindex
                patience_counter = 0
                print(f"         [Improved] +{improvement:.4f}")
            else:
                patience_counter += 1
                print(f"         [No improvement] patience {patience_counter}/{self.n_patience}")

            # -------------------------------------------------------------------------
            # Convergence check
            # -------------------------------------------------------------------------
            if patience_counter >= self.n_patience:
                print(f"\n[AKBO] Convergence reached ({self.n_patience} consecutive non-improving steps)")
                break

        return K_values, uindex_values, models

    def select_features(self, X, selected_indices, feature_names):
        """
        Select features by specified indices.

        Note: Feature selection has been completed during data preprocessing.
              This method only extracts the specified features.

        Parameters:
            X:                ndarray, raw feature data (N x D)
            selected_indices: list, feature index list
            feature_names:    list, feature name list

        Returns:
            X_selected:     ndarray, selected feature data (N x n_select)
            selected_names: list, selected feature names
        """
        X_selected     = X[:, selected_indices]
        selected_names = [feature_names[i] for i in selected_indices]

        self.selected_features = selected_indices

        print(f"\n[Feature Selection] {len(selected_indices)} features selected:")
        for i, idx in enumerate(selected_indices):
            print(f"  {i+1}. {feature_names[idx]}")

        return X_selected, selected_names

    def optimize(self, X, feature_names=None, selected_indices=None):
        """
        Run AKBO optimization.

        Workflow:
            1. Use specified features for clustering (feature selection done in preprocessing)
            2. Initial sampling
            3. Bayesian optimization
            4. Select optimal K value

        Parameters:
            X:                ndarray, standardized feature data (N x D)
            feature_names:    list, feature name list
            selected_indices: list, specified feature index list
                              (uses all features if None)

        Returns:
            best_k: int, optimal K value

        Note:
            Feature selection has been completed during data preprocessing.
            The 4 preset optimal features are:
            ['CON_SUB_DYNA', 'DIS_SUB_DYNA', 'HOM_SUB_DYNA', 'ENG_SUB_DYNA']
        """
        print("="*60)
        print("AKBO Clustering Optimization")
        print("="*60)
        print(f"K search range:          [{self.k_range[0]}, {self.k_range[1]}]")
        print(f"Initial sampling points: {self.n_init}")
        print(f"Maximum iterations:      {self.max_iter}")

        # -------------------------------------------------------------------------
        # Step 1: Determine features to use
        # -------------------------------------------------------------------------
        if selected_indices is not None and feature_names is not None:
            X_opt, selected_names = self.select_features(X, selected_indices, feature_names)
            print(f"\nOptimizing with {len(selected_names)} specified features")
        else:
            X_opt          = X
            selected_names = feature_names
            print("\n[Feature processing] Using all features for optimization.")

        # -------------------------------------------------------------------------
        # Step 2: Initial sampling (record history)
        # -------------------------------------------------------------------------
        K_values, uindex_values, models = self._initialize_sampling(X_opt)

        # Record initial sampling to history
        for i, (k, uindex) in enumerate(zip(K_values, uindex_values)):
            _, metrics = compute_uindex(X_opt, models[k].labels_)
            self.optimization_history.append({
                'iteration': f'initial_{i+1}',
                'K':         k,
                'UIndex':    uindex,
                'SI':        metrics['si'],
                'DBI':       metrics['dbi'],
                'DVI':       metrics['dvi'],
                'improved':  True
            })

        # -------------------------------------------------------------------------
        # Step 3: Bayesian optimization
        # -------------------------------------------------------------------------
        K_values, uindex_values, models = self._bayesian_optimization(
            X_opt, K_values, uindex_values, models
        )

        # -------------------------------------------------------------------------
        # Step 4: Select optimal result
        # -------------------------------------------------------------------------
        best_idx         = np.argmax(uindex_values)
        self.best_k      = K_values[best_idx]
        self.best_model  = models[self.best_k]
        self.best_labels = self.best_model.labels_

        # Compute optimal metrics
        _, self.best_metrics = compute_uindex(X_opt, self.best_labels)

        # Train GMM model for probability computation
        self.gmm_model = GaussianMixture(
            n_components=self.best_k,
            covariance_type='full',
            random_state=self.random_state,
            n_init=10
        )
        self.gmm_model.fit(X_opt)

        print("\n" + "="*60)
        print(f"[OK] AKBO optimization complete")
        print(f"Optimal K:              {self.best_k}")
        print(f"Optimal UIndex:         {self.best_metrics['uindex']:.4f}")
        print(f"  - Silhouette (SI):    {self.best_metrics['si']:.4f}")
        print(f"  - DBI:                {self.best_metrics['dbi']:.4f}")
        print(f"  - DVI:                {self.best_metrics['dvi']:.4f}")
        print("="*60)

        return self.best_k

    def fit(self, X):
        """
        Perform clustering (runs optimize() first if not yet done).

        Parameters:
            X: ndarray, feature data

        Returns:
            labels: ndarray, cluster labels (N,)
        """
        if self.best_model is None:
            print("[Warning] optimize() has not been called. Running automatic optimization...")
            self.optimize(X)

        return self.best_labels

    def predict(self, X):
        """
        Predict cluster labels for new samples.

        Parameters:
            X: ndarray, new sample feature data (M x D)

        Returns:
            labels: ndarray, predicted cluster labels (M,)
        """
        if self.best_model is None:
            raise ValueError("Optimization has not been run. Please call optimize() or fit() first.")

        return self.best_model.predict(X)

    def get_cluster_probs(self, X):
        """
        Get cluster probabilities (based on GMM posterior probabilities).

        Parameters:
            X: ndarray, feature data (N x D)

        Returns:
            probs: ndarray, cluster probabilities (N x K)
                   each row gives the probability of the sample belonging to each cluster

        Note:
            The GMM model is automatically trained at the end of optimize().
        """
        if self.gmm_model is None:
            raise ValueError("GMM model has not been trained. Please call optimize() first.")

        return self.gmm_model.predict_proba(X)


# ============================================================================
# Convenience Function
# ============================================================================
def akbo_clustering(X, k_range=(2, 10), max_iter=30, random_state=42):
    """
    Convenience function: run AKBO clustering.

    Parameters:
        X:            ndarray, standardized feature data (N x D)
        k_range:      tuple, K search range
        max_iter:     int, maximum iterations
        random_state: int, random seed

    Returns:
        labels:    ndarray, cluster labels (N,)
        optimal_k: int, optimal K value
        metrics:   dict, clustering quality metrics
    """
    clusterer = AKBOClusterer(
        k_range=k_range,
        max_iter=max_iter,
        random_state=random_state
    )

    optimal_k = clusterer.optimize(X)
    labels    = clusterer.fit(X)

    return labels, optimal_k, clusterer.best_metrics
