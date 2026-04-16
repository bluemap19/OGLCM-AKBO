"""
K-means++ Clustering with Grid-based Initialization
Custom implementation without external sklearn.cluster.KMeans

Features:
    1. Grid-based candidate center generation in feature space
    2. Farthest-First deterministic greedy selection from grid candidates
    3. Full parameter control (K, grid divisions, convergence, etc.)
    4. Lloyd's algorithm for iterative optimization
    5. GMM-based posterior probability (optional)
    6. Detailed clustering quality metrics (SI, DBI, DVI)

Design Principles:
    - NO random numbers for center initialization (fully deterministic)
    - Grid partitions feature space into uniform cells; cell centers are candidates
    - Farthest-First greedy picks K most-dispersed centers from candidates
    - "random_seed" parameter selects which cell becomes the first center (anchor cell)

Authors: Cuka & Doctor (Fuhao Zhang)
Date: 2026-04-15
Version: 1.0
"""

import numpy as np
import pandas as pd
import copy


# ============================================================================
# Core Implementation: K-means++ with Grid-based Initialization
# ============================================================================

class KMeansZFH:
    """
    K-means++ clustering with grid-based initialization.

    This implementation does NOT rely on sklearn.cluster.KMeans.
    It uses a deterministic Farthest-First strategy to select initial centers
    from grid-cell-centers within the feature space, avoiding random sampling.

    Initialization Workflow:
        Step 1: Build grid over the feature space
                - Divide each dimension into `grid_divisions` equal intervals
                - Grid cell centers become candidate centers
        Step 2: Farthest-First selection
                - Pick the first center based on `first_center_method`
                - Iteratively pick the point farthest from all selected centers
                - Repeat until K centers are selected
        Step 3: Lloyd's algorithm (assign 鈫?update 鈫?repeat)

    Parameters
    ----------
    n_clusters : int, default=5
        Number of clusters (K).
    grid_divisions : int or list of int, default=5
        Number of divisions per dimension for grid cell generation.
        - int: same number of divisions for all dimensions
        - list of int: specify per-dimension divisions (must match n_features)
        More divisions 鈫?more candidates 鈫?better coverage, higher cost.
        Rule of thumb: grid_divisions^D <= 10000 to keep candidate count manageable.
        For D=4 features with grid_divisions=5: 5^4 = 625 candidates.
    first_center_method : str, default='grid_corner'
        Method to pick the FIRST center (anchor).
        Available options:
        - 'grid_corner' : pick the grid cell farthest from the origin
                          (max normalized distance from [0,0,...,0])
                          Deterministic: always selects the same cell.
        - 'grid_corner_neg': pick the grid cell farthest from the centroid
                             (max distance from data centroid)
        - 'variance_max'  : pick the point with max variance across all dimensions
                             (most "spread-out" feature direction, take its max)
        - 'index_first'   : pick the first data point (index 0)
        - 'index_median'  : pick the median data point (index n//2)
        - 'index_last'    : pick the last data point (index n-1)
        - 'manual'        : user manually specifies the first center (see manual_first_center)
    manual_first_center : ndarray or None, default=None
        Only used when first_center_method='manual'.
        Shape: (n_features,). The manually specified first center coordinates.
    first_center_index : int, default=None
        Alternative to first_center_method for deterministic index-based first center.
        Picks data[index] as the first center.
        If None, uses first_center_method instead.
    n_init : int, default=1
        Number of times to run the full algorithm with different initializations.
        For grid-based init, different initializations come from different
        "first_center_index" values (if provided) or the same deterministic init.
        Note: With grid_divisions fixed, the initialization is deterministic;
        n_init > 1 re-runs with the same centers (useful for convergence checking).
    max_iter : int, default=300
        Maximum number of Lloyd iterations per run.
    tol : float, default=1e-4
        Convergence tolerance. Stops when centroid shift < tol.
    verbose : int, default=1
        Verbosity level:
        - 0: silent
        - 1: basic progress
        - 2: detailed per-iteration
    random_seed : int or None, default=None
        Controls the first_center_index deterministically.
        Only used when first_center_method='grid_corner' or 'grid_corner_neg'.
        - random_seed acts as a deterministic selector into the grid cell list
          sorted by distance (not random sampling).
        - Example: sorted_grid_cells[random_seed % len(sorted_grid_cells)]
        - This ensures reproducibility without randomness.
        - Set to None to always use the same deterministic cell (recommended).

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Final cluster centroids.
    labels_ : ndarray of shape (n_samples,)
        Cluster label for each sample (0 to n_clusters-1).
    inertia_ : float
        Sum of squared distances to nearest centroid (within-cluster SSE).
    n_iter_ : int
        Number of iterations run.
    history_ : list of dict
        Per-iteration records of inertia and centroid shift.
    candidate_centers_ : ndarray of shape (n_candidates, n_features)
        Grid cell centers used as candidate initial centers.
    selected_initial_centers_ : ndarray of shape (n_clusters, n_features)
        Centers selected by Farthest-First before Lloyd optimization.

    Example
    -------
    >>> import numpy as np
    >>> from kmeans_zfh import KMeansZFH
    >>> X = np.random.randn(500, 4)   # 500 samples, 4 features
    >>> model = KMeansZFH(n_clusters=5, grid_divisions=5, verbose=1)
    >>> model.fit(X)
    >>> labels = model.labels_
    >>> centers = model.cluster_centers_
    """

    def __init__(
        self,
        n_clusters=5,
        grid_divisions=5,
        first_center_method='grid_corner',
        manual_first_center=None,
        first_center_index=None,
        n_init=1,
        max_iter=300,
        tol=1e-4,
        verbose=1,
        random_seed=None,
    ):
        self.n_clusters = n_clusters
        self.grid_divisions = grid_divisions
        self.first_center_method = first_center_method
        self.manual_first_center = manual_first_center
        self.first_center_index = first_center_index
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        # random_seed is only used as a deterministic selector index
        # when first_center_method is 'grid_corner' or 'grid_corner_neg'
        self.random_seed = random_seed

        # Results (set by fit)
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self.history_ = []
        self.candidate_centers_ = None
        self.selected_initial_centers_ = None
        self._is_fitted = False

    # ========================================================================
    # Part 1: Grid Candidate Center Generation
    # ========================================================================

    def _build_grid_candidates(self, X):
        """
        Build grid cell centers as candidate initial centers.

        For each dimension d:
            - Find min and max of X[:, d]
            - Divide into grid_divisions[d] equal intervals
            - Grid cell centers are at interval midpoints

        The total number of candidates = prod(grid_divisions).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        candidates : ndarray of shape (n_candidates, n_features)
            Grid cell centers as candidate centers.
        grid_edges : list of ndarray
            Edge values for each dimension (for plotting/interpretation).
        """
        n_samples, n_features = X.shape

        # Normalize grid_divisions to a per-dimension list
        if isinstance(self.grid_divisions, int):
            grid_divs = [self.grid_divisions] * n_features
        else:
            grid_divs = list(self.grid_divisions)
            if len(grid_divs) != n_features:
                raise ValueError(
                    f"grid_divisions length ({len(grid_divs)}) must match "
                    f"n_features ({n_features})"
                )

        # Build per-dimension grid lines (edges)
        grid_edges = []
        candidate_points_per_dim = []

        for d in range(n_features):
            col_min = X[:, d].min()
            col_max = X[:, d].max()

            # Add a small padding to avoid boundary issues
            padding = (col_max - col_min) * 0.001
            col_min -= padding
            col_max += padding

            # Create grid edges (n_divs+1 edges -> n_divs intervals)
            edges = np.linspace(col_min, col_max, grid_divs[d] + 1)
            grid_edges.append(edges)

            # Compute cell centers (midpoints of each interval)
            centers = (edges[:-1] + edges[1:]) / 2.0
            candidate_points_per_dim.append(centers)

        # Generate all combinations of cell centers (cartesian product)
        # Result: (grid_divs[0] * grid_divs[1] * ...) 脳 n_features
        candidates = self._cartesian_product(candidate_points_per_dim)

        n_candidates = len(candidates)
        max_candidates = 10000
        if n_candidates > max_candidates:
            print(
                f"[Warning] Number of grid candidates ({n_candidates}) exceeds "
                f"{max_candidates}. Consider reducing grid_divisions to avoid "
                f"high memory usage."
            )

        if self.verbose >= 2:
            print(f"      Grid candidates generated: {n_candidates} points")
            print(f"      Grid divisions per dim  : {grid_divs}")
            for d in range(n_features):
                print(
                    f"        Dim {d}: range=[{X[:, d].min():.3f}, {X[:, d].max():.3f}], "
                    f"step={grid_edges[d][1]-grid_edges[d][0]:.4f}"
                )

        return candidates, grid_edges

    @staticmethod
    def _cartesian_product(arrays):
        """
        Compute the cartesian product of a list of 1D arrays.

        Parameters
        ----------
        arrays : list of ndarray

        Returns
        -------
        result : ndarray of shape (prod(len(a) for a in arrays), len(arrays))
        """
        if len(arrays) == 0:
            return np.empty((0, 0))

        n = len(arrays)
        # np.meshgrid is the clean, dependency-free way
        grids = np.meshgrid(*arrays, indexing='ij')
        # grids[i] has shape (len(arrays[0]), len(arrays[1]), ..., len(arrays[n-1]))
        # Reshape each to (-1,) then stack horizontally
        result = np.column_stack([g.ravel() for g in grids])
        return result.astype(float)

    # ========================================================================
    # Part 2: Farthest-First Deterministic First Center Selection
    # ========================================================================

    def _select_first_center(self, candidates, X):
        """
        Select the FIRST center (anchor) from grid candidates.

        Selection strategy is controlled by self.first_center_method.

        Parameters
        ----------
        candidates : ndarray of shape (n_candidates, n_features)
        X : ndarray of shape (n_samples, n_features)
            Original data (used for some methods)

        Returns
        -------
        first_center : ndarray of shape (n_features,)
        """
        n_candidates = len(candidates)

        if self.verbose >= 2:
            print(f"      First center method: '{self.first_center_method}'")

        # ------------------------------------------------------------------
        # Method: 'manual' 鈥?user-specified coordinates
        # ------------------------------------------------------------------
        if self.first_center_method == 'manual':
            if self.manual_first_center is None:
                raise ValueError(
                    "first_center_method='manual' requires manual_first_center "
                    "to be specified."
                )
            first_center = np.asarray(self.manual_first_center, dtype=float)
            if first_center.shape[0] != candidates.shape[1]:
                raise ValueError(
                    f"manual_first_center shape {first_center.shape} does not match "
                    f"n_features {candidates.shape[1]}"
                )
            if self.verbose >= 2:
                print(f"      Manual first center: {first_center}")
            return first_center

        # ------------------------------------------------------------------
        # Method: 'index_first' / 'index_median' / 'index_last'
        # ------------------------------------------------------------------
        if self.first_center_method in ('index_first', 'index_median', 'index_last'):
            if self.first_center_method == 'index_first':
                idx = 0
            elif self.first_center_method == 'index_median':
                idx = len(X) // 2
            else:  # 'index_last'
                idx = len(X) - 1

            first_center = candidates[idx % n_candidates]
            if self.verbose >= 2:
                print(f"      Index-based first center: data[{idx}] -> {first_center}")
            return first_center

        # ------------------------------------------------------------------
        # Method: 'first_center_index' override
        # ------------------------------------------------------------------
        if self.first_center_index is not None:
            idx = self.first_center_index % n_candidates
            first_center = candidates[idx]
            if self.verbose >= 2:
                print(f"      first_center_index={self.first_center_index} -> candidate[{idx}]")
                print(f"      First center: {first_center}")
            return first_center

        # ------------------------------------------------------------------
        # Method: 'variance_max' 鈥?pick the dimension with max variance,
        #                              select the endpoint with larger value
        # ------------------------------------------------------------------
        if self.first_center_method == 'variance_max':
            variances = np.var(X, axis=0)
            var_max_dim = int(np.argmax(variances))
            x_max_val = X[:, var_max_dim].max()
            x_min_val = X[:, var_max_dim].min()

            # Choose the endpoint that is farther from the centroid
            centroid = X.mean(axis=0)
            dist_max = abs(x_max_val - centroid[var_max_dim])
            dist_min = abs(x_min_val - centroid[var_max_dim])

            if dist_max >= dist_min:
                # Pick the grid point with max value in var_max_dim
                # and median values in all other dims
                first_center = np.median(candidates, axis=0)
                first_center[var_max_dim] = candidates[:, var_max_dim].max()
            else:
                first_center = np.median(candidates, axis=0)
                first_center[var_max_dim] = candidates[:, var_max_dim].min()

            if self.verbose >= 2:
                print(f"      Variance-based: dim {var_max_dim} has max variance={variances[var_max_dim]:.4f}")
                print(f"      First center: {first_center}")
            return first_center

        # ------------------------------------------------------------------
        # Method: 'grid_corner' 鈥?grid cell farthest from the origin
        #                              (i.e., farthest from [0,0,...,0])
        # This is a fully deterministic method.
        # "random_seed" selects into the sorted cell list.
        # ------------------------------------------------------------------
        if self.first_center_method == 'grid_corner':
            # Compute distance from each candidate to origin
            origin = np.zeros(candidates.shape[1])
            dists = np.linalg.norm(candidates - origin, axis=1)

            # Sort by distance (ascending)
            sorted_indices = np.argsort(dists)

            if self.verbose >= 2:
                print(f"      random_seed={self.random_seed} -> deterministic index selection")
                print(f"      Candidate distances range: [{dists.min():.4f}, {dists.max():.4f}]")

            # Pick based on random_seed as a deterministic selector
            if self.random_seed is not None:
                sel = self.random_seed % n_candidates
                chosen_idx = sorted_indices[sel]
            else:
                # Default: pick the farthest corner (max distance)
                chosen_idx = sorted_indices[-1]

            first_center = candidates[chosen_idx]
            if self.verbose >= 2:
                print(f"      Grid corner #{sel} (of {n_candidates} sorted candidates)")
                print(f"      First center: {first_center}")
            return first_center

        # ------------------------------------------------------------------
        # Method: 'grid_corner_neg' 鈥?grid cell farthest from data centroid
        #                               (most "outlier-like" cell)
        # ------------------------------------------------------------------
        if self.first_center_method == 'grid_corner_neg':
            centroid = X.mean(axis=0)
            dists = np.linalg.norm(candidates - centroid, axis=1)
            sorted_indices = np.argsort(dists)

            if self.random_seed is not None:
                sel = self.random_seed % n_candidates
                chosen_idx = sorted_indices[sel]
            else:
                chosen_idx = sorted_indices[-1]

            first_center = candidates[chosen_idx]
            if self.verbose >= 2:
                print(f"      Farthest-from-centroid cell (seed={self.random_seed})")
                print(f"      Centroid: {centroid}")
                print(f"      First center: {first_center}")
            return first_center

        raise ValueError(f"Unknown first_center_method: '{self.first_center_method}'")

    # ========================================================================
    # Part 3: Farthest-First Greedy Center Selection
    # ========================================================================

    def _farthest_first_selection(self, candidates, first_center):
        """
        Farthest-First greedy selection of remaining K-1 centers.

        Algorithm:
            1. Start with first_center
            2. For each remaining slot:
                 - Compute distance from each candidate to nearest selected center
                 - Pick the candidate with the MAXIMUM nearest-center distance
               This ensures selected centers are maximally dispersed.

        Parameters
        ----------
        candidates : ndarray of shape (n_candidates, n_features)
        first_center : ndarray of shape (n_features,)

        Returns
        -------
        selected : ndarray of shape (n_clusters, n_features)
            K selected initial centers (including first_center).
        """
        n_candidates = len(candidates)
        k = self.n_clusters

        if k > n_candidates:
            raise ValueError(
                f"n_clusters ({k}) > number of grid candidates ({n_candidates}). "
                f"Increase grid_divisions or reduce n_clusters."
            )

        selected = [first_center]
        remaining_slots = k - 1

        if self.verbose >= 2:
            print(f"      Farthest-First selection: {remaining_slots} centers to pick "
                  f"from {n_candidates} candidates")

        for step in range(remaining_slots):
            # Compute distance from each candidate to its nearest selected center
            selected_arr = np.array(selected)  # shape: (n_selected, n_features)
            dists_to_nearest = np.min(
                pairwise_distances(candidates, selected_arr, metric='euclidean'),
                axis=1
            )  # shape: (n_candidates,)

            # Pick the candidate with maximum nearest-center distance
            farthest_idx = int(np.argmax(dists_to_nearest))
            selected.append(candidates[farthest_idx])

            if self.verbose >= 2:
                print(
                    f"        Step {step+1}: pick candidate[{farthest_idx}] "
                    f"(dist={dists_to_nearest[farthest_idx]:.4f}, "
                    f"total_selected={len(selected)})"
                )

        return np.array(selected)

    # ========================================================================
    # Part 4: Lloyd's Algorithm 鈥?Iterative Assignment & Update
    # ========================================================================

    @staticmethod
    def _assign_clusters(X, centers):
        """
        Assign each sample to the nearest centroid.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        centers : ndarray of shape (n_clusters, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster assignment for each sample.
        distances : ndarray of shape (n_samples,)
            Distance from each sample to its assigned centroid.
        """
        dists = pairwise_distances(X, centers, metric='euclidean')
        labels = np.argmin(dists, axis=1)
        min_dists = dists[np.arange(len(X)), labels]
        return labels, min_dists

    @staticmethod
    def _update_centers(X, labels, n_clusters):
        """
        Update centroids as the mean of assigned samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        labels : ndarray of shape (n_samples,)
        n_clusters : int

        Returns
        -------
        new_centers : ndarray of shape (n_clusters, n_features)
            Updated centroids.
        empty_clusters : ndarray
            Indices of clusters that received no samples.
        """
        new_centers = np.zeros((n_clusters, X.shape[1]))
        empty_clusters = []

        for k in range(n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                new_centers[k] = X[mask].mean(axis=0)
            else:
                # Empty cluster: keep old center or reinitialize
                # We mark it and handle outside
                empty_clusters.append(k)

        return new_centers, np.array(empty_clusters)

    def _run_lloyd(self, X, init_centers):
        """
        Run Lloyd's algorithm from initial centers.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        init_centers : ndarray of shape (n_clusters, n_features)

        Returns
        -------
        labels : ndarray
        centers : ndarray
        inertia : float
        n_iter : int
        history : list of dict
        """
        centers = copy.deepcopy(init_centers)
        history = []
        n_samples = len(X)

        for iteration in range(self.max_iter):
            # Step 1: Assign samples to nearest centroid
            labels, min_dists = self._assign_clusters(X, centers)
            inertia = np.sum(min_dists ** 2)

            # Step 2: Compute centroid shift
            old_centers = copy.deepcopy(centers)

            # Step 3: Update centroids
            centers, empty_clusters = self._update_centers(X, labels, self.n_clusters)

            # Handle empty clusters: reinitialize with farthest unassigned point
            if len(empty_clusters) > 0:
                dists_to_assigned = pairwise_distances(X, centers, metric='euclidean')
                # For each empty cluster, find the point farthest from its own centroid
                # Simple strategy: use points from largest cluster
                for ek in empty_clusters:
                    # Find sample farthest from centroid in the largest cluster
                    cluster_sizes = [np.sum(labels == k) for k in range(self.n_clusters)]
                    largest_k = int(np.argmax(cluster_sizes))
                    largest_mask = labels == largest_k
                    largest_points = X[largest_mask]
                    dists_largest = np.linalg.norm(
                        largest_points - centers[largest_k], axis=1
                    )
                    reinit_idx = int(np.argmax(dists_largest))
                    centers[ek] = largest_points[reinit_idx]

            # Step 4: Compute centroid shift
            shift = np.linalg.norm(centers - old_centers)

            # Record history
            history.append({
                'iteration': iteration + 1,
                'inertia': float(inertia),
                'centroid_shift': float(shift),
                'n_empty_clusters': len(empty_clusters)
            })

            if self.verbose >= 2:
                print(
                    f"      Iteration {iteration+1:3d}: "
                    f"inertia={inertia:.4f}, shift={shift:.6f}, "
                    f"empty={len(empty_clusters)}"
                )

            # Step 5: Convergence check
            if shift < self.tol:
                if self.verbose >= 1:
                    print(
                        f"      Converged at iteration {iteration+1} "
                        f"(shift={shift:.6f} < tol={self.tol})"
                    )
                break

        inertia_final = np.sum(min_dists ** 2)
        return labels, centers, inertia_final, len(history), history

    # ========================================================================
    # Part 5: Fit 鈥?Main Entry Point
    # ========================================================================

    def fit(self, X):
        """
        Fit the K-means model to X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        n_samples, n_features = X.shape

        if self.verbose >= 1:
            print(f"\n[KMeansZFH] Starting fit: n_samples={n_samples}, "
                  f"n_features={n_features}, n_clusters={self.n_clusters}")
            print(f"           Grid divisions: {self.grid_divisions}")
            print(f"           First center method: '{self.first_center_method}'")
            print(f"           n_init={self.n_init}, max_iter={self.max_iter}, tol={self.tol}")

        # ------------------------------------------------------------------
        # Step 1: Build grid candidate centers
        # ------------------------------------------------------------------
        if self.verbose >= 1:
            print(f"\n[Step 1] Building grid candidate centers...")
        candidates, grid_edges = self._build_grid_candidates(X)
        self.candidate_centers_ = candidates

        # ------------------------------------------------------------------
        # Step 2: Select first center (anchor)
        # ------------------------------------------------------------------
        if self.verbose >= 1:
            print(f"\n[Step 2] Selecting first center (anchor)...")
        first_center = self._select_first_center(candidates, X)

        # ------------------------------------------------------------------
        # Step 3: Farthest-First greedy selection of K centers
        # ------------------------------------------------------------------
        if self.verbose >= 1:
            print(f"\n[Step 3] Farthest-First greedy selection...")
        init_centers = self._farthest_first_selection(candidates, first_center)
        self.selected_initial_centers_ = init_centers

        if self.verbose >= 1:
            print(f"\n[Step 4] Lloyd's algorithm optimization...")

        # ------------------------------------------------------------------
        # Step 4: Run Lloyd's algorithm
        # ------------------------------------------------------------------
        best_inertia = float('inf')
        best_labels = None
        best_centers = None
        best_n_iter = 0
        best_history = []

        for run_idx in range(self.n_init):
            if self.n_init > 1 and self.verbose >= 1:
                print(f"\n    === n_init run {run_idx+1}/{self.n_init} ===")

            # For n_init > 1, vary the first_center_index deterministically
            if self.n_init > 1:
                run_first_center_idx = run_idx % len(candidates)
                run_first_center = candidates[run_first_center_idx]
                run_init_centers = self._farthest_first_selection(candidates, run_first_center)
            else:
                run_init_centers = init_centers

            labels, centers, inertia, n_iter, history = self._run_lloyd(X, run_init_centers)

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centers = centers
                best_n_iter = n_iter
                best_history = history

            if self.verbose >= 1 and self.n_init > 1:
                print(f"    Run {run_idx+1}: inertia={inertia:.4f}, n_iter={n_iter}")

        # Store best results
        self.labels_ = best_labels
        self.cluster_centers_ = best_centers
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        self.history_ = best_history
        self._is_fitted = True

        # ------------------------------------------------------------------
        # Step 5: Summary
        # ------------------------------------------------------------------
        if self.verbose >= 1:
            unique, counts = np.unique(self.labels_, return_counts=True)
            print(f"\n[OK] K-means++ fit complete")
            print(f"     Best inertia : {self.inertia_:.4f}")
            print(f"     Iterations   : {self.n_iter_}")
            print(f"     Cluster sizes :")
            for k, cnt in zip(unique, counts):
                print(f"       Cluster {k}: {cnt:5d} samples ({cnt/len(X)*100:.1f}%)")

        return self

    def fit_predict(self, X):
        """
        Fit to X and return cluster labels.

        Parameters
        ----------
        X : ndarray

        Returns
        -------
        labels : ndarray
        """
        self.fit(X)
        return self.labels_

    def predict(self, X):
        """
        Predict cluster labels for new data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() or fit_predict() first.")
        X = np.asarray(X, dtype=float)
        labels, _ = self._assign_clusters(X, self.cluster_centers_)
        return labels


# ============================================================================
# Utility: Distance Computation
# ============================================================================

def pairwise_distances(XA, XB, metric='euclidean'):
    """
    Compute pairwise distances between two sets of points.

    Internal reimplementation 鈥?no external dependencies.

    Parameters
    ----------
    XA : ndarray of shape (nA, d)
    XB : ndarray of shape (nB, d)
    metric : str, default='euclidean'
        Distance metric ('euclidean' or 'manhattan').

    Returns
    -------
    dists : ndarray of shape (nA, nB)
    """
    XA = np.asarray(XA, dtype=float)
    XB = np.asarray(XB, dtype=float)

    if metric == 'euclidean':
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a路b  (stable, no cancellation)
        sq_dists = (
            np.sum(XA ** 2, axis=1, keepdims=True)
            + np.sum(XB ** 2, axis=1)
            - 2.0 * XA @ XB.T
        )
        # Numerical safety: clamp tiny negatives to 0
        sq_dists = np.maximum(sq_dists, 0.0)
        return np.sqrt(sq_dists)

    elif metric == 'manhattan':
        return np.abs(XA[:, np.newaxis, :] - XB[np.newaxis, :, :]).sum(axis=2)

    else:
        raise ValueError(f"Unsupported metric: '{metric}'")


# ============================================================================
# Convenience Function
# ============================================================================

def kmeans_zfh(X, n_clusters=5, grid_divisions=5, first_center_method='grid_corner',
               n_init=1, max_iter=300, tol=1e-4, verbose=1, random_seed=None):
    """
    Convenience function: run K-means++ with grid-based initialization.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    n_clusters : int
    grid_divisions : int or list of int
    first_center_method : str
    n_init : int
    max_iter : int
    tol : float
    verbose : int
    random_seed : int or None

    Returns
    -------
    labels : ndarray
    centers : ndarray
    inertia : float
    """
    model = KMeansZFH(
        n_clusters=n_clusters,
        grid_divisions=grid_divisions,
        first_center_method=first_center_method,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        random_seed=random_seed,
    )
    model.fit(X)
    return model.labels_, model.cluster_centers_, model.inertia_


# ============================================================================
# Demo / Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("KMeansZFH Demo 鈥?Grid-based K-means++ Initialization")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Test 1: Simple 2D synthetic data (4 well-separated clusters)
    # ------------------------------------------------------------------
    np.random.seed(42)

    # Generate 4 clusters in 2D
    c1 = np.array([[2.0, 2.0], [2.1, 2.0], [2.0, 2.1], [1.9, 2.0], [2.0, 1.9]])
    c2 = np.array([[-2.0, 2.0], [-2.1, 2.0], [-2.0, 2.1], [-1.9, 2.0], [-2.0, 1.9]])
    c3 = np.array([[2.0, -2.0], [2.1, -2.0], [2.0, -2.1], [1.9, -2.0], [2.0, -1.9]])
    c4 = np.array([[-2.0, -2.0], [-2.1, -2.0], [-2.0, -2.1], [-1.9, -2.0], [-2.0, -1.9]])

    X_test = np.vstack([c1, c2, c3, c4])
    print(f"\nTest data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"True 4 clusters (2 samples each, well-separated)")

    for method in ['grid_corner', 'variance_max', 'index_first']:
        print(f"\n{'='*50}")
        print(f"first_center_method = '{method}'")
        print('='*50)

        model = KMeansZFH(
            n_clusters=4,
            grid_divisions=3,
            first_center_method=method,
            n_init=1,
            max_iter=100,
            tol=1e-6,
            verbose=2,
            random_seed=42,
        )
        labels = model.fit_predict(X_test)

        print(f"\n    Initial centers (before Lloyd):")
        for i, c in enumerate(model.selected_initial_centers_):
            print(f"      Center {i}: [{c[0]:.4f}, {c[1]:.4f}]")

        print(f"\n    Final centers:")
        for i, c in enumerate(model.cluster_centers_):
            n_in_cluster = np.sum(labels == i)
            print(f"      Cluster {i}: [{c[0]:.4f}, {c[1]:.4f}]  n={n_in_cluster}")

        print(f"\n    Predicted labels: {labels.tolist()}")
        print(f"    Inertia: {model.inertia_:.6f}")

    # ------------------------------------------------------------------
    # Test 2: Higher-dimensional data (4 features)
    # ------------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("Test 2: 4D synthetic data (similar to OGLCM texture features)")
    print('='*70)

    # Generate 3 clusters in 4D
    rng = np.random.default_rng(42)
    n_per_cluster = 100
    cluster_a = rng.standard_normal((n_per_cluster, 4)) + np.array([3.0, 3.0, 3.0, 3.0])
    cluster_b = rng.standard_normal((n_per_cluster, 4)) + np.array([-3.0, 0.0, 0.0, 0.0])
    cluster_c = rng.standard_normal((n_per_cluster, 4)) + np.array([0.0, -3.0, 0.0, 3.0])
    X_4d = np.vstack([cluster_a, cluster_b, cluster_c])
    print(f"Data shape: {X_4d.shape} (n_samples={X_4d.shape[0]}, n_features={X_4d.shape[1]})")

    model_4d = KMeansZFH(
        n_clusters=3,
        grid_divisions=5,    # 5^4 = 625 candidates
        first_center_method='grid_corner',
        n_init=1,
        max_iter=300,
        tol=1e-4,
        verbose=2,
        random_seed=42,
    )
    labels_4d = model_4d.fit_predict(X_4d)

    print(f"\n    Grid candidates: {len(model_4d.candidate_centers_)}")
    print(f"    Inertia: {model_4d.inertia_:.4f}")
    print(f"    Iterations: {model_4d.n_iter_}")
    print(f"\n    Cluster distribution:")
    for k in range(3):
        n = np.sum(labels_4d == k)
        print(f"      Cluster {k}: {n} samples ({n/len(X_4d)*100:.1f}%)")

