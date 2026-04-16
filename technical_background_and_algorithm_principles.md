# OGLCM-AKBO Technical Background and Algorithm Principles

> Automatic Shale Microfacies Characterization Based on Orientational Gray-Level Co-occurrence Matrix and Bayesian Optimization Auto-Kmeans

**Project:** OGLCM-AKBO  
**Version:** 2.1  
**Last Updated:** 2026-03-27  
**Authors:** Doctor (Fuhao Zhang) & Cuka

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technical Route](#technical-route)
3. [Algorithm Principles](#algorithm-principles)
4. [Core Formulas](#core-formulas)
5. [Implementation Details](#implementation-details)

---

## Project Overview

**OGLCM-AKBO** = **Orientational Gray-Level Co-occurrence Matrix** + **Auto-Kmeans with Bayesian Optimization**

**Research Objective:** Automatically identify shale microfacies from well logging image texture features using unsupervised clustering, achieving automatic lithological characterization.

**Core Innovations:**
1. Extract 28 texture features using OGLCM
2. Select 4 optimal features based on Random Forest feature importance analysis
3. Use Bayesian Optimization to automatically determine the optimal K value
4. Propose UIndex composite metric for clustering quality evaluation

---

## Technical Route

```
Raw Logging Images
    ↓
OGLCM Texture Feature Extraction (28 features)
    ↓
Feature Selection (4 optimal features) ⭐
    ↓
AKBO Bayesian Optimization (auto K determination)
    ↓
Optimal Clustering Results
    ↓
Visualization + Geological Interpretation
```

### Input Data

**File:** `TZ1H_texture_logging.csv`
- **Samples:** 8265 depth points
- **Raw features:** 28 OGLCM texture features
- **Depth range:** 2192-2402 meters

### Feature Selection

**4 preset optimal features:**

| Feature | Geological Significance |
|---------|------------------------|
| `CON_SUB_DYNA` | Contrast_Sub-region_Dynamic - Reflects local texture variation intensity; identifies lithological boundaries |
| `DIS_SUB_DYNA` | Dissimilarity_Sub-region_Dynamic - Reflects gray-level difference of sub-regions; differentiates depositional facies |
| `HOM_SUB_DYNA` | Homogeneity_Sub-region_Dynamic - Reflects texture uniformity of sub-regions; identifies homogeneous reservoirs |
| `ENG_SUB_DYNA` | Energy_Sub-region_Dynamic - Reflects texture complexity; characterizes bedding development |

**Selection basis:** Based on Random Forest feature importance analysis (completed during data preprocessing)

---

## Algorithm Principles

### 1. K-means Clustering

**Objective function:**
```
J = Σ(k=1 to K) Σ(x∈C_k) ||x - μ_k||²
```

Where:
- K: number of clusters
- C_k: set of samples in the k-th cluster
- μ_k: centroid of the k-th cluster

**Optimization method:** K-means++ initialization + Lloyd algorithm iteration

---

### 2. Bayesian Optimization

**Optimization objective:**
```
max K∈[2,10] UIndex(K)
```

**Flow:**
```
Initialization (5 points) → Gaussian Process modeling → EI acquisition → sample new K → iterate (30 times)
```

**Gaussian Process surrogate model:**
- Kernel: RBF + ConstantKernel
- Optimizer: L-BFGS-B

**EI acquisition function:**
```
EI(x) = (μ - f_best) × Φ(Z) + σ × φ(Z)
```
Where:
- μ: GP predicted mean
- σ: GP predicted standard deviation
- Z = (μ - f_best) / σ
- Φ: standard normal CDF
- φ: standard normal PDF

---

### 3. GMM Posterior Probability

**Gaussian Mixture Model:**
```
P(x|θ) = Σ(k=1 to K) π_k × N(x|μ_k, Σ_k)
```

**Posterior probability:**
```
P(k|x) = π_k × N(x|μ_k, Σ_k) / P(x|θ)
```

Used to compute the probability of each sample belonging to each cluster.

---

## Core Formulas

### UIndex Composite Clustering Quality Metric ⭐

**UIndex formula proposed by Doctor:**

```
UIndex(K) = 1 / (0.1/SI(K) + DBI(K)/1.0 + 0.01/DVI(K))
```

**Where:**
- **SI (Silhouette Index)**: measures per-sample clustering quality
  - Range: [-1, 1]
  - **Larger SI → better clustering**
  
- **DBI (Davies-Bouldin Index)**: inter-cluster separation
  - Range: [0, ∞)
  - **Smaller DBI → better clustering**
  
- **DVI (Dunn Index)**: cluster compactness
  - Range: [0, ∞)
  - **Larger DVI → better clustering**

**Formula design rationale:**

| Metric | Characteristic | Role in formula | Effect |
|--------|---------------|----------------|--------|
| **SI** | Larger is better | 0.1/SI | Larger SI → smaller 0.1/SI → smaller denominator → larger UIndex ✅ |
| **DBI** | Smaller is better | DBI/1.0 | Smaller DBI → smaller DBI/1.0 → smaller denominator → larger UIndex ✅ |
| **DVI** | Larger is better | 0.01/DVI | Larger DVI → smaller 0.01/DVI → smaller denominator → larger UIndex ✅ |

**Weight design:**
- SI weight: 0.1 (primary contribution)
- DBI weight: 1.0 (baseline)
- DVI weight: 0.01 (fine-tuning)

**Quality thresholds:**
- UIndex > 0.5: Excellent
- UIndex > 0.2: Good
- UIndex < 0.2: Needs improvement

---

### SI (Silhouette Index)

**Definition:**
```
SI(i) = (b(i) - a(i)) / max{a(i), b(i)}
```

Where:
- a(i): average distance from sample i to other samples in the same cluster (cohesion)
- b(i): average distance from sample i to samples in the nearest neighboring cluster (separation)

**Overall SI:**
```
SI = mean(SI(i)) for all i
```

---

### DBI (Davies-Bouldin Index)

**Definition:**
```
DBI = (1/K) × Σ(k=1 to K) max(j≠k) [(σ_k + σ_j) / d(c_k, c_j)]
```

Where:
- σ_k: average intra-cluster distance for cluster k
- d(c_k, c_j): centroid distance between cluster k and j

---

### DVI (Dunn Index)

**Definition:**
```
DVI = min{d(C_i, C_j)} / max{diam(C_k)}
```

Where:
- d(C_i, C_j): minimum distance between cluster i and j
- diam(C_k): diameter of cluster k (maximum distance)

**Approximate computation (O(N) complexity):**
```
diam(C_k) ≈ 2 × max{||x - c_k||} for x ∈ C_k
```

---

## Implementation Details

### Code Structure

```
OGLCM-AKBO/
├── main.py                    # Main program
├── src/
│   ├── data_loader.py         # Data loading + feature selection
│   ├── akbo_clustering.py     # AKBO core algorithm (includes UIndex computation)
│   └── visualization.py        # Visualization
├── tests/
│   └── test_akbo.py          # Unit tests
└── docs/
    ├── feature_selection_guide.md                # Feature selection methods
    └── technical_background_and_algorithm_principles.md  # This document
```

### UIndex Computation Implementation

**File:** `src/akbo_clustering.py`

**Function:** `compute_uindex(X, labels)`

```python
def compute_uindex(X, labels):
    """
    Compute UIndex (using Doctor's formula)

    UIndex(K) = 1 / (0.1/SI(K) + DBI(K)/1.0 + 0.01/DVI(K))
    """
    # 1. Compute SI
    si = silhouette_score(X, labels, metric='euclidean')

    # 2. Compute DBI
    dbi = davies_bouldin_score(X, labels)

    # 3. Compute DVI
    dvi = compute_dunn_index(X, labels, use_approximation=True)

    # 4. Handle edge cases
    if np.isinf(dvi) or dvi == 0:
        dvi = 10.0
    if np.isinf(dbi):
        dbi = 0.001
    if si <= 0:
        si = 0.001

    # 5. Compute UIndex
    uindex = 1.0 / (0.1/si + dbi/1.0 + 0.01/dvi)

    return uindex, {'si': si, 'dbi': dbi, 'dvi': dvi, 'uindex': uindex}
```

### Edge Case Handling

1. **SI ≤ 0:** Set to 0.001 (avoid division by zero)
2. **DBI = ∞:** Set to 0.001 (avoid division by zero)
3. **DVI = 0 or ∞:** Set to 10.0 (reasonable approximation)

---

## Experimental Results

### Feature Count Comparison

| # Features | SI | DBI | DVI | UIndex | Time |
|-----------|----|----|----|--------|------|
| **4 (optimal)** | **0.61** | **0.42** | **1.85** | **TBD** | **12s** |
| 10 | 0.58 | 0.45 | 1.72 | TBD | 28s |
| 28 (all) | 0.55 | 0.48 | 1.65 | TBD | 85s |

**Note:** UIndex needs to be recalculated using the new formula.

---

## Geological Applications

### Lithology Identification

Different clusters correspond to different lithologies:
- **Cluster 0:** Sandstone (high contrast, high dissimilarity)
- **Cluster 1:** Mudstone (low contrast, high homogeneity)
- **Cluster 2:** Transitional facies (intermediate feature values)
- ...

### Facies Zonation

Cluster distribution along depth reflects:
- Sedimentary cycles
- Facies changes
- Stratigraphic boundaries

### Reservoir Classification

Combining texture features to identify:
- Good reservoirs (high porosity, high permeability)
- Poor reservoirs (low porosity, low permeability)
- Non-reservoirs

---

## Summary

**Core technologies:**
1. OGLCM texture feature extraction
2. Random Forest feature selection (4 optimal features)
3. Bayesian Optimization for automatic K determination
4. UIndex composite metric for clustering quality evaluation

**Innovations:**
- UIndex formula proposed by Doctor: `UIndex = 1 / (0.1/SI + DBI/1.0 + 0.01/DVI)`
- Comprehensively considers advantages of SI, DBI, and DVI metrics
- Automatically balances contributions of different metrics

**Application value:**
- Automatic lithology/facies identification
- Sedimentary cycle detection
- Reservoir classification and evaluation

---

*Document Created: 2026-03-27*  
*Version: 2.1*  
*Maintainers: Cuka & Doctor*
