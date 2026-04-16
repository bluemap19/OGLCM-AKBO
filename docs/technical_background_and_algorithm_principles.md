# OGLCM-AKBO Technical Background and Algorithm Principles

> Automatic Shale Microfacies Characterization Based on Orientational Gray-Level Co-occurrence Matrix and Bayesian Optimization Auto-Kmeans

**Project:** OGLCM-AKBO  
**Version:** 1.0  
**Created:** 2026-03-17  
**Authors:** Doctor (Fuhao Zhang) & Cuka

---

## 1. Project Overview

### 1.1 Research Background

Shale reservoirs exhibit strong heterogeneity and rapid lithofacies variations, making traditional well logging interpretation methods inadequate for fine-scale microfacies characterization. Borehole electrical imaging logging provides centimeter-scale resolution images of formation resistivity distribution, but automatically extracting lithofacies information from these images remains a challenge.

### 1.2 Research Objectives

Develop an **unsupervised learning** workflow to:
1. Automatically extract texture features from borehole image logging data
2. Automatically determine the optimal number of clusters without labeled data
3. Achieve automatic delineation of shale microfacies

### 1.3 Method Name Explanation

| Abbr. | Full Name | Meaning |
|-------|-----------|---------|
| **OGLCM** | Orientational Gray-Level Co-occurrence Matrix | Directional Gray-Level Co-occurrence Matrix |
| **AKBO** | Auto-Kmeans with Bayesian Optimization | Bayesian Optimization Auto-Adaptive K-means |

---

## 2. Algorithm Principles

### 2.1 OGLCM - Orientational Gray-Level Co-occurrence Matrix

#### 2.1.1 GLCM Fundamentals

The **Gray-Level Co-occurrence Matrix (GLCM)** is a second-order statistical texture analysis method that characterizes texture by quantifying the co-occurrence of gray levels between pixel pairs in an image.

**Mathematical Definition:**

For an image I, given the spatial relationship (d, θ) of pixel pairs, GLCM is defined as:

```
GLCM(i, j | d, θ) = #{(p₁, p₂) | I(p₁)=i, I(p₂)=j, dist(p₁,p₂)=d, angle(p₁,p₂)=θ}
```

Where:
- `i, j`: gray levels (typically quantized to 16/32/64 levels)
- `d`: pixel distance (positive integer, typically d=1,2,3)
- `θ`: direction angle (0°, 45°, 90°, 135°)

**Normalization:**

```
P(i,j) = GLCM(i,j) / k
```

Where k is the maximum gray level; after normalization, P(i,j) represents a probability distribution.

#### 2.1.2 Seven Core Texture Features

| Feature | Symbol | Formula | Physical Meaning | Range |
|---------|--------|---------|------------------|-------|
| **Energy** | ENG | Σ P(i,j)² | Texture uniformity / energy concentration | [0, 1] |
| **Contrast** | CON | Σ(i-j)²·P(i,j) | Texture sharpness / local gray variation | [0, ∞) |
| **Entropy** | ENT | -Σ P(i,j)·log(P(i,j)) | Information content / randomness | [0, ∞) |
| **ASM** | ASM | Σ P(i,j)² | Angular second-order moment, uniformity (same as ENG) | [0, 1] |
| **Correlation** | COR | Σ(i·j·P(i,j) - μ²)/σ² | Gray-level linear correlation | [-1, 1] |
| **Homogeneity** | HOM | Σ P(i,j)/(1+\|i-j\|) | Texture smoothness / local uniformity | [0, 1] |
| **Dissimilarity** | DIS | Σ \|i-j|·P(i,j) | Local gray difference intensity | [0, ∞) |

#### 2.1.3 Directional Feature Extraction Strategy

**Meaning of "Orientational":**

The OGLCM method in this research emphasizes **multi-directional** and **multi-property** texture feature extraction:

1. **Directional dimension:**
   - 0° (horizontal) - reflects horizontal bedding characteristics
   - 90° (vertical) - reflects vertical fractures/laminations
   - **Directional difference (X-Y)** - reflects anisotropy

2. **Statistical dimension:**
   - **MEAN** - multi-directional average, reflects overall texture
   - **SUB** - sub-region statistics, reflects local variations
   - **X** - X-direction (horizontal) features
   - **Y** - Y-direction (vertical) features

3. **Image type:**
   - **DYNA (Dynamic)** - dynamic imaging, reflects relative changes
   - **STAT (Static)** - static imaging, reflects absolute resistivity

**Feature Naming Convention:**

```
{Feature_Type}_{Statistical_Method}_{Image_Type}

Examples:
- CON_MEAN_DYNA  = Contrast_Mean_Dynamic
- ENT_X_DYNA     = Entropy_X-direction_Dynamic
- HOM_SUB_STAT   = Homogeneity_Sub-region_Static
```

**Total features:** 7 features × 4 statistical methods × 1 image type = **28 texture features**

---

### 2.2 AKBO - Auto-Kmeans with Bayesian Optimization

#### 2.2.1 Limitations of Traditional K-means

| Issue | Description | Impact |
|-------|-------------|--------|
| **K selection** | Requires pre-specifying the number of clusters | Relies on experience/trial-and-error |
| **Initial centroids** | Random initialization | Prone to local optima |
| **Subjective evaluation** | Elbow method/Silhouette coefficient requires manual judgment | Unstable results |

#### 2.2.2 Core Innovation of AKBO

**Formulating K selection and centroid initialization as a black-box optimization problem:**

```
max K∈[Kmin, Kmax]  UIndex(K)
```

Where UIndex is the composite clustering quality metric.

#### 2.2.3 Composite Clustering Quality Metric UIndex

**UIndex integrates three classical metrics:**

```
UIndex = SI × (1/DBI) × DVI
```

**1. Silhouette Index (SI)**

Measures the cohesion of a sample with its own cluster and its separation from neighboring clusters:

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))

SI = (1/N) × Σ s(i)
```

Where:
- `a(i)` = average distance from sample i to other samples in the same cluster
- `b(i)` = average distance from sample i to samples in the nearest neighboring cluster
- **SI ∈ [-1, 1]**, higher is better

**2. Davies-Bouldin Index (DBI)**

Measures the ratio of within-cluster distance to between-cluster distance:

```
DBI = (1/K) × Σ max(j≠i) [(d_i + d_j) / d(c_i, c_j)]
```

Where:
- `d_i` = average distance from samples in cluster i to the centroid
- `d(c_i, c_j)` = distance between centroids of cluster i and j
- **DBI ∈ [0, ∞)**, lower is better

**3. Dunn Index (DVI)**

Measures the ratio of minimum inter-cluster distance to maximum intra-cluster diameter:

```
DVI = min(i≠j) d(c_i, c_j) / max(k) diam(k)
```

Where:
- `diam(k)` = maximum diameter of cluster k
- **DVI ∈ [0, ∞)**, higher is better

**Advantages of UIndex:**
- SI focuses on per-sample clustering quality
- DBI focuses on inter-cluster separation (sensitive to noise)
- DVI focuses on cluster compactness (suitable for arbitrary shapes)
- The three are complementary, providing more comprehensive evaluation

#### 2.2.4 Bayesian Optimization Framework

**Gaussian Process (GP) surrogate model:**

```
UIndex ~ GP(μ(K), k(K, K'))
```

**Covariance function (RBF kernel):**

```
k(K, K') = σ² · exp(-||K - K'||² / (2l²))
```

**Acquisition function (Expected Improvement, EI):**

```
EI(K) = E[max(0, UIndex(K) - UIndex_best)]
```

Balances **exploration** and **exploitation**.

#### 2.2.5 AKBO Algorithm Flow

```
Algorithm: AKBO Unsupervised Clustering
Input:   data X, K search range [Kmin, Kmax], maximum iterations T
Output:  optimal K*, cluster labels L, centroids C

Step 1: Initialization
  1.1 Randomly select n_init K values ∈ [Kmin, Kmax]
  1.2 For each K:
      - Execute K-means++ initialization
      - Run K-means to convergence
      - Compute UIndex(K)
  1.3 Build initial dataset D = {(K_i, UIndex_i)}
  1.4 Initialize Gaussian Process model GP

Step 2: Bayesian Optimization Loop (t = 1 to T)
  2.1 Train GP model on D
  2.2 Select next K_t by maximizing the acquisition function:
      K_t = argmax_K EI(K; GP)
  2.3 Execute K-means++ clustering (K=K_t)
  2.4 Compute UIndex(K_t)
  2.5 Update dataset D = D ∪ {(K_t, UIndex_t)}
  2.6 Check convergence:
      - If UIndex shows no improvement for n_patience consecutive times, terminate

Step 3: Output Optimal Results
  3.1 K* = argmax_{K∈D} UIndex(K)
  3.2 Return corresponding cluster labels L and centroids C
```

#### 2.2.6 K-means++ Initialization

Compared to random initialization, K-means++ ensures well-spread initial centroids:

```
1. Randomly select the first centroid μ₁
2. For each sample x_i, compute D(x_i) = min_j ||x_i - μ_j||²
3. Select the next centroid with probability D(x_i)/ΣD(x)
4. Repeat until K centroids are selected
```

---

## 3. Technical Route

### 3.1 Overall Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    OGLCM-AKBO Workflow                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Data Loading and Preprocessing                      │
│  - Read CSV file (TZ1H_texture_logging.csv)                  │
│  - Separate depth column and feature columns                 │
│  - Data quality check (missing values, outliers)             │
│  - Feature standardization (Z-score)                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: OGLCM Texture Feature Analysis                      │
│  - Feature grouping (MEAN/SUB/X/Y)                          │
│  - Feature correlation analysis                              │
│  - Feature importance evaluation (optional)                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: AKBO Clustering Optimization                        │
│  - Define K search range [2, 10]                             │
│  - Initialize Gaussian Process model                         │
│  - Bayesian optimization loop:                               │
│    · Select candidate K values                                │
│    · Execute K-means++ clustering                             │
│    · Compute UIndex                                           │
│    · Update GP model                                          │
│  - Convergence check and optimal K selection                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Clustering Results Evaluation                        │
│  - Compute final UIndex and components (SI, DBI, DVI)         │
│  - Cluster distribution statistics                            │
│  - Feature distribution analysis                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Results Visualization and Export                     │
│  - Depth profile plot                                        │
│  - Feature distribution box plots                            │
│  - PCA dimensionality reduction scatter plot                  │
│  - Cluster centers radar chart                                │
│  - Export clustering results CSV                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow

```
Raw Data (CSV)
    │
    ▼
┌──────────────────┐
│  DataPreprocessor │
│  - load_data()    │
│  - preprocess()   │
└────────┬─────────┘
         │
         ▼
  Standardized Feature Matrix X (N×28)
         │
         ▼
┌──────────────────┐
│   AKBOClusterer  │
│  - optimize_K()  │
│  - fit()         │
└────────┬─────────┘
         │
         ▼
  Cluster Labels L (N×1)
         │
         ▼
┌──────────────────┐
│   Visualizer     │
│  - plot_all()    │
└────────┬─────────┘
         │
         ▼
  Visualization Figures + Results CSV
```

---

## 4. Data Structure

### 4.1 Input Data Format

**File:** `TZ1H_texture_logging.csv`

| Column | Type | Description |
|--------|------|-------------|
| DEPTH | float | Depth (meters) |
| {FEAT}_{STAT}_{TYPE} | float | Texture features (28 total) |

**Feature List:**

```python
features = [
    # MEAN group (7)
    'CON_MEAN_DYNA', 'DIS_MEAN_DYNA', 'HOM_MEAN_DYNA', 'ENG_MEAN_DYNA',
    'COR_MEAN_DYNA', 'ASM_MEAN_DYNA', 'ENT_MEAN_DYNA',

    # SUB group (7)
    'CON_SUB_DYNA', 'DIS_SUB_DYNA', 'HOM_SUB_DYNA', 'ENG_SUB_DYNA',
    'COR_SUB_DYNA', 'ASM_SUB_DYNA', 'ENT_SUB_DYNA',

    # X direction group (7)
    'CON_X_DYNA', 'DIS_X_DYNA', 'HOM_X_DYNA', 'ENG_X_DYNA',
    'COR_X_DYNA', 'ASM_X_DYNA', 'ENT_X_DYNA',

    # Y direction group (7)
    'CON_Y_DYNA', 'DIS_Y_DYNA', 'HOM_Y_DYNA', 'ENG_Y_DYNA',
    'COR_Y_DYNA', 'ASM_Y_DYNA', 'ENT_Y_DYNA',
]
```

### 4.2 Output Data Format

**File:** `results/clustering_results.csv`

| Column | Type | Description |
|--------|------|-------------|
| DEPTH | float | Depth |
| CLUSTER_LABEL | int | Cluster label (0, 1, ..., K-1) |
| CLUSTER_0_PROB | float | Probability of belonging to cluster 0 |
| CLUSTER_1_PROB | float | Probability of belonging to cluster 1 |
| ... | ... | ... |

**Metrics:** `results/clustering_metrics.json`

```json
{
  "optimal_k": 5,
  "n_samples": 8265,
  "n_features": 28,
  "metrics": {
    "uindex": 0.65,
    "silhouette": 0.52,
    "dbi": 0.85,
    "dvi": 1.47
  },
  "cluster_distribution": {
    "cluster_0": 1653,
    "cluster_1": 1821,
    ...
  }
}
```

---

## 5. Implementation Details

### 5.1 Key Parameter Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K_min` | 2 | Lower bound of K search |
| `K_max` | 10 | Upper bound of K search |
| `n_init` | 5 | Number of initial sampling points |
| `max_iter` | 30 | Maximum Bayesian optimization iterations |
| `n_patience` | 5 | Convergence patience count |
| `tol` | 1e-4 | Convergence threshold |
| `random_state` | 42 | Random seed |

### 5.2 Dependencies

```python
Core dependencies:
- numpy >= 1.20.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0

Visualization:
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
```

### 5.3 Performance Optimization

1. **Vectorized computation:** Use NumPy array operations instead of loops
2. **Parallel initialization:** K-means supports multi-initialization parallelization
3. **Incremental updates:** GP model supports incremental training
4. **Early stopping:** Terminate early once UIndex converges

---

## 6. Expected Results

### 6.1 Clustering Quality Metrics

| Metric | Excellent | Good | Acceptable |
|--------|-----------|------|------------|
| SI | >0.7 | >0.5 | >0.25 |
| DBI | <0.5 | <1.0 | <1.5 |
| DVI | >2.0 | >1.0 | >0.5 |

### 6.2 Geological Interpretation

Clustering results can correspond to different lithofacies types:
- **High-resistivity bedding facies** - developed lamellation, good reservoir potential
- **Massive high-resistivity facies** - dense intervals
- **Low-resistivity mudstone facies** - high clay content
- **Fractured facies** - developed vertical fractures
- **Transitional facies** - mixed characteristics

---

## 7. Correspondence with the Paper

| Paper Section | Corresponding Code Module | Description |
|---------------|--------------------------|-------------|
| Texture feature extraction | Completed (CSV data) | Texture features already extracted |
| GLCM principles | Section 2.1 of this document | Theoretical explanation |
| AKBO algorithm | `akbo_clustering.py` | Core implementation |
| UIndex definition | Formula (14) | Composite metric |
| Workflow | Section 3 | Technical route |
| Experimental results | `results/` | Output directory |

---

## 8. Future Extensions

1. **Feature selection:** Use PCA or feature importance to filter key features
2. **Anomaly detection:** Identify logging anomaly intervals
3. **Facies correlation:** Validate with neighboring wells/core data
4. **3D modeling:** Extend to multi-well joint interpretation

---

*Document Version: 1.0 | Last Updated: 2026-03-17*
