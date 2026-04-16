# OGLCM-AKBO

**Orientational Gray-Level Co-occurrence Matrix (OGLCM) + Auto-Kmeans with Bayesian Optimization (AKBO)**

An Automatic Shale Microfacies Characterization Method Based on Logging Image Texture Features

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version 2.1](https://img.shields.io/badge/version-2.1-yellow.svg)](https://github.com/bluemap19/OGLCM-AKBO)

---

## Project Overview

OGLCM-AKBO is an **unsupervised clustering algorithm** for automatic shale microfacies identification based on logging image texture features.

**Core Innovations:**

1. Extracts 28 texture features using OGLCM
2. Selects 4 optimal features based on Random Forest feature importance analysis
3. Automatically determines the optimal K value using Bayesian Optimization
4. Proposes the UIndex composite metric for clustering quality evaluation

**Implemented Features:**

1. PCA data dimensionality reduction
2. Constructs the KMeans parameter space
3. Uses Bayesian Optimization to automatically determine optimal KMeans parameter values

---

## Quick Start

### Environment Requirements

- Python 3.8+
- NumPy >= 1.20.0
- Pandas >= 1.3.0
- Scikit-learn >= 0.24.0
- Matplotlib >= 3.4.0

### Installation

```bash
# Clone the repository
git clone https://github.com/bluemap19/OGLCM-AKBO.git
cd OGLCM-AKBO

# Install dependencies
pip install -r requirements.txt
```

### Running the Example

```bash
# Run the main program
python main.py
```

**Input:** `TZ1H_texture_logging.csv` (logging texture feature data)

**Output:**
- `results/clustering_results.csv` - Clustering results
- `results/test_report.md` - Detailed test report (including complete iteration history)
- `results/figures/*.png` - Visualization figures

---

## Data Description

### Input Data Format

| Column | Description | Example |
|--------|-------------|---------|
| DEPTH | Depth (m) | 2192.21 |
| CON_SUB_DYNA | Contrast_sub-region_dynamic | 0.523 |
| DIS_SUB_DYNA | Dissimilarity_sub-region_dynamic | 0.412 |
| HOM_SUB_DYNA | Homogeneity_sub-region_dynamic | 0.678 |
| ENG_SUB_DYNA | Energy_sub-region_dynamic | 0.345 |
| ... | Other features (28 total) | ... |

### Optimal Features (4 features)

| Feature | Geological Significance |
|---------|------------------------|
| `CON_SUB_DYNA` | Contrast_sub-region_dynamic - Reflects local texture variation intensity |
| `DIS_SUB_DYNA` | Dissimilarity_sub-region_dynamic - Reflects sub-region grayscale difference |
| `HOM_SUB_DYNA` | Homogeneity_sub-region_dynamic - Reflects sub-region texture uniformity |
| `ENG_SUB_DYNA` | Energy_sub-region_dynamic - Reflects texture complexity |

---

## Core Algorithm

### UIndex Composite Metric

**Proposed composite clustering quality metric:**

```
UIndex(K) = 1 / (0.1/SI(K) + DBI(K)/1.0 + 0.01/DVI(K))
```

| Metric | Description | Characteristics |
|--------|-------------|-----------------|
| **SI** | Silhouette Index | Higher is better |
| **DBI** | Davies-Bouldin Index | Lower is better |
| **DVI** | Dunn Index | Higher is better |

**Quality Evaluation Criteria:**
- UIndex > 0.5: Excellent
- UIndex > 0.2: Good
- UIndex < 0.2: Needs improvement

### Bayesian Optimization Workflow

```
Initialization (5 points) -> Gaussian Process Modeling -> EI Acquisition Function -> Iteration (30 times) -> Optimal K value
```

---

## Project Structure

```
OGLCM-AKBO/
├── main.py                      # Main program
├── src/
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── akbo_clustering.py       # AKBO core algorithm
│   └── visualization.py         # Visualization module
├── tests/
│   └── test_akbo.py            # Unit tests
├── docs/
│   ├── feature_selection_guide.md      # Feature selection methods
│   ├── technical_background_and_algorithm_principles.md  # Algorithm principles
│   └── version_2.1_changelog.md              # Version changelog
├── results/                     # Output results (git ignored)
├── .gitignore                   # Git ignore file
├── requirements.txt             # Dependencies
├── LICENSE                      # License
└── README.md                    # Project documentation
```

---

## Test Results

### Sample Data (Well TZ1H)

| Metric | Value | Evaluation |
|--------|-------|------------|
| **Samples** | 8265 | - |
| **Features** | 4 | - |
| **Optimal K** | 5 | - |
| **UIndex** | 0.9828 | Excellent |
| **SI** | 0.4552 | Moderate |
| **DBI** | 0.7643 | Good |
| **DVI** | 0.2987 | Low |

### Cluster Distribution

| Cluster | Samples | Proportion |
|---------|---------|------------|
| 0 | 2629 | 31.8% |
| 1 | 951 | 11.5% |
| 2 | 1377 | 16.7% |
| 3 | 2543 | 30.8% |
| 4 | 765 | 9.3% |

---

## Documentation

- [Feature Selection Guide](docs/feature_selection_guide.md) - Feature selection methods and geological significance
- [Technical Background and Algorithm Principles](docs/technical_background_and_algorithm_principles.md) - Complete algorithm principles and formulas
- [Version 2.1 Changelog](docs/version_2.1_changelog.md) - Code change log

---

## Version History

### v2.1 (2026-03-27)
- Updated UIndex calculation formula
- Removed feature selection from the clusterer (already completed in preprocessing)
- Used preset 4 optimal features
- Added detailed iteration history to test report

### v2.0 (2026-03-15)
- Fixed EI acquisition function error
- Used GMM to compute true posterior probabilities
- Added Random Forest feature selection
- Added unit tests (15/15 passed)

---

## Authors

- **Fuhao Zhang**
  - GitHub: [@bluemap19](https://github.com/bluemap19)
  - Email: puremaple19@outlook.com
  - Research Area: Well Logging Engineering, Geological Engineering, Deep Learning

- **Cuka** (OpenClaw AI Assistant)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Thanks to all researchers and technical supporters who contributed to this project!

---

## Contact

For questions or suggestions, please contact:

- GitHub Issues: https://github.com/bluemap19/OGLCM-AKBO/issues
- Email: puremaple19@outlook.com

---

*Last Updated: 2026-04-14*
