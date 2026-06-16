# Graph Neural Network-Based Representation Learning for Multi-omics Data

A comprehensive method for representation learning using Graph Neural Networks (GNNs) to integrate spatial multi-omics data and identify spatial domains.

## Overview

This thesis project implements various GNN-based methods for analyzing spatial multi-omics datasets, enabling the discovery of biologically meaningful spatial domains through advanced representation learning techniques. The framework integrates multiple graph representation learning approaches including the proposed VGAE model, and 4 baselines: GAE, GATE, BGRL, DGI.

## Key Features

- **Multi-GNN Architecture**: Implementation of multiple graph neural network models (VGAE, GAE, GATE, BGRL, DGI)
- **Multi-omics Integration**: Integration of spatial transcriptomics and proteomics/epigenomics data
- **Spatial Domain Identification**: Advanced methods for discovering spatial domains in tissue samples
- **Flexible Preprocessing**: Comprehensive utilities for data normalization, graph construction, and feature engineering
- **Evaluation Metrics**: Built-in evaluation functions including ARI, NMI, and HOM

## Project Structure

```
thesis/
├── src/
│   ├── Model/
│   │   ├── BGRL.py          # Bootstrapped Graph Representation Learning
│   │   ├── DGI.py           # Deep Graph Infomax
│   │   ├── GAE.py           # Graph Autoencoder
│   │   ├── GATE.py          # Graph Attention-based autoencoder
│   │   └── VGAE.py          # Variational Graph Autoencoder
│   ├── utils.py             # Preprocessing and evaluation utilities
│   └── boxplot.py           # Visualization helper
├── notebook/
│   ├── all_models.ipynb     # Comprehensive model comparison notebook
│   └── test_data.ipynb      # Test data and quick start guide
├── dataset/                 # Multi-omics spatial datasets
├── figure/                  # Generated figures and visualizations
├── requirements.txt         # Python package dependencies
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9 or higher
- CUDA support (optional, for GPU acceleration)

### Setup Instructions

1. **Clone the repository**
```bash
   git clone https://github.com/watermelon-3012/thesis-gnn-multiomics
```

2. **Create a virtual environment (recommended)**

    2.1 Create conda environment and install package
```bash
    #Create an environment called SpatialGlue

    conda create -n thesis python=3.8

    #Activate your environment

    conda activate thesis
```
    2.2 Install dependencies
```bash
    pip install -r requirements.txt
```

    2.3 To use the environment in jupyter notebook, add python kernel for this environment.
```bash
    pip install ipykernel

    python -m ipykernel install --user --name=thesis  
```

### Dependencies

Key packages required:
- `torch` - Deep learning framework
- `torch_geometric` - Graph neural network library
- `scanpy` - Single-cell analysis in Python
- `anndata` - Annotated data structure
- `pandas` - Data manipulation
- `scipy` - Scientific computing
- `scikit-learn` - Machine learning utilities
- `numpy` - Numerical computing
- `matplotlib` - Visualization

See `requirements.txt` for complete dependency list.

## Usage

### Basic Workflow

1. Load and Prepare Data
2. Build Spatial Graph
3. Train a Model
4. Identify Spatial Domains by Clustering
5. Evaluate Results

### Comprehensive Model Comparison

To run all models and dataset and compare their performance:

```bash
jupyter notebook notebook/all_models.ipynb
```

This notebook demonstrates:
- Data loading and preprocessing
- Model initialization for all five GNN architectures
- Training procedures
- Performance comparison and visualization

## Available Models

### 1. **VGAE** - Variational Graph Autoencoder
Probabilistic graph autoencoder using variational inference for robust latent representations.

### 2. **GAE** - Graph Autoencoder
Standard graph autoencoder for unsupervised representation learning with deterministic encoder/decoder.

### 3. **GATE** - Graph Attention-based Autoencoder
Incorporates attention mechanisms to learn adaptive node importance in the neighborhood.

### 4. **BGRL** - Bootstrapped Graph Representation Learning
Self-supervised learning approach using bootstrapping for unsupervised graph representation learning.

### 5. **DGI** - Deep Graph Infomax
Maximizes mutual information between node representations and graph-level summary statistics.

## Reproducing Results

### Step-by-Step Reproduction

1. **Prepare Dataset**
   - Place your spatial multi-omics data in the `dataset/` folder (h5ad, csv, or h5 format)
   - Ensure data contains spatial coordinates and gene/protein expression matrices

2. **Run Analysis Pipeline**
```bash
   # Execute the complete analysis
   jupyter notebook notebook/all_models.ipynb
```

3. **Configuration**
   Edit parameters in the notebooks or create a config file

4. **Generate Results**
   - Model embeddings are saved automatically
   - Visualizations are stored in the `figure/` folder
   - Evaluation metrics are printed and can be exported
   
## Data Formats

Supported input formats:
- **H5AD**: AnnData format (recommended) - `scanpy.read_h5ad()`
- **H5**: HDF5 format for large datasets
- **CSV**: Comma-separated values with metadata

Expected data structure:
- Gene/protein/epigenome expression matrix (cells × genes/proteins/epigenome features)
- Spatial coordinates (x, y coordinates for each cell)
- Spatial domain labels (for evaluation)

## Evaluation Metrics

The method includes standard clustering evaluation metrics:
- **ARI** (Adjusted Rand Index): Measures agreement between predicted and true labels
- **NMI** (Normalized Mutual Information): Quantifies mutual dependence between label sets
- **HOM** (Homogeneity Score): Ensures all clusters contain only members of a single class

## Contributing

For questions or issues, please contact the thesis author or open an issue in the repository.