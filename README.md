# Graph Neural Network-Based Representation Learning for Spatial Multi-omics Data

A Graph Neural Network (GNN) framework for integrating spatial multi-omics data and identifying biologically meaningful spatial domains through representation learning.

## Introduction

Spatial multi-omics technologies enable simultaneous measurement of multiple molecular modalities, such as transcriptomics, proteomics, and epigenomics, while preserving spatial information within tissues. Integrating these heterogeneous data sources remains a challenging task due to their high dimensionality and complex relationships.

This project investigates Graph Neural Network (GNN)-based representation learning methods for spatial multi-omics integration and spatial domain identification. The framework includes a proposed Variational Graph Autoencoder (VGAE) model and four baseline methods:

* VGAE (Variational Graph Autoencoder)
* GAE (Graph Autoencoder)
* GATE (Graph Attention Autoencoder)
* BGRL (Bootstrapped Graph Representation Learning)
* DGI (Deep Graph Infomax)

The learned latent representations are used for clustering and spatial domain discovery in tissue samples.

---

## Features

* Multiple GNN architectures implemented within a unified framework
* Integration of spatial transcriptomics with proteomics or epigenomics data
* Spatial graph construction and representation learning
* Spatial domain identification through clustering
* Comprehensive preprocessing utilities
* Built-in evaluation metrics:

  * Adjusted Rand Index (ARI)
  * Normalized Mutual Information (NMI)
  * Homogeneity Score (HOM)
* Visualization and model comparison tools

---

## Repository Structure

```text
thesis/
├── src/
│   ├── Model/
│   │   ├── BGRL.py
│   │   ├── DGI.py
│   │   ├── GAE.py
│   │   ├── GATE.py
│   │   └── VGAE.py
│   ├── utils.py
│   └── boxplot.py
│
├── notebook/
│   ├── all_models.ipynb
│   └── test_data.ipynb
│
├── dataset/
├── figure/
├── requirements.txt
└── README.md
```

### Directory Description

| Directory/File              | Description                                                      |
| --------------------------- | ---------------------------------------------------------------- |
| `src/Model/`                | Implementation of all GNN models                                 |
| `src/utils.py`              | Data preprocessing, graph construction, and evaluation utilities |
| `src/boxplot.py`            | Visualization helper functions                                   |
| `notebook/all_models.ipynb` | Complete experimental pipeline and model comparison              |
| `notebook/test_data.ipynb`  | Quick-start notebook for testing                                 |
| `dataset/`                  | Spatial multi-omics datasets                                     |
| `figure/`                   | Generated figures and visualizations                             |

---

## Installation

### Prerequisites

* Python 3.8+
* PyTorch 1.9+
* CUDA-enabled GPU (optional)

### Clone the Repository

```bash
git clone https://github.com/watermelon-3012/thesis-gnn-multiomics.git
cd thesis-gnn-multiomics
```

### Create a Conda Environment

```bash
conda create -n thesis python=3.8
conda activate thesis
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Register Jupyter Kernel

```bash
pip install ipykernel

python -m ipykernel install \
    --user \
    --name thesis \
    --display-name "Python (thesis)"
```

---

## Required Dependencies

Major packages used in this project include:

* `torch`
* `torch_geometric`
* `scanpy`
* `anndata`
* `numpy`
* `pandas`
* `scipy`
* `scikit-learn`
* `matplotlib`

See `requirements.txt` for the complete dependency list.

---

## Data Requirements

Supported input formats:

* H5AD (`.h5ad`)
* HDF5 (`.h5`)
* CSV (`.csv`)

Each dataset should contain:

1. Expression matrix (cells × features)
2. Spatial coordinates (x, y)
3. Multi-omics modalities (e.g., gene expression and protein abundance)
4. Ground-truth spatial domain labels (optional, for evaluation)

---

## Workflow

The typical analysis pipeline consists of the following steps:

1. Load spatial multi-omics data
2. Preprocess and normalize features
3. Construct spatial neighborhood graphs
4. Train a GNN model
5. Learn latent node representations
6. Cluster embeddings to identify spatial domains
7. Evaluate clustering performance
8. Visualize spatial domains and embeddings

---

## Available Models

### VGAE – Variational Graph Autoencoder

A probabilistic graph autoencoder that uses variational inference to learn robust latent representations.

### GAE – Graph Autoencoder

A deterministic graph autoencoder for unsupervised node representation learning.

### GATE – Graph Attention Autoencoder

Uses graph attention mechanisms to adaptively weight neighboring nodes during representation learning.

### BGRL – Bootstrapped Graph Representation Learning

A self-supervised method that learns node embeddings through bootstrapping without negative samples.

### DGI – Deep Graph Infomax

Learns representations by maximizing mutual information between local node embeddings and global graph summaries.

---

## Running Experiments

### Quick Start

Launch the main notebook:

```bash
jupyter notebook notebook/all_models.ipynb
```

The notebook performs:

* Data loading
* Feature preprocessing
* Graph construction
* Model training
* Embedding extraction
* Clustering
* Performance evaluation
* Result visualization

---

## Reproducing Experimental Results

### Step 1: Prepare Data

Place datasets in the `dataset/` directory.

Example:

```text
dataset/
├── sample_1.h5ad
├── sample_2.h5ad
└── ...
```

### Step 2: Configure Parameters

Modify hyperparameters directly in:

```text
notebook/all_models.ipynb
```

Examples include:

* Learning rate
* Hidden dimensions
* Number of epochs
* Graph construction settings
* Clustering parameters

### Step 3: Run the Pipeline

```bash
jupyter notebook notebook/all_models.ipynb
```

### Step 4: Collect Outputs

Generated outputs include:

* Learned embeddings
* Spatial domain predictions
* Evaluation metrics
* Visualization figures

Results are automatically saved to the corresponding output directories.

---

## Evaluation Metrics

### Adjusted Rand Index (ARI)

Measures agreement between predicted and true labels while correcting for random chance.

### Normalized Mutual Information (NMI)

Measures the mutual dependence between predicted clusters and ground-truth labels.

### Homogeneity Score (HOM)

Measures whether each cluster contains only samples from a single class.

---

## Contact

For questions, suggestions, or bug reports, please open an issue in the repository or contact the author.

GitHub Repository:

https://github.com/watermelon-3012/thesis-gnn-multiomics
