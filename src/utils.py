import os
import torch
import pandas as pd
import scanpy as sc
import skmisc
import numpy as np
import torch_geometric
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from scipy.spatial import cKDTree
from scipy.stats import mode
from collections import Counter
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score
from torch.backends import cudnn
import random

def clr_normalize_each_cell(adata, inplace=True):
    
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()
    
    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata 

def pca(adata, use_reps=None, n_comps=10):
    
    """Dimension reduction with PCA algorithm"""
    
    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps)
    if use_reps is not None:
       feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else: 
       if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
          feat_pca = pca.fit_transform(adata.X.toarray()) 
       else:   
          feat_pca = pca.fit_transform(adata.X)
    
    return feat_pca

def cKD_refine_label(coords, labels, k):
    # Step 1: Build KD-Tree
    tree = cKDTree(coords.copy())
    # Step 2: Find k-nearest neighbors for each spot
    # k+1 because the closest point is itself
    distances, neighbors = tree.query(coords, k=k+1)
    # Exclude self-neighbor (first column)
    neighbors = neighbors[:, 1:]
    # Step 3: Reassign labels
    new_labels = labels.copy()
    for i, nbrs in enumerate(neighbors):
        # Get the labels of neighboring spots
        neighbor_labels = labels[nbrs]
        # Find the most common label among neighbors
        # most_common_label = mode(neighbor_labels).mode[0]
        most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
        # Reassign the label
        new_labels[i] = most_common_label
    return (new_labels)

def buildGraph(adata_omics1, adata_omics2, k = 6):

    """Build graph by spatial coordinate with K-NN"""

    coords = adata_omics1.obsm['spatial'] #Extract spatial coordinate
    features_omics1 = adata_omics1.obsm['feat'] #Extract features of omics 1
    features_omics2 = adata_omics2.obsm['feat'] #Extract features of omics 2

    #Fit KNN model using spatial coordinates
    #Each points will connect to its k nearest neighbors
    knn = NearestNeighbors(n_neighbors = k).fit(coords)

    #Build spares adjacency matrix
    edge_index_np = knn.kneighbors_graph(coords, mode = 'connectivity').tocoo()

    #Convert to pytorch edge_index and make the graph undirected
    edge_index = torch.tensor(np.vstack((
        np.concatenate([edge_index_np.row, edge_index_np.col]),
        np.concatenate([edge_index_np.col, edge_index_np.row]))),
        dtype = torch.long)
        
    #Convert features to tensor
    X_omics1 = torch.tensor(features_omics1, dtype = torch.float)
    X_omics2 = torch.tensor(features_omics2, dtype = torch.float)

    #Create data object
    data = Data()
    data.num_nodes = X_omics1.shape[0]
    data.edge_index = edge_index
    data.x_omics1 = X_omics1   
    data.x_omics2 = X_omics2 

    return data

def compute_metrics(labels_true, labels_pred):
    """
    labels_true: ground truth
    labels_pred: prediction
    """

    ARI = adjusted_rand_score(labels_true, labels_pred)
    NMI = normalized_mutual_info_score(labels_true, labels_pred)
    HOM = homogeneity_score(labels_true, labels_pred)

    return {"ARI": ARI, "NMI": NMI, "HOM": HOM}

def fix_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'    