import numpy as np
from scipy.sparse import csr_matrix

def load_hyperedges(file_path):
    """Load hyperedges from file and convert to sets."""
    hyperedges = []
    with open(file_path, 'r') as f:
        for line in f:
            # Convert each line to a set of integers
            nodes = set(map(int, line.strip().split(',')))
            hyperedges.append(nodes)
    return hyperedges

def hyperedges_to_sparse_matrix(hyperedges):
    """Convert hyperedges to sparse matrix representation."""
    # Get all unique nodes
    all_nodes = set()
    for edge in hyperedges:
        all_nodes.update(edge)
    nodes = sorted(list(all_nodes))
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # Create sparse matrix
    n_nodes = len(nodes)
    n_hyperedges = len(hyperedges)
    data = []
    row_ind = []
    col_ind = []
    
    for edge_idx, edge in enumerate(hyperedges):
        for node in edge:
            row_ind.append(edge_idx)
            col_ind.append(node_to_idx[node])
            data.append(1)
    
    return csr_matrix((data, (row_ind, col_ind)), shape=(n_hyperedges, n_nodes))

def load_labels(file_path):
    """Load node labels from file."""
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            labels.append(int(line.strip()))
    return labels