import time
import numpy as np
from scipy.sparse import csr_matrix
from mst_algorithms import MFCApprox, JaccardWeight

def load_hyperedges(file_path):
    """Load hyperedges from file and convert to sets."""
    hyperedges = []
    with open(file_path, 'r') as f:
        for line in f:
            # Convert each line to a set of integers
            nodes = set(map(int, line.strip().split(',')))
            hyperedges.append(nodes)
    return hyperedges

def load_labels(file_path):
    """Load node labels from file."""
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            labels.append(int(line.strip()))
    return labels

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

def main():

    
    output_prims = 'cooking_10pct_prims_jaccard.txt'
    output_kruskals = 'cooking_10pct_kruskals_jaccard.txt'
    file_hyperedges = 'cooking-samples/hyperedges-sample10pct.txt' # 10 % FILE
    file_labels = 'cooking-samples/node-labels-sample10pct.txt' # 10 % FILE
    #file_path = 'cooking-samples/hyperedges-sample20pct.txt' # 10 % FILE
    #file_path = 'cooking-samples/hyperedges-sample100pct.txt' # 10 % FILE

    # Load and process data
    print("Loading hyperedges...")
    hyperedges = load_hyperedges(file_hyperedges)
    print(f"Loaded {len(hyperedges)} hyperedges")
    
    print("Loading labels...")
    labels = load_labels(file_labels)
    print(f"Loaded {len(labels)} labels")
    
    print("Converting to sparse matrix...")
    sparse_matrix = hyperedges_to_sparse_matrix(hyperedges)
    print(f"Created sparse matrix with shape {sparse_matrix.shape}")
    
    # Run MFC-Approx algorithm
    print("\nRunning MFC-Approx algorithm...")
    start_time = time.time()
    
    mfc = MFCApprox(sparse_matrix, labels, JaccardWeight())
    final_tree = mfc.run()
    
    end_time = time.time()
    print(f"\nMFC-Approx runtime: {end_time - start_time:.2f} seconds")
    
    # Calculate total weight
    total_weight = sum(weight for _, _, weight in final_tree)
    print(f"Total weight of MFC-Approx tree: {total_weight:.4f}")

if __name__ == "__main__":
    main() 