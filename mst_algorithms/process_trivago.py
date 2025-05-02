import time
import numpy as np
from scipy.sparse import csr_matrix
from mst_algorithms import PrimsMST, KruskalsMST, JaccardWeight

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

def run_mst_algorithm(algorithm_class, sparse_matrix, weight_function):
    """Run MST algorithm and return results."""
    start_time = time.time()
    mst = algorithm_class(sparse_matrix, weight_function)
    edges = mst.find_mst()
    end_time = time.time()
    
    # Calculate total weight
    total_weight = sum(weight for _, _, weight in edges)
    
    return {
        'edges': edges,
        'total_weight': total_weight,
        'runtime': end_time - start_time
    }

def save_tree(edges, filename):
    """Save MST edges to file."""
    with open(filename, 'w') as f:
        for u, v, weight in edges:
            f.write(f"{u},{v},{weight}\n")

def main():
    # Load and process data
    print("Loading hyperedges...")
    hyperedges = load_hyperedges('trivago-samples/hyperedges-sample5pct.txt')
    print(f"Loaded {len(hyperedges)} hyperedges")
    
    print("Converting to sparse matrix...")
    sparse_matrix = hyperedges_to_sparse_matrix(hyperedges)
    print(f"Created sparse matrix with shape {sparse_matrix.shape}")
    
    # Run algorithms
    weight_function = JaccardWeight()
    
  #  print("\nRunning Prim's algorithm...")
  #  prims_result = run_mst_algorithm(PrimsMST, sparse_matrix, weight_function)
  #  print(f"Prim's runtime: {prims_result['runtime']:.2f} seconds")
  #  print(f"Prim's total weight: {prims_result['total_weight']:.4f}")
  #  save_tree(prims_result['edges'], 'trivago_prims_jaccard.txt')
    
    print("\nRunning Kruskal's algorithm...")
    kruskals_result = run_mst_algorithm(KruskalsMST, sparse_matrix, weight_function)
    print(f"Kruskal's runtime: {kruskals_result['runtime']:.2f} seconds")
    print(f"Kruskal's total weight: {kruskals_result['total_weight']:.4f}")
    save_tree(kruskals_result['edges'], 'trivago_kruskals_jaccard.txt')

if __name__ == "__main__":
    main() 