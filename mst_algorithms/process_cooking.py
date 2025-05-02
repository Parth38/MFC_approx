import time

from mst_algorithms import PrimsMST, KruskalsMST, JaccardWeight
from load_hyperedges import load_hyperedges, hyperedges_to_sparse_matrix


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


    output_prims = 'cooking_10pct_prims_jaccard.txt'
    output_kruskals = 'cooking_10pct_kruskals_jaccard.txt'
    file_path = 'cooking-samples/hyperedges-sample10pct.txt' # 10 % FILE
    #file_path = 'cooking-samples/hyperedges-sample20pct.txt' # 10 % FILE
    #file_path = 'cooking-samples/hyperedges-sample100pct.txt' # 10 % FILE

    print("Loading hyperedges...")
    hyperedges = load_hyperedges(file_path)
    print(f"Loaded {len(hyperedges)} hyperedges")
    
    print("Converting to sparse matrix...")
    sparse_matrix = hyperedges_to_sparse_matrix(hyperedges)
    print(f"Created sparse matrix with shape {sparse_matrix.shape}")
    
    # Run algorithms
    weight_function = JaccardWeight()
    
    print("\nRunning Prim's algorithm...")
    prims_result = run_mst_algorithm(PrimsMST, sparse_matrix, weight_function)
    print(f"Prim's runtime: {prims_result['runtime']:.2f} seconds")
    print(f"Prim's total weight: {prims_result['total_weight']:.4f}")
    save_tree(prims_result['edges'], output_prims)
    
    print("\nRunning Kruskal's algorithm...")
    kruskals_result = run_mst_algorithm(KruskalsMST, sparse_matrix, weight_function)
    print(f"Kruskal's runtime: {kruskals_result['runtime']:.2f} seconds")
    print(f"Kruskal's total weight: {kruskals_result['total_weight']:.4f}")
    save_tree(kruskals_result['edges'], output_kruskals)

if __name__ == "__main__":
    main() 