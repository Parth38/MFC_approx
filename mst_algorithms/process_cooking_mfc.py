import time
import numpy as np
from scipy.sparse import csr_matrix
from mst_algorithms import MFCApprox, JaccardWeight
from load_hyperedges import load_hyperedges, hyperedges_to_sparse_matrix, load_labels


def main():

    output_prims = 'cooking_10pct_prims_jaccard.txt'
    output_kruskals = 'cooking_10pct_kruskals_jaccard.txt'


    file_hyperedges = 'cooking-samples/hyperedge_sample_10.txt' # 10 % FILE
    file_labels = 'cooking-samples/hyperedge_label_sample_10.txt' # 10 % FILE
    #file_hyperedges = 'trivago-samples/hyperedges-sample5pct.txt' # 10 % FILE
    #file_labels = 'trivago-samples/node-labels-sample5pct.txt' # 10 % FILE
    # Load and process data
    print("Loading hyperedges...")
    hyperedges = load_hyperedges(file_hyperedges)
    print(f"Loaded {len(hyperedges)} hyperedges")
    
    print("Loading labels...")
    labels = load_labels(file_labels)
    print(f"Loaded {len(labels)} labels")
    

    labels = labels[:len(hyperedges)]
    number_of_original_partitions = len(set(labels))
    while len(labels) < len(hyperedges) :
        labels.append(number_of_original_partitions + 1)
 
    print(len(labels))
 

    print("Converting to sparse matrix...")
    sparse_matrix = hyperedges_to_sparse_matrix(hyperedges)
    print(f"Created sparse matrix with shape {sparse_matrix.shape}")
    
    # Run MFC-Approx algorithm
    print("\nRunning TRUE MST, OPTIMAL MFC AND APPROX MFC ...")


 
    
    
    mfc = MFCApprox(sparse_matrix, labels, JaccardWeight())
    #final_tree = mfc.runOptimal()


    ###TRUE MST
    str_output = 'TRUE MST'
    start_time = time.time()
    true_mst = mfc.compute_true_mst()
    
    end_time = time.time()
    print(f"\n{str_output} runtime: {end_time - start_time:.2f} seconds")
    total_weight = sum(weight for _, _, weight in true_mst)    # Calculate total weight
    print(f"Total weight of {str_output}: {total_weight:.4f}")


    ###OPTIMAL MFC
    #str_output = 'OPTIMAL MFC T*'
    #start_time = time.time()
    #optimal_mfc = mfc.runOptimal()
    
    #end_time = time.time()
   # print(f"\n{str_output} runtime: {end_time - start_time:.2f} seconds")
    #total_weight = sum(weight for _, _, weight in optimal_mfc)    # Calculate total weight
    #print(f"Total weight of {str_output}: {total_weight:.4f}")

    ###APPROX MFC
    str_output = 'APPROX MFC T^'
    start_time = time.time()
    approx_mfc = mfc.run()
    end_time = time.time()
    print(f"\n{str_output} runtime: {end_time - start_time:.2f} seconds")
    total_weight = sum(weight for _, _, weight in approx_mfc)    # Calculate total weight
    print(f"Total weight of {str_output}: {total_weight:.4f}")

if __name__ == "__main__":
    main() 