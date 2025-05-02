import time
from mst_algorithms import MFCApprox, JaccardWeight
from load_files import load_hyperedges, hyperedges_to_sparse_matrix, load_node_labels

def main():

    # --------‑ paths -------------------------------------------------------
    file_hyperedges = 'cooking-samples/hyperedges-sample10pct.txt'
    file_labels     = 'cooking-samples/node-labels-sample10pct.txt'   # cuisine of each recipe
    # ----------------------------------------------------------------------

    # 1. load
    print("Loading hyperedges...")
    hyperedges = load_hyperedges(file_hyperedges)          # 0‑based
    print(f"Loaded {len(hyperedges)} hyperedges")

    print("Loading labels...")
    labels = load_node_labels(file_labels)                 # 0‑based, len == #hyperedges
    print(f"Loaded {len(labels)} labels")

    # sanity check
    assert len(labels) == len(hyperedges), (
        "Label file must have one line per hyper‑edge!"
    )

    # 2. sparse matrix (rows = nodes, cols = hyper‑edges)
    print("Converting to sparse matrix...")
    sparse_matrix = hyperedges_to_sparse_matrix(hyperedges)
    print(f"Created sparse matrix with shape {sparse_matrix.shape}")

    # 3. run the three trees
    print("\nRunning TRUE MST, OPTIMAL MFC AND APPROX MFC ...")

    mfc = MFCApprox(sparse_matrix, labels.tolist(), JaccardWeight())

    # TRUE MST
    t0 = time.time()
    true_mst = mfc.compute_true_mst()
    print(f"\nTRUE MST   runtime: {time.time() - t0:.2f}s, "
          f"weight: {sum(w for _,_,w in true_mst):.4f}")

    # OPTIMAL MFC
    t0 = time.time()
    opt_mfc = mfc.compute_optimal_mfc()
    print(f"OPTIMAL MFC runtime: {time.time() - t0:.2f}s, "
          f"weight: {sum(w for _,_,w in opt_mfc):.4f}")

    # APPROX MFC
    t0 = time.time()
    approx_mfc = mfc.compute_mfc_approx()
    print(f"APPROX MFC runtime: {time.time() - t0:.2f}s, "
          f"weight: {sum(w for _,_,w in approx_mfc):.4f}")

if __name__ == "__main__":
    main()
