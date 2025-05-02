import numpy as np
 
from typing import List
 

 
# ---------------------------------------------------------------------------
# 1) hyper‑edges  (comma‑separated integers, 1‑based in the file)
#    returns  List[List[int]]  of 0‑based node IDs
# ---------------------------------------------------------------------------
def load_hyperedges(path: str, zero_index: bool = True) -> List[List[int]]:
    edges: List[List[int]] = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                row = [int(x) - (1 if zero_index else 0) for x in ln.split(",")]
                edges.append(row)
    return edges

# ---------------------------------------------------------------------------
# 2) node‑labels (one integer per line, 1‑based in the file)
#    returns  np.ndarray[int]  of 0‑based labels
# ---------------------------------------------------------------------------
def load_node_labels(path: str, zero_index: bool = True) -> np.ndarray:
    lbls = np.loadtxt(path, dtype=np.int32)
    if zero_index:
        lbls -= 1
    return lbls

# ---------------------------------------------------------------------------
# 3) helper – build CSR incidence matrix  (optional, same as before)
# ---------------------------------------------------------------------------
from scipy.sparse import coo_matrix, csr_matrix
def hyperedges_to_sparse_matrix(edges: List[List[int]]) -> csr_matrix:
    rows, cols, data = [], [], []
    for e_id, verts in enumerate(edges):
        rows.extend(verts)
        cols.extend([e_id] * len(verts))
        data.extend([1] * len(verts))
    m = max(max(row) for row in edges) + 1     # node count
    n = len(edges)                             # edge count
    return coo_matrix((data, (rows, cols)), shape=(m, n)).tocsr()
