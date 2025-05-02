import numpy as np
import networkx as nx
from collections import defaultdict, Counter
from itertools import combinations
from scipy.sparse import csr_matrix
import random  # Add import for random selection

# Load label names

LABEL_FILE = './cooking-samples/label-names-sample10pct.txt'
NODE_LABEL_FILE = './cooking-samples/node-labels-sample10pct.txt'
HYPEREDGES_FILE = './cooking-samples/hyperedges-sample10pct.txt'


with open(LABEL_FILE) as f:
    label_names = [line.strip() for line in f]

# Load node labels
with open(NODE_LABEL_FILE) as f:
    node_labels = [int(line.strip()) for line in f]

# Load hyperedges
def load_hyperedges(file_path):
    """Load hyperedges from file and convert to sets."""
    hyperedges = []
    with open(file_path, 'r') as f:
        for line in f:
            # Split by comma and convert to integers
            nodes = list(map(int, line.strip().split(',')))
            hyperedges.append(nodes)
    return hyperedges

hyperedges = load_hyperedges(HYPEREDGES_FILE)


# Build co-occurrence counter and neighbors map
cooccur = Counter()
node_neighbors = defaultdict(set)
for hedge in hyperedges:
    for u, v in combinations(hedge, 2):
        if u != v:
            cooccur[(u, v)] += 1
            cooccur[(v, u)] += 1
            node_neighbors[u].add(v)
            node_neighbors[v].add(u)

def jaccard(u, v):
    A, B = node_neighbors[u], node_neighbors[v]
    return len(A & B) / len(A | B) if A | B else 0

# Partition nodes by labels
partitions = defaultdict(list)
for idx, label in enumerate(node_labels):
    partitions[label].append(idx)

# Build local MSTs within partitions
local_trees = {}
for label, nodes in partitions.items():
    G = nx.Graph()
    for u, v in combinations(nodes, 2):
        if (u, v) in cooccur:
            weight = 1 - jaccard(u, v)
            G.add_edge(u, v, weight=weight)
    if G.number_of_edges() > 0:
        mst = nx.minimum_spanning_tree(G, weight='weight')
        local_trees[label] = mst

        # Save local MST to file
        with open(f'partition_tree_{label}.txt', 'w') as f:
            for u, v, d in mst.edges(data=True):
                f.write(f"{u},{v},{d['weight']:.6f}\n")

# Choose representative randomly for each partition
representatives = {label: random.choice(nodes) for label, nodes in partitions.items()}

# Compute approximate inter-partition distances
inter_partition_edges = []
labels = list(partitions.keys())
for i, label_i in enumerate(labels):
    for j in range(i + 1, len(labels)):
        label_j = labels[j]
        si, sj = representatives[label_i], representatives[label_j]
        
        dists = []
        if any((si, v) in cooccur for v in partitions[label_j]):
            dist_i_to_j = min(1 - jaccard(si, v) for v in partitions[label_j] if (si, v) in cooccur)
            dists.append(dist_i_to_j)
        if any((sj, u) in cooccur for u in partitions[label_i]):
            dist_j_to_i = min(1 - jaccard(sj, u) for u in partitions[label_i] if (sj, u) in cooccur)
            dists.append(dist_j_to_i)

        if dists:
            wij = min(dists)
            inter_partition_edges.append((label_i, label_j, wij))

# Build coarsened graph
GP = nx.Graph()
for u, v, w in inter_partition_edges:
    GP.add_edge(u, v, weight=w)

coarsened_mst = nx.minimum_spanning_tree(GP, weight='weight')

# Combine all local MSTs and inter-partition edges into global tree
global_tree = nx.Graph()
for tree in local_trees.values():
    global_tree.add_edges_from(tree.edges(data=True))

for u, v in coarsened_mst.edges():
    rep_u = representatives[u]
    rep_v = representatives[v]
    weight = next(w for x, y, w in inter_partition_edges if {x, y} == {u, v})
    global_tree.add_edge(rep_u, rep_v, weight=weight)

# Save final global tree to file
with open('mfc_approx_final_tree.txt', 'w') as f:
    for u, v, d in global_tree.edges(data=True):
        f.write(f"{u},{v},{d['weight']:.6f}\n")

# Report total cost of the global tree
total_cost = sum(d['weight'] for _, _, d in global_tree.edges(data=True))
print(f"Total cost of approximate MST: {total_cost:.4f}")
print(f"Total number of nodes: {global_tree.number_of_nodes()}")
print(f"Total number of edges: {global_tree.number_of_edges()}")

def hyperedges_to_sparse_matrix(hyperedges):
    """Convert hyperedges to sparse matrix representation.
    
    Args:
        hyperedges: List of hyperedges, where each hyperedge is a list of node IDs
        
    Returns:
        csr_matrix: Sparse matrix where:
            - Rows represent hyperedges
            - Columns represent nodes
            - Value 1 indicates node is in hyperedge
    """
    # Get all unique nodes
    all_nodes = set()
    for edge in hyperedges:
        all_nodes.update(edge)
    nodes = sorted(list(all_nodes))
    
    # Create mapping from node IDs to matrix indices
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # Create sparse matrix
    n_nodes = len(nodes)
    n_hyperedges = len(hyperedges)
    data = []
    row_ind = []
    col_ind = []
    
    # For each hyperedge and each node in it, add a 1 to the matrix
    for edge_idx, edge in enumerate(hyperedges):
        for node in edge:
            row_ind.append(edge_idx)
            col_ind.append(node_to_idx[node])
            data.append(1)
    
    # Create and return the sparse matrix
    return csr_matrix((data, (row_ind, col_ind)), shape=(n_hyperedges, n_nodes))
