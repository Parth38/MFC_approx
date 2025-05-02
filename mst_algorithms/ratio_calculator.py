import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict, Tuple, Set
import os
import glob

#CONSTANTS

PARTITIONS_FOLDER = './cooking-data/partitions/'                            # FOLDER CONTAINING THE PARTITIONS OF THE INITIAL FOREST (Partition per file)
#MFC_APPROX_TREE_FILE =  './trivago-data/mfc_approx_final_tree.txt'          # MFC-APPROX TREE
#OPTIMAL_MST_FILE = './trivago-data/true_mst.txt'     # OPTIMAL MST    
MFC_APPROX_TREE_FILE =  './cooking-data/mfc_approx_final_tree.txt'          # MFC-APPROX TREE
OPTIMAL_MST_FILE = './cooking-data/true_mst.txt'     # OPTIMAL MST    

def calculate_gamma(
        partitions: Dict[int, List[Tuple[int, int, float]]],
        optimal_mst: List[Tuple[int, int, float]],
        initial_forest: List[Tuple[int, int, float]]) -> float:
    """
    γ =   Σ_{(u,v,w) ∈ F  :  pid(u)=pid(v)}  w
        -------------------------------------------------
          Σ_{(u,v,w) ∈ T* :  pid(u)=pid(v)}  w

    • partitions : {pid: [(u,v,w), …]}   intra‑partition trees (F)
    • optimal_mst: [(u,v,w), …]          global MST   (T*)
    • initial_forest should be the union of all partition trees.

    Any edge whose endpoints belong to *different* partitions is ignored
    in both numerator and denominator.
    """

    # ------------------------------------------------------------------ A. node → partition id
    node2pid: Dict[int, int] = {}
    for pid, edges in partitions.items():
        for u, v, _ in edges:
            node2pid[u] = pid
            node2pid[v] = pid

    # ------------------------------------------------------------------ B. numerator  (inside‑P edges of F)
    num = 0.0
    for u, v, w in initial_forest:
        pid = node2pid.get(u)
        if pid is not None and pid == node2pid.get(v):
            num += w                                            # same partition

    # ------------------------------------------------------------------ C. denominator  (inside‑P edges of T*)
    den = 0.0
    for u, v, w in optimal_mst:
        pid = node2pid.get(u)
        if pid is not None and pid == node2pid.get(v):
            den += w

    if den == 0:
        raise ValueError("Denominator in γ is zero – check partitions!")

    return num / den

def calculate_beta(mfc_mst: List[Tuple[int, int, float]], optimal_mst: List[Tuple[int, int, float]]) -> float:
    """Calculate Beta value for the MFC-Approx algorithm.                       
    
    Args:
        mfc_mst: List of edges in the MFC-Approx MST, each edge is a tuple (u, v, w)
        optimal_mst: List of edges in the optimal MST, each edge is a tuple (u, v, w)
        
    Returns:
        float: Beta value
    """         

    weight_of_mfc_mst = sum(w for _, _, w in mfc_mst)
    weight_of_optimal_mst = sum(w for _, _, w in optimal_mst)

    return weight_of_mfc_mst / weight_of_optimal_mst    


def load_partitions(folder_path: str) -> Dict[int, List[Tuple[int, int, float]]]:
    """Load partitions from files in the specified folder.
    
    Args:
        folder_path: Path to the folder containing partition files
        
    Returns:
        Dict[int, List[Tuple[int, int, float]]]: Dictionary mapping partition IDs to lists of edges
        where each edge is a tuple (node1, node2, weight)
    """
    partitions = {}
    
    # Get all partition files in the folder
    partition_files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    for file_path in partition_files:
        # Extract partition ID from filename (assuming format like "partition_tree_1.txt")
        filename = os.path.basename(file_path)
        try:
            # Try to extract the number from the filename
            partition_id = int(''.join(filter(str.isdigit, filename)))
        except ValueError:
            # If no number found, use the index in the list
            partition_id = len(partitions)
 
        # Store the partition's edges
        partitions[partition_id] = load_mst_file(file_path)
    
    return partitions

def load_mst_file(file_path: str) -> List[Tuple[int, int, float]]:
    """Load an MST
    Args:
        file_path: Path to the file containing an MST (node1, node2, weight)
        
    Returns:
        List[Tuple[int, int, float]]: List of edges in the mst 
    """
    edges = []
    with open(file_path, 'r') as f:
        for line in f:
            # Split the line into node1, node2, weight
            parts = line.strip().split(',')
            if len(parts) >= 3:  # Ensure we have all three components
                node1 = int(parts[0])
                node2 = int(parts[1])
                weight = float(parts[2])
                edges.append((node1, node2, weight))
    return edges

def is_connected(edges: List[Tuple[int, int, float]]) -> bool:
    """Check if a graph represented by edges is connected.
    
    Args:
        edges: List of edges, where each edge is a tuple (u, v, w)
        
    Returns:
        bool: True if the graph is connected, False otherwise
    """
    if not edges:
        return False
    
    # Get all unique nodes
    nodes = set()
    for u, v, _ in edges:
        nodes.add(u)
        nodes.add(v)
    
    if not nodes:
        return False
    
    # Start BFS from the first node
    start_node = next(iter(nodes))
    visited = {start_node}
    queue = [start_node]
    
    # Create adjacency list
    adj = {node: set() for node in nodes}
    for u, v, _ in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Perform BFS
    while queue:
        current = queue.pop(0)
        for neighbor in adj[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    # Graph is connected if all nodes were visited
    return len(visited) == len(nodes)


def gamma_calculator(optimal_mst: List[Tuple[int, int, float]], partitions: Dict[int, List[Tuple[int, int, float]]]) -> float:
    summ_of_weights_in_partitions = 0
    summ_of_weights_in_optimal_mst = 0

    # Iterate through each partition's edges
    partition_nodes = {}
    for key, partition_edges in partitions.items():
        # Create a set of nodes in this partition
        pNodes = {node for edge in partition_edges for node in edge[:2]}
        partition_nodes[key] = pNodes
        
        for edge in partition_edges:
            summ_of_weights_in_partitions += edge[2]

    for edge in optimal_mst:
        # Check if both nodes are in the same partition
        for partition_set in partition_nodes.values():
            if edge[0] in partition_set and edge[1] in partition_set:
                summ_of_weights_in_optimal_mst += edge[2]
                break

    if summ_of_weights_in_optimal_mst == 0:
        return float('inf')
    return summ_of_weights_in_partitions / summ_of_weights_in_optimal_mst

        
def main():
    # Load partitions
    partitions = load_partitions(PARTITIONS_FOLDER)
    
    print(f"Loaded {len(partitions)} partitions")
    
    # Load MFC-Approx tree
    mfc_approx_mst = load_mst_file(MFC_APPROX_TREE_FILE)
    print(f"Loaded MFC-Approx tree with {len(mfc_approx_mst)} edges")
    
    # Check if MFC-Approx tree is connected
    is_mfc_connected = is_connected(mfc_approx_mst)
    print(f"MFC-Approx tree is {'connected' if is_mfc_connected else 'disconnected'}")
    
    # Load optimal MST
    optimal_mst = load_mst_file(OPTIMAL_MST_FILE)
    print(f"Loaded optimal MST with {len(optimal_mst)} edges")
    
    # Check if optimal MST is connected
    is_optimal_connected = is_connected(optimal_mst)
    print(f"Optimal MST is {'connected' if is_optimal_connected else 'disconnected'}")
    
    # Example usage
    for partition_id, edges in partitions.items():
        print(f"Partition {partition_id}: {len(edges)} edges")
    
    #calculate beta
    beta = calculate_beta(mfc_approx_mst, optimal_mst)
    print(f"Beta: {beta}")
    
    #calculate gamma
    initial_forest = [e for edges in partitions.values() for e in edges]
    gamma = gamma_calculator(optimal_mst, partitions)
   
    print(f"Gamma: {gamma}")

if __name__ == "__main__":
    main()