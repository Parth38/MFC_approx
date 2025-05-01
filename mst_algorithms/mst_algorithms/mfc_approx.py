import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict, Tuple, Set
from .kruskals import KruskalsMST
from .weight_functions import WeightFunction

class MFCApprox:
    """Implementation of the MFC-Approx algorithm."""
    
    def __init__(self, sparse_matrix: csr_matrix, labels: List[int], weight_function: WeightFunction):
        """Initialize MFC-Approx algorithm.
        
        Args:
            sparse_matrix: Input sparse matrix where each row is a point
            labels: List of labels for each row in the matrix
            weight_function: Weight function to compute distances
        """
        self.sparse_matrix = sparse_matrix
        self.labels = labels
        self.weight_function = weight_function
        self.n_points = sparse_matrix.shape[0]
        
        # Create partitions based on labels
        self.partitions: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            if label not in self.partitions:
                self.partitions[label] = []
            self.partitions[label].append(idx)
            
        self.partition_trees: Dict[int, List[Tuple[int, int, float]]] = {}
        self.coarsened_graph: Dict[Tuple[int, int], Tuple[float, Tuple[int, int]]] = {}
        
    def compute_partition_trees(self):
        """Compute MST for each partition using Kruskal's algorithm."""
        for label, indices in self.partitions.items():
            # Create submatrix for partition
            partition_matrix = self.sparse_matrix[indices]
            
            # Compute MST for partition
            mst = KruskalsMST(partition_matrix, self.weight_function)
            edges = mst.find_mst()
            
            # Convert edge indices back to original matrix indices
            original_edges = [
                (indices[u], indices[v], weight)
                for u, v, weight in edges
            ]
            
            self.partition_trees[label] = original_edges
            
            # Save partition tree
            self._save_tree(original_edges, f'partition_tree_{label}.txt')
            
    def compute_coarsened_graph(self):
        """Compute the coarsened graph between partitions."""
        # Get representatives for each partition
        representatives = {
            label: indices[0]  # Using first element as representative
            for label, indices in self.partitions.items()
        }
        
        # Compute inter-partition weights
        labels = sorted(self.partitions.keys())
        for i, label_i in enumerate(labels):
            for label_j in labels[i+1:]:
                # Find closest node in C_i to r_j
                a_ij = float('inf')
                a_ij_pair = None
                for u in self.partitions[label_i]:
                    dist = self.weight_function.compute(
                        self.sparse_matrix[u],
                        self.sparse_matrix[representatives[label_j]]
                    )
                    if dist < a_ij:
                        a_ij = dist
                        a_ij_pair = (u, representatives[label_j])
                
                # Find closest node in C_j to r_i
                b_ij = float('inf')
                b_ij_pair = None
                for v in self.partitions[label_j]:
                    dist = self.weight_function.compute(
                        self.sparse_matrix[representatives[label_i]],
                        self.sparse_matrix[v]
                    )
                    if dist < b_ij:
                        b_ij = dist
                        b_ij_pair = (representatives[label_i], v)
                
                # Store the minimum weight and corresponding edge
                if a_ij < b_ij:
                    self.coarsened_graph[(label_i, label_j)] = (a_ij, a_ij_pair)
                else:
                    self.coarsened_graph[(label_i, label_j)] = (b_ij, b_ij_pair)
        
        # Save coarsened graph
        self._save_coarsened_graph()
        
    def compute_final_tree(self) -> List[Tuple[int, int, float]]:
        """Compute the final MST using the coarsened graph."""
        # Create edges for the coarsened graph
        coarsened_edges = [
            (u, v, weight)
            for (_, _), (weight, (u, v)) in self.coarsened_graph.items()
        ]
        
        # Compute MST of coarsened graph
        coarsened_matrix = self._create_coarsened_matrix()
        mst = KruskalsMST(coarsened_matrix, self.weight_function)
        coarsened_mst = mst.find_mst()
        
        # Convert coarsened MST edges back to original indices
        final_edges = []
        for u, v, weight in coarsened_mst:
            # Get the actual edge that realizes this weight
            for (label_i, label_j), (w, (actual_u, actual_v)) in self.coarsened_graph.items():
                if w == weight:
                    final_edges.append((actual_u, actual_v, weight))
                    break
        
        # Combine partition trees and coarsened MST edges
        all_edges = []
        for tree in self.partition_trees.values():
            all_edges.extend(tree)
        all_edges.extend(final_edges)
        
        # Save final tree
        self._save_tree(all_edges, 'mfc_approx_final_tree.txt')
        
        # Calculate total weight
        total_weight = sum(weight for _, _, weight in all_edges)
        print(f"Total weight of MFC-Approx tree: {total_weight:.4f}")
        
        return all_edges
    
    def _create_coarsened_matrix(self) -> csr_matrix:
        """Create sparse matrix representation of coarsened graph."""
        n_partitions = len(self.partitions)
        data = []
        row_ind = []
        col_ind = []
        
        for (i, j), (weight, _) in self.coarsened_graph.items():
            row_ind.append(i)
            col_ind.append(j)
            data.append(weight)
            
        return csr_matrix((data, (row_ind, col_ind)), shape=(n_partitions, n_partitions))
    
    def _save_tree(self, edges: List[Tuple[int, int, float]], filename: str):
        """Save tree edges to file."""
        with open(filename, 'w') as f:
            for u, v, weight in edges:
                f.write(f"{u},{v},{weight}\n")
    
    def _save_coarsened_graph(self):
        """Save coarsened graph to file."""
        with open('coarsened_graph.txt', 'w') as f:
            for (i, j), (weight, (u, v)) in self.coarsened_graph.items():
                f.write(f"{i},{j},{weight},{u},{v}\n")
    
    def run(self) -> List[Tuple[int, int, float]]:
        """Run the complete MFC-Approx algorithm."""
        print("Computing partition trees...")
        self.compute_partition_trees()
        
        print("Computing coarsened graph...")
        self.compute_coarsened_graph()
        
        print("Computing final tree...")
        return self.compute_final_tree() 