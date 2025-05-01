from typing import List, Tuple
from scipy.sparse import csr_matrix
from .weight_functions import WeightFunction

class MST:
    """Base class for Minimum Spanning Tree algorithms."""
    
    def __init__(self, sparse_matrix: csr_matrix, weight_function: WeightFunction):
        """Initialize the MST algorithm.
        
        Args:
            sparse_matrix: Input sparse matrix where each row is a point
            weight_function: Weight function to compute distances between points
        """
        self.sparse_matrix = sparse_matrix
        self.weight_function = weight_function
        self.n_points = sparse_matrix.shape[0]
        
    def find_mst(self) -> List[Tuple[int, int, float]]:
        """Find the Minimum Spanning Tree.
        
        Returns:
            List of edges in the MST, where each edge is represented as
            (vertex1, vertex2, weight)
        """
        raise NotImplementedError("Subclasses must implement find_mst()")
        
    def _get_edge_weight(self, i: int, j: int) -> float:
        """Compute the weight of an edge between two points.
        
        Args:
            i: Index of first point
            j: Index of second point
            
        Returns:
            float: Weight of the edge
        """
        return self.weight_function.compute(
            self.sparse_matrix[i],
            self.sparse_matrix[j]
        )
