import heapq
from typing import List, Tuple, Set
from scipy.sparse import csr_matrix
from .mst import MST

class PrimsMST(MST):
    """Implementation of Prim's algorithm for finding Minimum Spanning Tree."""
    
    def find_mst(self) -> List[Tuple[int, int, float]]:
        """Find MST using Prim's algorithm.
        
        Returns:
            List of edges in the MST, where each edge is represented as
            (vertex1, vertex2, weight)
        """
        # Initialize result list and visited set
        mst_edges: List[Tuple[int, int, float]] = []
        visited: Set[int] = set()
        
        # Priority queue for edges (weight, from_vertex, to_vertex)
        heap: List[Tuple[float, int, int]] = []
        
        # Start with vertex 0
        start_vertex = 0
        visited.add(start_vertex)
        
        # Add all edges from start_vertex to the heap
        for j in range(self.n_points):
            if j != start_vertex:
                weight = self._get_edge_weight(start_vertex, j)
                heapq.heappush(heap, (weight, start_vertex, j))
        
        # Continue until we have n-1 edges or the heap is empty
        while len(mst_edges) < self.n_points - 1 and heap:
            weight, u, v = heapq.heappop(heap)
            
            # Skip if both vertices are already in the MST
            if v in visited:
                continue
                
            # Add edge to MST
            mst_edges.append((u, v, weight))
            visited.add(v)
            
            # Add new edges from v to unvisited vertices
            for j in range(self.n_points):
                if j not in visited:
                    weight = self._get_edge_weight(v, j)
                    heapq.heappush(heap, (weight, v, j))
        
        return mst_edges
