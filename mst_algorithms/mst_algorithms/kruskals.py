from typing import List, Tuple, Dict
from scipy.sparse import csr_matrix
from .mst import MST

class UnionFind:
    """Union-Find data structure for Kruskal's algorithm."""
    
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size
        
    def find(self, x: int) -> int:
        """Find the root of the set containing x."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
        
    def union(self, x: int, y: int) -> bool:
        """Union the sets containing x and y.
        
        Returns:
            True if the sets were merged, False if they were already in the same set
        """
        x_root = self.find(x)
        y_root = self.find(y)
        
        if x_root == y_root:
            return False
            
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        elif self.rank[x_root] > self.rank[y_root]:
            self.parent[y_root] = x_root
        else:
            self.parent[y_root] = x_root
            self.rank[x_root] += 1
            
        return True

class KruskalsMST(MST):
    """Implementation of Kruskal's algorithm for finding Minimum Spanning Tree."""
    
    def find_mst(self) -> List[Tuple[int, int, float]]:
        """Find MST using Kruskal's algorithm.
        
        Returns:
            List of edges in the MST, where each edge is represented as
            (vertex1, vertex2, weight)
        """
        # Initialize result list and Union-Find structure
        mst_edges: List[Tuple[int, int, float]] = []
        uf = UnionFind(self.n_points)
        
        # Generate all possible edges with their weights
        edges: List[Tuple[float, int, int]] = []
        for i in range(self.n_points):
            for j in range(i + 1, self.n_points):
                weight = self._get_edge_weight(i, j)
                edges.append((weight, i, j))
        
        # Sort edges by weight
        edges.sort()
        
        # Process edges in order of increasing weight
        for weight, u, v in edges:
            if uf.union(u, v):
                mst_edges.append((u, v, weight))
                if len(mst_edges) == self.n_points - 1:
                    break
        
        return mst_edges
