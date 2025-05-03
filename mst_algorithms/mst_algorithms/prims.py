import heapq
from typing import List, Tuple, Set
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
from .mst import MST

class PrimsMST(MST):
    """Optimized implementation of Prim's algorithm for Minimum Spanning Tree with progress bar."""
    
    def find_mst(self) -> List[Tuple[int, int, float]]:
        """Find MST using Prim's algorithm.
        
        Returns:
            List of edges in the MST, where each edge is represented as
            (vertex1, vertex2, weight)
        """
        # Initialize result list and visited array (faster than set for lookups)
        mst_edges = []
        n = self.n_points
        visited = np.zeros(n, dtype=bool)
        
        # Use key array to track minimum edge weight to each vertex
        key = np.full(n, np.inf)
        parent = np.full(n, -1)
        
        # Start with vertex 0
        start_vertex = 0
        key[start_vertex] = 0
        
        # Priority queue for vertices (key, vertex)
        # This approach avoids duplicate vertices in the queue
        pq = [(0, start_vertex)]
        
        # Setup progress bar - we need n-1 edges for a complete MST
        with tqdm(total=n-1, desc="Finding MST using Prim's algorithm") as pbar:
            while pq:
                _, u = heapq.heappop(pq)
                
                # Skip if already processed
                if visited[u]:
                    continue
                    
                visited[u] = True
                
                # Add edge to MST (except for root)
                if parent[u] != -1:
                    mst_edges.append((parent[u], u, key[u]))
                    # Update progress bar each time we add an edge
                    pbar.update(1)
                
                # Check all neighbors of u
                for v in range(n):
                    if not visited[v]:
                        weight = self._get_edge_weight(u, v)
                        if weight < key[v]:
                            key[v] = weight
                            parent[v] = u
                            heapq.heappush(pq, (key[v], v))
        
        return mst_edges