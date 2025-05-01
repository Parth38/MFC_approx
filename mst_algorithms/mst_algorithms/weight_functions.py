from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import csr_matrix

class WeightFunction(ABC):
    """Abstract base class for weight functions."""
    
    @abstractmethod
    def compute(self, point1: csr_matrix, point2: csr_matrix) -> float:
        """Compute the weight/distance between two points.
        
        Args:
            point1: First point as a sparse matrix row
            point2: Second point as a sparse matrix row
            
        Returns:
            float: The computed weight/distance
        """
        pass

class JaccardWeight(WeightFunction):
    """Jaccard similarity-based weight function."""
    
    def compute(self, point1: csr_matrix, point2: csr_matrix) -> float:
        """Compute Jaccard similarity between two points.
        
        The Jaccard similarity is defined as:
        |A ∩ B| / |A ∪ B|
        
        Returns:
            float: 1 - Jaccard similarity (to convert to distance)
        """
        # Convert to dense arrays for intersection and union operations
        p1 = point1.toarray().flatten()
        p2 = point2.toarray().flatten()
        
        # Compute intersection and union
        intersection = np.sum(np.minimum(p1, p2))
        union = np.sum(np.maximum(p1, p2))
        
        if union == 0:
            return 1.0  # Maximum distance when both points are zero vectors
            
        jaccard_similarity = intersection / union
        return 1.0 - jaccard_similarity  # Convert to distance

class HammingWeight(WeightFunction):
    """Hamming distance-based weight function."""
    
    def compute(self, point1: csr_matrix, point2: csr_matrix) -> float:
        """Compute Hamming distance between two points.
        
        The Hamming distance is the number of positions at which
        the corresponding values are different.
        
        Returns:
            float: Normalized Hamming distance
        """
        # Convert to dense arrays
        p1 = point1.toarray().flatten()
        p2 = point2.toarray().flatten()
        
        # Compute Hamming distance
        distance = np.sum(p1 != p2)
        
        # Normalize by the length of the vectors
        return distance / len(p1)
