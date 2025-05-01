# MST Algorithms Implementation

This project implements two Minimum Spanning Tree (MST) algorithms - Prim's and Kruskal's - with support for different weight functions. This was developed as part of a college course project.

## Features

- Implementation of Prim's algorithm using priority queue
- Implementation of Kruskal's algorithm using Union-Find data structure
- Support for different weight functions:
  - Jaccard similarity/distance
  - Hamming distance
- Works with sparse matrices where each row represents a point in metric space

## Project Structure

```
mst_algorithms/
├── mst_algorithms/
│   ├── __init__.py
│   ├── weight_functions.py    # Weight function implementations
│   ├── mst.py                 # Base MST class
│   ├── prims.py              # Prim's algorithm implementation
│   └── kruskals.py           # Kruskal's algorithm implementation
└── requirements.txt          # Project dependencies
```

## Requirements

- Python 3.8+
- numpy>=1.21.0
- scipy>=1.7.0

## Usage Example

```python
from scipy.sparse import csr_matrix
from mst_algorithms import PrimsMST, KruskalsMST, JaccardWeight

# Create your sparse matrix
sparse_matrix = csr_matrix(...)

# Create MST instance with Jaccard weight function
mst = PrimsMST(sparse_matrix, JaccardWeight())

# Find MST
edges = mst.find_mst()
```

## Implementation Details

### Prim's Algorithm
- Time Complexity: O(E log V)
- Uses priority queue for efficient edge selection
- Greedy algorithm that grows the MST from a starting vertex

### Kruskal's Algorithm
- Time Complexity: O(E log E)
- Uses Union-Find data structure for cycle detection
- Greedy algorithm that sorts edges by weight

### Weight Functions
- Jaccard: Computes similarity based on intersection over union
- Hamming: Computes normalized Hamming distance between points 