from pathlib import Path
from typing import Tuple, Dict, List
from itertools import combinations   
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_points(file_path: Path, delimiter: str = ",") -> np.ndarray:
    return np.loadtxt(file_path, delimiter=delimiter)

def cluster_points(points: np.ndarray, k: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(points)
    return labels, kmeans.cluster_centers_

def build_graph(points: np.ndarray, labels: np.ndarray) -> nx.Graph:
    G = nx.Graph()
    for i, (x, y) in enumerate(points):
        G.add_node(
            i,
            pos=(x, y),
            cluster=int(labels[i]),
        )
    return G

def plot_clusters(
    G: nx.Graph,
    msts: Dict[int, nx.Graph],
    mfc_edges: List[Tuple[int, int]],
    figsize: Tuple[int, int] = (7, 7),
    marker_size: int = 80,
):
    palette = [
        (0.121, 0.466, 0.705),
        (1.000, 0.498, 0.055),
        (0.172, 0.627, 0.172),
        (0.839, 0.153, 0.157),
        (0.580, 0.404, 0.741),
        (0.549, 0.337, 0.294),
        (0.890, 0.467, 0.761),
        (0.498, 0.498, 0.498),
        (0.737, 0.741, 0.133),
        (0.090, 0.745, 0.811),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.2, 1.0),
        (0.75, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.6, 1.0, 0.6),
        (1.0, 0.4, 0.0),
        (0.0, 1.0, 0.5),
        (0.8, 1.0, 1.0),
    ]

    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.get_node_attributes(G, "pos")

    for cid, mst in msts.items():
        color = palette[cid % len(palette)]
        nx.draw_networkx_edges(
            mst, pos, ax=ax,
            edge_color=[color] * mst.number_of_edges(),
            width=2.0, alpha=0.9,
        )

    nx.draw_networkx_edges(
        G, pos, edgelist=mfc_edges, ax=ax,
        edge_color="black", width=4.0, alpha=1.0
    )

    node_colors = [palette[G.nodes[n]["cluster"] % len(palette)] for n in G.nodes]
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=marker_size,
        node_color=node_colors,
        edgecolors="black", linewidths=0.5,
    )

    ax.set_title("Clusters, per‑cluster MSTs, and Metric Forest Completion edges")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    plt.show()

def _complete_graph_for_cluster(
    point_indices: np.ndarray,
    points: np.ndarray
) -> nx.Graph:
    H = nx.Graph()
    for idx in point_indices:
        x, y = points[idx]
        H.add_node(idx, pos=(x, y))
    for u, v in combinations(point_indices, 2):
        x1, y1 = points[u]
        x2, y2 = points[v]
        w = np.hypot(x1 - x2, y1 - y2)
        H.add_edge(u, v, weight=w)
    return H

def compute_msts_per_partition(
    points: np.ndarray,
    labels: np.ndarray
) -> Dict[int, nx.Graph]:
    msts = {}
    unique_clusters = np.unique(labels)
    for c in unique_clusters:
        idx = np.where(labels == c)[0]
        complete_g = _complete_graph_for_cluster(idx, points)
        mst = nx.minimum_spanning_tree(complete_g, weight="weight", algorithm="kruskal")
        msts[c] = mst
    return msts

def compute_mfc_edges(
    points: np.ndarray,
    labels: np.ndarray
) -> List[Tuple[int, int]]:
    clusters = np.unique(labels)
    super_G = nx.Graph()
    super_G.add_nodes_from(clusters)

    best_edge: Dict[Tuple[int, int], Tuple[float, Tuple[int, int]]] = {}
    for c1, c2 in combinations(clusters, 2):
        idx1 = np.where(labels == c1)[0]
        idx2 = np.where(labels == c2)[0]

        d_min, u_best, v_best = float("inf"), None, None
        for u in idx1:
            for v in idx2:
                d = np.hypot(*(points[u] - points[v]))
                if d < d_min:
                    d_min, u_best, v_best = d, u, v
        best_edge[(c1, c2)] = (d_min, (u_best, v_best))
        super_G.add_edge(c1, c2, weight=d_min)

    super_mst = nx.minimum_spanning_tree(super_G, algorithm="kruskal")

    mfc_edges = [
        best_edge[tuple(sorted(e))][1]
        for e in super_mst.edges()
    ]
    return mfc_edges

def main():
    file_path = './gaussian-samples/nodes-100.txt'
    points = load_points(file_path)
    t = 1
    labels, centers = cluster_points(points, t)
    G = build_graph(points, labels)
    msts = compute_msts_per_partition(points, labels)
    mfc_edges = compute_mfc_edges(points, labels)
    G = build_graph(points, labels)
    plot_clusters(G, msts, mfc_edges)

if __name__ == "__main__":
    main()
