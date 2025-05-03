
from pathlib import Path
from itertools import combinations
import time
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


FILE = Path("./gaussian-samples/nodes-100.txt")   
T_VALUES = [2, 4, 6, 8, 10]                       
SEED = 42                                        


def load_points(file_path: Path, delim: str = ",") -> np.ndarray:
    return np.loadtxt(file_path, delimiter=delim)

def pairwise_cost(u: int, v: int, pts: np.ndarray) -> float:
    return float(np.hypot(*(pts[u] - pts[v])))

def mst_cost(edge_list, pts):
    return sum(pairwise_cost(u, v, pts) for u, v in edge_list)

def full_graph_mst(points: np.ndarray):
    """Exact MST on the complete graph (Kruskal)."""
    n = len(points)
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=pairwise_cost(i, j, points))
    return nx.minimum_spanning_tree(G, algorithm="kruskal")

def cluster_points(points, k, seed=SEED):
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    return km.fit_predict(points)

def intra_partition_msts(points, labels):
    msts = {}
    for cid in np.unique(labels):
        idx = np.where(labels == cid)[0]
        H = nx.Graph()
        for i, j in combinations(idx, 2):
            H.add_edge(i, j, weight=pairwise_cost(i, j, points))
        msts[cid] = nx.minimum_spanning_tree(H, algorithm="kruskal")
    return msts

def representative_edges(points, labels, msts, seed=SEED):
    # Pick one random node per cluster
    rng = np.random.default_rng(seed)
    reps = {cid: int(rng.choice(np.where(labels == cid)[0]))
            for cid in np.unique(labels)}
    # Build complete graph on reps
    G = nx.Graph()
    for cid, rid in reps.items():
        G.add_node(cid, rep=rid)
    for c1, c2 in combinations(reps, 2):
        u, v = reps[c1], reps[c2]
        G.add_edge(c1, c2,
                   weight=pairwise_cost(u, v, points),
                   endpoints=(u, v))
    rep_mst = nx.minimum_spanning_tree(G, algorithm="kruskal")
    return [data["endpoints"] for _, _, data in rep_mst.edges(data=True)]

pts = load_points(FILE)
n = len(pts)


t0 = time.perf_counter()
exact_MST = full_graph_mst(pts)
exact_time = time.perf_counter() - t0
exact_edges = list(exact_MST.edges())
exact_cost = mst_cost(exact_edges, pts)
exact_edge_set = {tuple(sorted(e)) for e in exact_edges}


rows = []
for t in T_VALUES:
    start = time.perf_counter()

    # clustering
    lbl = cluster_points(pts, t)

    # per‑cluster MSTs
    intra = intra_partition_msts(pts, lbl)

    # edges that join clusters (Metric‑Forest Completion)
    inter = representative_edges(pts, lbl, intra)

    # put everything together
    approx_edges = [e for mst in intra.values() for e in mst.edges()] + inter
    approx_cost = mst_cost(approx_edges, pts)
    approx_time = time.perf_counter() - start

    # γ‑overlap (initial forest vs exact MST)
    forest_edge_set = {tuple(sorted(e))
                       for mst in intra.values()
                       for e in mst.edges()}
    gamma =  (n - 2) /  len(forest_edge_set & exact_edge_set) 

    # partition quality
    sil = silhouette_score(pts, lbl)

    rows.append({
        "t": t,
        "approx_cost": approx_cost,
        "cost_ratio": approx_cost / exact_cost,
        "approx_time (s)": approx_time,
        "runtime_ratio": exact_time / approx_time,
        "gamma_overlap": gamma,
        "silhouette": sil,
    })

df = pd.DataFrame(rows)
print(df)


fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Runtime ratio
axes[0].plot(df["t"], df["runtime_ratio"], marker="o")
axes[0].set_title("Runtime ratio (exact / approx)")
axes[0].set_xlabel("Number of clusters $t$")
axes[0].set_ylabel("Runtime ratio")
axes[0].grid(alpha=0.4)

# Cost ratio
axes[1].plot(df["t"], df["cost_ratio"], marker="o")
axes[1].set_title("Cost ratio (approx / exact)")
axes[1].set_xlabel("Number of clusters $t$")
axes[1].set_ylabel("Cost ratio")
axes[1].grid(alpha=0.4)

# γ vs silhouette
sc = axes[2].scatter(df["silhouette"], df["gamma_overlap"],
                     c=df["t"], cmap="viridis", s=70)
for _, r in df.iterrows():
    axes[2].annotate(f"t={int(r['t'])}",
                     (r["silhouette"], r["gamma_overlap"]),
                     textcoords="offset points", xytext=(5, -8), fontsize=8)
axes[2].set_title("γ‑overlap vs partition quality")
axes[2].set_xlabel("Silhouette score")
axes[2].set_ylabel("γ‑overlap")
axes[2].grid(alpha=0.4)

plt.colorbar(sc, ax=axes[2], label="$t$")
plt.tight_layout()
plt.show()
