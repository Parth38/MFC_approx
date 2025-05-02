
import random
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict, Tuple
from .kruskals import KruskalsMST
from .weight_functions import WeightFunction

class MFCApprox:
    def __init__(self, sparse_matrix: csr_matrix, labels: List[int], weight_function: WeightFunction):
        self.sparse_matrix = sparse_matrix
        self.labels = labels
        self.weight_function = weight_function
        self.n_points = sparse_matrix.shape[0]

        self.partitions: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            self.partitions.setdefault(label, []).append(idx)
 
        # ── MFCApprox.__init__  (after you filled self.partitions) ────────────────
        # 1. collect singleton vertices
        singleton_vertices = []
        for pid, verts in list(self.partitions.items()):   # list() → safe to delete
            if len(verts) == 1:
                singleton_vertices.extend(verts)
                del self.partitions[pid]                   # remove the 1‑element part

        # 2.   + any vertex that never got a label at all
        assigned = {v for verts in self.partitions.values() for v in verts}
        unassigned = [v for v in range(self.n_points) if v not in assigned]
        singleton_vertices.extend(unassigned)

        # 3. create one unified “catch‑all” partition if needed
        if singleton_vertices:
            extra_pid = max(self.partitions.keys(), default=-1) + 1
            self.partitions[extra_pid] = singleton_vertices
            print(f"Grouped {len(singleton_vertices)} singleton vertices into "
                f'partition {extra_pid}')


        self.partition_trees: Dict[int, List[Tuple[int, int, float]]] = {}
        self.coarsened_graph: Dict[Tuple[int, int], Tuple[float, Tuple[int, int]]] = {}

    def compute_partition_trees(self):
        for label, indices in self.partitions.items():
            partition_matrix = self.sparse_matrix[indices]
            mst = KruskalsMST(partition_matrix, self.weight_function)
            edges = mst.find_mst()
            original_edges = [(indices[u], indices[v], weight) for u, v, weight in edges]
            self.partition_trees[label] = original_edges
            self._save_tree(original_edges, f'partition_tree_{label}.txt')

    def compute_coarsened_graph(self):
        """
        Build the two‑level (partition‑level) graph used by MFC‑Approx.

        For every ordered pair of partitions (Pi, Pj) we keep the lightest of
            • d(xi*, sj)  –  nearest point in Pi to the representative of Pj
            • d(xj*, si)  –  nearest point in Pj to the representative of Pi
            • d(xi*, xj*) –  direct edge between the two nearest points
        and remember the vertex pair that realises that distance.
        """
        import random

        # ── random representative for each partition ──────────────────────────
        reps = {lbl: random.choice(nodes) for lbl, nodes in self.partitions.items()}
        labels = sorted(self.partitions.keys())

        self.coarsened_graph = {}          # clear any previous contents

        # ── consider every unordered pair of partitions ───────────────────────
        for idx, li in enumerate(labels):
            for lj in labels[idx + 1:]:
                # nearest point in Pi to rep(Pj)
                xi_star = min(
                    self.partitions[li],
                    key=lambda x: self.weight_function.compute(
                        self.sparse_matrix[x], self.sparse_matrix[reps[lj]]
                    )
                )
                # nearest point in Pj to rep(Pi)
                xj_star = min(
                    self.partitions[lj],
                    key=lambda x: self.weight_function.compute(
                        self.sparse_matrix[x], self.sparse_matrix[reps[li]]
                    )
                )

                # three candidate edge weights
                w_i_to_j = self.weight_function.compute(
                    self.sparse_matrix[xi_star], self.sparse_matrix[reps[lj]]
                )
                w_j_to_i = self.weight_function.compute(
                    self.sparse_matrix[xj_star], self.sparse_matrix[reps[li]]
                )
                w_star_pair = self.weight_function.compute(
                    self.sparse_matrix[xi_star], self.sparse_matrix[xj_star]
                )

                # keep the lightest and remember which two vertices gave it
                min_w = min(w_i_to_j, w_j_to_i, w_star_pair)
                if min_w == w_i_to_j:
                    best_pair = (xi_star, reps[lj])
                elif min_w == w_j_to_i:
                    best_pair = (reps[li], xj_star)
                else:
                    best_pair = (xi_star, xj_star)

                self.coarsened_graph[(li, lj)] = (min_w, best_pair)

            self._save_coarsened_graph()

    def compute_final_tree(self) -> List[Tuple[int, int, float]]:
        """
        Step 3 of Algorithm 1 (MFC‑Approx).

        • Take the complete graph of partitions with weights we already stored
        in `self.coarsened_graph`.
        • Run Kruskal *directly* on those pre‑computed edges (no recomputation).
        • Map each chosen inter‑partition edge (Pi, Pj) back to the concrete
        vertex pair (u, v) that realised its weight.
        • Union that set with all intra‑partition MST edges.
        """

        # --- 1. Relabel partition IDs to 0..p-1 --------------------------------
        labels         = sorted(self.partitions)              # stable order
        label_to_idx   = {lbl: i for i, lbl in enumerate(labels)}
        idx_to_label   = {i: lbl for lbl, i in label_to_idx.items()}
        n_partitions   = len(labels)

        # --- 2. Build explicit edge list (u_idx, v_idx, w_uv) -------------------
        coarsened_edges: List[Tuple[int, int, float]] = [
            (label_to_idx[i], label_to_idx[j], w_uv[0])      # w_uv = (weight, (u,v))
            for (i, j), w_uv in self.coarsened_graph.items()
        ]
        coarsened_edges.sort(key=lambda e: e[2])             # Kruskal needs sorting

        # --- 3. Kruskal on the partition graph ---------------------------------
        parent = list(range(n_partitions))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]                # path compression
                x = parent[x]
            return x

        chosen_partition_edges: List[Tuple[int, int, float]] = []
        for u_idx, v_idx, w in coarsened_edges:
            ru, rv = find(u_idx), find(v_idx)
            if ru != rv:
                parent[ru] = rv
                chosen_partition_edges.append((u_idx, v_idx, w))
                if len(chosen_partition_edges) == n_partitions - 1:
                    break

        # --- 4. Convert each partition edge back to its concrete vertex pair ---
        final_edges: List[Tuple[int, int, float]] = []
        for u_idx, v_idx, _ in chosen_partition_edges:
            li, lj = idx_to_label[u_idx], idx_to_label[v_idx]
            key     = (li, lj) if (li, lj) in self.coarsened_graph else (lj, li)
            w, (u, v) = self.coarsened_graph[key]            # safe – key exists
            final_edges.append((u, v, w))

        # --- 5. Union intra‑ and inter‑partition edges -------------------------
        all_edges = [e for tree in self.partition_trees.values() for e in tree]
        all_edges.extend(final_edges)

        # (Optional) save and report
        self._save_tree(all_edges, 'mfc_approx_final_tree.txt')
        total_weight = sum(w for _, _, w in all_edges)
        print(f"Total weight of MFC‑Approx tree: {total_weight:.4f}")

        return all_edges

    def compute_mfc_approx(self) -> List[Tuple[int, int, float]]:
        """
        Fast representative‑only variant of MFC‑Approx (per your clarification):

        • One random representative  s_i  ←  P_i  for every partition.
        • Build the complete p‑node graph on those reps (p = #partitions).
        • Run a tiny MST (Kruskal) on that graph to choose p−1 edges.
        • Union those edges with every intra‑partition MST edge (T_i).

        Returns: list of  (u, v, w)  edges over ORIGINAL node indices.
        """
        # --------------------------------------------------  A. intra‑partition MSTs
        if not self.partition_trees:
            self.compute_partition_trees()       # fills self.partition_trees

        # --------------------------------------------------  B. pick representatives
        reps: Dict[int, int] = {
            label: random.choice(nodes) for label, nodes in self.partitions.items()
        }
        labels     = list(reps.keys())               # arbitrary order is fine
        idx_of     = {lbl: i for i, lbl in enumerate(labels)}
        label_of   = {i: lbl for lbl, i in idx_of.items()}
        p          = len(labels)

        # --------------------------------------------------  C. build complete graph
        rep_edges: List[Tuple[int, int, float]] = []   # (idx_i, idx_j, w)
        for a in range(p):
            u = reps[label_of[a]]
            for b in range(a + 1, p):
                v = reps[label_of[b]]
                w = self.weight_function.compute(self.sparse_matrix[u],
                                                self.sparse_matrix[v])
                rep_edges.append((a, b, w))

        # --------------------------------------------------  D. MST on reps (tiny)
        rep_edges.sort(key=lambda e: e[2])             # Kruskal sort
        parent = list(range(p))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        inter_edges: List[Tuple[int, int, float]] = []  # (u, v, w) with global IDs
        for i_idx, j_idx, w in rep_edges:
            ri, rj = find(i_idx), find(j_idx)
            if ri != rj:
                parent[ri] = rj
                inter_edges.append(
                    (reps[label_of[i_idx]], reps[label_of[j_idx]], w)
                )
                if len(inter_edges) == p - 1:
                    break

        # --------------------------------------------------  E. union + return
        final_edges = [e for tree in self.partition_trees.values() for e in tree]
        final_edges.extend(inter_edges)

        self._save_tree(final_edges, 'mfc_approx_final_tree.txt')
        print("MFC‑Approx (rep‑only) total weight:",
            sum(w for _, _, w in final_edges))
        return final_edges

    def compute_optimal_mfc(self) -> List[Tuple[int, int, float]]:
        """
        Optimal baseline:

        • For every unordered pair of partitions (P_i, P_j) find the
            TRUE minimum‑distance edge  (u, v).
        • Run Kruskal on that dense p‑node graph (p = #partitions).
        • Union the chosen p−1 edges with every intra‑partition MST edge.

        No use of self.coarsened_graph; only fresh distance calls.
        """
        # ---------------------------------------------  A. intra‑partition MSTs
        if not self.partition_trees:
            self.compute_partition_trees()

        labels   = sorted(self.partitions)
        idx_of   = {lbl: i for i, lbl in enumerate(labels)}
        label_of = {i: lbl for lbl, i in idx_of.items()}
        p        = len(labels)

        # ---------------------------------------------  B. true min edges between Pi,Pj
        true_edges: List[Tuple[int, int, float, int, int]] = []  # (idx_i, idx_j, w, u, v)
        for a in range(p):
            li = labels[a]
            for b in range(a + 1, p):
                lj = labels[b]
                best_w  = float('inf')
                best_uv = (-1, -1)
                for u in self.partitions[li]:
                    for v in self.partitions[lj]:
                        w = self.weight_function.compute(
                            self.sparse_matrix[u], self.sparse_matrix[v]
                        )
                        if w < best_w:
                            best_w, best_uv = w, (u, v)
                true_edges.append((a, b, best_w, *best_uv))

        # ---------------------------------------------  C. Kruskal on partition graph
        true_edges.sort(key=lambda e: e[2])
        parent = list(range(p))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        inter_edges: List[Tuple[int, int, float]] = []
        for i_idx, j_idx, w, u, v in true_edges:
            ri, rj = find(i_idx), find(j_idx)
            if ri != rj:
                parent[ri] = rj
                inter_edges.append((u, v, w))
                if len(inter_edges) == p - 1:
                    break

        # ---------------------------------------------  D. union + return
        final_edges = [e for tree in self.partition_trees.values() for e in tree]
        final_edges.extend(inter_edges)

        self._save_tree(final_edges, 'optimal_mfc_tree.txt')
        print("Optimal‑MFC total weight:", sum(w for _, _, w in final_edges))
        return final_edges

    def compute_true_mst(self) -> List[Tuple[int, int, float]]:
        n = self.n_points
        X = self.sparse_matrix
        d = self.weight_function.compute           # alias for brevity

        edges: List[Tuple[float, int, int]] = []
        for u in range(n):
            Xu = X[u]                             # row → CSR slice, avoids copy
            for v in range(u + 1, n):
                w = d(Xu, X[v])
                edges.append((w, u, v))

        # ------------------------------------------------------------------ B. sort edges by weight (Kruskal prerequisite)
        edges.sort(key=lambda triple: triple[0])   # O(m log m) with m = n(n−1)/2

        # ------------------------------------------------------------------ C. union–find helpers (path compression + size)
        parent = list(range(n))
        size   = [1] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]      # path compression
                x = parent[x]
            return x

        def union(a: int, b: int) -> bool:
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            # union by size (keep tree shallow)
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]
            return True

        # ------------------------------------------------------------------ D. Kruskal main loop
        mst_edges: List[Tuple[int, int, float]] = []
        for w, u, v in edges:
            if union(u, v):
                mst_edges.append((u, v, w))
                if len(mst_edges) == n - 1:       # tree complete
                    break
        self._save_tree(mst_edges, 'true_mst.txt')
        return mst_edges

    def _create_coarsened_matrix(self) -> csr_matrix:
        labels = sorted(self.partitions)
        label_to_idx = {label: i for i, label in enumerate(labels)}
        n = len(labels)

        data, row_ind, col_ind = [], [], []
        for (label_i, label_j), (weight, _) in self.coarsened_graph.items():
            i, j = label_to_idx[label_i], label_to_idx[label_j]
            row_ind += [i, j]
            col_ind += [j, i]
            data += [weight, weight]

        return csr_matrix((data, (row_ind, col_ind)), shape=(n, n))

    def calculate_beta_gamma(self) -> Tuple[int, int]:
        gamma = max(len(indices) for indices in self.partitions.values())
        node_partition_count = {}
        for indices in self.partitions.values():
            for node in indices:
                node_partition_count[node] = node_partition_count.get(node, 0) + 1
        beta = max(node_partition_count.values(), default=0)
        return beta, gamma

    def calculate_gamma(self, optimal_mst: List[Tuple[int, int, float]]) -> float:
        forest_weight = sum(weight for tree in self.partition_trees.values() for _, _, weight in tree)
        mst_inside_partitions_weight = 0
        for u, v, w in optimal_mst:
            for partition in self.partitions.values():
                part_set = set(partition)
                if u in part_set and v in part_set:
                    mst_inside_partitions_weight += w
                    break
        return float('inf') if mst_inside_partitions_weight == 0 else forest_weight / mst_inside_partitions_weight

    def run(self) -> List[Tuple[int, int, float]]:
        print("Computing partition trees...")
        self.compute_partition_trees()

        print("Computing coarsened graph...")
        self.compute_coarsened_graph()

        #beta, gamma = self.calculate_beta_gamma()
        #print(f"\nBeta (max partitions per node): {beta}")
        #print(f"Gamma (max partition size): {gamma}")

        print("Computing Aprox final tree...")
        final_tree = self.compute_mfc_approx()

        # Optionally calculate gamma (approximate)
        #gamma_approx = self.calculate_gamma(final_tree)
        #print(f"\nApproximate gamma (forest overlap): {gamma_approx:.4f}")

        return final_tree
    
    def runOptimal(self) -> List[Tuple[int, int, float]]:
        self.compute_partition_trees()

        print("Computing coarsened graph...")
        self.compute_coarsened_graph()

 

        print("Computing Aprox final tree...")
        final_tree = self.compute_optimal_mfc()

 

        return final_tree

    def _save_tree(self, edges: List[Tuple[int, int, float]], filename: str):
        with open(filename, 'w') as f:
            for u, v, w in edges:
                f.write(f"{u},{v},{w}\n")

    def _save_coarsened_graph(self):
        with open('coarsened_graph.txt', 'w') as f:
            for (i, j), (weight, (u, v)) in self.coarsened_graph.items():
                f.write(f"{i},{j},{weight},{u},{v}\n")
