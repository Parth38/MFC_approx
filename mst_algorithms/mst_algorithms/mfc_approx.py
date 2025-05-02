
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
        # Choose representative randomly for each partition
        representatives = {label: random.choice(nodes) for label, nodes in self.partitions.items()}
        labels = sorted(self.partitions.keys())

        for i, label_i in enumerate(labels):
            for label_j in labels[i + 1:]:
                x̃i = min(
                    self.partitions[label_i],
                    key=lambda x: self.weight_function.compute(self.sparse_matrix[x], self.sparse_matrix[representatives[label_j]])
                )
                x̃j = min(
                    self.partitions[label_j],
                    key=lambda x: self.weight_function.compute(self.sparse_matrix[x], self.sparse_matrix[representatives[label_i]])
                )
                wi_to_j = self.weight_function.compute(self.sparse_matrix[x̃i], self.sparse_matrix[representatives[label_j]])
                wj_to_i = self.weight_function.compute(self.sparse_matrix[x̃j], self.sparse_matrix[representatives[label_i]])
                w_x̃ix̃j = self.weight_function.compute(self.sparse_matrix[x̃i], self.sparse_matrix[x̃j])

                min_weight = min(wi_to_j, wj_to_i, w_x̃ix̃j)
                if min_weight == wi_to_j:
                    best_pair = (x̃i, representatives[label_j])
                elif min_weight == wj_to_i:
                    best_pair = (representatives[label_i], x̃j)
                else:
                    best_pair = (x̃i, x̃j)

                self.coarsened_graph[(label_i, label_j)] = (min_weight, best_pair)

        self._save_coarsened_graph()

    def compute_final_tree(self) -> List[Tuple[int, int, float]]:
        label_to_idx = {label: i for i, label in enumerate(sorted(self.partitions))}
        idx_to_label = {i: label for label, i in label_to_idx.items()}

        coarsened_edges = []
        for (label_i, label_j), (weight, _) in self.coarsened_graph.items():
            i, j = label_to_idx[label_i], label_to_idx[label_j]
            coarsened_edges.append((i, j, weight))

        n_partitions = len(label_to_idx)
        coarsened_matrix = self._create_coarsened_matrix()

        mst = KruskalsMST(coarsened_matrix, self.weight_function)
        coarsened_mst = mst.find_mst()

        final_edges = []
        for i, j, weight in coarsened_mst:
            label_i, label_j = idx_to_label[i], idx_to_label[j]
            w, (u, v) = self.coarsened_graph.get((label_i, label_j), self.coarsened_graph.get((label_j, label_i)))
            final_edges.append((u, v, w))

        all_edges = [edge for tree in self.partition_trees.values() for edge in tree]
        all_edges.extend(final_edges)

        self._save_tree(all_edges, 'mfc_approx_final_tree.txt')
        total_weight = sum(weight for _, _, weight in all_edges)
        print(f"Total weight of MFC-Approx tree: {total_weight:.4f}")

        return all_edges

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

        beta, gamma = self.calculate_beta_gamma()
        print(f"\nBeta (max partitions per node): {beta}")
        print(f"Gamma (max partition size): {gamma}")

        print("Computing final tree...")
        final_tree = self.compute_final_tree()

        # Optionally calculate gamma (approximate)
        gamma_approx = self.calculate_gamma(final_tree)
        print(f"\nApproximate gamma (forest overlap): {gamma_approx:.4f}")

        return final_tree

    def _save_tree(self, edges: List[Tuple[int, int, float]], filename: str):
        with open(filename, 'w') as f:
            for u, v, w in edges:
                f.write(f"{u},{v},{w}\n")

    def _save_coarsened_graph(self):
        with open('coarsened_graph.txt', 'w') as f:
            for (i, j), (weight, (u, v)) in self.coarsened_graph.items():
                f.write(f"{i},{j},{weight},{u},{v}\n")
