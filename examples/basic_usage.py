"""Basic usage of incremental-ph.

Builds a small random complex, inserts points one by one, and removes
edges — showing that the barcode stays exact throughout.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix

from incremental_ph import build, insert_point, remove_edges, warmup


def make_sparse(D, epsilon):
    """Sparsify a distance matrix with an epsilon-ball."""
    n = D.shape[0]
    rows, cols, data = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= epsilon:
                rows += [i, j]
                cols += [j, i]
                data += [D[i, j], D[i, j]]
    if not rows:
        return coo_matrix((n, n)).tocsr()
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def main():
    np.random.seed(42)
    warmup()

    # --- Build initial complex ---
    n_base = 30
    points = np.random.randn(n_base, 5) * 0.5
    D = cdist(points, points)
    epsilon = 1.2
    sparse_D = make_sparse(D, epsilon)

    state = build(sparse_D, n_points=n_base)
    bars = state.barcode_h1(min_persistence=1e-6)
    print(f"\nInitial: {n_base} points, {len(state.edge_set)} edges, {len(bars)} H1 bars")

    # --- Insert 10 points one by one ---
    print("\nInserting 10 points:")
    for i in range(10):
        new_point = np.random.randn(5) * 0.5
        new_dists = np.linalg.norm(points[:state.n_points] - new_point, axis=1)
        # Pad to current size
        full_dists = np.zeros(state.n_points)
        full_dists[:len(new_dists)] = new_dists

        stats = insert_point(state, full_dists, epsilon=epsilon)
        bars = state.barcode_h1(min_persistence=1e-6)
        print(f"  Point {n_base + i}: {stats['n_new_edges']} edges, "
              f"{stats['n_new_triangles']} tris, "
              f"{len(bars)} H1 bars, {stats['time_total_ms']:.1f}ms")

        # Track the new point for future distance calculations
        points = np.vstack([points, new_point.reshape(1, -1)])

    # --- Remove 5 random edges ---
    print("\nRemoving 5 random edges:")
    edges = list(state.edge_set)
    np.random.shuffle(edges)
    for edge in edges[:5]:
        stats = remove_edges(state, [edge])
        bars = state.barcode_h1(min_persistence=1e-6)
        print(f"  Removed {edge}: {stats['n_tris_removed']} tris removed, "
              f"{len(bars)} H1 bars, {stats['time_total_ms']:.1f}ms")

    print(f"\nFinal: {state.n_points} points, {len(state.edge_set)} edges, "
          f"{len(bars)} H1 bars")


if __name__ == "__main__":
    main()
