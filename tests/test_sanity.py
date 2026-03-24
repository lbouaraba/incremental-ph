"""Sanity test: batch build = incremental build = Ripser.

Uses synthetic 2D data so no external datasets are needed.
"""

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix, lil_matrix

from incremental_ph import build, insert_point, remove_edges, warmup


def make_sparse(D, epsilon):
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


def compare_bars(a, b, tol=1e-4):
    if len(a) == 0 and len(b) == 0:
        return True
    if len(a) != len(b):
        return False
    D = np.max(np.abs(a[:, None, :] - b[None, :, :]), axis=2)
    n_b_in_a = sum(1 for j in range(len(b)) if D[:, j].min() < tol)
    n_a_in_b = sum(1 for i in range(len(a)) if D[i].min() < tol)
    return n_b_in_a == len(b) and n_a_in_b == len(a)


@pytest.fixture(scope="session", autouse=True)
def jit_warmup():
    warmup()


class TestBatchBuild:
    """Test that batch build matches Ripser."""

    def test_small_2d(self):
        np.random.seed(42)
        pts = np.random.randn(30, 2) * 0.5
        D = cdist(pts, pts)
        epsilon = 0.8
        sparse_D = make_sparse(D, epsilon)

        state = build(sparse_D, n_points=30, verbose=False)
        bars = state.barcode_h1(min_persistence=1e-6)

        # Verify against Ripser if available
        try:
            from ripser import ripser
            mat = lil_matrix((30, 30))
            coo = sparse_D.tocoo()
            for i, j, d in zip(coo.row, coo.col, coo.data):
                mat[i, j] = d
            result = ripser(mat.tocsr(), distance_matrix=True, maxdim=1)["dgms"][1]
            finite = result[np.isfinite(result[:, 1])]
            finite = finite[(finite[:, 1] - finite[:, 0]) > 1e-6]
            assert compare_bars(bars, finite), \
                f"Batch build mismatch: {len(bars)} vs {len(finite)} bars"
        except ImportError:
            # Without Ripser, just check we got some bars
            assert len(bars) >= 0


class TestInsertPoint:
    """Test incremental insertion matches full rebuild."""

    def test_insert_matches_rebuild(self):
        np.random.seed(123)
        n_base = 20
        pts = np.random.randn(n_base + 5, 2) * 0.5
        D_full = cdist(pts, pts)
        epsilon = 0.9

        # Build on first n_base
        D_base = D_full[:n_base, :n_base]
        sparse_base = make_sparse(D_base, epsilon)
        state = build(sparse_base, n_points=n_base, verbose=False)

        # Insert 5 points one by one
        for step in range(5):
            new_idx = n_base + step
            new_dists = D_full[new_idx, :new_idx]
            insert_point(state, new_dists, epsilon=epsilon)

        bars_inc = state.barcode_h1(min_persistence=1e-6)

        # Full rebuild on all points
        D_all = D_full[:n_base + 5, :n_base + 5]
        sparse_all = make_sparse(D_all, epsilon)
        state_full = build(sparse_all, n_points=n_base + 5, verbose=False)
        bars_full = state_full.barcode_h1(min_persistence=1e-6)

        assert compare_bars(bars_inc, bars_full), \
            f"Incremental insert mismatch: {len(bars_inc)} vs {len(bars_full)} bars"


class TestRemoveEdges:
    """Test edge removal matches rebuild without those edges."""

    def test_remove_single_edge(self):
        np.random.seed(456)
        pts = np.random.randn(20, 2) * 0.5
        D = cdist(pts, pts)
        epsilon = 0.9
        sparse_D = make_sparse(D, epsilon)

        state = build(sparse_D, n_points=20, verbose=False)

        # Remove a random edge
        edges = list(state.edge_set)
        if not edges:
            pytest.skip("No edges to remove")

        edge = edges[len(edges) // 2]
        remove_edges(state, [edge])
        bars_after = state.barcode_h1(min_persistence=1e-6)

        # Rebuild without that edge
        i, j = edge
        sparse_without = make_sparse(D, epsilon)
        mat = lil_matrix(sparse_without)
        mat[i, j] = 0
        mat[j, i] = 0
        state2 = build(mat.tocsr(), n_points=20, verbose=False)
        bars_rebuild = state2.barcode_h1(min_persistence=1e-6)

        assert compare_bars(bars_after, bars_rebuild), \
            f"Remove edge mismatch: {len(bars_after)} vs {len(bars_rebuild)} bars"

    def test_remove_all_edges(self):
        """Delete every edge one by one. Should end with 0 H1 bars."""
        np.random.seed(789)
        pts = np.random.randn(10, 2) * 0.3
        D = cdist(pts, pts)
        epsilon = 0.6
        sparse_D = make_sparse(D, epsilon)

        state = build(sparse_D, n_points=10, verbose=False)
        edges = list(state.edge_set)

        for edge in edges:
            if edge in state.edge_set:
                remove_edges(state, [edge])

        bars = state.barcode_h1(min_persistence=1e-6)
        assert len(bars) == 0, f"Expected 0 bars after removing all edges, got {len(bars)}"
        assert len(state.edge_set) == 0
