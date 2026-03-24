"""Incremental persistent homology with exact updates.

Insert and remove simplices from a Vietoris-Rips complex without batch
recomputation. Maintains the reduced boundary matrix R and basis change
matrix V incrementally, producing barcodes that exactly match Ripser.

Typical usage:

    from incremental_ph import build, insert_point, remove_edges, warmup
    from scipy.sparse import coo_matrix

    warmup()  # pre-compile Numba JIT (once)

    state = build(sparse_distance_matrix, n_points=200)
    barcode = state.barcode_h1(min_persistence=1e-6)

    # Insert a new point
    stats = insert_point(state, new_distances, epsilon=0.5)
    barcode = state.barcode_h1(min_persistence=1e-6)

    # Remove edges
    stats = remove_edges(state, [(3, 7), (12, 45)])
    barcode = state.barcode_h1(min_persistence=1e-6)
"""

from incremental_ph.core import (
    NumbaState,
    build,
    build_initial_numba,
    insert_point,
    remove_edges,
    knn_update,
    warmup,
)

__version__ = "0.1.0"
__all__ = [
    "NumbaState",
    "build",
    "build_initial_numba",
    "insert_point",
    "remove_edges",
    "knn_update",
    "warmup",
]
