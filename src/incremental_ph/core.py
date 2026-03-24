"""
Incremental Persistent Homology — Numba-accelerated core.

Maintains the reduced boundary matrix R and basis change matrix V from
the standard PH reduction R = D * V. When simplices are inserted or
removed, only the affected columns are re-reduced — the rest of the
decomposition is untouched.

Algorithm:
  - Insertion: new simplices are appended, then reduced via
    `reduce_column_incremental` which handles pivot displacement
    (when a new column claims a pivot already owned by an existing
    column, the old owner is displaced and re-reduced).
  - Removal: for each surviving column whose V contains a removed
    simplex, XOR the removed simplex's boundary back into R, then
    re-reduce. This is the V-based XOR-back algorithm.

All arithmetic is over Z/2Z (XOR of sorted integer arrays), so there
is zero numerical drift regardless of how many updates are applied.

Validated against Ripser across 220+ individual checks with zero
mismatches. See FINDINGS.md for the full test suite.
"""

import numpy as np
import time
from collections import defaultdict
from dataclasses import dataclass, field
from numba import njit, types
from numba.typed import List as NumbaList


# -----------------------------------------------------------------------
# Numba-JIT core operations
# -----------------------------------------------------------------------

@njit(cache=True)
def xor_sorted(a, b):
    """Symmetric difference of two sorted int32 arrays (Z/2Z addition)."""
    out = np.empty(len(a) + len(b), dtype=np.int32)
    i = 0
    j = 0
    k = 0
    na = len(a)
    nb = len(b)
    while i < na and j < nb:
        if a[i] < b[j]:
            out[k] = a[i]; i += 1; k += 1
        elif a[i] > b[j]:
            out[k] = b[j]; j += 1; k += 1
        else:  # equal -> cancel (Z/2Z)
            i += 1; j += 1
    while i < na:
        out[k] = a[i]; i += 1; k += 1
    while j < nb:
        out[k] = b[j]; j += 1; k += 1
    return out[:k]


@njit(cache=True)
def reduce_all(n_simps, boundary_ids, boundary_lens, boundary_data,
               simp_dims, max_boundary):
    """Full left-to-right column reduction (standard PH algorithm).

    Returns:
        pivot_owner: pivot_owner[row_id] = col_id that owns this pivot
        col_pivot: col_pivot[col_id] = row_id that is this column's pivot
        R: reduced boundary columns (list of sorted int32 arrays)
        V: basis change columns (list of sorted int32 arrays)
    """
    pivot_owner = np.full(n_simps, -1, dtype=np.int32)
    col_pivot = np.full(n_simps, -1, dtype=np.int32)

    R = NumbaList()
    V = NumbaList()
    empty = np.empty(0, dtype=np.int32)
    for i in range(n_simps):
        R.append(empty.copy())
        V.append(empty.copy())

    for col in range(n_simps):
        dim = simp_dims[col]
        if dim == 0:
            V[col] = np.array([col], dtype=np.int32)
            continue

        # Initialize R[col] = sorted boundary face IDs
        start = boundary_data[col]
        blen = boundary_lens[col]
        bdry = np.empty(blen, dtype=np.int32)
        for bi in range(blen):
            bdry[bi] = boundary_ids[start + bi]
        for bi in range(blen):
            for bj in range(bi + 1, blen):
                if bdry[bi] > bdry[bj]:
                    tmp = bdry[bi]
                    bdry[bi] = bdry[bj]
                    bdry[bj] = tmp
        R[col] = bdry
        V[col] = np.array([col], dtype=np.int32)

        # Reduce: add columns until pivot is unique or column is zero
        while len(R[col]) > 0:
            lo = R[col][len(R[col]) - 1]
            owner = pivot_owner[lo]
            if owner >= 0:
                R[col] = xor_sorted(R[col], R[owner])
                V[col] = xor_sorted(V[col], V[owner])
            else:
                pivot_owner[lo] = col
                col_pivot[col] = lo
                break

    return pivot_owner, col_pivot, R, V


@njit(cache=True)
def _filt_earlier(a, b, simplex_dists, simp_dims):
    """Is simplex a earlier than b in filtration order?

    Compares (distance, dimension, ID) lexicographically.
    """
    da = simplex_dists[a]
    db = simplex_dists[b]
    if da < db:
        return True
    if da > db:
        return False
    dima = simp_dims[a]
    dimb = simp_dims[b]
    if dima < dimb:
        return True
    if dima > dimb:
        return False
    return a < b


@njit(cache=True)
def _find_low(R_col, simplex_dists, simp_dims):
    """Find the element with the MAX filtration key in an R column.

    In batch builds, max ID = max filtration key, so R_col[-1] works.
    After incremental insertion, new IDs are appended at the end but may
    have smaller distances than existing simplices. Must scan to find
    the true filtration-latest element.
    """
    best = R_col[0]
    for i in range(1, len(R_col)):
        cand = R_col[i]
        if not _filt_earlier(cand, best, simplex_dists, simp_dims):
            best = cand
    return best


@njit(cache=True)
def reduce_column_incremental(col_id, R_col, V_col,
                              pivot_owner, col_pivot,
                              R, V, n_simps,
                              simplex_dists, simp_dims):
    """Reduce a single column with pivot displacement.

    When a new column claims a pivot already owned by a later column in
    the filtration, the old owner is displaced: its R and V are XOR'd
    with the new column's, and it must be re-reduced.

    Returns: array of displaced column IDs that need cascade re-reduction.
    """
    displaced = np.empty(64, dtype=np.int32)
    n_displaced = 0

    while len(R_col) > 0:
        lo = _find_low(R_col, simplex_dists, simp_dims)
        owner = pivot_owner[lo]

        if owner >= 0:
            if _filt_earlier(col_id, owner, simplex_dists, simp_dims):
                # col_id is earlier in filtration -> TAKES the pivot
                pivot_owner[lo] = col_id
                col_pivot[col_id] = lo
                col_pivot[owner] = -1
                R[owner] = xor_sorted(R[owner], R_col)
                V[owner] = xor_sorted(V[owner], V_col)
                displaced[n_displaced] = owner
                n_displaced += 1
                break
            else:
                # Standard: add owner's column to ours
                R_col = xor_sorted(R_col, R[owner])
                V_col = xor_sorted(V_col, V[owner])
        else:
            pivot_owner[lo] = col_id
            col_pivot[col_id] = lo
            break

    R[col_id] = R_col
    V[col_id] = V_col

    if len(R_col) == 0 and col_pivot[col_id] >= 0:
        col_pivot[col_id] = -1

    return displaced[:n_displaced]


@njit(cache=True)
def _find_affected_and_fix(remove_mask, R, V, pivot_owner, col_pivot,
                           boundary_ids, boundary_starts, boundary_lens,
                           n_simps):
    """Core of removal: scan V, XOR back boundaries of removed simplices.

    For each surviving column whose V contains removed simplices:
    1. XOR the removed simplices' boundaries back into R (undo their contribution)
    2. Strip removed IDs from V
    3. Clear the column's pivot (it will be re-reduced)

    Returns array of affected column IDs that need re-reduction.
    """
    affected = np.empty(n_simps, dtype=np.int32)
    n_affected = 0

    for col in range(n_simps):
        if remove_mask[col]:
            continue

        v_col = V[col]
        has_removed = False
        for vi in range(len(v_col)):
            if remove_mask[v_col[vi]]:
                has_removed = True
                break

        if not has_removed:
            continue

        # XOR back boundaries of removed simplices in V[col]
        for vi in range(len(v_col)):
            s = v_col[vi]
            if not remove_mask[s]:
                continue
            bstart = boundary_starts[s]
            blen = boundary_lens[s]
            if blen == 0:
                continue
            bdry = np.empty(blen, dtype=np.int32)
            for bi in range(blen):
                bdry[bi] = boundary_ids[bstart + bi]
            for bi in range(1, blen):
                key = bdry[bi]
                bj = bi - 1
                while bj >= 0 and bdry[bj] > key:
                    bdry[bj + 1] = bdry[bj]
                    bj -= 1
                bdry[bj + 1] = key
            R[col] = xor_sorted(R[col], bdry)

        # Remove dead IDs from V[col]
        new_v = np.empty(len(v_col), dtype=np.int32)
        k = 0
        for vi in range(len(v_col)):
            if not remove_mask[v_col[vi]]:
                new_v[k] = v_col[vi]
                k += 1
        V[col] = new_v[:k]

        # Clear pivot
        old_pivot = col_pivot[col]
        if old_pivot >= 0:
            if pivot_owner[old_pivot] == col:
                pivot_owner[old_pivot] = -1
            col_pivot[col] = -1

        affected[n_affected] = col
        n_affected += 1

    # Clear pivots for removed simplices
    for s in range(n_simps):
        if not remove_mask[s]:
            continue
        old_pivot = col_pivot[s]
        if old_pivot >= 0:
            if pivot_owner[old_pivot] == s:
                pivot_owner[old_pivot] = -1
            col_pivot[s] = -1
        owner = pivot_owner[s]
        if owner >= 0 and not remove_mask[owner]:
            col_pivot[owner] = -1
            pivot_owner[s] = -1

    return affected[:n_affected]


# -----------------------------------------------------------------------
# State
# -----------------------------------------------------------------------

@dataclass
class NumbaState:
    """Persistent homology state backed by Numba arrays.

    Holds the full reduction state (R, V, pivots) plus the simplicial
    complex (edges, triangles, adjacency). All incremental operations
    mutate this state in-place.
    """
    # Simplex catalog
    id_to_simplex: list
    simplex_to_id: dict
    simplex_dists: np.ndarray   # filtration distance per simplex
    simp_dims: np.ndarray       # 0=vertex, 1=edge, 2=triangle
    n_simps: int

    # Reduction state
    pivot_owner: np.ndarray     # pivot_owner[row] = col (-1 if free)
    col_pivot: np.ndarray       # col_pivot[col] = row (-1 if zero)
    R: object                   # NumbaList of sorted int32 arrays
    V: object                   # NumbaList of sorted int32 arrays

    # Boundary arrays (for removal XOR-back)
    boundary_ids: np.ndarray
    boundary_lens: np.ndarray
    boundary_starts: np.ndarray

    # Graph state
    edge_set: set
    adj: dict
    edge_dists: dict
    n_points: int

    # Lifecycle
    alive: np.ndarray           # alive[id] = True if simplex exists

    def barcode_h1(self, min_persistence=1e-6):
        """Extract H1 persistence barcode.

        Args:
            min_persistence: minimum (death - birth) to include a bar.
                Default 1e-6 filters out zero-persistence noise bars
                that appear in dense graphs. Set to 0.0 to include all.

        Returns:
            np.ndarray of shape (n_bars, 2) with columns [birth, death].
            Each row is a 1-dimensional homology class (loop).
        """
        bars = []
        for row_id in range(self.n_simps):
            if not self.alive[row_id]:
                continue
            col_id = self.pivot_owner[row_id]
            if col_id < 0:
                continue
            if not self.alive[col_id]:
                continue
            dim_birth = self.simp_dims[row_id]
            dim_death = self.simp_dims[col_id]
            if dim_birth == 1 and dim_death == 2:
                d_birth = self.simplex_dists[row_id]
                d_death = self.simplex_dists[col_id]
                pers = d_death - d_birth
                if pers >= min_persistence:
                    bars.append([d_birth, d_death])
        return np.array(bars) if bars else np.empty((0, 2))


# -----------------------------------------------------------------------
# Filtration construction
# -----------------------------------------------------------------------

def _build_filtration(edge_set, edge_dists, adj, n_points):
    """Enumerate all simplices, sort by filtration key, assign integer IDs."""
    simplex_list = []

    for i in range(n_points):
        simplex_list.append(((i,), 0.0, 0))

    for edge in edge_set:
        simplex_list.append((edge, edge_dists[edge], 1))

    for i in range(n_points):
        if i not in adj:
            continue
        for j in adj[i]:
            if j <= i:
                continue
            common = adj[i] & adj[j] if j in adj else set()
            for k in common:
                if k <= j:
                    continue
                tri = (i, j, k)
                e1, e2, e3 = (i, j), (i, k), (j, k)
                d_tri = max(edge_dists[e1], edge_dists[e2], edge_dists[e3])
                simplex_list.append((tri, d_tri, 2))

    simplex_list.sort(key=lambda x: (x[1], x[2], x[0]))

    id_to_simplex = [s[0] for s in simplex_list]
    simplex_to_id = {s[0]: i for i, s in enumerate(simplex_list)}
    simplex_dists = np.array([s[1] for s in simplex_list], dtype=np.float64)
    simp_dims = np.array([s[2] for s in simplex_list], dtype=np.int32)

    return id_to_simplex, simplex_to_id, simplex_dists, simp_dims


def _build_boundary_arrays(id_to_simplex, simplex_to_id, simp_dims, n_simps):
    """Precompute boundary face IDs for every simplex."""
    boundary_ids_list = []
    boundary_lens = np.zeros(n_simps, dtype=np.int32)
    boundary_starts = np.zeros(n_simps, dtype=np.int32)
    pos = 0

    for simp_id in range(n_simps):
        simp = id_to_simplex[simp_id]
        dim = simp_dims[simp_id]

        if dim == 0:
            boundary_starts[simp_id] = pos
            boundary_lens[simp_id] = 0
        elif dim == 1:
            i, j = simp
            face_ids = [simplex_to_id[(i,)], simplex_to_id[(j,)]]
            boundary_starts[simp_id] = pos
            boundary_lens[simp_id] = 2
            boundary_ids_list.extend(face_ids)
            pos += 2
        elif dim == 2:
            i, j, k = simp
            faces = [tuple(sorted((i, j))), tuple(sorted((i, k))), tuple(sorted((j, k)))]
            face_ids = [simplex_to_id[f] for f in faces]
            boundary_starts[simp_id] = pos
            boundary_lens[simp_id] = 3
            boundary_ids_list.extend(face_ids)
            pos += 3

    boundary_ids = np.array(boundary_ids_list, dtype=np.int32) if boundary_ids_list else np.empty(0, dtype=np.int32)
    return boundary_ids, boundary_lens, boundary_starts


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def build(sparse_D, n_points, verbose=True):
    """Build PH state from a sparse distance matrix.

    Args:
        sparse_D: scipy sparse matrix (CSR/COO/etc.) where entry (i,j)
            is the distance between points i and j. Only upper triangle
            is needed; both directions are accepted.
        n_points: number of vertices in the complex.
        verbose: print timing and size info.

    Returns:
        NumbaState ready for incremental updates.
    """
    return build_initial_numba(sparse_D, n_points, verbose=verbose)


def build_initial_numba(sparse_D, n_points, verbose=True):
    """Build PH state from sparse distance matrix using Numba-accelerated reduction."""
    t0 = time.perf_counter()

    coo = sparse_D.tocoo()
    edge_set = set()
    edge_dists = {}
    adj = defaultdict(set)

    for i, j, d in zip(coo.row, coo.col, coo.data):
        if i < j:
            edge = (i, j)
            if edge not in edge_set:
                edge_set.add(edge)
                edge_dists[edge] = float(d)
                adj[i].add(j)
                adj[j].add(i)

    t_graph = time.perf_counter() - t0

    id_to_simplex, simplex_to_id, simplex_dists, simp_dims = \
        _build_filtration(edge_set, edge_dists, adj, n_points)
    n_simps = len(id_to_simplex)

    t_filt = time.perf_counter() - t0 - t_graph

    boundary_ids, boundary_lens, boundary_starts = \
        _build_boundary_arrays(id_to_simplex, simplex_to_id, simp_dims, n_simps)

    t_bdry = time.perf_counter() - t0 - t_graph - t_filt

    pivot_owner, col_pivot, R, V = reduce_all(
        n_simps, boundary_ids, boundary_lens, boundary_starts,
        simp_dims, 3
    )

    t_reduce = time.perf_counter() - t0 - t_graph - t_filt - t_bdry
    t_total = time.perf_counter() - t0

    n_bars = 0
    for row_id in range(n_simps):
        col_id = pivot_owner[row_id]
        if col_id >= 0 and simp_dims[row_id] == 1 and simp_dims[col_id] == 2:
            if simplex_dists[col_id] - simplex_dists[row_id] > 1e-6:
                n_bars += 1

    if verbose:
        print(f"  [incremental-ph] {n_points} pts, {len(edge_set)} edges, "
              f"{n_simps} simplices, {n_bars} H1 bars")
        print(f"  Time: {t_graph*1000:.0f}ms graph + {t_filt*1000:.0f}ms filt + "
              f"{t_bdry*1000:.0f}ms bdry + {t_reduce*1000:.0f}ms reduce = "
              f"{t_total*1000:.0f}ms total")

    alive = np.ones(n_simps, dtype=np.bool_)

    state = NumbaState(
        id_to_simplex=id_to_simplex,
        simplex_to_id=simplex_to_id,
        simplex_dists=simplex_dists,
        simp_dims=simp_dims,
        n_simps=n_simps,
        pivot_owner=pivot_owner,
        col_pivot=col_pivot,
        R=R, V=V,
        boundary_ids=boundary_ids,
        boundary_lens=boundary_lens,
        boundary_starts=boundary_starts,
        edge_set=edge_set,
        adj=dict(adj),
        edge_dists=edge_dists,
        n_points=n_points,
        alive=alive,
    )
    return state


def insert_point(state, new_distances, epsilon=None, k=None):
    """Insert a new point into the complex with exact barcode update.

    Adds the vertex, edges to neighbors (within epsilon-ball or k-nearest),
    and all induced triangles. Then reduces the new columns incrementally.

    Args:
        state: NumbaState from build().
        new_distances: array of distances from new point to all existing
            points. Length must equal state.n_points. Zero entries are
            treated as "no edge" (use epsilon or k to control connectivity).
        epsilon: distance threshold for edge creation. Edges are created
            to all existing points within this distance.
        k: number of nearest neighbors. Edges to the k closest points.
        Must specify exactly one of epsilon or k.

    Returns:
        dict with keys: new_point, n_new_edges, n_new_triangles,
        n_new_simplices, n_displaced, time_total_ms.
    """
    t0 = time.perf_counter()
    new_pt = state.n_points
    state.n_points += 1

    if epsilon is not None:
        neighbors = [int(i) for i in range(len(new_distances))
                     if 0 < new_distances[i] <= epsilon]
    elif k is not None:
        k_nearest = np.argsort(new_distances)[:k]
        neighbors = [int(i) for i in k_nearest if new_distances[i] > 0]
    else:
        raise ValueError("Must specify either epsilon or k")

    new_simps = []

    # Vertex
    vtx = (new_pt,)
    new_simps.append((vtx, 0.0, 0))

    # Edges
    new_edges = []
    for qi in neighbors:
        d = float(new_distances[qi])
        edge = tuple(sorted((new_pt, qi)))
        state.edge_set.add(edge)
        state.edge_dists[edge] = d
        if new_pt not in state.adj:
            state.adj[new_pt] = set()
        state.adj[new_pt].add(qi)
        if qi not in state.adj:
            state.adj[qi] = set()
        state.adj[qi].add(new_pt)
        new_edges.append(edge)
        new_simps.append((edge, d, 1))

    # Triangles
    new_tris = []
    for qi in sorted(neighbors):
        for qj in sorted(neighbors):
            if qj <= qi:
                continue
            existing_edge = tuple(sorted((qi, qj)))
            if existing_edge in state.edge_set:
                tri = tuple(sorted((new_pt, qi, qj)))
                d_tri = max(new_distances[qi], new_distances[qj],
                            state.edge_dists[existing_edge])
                new_tris.append(tri)
                new_simps.append((tri, float(d_tri), 2))

    if len(new_simps) <= 1:
        _grow_state(state, new_simps)
        return {"new_point": new_pt, "n_new_edges": 0, "n_new_triangles": 0,
                "n_new_simplices": 1, "n_displaced": 0, "time_total_ms": 0}

    new_simps.sort(key=lambda x: (x[1], x[2], x[0]))
    new_ids = _grow_state(state, new_simps)

    # Reduce new columns
    n_displaced = 0
    for new_id in new_ids:
        dim = state.simp_dims[new_id]
        if dim == 0:
            continue

        simp = state.id_to_simplex[new_id]
        if dim == 1:
            i, j = simp
            face_ids = sorted([state.simplex_to_id[(i,)], state.simplex_to_id[(j,)]])
        else:
            i, j, k = simp
            faces = [tuple(sorted((i, j))), tuple(sorted((i, k))), tuple(sorted((j, k)))]
            face_ids = sorted([state.simplex_to_id[f] for f in faces])
        R_col = np.array(face_ids, dtype=np.int32)
        V_col = np.array([new_id], dtype=np.int32)

        displaced = reduce_column_incremental(
            new_id, R_col, V_col,
            state.pivot_owner, state.col_pivot,
            state.R, state.V, state.n_simps,
            state.simplex_dists, state.simp_dims
        )

        while len(displaced) > 0:
            next_displaced_list = []
            for disp_id in displaced:
                n_displaced += 1
                more = reduce_column_incremental(
                    disp_id, state.R[disp_id], state.V[disp_id],
                    state.pivot_owner, state.col_pivot,
                    state.R, state.V, state.n_simps,
                    state.simplex_dists, state.simp_dims
                )
                if len(more) > 0:
                    next_displaced_list.extend(more)
            if next_displaced_list:
                displaced = np.array(next_displaced_list, dtype=np.int32)
            else:
                break

    t_total = time.perf_counter() - t0
    return {
        "new_point": new_pt,
        "n_new_edges": len(new_edges),
        "n_new_triangles": len(new_tris),
        "n_new_simplices": len(new_simps),
        "n_displaced": n_displaced,
        "time_total_ms": t_total * 1000,
    }


def _grow_state(state, new_simps):
    """Grow all state arrays to accommodate new simplices."""
    n_new = len(new_simps)
    old_n = state.n_simps
    new_n = old_n + n_new
    empty = np.empty(0, dtype=np.int32)

    state.simplex_dists = np.concatenate([state.simplex_dists,
                                          np.array([s[1] for s in new_simps], dtype=np.float64)])
    state.simp_dims = np.concatenate([state.simp_dims,
                                      np.array([s[2] for s in new_simps], dtype=np.int32)])
    state.alive = np.concatenate([state.alive, np.ones(n_new, dtype=np.bool_)])
    state.pivot_owner = np.concatenate([state.pivot_owner, np.full(n_new, -1, dtype=np.int32)])
    state.col_pivot = np.concatenate([state.col_pivot, np.full(n_new, -1, dtype=np.int32)])
    state.boundary_lens = np.concatenate([state.boundary_lens, np.zeros(n_new, dtype=np.int32)])
    state.boundary_starts = np.concatenate([state.boundary_starts, np.zeros(n_new, dtype=np.int32)])

    new_ids = []
    for idx, (simp, dist, dim) in enumerate(new_simps):
        new_id = old_n + idx
        state.id_to_simplex.append(simp)
        state.simplex_to_id[simp] = new_id
        new_ids.append(new_id)

        if dim == 1:
            i, j = simp
            face_ids = sorted([state.simplex_to_id[(i,)], state.simplex_to_id[(j,)]])
            bdry_arr = np.array(face_ids, dtype=np.int32)
        elif dim == 2:
            i, j, k = simp
            faces = [tuple(sorted((i, j))), tuple(sorted((i, k))), tuple(sorted((j, k)))]
            face_ids = sorted([state.simplex_to_id[f] for f in faces])
            bdry_arr = np.array(face_ids, dtype=np.int32)
        else:
            bdry_arr = empty

        bstart = len(state.boundary_ids)
        state.boundary_ids = np.concatenate([state.boundary_ids, bdry_arr])
        state.boundary_starts[new_id] = bstart
        state.boundary_lens[new_id] = len(bdry_arr)

        state.R.append(empty.copy())
        state.V.append(np.array([new_id], dtype=np.int32))

    state.n_simps = new_n
    return new_ids


def remove_edges(state, edges):
    """Remove edges and their coface triangles with exact barcode update.

    Uses the V-based XOR-back algorithm: for each surviving column whose
    basis change matrix V references a removed simplex, the removed
    simplex's boundary is XOR'd back into R to undo its contribution,
    then the column is re-reduced.

    Args:
        state: NumbaState from build().
        edges: list of edges to remove, each as (i, j) tuple.

    Returns:
        dict with keys: n_edges_removed, n_tris_removed, n_total_removed,
        n_affected, n_displaced, time_fix_ms, time_total_ms.
    """
    t0 = time.perf_counter()

    all_to_remove = set()
    n_tris_removed = 0

    for edge in edges:
        edge = tuple(sorted(edge))
        if edge not in state.simplex_to_id:
            continue
        all_to_remove.add(edge)

        i, j = edge
        if i in state.adj and j in state.adj:
            common = state.adj[i] & state.adj[j]
            for k in common:
                tri = tuple(sorted((i, j, k)))
                if tri in state.simplex_to_id:
                    all_to_remove.add(tri)
                    n_tris_removed += 1

    if not all_to_remove:
        return {"n_edges_removed": 0, "n_tris_removed": 0,
                "n_affected": 0, "n_displaced": 0, "time_total_ms": 0}

    n_edges_removed = sum(1 for s in all_to_remove if len(s) == 2)

    remove_mask = np.zeros(state.n_simps, dtype=np.bool_)
    for s in all_to_remove:
        remove_mask[state.simplex_to_id[s]] = True

    # Numba JIT: scan V, XOR back, clear pivots
    affected_ids = _find_affected_and_fix(
        remove_mask, state.R, state.V,
        state.pivot_owner, state.col_pivot,
        state.boundary_ids, state.boundary_starts, state.boundary_lens,
        state.n_simps
    )

    t_fix = time.perf_counter() - t0

    # Mark removed simplices as dead
    for s in all_to_remove:
        sid = state.simplex_to_id[s]
        state.alive[sid] = False
        state.R[sid] = np.empty(0, dtype=np.int32)
        state.V[sid] = np.empty(0, dtype=np.int32)

    # Update graph state
    for s in all_to_remove:
        if len(s) == 2:
            i, j = s
            state.edge_set.discard(s)
            state.edge_dists.pop(s, None)
            if i in state.adj:
                state.adj[i].discard(j)
            if j in state.adj:
                state.adj[j].discard(i)

    # Re-reduce affected columns
    affected_ids.sort()
    n_displaced = 0

    for col_id in affected_ids:
        displaced = reduce_column_incremental(
            col_id, state.R[col_id], state.V[col_id],
            state.pivot_owner, state.col_pivot,
            state.R, state.V, state.n_simps,
            state.simplex_dists, state.simp_dims
        )
        while len(displaced) > 0:
            next_list = []
            for d in displaced:
                n_displaced += 1
                more = reduce_column_incremental(
                    d, state.R[d], state.V[d],
                    state.pivot_owner, state.col_pivot,
                    state.R, state.V, state.n_simps,
                    state.simplex_dists, state.simp_dims
                )
                if len(more) > 0:
                    next_list.extend(more)
            if next_list:
                displaced = np.array(next_list, dtype=np.int32)
            else:
                break

    t_total = time.perf_counter() - t0
    return {
        "n_edges_removed": n_edges_removed,
        "n_tris_removed": n_tris_removed,
        "n_total_removed": len(all_to_remove),
        "n_affected": len(affected_ids),
        "n_displaced": n_displaced,
        "time_fix_ms": t_fix * 1000,
        "time_total_ms": t_total * 1000,
    }


def knn_update(state, new_distances, local_eps_old, k=15):
    """Full KNN incremental update: evict edges + insert new point.

    Computes the new point's local-epsilon, finds edges that no longer
    pass the KNN threshold, removes them, then inserts the new point
    with KNN-filtered edges.

    Args:
        state: NumbaState from build().
        new_distances: distances from new point to all existing points.
        local_eps_old: current epsilon values for existing points.
        k: KNN parameter (default 15).

    Returns:
        dict with timing, counts, and updated local_eps array.
    """
    t0 = time.perf_counter()
    n_existing = state.n_points

    k_actual = min(k, n_existing)
    new_eps = float(np.sort(new_distances[:n_existing])[:k_actual][-1])
    local_eps = np.concatenate([local_eps_old, [new_eps]])

    # Find evicted edges
    evicted = []
    for edge in list(state.edge_set):
        i, j = edge
        if i >= n_existing or j >= n_existing:
            continue
        d = state.edge_dists[edge]
        if d > max(local_eps[i], local_eps[j]):
            evicted.append(edge)

    t_find = time.perf_counter() - t0

    stats_remove = None
    if evicted:
        stats_remove = remove_edges(state, evicted)

    t_remove = time.perf_counter() - t0

    # Insert with KNN-filtered edges
    masked_dists = np.zeros(n_existing)
    for j in range(n_existing):
        d = new_distances[j]
        if d <= max(new_eps, local_eps[j]):
            masked_dists[j] = d

    stats_insert = insert_point(state, masked_dists, epsilon=2.0)

    t_total = time.perf_counter() - t0

    return {
        "n_evicted": len(evicted),
        "remove": stats_remove,
        "insert": stats_insert,
        "new_eps": new_eps,
        "local_eps": local_eps,
        "time_find_ms": t_find * 1000,
        "time_remove_ms": (t_remove - t_find) * 1000,
        "time_insert_ms": (t_total - t_remove) * 1000,
        "time_total_ms": t_total * 1000,
    }


# -----------------------------------------------------------------------
# Warmup
# -----------------------------------------------------------------------

def warmup():
    """Pre-compile all Numba JIT functions with tiny data.

    Call once at startup to avoid JIT compilation latency on the first
    real call. Takes ~100ms.
    """
    t0 = time.perf_counter()
    boundary_ids = np.array([0, 1, 0, 2, 1, 2, 3, 4, 5], dtype=np.int32)
    boundary_lens = np.array([0, 0, 0, 2, 2, 2, 3], dtype=np.int32)
    boundary_starts = np.array([0, 0, 0, 0, 2, 4, 6], dtype=np.int32)
    simp_dims = np.array([0, 0, 0, 1, 1, 1, 2], dtype=np.int32)
    _, _, R, V = reduce_all(7, boundary_ids, boundary_lens, boundary_starts, simp_dims, 3)
    a = np.array([1, 3, 5], dtype=np.int32)
    b = np.array([2, 3, 4], dtype=np.int32)
    xor_sorted(a, b)
    rm = np.zeros(7, dtype=np.bool_)
    rm[6] = True
    po = np.full(7, -1, dtype=np.int32)
    cp = np.full(7, -1, dtype=np.int32)
    _find_affected_and_fix(rm, R, V, po, cp, boundary_ids, boundary_starts, boundary_lens, 7)
    R_col = np.array([3, 4], dtype=np.int32)
    V_col = np.array([5], dtype=np.int32)
    sd = np.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.15, 0.2], dtype=np.float64)
    dims = np.array([0, 0, 0, 1, 1, 1, 2], dtype=np.int32)
    reduce_column_incremental(5, R_col, V_col, po, cp, R, V, 7, sd, dims)
    _filt_earlier(3, 4, sd, dims)
    ms = (time.perf_counter() - t0) * 1000
    print(f"  [incremental-ph] Numba warmup: {ms:.0f}ms")
