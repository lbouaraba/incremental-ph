# incremental-ph

Incremental persistent homology with exact updates. Insert and remove simplices from a Vietoris-Rips complex without batch recomputation.

This library maintains the reduced boundary matrix (R=DV decomposition) incrementally, updating only the affected columns when simplices are inserted or removed. The result is exact (matches Ripser bit-for-bit over Z/2Z) and 10-30x faster, and provides a practical, pip-installable Python package combining insertion, removal, and KNN updates for high-dimensional point clouds.

## Related work

The theoretical foundations for incremental persistent homology are established:

- **Insertion** into a filtration is the standard case, handled by extending the R=DV decomposition with new columns (Edelsbrunner & Harer, 2010).
- **Removal** of simplices was solved by [SiRUP](https://arxiv.org/abs/2312.03925) (Giunti & Lazovskis, 2023), which provides a provably optimal algorithm with minimal column additions.
- **Vineyard updates** for transpositions were introduced by Cohen-Steiner, Edelsbrunner & Morozov (2006).

This library was developed independently — the removal mechanism (V-based XOR-back) was discovered empirically before we became aware of SiRUP. The practical contribution is the combination of insertion + removal + KNN pipeline in a single Numba-accelerated package, tested on high-dimensional data (1024D) with 220+ exact validation checks against Ripser.

## What it does

- **Insert a point** into a live complex: ~3ms (vs ~105ms for Ripser rebuild)
- **Remove edges** in any order: ~1-8ms depending on edge degree
- **KNN pipeline**: automatic edge eviction + insertion for streaming data
- **Exact**: Z/2Z arithmetic has zero numerical drift. Validated against Ripser across 220+ individual checks with zero mismatches.

## Install

```bash
pip install incremental-ph
```

Or from source:

```bash
git clone https://github.com/lbouaraba/incremental-ph
cd incremental-ph
pip install -e ".[dev]"
```

Requires Python 3.9+ and a C compiler for Numba.

## Quick start

```python
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
from incremental_ph import build, insert_point, remove_edges, warmup

# Pre-compile Numba JIT functions (once, ~100ms)
warmup()

# Create a sparse distance matrix from points
points = np.random.randn(100, 10)
D = cdist(points, points)
epsilon = 1.5

# Sparsify: only keep edges within epsilon
rows, cols, data = [], [], []
for i in range(len(D)):
    for j in range(i + 1, len(D)):
        if D[i, j] <= epsilon:
            rows += [i, j]
            cols += [j, i]
            data += [D[i, j], D[i, j]]
sparse_D = coo_matrix((data, (rows, cols)), shape=D.shape).tocsr()

# Build initial PH state
state = build(sparse_D, n_points=100)
bars = state.barcode_h1(min_persistence=1e-6)
print(f"Initial: {len(bars)} H1 bars")

# Insert a new point
new_point = np.random.randn(10)
new_dists = np.array([np.linalg.norm(new_point - p) for p in points])
stats = insert_point(state, new_dists, epsilon=epsilon)
bars = state.barcode_h1(min_persistence=1e-6)
print(f"After insert: {len(bars)} H1 bars ({stats['time_total_ms']:.1f}ms)")

# Remove an edge
edge = list(state.edge_set)[0]
stats = remove_edges(state, [edge])
bars = state.barcode_h1(min_persistence=1e-6)
print(f"After remove: {len(bars)} H1 bars ({stats['time_total_ms']:.1f}ms)")
```

## API

### `warmup()`

Pre-compiles all Numba JIT functions. Call once at startup (~100ms).

### `build(sparse_D, n_points, verbose=True) -> NumbaState`

Build PH state from a scipy sparse distance matrix.

### `insert_point(state, new_distances, epsilon=None, k=None) -> dict`

Insert a new vertex. Creates edges to neighbors within `epsilon` distance or to `k` nearest neighbors, plus all induced triangles. Returns timing and count stats.

### `remove_edges(state, edges) -> dict`

Remove edges (and their coface triangles). Edges can be removed in any order — not just reverse filtration. Returns timing and count stats.

### `knn_update(state, new_distances, local_eps_old, k=15) -> dict`

Full KNN incremental update: evicts edges that no longer pass the KNN threshold, then inserts the new point with KNN-filtered edges.

### `state.barcode_h1(min_persistence=1e-6) -> np.ndarray`

Extract the H1 persistence barcode. Returns array of shape `(n_bars, 2)` with `[birth, death]` columns. The default `min_persistence=1e-6` filters out zero-persistence noise bars that appear in dense graphs. Set to `0.0` to include all bars.

## How it works

Standard PH reduces a boundary matrix D into R = D * V by column operations over Z/2Z. This library maintains R and V across updates:

**Insertion**: New simplices are appended and reduced via `reduce_column_incremental`, which handles *pivot displacement* — when a new column claims a pivot already owned by a later column in the filtration, the old owner is displaced and re-reduced.

**Removal**: For each surviving column whose V references a removed simplex, the removed simplex's boundary is XOR'd back into R (undoing its contribution), then the column is re-reduced. This is the *V-based XOR-back* algorithm. See [SiRUP](https://arxiv.org/abs/2312.03925) (Giunti & Lazovskis, 2023) for a theoretically optimal treatment of the removal problem.

Since all arithmetic is XOR over Z/2Z, there is zero numerical drift regardless of how many updates are applied.

## Performance

Benchmarked on 500 arXiv papers embedded in 1024D (cosine distance, KNN k=15):

| Operation | Time | Notes |
|-----------|------|-------|
| Build (200 pts) | 30ms | Initial reduction |
| Insert point | 3.7ms mean | 300 sequential insertions |
| Remove edge (low degree) | 0.6ms | 1 triangle affected |
| Remove edge (high degree) | 7.8ms | 43 triangles affected |
| KNN update (evict + insert) | 3.7ms mean | Including edge eviction |

Scaling is sub-linear in edge degree: 43x more triangles costs only 14x more time.

## Validation

The test suite runs 220+ individual comparisons against Ripser with zero mismatches:

| Test | What | Result |
|------|------|--------|
| Sanity check | Batch = incremental = Ripser | 3/3 EXACT |
| Cascade bomb | Worst-case hub edge removal | EXACT, 4ms |
| KNN torture | 100 sequential KNN insertions | 10/10 EXACT |
| Void death | Semantic topology validation | 4/4 EXACT |
| Degree scaling | Low/mid/high degree edges | Sub-linear, all < 8ms |
| Empty complex | Delete all edges one by one | 45/45 EXACT |
| Wrong order | Random deletion order | 50/50 EXACT |
| Real data | 300 KNN insertions, 200->500 pts | 6/6 EXACT, 3.66ms |

## Acknowledgments

[Ripser](https://github.com/Ripser/ripser) by Ulrich Bauer serves as the ground truth for all validation. The removal mechanism was developed independently using AI-assisted development (Claude, Kimi); we acknowledge [SiRUP](https://arxiv.org/abs/2312.03925) by Giunti & Lazovskis as prior work addressing the same problem with provably optimal guarantees.

## License

[Hippocratic License 3.0](LICENSE.md) — free to use, copy, and modify, with ethical use conditions.
