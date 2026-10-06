# Constrained K-Means Clustering

`Ai4r::Clusterers::ConstrainedKMeans` is a standalone entry point for
k-means when some records are known to describe the same object (must-link)
or are known to belong to different clusters (cannot-link). It does **not**
run ordinary k-means and then move violating points: every allocation pass
places groups while respecting the constraints.

## Constraints

- **must-link** pairs are treated as an equivalence relation. The transitive
  closure is computed up front, and each equivalence group is placed into a
  cluster as a whole. Every member participates in centroid computation, so
  larger groups move centroids more.
- **cannot-link** pairs forbid two records from sharing a cluster.
- A cannot-link pair whose endpoints are already connected by a must-link
  chain is a direct contradiction. It raises
  `Ai4r::Clusterers::ConstraintConflict` **before any iteration**, with a
  `conflicts` array that reports each offending pair together with the
  must-link chain proving the contradiction.

## Allocation and feasibility

Each iteration first tries a deterministic greedy placement of the
must-link groups (cheapest feasible cluster in group order). When the greedy
pass paints itself into a corner, a budgeted backtracking search takes over
and is allowed to undo earlier greedy choices. The search uses a
most-constrained-group ordering and never violates cannot-link edges.

The end reason distinguishes the two failure modes:

- `:infeasible` — the search exhausted the whole decision tree within
  budget, proving no valid allocation exists for the current centroids.
- `:budget_exhausted` — the configured `search_budget` (default 10000 group
  placements per pass, `nil` for unbounded) ran out before the tree could be
  fully explored; feasibility is therefore unknown.

Other end reasons are `:converged`, `:max_iterations`, and
`:empty_cluster` (with `on_empty: 'terminate'`).

Empty clusters follow the configured `on_empty` policy (`'terminate'`,
`'eliminate'`, `'random'`, `'outlier'`). Relocation is constraint aware:
after moving an empty centroid, the allocation is recomputed so the moved
group still satisfies must-link and cannot-link constraints.

## Result

`build_constrained` returns a `ConstrainedResult`:

- `assignments` — cluster index (0-based) for **every original sample**, or
  `nil` for samples that could not be placed.
- `unassigned_indices` — the explicit list of unplaced samples, so no record
  is ever hidden.
- `objective_value` (aliased as `sse`) — final total distance to centroids,
  or `nil` when some sample is unassigned.
- `end_reason`, `iterations`, `number_of_clusters`, `centroids`,
  `clusters`, `must_link_groups`, and `stats` (including whether greedy
  failed and how many search placements were tried).

## Example

```ruby
require 'ai4r'
include Ai4r::Clusterers
include Ai4r::Data

data = DataSet.new(data_items: [[0.0], [10.0], [20.0], [19.0]])
cannot_link = [[0, 1], [1, 2], [0, 3], [1, 3], [2, 3]]

result = ConstrainedKMeans.new
  .set_parameters(centroid_indices: [0, 1, 2])
  .build_constrained(data, 3, cannot_link: cannot_link)

result.assignments   # e.g. [0, 1, 0, 2] — group 2 was moved back to share
                     # cluster 0 after the greedy pass got stuck
result.end_reason    # :converged
result.objective_value
```

Direct contradictions are reported before iterating:

```ruby
begin
  ConstrainedKMeans.new.build_constrained(
    data, 2, must_link: [[0, 1], [1, 2]], cannot_link: [[0, 2]]
  )
rescue ConstraintConflict => e
  e.conflicts
  # [{ pair: [0, 2], must_link_path: [0, 1, 2] }]
end
```
