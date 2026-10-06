# KMeans Clustering

`Ai4r::Clusterers::KMeans` partitions data into K clusters by minimizing the
within-cluster variance. Points are assigned to the nearest centroid and the
centroids move to the mean of their assigned points until convergence.

## Example

```ruby
require 'ai4r'
include Ai4r::Clusterers
include Ai4r::Data

points = [[1,1],[1,2],[9,8],[9,9]]
set = DataSet.new(data_items: points)
clusterer = KMeans.new
clusterer.build(set, 2)
pp clusterer.clusters.map(&:data_items)
```

`build` returns the clusterer itself so you can inspect `clusters`,
`iterations`, and the sum of squared errors via `sse`.

See `examples/clusterers/kmeans_custom_example.rb` for a variant using
Manhattan distance.

## Constrained clustering (must-link / cannot-link)

`ConstrainedKMeans#build_constrained` is a separate entry point for
instance-level constraints. It does not run ordinary k-means first and
repair violations afterwards: must-link pairs are closed transitively into
groups with a union-find before any iteration, each group is assigned to a
single cluster as an atomic unit, and every member of the group contributes
to that cluster's centroid.

```ruby
require 'ai4r'
points = [[0,0],[0,100],[100,0],[100,100]]
set = DataSet.new(data_items: points)
result = ConstrainedKMeans.new
  .set_parameters(random_seed: 7)
  .build_constrained(
    set, 3,
    must_link: [[0, 1]],          # 0 and 1 stay together (transitive)
    cannot_link: [[0, 3], [2, 3]] # 3 never shares a cluster with 0 or 2
  )

result.assignments      # cluster index per original item, e.g. [0, 0, 2, 1]
result.objective        # final objective (sum of distances to centroids)
result.reason           # :converged | :max_iterations | :infeasible |
                        # :budget_exhausted
result.clusters.map { |c| c.data_items.size }
```

Behavior worth knowing:

- A cannot-link pair contradicted by the must-link closure raises
  `ConstraintConflict` before centroids are initialized; the exception
  exposes every offending pair and the closed groups.
- Each iteration assigns groups greedily. If the greedy order paints a
  group into a corner, a budgeted backtracking search (MRV ordering) can
  withdraw earlier choices. The acceptance suite includes data where only
  backtracking succeeds.
- `:infeasible` means the complete search space was exhausted (the group
  constraint graph is not K-colorable into K non-empty clusters).
  `:budget_exhausted` means the `search_budget` node limit was reached
  first; `unassigned_indices` then lists every record without a cluster,
  so no data item is hidden.
