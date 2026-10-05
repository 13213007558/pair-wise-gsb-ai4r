# Anytime Repairing A* (ARA*)

`Ai4r::Search::AraStarSession` is an anytime variant of A*. It returns a
feasible route quickly using an inflated heuristic, then spends any remaining
node-expansion budget tightening the inflation factor toward 1 and repairing
the route until it is proven optimal.

The search is driven explicitly by a node-expansion budget. There are no
threads or timers: the caller decides when to advance and can pause between
any two expansions.

See [A* Search](a_star_search.md) for the one-shot A* implementation and
[Search Algorithms](search_algorithms.md) for BFS and DFS.

## Creating a session

The constructor takes the same callbacks as `AStar`, plus an optional
inflation schedule:

```ruby
require 'ai4r/search'

graph = {
  'S' => { 'A' => 1, 'B' => 1 },
  'A' => { 'G' => 9 },
  'B' => { 'G' => 4 },
  'G' => {}
}
heuristic = { 'S' => 4, 'A' => 9, 'B' => 4, 'G' => 0 }

session = Ai4r::Search::AraStarSession.new(
  'S',
  ->(node) { node == 'G' },
  ->(node) { graph[node] || {} },
  ->(node) { heuristic[node] || 0 },
  initial_epsilon: 3.0, # start with a strongly inflated heuristic
  epsilon_step: 1.0    # remove this much inflation after each round
)
```

Edge weights must be non-negative. The heuristic must be admissible; it does
not need to be consistent.

## Advancing by an expansion budget

`advance(expansions)` performs at most that many node expansions and returns a
`Result` with:

* `path` — current best route, or `nil` while no route has been found.
* `cost` — true edge cost of the current route.
* `suboptimality_bound` — proven ratio guarantee (`cost / optimum` is no
  larger than this); `nil` until the first route exists, so no bound is ever
  fabricated.
* `complete?` — true once the route is proven optimal, or when the reachable
  search space has been exhausted without finding a goal.

```ruby
result = session.advance(20)
result.path                  # => ['S', 'B', 'G']
result.cost                  # => 5
result.suboptimality_bound   # => 1.25
result.complete?             # => false

# ... later, resume with whatever budget is available
result = session.advance(50)
result.cost                  # => 5
result.complete?             # => true
```

The current route, cost, bound and completion flag are also queryable without
advancing via `path`, `cost`, `suboptimality_bound` and `complete?`.

## Pausing, resuming and repairing

All accumulated `g` values, backpointers, the `OPEN` frontier and the
inconsistent (`INCONS`) set survive between calls. When a node that was already
expanded in an earlier round is later reached by a cheaper path (which can
happen with admissible but inconsistent heuristics), it is moved to the
inconsistent set and reconsidered in the next round. Successive routes are
therefore always at least as good as earlier ones, and a session paused after
every expansion finishes with the same optimal cost as an uninterrupted run.

The inflation factor is available as `epsilon` and decreases by
`epsilon_step` after each completed round until it reaches 1 and optimality
is proven.
