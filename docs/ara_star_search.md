# ARA* Search (Anytime Repairing A*)

Ai4r::Search::AraStar is an anytime variant of [A*](a_star_search.md).
It first returns a route quickly using an inflated heuristic, then spends any
remaining node-expansion budget proving tighter suboptimality bounds and
improving the route until it is optimal. Unlike a plain A* run, the search is
exposed as a **session** that the caller advances manually: there are no
threads or timers, and pausing between calls loses no work.

Edge weights must be non-negative and the heuristic must be admissible; it
does not need to be consistent. Whenever a cheaper way to reach an already
expanded node is found, that node is reopened, so a consistent heuristic is
never required.

## Usage

~~~ruby
require 'ai4r/search'

graph = {
  'T' => { 'S' => 10, 'B' => 4 },
  'S' => { 'G' => 10 },
  'B' => { 'S' => 1 }
}
heuristic = { 'T' => 10, 'S' => 0, 'B' => 8, 'G' => 0 }

session = Ai4r::Search::AraStar.new(
  'T',
  ->(node) { node == 'G' },
  ->(node) { graph[node] || {} },
  ->(node) { heuristic.fetch(node, 0) },
  3.0 # initial inflation factor epsilon
)

# Spend a budget of two node expansions.
status = session.advance(2)
status.path     # ['T', 'S', 'G']  - first feasible route
status.cost     # 20.0              - its true cost
status.bound    # 1.67              - cost is within 1.67x of optimum
status.epsilon  # 1.67              - current inflation weight
status.finished # false

# Later, spend more budget. Costs and bounds are reused across calls.
status = session.advance(10)
status.path     # ['T', 'B', 'S', 'G']
status.cost     # 15.0
status.bound    # 1.0
status.finished # true
~~~

## Status fields

Each call to +advance+ (or the convenience runner +run+) returns an
+AraStar::Status+:

- +path+ - best route found so far as an array of nodes, or +nil+.
- +cost+ - true accumulated edge cost of +path+ (the goal g-value).
- +bound+ - the best suboptimality factor that can currently be proved:
  +cost / bound+ is a lower bound on the optimum, so +cost <= bound * C*+.
  It is +nil+ until the first route exists; no bound is fabricated before a
  solution is found. It reaches exactly +1.0+ once the optimum is proved.
- +finished+ - +true+ once the route is provably optimal, or once the graph
  has been exhausted without reaching the goal.
- +epsilon+ - inflation weight of the current/last weighted iteration,
  monotonically decreasing from the initial value toward +1+.

## How pausing works

+advance(expansions)+ expands at most +expansions+ nodes and returns
immediately. When an iteration proves a route before the budget is spent,
the remaining expansions are used for the next, tighter iteration inside the
same call. Between calls the session keeps:

- the best g-value and back-pointer known for every node,
- the persistent frontier of nodes that may still improve a route,
- the incumbent route and the last proved suboptimality factor.

Resuming a paused session therefore produces exactly the same final result as
letting the search run without interruption, and intermediate routes can
only improve (their cost never increases). To run to completion in one go,
call +run+ or keep calling +advance+ until +finished+.

## When to use it

Reach for ARA* when a usable route is needed quickly but more time may be
available, for example interactive route planning or bounded-time agents.
When only the optimum matters and a single blocking call is acceptable,
plain [A*](a_star_search.md) is simpler; when no informative heuristic is
available, start from [BFS/DFS](search_algorithms.md).
