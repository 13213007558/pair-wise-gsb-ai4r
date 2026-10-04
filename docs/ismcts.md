# Information Set Monte Carlo Tree Search

`Ai4r::Search::ISMCTS` implements single-observer Information Set MCTS for
games with hidden information (e.g. card games), where plain
[Monte Carlo Tree Search](monte_carlo_tree_search.md) would need the
opponent's hidden cards to make a decision.

Instead of searching over full game states, the tree is built over
*information sets* — everything the observing player is allowed to know.
Each iteration samples one complete state (a determinization) consistent
with the observer's information, and the statistics of all samples mapping
to the same information set are merged into a single node.

```ruby
require 'ai4r/search'

ismcts = Ai4r::Search::ISMCTS.new(
  info_set_fn: ->(s) { s[:public] },
  sample_fn: ->(info_set) { deal_consistent_with(info_set) },
  actions_fn: ->(s) { legal_actions(s) },
  transition_fn: ->(s, a) { apply(s, a) },
  terminal_fn: ->(s) { s[:over] },
  reward_fn: ->(s) { s[:my_score] }
)

best = ismcts.search(current_state, 100)
ismcts.statistics
# => { bet: { visits: 61, value: 30.5, mean: 0.5, availability: 100 }, ... }
```

The callbacks are:

* `info_set_fn.call(state)` – public information-set identifier. Hidden
  components of the state must not influence the returned value.
* `sample_fn.call(info_set)` – samples a full state consistent with the
  observer's information. This is the only source of hidden information.
* `actions_fn.call(state)` – legal actions for a sampled state. An action
  may be legal in only some samples; the search tracks an availability
  count per action and only selects among actions legal in the current
  sample.
* `transition_fn.call(state, action)` – computes the next state.
* `terminal_fn.call(state)` – whether the state is terminal.
* `reward_fn.call(state)` – payoff of a terminal state, always from the
  designated (observing) player's perspective. The search never negates
  or reinterprets it.

Guarantees:

* Tree nodes store only the public information-set identifier, the
  incoming action and aggregate statistics — never a full state — so
  hidden information cannot leak through stored node state.
* The root state, sampled states and any object returned by the callbacks
  are never mutated by the search.
* Pass `random: Random.new(seed)` for reproducible searches: identical
  public information and identical sample sequences always yield identical
  root decisions, regardless of the true hidden state.
