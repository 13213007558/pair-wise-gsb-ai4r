# Reinforcement Learning

The `Ai4r::Reinforcement` module provides basic agents for Markov decision processes.

## Q-Learning

`Ai4r::Reinforcement::QLearning` implements the classic tabular algorithm.

### Parameters

* `learning_rate` – update step size (default `0.1`).
* `discount` – discount factor for future rewards (default `0.9`).
* `exploration` – probability of choosing a random action (default `0.1`).

```ruby
require 'ai4r/reinforcement/q_learning'

agent = Ai4r::Reinforcement::QLearning.new
agent.update(:s1, :a, 1, :s2)
agent.choose_action(:s1)
```

See `examples/reinforcement/q_learning_example.rb` for a runnable script.

## Prioritized Sweeping

`Ai4r::Reinforcement::PrioritizedSweeping` extends tabular Q-learning with a
learned sample model and a priority queue, so a reward discovered at the goal
sweeps backwards along already observed predecessors instead of waiting for
future visits.

* Each observation performs one *interaction update* and stores the outcome in
  the model. The same state-action pair may accumulate different successors
  and rewards, which are averaged by their observed counts.
* `update(..., planning_budget: n)` lets the caller decide how many
  *planning updates* (model-based) may run per observation; `plan(n)` spends
  more budget later. Interaction and planning updates are counted separately
  in `interaction_updates` and `planning_updates`.
* Terminal states never bootstrap, and only actions actually observed from a
  state take part in its value maximum.
* Queue entries whose priority became stale after model changes are discarded
  without consuming planning budget.

### Parameters

* `learning_rate` – update step size (default `0.1`).
* `discount` – discount factor for future rewards (default `0.9`).
* `exploration` – probability of choosing a random action (default `0.1`).
* `threshold` – minimum priority to keep a pair queued (default `1e-6`).

```ruby
require 'ai4r/reinforcement/prioritized_sweeping'

agent = Ai4r::Reinforcement::PrioritizedSweeping.new
agent.update(:s1, :go, 1, :goal, done: true, planning_budget: 10)
agent.plan(5)                       # resume planning later
agent.choose_action(:s1, random: Random.new(42))
```

## Policy Iteration

`Ai4r::Reinforcement::PolicyIteration` alternates policy evaluation and improvement.
Supply state lists, possible actions, transition probabilities and rewards.

### Parameters

* `discount` – discount factor for future rewards (default `0.9`).

```ruby
require 'ai4r/reinforcement/policy_iteration'

pi = Ai4r::Reinforcement::PolicyIteration.new
policy = pi.policy_iteration(states, actions, transition, reward)
```

For algorithms based on random playouts see [Monte Carlo Tree Search](monte_carlo_tree_search.md).
For classical graph search techniques see [Search Algorithms](search_algorithms.md).
