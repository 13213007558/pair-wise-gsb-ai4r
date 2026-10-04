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

`Ai4r::Reinforcement::PrioritizedSweepingAgent` extends Q-learning with a
learned MDP model. Every observed transition is stored, together with its
predecessors. Each observation still performs exactly one Q-learning backup,
but the resulting temporal-difference error is pushed onto a priority queue.
The caller grants a planning budget per observation; the agent then sweeps the
largest residuals first and propagates them backwards through already observed
predecessors, so a reward discovered at the end of a maze can update earlier
states immediately instead of waiting for them to be revisited.

### Parameters

* `learning_rate` – update step size (default `0.1`).
* `discount` – discount factor for future rewards (default `0.9`).
* `exploration` – probability of choosing a random action (default `0.1`).
* `threshold` – residuals at or below this value are never queued (default `0.0`).

The constructor accepts an external `Random` (`rng:`) used both for
exploration and for sampling stochastic model outcomes, which keeps experiments
reproducible. The model stores every distinct `(next_state, terminal)` outcome
with its observed counts and rewards, so the same state-action can lead to
different successors. Terminal transitions are not bootstrapped. Declare the
actions available in each state with `declare_available_actions` (pass an
empty array for terminal states); only declared actions participate in the
maximum and in action selection.

```ruby
require 'ai4r/reinforcement/prioritized_sweeping_agent'

agent = Ai4r::Reinforcement::PrioritizedSweepingAgent.new(rng: Random.new(7))
agent.set_parameters(learning_rate: 0.5, discount: 1.0)
agent.declare_available_actions(:s1, [:right])
agent.declare_available_actions(:goal, [])

# One real observation, followed by up to 10 model-based backups.
agent.update(:s1, :right, 1, :goal, terminal: true, planning_budget: 10)
agent.interaction_updates # => 1
agent.planning_updates    # => 1

# Planning can also be driven later, returning the swept state-actions.
agent.plan(5)
```

`queue_size` reports physical heap entries, `pending_queue_size` reports the
state-actions still waiting to be swept, and `observed_outcomes` exposes the
stored stochastic model. Stale heap copies left behind by promoted priorities
or by model changes are discarded on pop without consuming planning budget.

See `examples/reinforcement/prioritized_sweeping_example.rb` for a runnable
corridor demo.

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
