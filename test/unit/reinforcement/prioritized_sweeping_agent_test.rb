# frozen_string_literal: true

require_relative '../../test_helper'
require 'ai4r/reinforcement/prioritized_sweeping_agent'

class PrioritizedSweepingAgentUnitTest < Minitest::Test
  include Ai4r::Reinforcement

  # Seven non-terminal corridor states s0..s6 leading into a terminal goal.
  CORRIDOR_STATES = (0..6).map { |i| :"s#{i}" }.freeze

  def make_agent(seed: 1, learning_rate: 1.0, discount: 1.0, threshold: 0.0)
    agent = PrioritizedSweepingAgent.new(rng: Random.new(seed))
    agent.set_parameters(learning_rate: learning_rate, discount: discount,
                         exploration: 0.0, threshold: threshold)
    CORRIDOR_STATES.each { |state| agent.declare_available_actions(state, [:right]) }
    agent.declare_available_actions(:goal, [])
    agent
  end

  # Walk the agent through the corridor once, observing the reward only at the
  # final terminal transition. No planning happens while walking.
  def walk_corridor(agent)
    CORRIDOR_STATES.each_cons(2) do |state, next_state|
      agent.update(state, :right, 0, next_state,
                   available_actions: [:right], planning_budget: 0)
    end
    agent.update(:s6, :right, 1, :goal, terminal: true, planning_budget: 0)
  end

  def test_interaction_and_planning_updates_are_counted_separately
    agent = make_agent
    walk_corridor(agent)

    assert_equal 7, agent.interaction_updates
    assert_equal 0, agent.planning_updates

    swept = agent.plan(4)
    assert_equal 7, agent.interaction_updates
    assert_equal 4, agent.planning_updates
    assert_equal 4, swept.length
  end

  def test_planning_budget_can_be_attached_to_an_observation
    agent = make_agent
    CORRIDOR_STATES.each_cons(2) do |state, next_state|
      agent.update(state, :right, 0, next_state,
                   available_actions: [:right], planning_budget: 0)
    end

    agent.update(:s6, :right, 1, :goal, terminal: true, planning_budget: 3)

    assert_equal 7, agent.interaction_updates
    assert_equal 3, agent.planning_updates
  end

  def test_reward_propagates_backwards_through_observed_predecessors
    agent = make_agent
    walk_corridor(agent)

    assert_equal [%i[s6 right], %i[s5 right], %i[s4 right], %i[s3 right]],
                 agent.plan(4)
    assert_equal 1.0, agent.q[:s6][:right]
    assert_equal 1.0, agent.q[:s5][:right]
    assert_equal 1.0, agent.q[:s4][:right]
    assert_equal 1.0, agent.q[:s3][:right]
    # States not reached by the sweep yet are still untouched.
    CORRIDOR_STATES[0..2].each do |state|
      assert_equal 0.0, agent.q[state][:right]
    end
    # Only the immediate next predecessor waits; s1 and s0 enter the queue
    # once the wave reaches them.
    assert_equal 1, agent.pending_queue_size
  end

  def test_budget_exhaustion_leaves_partial_propagation_then_converges
    agent = make_agent
    walk_corridor(agent)

    agent.plan(4)
    assert_equal 1, agent.pending_queue_size
    assert_in_delta 0.0, agent.q[:s0][:right], 1e-9

    rest = agent.plan(3)
    assert_equal [%i[s2 right], %i[s1 right], %i[s0 right]], rest
    CORRIDOR_STATES.each do |state|
      assert_in_delta 1.0, agent.q[state][:right], 1e-9
    end
    assert_equal 0, agent.queue_size

    # Once the queue is drained, extra planning has nothing left to sweep.
    assert_empty agent.plan(10)
    assert_equal 7, agent.planning_updates
  end

  def test_terminal_states_are_not_bootstrapped
    agent = make_agent
    # Poison the terminal state's table: its declared action set is empty,
    # so this value must never enter the max during bootstrap backups.
    agent.q[:goal][:impossible] = 100.0
    walk_corridor(agent)

    agent.plan(7)

    assert_equal 0.0, agent.value(:goal)
    assert_in_delta 1.0, agent.q[:s6][:right], 1e-9
  end

  def test_unavailable_actions_do_not_participate_in_the_maximum
    agent = make_agent
    agent.declare_available_actions(:s0, [:right])
    agent.q[:s0][:forbidden] = 50.0
    agent.q[:s0][:right] = 0.25

    assert_in_delta 0.25, agent.value(:s0), 1e-9
    assert_equal :right, agent.choose_action(:s0)
  end

  def test_model_keeps_multiple_stochastic_outcomes
    agent = build_fork_agent

    outcomes = agent.observed_outcomes(:s0, :go)
    counts = {}
    outcomes.each { |outcome| counts[outcome[:next_state]] = outcome[:count] }

    assert_equal 2, outcomes.length
    assert_equal 11, counts[:s1]
    assert_equal 9, counts[:win]
    win = outcomes.find { |outcome| outcome[:next_state] == :win }
    assert_equal Array.new(9, 1.0), win[:rewards]
    detour = outcomes.find { |outcome| outcome[:next_state] == :s1 }
    assert(detour[:rewards].all?(&:zero?))
  end

  def test_stochastic_fork_propagates_in_priority_order_then_converges
    agent = build_fork_agent

    assert_equal [%i[s0 go], %i[s1 back], %i[s0 go]], agent.plan(3)
    assert_equal 1, agent.pending_queue_size
    assert_equal 3, agent.planning_updates
    assert_in_delta 0.5052174173702969, agent.q[:s0][:go], 1e-9
    assert_in_delta 0.37511657851615116, agent.q[:s1][:back], 1e-9

    # Continue planning after the small budget was exhausted: sample backups
    # keep cycling through the observed s0 <-> s1 predecessors and move both
    # values toward their Bellman fixed points (s0 ~= 0.840, s1 ~= 0.756).
    agent.plan(3000)
    assert(agent.q[:s0][:go] > 0.5052)
    assert(agent.q[:s1][:back] > 0.3751)
    assert_in_delta 0.7295214430984736, agent.q[:s0][:go], 1e-6
    assert_in_delta 0.6822821196909513, agent.q[:s1][:back], 1e-6
  end

  def test_stochastic_sampling_uses_the_external_random
    first  = run_fork_with_seed(4242)
    second = run_fork_with_seed(4242)
    other  = run_fork_with_seed(9999)

    assert_equal first, second
    refute_equal first, other
  end

  def test_stale_priorities_are_discarded_without_spending_budget
    agent = make_agent(learning_rate: 0.5, threshold: 0.05)
    # The first observation enqueues a residual of 1.0. The second identical
    # observation arrives while that entry is still queued: its smaller
    # residual must not replace the pending one. Popping the pending entry
    # processes it exactly once, and the stale heap copy is discarded without
    # spending any further planning budget.
    agent.update(:s6, :right, 1, :goal, terminal: true,
                                        available_actions: [], planning_budget: 0)
    agent.update(:s6, :right, 1, :goal, terminal: true,
                                        planning_budget: 0)
    # The smaller second residual does not replace the already pending entry.
    assert_equal 1, agent.queue_size
    assert_equal 1, agent.pending_queue_size

    swept = agent.plan(20)

    assert_equal [%i[s6 right]], swept
    assert_equal 1, agent.planning_updates
    assert_equal 0, agent.pending_queue_size
  end

  def test_sub_threshold_entries_are_skipped_without_spending_budget
    agent = make_agent(learning_rate: 0.5, threshold: 0.2)
    # Real observations drive decreasing interaction residuals.
    agent.update(:s6, :right, 1, :goal, terminal: true, planning_budget: 0)
    agent.update(:s6, :right, 1, :goal, terminal: true, planning_budget: 0)
    agent.update(:s6, :right, 1, :goal, terminal: true, planning_budget: 0)
    assert_equal 1, agent.pending_queue_size

    # One planning backup processes the largest pending residual.
    assert_equal 1, agent.plan(20).length

    # Further observations only produce residuals under the threshold; they
    # never enter the queue and cannot spend planning budget.
    3.times do
      agent.update(:s6, :right, 1, :goal, terminal: true, planning_budget: 0)
    end
    assert_equal 0, agent.pending_queue_size
    assert_empty agent.plan(10)
    assert_equal 1, agent.planning_updates
  end

  private

  # Small stochastic fork with a cycle: from s0, :go either wins immediately
  # or reaches s1, whose :back action returns to s0. The model keeps both
  # outcomes, and repeated planning keeps propagating through predecessors.
  def build_fork_agent
    agent = PrioritizedSweepingAgent.new(rng: Random.new(2024))
    agent.set_parameters(learning_rate: 0.2, discount: 0.9,
                         exploration: 0.0, threshold: 1e-9)
    agent.declare_available_actions(:s0, [:go])
    agent.declare_available_actions(:s1, [:back])
    agent.declare_available_actions(:win, [])

    environment = Random.new(77)
    20.times do
      if environment.rand < 0.5
        agent.update(:s0, :go, 1, :win, terminal: true,
                                        available_actions: [], planning_budget: 0)
      else
        agent.update(:s0, :go, 0, :s1,
                     available_actions: [:back], planning_budget: 0)
        agent.update(:s1, :back, 0, :s0,
                     available_actions: [:go], planning_budget: 0)
      end
    end
    agent
  end

  def run_fork_with_seed(seed)
    agent = PrioritizedSweepingAgent.new(rng: Random.new(seed))
    agent.set_parameters(learning_rate: 0.2, discount: 0.9, threshold: 1e-9)
    agent.declare_available_actions(:s0, [:go])
    agent.declare_available_actions(:s1, [:back])
    agent.declare_available_actions(:win, [])
    environment = Random.new(seed)
    6.times do
      if environment.rand < 0.5
        agent.update(:s0, :go, 1, :win, terminal: true,
                                        available_actions: [], planning_budget: 0)
      else
        agent.update(:s0, :go, 0, :s1,
                     available_actions: [:back], planning_budget: 0)
        agent.update(:s1, :back, 0, :s0,
                     available_actions: [:go], planning_budget: 0)
      end
    end
    agent.plan(5).map(&:last) + [agent.q[:s0][:go], agent.q[:s1][:back]]
  end
end
