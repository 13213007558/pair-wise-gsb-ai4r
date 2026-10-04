# frozen_string_literal: true

require_relative '../../test_helper'
require 'ai4r/reinforcement/prioritized_sweeping'

class PrioritizedSweepingTest < Minitest::Test
  include Ai4r::Reinforcement

  def build_agent
    agent = PrioritizedSweeping.new
    agent.set_parameters(learning_rate: 1.0, discount: 1.0, exploration: 0.0,
                         threshold: 1e-9)
    agent
  end

  # Walk a deterministic corridor s0 -> s1 -> ... -> s4 with no reward,
  # then discover reward 1 at the goal. The reward must sweep backwards
  # one predecessor per planning update.
  def build_corridor_agent
    agent = build_agent
    4.times do |i|
      agent.update(:"s#{i}", :forward, 0, :"s#{i + 1}", planning_budget: 0)
    end
    agent
  end

  def test_reward_propagates_backwards_in_priority_order
    agent = build_corridor_agent
    assert_equal 4, agent.interaction_updates
    assert_equal 0, agent.planning_updates

    updated = agent.update(:s4, :forward, 1, :goal, done: true, planning_budget: 2)

    assert_equal [[:s3, :forward], [:s2, :forward]], updated
    assert_equal 5, agent.interaction_updates
    assert_equal 2, agent.planning_updates
    assert_in_delta 1.0, agent.q[:s4][:forward], 1e-9
    assert_in_delta 1.0, agent.q[:s3][:forward], 1e-9
    assert_in_delta 1.0, agent.q[:s2][:forward], 1e-9
  end

  def test_budget_exhaustion_leaves_pending_work_resumable
    agent = build_corridor_agent
    agent.update(:s4, :forward, 1, :goal, done: true, planning_budget: 2)

    # Budget ran out: s1 and s0 are still untouched, work remains queued.
    assert_in_delta 0.0, agent.q[:s1][:forward], 1e-9
    assert_in_delta 0.0, agent.q[:s0][:forward], 1e-9
    assert_equal 1, agent.queue_size

    # Continuing planning finishes the sweep without new observations.
    updated = agent.plan(5)
    assert_equal [[:s1, :forward], [:s0, :forward]], updated
    assert_equal 4, agent.planning_updates
    assert_equal 0, agent.queue_size
    5.times do |i|
      assert_in_delta 1.0, agent.q[:"s#{i}"][:forward], 1e-9
    end
  end

  def test_stochastic_model_averages_successors_and_rewards
    agent = build_agent
    rng = Random.new(42)
    counts = { good: 0, bad: 0 }
    20.times do
      if rng.rand < 0.75
        agent.update(:start, :go, 1, :good, done: true, planning_budget: 1)
        counts[:good] += 1
      else
        agent.update(:start, :go, 0, :bad, done: true, planning_budget: 1)
        counts[:bad] += 1
      end
    end

    # The same state-action pair keeps both observed outcomes.
    assert_equal counts[:good], agent.model[:start][:go][[:good, 1]]
    assert_equal counts[:bad], agent.model[:start][:go][[:bad, 0]]
    # Planning drives Q towards the model's expected value.
    expected = counts[:good] / 20.0
    assert_in_delta expected, agent.q[:start][:go], 1e-9
  end

  def test_terminal_state_does_not_bootstrap
    agent = build_agent
    agent.update(:mid, :side, 0, :elsewhere, planning_budget: 0)
    agent.q[:mid][:side] = 5.0

    agent.update(:start, :go, 1, :mid, done: true, planning_budget: 0)

    # :mid is terminal, so its Q-values must not bootstrap :start.
    assert_in_delta 1.0, agent.q[:start][:go], 1e-9
  end

  def test_unavailable_actions_do_not_participate_in_max
    agent = build_agent
    agent.update(:mid, :known, 0, :elsewhere, planning_budget: 0)
    agent.q[:mid][:known] = 1.0
    agent.q[:mid][:ghost] = 100.0 # never observed from :mid

    agent.update(:start, :go, 0, :mid, planning_budget: 1)

    assert_in_delta 1.0, agent.q[:start][:go], 1e-9
  end

  def test_stale_queue_entries_do_not_consume_budget
    agent = build_agent
    agent.update(:s0, :go, 0, :s1, planning_budget: 0)
    agent.update(:s1, :go, 1, :s2, done: true, planning_budget: 0)
    assert_equal 1, agent.queue_size

    # A newer observation pulls Q(s0, :go) up to date, making the queued
    # entry irrelevant before any planning happens.
    agent.update(:s0, :go, 0, :s1, planning_budget: 0)
    assert_in_delta 1.0, agent.q[:s0][:go], 1e-9

    updated = agent.plan(5)
    assert_equal [], updated
    assert_equal 0, agent.planning_updates
    assert_equal 0, agent.queue_size
  end

  def test_choose_action_is_greedy_without_exploration
    agent = build_agent
    agent.update(:s, :a, 0, :t, planning_budget: 0)
    agent.update(:s, :b, 0, :t, planning_budget: 0)
    agent.q[:s][:b] = 2.0

    assert_equal :b, agent.choose_action(:s, random: Random.new(1))
    assert_nil agent.choose_action(:nowhere, random: Random.new(1))
  end

  def test_choose_action_accepts_external_random
    agent = build_agent
    agent.set_parameters(exploration: 1.0)
    agent.update(:s, :a, 0, :t, planning_budget: 0)
    agent.update(:s, :b, 0, :t, planning_budget: 0)

    rng1 = Random.new(1234)
    rng2 = Random.new(1234)
    first = 10.times.map { agent.choose_action(:s, random: rng1) }
    second = 10.times.map { agent.choose_action(:s, random: rng2) }
    assert_equal first, second
    assert(first.all? { |action| %i[a b].include?(action) })
  end
end
