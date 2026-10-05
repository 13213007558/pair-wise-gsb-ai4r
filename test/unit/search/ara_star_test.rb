# frozen_string_literal: true

require_relative '../../test_helper'
require 'ai4r/search'

class AraStarSessionUnitTest < Minitest::Test
  include Ai4r::Search

  # Weighted search first commits to the inflated S -> B -> X -> G route
  # (cost 8). Later A -> D -> X reaches the already expanded X more cheaply
  # (g 7 -> 3), forcing X into INCONS and back into OPEN. The repaired route
  # S -> A -> D -> X -> G costs 4. The heuristic is admissible but the edge
  # A -> D violates consistency: h(A) = 5 > 1 + h(D) = 3.
  GRAPH = {
    'S' => { 'A' => 1, 'B' => 1 },
    'A' => { 'G' => 9, 'D' => 1 },
    'D' => { 'X' => 1 },
    'B' => { 'X' => 6 },
    'X' => { 'G' => 1 },
    'G' => {}
  }.freeze

  HEURISTIC = {
    'S' => 0.0, 'A' => 5.0, 'B' => 0.0, 'D' => 2.0,
    'X' => 1.0, 'G' => 0.0
  }.freeze

  def neighbor_fn(node)
    GRAPH[node] || {}
  end

  def heuristic_fn(node)
    HEURISTIC[node] || 0.0
  end

  def goal_test(node)
    node == 'G'
  end

  def new_session(**options)
    defaults = { initial_epsilon: 3.0, epsilon_step: 1.0 }
    AraStarSession.new(
      'S', method(:goal_test), method(:neighbor_fn), method(:heuristic_fn),
      **defaults.merge(options)
    )
  end

  def test_returns_no_bound_or_route_before_a_route_exists
    session = new_session
    result = session.advance(2)

    assert_nil result.path
    assert_nil result.cost
    assert_nil result.suboptimality_bound
    refute result.complete?
  end

  def test_first_route_is_suboptimal_while_epsilon_stays_above_one
    session = new_session
    result = session.advance(4)

    assert_equal %w[S B X G], result.path
    assert_equal 8, result.cost
    assert_in_delta 4.0 / 3.0, result.suboptimality_bound, 1e-9
    assert_in_delta 2.0, session.epsilon, 1e-9
    refute result.complete?
  end

  def test_inconsistent_node_is_reopened_and_repairs_the_route
    session = new_session
    session.advance(4)
    assert_equal %w[S B X G], session.path

    result = session.advance(10)

    assert_equal %w[S A D X G], result.path
    assert_equal 4, result.cost
    assert_in_delta 1.0, result.suboptimality_bound, 1e-9
    assert result.complete?
  end

  def test_cheaper_path_to_a_closed_node_goes_through_inconsistent_set
    session = new_session
    session.advance(4)

    # X was expanded in the first round with g = 7 via B. Lowering epsilon
    # and advancing through A -> D reopens it with g = 3.
    session.advance(2)
    inconsistent = session.instance_variable_get(:@inconsistent)
    open = session.instance_variable_get(:@open)
    g = session.instance_variable_get(:@g)

    assert(inconsistent.include?('X') || open.include?('X'))
    assert_equal 3, g['X']
  end

  def test_mid_round_pause_keeps_the_last_proven_bound
    session = new_session
    session.advance(4)
    proven_bound = session.suboptimality_bound

    # One expansion into the next round must not fabricate a tighter bound
    # before the round proves one.
    session.advance(1)
    assert_in_delta proven_bound, session.suboptimality_bound, 1e-12
  end

  def test_paused_and_resumed_search_matches_uninterrupted_run
    paused = new_session
    costs = []
    bounds = []
    20.times do
      result = paused.advance(1)
      next if result.cost.nil?

      costs << result.cost
      bounds << result.suboptimality_bound
      assert result.cost <= costs.first
    end

    uninterrupted = new_session.advance(100)

    assert_equal uninterrupted.path, paused.path
    assert_equal uninterrupted.cost, paused.cost
    assert_equal 4, paused.cost
    assert paused.complete?
    assert_equal costs, costs.sort.reverse
    assert(bounds.each_cons(2).all? { |earlier, later| later <= earlier + 1e-12 })
  end

  def test_intermediate_routes_only_improve
    session = new_session
    seen_costs = []
    20.times do
      result = session.advance(1)
      seen_costs << result.cost unless result.cost.nil?
    end

    assert_equal [8, 4], seen_costs.uniq
  end

  def test_zero_expansion_budget_keeps_state_unchanged
    session = new_session
    result = session.advance(0)

    assert_nil result.path
    assert_equal 0, session.expansions
  end

  def test_reuses_accumulated_costs_and_frontier
    session = new_session
    session.advance(2)
    assert_equal 2, session.expansions

    session.advance(2)
    assert_equal 4, session.expansions
  end

  def test_reports_exhaustion_when_goal_is_unreachable
    graph = { 'S' => { 'A' => 1 }, 'A' => {} }
    session = AraStarSession.new(
      'S', method(:goal_test), ->(node) { graph[node] || {} }, ->(_) { 0 }
    )
    result = session.advance(10)

    assert_nil result.path
    assert_nil result.cost
    assert_nil result.suboptimality_bound
    assert result.complete?
  end

  def test_rejects_negative_edge_costs
    bad_neighbors = ->(node) { node == 'S' ? { 'A' => -1 } : {} }
    session = AraStarSession.new(
      'S', method(:goal_test), bad_neighbors, ->(_) { 0 }
    )

    assert_raises(ArgumentError) { session.advance(1) }
  end

  def test_rejects_invalid_initial_epsilon
    assert_raises(ArgumentError) do
      new_session(initial_epsilon: 0.5)
    end
  end

  def test_result_snapshot_is_queryable_without_advancing
    session = new_session
    session.advance(4)

    assert_equal %w[S B X G], session.path
    assert_equal 8, session.cost
    assert_in_delta 4.0 / 3.0, session.suboptimality_bound, 1e-9
    refute session.complete?
  end
end
