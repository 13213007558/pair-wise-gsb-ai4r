# frozen_string_literal: true

require_relative '../../test_helper'
require 'ai4r/search'

class AraStarUnitTest < Minitest::Test
  include Ai4r::Search

  # Directed graph. The inflated search first commits to T -> S -> G
  # (cost 20); the optimum T -> B -> S -> G costs 15 and reopens S once
  # B is expanded in the next iteration.
  #
  # The heuristic is admissible but deliberately inconsistent:
  #   h(B) = 8 > 1 + h(S) = 1.
  # Under epsilon = 3 B is heavily inflated, so the search expands the
  # expensive direct route to S first; later B reopens the closed S.
  GRAPH = {
    'T' => { 'S' => 10, 'B' => 4 },
    'S' => { 'G' => 10 },
    'B' => { 'S' => 1 }
  }.freeze

  HEURISTIC = { 'T' => 10, 'S' => 0, 'B' => 8, 'G' => 0 }.freeze

  def neighbor_fn(node)
    GRAPH[node] || {}
  end

  def heuristic_fn(node)
    HEURISTIC.fetch(node, 0)
  end

  def goal?(node)
    node == 'G'
  end

  def new_session(epsilon = 3.0)
    AraStar.new('T', method(:goal?), method(:neighbor_fn),
                method(:heuristic_fn), epsilon)
  end

  def test_runs_to_the_optimal_route
    status = new_session.run
    assert status.finished
    assert_equal %w[T B S G], status.path
    assert_float_equal 15.0, status.cost
    assert_float_equal 1.0, status.epsilon
    assert_float_equal 1.0, status.bound
  end

  def test_first_route_is_suboptimal_and_only_improves
    session = new_session

    status = session.advance(1)
    refute status.finished
    assert_nil status.path
    assert_nil status.cost
    assert_nil status.bound, 'no bound may be reported before a route exists'

    status = session.advance(2)
    assert_equal %w[T S G], status.path
    assert_float_equal 20.0, status.cost
    assert status.bound >= 1.0
    assert status.bound > 1.0

    status = session.advance(1)
    assert_equal %w[T S G], status.path
    assert_float_equal 20.0, status.cost

    costs = [status.cost]
    bounds = [status.bound]
    loop do
      status = session.advance(1)
      costs << status.cost if status.cost
      bounds << status.bound if status.bound
      break if status.finished
    end

    assert_equal %w[T B S G], status.path
    assert_float_equal 15.0, status.cost
    assert costs.each_cons(2).all? { |before, after| after <= before },
           'route costs must never worsen'
    assert_includes costs, 20.0
    assert bounds.each_cons(2).all? { |before, after| after <= before + 1e-9 },
           'bounds must tighten'
  end

  def test_paused_session_resumes_to_same_optimum
    uninterrupted = new_session.run

    paused = new_session
    early = paused.advance(3)
    refute early.finished
    assert_equal %w[T S G], early.path
    assert_float_equal 20.0, early.cost

    final = nil
    final = paused.advance(1) until final&.finished

    assert_equal uninterrupted.path, final.path
    assert_float_equal uninterrupted.cost, final.cost
    assert_float_equal uninterrupted.bound, final.bound
  end

  def test_chunking_does_not_change_the_result
    reference = new_session
    reference_status = reference.advance(100)

    chunked = new_session
    status = nil
    status = chunked.advance(1) until status&.finished
    assert_float_equal reference_status.cost, status.cost
    assert_equal reference_status.path, status.path

    alternate = new_session
    status = alternate.advance(2)
    until status.finished
      status = alternate.advance(3)
    end
    assert_float_equal reference_status.cost, status.cost
    assert_equal reference_status.path, status.path
  end

  def test_reopens_a_node_improved_after_it_was_closed
    session = new_session
    session.advance(3) # T, S and G expanded: incumbent T-S-G is recorded
    session.advance(1) # B expands and offers a cheaper route to S

    g_values = session.instance_variable_get(:@g)
    assert_float_equal 5.0, g_values['S']
    assert_includes session.instance_variable_get(:@frontier), 'S'

    status = session.run
    assert_equal %w[T B S G], status.path
  end

  def test_unreachable_goal_finishes_without_a_route_or_bound
    neighbors = { 'A' => { 'X' => 5 }, 'X' => {} }
    session = AraStar.new('A', method(:goal?),
                          ->(n) { neighbors[n] || {} },
                          method(:heuristic_fn), 2.0)
    status = session.run
    assert status.finished
    assert_nil status.path
    assert_nil status.cost
    assert_nil status.bound
    assert_float_equal 1.0, status.epsilon
  end

  def test_start_is_the_goal
    session = AraStar.new('G', method(:goal?), method(:neighbor_fn),
                          method(:heuristic_fn), 3.0)
    status = session.advance(5)
    assert status.finished
    assert_equal ['G'], status.path
    assert_float_equal 0.0, status.cost
    assert_float_equal 1.0, status.bound
  end

  def test_zero_budget_keeps_state
    session = new_session
    first = session.advance(0)
    second = session.advance(0)
    assert_nil first.path
    assert_nil second.path
    refute first.finished
  end

  def test_epsilon_one_matches_direct_optimal_search
    status = new_session(1.0).run
    assert status.finished
    assert_equal %w[T B S G], status.path
    assert_float_equal 15.0, status.cost
  end

  def test_rejects_invalid_arguments
    assert_raises(ArgumentError) { new_session(0.5) }
    session = new_session
    assert_raises(ArgumentError) { session.advance(-1) }
  end
end
