# frozen_string_literal: true

# Author::    OpenAI Assistant
# License::   MPL 1.1
# Project::   ai4r

require 'set'

module Ai4r
  module Search
    # Anytime Repairing A* (ARA*) driven by an explicit node-expansion budget.
    #
    # Each call to {#advance} performs at most +expansions+ node expansions and
    # returns the best route found so far. Between calls the session keeps its
    # accumulated +g+ values, backpointers and search frontier, so computation
    # can be paused and resumed cheaply. The heuristic inflation factor starts
    # at +initial_epsilon+ and is lowered toward 1 after each search round,
    # reusing the work already paid for.
    #
    # Edge weights must be non-negative and the heuristic must be admissible
    # (it may be inconsistent). When a node that was already expanded is later
    # reached by a cheaper path, it is placed in an inconsistent set and is
    # reconsidered in the next round, so improved routes repair earlier
    # decisions.
    #
    # Callbacks follow {AStar}: +neighbor_fn+ returns a hash of
    # +neighbor => edge_cost+ pairs and +heuristic_fn+ estimates the remaining
    # cost to the goal.
    class AraStarSession
      INFINITY = Float::INFINITY
      private_constant :INFINITY

      # Snapshot returned by every {AraStarSession#advance} call.
      Result = Struct.new(:path, :cost, :suboptimality_bound, :complete) do
        def complete?
          complete
        end
      end

      OPTIMALITY_TOLERANCE = 1e-12
      private_constant :OPTIMALITY_TOLERANCE

      # @return [Numeric] inflation factor used by the current search round
      attr_reader :epsilon

      # @return [Integer] total node expansions performed so far
      attr_reader :expansions

      # @param start [Object] initial node
      # @param goal_test [Proc] predicate returning true when a node is a goal
      # @param neighbor_fn [Proc] -> node { neighbor => cost, ... }
      # @param heuristic_fn [Proc] -> node => admissible estimated remaining cost
      # @param initial_epsilon [Numeric] starting inflation factor (>= 1)
      # @param epsilon_step [Numeric] maximum inflation removed after each round
      def initialize(start, goal_test, neighbor_fn, heuristic_fn,
                     initial_epsilon: 3.0, epsilon_step: 1.0)
        raise ArgumentError, 'initial_epsilon must be >= 1' if initial_epsilon < 1
        raise ArgumentError, 'epsilon_step must be > 0' if epsilon_step <= 0

        @start = start
        @goal_test = goal_test
        @neighbor_fn = neighbor_fn
        @heuristic_fn = heuristic_fn
        @epsilon_step = epsilon_step

        @epsilon = initial_epsilon.to_f
        @expansions = 0

        @g = { start => 0 }
        @came_from = {}
        @f = { start => heuristic(start) }
        @open = [start]
        @closed = Set.new
        @inconsistent = Set.new

        @incumbent_node = nil
        @incumbent_cost = nil
        @incumbent_path = nil
        @best_bound = nil
        @round_active = false
        @complete = false
      end

      # Continue the search, expanding at most +expansions+ nodes.
      #
      # The session automatically finishes a search round and lowers the
      # inflation factor whenever a round ends, using any leftover budget to
      # keep improving the route.
      #
      # @param expansions [Integer] maximum number of node expansions
      # @return [Result] current route, its true cost, the provable
      #   suboptimality bound (nil until a route exists), and completion flag
      def advance(expansions = 1)
        raise ArgumentError, 'expansions must be non-negative' if expansions.negative?

        budget = expansions
        until @complete || budget.zero?
          start_round unless @round_active
          budget = run_round(budget)
          finish_round unless @round_active
        end
        result
      end

      # @return [Array<Object>, nil] best route found so far, or nil
      def path
        @incumbent_path&.dup
      end

      # @return [Numeric, nil] true edge cost of the current route, or nil
      def cost
        @incumbent_cost
      end

      # @return [Numeric, nil] proven suboptimality bound (cost / optimal is no
      #   larger than this); nil while no route has been found yet
      def suboptimality_bound
        @best_bound
      end

      # @return [Boolean] true when the route is proven optimal or, when no
      #   route exists, the search space has been exhausted
      def complete?
        @complete
      end

      private

      def start_round
        @round_active = true
      end

      def run_round(budget)
        until budget.zero?
          if @open.empty?
            @round_active = false
            return budget
          end

          current = pop_open
          if stop_before_expansion?(current)
            @open << current
            @round_active = false
            return budget
          end
          if @goal_test.call(current)
            update_incumbent(current)
            @expansions += 1
            budget -= 1
            @round_active = false
            return budget
          end

          expand(current)
          @expansions += 1
          budget -= 1
        end
        0
      end

      def stop_before_expansion?(node)
        return false if @incumbent_cost.nil?

        cutoff = @incumbent_cost
        inflated_start = @epsilon * @f[@start]
        cutoff = inflated_start if inflated_start.positive? && inflated_start < cutoff
        weighted_f(node) >= cutoff - OPTIMALITY_TOLERANCE
      end

      def expand(node)
        @closed << node
        @neighbor_fn.call(node).each do |neighbor, edge_cost|
          validate_edge_cost(edge_cost)
          tentative_g = @g[node] + edge_cost
          next if @g.key?(neighbor) && tentative_g >= @g[neighbor]

          @g[neighbor] = tentative_g
          @came_from[neighbor] = node
          @f[neighbor] = tentative_g + heuristic(neighbor)

          if @closed.include?(neighbor)
            @open.delete(neighbor)
            @inconsistent << neighbor
          else
            @open << neighbor unless @open.include?(neighbor)
          end
        end
      end

      def update_incumbent(node)
        goal_cost = @g[node]
        return false if @incumbent_cost && goal_cost >= @incumbent_cost

        @incumbent_node = node
        @incumbent_cost = goal_cost
        @incumbent_path = reconstruct_path(node)
        true
      end

      def finish_round
        @round_active = false
        candidate = candidate_bound
        @inconsistent.each { |node| @open << node unless @open.include?(node) }
        @inconsistent.clear
        @closed.clear

        if candidate
          proven_bound = [@epsilon, candidate].min
          update_bound(proven_bound)
          @complete = proven_bound <= 1.0 + OPTIMALITY_TOLERANCE
          decrease_epsilon unless @complete
        elsif @open.empty?
          @complete = true
        else
          decrease_epsilon
        end
      end

      def decrease_epsilon
        candidate = @epsilon - @epsilon_step
        @epsilon = candidate <= 1.0 + OPTIMALITY_TOLERANCE ? 1.0 : candidate
      end

      def candidate_bound
        return nil unless @incumbent_cost

        frontier_value = frontier_f_value
        return 1.0 if frontier_value.infinite?

        [1.0, @incumbent_cost / frontier_value].max
      end

      def frontier_f_value
        values = @open.map { |node| @f[node] } +
                 @inconsistent.map { |node| @f[node] }
        values.min || INFINITY
      end

      def update_bound(bound)
        @best_bound = bound if @best_bound.nil? || bound < @best_bound
      end

      def pop_open
        selected = @open.min_by { |node| [weighted_f(node), @f[node], @g[node]] }
        @open.delete(selected)
        selected
      end

      def weighted_f(node)
        @g[node] + @epsilon * (@f[node] - @g[node])
      end

      def heuristic(node)
        @heuristic_fn.call(node).to_f
      end

      def validate_edge_cost(edge_cost)
        return unless edge_cost.negative?

        raise ArgumentError, "edge costs must be non-negative, got #{edge_cost}"
      end

      def reconstruct_path(node)
        [node].tap do |path|
          while @came_from.key?(node)
            node = @came_from[node]
            path.unshift(node)
          end
        end
      end

      def result
        Result.new(path, cost, suboptimality_bound, @complete)
      end
    end
  end
end
