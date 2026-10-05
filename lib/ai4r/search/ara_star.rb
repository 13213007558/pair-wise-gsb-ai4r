# frozen_string_literal: true

# Author::    OpenAI Assistant
# License::   MPL 1.1
# Project::   ai4r

require 'set'

module Ai4r
  module Search
    # Incremental Anytime Repairing A* (ARA*) search session.
    #
    # A weighted A* search is run with an inflated heuristic weight
    # +epsilon+ (>= 1).  As soon as a goal node is expanded, the current
    # route is returned with a suboptimality certificate: its true cost
    # is provably within a factor of +epsilon+ of the optimum.  Each later
    # iteration lowers +epsilon+ towards 1, reusing the path costs
    # (g-values) and the search frontier (OPEN + INCONS) already computed,
    # and reopening expanded nodes when a cheaper route to them is found.
    # When +epsilon+ reaches 1 the route is optimal.
    #
    # The session is advanced manually with #advance using a budget of
    # node expansions, so it can be paused and resumed without threads or
    # timers.  Edge costs must be non-negative and the heuristic must be
    # admissible; it does not need to be consistent because improved nodes
    # are always reopened.
    #
    # Example:
    #   neighbors = { 'A' => { 'B' => 1 }, 'B' => { 'G' => 1 } }
    #   session = Ai4r::Search::AraStar.new(
    #     'A', ->(n) { n == 'G' },
    #     ->(n) { neighbors[n] || {} },
    #     ->(n) { n == 'G' ? 0 : 1 },
    #     3.0
    #   )
    #   status = session.advance(10) # budget of ten node expansions
    #   status.path     # ['A', 'B', 'G']
    #   status.cost     # 2.0
    #   status.bound    # provable factor, nil before the first route
    #   status.finished # true once optimality is proved
    class AraStar
      # Snapshot returned by every #advance call.
      #
      # +path+::     best route found so far, or nil before the first one
      # +cost+::     true accumulated edge cost of +path+
      # +bound+::    provable suboptimality factor; +cost/bound+ is a
      #             lower bound on the optimum, so cost <= bound * C*.
      #             Nil while no route exists yet: no bound is invented.
      # +finished+:: true once the optimum is proved, or once the graph
      #             has been exhausted and proved to have no route
      # +epsilon+::  inflation weight of the current/last iteration,
      #             monotonically decreasing to 1
      Status = Struct.new(:path, :cost, :bound, :finished, :epsilon)

      EPSILON_TOLERANCE = 1e-9
      COST_TOLERANCE = 1e-12

      # @param start [Object] initial node
      # @param goal_test [Proc] predicate returning true for a goal node
      # @param neighbor_fn [Proc] -> node { neighbor => non_negative_cost }
      # @param heuristic_fn [Proc] -> node => admissible remaining cost
      # @param initial_epsilon [Numeric] first inflation factor (>= 1)
      def initialize(start, goal_test, neighbor_fn, heuristic_fn,
                     initial_epsilon = 3.0)
        raise ArgumentError, 'initial_epsilon must be >= 1' if initial_epsilon < 1

        @start = start
        @goal_test = goal_test
        @neighbor_fn = neighbor_fn
        @heuristic_fn = heuristic_fn
        @epsilon = initial_epsilon.to_f

        @g = { start => 0.0 }
        @bp = {}
        @h = { start => heuristic_fn.call(start).to_f }
        @frontier = Set[start]
        @closed = Set.new
        @heap = nil
        @generation = 0
        @seq = 0
        @incumbent = nil
        @incumbent_cost = nil
        @certified = nil
        @finished = false
      end

      # Expand up to +expansions+ nodes and return the current Status.
      #
      # Expansions left over after an iteration is proved are spent on
      # the next, tighter iteration.  Advancing a finished session is a
      # no-op.  Expanding a goal node counts as one expansion.
      def advance(expansions = 1)
        raise ArgumentError, 'expansions must be non-negative' if expansions.negative?
        return status if @finished

        if @goal_test.call(@start)
          record_incumbent(@start)
          @epsilon = 1.0
          @certified = 1.0
          @frontier.clear
          @finished = true
          return status
        end

        setup_iteration if @heap.nil?
        remaining = expansions
        while remaining.positive?
          entry = pop_frontier
          if entry.nil?
            if @incumbent_cost.nil?
              finish_without_route
            else
              close_proven_iteration
            end
            break if @finished
          elsif @goal_test.call(entry.node)
            record_incumbent(entry.node)
            remaining -= 1
            close_proven_iteration
          else
            remaining -= 1
            expand(entry.node)
          end
          break if @finished || remaining.zero?

          setup_iteration if @heap.nil?
        end
        status
      end

      # Keep advancing until the session is finished and return the final
      # Status.
      def run
        advance(1) until @finished
        status
      end

      private

      def expand(node)
        @neighbor_fn.call(node).each do |neighbor, edge_cost|
          @h[neighbor] ||= @heuristic_fn.call(neighbor).to_f
          tentative = @g[node] + edge_cost.to_f
          next if !@g[neighbor].nil? && tentative >= @g[neighbor] - COST_TOLERANCE

          @g[neighbor] = tentative
          @bp[neighbor] = node
          # Improved nodes are reopened even if already expanded in this
          # iteration; this keeps the search correct for admissible but
          # inconsistent heuristics.
          @closed.delete(neighbor)
          @frontier.add(neighbor)
          enqueue(neighbor)
        end
      end

      # Rebuild the open heap from the persistent frontier (OPEN and
      # INCONS nodes carried over from previous iterations).
      def setup_iteration
        @closed = Set.new
        @generation += 1
        @heap = MinHeap.new
        @frontier.each { |node| enqueue(node) }
      end

      # A goal has been expanded, or the heap of a solvable search ran
      # dry: certify the incumbent and lower the inflation weight.
      def close_proven_iteration
        f_min = frontier_f_min
        if f_min.nil? || f_min <= COST_TOLERANCE
          @epsilon = 1.0
          @certified = 1.0
          @finished = true
          return
        end

        ratio = @incumbent_cost / f_min
        @epsilon = [[1.0, ratio].max, @epsilon].min
        if ratio <= 1.0 + EPSILON_TOLERANCE || @epsilon <= 1.0 + EPSILON_TOLERANCE
          # incumbent_cost <= f_min certifies the incumbent as optimal.
          @epsilon = 1.0
          @certified = 1.0
          @finished = true
          return
        end
        @certified = @epsilon
        @heap = nil
      end

      def finish_without_route
        @epsilon = 1.0
        @finished = true
      end

      def pop_frontier
        loop do
          entry = @heap.pop
          return nil if entry.nil?
          next unless entry.generation == @generation
          next if @closed.include?(entry.node)
          next unless (entry.pushed_g - @g[entry.node]).abs <= COST_TOLERANCE

          @closed.add(entry.node)
          @frontier.delete(entry.node)
          return entry
        end
      end

      def enqueue(node)
        @seq += 1
        key = @g[node] + @epsilon * @h[node]
        @heap.push(HeapEntry.new(key, @seq, @generation, node, @g[node]))
      end

      def record_incumbent(node)
        @incumbent_cost = @g[node]
        @incumbent = reconstruct_path(node)
      end

      def reconstruct_path(node)
        path = [node]
        while @bp.key?(node)
          node = @bp[node]
          path.unshift(node)
        end
        path
      end

      def status
        bound = nil
        if @incumbent_cost && @finished
          bound = 1.0
        elsif @incumbent_cost
          f_min = frontier_f_min
          bound = if f_min.nil? || f_min <= COST_TOLERANCE
                    @certified
                  else
                    # f_min is an admissible lower bound on every route still
                    # reachable through the frontier, so cost / f_min is itself a
                    # valid certificate.  A factor is never reported below 1.
                    [@certified, [1.0, @incumbent_cost / f_min].max].min
                  end
        end
        Status.new(@incumbent&.dup, @incumbent_cost, bound, @finished, @epsilon)
      end

      def frontier_f_min
        return nil if @frontier.empty?

        @frontier.map { |node| @g[node] + @h[node] }.min
      end

      HeapEntry = Struct.new(:key, :seq, :generation, :node, :pushed_g)

      # Binary min-heap ordered by inflated f-value, then by insertion
      # sequence so ties expand deterministically.
      class MinHeap
        def initialize
          @data = []
        end

        def push(entry)
          @data << entry
          sift_up(@data.size - 1)
        end

        def pop
          return nil if @data.empty?

          root = @data.first
          last = @data.pop
          unless @data.empty?
            @data[0] = last
            sift_down(0)
          end
          root
        end

        private

        def sift_up(index)
          while index.positive?
            parent = (index - 1) / 2
            break unless earlier?(@data[index], @data[parent])

            @data[index], @data[parent] = @data[parent], @data[index]
            index = parent
          end
        end

        def sift_down(index)
          loop do
            left = index * 2 + 1
            break if left >= @data.size

            right = left + 1
            child = if right < @data.size && earlier?(@data[right], @data[left])
                      right
                    else
                      left
                    end
            break unless earlier?(@data[child], @data[index])

            @data[index], @data[child] = @data[child], @data[index]
            index = child
          end
        end

        def earlier?(a, b)
          a.key < b.key || (a.key == b.key && a.seq < b.seq)
        end
      end
    end
  end
end
