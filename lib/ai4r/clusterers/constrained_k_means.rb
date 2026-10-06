# frozen_string_literal: true

# Author::    ai4r contributors
# License::   MPL 1.1
# Project::   ai4r
# Url::       https://github.com/SergioFierens/ai4r
#
# You can redistribute it and/or modify it under the terms of
# the Mozilla Public License version 1.1  as published by the
# Mozilla Foundation at http://www.mozilla.org/MPL/MPL-1.1.txt

require_relative 'k_means'

module Ai4r
  module Clusterers
    # Raised before any iteration starts when a cannot-link constraint
    # contradicts the transitive closure of the must-link constraints.
    # Every reported conflict is independently verifiable through the
    # must-link chain stored in +conflicts+.
    class ConstraintConflict < ArgumentError
      attr_reader :conflicts

      def initialize(conflicts)
        @conflicts = conflicts
        first = conflicts.first
        pair = first[:pair].inspect
        chain = first[:must_link_path].inspect
        super("cannot-link constraint #{pair} conflicts with must-link chain #{chain}")
      end
    end

    # Outcome of a constrained clustering run. Every original data item is
    # accounted for: +assignments[i]+ is the 0-based cluster of item +i+, or
    # +nil+ when no feasible full assignment was found; such indices are
    # always also listed in +unassigned_indices+.
    class ConstrainedResult
      attr_reader :assignments, :unassigned_indices, :clusters, :centroids,
                  :objective_value, :end_reason, :iterations,
                  :number_of_clusters, :must_link_groups, :stats

      def initialize(attributes)
        attributes.each { |key, value| instance_variable_set("@#{key}", value) }
      end

      alias labels assignments
      alias sse objective_value

      def fully_assigned?
        @unassigned_indices.empty?
      end
    end

    # K-means with instance-level must-link / cannot-link constraints.
    #
    # Must-link is treated as an equivalence relation: the transitive
    # closure is computed up front and every equivalence group is placed
    # into one cluster as a whole. All group members participate in centroid
    # computation, so group size moves centroids. A cannot-link pair whose
    # endpoints are already connected by a must-link chain is a direct
    # contradiction and raises ConstraintConflict before iterating.
    #
    # Allocation never runs ordinary k-means first and moves violating
    # points afterwards. Each iteration starts with a deterministic greedy
    # placement over must-link groups (cheapest feasible cluster); when it
    # paints itself into a corner, a budgeted backtracking search takes
    # over and may undo earlier greedy choices. The reported end reason
    # distinguishes a proven infeasible instance (:infeasible) from a search
    # that simply spent its budget (:budget_exhausted).
    class ConstrainedKMeans < KMeans
      END_REASONS = %i[
        converged
        max_iterations
        infeasible
        budget_exhausted
        empty_cluster
      ].freeze

      parameters_info(
        search_budget: 'Maximum number of group placements the ' \
                       'backtracking feasibility search may try per ' \
                       'allocation pass. nil means unbounded; defaults to ' \
                       '10000. A search that exhausts the tree within ' \
                       'budget proves :infeasible, otherwise the run ends ' \
                       'with :budget_exhausted.'
      )

      attr_reader :assignments, :objective_value, :end_reason, :must_link_groups

      def initialize
        super
        @search_budget = 10_000
        @last_stats = {}
      end

      # Build a clustering that respects the given constraints.
      # @param data_set [Ai4r::Data::DataSet] data items to cluster
      # @param number_of_clusters [Integer] desired number of clusters (k)
      # @param must_link [Array<Array(Integer, Integer)>] item index pairs
      #   forced into the same cluster
      # @param cannot_link [Array<Array(Integer, Integer)>] item index pairs
      #   forbidden from sharing a cluster
      # @return [ConstrainedResult] run outcome; the clusterer also exposes
      #   #clusters, #centroids, #assignments, #objective_value and
      #   #end_reason for the same run.
      def build_constrained(data_set, number_of_clusters, must_link: [], cannot_link: [])
        @data_set = data_set
        @number_of_clusters = number_of_clusters
        @items = data_set.data_items
        validate_constraints(must_link, cannot_link)

        @must_link_groups, @must_link_adjacency = build_must_link_groups(must_link)
        detect_direct_conflicts(cannot_link)
        @group_forbidden = build_group_forbidden(cannot_link)
        @relocation_count = 0
        @relocation_nodes = 0
        @last_stats = {}

        @rng = @random_seed.nil? ? Random.new : Random.new(@random_seed)
        @iterations = 0
        @history = [] if @track_history
        calc_initial_centroids

        run_iterations
        finalize_result
      end

      protected

      def validate_constraints(must_link, cannot_link)
        raise ArgumentError, 'Data set has no data items' if @items.empty?
        raise ArgumentError, 'Number of clusters larger than data items' if @number_of_clusters > @items.length
        raise ArgumentError, 'Invalid value for on_empty' unless @on_empty == 'eliminate' || @on_empty == 'terminate' || @on_empty == 'random' || @on_empty == 'outlier'

        valid_indices = @centroid_indices.all? { |i| i.is_a?(Integer) && i >= 0 && i < @items.length }
        distinct_items = valid_indices &&
                         @centroid_indices.map { |i| @items[i] }.uniq.length == @number_of_clusters
        unless @centroid_indices.empty? ||
               (valid_indices &&
                @centroid_indices.length == @number_of_clusters &&
                @centroid_indices.uniq.length == @centroid_indices.length &&
                distinct_items)
          raise ArgumentError,
                'centroid_indices must provide one distinct data item per cluster for constrained clustering'
        end

        (must_link + cannot_link).each do |pair|
          raise ArgumentError, "Constraint pair must contain two integer indices: #{pair.inspect}" unless pair.is_a?(Array) && pair.length == 2 && pair.all? { |i| i.is_a?(Integer) }

          pair.each do |index|
            next if index >= 0 && index < @items.length

            raise ArgumentError, "Constraint index #{index} out of range (0...#{@items.length})"
          end
        end
      end

      # Union-find over must-link edges. Returns the equivalence groups
      # (sorted by smallest member) and the adjacency list used to report
      # verifiable must-link chains on conflicts.
      def build_must_link_groups(must_link)
        parent = (0...@items.length).to_a
        rank = Array.new(@items.length, 0)
        find_root = lambda do |x|
          root = x
          root = parent[root] while parent[root] != root
          while parent[x] != root
            next_parent = parent[x]
            parent[x] = root
            x = next_parent
          end
          root
        end
        must_link.each do |a, b|
          root_a = find_root.call(a)
          root_b = find_root.call(b)
          next if root_a == root_b

          if rank[root_a] < rank[root_b]
            parent[root_a] = root_b
          elsif rank[root_a] > rank[root_b]
            parent[root_b] = root_a
          else
            parent[root_b] = root_a
            rank[root_a] += 1
          end
        end
        grouped = Hash.new { |hash, key| hash[key] = [] }
        @items.each_index { |i| grouped[find_root.call(i)] << i }

        adjacency = Array.new(@items.length) { [] }
        must_link.each do |a, b|
          next if a == b

          adjacency[a] << b unless adjacency[a].include?(b)
          adjacency[b] << a unless adjacency[b].include?(a)
        end
        [grouped.values.sort_by(&:first), adjacency]
      end

      # cannot-link pairs inside one must-link equivalence group are direct
      # contradictions. All of them (with their must-link chain) are reported
      # before any iteration runs.
      def detect_direct_conflicts(cannot_link)
        group_of = Hash.new { |hash, index| hash[index] = @must_link_groups.find_index { |group| group.include?(index) } }
        conflicts = cannot_link.filter_map do |a, b|
          next unless a == b || group_of[a] == group_of[b]

          { pair: [a, b], must_link_path: a == b ? [a] : must_link_path(a, b) }
        end
        raise ConstraintConflict, conflicts unless conflicts.empty?
      end

      # BFS over must-link edges, returning the chain from a to b.
      def must_link_path(a, b)
        previous = { a => nil }
        queue = [a]
        until queue.empty?
          node = queue.shift
          break if node == b

          @must_link_adjacency[node].each do |neighbor|
            next if previous.key?(neighbor)

            previous[neighbor] = node
            queue << neighbor
          end
        end
        path = []
        cursor = b
        while cursor
          path << cursor
          cursor = previous[cursor]
        end
        path.reverse
      end

      # Translate item-level cannot-link edges to forbidden group pairs.
      def build_group_forbidden(cannot_link)
        group_index = Array.new(@items.length)
        @must_link_groups.each_with_index { |group, gi| group.each { |i| group_index[i] = gi } }
        forbidden = Array.new(@must_link_groups.length) { Set.new }
        cannot_link.each do |a, b|
          ga = group_index[a]
          gb = group_index[b]
          next if ga == gb

          forbidden[ga] << gb
          forbidden[gb] << ga
        end
        forbidden
      end

      def run_iterations
        @end_reason = nil
        @latest_placement = {}
        loop do
          costs = group_costs
          status, placement, stats = allocate_groups(costs)
          status, placement = handle_empty_clusters(status, placement) unless status == :budget
          @last_stats = stats
          @latest_placement = placement || @latest_placement
          if status == :budget
            @end_reason = :budget_exhausted
            break
          end
          if END_REASONS.include?(status)
            @end_reason = status
            break
          end

          @iterations += 1
          @old_centroids = @centroids
          materialize_clusters(placement)
          @centroids = @centroid_function.call(@clusters)
          snapshot_history(placement)
          if @old_centroids == @centroids
            @end_reason = :converged
            break
          end
          if @max_iterations && @max_iterations <= @iterations
            @end_reason = :max_iterations
            break
          end
        end
        @run_iterations ||= (@terminal_status == :budget ? :budget_exhausted : @terminal_status)
      end

      # One allocation pass: canonical greedy first, budgeted backtracking
      # (which may retract early greedy choices) on failure.
      def allocate_groups(costs)
        greedy_status, placement, greedy_failed = canonical_greedy(costs)
        nodes_visited = 0
        if greedy_status != :solved
          search_status, placement, nodes_visited = feasibility_search(costs, force_full_cover: false)
          greedy_status = search_status
        end
        stats = { greedy_failed: greedy_failed, search_nodes_visited: nodes_visited,
                  groups: @must_link_groups.length }
        [greedy_status, placement, stats]
      end

      # Cost of placing each whole group on each centroid. Summing over all
      # members makes the group size part of the allocation decision.
      def group_costs
        @must_link_groups.map do |group|
          (0...@number_of_clusters).map do |cluster|
            group.sum { |i| distance(@items[i], @centroids[cluster]) }
          end
        end
      end

      # Greedy pass in canonical group order: cheapest currently feasible
      # cluster. Returns a partial placement when it gets stuck, so the
      # search never mistakes its failure for infeasibility.
      def canonical_greedy(costs)
        placement = {}
        @must_link_groups.each_index do |group|
          allowed = feasible_clusters(group, placement)
          return [:partial, placement, true] if allowed.empty?

          cluster = allowed.min_by { |c| costs[group][c] }
          placement[group] = cluster
        end
        [:solved, placement, false]
      end

      # Clusters on which +group+ may be placed given the partial placement:
      # no already-placed cannot-link group is there.
      def feasible_clusters(group, placement)
        blocked = Set.new
        @group_forbidden[group].each do |other|
          blocked << placement[other] if placement.key?(other)
        end
        (0...@number_of_clusters).reject { |cluster| blocked.include?(cluster) }
      end

      # Depth-first feasibility search with most-constrained-group (MRV)
      # ordering. Exhausting the whole tree proves infeasibility; exceeding
      # search_budget placements reports :budget instead. When
      # force_full_cover is true every cluster must receive a group, which
      # is only used while repairing empty clusters.
      def feasibility_search(costs, force_full_cover:)
        @search_nodes_visited = 0
        placement = {}
        outcome = dfs_place(placement, costs, force_full_cover)
        [outcome, placement, @search_nodes_visited]
      end

      # Recursive backtracking step. +placement+ is mutated while descending
      # and restored after each branch, so siblings never see stale state.
      def dfs_place(placement, costs, force_full_cover)
        if placement.length == @must_link_groups.length
          covered_all = placement.values.uniq.length == @number_of_clusters
          return !force_full_cover || covered_all ? :solved : :infeasible
        end

        chosen = choose_group(placement, costs)
        return :infeasible if chosen.nil?

        group, allowed = chosen
        allowed.each do |cluster|
          @search_nodes_visited += 1
          return :budget if @search_budget && @search_nodes_visited > @search_budget

          placement[group] = cluster
          result = dfs_place(placement, costs, force_full_cover)
          if result == :solved
            return :solved
          elsif result == :budget
            return :budget
          end

          placement.delete(group)
        end
        :infeasible
      end

      # MRV variable selection: group with fewest feasible clusters, ties by
      # group index for determinism. Returns nil if some group has no option.
      def choose_group(placement, costs)
        remaining = @must_link_groups.each_index.reject { |group| placement.key?(group) }
        chosen = nil
        allowed_for_chosen = nil
        remaining.each do |group|
          allowed = feasible_clusters(group, placement)
          return nil if allowed.empty?

          allowed.sort_by! { |cluster| [costs[group][cluster], cluster] }
          if chosen.nil? || allowed.length < allowed_for_chosen.length
            chosen = group
            allowed_for_chosen = allowed
          end
        end
        [chosen, allowed_for_chosen]
      end

      # Apply the configured on_empty policy while keeping every constraint.
      def handle_empty_clusters(status, placement)
        return [status, placement] unless status == :solved

        used = placement.values.uniq
        empty = (0...@number_of_clusters).reject { |cluster| used.include?(cluster) }
        return [:ok, placement] if empty.empty?

        case @on_empty
        when 'terminate'
          [:empty_cluster, placement]
        when 'eliminate'
          eliminate_empty_clusters(placement)
          [:ok, placement]
        else
          relocate_empty_clusters(empty, placement)
        end
      end

      # Drop empty clusters and remap every assigned item to the new indices.
      def eliminate_empty_clusters(placement)
        used = placement.values.uniq.sort
        remap = {}
        used.each_with_index { |cluster, new_index| remap[cluster] = new_index }
        @centroids = used.map { |cluster| @centroids[cluster] }
        @number_of_clusters = @centroids.length
        placement.transform_values! { |cluster| remap[cluster] }
      end

      # Relocate empty centroid slots to unused data items ('random' /
      # 'outlier') and re-run the constrained allocation. Relocation is
      # itself constraint aware: if no reallocation covers every cluster,
      # the instance is reported infeasible; budget overruns are reported
      # separately.
      def relocate_empty_clusters(empty_clusters, placement)
        new_centroid_indices = relocation_candidates(empty_clusters, placement)
        return [:infeasible, placement] if new_centroid_indices.length < empty_clusters.length

        attempt_centroids = @centroids.dup
        empty_clusters.each_with_index do |cluster, position|
          attempt_centroids[cluster] = @items[new_centroid_indices[position]]
        end
        costs = group_costs_with_centroids(attempt_centroids)
        status, new_placement, nodes = feasibility_search(costs, force_full_cover: true)
        return [:budget, nil] if status == :budget
        return [:infeasible, placement] if status == :infeasible

        @centroids = attempt_centroids
        @relocation_nodes = (@relocation_nodes || 0) + nodes
        @relocation_count = (@relocation_count || 0) + 1
        [:ok, new_placement]
      end

      # Candidate data-item indices to fill the empty slots. 'outlier' tries
      # items furthest from their assigned centroid first; 'random' shuffles
      # deterministically. Each candidate index identifies the must-link
      # group that must be moved onto the empty slot; items already used as
      # centroids are skipped.
      def relocation_candidates(empty_clusters, placement)
        centroid_items = @centroids
        available = @items.each_index.reject { |i| centroid_items.include?(@items[i]) }
        ordered = if @on_empty == 'outlier'
                    group_of_item = Array.new(@items.length)
                    placement.each do |group, cluster|
                      @must_link_groups[group].each { |i| group_of_item[i] = cluster }
                    end
                    available.sort_by { |i| [-distance(@items[i], @centroids[group_of_item[i]]), -i] }
                  else
                    available.shuffle(random: @rng)
                  end
        ordered.first(empty_clusters.length)
      end

      def group_costs_with_centroids(centroids)
        @must_link_groups.map do |group|
          (0...centroids.length).map do |cluster|
            group.sum { |i| distance(@items[i], centroids[cluster]) }
          end
        end
      end

      # Build the public cluster objects from a group placement.
      def materialize_clusters(placement)
        @clusters = Array.new(@number_of_clusters) do
          Ai4r::Data::DataSet.new data_labels: @data_set.data_labels
        end
        @cluster_indices = Array.new(@number_of_clusters) { [] }
        @assignments = Array.new(@items.length)
        placement.each do |group, cluster|
          @must_link_groups[group].each do |item_index|
            @clusters[cluster] << @items[item_index]
            @cluster_indices[cluster] << item_index
            @assignments[item_index] = cluster
          end
        end
      end

      def snapshot_history(placement)
        return unless @track_history

        @history << {
          centroids: @centroids.collect(&:dup),
          assignments: assignments_from_placement(placement),
          must_link_groups: @must_link_groups
        }
      end

      def assignments_from_placement(placement)
        labels = Array.new(@items.length)
        placement.each do |group, cluster|
          @must_link_groups[group].each { |i| labels[i] = cluster }
        end
        labels
      end

      # Freeze the run into clusterer state and a ConstrainedResult. No
      # sample is hidden: indices without a cluster appear both as nil in
      # assignments and in unassigned_indices.
      def finalize_result
        final_placement = @latest_placement
        materialize_clusters(final_placement)
        unassigned = (0...@items.length).select { |i| @assignments[i].nil? }
        @objective_value = unassigned.empty? ? sse : nil
        stats = @last_stats.merge(
          relocations: @relocation_count || 0,
          relocation_search_nodes: @relocation_nodes || 0
        )

        result = ConstrainedResult.new(
          assignments: @assignments.dup,
          unassigned_indices: unassigned,
          clusters: @clusters,
          centroids: @centroids,
          objective_value: @objective_value,
          end_reason: @end_reason,
          iterations: @iterations,
          number_of_clusters: @number_of_clusters,
          must_link_groups: @must_link_groups,
          stats: stats
        )
        @result = result
      end
    end
  end
end
