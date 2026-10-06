# frozen_string_literal: true

# Author::    ai4r contributors
# License::   MPL 1.1
# Project::   ai4r
#
# You can redistribute it and/or modify it under the terms of
# the Mozilla Public License version 1.1 as published by the
# Mozilla Foundation at http://www.mozilla.org/MPL/MPL-1.1.txt

require_relative 'k_means'

module Ai4r
  module Clusterers
    # Raised before any iteration starts when a cannot-link pair is
    # already contradicted by the (transitively closed) must-link groups,
    # or when constraint indices refer to data items that do not exist.
    #
    # +conflicts+ lists every offending cannot-link pair as
    # <code>{ pair: [i, j], must_link_group: [..], type: :direct_conflict }</code>,
    # so callers can verify the contradiction against +must_link_groups+.
    class ConstraintConflict < ArgumentError
      attr_reader :conflicts, :must_link_groups

      def initialize(conflicts, must_link_groups)
        @conflicts = conflicts
        @must_link_groups = must_link_groups
        first = conflicts.first
        super("#{conflicts.length} constraint conflict(s) before iteration; first: #{first[:type]} #{first[:pair].inspect}")
      end
    end

    # Outcome of a constrained k-means run. Every original data item is
    # always represented: +assignments[i]+ is the cluster of item +i+, or
    # +nil+ when no complete assignment could be produced (nothing is
    # silently dropped).
    class ConstrainedKMeansResult
      attr_reader :assignments, :clusters, :centroids, :objective,
                  :reason, :iterations, :unassigned_indices,
                  :must_link_groups, :cannot_link, :conflicts,
                  :backtracks, :search_nodes_visited, :search_calls

      def initialize(assignments:, clusters:, centroids:, objective:, reason:,
                     iterations: 0, unassigned_indices: [], must_link_groups: [],
                     cannot_link: [], conflicts: [], backtracks: 0,
                     search_nodes_visited: 0, search_calls: 0)
        @assignments = assignments
        @clusters = clusters
        @centroids = centroids
        @objective = objective
        @reason = reason
        @iterations = iterations
        @unassigned_indices = unassigned_indices
        @must_link_groups = must_link_groups
        @cannot_link = cannot_link
        @conflicts = conflicts
        @backtracks = backtracks
        @search_nodes_visited = search_nodes_visited
        @search_calls = search_calls
      end

      # A complete, constraint-satisfying assignment into K non-empty
      # clusters. +:infeasible+ and +:budget_exhausted+ results return false.
      def complete?
        %i[converged max_iterations].include?(reason)
      end

      def to_h
        {
          assignments: @assignments,
          centroids: @centroids,
          objective: @objective,
          reason: @reason,
          iterations: @iterations,
          unassigned_indices: @unassigned_indices,
          must_link_groups: @must_link_groups,
          cannot_link: @cannot_link,
          conflicts: @conflicts,
          backtracks: @backtracks,
          search_nodes_visited: @search_nodes_visited,
          search_calls: @search_calls
        }
      end
    end

    # K-means with instance-level must-link / cannot-link constraints.
    #
    # Must-link pairs are closed transitively (union-find) *before* any
    # iteration: the resulting equivalence groups are the atomic units of
    # assignment, and every member of a group contributes to the same
    # centroid on every iteration. Cannot-link pairs forbid two groups from
    # sharing a cluster. A cannot-link pair contradicted by the must-link
    # closure raises ConstraintConflict before centroids are initialized.
    #
    # Assigning groups to clusters is a graph coloring problem. Each
    # iteration starts with a deterministic greedy assignment; if that
    # leaves a group without a legal cluster (or an empty cluster), a
    # budgeted backtracking search with MRV ordering looks for a feasible
    # assignment, allowing earlier greedy choices to be withdrawn. The run
    # distinguishes a proven impossibility (:infeasible, the whole search
    # tree was explored) from :budget_exhausted (the node budget ran out
    # first). Empty clusters are never silently eliminated: the same search
    # is required to produce K non-empty clusters, so repairs stay
    # constraint-aware.
    class ConstrainedKMeans < KMeans
      attr_reader :result, :must_link_groups, :group_index_by_item

      parameters_info(
        search_budget: 'Maximum number of group-assignment search nodes ' \
                       'explored each time the greedy assignment needs a ' \
                       'feasibility search. nil (default) means exhaustive; ' \
                       'an Integer limits the search so a run can terminate ' \
                       'with reason :budget_exhausted.',
        verify_feasibility: 'When true (default) the closed constraints are ' \
                            'checked for a K-cluster coloring before the ' \
                            'first iteration, using the same search budget. ' \
                            'Set false to skip the pre-iteration check.'
      )

      def initialize
        super
        @search_budget = nil
        @verify_feasibility = true
      end

      # Run constrained k-means and return a ConstrainedKMeansResult.
      # Unlike the ordinary entry point, this never performs an
      # unconstrained run: every assignment decision is made through the
      # constrained group assignment.
      #
      # @param data_set [Ai4r::Data::DataSet] data items to cluster
      # @param number_of_clusters [Integer] K; exactly K non-empty
      #   clusters are required for a complete result
      # @param must_link [Array<Array(Integer, Integer)>] pairs of item
      #   indices that must end up in the same cluster (transitive)
      # @param cannot_link [Array<Array(Integer, Integer)>] pairs of item
      #   indices that must end up in different clusters
      # @return [ConstrainedKMeansResult]
      def build_constrained(data_set, number_of_clusters, must_link: [], cannot_link: [])
        @data_set = data_set
        @number_of_clusters = number_of_clusters
        must_link = normalize_pairs(must_link, :must_link)
        cannot_link = normalize_pairs(cannot_link, :cannot_link)
        validate_inputs

        @must_link_groups = build_must_link_groups(must_link)
        @group_index_by_item = map_items_to_groups(@must_link_groups)
        @group_adjacency = build_group_adjacency(cannot_link)
        validate_no_direct_conflict(cannot_link)

        @backtracks = 0
        @search_nodes = 0
        @search_calls = 0
        @iterations = 0
        @rng = @random_seed.nil? ? Random.new : Random.new(@random_seed)

        if @verify_feasibility
          precheck = feasibility_precheck
          return precheck unless precheck.nil?
        end

        calc_initial_centroids
        loop do
          status, group_to_cluster, detail = assign_groups
          unless status == :feasible
            result = failed_result(status, detail)
            sync_state(result)
            return result
          end
          apply_assignment(group_to_cluster)
          recompute_centroids
          break if stop_criteria_met
        end

        reason = @old_centroids == @centroids ? :converged : :max_iterations
        result = success_result(reason)
        sync_state(result)
        result
      end

      protected

      def validate_inputs
        n = @data_set.data_items.length
        raise ArgumentError, 'Data set must not be empty.' if n.zero?
        return unless @number_of_clusters > n

        raise ArgumentError, 'Number of clusters larger than data items'
      end

      def normalize_pairs(pairs, kind)
        seen = Set.new
        n = @data_set.data_items.length
        Array(pairs).map do |pair|
          unless pair.is_a?(Array) && pair.length == 2 &&
                 pair.all? { |x| x.is_a?(Integer) }
            raise ArgumentError, "#{kind} pairs must be [Integer, Integer] pairs: #{pair.inspect}"
          end

          i, j = pair
          raise ArgumentError, "#{kind} pair #{pair.inspect} references an index outside 0..#{n - 1}" unless i.between?(0, n - 1) && j.between?(0, n - 1)
          raise ArgumentError, "#{kind} pairs must reference two distinct items: #{pair.inspect}" if i == j

          ordered = [i, j].sort
          seen.add?(ordered) ? ordered : nil
        end.compact
      end

      # Transitive closure of must-link pairs via union-find.
      def build_must_link_groups(must_link)
        n = @data_set.data_items.length
        parent = Array.new(n) { |i| i }
        find = lambda do |x|
          root = x
          root = parent[root] while parent[root] != root
          while parent[x] != x
            nxt = parent[x]
            parent[x] = root
            x = nxt
          end
          root
        end
        must_link.each do |i, j|
          ri = find.call(i)
          rj = find.call(j)
          parent[rj] = ri unless ri == rj
        end
        buckets = {}
        n.times { |i| (buckets[find.call(i)] ||= []) << i }
        buckets.values.map(&:sort).sort_by(&:first)
      end

      def map_items_to_groups(groups)
        mapping = Array.new(@data_set.data_items.length)
        groups.each_with_index { |group, gi| group.each { |i| mapping[i] = gi } }
        mapping
      end

      # cannot-link graph between must-link groups (sets of group indices).
      def build_group_adjacency(cannot_link)
        adjacency = Array.new(@must_link_groups.length) { Set.new }
        cannot_link.each do |i, j|
          gi = @group_index_by_item[i]
          gj = @group_index_by_item[j]
          next if gi == gj # handled as a direct conflict

          adjacency[gi] << gj
          adjacency[gj] << gi
        end
        adjacency
      end

      # A cannot-link pair whose endpoints belong to the same must-link
      # group is an outright contradiction, reported before iteration.
      def validate_no_direct_conflict(cannot_link)
        conflicts = cannot_link.filter_map do |i, j|
          next unless @group_index_by_item[i] == @group_index_by_item[j]

          group = @must_link_groups[@group_index_by_item[i]]
          { type: :direct_conflict, pair: [i, j], must_link_group: group }
        end
        return if conflicts.empty?

        raise ConstraintConflict.new(conflicts, @must_link_groups)
      end

      # Structural check independent of the current centroids: the group
      # graph must admit a K-coloring using every cluster at least once.
      # Returns nil when feasible, otherwise a failure result.
      def feasibility_precheck
        if @must_link_groups.length < @number_of_clusters
          detail = { type: :too_few_groups,
                     groups: @must_link_groups.length,
                     number_of_clusters: @number_of_clusters }
          return failed_result(:infeasible, detail)
        end
        status, _mapping, detail = color_groups(require_surjective: true)
        return nil if status == :feasible

        failed_result(status, detail || { type: :not_k_colorable })
      end

      # One constrained assignment pass against the current centroids.
      # Greedy first; on failure, a budgeted search may withdraw earlier
      # choices. Returns [status, group_to_cluster_or_nil, detail_or_nil].
      def assign_groups
        greedy = greedy_assignment
        return greedy if greedy.first == :feasible

        @backtracks += 1
        status, mapping, detail = color_groups(require_surjective: true)
        [status, mapping, detail || greedy.last]
      end

      # Deterministic greedy pass: groups in data order, each to its
      # cheapest cluster not forbidden by an already placed neighbor.
      def greedy_assignment
        mapping = Array.new(@must_link_groups.length)
        used = Array.new(@number_of_clusters, false)
        @must_link_groups.each_with_index do |_group, gi|
          forbidden = neighbor_clusters(gi, mapping)
          options = (0...@number_of_clusters).reject { |c| forbidden.include?(c) }
          if options.empty?
            detail = { type: :greedy_blocked, group: gi,
                       members: @must_link_groups[gi],
                       forbidden_by: forbidden_groups(gi, mapping) }
            return [:infeasible, nil, detail]
          end
          chosen = options.min_by { |c| group_cluster_cost(gi, c) }
          mapping[gi] = chosen
          used[chosen] = true
        end
        empty = used.each_index.reject { |c| used[c] }
        return [:infeasible, nil, { type: :empty_clusters, clusters: empty }] unless empty.empty?

        [:feasible, mapping, nil]
      end

      def forbidden_groups(gi, mapping)
        @group_adjacency[gi].reject { |gj| mapping[gj].nil? }
                            .map { |gj| { group: gj, cluster: mapping[gj] } }
      end

      # Budgeted backtracking K-coloring of the must-link groups.
      # With +require_surjective+ every cluster must receive a group, so
      # empty clusters are repaired without breaking constraints.
      # Returns one of:
      # * [:feasible, mapping, nil]
      # * [:infeasible, nil, witness] (whole search tree explored)
      # * [:budget_exhausted, partial_mapping, witness] (budget hit first)
      def color_groups(require_surjective:)
        groups_count = @must_link_groups.length
        if require_surjective && groups_count < @number_of_clusters
          return [:infeasible, nil,
                  { type: :too_few_groups, groups: groups_count,
                    number_of_clusters: @number_of_clusters }]
        end

        @search_calls += 1
        nodes = 0
        mapping = Array.new(groups_count)
        unassigned = Set.new((0...groups_count).to_a)
        used = Array.new(@number_of_clusters, false)

        search = lambda do
          empty_clusters = used.count(&:!)
          if unassigned.empty?
            return require_surjective && empty_clusters.positive? ? :infeasible : :feasible
          end
          return :infeasible if require_surjective && unassigned.length < empty_clusters

          return :budget_exhausted if !@search_budget.nil? && nodes >= @search_budget

          nodes += 1

          gi = choose_group(unassigned, mapping)
          options = legal_clusters_for(gi, mapping, used, require_surjective, unassigned.length - 1)
          return :infeasible if options.empty?

          options.each do |cluster|
            mapping[gi] = cluster
            unassigned.delete(gi)
            was_unused = !used[cluster]
            used[cluster] = true
            outcome = search.call
            return :feasible if outcome == :feasible
            return :budget_exhausted if outcome == :budget_exhausted

            used[cluster] = false if was_unused
            unassigned.add(gi)
            mapping[gi] = nil
          end
          :infeasible
        end

        outcome = search.call
        @search_nodes += nodes
        case outcome
        when :feasible
          [:feasible, mapping, nil]
        when :budget_exhausted
          witness = { type: :budget_exhausted, nodes_visited: nodes, budget: @search_budget,
                      assigned_groups: mapping.compact.length, total_groups: groups_count,
                      partial_mapping: mapping }
          [:budget_exhausted, mapping, witness]
        else
          witness = { type: :not_k_colorable, number_of_clusters: @number_of_clusters,
                      total_groups: groups_count, nodes_visited: nodes }
          [:infeasible, nil, witness]
        end
      end

      # Minimum-remaining-values heuristic; ties favor the most connected
      # unassigned group, then the largest group, then stable index order.
      def choose_group(unassigned, mapping)
        unassigned.min_by do |gi|
          [
            legal_clusters_for(gi, mapping, nil, false, nil).length,
            -@group_adjacency[gi].count { |gj| unassigned.include?(gj) },
            -@must_link_groups[gi].length,
            gi
          ]
        end
      end

      def legal_clusters_for(gi, mapping, used, require_surjective, groups_left)
        forbidden = neighbor_clusters(gi, mapping)
        candidates = (0...@number_of_clusters).select do |cluster|
          next false if forbidden.include?(cluster)

          if require_surjective && !used.nil? && !used[cluster]
            # Taking a still-empty cluster must leave enough groups
            # behind to occupy every other empty cluster.
            empties_after = used.count(&:!) - 1
            next false if groups_left < empties_after
          end
          true
        end
        candidates.sort_by { |cluster| group_cluster_cost(gi, cluster) }
      end

      def neighbor_clusters(gi, mapping)
        @group_adjacency[gi].filter_map { |gj| mapping[gj] }.uniq
      end

      # Total contribution of a whole must-link group to a cluster, so
      # large groups naturally influence the placement and, through it,
      # the centroid computation.
      def group_cluster_cost(gi, cluster)
        return cluster.to_f if @centroids.nil? # structural precheck: no centroids yet

        centroid = @centroids[cluster]

        @must_link_groups[gi].sum { |i| distance(@data_set.data_items[i], centroid) }.to_f
      end

      # Materialize a feasible group->cluster mapping into per-item
      # assignments, cluster data sets and indices.
      def apply_assignment(group_to_cluster)
        n = @data_set.data_items.length
        @assignments = Array.new(n)
        @clusters = Array.new(@number_of_clusters) do
          Ai4r::Data::DataSet.new data_labels: @data_set.data_labels
        end
        @cluster_indices = Array.new(@number_of_clusters) { [] }
        group_to_cluster.each_with_index do |cluster, gi|
          @must_link_groups[gi].each do |i|
            item = @data_set.data_items[i]
            @assignments[i] = cluster
            @clusters[cluster] << item
            @cluster_indices[cluster] << i
          end
        end
      end

      def failed_result(status, detail)
        mapping = detail && detail[:partial_mapping]
        assignments = Array.new(@data_set.data_items.length)
        clusters = Array.new(@number_of_clusters) do
          Ai4r::Data::DataSet.new data_labels: @data_set.data_labels
        end
        unassigned = (0...@data_set.data_items.length).to_a
        unless mapping.nil?
          unassigned = []
          @must_link_groups.each_with_index do |group, gi|
            cluster = mapping[gi]
            if cluster.nil?
              unassigned.concat(group)
            else
              group.each do |i|
                assignments[i] = cluster
                clusters[cluster] << @data_set.data_items[i]
              end
            end
          end
          unassigned.sort!
        end
        ConstrainedKMeansResult.new(
          assignments: assignments,
          clusters: clusters,
          centroids: @centroids,
          objective: objective_for(assignments),
          reason: status,
          iterations: @iterations,
          unassigned_indices: unassigned,
          must_link_groups: @must_link_groups,
          cannot_link: cannot_link_pairs,
          conflicts: detail.nil? ? [] : [detail],
          backtracks: @backtracks,
          search_nodes_visited: @search_nodes,
          search_calls: @search_calls
        )
      end

      def success_result(reason)
        ConstrainedKMeansResult.new(
          assignments: @assignments,
          clusters: @clusters,
          centroids: @centroids,
          objective: objective_for(@assignments),
          reason: reason,
          iterations: @iterations,
          unassigned_indices: [],
          must_link_groups: @must_link_groups,
          cannot_link: cannot_link_pairs,
          backtracks: @backtracks,
          search_nodes_visited: @search_nodes,
          search_calls: @search_calls
        )
      end

      def cannot_link_pairs
        @group_adjacency.each_with_index.flat_map do |neighbors, gi|
          neighbors.select { |gj| gj > gi }.flat_map do |gj|
            @must_link_groups[gi].product(@must_link_groups[gj]).map(&:sort)
          end
        end.uniq.sort
      end

      def objective_for(assignments)
        @data_set.data_items.each_with_index.sum do |item, i|
          cluster = assignments[i]
          cluster.nil? ? 0.0 : distance(item, @centroids[cluster]).to_f
        end
      end

      # Mirror the result onto the standard KMeans attributes so that
      # sse/eval/centroids/clusters keep working after a constrained run.
      def sync_state(result)
        @result = result
        @clusters = result.clusters
        @centroids = result.centroids
        @assignments = result.assignments
        @number_of_clusters = result.clusters.length
      end
    end
  end
end
