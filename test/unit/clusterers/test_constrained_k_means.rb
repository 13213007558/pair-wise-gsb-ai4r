# frozen_string_literal: true

require_relative '../../test_helper'
require 'ai4r/clusterers/constrained_k_means'

class TestConstrainedKMeans < Minitest::Test
  include Ai4r::Data
  include Ai4r::Clusterers

  # Four corners. Items 0,1,2 are the initial centroids, so a greedy pass
  # assigns each of them to a different cluster at zero cost. Item 3 has
  # cannot-links to 0,1,2, leaving greedy no legal cluster - yet the star
  # graph is 3-colorable: feasibility is restored only by withdrawing an
  # early choice (two of 0,1,2 must share a cluster).
  STAR_POINTS = [[0, 0], [0, 100], [100, 0], [100, 100]].freeze
  STAR_CANNOT_LINK = [[0, 3], [1, 3], [2, 3]].freeze
  K4_CANNOT_LINK = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]].freeze

  def dataset(items = STAR_POINTS)
    DataSet.new(data_items: items)
  end

  def build(**opts)
    km = ConstrainedKMeans.new
    km.centroid_indices = [0, 1, 2]
    km.set_parameters(**opts) unless opts.empty?
    km
  end

  def assert_constraints_satisfied(result, cannot_link)
    cannot_link.each do |i, j|
      refute_nil result.assignments[i], "item #{i} must be assigned"
      refute_nil result.assignments[j], "item #{j} must be assigned"
      refute_equal result.assignments[i], result.assignments[j],
                   "cannot-link #{[i, j].inspect} violated"
    end
  end

  def test_backtracks_to_withdraw_an_early_greedy_choice
    result = build.build_constrained(dataset, 3, cannot_link: STAR_CANNOT_LINK)

    assert result.complete?, result.reason
    assert_operator result.backtracks, :>=, 1
    assert_operator result.search_calls, :>=, 1
    assert_equal [0, 1, 2, 3].length, result.assignments.length
    refute_includes result.assignments, nil
    assert_equal(3, result.clusters.count { |c| !c.data_items.empty? })
    assert_constraints_satisfied(result, STAR_CANNOT_LINK)
    # 0,1,2 cannot occupy three different clusters because 3 needs one,
    # so at least two of them share a cluster after the withdrawal.
    assert(result.assignments.values_at(0, 1, 2).uniq.length < 3)
    assert_kind_of Float, result.objective
  end

  def test_must_link_is_transitive_and_group_shapes_centroid
    points = [[0, 0], [10, 0], [2, 0], [100, 0]]
    km = ConstrainedKMeans.new
    km.centroid_indices = [0, 3]
    result = km.build_constrained(
      dataset(points), 2,
      must_link: [[0, 1], [1, 2]], cannot_link: [[0, 3]]
    )

    assert result.complete?, result.reason
    assert_equal [[0, 1, 2], [3]], result.must_link_groups
    assert_equal [0, 0, 0, 1], result.assignments
    assert_in_epsilon 4.0, result.centroids[0][0]
    assert_in_epsilon 100.0, result.centroids[1][0]
    assert_constraints_satisfied(result, [[0, 3]])
  end

  def test_direct_conflict_raised_before_iteration_with_verifiable_details
    err = assert_raises(ConstraintConflict) do
      ConstrainedKMeans.new.build_constrained(
        dataset, 2,
        must_link: [[0, 1], [1, 2]], cannot_link: [[0, 2]]
      )
    end
    assert_equal :direct_conflict, err.conflicts.first[:type]
    assert_equal [0, 2], err.conflicts.first[:pair]
    assert_equal [0, 1, 2], err.conflicts.first[:must_link_group]
    assert_equal [[0, 1, 2], [3]], err.must_link_groups
  end

  def test_self_pair_is_rejected
    assert_raises(ArgumentError) do
      ConstrainedKMeans.new.build_constrained(dataset, 2, must_link: [[1, 1]])
    end
    assert_raises(ArgumentError) do
      ConstrainedKMeans.new.build_constrained(dataset, 2, cannot_link: [[9, 0]])
    end
  end

  def test_k4_into_three_clusters_is_proven_infeasible
    result = build.build_constrained(dataset, 3, cannot_link: K4_CANNOT_LINK)

    refute result.complete?
    assert_equal :infeasible, result.reason
    assert_equal :not_k_colorable, result.conflicts.first[:type]
    assert_equal [0, 1, 2, 3], result.unassigned_indices
    assert_equal [nil, nil, nil, nil], result.assignments
  end

  def test_too_few_groups_for_k_nonempty_clusters_is_infeasible
    points = [[0, 0], [1, 1], [100, 100]]
    result = ConstrainedKMeans.new.build_constrained(
      dataset(points), 3, must_link: [[0, 1], [1, 2]]
    )

    assert_equal :infeasible, result.reason
    assert_equal :too_few_groups, result.conflicts.first[:type]
    refute result.complete?
  end

  def test_budget_zero_is_exhausted_and_hides_no_records
    km = build(search_budget: 0)
    result = km.build_constrained(dataset, 3, cannot_link: STAR_CANNOT_LINK)

    assert_equal :budget_exhausted, result.reason
    refute result.complete?
    assert_equal [0, 1, 2, 3].sort, result.unassigned_indices.sort
    assert_equal 4, result.assignments.length
    assert_equal :budget_exhausted, result.conflicts.first[:type]
  end

  def test_infeasible_during_iteration_when_precheck_is_skipped
    result = build(verify_feasibility: false)
             .build_constrained(dataset, 3, cannot_link: K4_CANNOT_LINK)

    assert_equal :infeasible, result.reason
    assert_equal 4, result.unassigned_indices.length
  end

  def test_budget_exhausted_reports_partial_assignment
    result = build(search_budget: 3, verify_feasibility: false)
             .build_constrained(dataset, 3, cannot_link: STAR_CANNOT_LINK)

    assert_equal :budget_exhausted, result.reason
    assert result.unassigned_indices.any?
    assert result.assignments.any?
    result.cannot_link.each do |i, j|
      next if result.assignments[i].nil? || result.assignments[j].nil?

      refute_equal result.assignments[i], result.assignments[j]
    end
    # The union of assigned and unassigned records is exactly all items.
    covered = result.assignments.each_index.reject do |i|
      result.assignments[i].nil?
    end + result.unassigned_indices
    assert_equal (0..3).to_a, covered.sort
  end

  def test_converged_assignment_is_stable_and_matches_centroids
    result = build(random_seed: 3)
             .build_constrained(dataset, 2, cannot_link: [[0, 3]])

    assert result.complete?
    assert_constraints_satisfied(result, [[0, 3]])
    expected = result.assignments.each_with_index.sum do |c, i|
      Ai4r::Data::Proximity.squared_euclidean_distance(
        STAR_POINTS[i], result.centroids[c]
      )
    end
    assert_in_epsilon expected, result.objective
    assert_operator result.iterations, :>=, 1
  end

  def test_empty_cluster_repair_respects_constraints
    # Two groups forced apart, K=2: both clusters must become non-empty.
    points = [[0, 0], [1, 0], [100, 0], [101, 0]]
    km = ConstrainedKMeans.new
    km.centroid_indices = [0, 2]
    result = km.build_constrained(
      dataset(points), 2,
      must_link: [[0, 1], [2, 3]], cannot_link: [[0, 2]]
    )

    assert result.complete?, result.reason
    sizes = result.clusters.map { |c| c.data_items.size }
    assert_equal [2, 2], sizes.sort
  end

  def test_result_to_h_exposes_assignments_objective_and_reason
    result = build.build_constrained(dataset, 3, cannot_link: STAR_CANNOT_LINK)
    hash = result.to_h
    assert_equal result.assignments, hash[:assignments]
    assert_equal result.objective, hash[:objective]
    assert_equal result.reason, hash[:reason]
    assert_equal result.must_link_groups, hash[:must_link_groups]
  end

  def test_standard_kmeans_state_is_synced_after_constrained_run
    km = build
    result = km.build_constrained(dataset, 2, cannot_link: [[0, 3]])

    assert_equal(result.assignments,
                 STAR_POINTS.map { |point| km.eval(point) })
    assert_equal result.objective, km.sse
  end

  def test_invalid_arguments
    assert_raises(ArgumentError) do
      ConstrainedKMeans.new.build_constrained(
        DataSet.new(data_items: []), 1
      )
    end
    assert_raises(ArgumentError) do
      ConstrainedKMeans.new.build_constrained(dataset, 9)
    end
  end
end
