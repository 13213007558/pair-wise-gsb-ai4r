# frozen_string_literal: true

require_relative '../../test_helper'
require 'ai4r/clusterers/constrained_k_means'

class TestConstrainedKMeans < Minitest::Test
  include Ai4r::Data
  include Ai4r::Clusterers

  def dataset(items)
    DataSet.new(data_items: items)
  end

  # The canonical greedy placement (groups in index order, cheapest feasible
  # cluster first) fails on this cannot-link graph with these initial
  # centroids: groups 0,1,2 greedily take clusters 0,1,2 and group 3 is
  # adjacent to all of them. The graph is 3-colorable ({0,2} share a
  # cluster), so the backtracking search must undo the early choice made for
  # group 2 to produce a complete assignment.
  def test_backtracking_retracts_early_greedy_choice
    items = [[0.0], [10.0], [20.0], [19.0]]
    cannot_link = [[0, 1], [1, 2], [0, 3], [1, 3], [2, 3]]
    result = ConstrainedKMeans.new
                              .set_parameters(centroid_indices: [0, 1, 2])
                              .build_constrained(dataset(items), 3, cannot_link: cannot_link)

    assert result.fully_assigned?
    assert_equal :converged, result.end_reason
    assert result.stats[:greedy_failed], 'canonical greedy was expected to get stuck'
    assert_operator result.stats[:search_nodes_visited], :>, 0,
                    'backtracking search was expected to run'
    # groups 0 and 2 must share a cluster so group 3 gets its own slot
    assert_equal result.assignments[0], result.assignments[2]
    refute_equal result.assignments[3], result.assignments[0]
    refute_equal result.assignments[3], result.assignments[1]
    # every recorded assignment respects every cannot-link edge
    cannot_link.each do |a, b|
      refute_equal result.assignments[a], result.assignments[b]
    end
  end

  def test_must_link_groups_always_stay_together
    items = [[0.0], [1.0], [2.0], [10.0], [11.0]]
    result = ConstrainedKMeans.new
                              .set_parameters(centroid_indices: [0, 3])
                              .build_constrained(dataset(items), 2,
                                                 must_link: [[0, 1], [1, 2], [3, 4]])

    assert_equal [0, 0, 0, 1, 1], result.assignments
    # group size participates in centroid computation: mean of 0,1,2 = 1
    assert_equal [[1.0], [10.5]], result.centroids
    assert_equal :converged, result.end_reason
  end

  def test_cannot_link_between_must_link_groups_is_respected
    items = [[0.0], [1.0], [2.0], [10.0], [11.0]]
    result = ConstrainedKMeans.new
                              .set_parameters(centroid_indices: [0, 3])
                              .build_constrained(
                                dataset(items), 2,
                                must_link: [[0, 1], [1, 2], [3, 4]],
                                cannot_link: [[0, 4]]
                              )

    assert_equal :converged, result.end_reason
    refute_equal result.assignments[0], result.assignments[4]
  end

  def test_direct_must_link_cannot_link_conflict_is_reported_before_iterating
    items = [[0.0], [1.0], [2.0]]
    error = assert_raises(ConstraintConflict) do
      ConstrainedKMeans.new.build_constrained(
        dataset(items), 2,
        must_link: [[0, 1], [1, 2]],
        cannot_link: [[0, 2]]
      )
    end
    conflict = error.conflicts.first
    assert_equal [0, 2], conflict[:pair]
    # transitive must-link chain is reported so the conflict is checkable
    assert_equal [0, 1, 2], conflict[:must_link_path]
  end

  def test_self_cannot_link_is_a_direct_conflict
    assert_raises(ConstraintConflict) do
      ConstrainedKMeans.new.build_constrained(
        dataset([[0.0]]), 1, cannot_link: [[0, 0]]
      )
    end
  end

  def test_proven_infeasible_distinguishes_from_budget
    items = [[0.0], [10.0], [20.0]]
    triangle = [[0, 1], [1, 2], [0, 2]]

    unlimited = ConstrainedKMeans.new
                                 .set_parameters(centroid_indices: [0, 1], search_budget: nil)
                                 .build_constrained(dataset(items), 2, cannot_link: triangle)
    assert_equal :infeasible, unlimited.end_reason
    assert_equal [0, 1, 2], unlimited.unassigned_indices
    assert_nil unlimited.objective_value

    budgeted = ConstrainedKMeans.new
                                .set_parameters(centroid_indices: [0, 1], search_budget: 1)
                                .build_constrained(dataset(items), 2, cannot_link: triangle)
    assert_equal :budget_exhausted, budgeted.end_reason
    refute_empty budgeted.unassigned_indices
    # no sample is hidden: assignments length matches the data
    assert_equal items.length, budgeted.assignments.length
  end

  def test_every_original_sample_is_accounted_for
    items = [[0.0], [1.0], [2.0]]
    result = ConstrainedKMeans.new
                              .set_parameters(centroid_indices: [0, 1], search_budget: nil)
                              .build_constrained(dataset(items), 2,
                                                 cannot_link: [[0, 1], [1, 2], [0, 2]])
    accounted = result.unassigned_indices.length +
                result.assignments.compact.length
    assert_equal items.length, accounted
  end

  def test_empty_cluster_terminate_keeps_constraints
    items = [[0.0], [1.0], [10.0], [11.0]]
    result = ConstrainedKMeans.new
                              .set_parameters(centroid_indices: [0, 2, 1], on_empty: 'terminate')
                              .build_constrained(dataset(items), 3, must_link: [[0, 1], [2, 3]])

    assert_equal :empty_cluster, result.end_reason
    assert_equal [0, 0, 1, 1], result.assignments
  end

  def test_empty_cluster_eliminate_reduces_k_and_converges
    items = [[0.0], [1.0], [10.0], [11.0]]
    result = ConstrainedKMeans.new
                              .set_parameters(centroid_indices: [0, 2, 1], on_empty: 'eliminate')
                              .build_constrained(dataset(items), 3, must_link: [[0, 1], [2, 3]])

    assert_equal :converged, result.end_reason
    assert_equal 2, result.number_of_clusters
    assert_equal [0, 0, 1, 1], result.assignments
  end

  def test_empty_cluster_relocation_respects_constraints
    # two indivisible groups cannot cover three clusters, so no relocation
    # can ever succeed: result must be infeasible, not a silent forced move
    items = [[0.0], [1.0], [10.0], [11.0]]
    %w[random outlier].each do |policy|
      result = ConstrainedKMeans.new
                                .set_parameters(centroid_indices: [0, 2, 1], on_empty: policy)
                                .build_constrained(dataset(items), 3,
                                                   must_link: [[0, 1], [2, 3]])
      assert_equal :infeasible, result.end_reason, "policy #{policy}"
      assert_equal [0, 0, 1, 1], result.assignments
    end
  end

  def test_reject_non_distinct_initial_centroids_for_constrained_run
    items = [[0.0], [0.0], [1.0]]
    error = assert_raises(ArgumentError) do
      ConstrainedKMeans.new
                       .set_parameters(centroid_indices: [0, 1, 2])
                       .build_constrained(dataset(items), 3)
    end
    assert_match(/distinct data item per cluster/, error.message)
  end

  def test_outlier_relocation_fills_empty_slot_without_violating_constraints
    # Three must-link groups A{0,1}, B{2,3,4}, C{5}, k=3. The c2 seed sits on
    # item 2 (19), but B as a whole is cheaper to place on c1 (20), so c2 is
    # initially empty. 'outlier' relocates c2 to the furthest unused point
    # (100) and the constraint-aware full-cover reallocation fills it; the
    # same run with 'terminate' keeps the empty cluster instead.
    items = [[0.0], [1.0], [19.0], [20.0], [21.0], [100.0]]
    must_link = [[0, 1], [2, 3], [3, 4]]

    outlier = ConstrainedKMeans.new
                               .set_parameters(centroid_indices: [0, 3, 2], on_empty: 'outlier')
                               .build_constrained(dataset(items), 3, must_link: must_link)
    assert_equal :converged, outlier.end_reason
    assert_equal 3, outlier.number_of_clusters
    assert_equal [0, 0, 1, 1, 1, 2], outlier.assignments
    assert_equal [100.0], outlier.centroids.last
    assert_operator outlier.stats[:relocation_search_nodes], :>, 0

    terminate = ConstrainedKMeans.new
                                 .set_parameters(centroid_indices: [0, 3, 2], on_empty: 'terminate')
                                 .build_constrained(dataset(items), 3, must_link: must_link)
    assert_equal :empty_cluster, terminate.end_reason
  end

  def test_invalid_index_rejected
    error = assert_raises(ArgumentError) do
      ConstrainedKMeans.new.build_constrained(
        dataset([[0.0], [1.0]]), 2, must_link: [[0, 5]]
      )
    end
    assert_match(/out of range/, error.message)
  end

  def test_result_exposes_final_objective
    items = [[0.0], [1.0], [10.0], [11.0]]
    result = ConstrainedKMeans.new
                              .set_parameters(centroid_indices: [0, 2])
                              .build_constrained(dataset(items), 2, must_link: [[0, 1], [2, 3]])

    assert_in_delta 1.0, result.objective_value
    assert_equal result.objective_value, result.sse
    assert_equal result.assignments, result.labels
  end
end
