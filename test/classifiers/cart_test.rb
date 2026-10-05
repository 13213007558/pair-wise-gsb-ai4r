# frozen_string_literal: true

require 'ai4r/classifiers/cart'
require_relative '../test_helper'

module Ai4r
  module Classifiers
    class CartTest < Minitest::Test
      LABELS = %w[x model y].freeze

      def build(items, **options)
        CART.new.build(Ai4r::Data::DataSet.new(data_items: items, data_labels: LABELS), **options)
      end

      # Duplicating a row and assigning it an equivalent integer weight must
      # yield identical trees and leaf distributions.
      def test_duplicated_row_equals_integer_weight
        base = [[1, 'a', 'A'], [8, 'b', 'B'], [9, 'b', 'B'], [3, 'a', 'A']]
        duplicated = build(base + [[1, 'a', 'A']])
        weighted = build(base, weights: [2, 1, 1, 1])

        assert_equal duplicated.to_h, weighted.to_h
        assert_equal [1, 'a'].then { |row| duplicated.eval(row) }, weighted.eval([1, 'a'])
        # Every A lands on the left leaf: three weighted rows in both builds.
        assert_equal 3.0, weighted.tree.left_child.distribution['A']
        assert_equal 3.0, duplicated.tree.left_child.distribution['A']
        assert_equal 'A', weighted.eval([2, 'a'])
        assert_equal 'B', weighted.eval([9, 'b'])
      end

      # The root splits on the numeric sensor x; when x is missing, the highly
      # correlated backup sensor z (surrogate) must decide the branch.
      def test_missing_primary_feature_uses_surrogate
        items = [
          [1, 10, 'A'], [2, 20, 'A'], [5, 50, 'A'],
          [8, 80, 'B'], [9, 90, 'B']
        ]
        cart = build(items)
        assert_equal 0, cart.tree.feature_index
        refute_empty cart.tree.surrogates
        surrogate = cart.tree.surrogates.first
        assert_equal 1, surrogate.feature_index

        path = cart.path_for([nil, 85])
        assert path.first[:primary_missing]
        assert path.first[:surrogate_used]
        assert_equal 'B', cart.eval([nil, 85])
        assert_equal 'A', cart.eval([nil, 15])
      end

      # When the primary feature and every surrogate are unavailable, the
      # explicitly trained default direction is used.
      def test_default_direction_when_all_splits_missing
        items = [[1, 10, 'A'], [2, 20, 'A'], [8, 80, 'B'], [9, 90, 'B']]
        cart = build(items)
        path = cart.path_for([nil, nil])
        refute path.first[:surrogate_used]
        assert_equal cart.tree.default_direction, path.first[:direction]
        assert_equal cart.eval([1, 10]), cart.eval([nil, nil])
      end

      # Increasing alpha only collapses branches; leaves never grow back and
      # the selected subtrees form a nested sequence.
      def test_pruning_sequence_is_nested_and_monotone
        items = [
          [1, 'A'], [2, 'A'], [3, 'B'], [4, 'A'], [8, 'B'], [9, 'B']
        ]
        labels = %w[x y]
        cart = CART.new.build(Ai4r::Data::DataSet.new(data_items: items, data_labels: labels))
        sequence = cart.pruning_sequence

        alphas = sequence.map { |entry| entry[:alpha] }
        leaves = sequence.map { |entry| entry[:leaves] }
        assert_equal alphas, alphas.sort
        assert_equal leaves, leaves.sort.reverse
        assert leaves.first > leaves.last

        cut_alpha = sequence[1][:alpha]
        pruned_root = cart.tree_for_alpha(cut_alpha + 1_000_000)
        assert pruned_root.leaf?
        # A subtree removed at a smaller alpha never reappears at a larger alpha.
        mid_pred = cart.eval_with_alpha([8], cut_alpha)
        root_pred = cart.eval_with_alpha([8], cut_alpha + 1_000_000)
        refute_equal mid_pred, root_pred
        # Selecting the same alpha again gives the same nested tree, never a larger one.
        assert_equal cart.tree_for_alpha(cut_alpha).then(&:left_child).label,
                     cart.eval_with_alpha([1], cut_alpha)
        assert_equal leaves.last,
                     (1..5).map { |step| cart.tree_for_alpha(sequence.last[:alpha] + step).then { |t| t.leaf? ? 1 : 0 } }.max
      end

      def test_categorical_binary_split
        items = [%w[red A], %w[red A], %w[blue B], %w[green B]]
        labels = %w[color y]
        cart = CART.new.build(Ai4r::Data::DataSet.new(data_items: items, data_labels: labels))
        refute cart.tree.numeric?
        assert_equal %w[red], cart.tree.subset
        assert_equal 'A', cart.eval(['red'])
        assert_equal 'B', cart.eval(['blue'])
        assert_includes cart.get_rules, 'color in [red]'
      end

      # Weights are reused for the leaf decision through misclassification costs.
      def test_weights_used_for_cost_weighted_leaf_label
        items = [[1, 'A'], [1, 'A'], [1, 'B']]
        labels = %w[x y]
        cart = CART.new.build(
          Ai4r::Data::DataSet.new(data_items: items, data_labels: labels),
          costs: { 'A' => { 'B' => 5 } }
        )
        assert cart.tree.leaf?
        assert_equal 'B', cart.eval([1])
        assert_equal 2.0, cart.tree.distribution['A']
      end

      def test_rare_fault_weighting_changes_the_tree
        # A single rare fault sits next to a normal reading at high x, so without
        # weighting the leaf majority stays 'ok'; the fault weight flips it.
        items = [[1, 'ok'], [2, 'ok'], [3, 'ok'], [4, 'ok'],
                 [8, 'ok'], [8, 'ok'], [9, 'ok'], [9, 'fault']]
        labels = %w[x y]
        unweighted = CART.new.build(Ai4r::Data::DataSet.new(data_items: items, data_labels: labels))
        weighted = CART.new.build(
          Ai4r::Data::DataSet.new(data_items: items, data_labels: labels),
          weights: [1, 1, 1, 1, 1, 1, 1, 20]
        )
        assert_equal 'ok', unweighted.eval([9])
        assert_equal 'fault', weighted.eval([9])
      end

      def test_rejects_negative_and_mismatched_weights
        items = [[1, 'A'], [2, 'B']]
        assert_raises(ArgumentError) { build(items, weights: [1, -1]) }
        assert_raises(ArgumentError) { build(items, weights: [1]) }
        assert_raises(ArgumentError) { build(items, costs: { 'A' => 5 }) }
      end

      def test_rules_and_graphviz_describe_conditions
        items = [[1, 'a', 'A'], [9, 'b', 'B']]
        cart = build(items)
        assert_includes cart.get_rules, 'x <= 5.0'
        assert_includes cart.get_rules, 'x > 5.0'
        assert_match(/digraph G/, cart.to_graphviz)
        assert_equal 'x', cart.to_h[:feature]
      end
    end
  end
end
