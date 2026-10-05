# frozen_string_literal: true

require_relative '../data/data_set'
require_relative '../classifiers/classifier'

module Ai4r
  module Classifiers
    # CART (Classification And Regression Trees) binary decision tree.
    #
    # Every internal node makes one yes/no test:
    # * numeric feature: feature <= threshold goes left, otherwise right
    # * discrete feature: membership in a trained subset goes left
    #
    # Each example can carry a non-negative weight. The same weights are used
    # for training, for the class distribution stored at every leaf, and for
    # misclassification costs. Missing feature values are represented by nil.
    # A primary split supports surrogate splits; only when every available
    # surrogate is also missing does an example take the node trained default
    # direction.
    class CART < Classifier
      attr_reader :data_set, :tree, :classes, :weights, :costs, :pruning_sequence

      parameters_info max_depth: 'Maximum recursion depth. Default nil (no limit).',
                      min_weight: 'Minimum total weight required to split a node. Default 1.',
                      min_gain: 'Minimum weighted Gini gain required to split. Default 0.',
                      max_categorical_values: 'Domains up to this size use an exhaustive subset ' \
                                              'search; larger domains use one-vs-rest. Default 12.',
                      weights: 'Array of non-negative per-example weights. Defaults to all ones.',
                      costs: 'Per-class misclassification costs, e.g. { B => { A => 5 } }.'

      def initialize
        super()
        @max_depth = nil
        @min_weight = 1
        @min_gain = 0
        @max_categorical_values = 12
        @weights = nil
        @costs = {}
      end

      # Build the fully grown tree and its weakest-link pruning sequence.
      # The last column of each data item is its class.
      def build(data_set, weights: @weights, costs: @costs, **parameters)
        set_parameters(parameters)
        data_set.check_not_empty
        @data_set = data_set
        @weights = normalize_weights(weights)
        @costs = costs || {}
        validate_weights_and_costs
        @classes = data_items.map(&:last).uniq
        @tree = grow(all_indices, 0)
        @pruning_sequence = build_pruning_sequence
        self
      end

      # Predict the class of a new example. Missing values are represented by nil.
      def eval(data, node = @tree)
        return node.label if node.leaf?

        child = route(node, data).zero? ? node.left_child : node.right_child
        eval(data, child)
      end

      # Root node of the fully grown tree.
      def root
        @tree
      end

      # Nested subtree selected by a cost-complexity alpha. Returns a copy.
      def tree_for_alpha(alpha)
        entry = @pruning_sequence.reverse.find { |item| item[:alpha] <= alpha }
        deep_copy(entry[:tree])
      end

      # Predict using the nested subtree selected by alpha.
      def eval_with_alpha(data, alpha)
        eval(data, tree_for_alpha(alpha))
      end

      # Human-readable rule for every leaf of the current tree.
      def get_rules
        paths = leaf_paths(@tree).map do |conditions, leaf|
          body = conditions.empty? ? 'true' : conditions.join(' and ')
          dist = leaf.distribution.map { |k, w| "#{k}:#{format_weight(w)}" }.join(', ')
          "if #{body} then #{category_label}='#{leaf.label}' (#{dist})"
        end
        return '' if paths.empty?

        rest = paths[1..].map { |rule| "els#{rule}" }.join("\n")
        "#{paths.first}\n#{rest}"
      end

      # Conditions used while evaluating one example, including surrogate use.
      def path_for(data)
        node = @tree
        path = []
        until node.leaf?
          direction = route(node, data)
          surrogate = data[node.feature_index].nil? ? find_surrogate(node, data) : nil
          path << {
            feature: label(node.feature_index),
            feature_index: node.feature_index,
            condition: node.condition(label(node.feature_index), direction),
            direction: direction,
            primary_missing: data[node.feature_index].nil?,
            surrogate_used: !surrogate.nil?
          }
          node = direction.zero? ? node.left_child : node.right_child
        end
        path
      end

      # Nested Hash representation of the fitted tree.
      def to_h(node = @tree)
        return { leaf: node.label, distribution: node.distribution } if node.leaf?

        split = node.numeric? ? { threshold: node.threshold } : { subset: node.subset }
        {
          feature: label(node.feature_index),
          feature_index: node.feature_index,
          split: split,
          default_direction: node.default_direction,
          surrogates: node.surrogates.map { |s| s.to_h(@data_set.data_labels) },
          left: to_h(node.left_child),
          right: to_h(node.right_child)
        }
      end

      # GraphViz DOT representation of the current tree.
      def to_graphviz
        lines = ['digraph G {']
        render_graphviz(@tree, nil, nil, lines, [0])
        lines << '}'
        lines.join("\n")
      end

      private

      def all_indices
        (0...data_items.length).to_a
      end

      def data_items
        @data_set.data_items
      end

      def feature_count
        @data_set.data_labels.length - 1
      end

      def category_label
        @data_set.data_labels.last
      end

      def label(index)
        @data_set.data_labels[index]
      end

      def format_weight(weight)
        weight == weight.to_i ? weight.to_i.to_s : format('%.4g', weight)
      end

      def normalize_weights(weights)
        return Array.new(data_items.length, 1.0) if weights.nil?

        raise ArgumentError, 'weights must be an array with one non-negative value per example' unless weights.is_a?(Array) && weights.length == data_items.length

        weights.map do |weight|
          value = Float(weight)
          raise ArgumentError, 'weights must be non-negative' if value.negative?

          value
        end
      end

      def validate_weights_and_costs
        raise ArgumentError, 'costs must be a Hash of Hashes keyed by class labels' unless @costs.is_a?(Hash) && @costs.values.all?(Hash)

        @costs.each_value do |row|
          row.each_value do |cost|
            value = Float(cost)
            raise ArgumentError, 'misclassification costs must be non-negative' if value.negative?
          end
        end
      end

      def cost(predicted, actual)
        @costs.dig(predicted, actual) || (predicted == actual ? 0.0 : 1.0)
      end

      def grow(indices, depth)
        distribution = weighted_distribution(indices)
        leaf_label, leaf_cost = best_label(distribution)
        return CartNode.leaf(leaf_label, distribution, leaf_cost) if stop_growing?(indices, depth)

        split = find_best_split(indices)
        return CartNode.leaf(leaf_label, distribution, leaf_cost) if split.nil? || split[:gain] <= @min_gain

        left_indices, right_indices = partition_primary(indices, split)
        split[:surrogates] = fit_surrogates(indices, split)
        split[:default_direction] = default_direction(left_indices, right_indices)
        assign_missing_primary(indices, split, left_indices, right_indices)

        left_child = grow(left_indices, depth + 1)
        right_child = grow(right_indices, depth + 1)
        CartNode.internal(split, left_child, right_child, distribution, leaf_cost)
      end

      def stop_growing?(indices, depth)
        total_weight = sum_weights(indices)
        return true if total_weight.zero?
        return true if @max_depth && depth >= @max_depth
        return true if total_weight < @min_weight

        seen = Set.new
        indices.each { |i| seen << data_items[i].last unless @weights[i].zero? }
        seen.length <= 1
      end

      def weighted_distribution(indices)
        distribution = Hash.new(0.0)
        indices.each { |i| distribution[data_items[i].last] += @weights[i] }
        distribution
      end

      # Minimize weighted misclassification cost; ties follow training class order.
      def best_label(distribution)
        candidates = distribution.empty? ? @classes : distribution.keys
        best = candidates.min_by do |predicted|
          risk = distribution.sum { |actual, weight| weight * cost(predicted, actual) }
          [risk, @classes.index(predicted) || Float::INFINITY]
        end
        risk = distribution.sum { |actual, weight| weight * cost(best, actual) }
        [best, risk]
      end

      def sum_weights(indices)
        indices.sum(0.0) { |i| @weights[i] }
      end

      def find_best_split(indices)
        best = nil
        feature_count.times do |feature_index|
          observed = indices.reject { |i| data_items[i][feature_index].nil? }
          next if observed.empty?

          candidates = if numeric_feature?(observed, feature_index)
                         numeric_split_candidates(observed, feature_index)
                       else
                         categorical_split_candidates(observed, feature_index)
                       end
          candidates.each do |candidate|
            candidate[:gain] = gini_gain(observed, candidate[:left], candidate[:right])
            best = candidate if best.nil? || candidate[:gain] > best[:gain]
          end
        end
        best
      end

      def numeric_split_candidates(observed, feature_index)
        values = observed.map { |i| data_items[i][feature_index] }.uniq.sort
        values.each_cons(2).map do |lower, higher|
          threshold = (lower + higher) / 2.0
          left, right = observed.partition { |i| data_items[i][feature_index] <= threshold }
          candidate_split(feature_index, true, threshold, nil, left, right)
        end
      end

      def categorical_split_candidates(observed, feature_index)
        values = observed.map { |i| data_items[i][feature_index] }.uniq
        candidate_subsets(values).map do |subset|
          left, right = observed.partition { |i| subset.include?(data_items[i][feature_index]) }
          candidate_split(feature_index, false, nil, subset, left, right)
        end
      end

      # Canonical half of the binary power set avoids evaluating a subset and
      # its complement twice.
      def candidate_subsets(values)
        rest = values[1..] || []
        if values.length <= @max_categorical_values
          (0..rest.length).flat_map do |size|
            rest.combination(size).map { |tail| [values.first, *tail] }
          end
        else
          values.map { |value| [value] }
        end
      end

      def candidate_split(feature_index, numeric, threshold, subset, left, right)
        {
          feature_index: feature_index,
          numeric: numeric,
          threshold: threshold,
          subset: subset,
          left: left,
          right: right
        }
      end

      def gini_gain(indices, left, right)
        total = sum_weights(left) + sum_weights(right)
        return 0.0 if total.zero?

        left_total = sum_weights(left)
        right_total = sum_weights(right)
        parent = node_impurity(indices)
        weighted = (left_total * node_impurity(left)) + (right_total * node_impurity(right))
        parent - (weighted / total)
      end

      def gini(indices)
        total = sum_weights(indices)
        return 0.0 if total.zero?

        distribution = weighted_distribution(indices)
        1.0 - distribution.sum { |_, weight| (weight / total)**2 }
      end

      def node_impurity(indices)
        @costs.empty? ? gini(indices) : cost_impurity(indices)
      end

      # Cost-weighted impurity: probability of a class in a node times the
      # weighted cost of assigning each possible class. This is Gini impurity
      # with the misclassification cost matrix folded in.
      def cost_impurity(indices)
        total = sum_weights(indices)
        return 0.0 if total.zero?

        distribution = weighted_distribution(indices)
        probabilities = distribution.transform_values { |weight| weight / total }
        candidates = distribution.keys
        candidates.sum do |predicted|
          probabilities[predicted] *
            distribution.sum { |actual, weight| (weight / total) * cost(predicted, actual) }
        end
      end

      def numeric_feature?(observed, feature_index)
        observed.all? { |i| data_items[i][feature_index].is_a?(Numeric) }
      end

      def partition_primary(indices, split)
        observed = indices.reject { |i| data_items[i][split[:feature_index]].nil? }
        partition_by(observed, split)
      end

      def partition_by(indices, split)
        feature_index = split[:feature_index]
        if split[:numeric]
          indices.partition { |i| data_items[i][feature_index] <= split[:threshold] }
        else
          indices.partition { |i| split[:subset].include?(data_items[i][feature_index]) }
        end
      end

      def assign_missing_primary(indices, split, left_indices, right_indices)
        feature_index = split[:feature_index]
        missing = indices.select { |i| data_items[i][feature_index].nil? }
        missing.each do |index|
          surrogate = find_available_surrogate(split[:surrogates], data_items[index])
          direction = surrogate ? surrogate.direction(data_items[index]) : split[:default_direction]
          (direction.zero? ? left_indices : right_indices) << index
        end
      end

      def default_direction(left_indices, right_indices)
        sum_weights(left_indices) >= sum_weights(right_indices) ? 0 : 1
      end

      # Surrogates are other features whose splits best mimic the primary
      # split on examples where both values are observed.
      def fit_surrogates(indices, primary)
        primary_feature = primary[:feature_index]
        complete = indices.reject do |i|
          data_items[i][primary_feature].nil? || @weights[i].zero?
        end
        return [] if complete.empty?

        target_directions = complete.map do |i|
          primary_direction(data_items[i], primary)
        end
        majority_weight = [
          complete.each_with_index.sum { |_, idx| target_directions[idx].zero? ? @weights[complete[idx]] : 0 },
          complete.each_with_index.sum { |_, idx| target_directions[idx] == 1 ? @weights[complete[idx]] : 0 }
        ].max

        surrogates = []
        feature_count.times do |feature_index|
          next if feature_index == primary_feature

          candidate = best_surrogate_split(complete, feature_index, target_directions)
          next if candidate.nil?
          next unless candidate[:agreement] > majority_weight

          candidate[:measure] = candidate[:agreement] / sum_weights(complete)
          surrogates << SurrogateSplit.new(
            feature_index, candidate[:numeric], candidate[:threshold],
            candidate[:subset], candidate[:polarity], candidate[:agreement], candidate[:measure]
          )
        end
        surrogates.sort_by { |s| [-s.measure, s.feature_index] }
      end

      def best_surrogate_split(complete, feature_index, target_directions)
        observed = []
        complete.each_with_index do |item_index, idx|
          observed << [item_index, idx] unless data_items[item_index][feature_index].nil?
        end
        return nil if observed.empty?

        dataset_indices = observed.map(&:first)
        splits = if dataset_indices.all? { |i| data_items[i][feature_index].is_a?(Numeric) }
                   numeric_surrogate_candidates(observed, feature_index)
                 else
                   categorical_surrogate_candidates(observed, feature_index)
                 end

        best = nil
        splits.each do |candidate|
          agreement = observed.sum(0.0) do |item_index, idx|
            split_left = if candidate[:numeric]
                           data_items[item_index][feature_index] <= candidate[:threshold]
                         else
                           candidate[:subset].include?(data_items[item_index][feature_index])
                         end
            predicted = split_left == candidate[:polarity] ? 0 : 1
            predicted == target_directions[idx] ? @weights[item_index] : 0.0
          end
          candidate[:agreement] = agreement
          best = candidate if best.nil? || agreement > best[:agreement]
        end
        best
      end

      # A candidate maps its own left side either to the primary left (polarity +)
      # or to the primary right (polarity -), allowing negatively correlated surrogates.
      def numeric_surrogate_candidates(observed, feature_index)
        values = observed.map { |item_index, _| data_items[item_index][feature_index] }.uniq.sort
        values.each_cons(2).flat_map do |lower, higher|
          threshold = (lower + higher) / 2.0
          [true, false].map do |polarity|
            { numeric: true, threshold: threshold, subset: nil, polarity: polarity }
          end
        end
      end

      def categorical_surrogate_candidates(observed, feature_index)
        values = observed.map { |item_index, _| data_items[item_index][feature_index] }.uniq
        candidate_subsets(values).flat_map do |subset|
          [true, false].map do |polarity|
            { numeric: false, threshold: nil, subset: polarity ? subset : values - subset, polarity: polarity }
          end
        end
      end

      def primary_direction(item, split)
        value = item[split[:feature_index]]
        if split[:numeric]
          value <= split[:threshold] ? 0 : 1
        else
          split[:subset].include?(value) ? 0 : 1
        end
      end

      def find_available_surrogate(surrogates, item)
        surrogates.find { |surrogate| !item[surrogate.feature_index].nil? }
      end

      def find_surrogate(node, data)
        find_available_surrogate(node.surrogates, data)
      end

      def route(node, data)
        value = data[node.feature_index]
        unless value.nil?
          direction = if node.numeric?
                        value <= node.threshold ? 0 : 1
                      else
                        node.subset.include?(value) ? 0 : 1
                      end
          return direction
        end

        surrogate = find_surrogate(node, data)
        return surrogate.direction(data) if surrogate

        node.default_direction
      end

      # Weakest-link cost-complexity pruning (Breiman, Friedman, Olshen, Stone).
      # Each step collapses one weakest internal node and snapshots the result,
      # yielding a non-decreasing sequence of nested subtrees.
      def build_pruning_sequence
        sequence = [{ alpha: 0.0, leaves: count_leaves(@tree), tree: deep_copy(@tree) }]
        current = deep_copy(@tree)
        until current.leaf?
          annotate_weakness(current) unless current.leaf?
          weakest = find_weakest(current)
          alpha = weakest.g_value
          collapse(weakest)
          sequence << {
            alpha: alpha,
            leaves: count_leaves(current),
            tree: deep_copy(current)
          }
        end
        sequence
      end

      def annotate_weakness(node)
        return if node.leaf?

        annotate_weakness(node.left_child)
        annotate_weakness(node.right_child)
        leaves = count_leaves(node)
        subtree_risk = sum_leaf_risk(node)
        node.leaf_count = leaves
        node.subtree_risk = subtree_risk
        node.g_value = (node.leaf_risk - subtree_risk) / (leaves - 1)
      end

      # Post-order traversal makes the deterministic first-weakest choice on ties.
      def find_weakest(node)
        return node if node.leaf?

        candidates = [node]
        candidates << find_weakest(node.left_child) unless node.left_child.leaf?
        candidates << find_weakest(node.right_child) unless node.right_child.leaf?
        child_weakest = candidates[1..].min_by(&:g_value)
        return child_weakest if child_weakest && child_weakest.g_value < node.g_value

        node
      end

      def collapse(node)
        label, leaf_risk = best_label(node.distribution)
        node.collapse_to_leaf(label, leaf_risk)
      end

      def count_leaves(node)
        return 1 if node.leaf?

        count_leaves(node.left_child) + count_leaves(node.right_child)
      end

      def count_internal(node)
        return 0 if node.leaf?

        1 + count_internal(node.left_child) + count_internal(node.right_child)
      end

      def sum_leaf_risk(node)
        return node.leaf_risk if node.leaf?

        sum_leaf_risk(node.left_child) + sum_leaf_risk(node.right_child)
      end

      def deep_copy(node)
        return CartNode.leaf(node.label, node.distribution.dup, node.leaf_risk) if node.leaf?

        surrogates = node.surrogates.map(&:dup)
        CartNode.internal(
          node.split_hash,
          deep_copy(node.left_child),
          deep_copy(node.right_child),
          node.distribution.dup,
          node.leaf_risk,
          surrogates: surrogates
        )
      end

      def leaf_paths(node, conditions = [])
        return [[conditions, node]] if node.leaf?

        feature = label(node.feature_index)
        left_conditions = conditions + [node.condition(feature, 0)]
        right_conditions = conditions + [node.condition(feature, 1)]
        leaf_paths(node.left_child, left_conditions) +
          leaf_paths(node.right_child, right_conditions)
      end

      def render_graphviz(node, parent_id, edge_label, lines, counter)
        id = counter[0]
        counter[0] += 1
        if node.leaf?
          lines << "  node#{id} [label=\"#{node.label}\", shape=box]"
        else
          lines << "  node#{id} [label=\"#{label(node.feature_index)}\"]"
          render_graphviz(node.left_child, id, node.condition(label(node.feature_index), 0), lines, counter)
          render_graphviz(node.right_child, id, node.condition(label(node.feature_index), 1), lines, counter)
        end
        lines << "  node#{parent_id} -> node#{id} [label=\"#{edge_label}\"]" if parent_id
        id
      end
    end

    # A surrogate split reproduces a primary split using another feature.
    class SurrogateSplit
      attr_reader :feature_index, :numeric, :threshold, :subset, :polarity,
                  :agreement, :measure

      def initialize(feature_index, numeric, threshold, subset, polarity, agreement, measure)
        @feature_index = feature_index
        @numeric = numeric
        @threshold = threshold
        @subset = subset
        @polarity = polarity
        @agreement = agreement
        @measure = measure
      end

      # 0 routes to the primary left child, 1 to the primary right child.
      def direction(data)
        value = data[@feature_index]
        goes_left = @numeric ? value <= @threshold : @subset.include?(value)
        goes_left == @polarity ? 0 : 1
      end

      def condition(labels, direction)
        name = labels[@feature_index]
        if @numeric
          operator = direction.zero? == @polarity ? '<=' : '>'
          value = @threshold
          "#{name} #{operator} #{value}"
        else
          values = direction.zero? == @polarity ? @subset : nil
          values ||= :complement
          values == :complement ? "#{name} not in [#{@subset.join(', ')}]" : "#{name} in [#{values.join(', ')}]"
        end
      end

      def to_h(labels)
        result = {
          feature: labels[@feature_index],
          feature_index: @feature_index,
          polarity: @polarity ? :same : :inverted,
          agreement: @agreement,
          measure: @measure
        }
        @numeric ? result.merge(threshold: @threshold) : result.merge(subset: @subset)
      end
    end

    # Binary tree node. Leaves hold a class label, weighted class distribution,
    # and their resubstitution (weighted misclassification) risk.
    class CartNode
      attr_accessor :left_child, :right_child, :leaf_count, :subtree_risk, :g_value
      attr_reader :feature_index, :numeric, :threshold, :subset, :surrogates,
                  :default_direction, :distribution, :leaf_risk, :label

      def self.leaf(label, distribution, leaf_risk = 0.0)
        new(nil, distribution, label, leaf_risk)
      end

      def self.internal(split, left_child, right_child, distribution, leaf_risk, surrogates: nil)
        node = new(split, distribution, nil, leaf_risk)
        node.left_child = left_child
        node.right_child = right_child
        node.instance_variable_set(:@surrogates, surrogates || split[:surrogates] || [])
        node
      end

      def initialize(split, distribution, label, leaf_risk)
        @split = split
        @distribution = distribution
        @label = label
        @leaf_risk = leaf_risk
        @surrogates = []
        @default_direction = split&.dig(:default_direction) || 0
        @numeric = split&.dig(:numeric)
        @feature_index = split&.dig(:feature_index)
        @threshold = split&.dig(:threshold)
        @subset = split&.dig(:subset)
      end

      def leaf?
        !@label.nil?
      end

      def numeric?
        @numeric == true
      end

      def collapse_to_leaf(label, risk)
        @label = label
        @leaf_risk = risk
        @left_child = nil
        @right_child = nil
        @surrogates = []
        @feature_index = nil
        @numeric = nil
        @threshold = nil
        @subset = nil
        @g_value = nil
        @leaf_count = nil
        @subtree_risk = nil
      end

      def condition(name, direction)
        if numeric?
          operator = direction.zero? ? '<=' : '>'
          "#{name} #{operator} #{@threshold}"
        else
          values = direction.zero? ? @subset : nil
          values ? "#{name} in [#{values.join(', ')}]" : "#{name} not in [#{@subset.join(', ')}]"
        end
      end

      def split_hash
        {
          feature_index: @feature_index,
          numeric: @numeric,
          threshold: @threshold,
          subset: @subset,
          surrogates: @surrogates,
          default_direction: @default_direction
        }
      end
    end
  end
end
