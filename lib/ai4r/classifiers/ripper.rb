# frozen_string_literal: true

# Author::    AI4R contributors (educational implementation of the
# RIPPER algorithm created by William W. Cohen)
# License::   MPL 1.1
# Project::   ai4r
#
# You can redistribute it and/or modify it under the terms of
# the Mozilla Public License version 1.1  as published by the
# Mozilla Foundation at http://www.mozilla.org/MPL/MPL-1.1.txt
#
# W. W. Cohen (1995). Fast Effective Rule Induction. Proceedings of
# the 12th International Conference on Machine Learning, 115-123.

require 'set'
require_relative '../data/data_set'
require_relative '../classifiers/classifier'

module Ai4r
  module Classifiers
    # RIPPER-style ordered rule learner for noisy data. See the class overview
    # in the repository documentation for the fixed rule semantics.
    class Ripper < Classifier
      attr_reader :data_set, :rules, :class_order, :default_class, :growth_items, :prune_items, :trace

      parameters_info(
        prune_ratio: 'Fraction of training data reserved for pruning and ' \
                     'description-length decisions (the rest is used to grow rules).',
        seed: 'Seed for the deterministic growth/prune split.',
        optimize: 'Run the RIPPER optimization pass (rule replacement and revision).',
        default_class: 'Force the class returned when no rule matches. ' \
                       'By default it is the most frequent training class.',
        max_rules_per_class: 'Safety cap on the number of rules induced per class.'
      )

      def initialize
        super()
        @prune_ratio = 0.33
        @seed = nil
        @optimize = true
        @default_class = nil
        @max_rules_per_class = nil
      end

      # Builds the ordered rule set. The last attribute of every item is the
      # class label. Returns self.
      def build(data_set)
        data_set.check_not_empty
        @data_set = data_set
        @rules = []
        @trace = []

        all_items = data_set.data_items.map(&:dup)
        @numeric = detect_numeric_attributes
        @class_order, @default_class = class_ordering(all_items)

        @growth_items, @prune_items = split_items(all_items)
        uncovered_growth = @growth_items.map(&:dup)
        initial_rules = []

        @class_order.each do |target|
          class_rules, uncovered_growth = induce_class_rules(target, uncovered_growth, phase: :grow)
          initial_rules.concat(class_rules)
          @trace << { stage: :growth, class: target, rules: class_rules.map { |r| deep_rule(r) } }
        end

        @rules =
          if @optimize
            optimize_rules(initial_rules)
          else
            initial_rules
          end

        self
      end

      # Predict the class of +data+ (an array of attribute values, no label).
      # The stored rules are evaluated in order and the first match wins; the
      # default class is returned if nothing matches. This is a plain Ruby
      # traversal of the rule data structure - it never calls Kernel#eval.
      def eval(data)
        @rules.each do |rule|
          return rule[:class] if rule_matches?(data, rule[:conditions])
        end
        @default_class
      end

      # Exports the rules as plain data: an ordered list of hashes (each with
      # +:class+ and +:conditions+) plus the +default_class+. Running the
      # conditions in the listed order reproduces #eval exactly.
      def to_rules
        {
          rules: @rules.map { |rule| deep_rule(rule) },
          default_class: @default_class
        }
      end

      # Renders executable Ruby code with an explicit if/elsif/else chain.
      # Useful for inspection. Prediction inside the classifier does not use
      # eval; this string is only produced on demand.
      def get_rules
        label = @data_set.category_label
        clauses = @rules.map { |rule| "#{conditions_to_ruby(rule[:conditions])} then #{label} = #{ruby_literal(rule[:class])}" }
        return "#{label} = #{ruby_literal(@default_class)}" if clauses.empty?

        out = +"if #{clauses.first}"
        clauses[1..].each { |clause| out << "\nelsif #{clause}" }
        out << "\nelse #{label} = #{ruby_literal(@default_class)}"
        out << "\nend"
        out
      end

      protected

      # A condition is a hash of the form:
      #   { attr_index: 1, op: :eq, value: 'red' }
      #   { attr_index: 0, op: :lte, value: 12.5 }
      #   { attr_index: 0, op: :gte, value: 12.5 }

      def induce_class_rules(target, uncovered_growth, phase:)
        class_rules = []
        pool = uncovered_growth
        loop do
          break if @max_rules_per_class && class_rules.size >= @max_rules_per_class
          break unless pool.any? { |item| item.last == target }

          rule = grow_rule(target, pool)
          pruned_rule, removed = prune_rule(rule, target, @prune_items)
          break unless acceptable_rule?(pruned_rule)

          current_dl = class_description_length(target, class_rules, @prune_items)
          candidate_dl = class_description_length(target, class_rules + [pruned_rule], @prune_items)
          if candidate_dl > current_dl
            @trace << { stage: :stop_dl, class: target, rule: deep_rule(pruned_rule),
                        removed: removed, current_dl: current_dl, candidate_dl: candidate_dl }
            break
          end

          class_rules << pruned_rule
          @trace << { stage: phase, class: target, rule: deep_rule(pruned_rule),
                      removed: removed, current_dl: current_dl, candidate_dl: candidate_dl }
          pool = pool.reject { |item| rule_matches?(item, pruned_rule[:conditions]) }
        end
        [class_rules, pool]
      end

      # RIPPER optimization pass. For every class the phase-1 rules are walked
      # in order. At each position the original rule competes against a
      # replacement grown from scratch and a revision grown on top of the
      # original rule's conditions; all three are pruned on the prune data and
      # the one with the smallest DL for the remaining data wins. Ties prefer
      # the shorter rule and then the revision, so the pass only churns when
      # it removes model cost without paying more error cost.
      def optimize_rules(initial_rules)
        final_rules = []
        @class_order.each do |target|
          class_rules = initial_rules.select { |rule| rule[:class] == target }
          pool = @growth_items.map(&:dup)

          class_rules.each do |original|
            replacement = grow_rule(target, pool)
            revision = grow_rule(target, pool, seed_conditions: original[:conditions])
            replacement, rep_removed = prune_rule(replacement, target, @prune_items)
            revision, rev_removed = prune_rule(revision, target, @prune_items)

            chosen, kind = choose_optimized(target, final_rules + [original],
                                            final_rules + [replacement],
                                            final_rules + [revision])
            if chosen.nil?
              chosen = original
              kind = :kept
            end
            @trace << { stage: :optimize, class: target, kind: kind,
                        original: deep_rule(original), replacement: deep_rule(replacement),
                        replacement_removed: rep_removed, revision: deep_rule(revision),
                        revision_removed: rev_removed, chosen: deep_rule(chosen) }
            final_rules << chosen
            pool = pool.reject { |item| rule_matches?(item, chosen[:conditions]) }
          end
        end
        final_rules
      end

      def choose_optimized(target, original_set, replacement_set, revision_set)
        candidates = [
          [original_set.last, :kept, class_description_length(target, original_set, @prune_items)],
          [replacement_set.last, :replacement, class_description_length(target, replacement_set, @prune_items)],
          [revision_set.last, :revision, class_description_length(target, revision_set, @prune_items)]
        ]
        valid = candidates.select { |rule, _kind, _dl| acceptable_rule?(rule) }
        return nil if valid.empty?

        best = valid.min_by { |rule, kind, dl| [dl, rule[:conditions].size, kind_priority(kind)] }
        [best[0], best[1]]
      end

      def kind_priority(kind)
        { revision: 0, replacement: 1, kept: 2 }.fetch(kind, 3)
      end

      # Greedy rule growing. Starts with an empty body (everything matches) and
      # repeatedly adds the condition with the highest positive FOIL
      # information gain until no negatives are covered, no attributes are
      # left, or no candidate improves the score. When +seed_conditions+ is
      # given (a revision), deleting an existing seed condition is also a
      # legal specialization step, which lets the optimizer escape a bad
      # condition carried over from phase 1.
      def grow_rule(target, items, seed_conditions: nil)
        @seed_conditions = seed_conditions ? deep_conditions(seed_conditions) : nil
        conditions = seed_conditions ? deep_conditions(seed_conditions) : []
        covered = items.select { |item| conditions_match?(item, conditions) }

        loop do
          positives = covered.count { |item| item.last == target }
          negatives = covered.size - positives
          break if negatives.zero?

          used = conditions.map { |condition| condition[:attr_index] }
          candidates = candidate_conditions(items, covered, target, used)
          add_deletion_candidates(candidates, conditions, items, target, @seed_conditions)
          break if candidates.empty?

          best = candidates.max_by { |candidate| [candidate[:gain], candidate[:positives], candidate[:tie_index]] }
          break if best[:gain] <= 0

          if best[:delete]
            deleted = conditions.delete_at(best[:delete])
            @seed_conditions&.delete(deleted)
            covered = items.select { |item| conditions_match?(item, conditions) }
          else
            conditions << best[:condition]
            covered = covered.select { |item| condition_matches?(item, best[:condition]) }
          end
        end

        @seed_conditions = nil
        { class: target, conditions: conditions }
      end

      # Deleting a seeded condition widens coverage, so its FOIL gain is
      # measured against the item set after deletion rather than the current
      # covered slice.
      def add_deletion_candidates(candidates, conditions, items, target, seed_conditions)
        return unless seed_conditions

        seed_conditions.each_with_index do |seed_condition, seed_index|
          index = conditions.index(seed_condition)
          next unless index

          widened = conditions.reject.with_index { |_, i| i == index }
          selected = items.select { |item| conditions_match?(item, widened) }
          positives = selected.count { |item| item.last == target }
          negatives = selected.size - positives
          gain = foil_gain(positives, negatives,
                           items.count { |item| item.last == target },
                           items.count { |item| item.last != target })
          candidates << { condition: nil, delete: index, gain: gain,
                          positives: positives, tie_index: 10_000 + seed_index }
        end
      end

      # Reduced-error pruning on the data held out from growing. Repeatedly
      # deletes whichever single condition maximizes
      # (p - n) / (p + n) on the prune data; equal scores still shorten the
      # rule, since an extra condition is pure overhead on noisy data.
      def prune_rule(rule, target, prune_data)
        conditions = deep_conditions(rule[:conditions])
        removed = []
        loop do
          current_score = pruning_score(conditions, target, prune_data)
          best_index = nil
          best_score = current_score
          conditions.each_index do |index|
            candidate_conditions = conditions.reject.with_index { |_, i| i == index }
            score = pruning_score(candidate_conditions, target, prune_data)
            next if score < best_score

            best_score = score
            best_index = index
          end
          break if best_index.nil?

          removed << conditions.delete_at(best_index)
        end
        [{ class: target, conditions: conditions }, removed]
      end

      def pruning_score(conditions, target, prune_data)
        covered = prune_data.select { |item| conditions_match?(item, conditions) }
        return -2.0 if covered.empty?

        positives = covered.count { |item| item.last == target }
        negatives = covered.size - positives
        (positives - negatives).to_f / covered.size
      end

      def acceptable_rule?(rule)
        rule && !rule[:conditions].empty?
      end

      def candidate_conditions(items, covered, target, used_attributes)
        candidates = []
        attribute_count = @data_set.num_attributes - 1
        total_positives = items.count { |item| item.last == target }
        total_negatives = items.size - total_positives

        attribute_count.times do |attr_index|
          next if used_attributes.include?(attr_index)

          if @numeric[attr_index]
            numeric_candidates(attr_index, covered, target,
                               total_positives, total_negatives, candidates)
          else
            nominal_candidates(attr_index, covered, target,
                               total_positives, total_negatives, candidates)
          end
        end
        candidates
      end

      def numeric_candidates(attr_index, covered, target,
                             total_positives, total_negatives, candidates)
        values = covered.map { |item| item[attr_index] }.uniq.sort
        values.each_cons(2).with_index do |(low, high), pair_index|
          next if low == high

          threshold = (low + high).to_f / 2
          [[:lte, threshold], [:gte, threshold]].each_with_index do |(op, _value), direction|
            condition = { attr_index: attr_index, op: op, value: threshold }
            selected = covered.select { |item| condition_matches?(item, condition) }
            next if selected.empty?

            positives = selected.count { |item| item.last == target }
            gain = foil_gain(positives, selected.size - positives,
                             total_positives, total_negatives)
            candidates << { condition: condition, gain: gain, positives: positives,
                            tie_index: -(attr_index * 1000 + pair_index * 2 + direction) }
          end
        end
      end

      def nominal_candidates(attr_index, covered, target,
                             total_positives, total_negatives, candidates)
        covered.map { |item| item[attr_index] }.uniq.each_with_index do |value, value_index|
          condition = { attr_index: attr_index, op: :eq, value: value }
          selected = covered.select { |item| condition_matches?(item, condition) }
          next if selected.empty?

          positives = selected.count { |item| item.last == target }
          gain = foil_gain(positives, selected.size - positives,
                           total_positives, total_negatives)
          candidates << { condition: condition, gain: gain, positives: positives,
                          tie_index: -(attr_index * 1000 + value_index) }
        end
      end

      def foil_gain(positives, negatives, total_positives, total_negatives)
        return 0.0 if positives.zero?

        prior = safe_ratio(total_positives, total_positives + total_negatives)
        current = safe_ratio(positives, positives + negatives)
        positives * (Math.log2(current) - Math.log2(prior))
      end

      def safe_ratio(numerator, denominator)
        denominator.zero? ? 0.5 : numerator.to_f / denominator
      end

      # Description length used both to stop adding rules and to choose between
      # replacement / revision / original in the optimization pass. It is the
      # classic simplified RIPPER measure:
      #
      # * each surviving condition costs a fixed number of bits to transmit;
      # * false positives (negatives covered by the rule set) are encoded at
      #   one bit each, as are target-class positives left uncovered;
      # * the genuine positives covered by the rules cost nothing.
      #
      # Consequently a rule that only memorizes noise pays model cost without
      # reducing error cost and is rejected.
      def class_description_length(target, rules, prune_data)
        covered_indices = rules.flat_map do |rule|
          prune_data.each_index.select { |index| rule_matches?(prune_data[index], rule[:conditions]) }
        end.uniq

        covered = covered_indices.map { |index| prune_data[index] }
        false_positives = covered.count { |item| item.last != target }
        total_positives = prune_data.count { |item| item.last == target }
        true_positives = covered.size - false_positives
        false_negatives = total_positives - true_positives

        model_cost = rules.sum { |rule| rule[:conditions].size }
        model_cost.to_f + false_positives + false_negatives
      end

      def rule_matches?(item_or_data, conditions)
        conditions.all? { |condition| condition_matches?(item_or_data, condition) }
      end

      def conditions_match?(item, conditions)
        conditions.all? { |condition| condition_matches?(item, condition) }
      end

      def condition_matches?(item, condition)
        value = item[condition[:attr_index]]
        case condition[:op]
        when :eq then value == condition[:value]
        when :lte then value.is_a?(Numeric) && value <= condition[:value]
        when :gte then value.is_a?(Numeric) && value >= condition[:value]
        else false
        end
      end

      def detect_numeric_attributes
        first_item = @data_set.data_items.first
        (0...(@data_set.num_attributes - 1)).map do |index|
          first_item[index].is_a?(Numeric)
        end
      end

      # Least frequent classes are learned first; ties follow first appearance
      # in the training data. The most frequent class becomes the default.
      def class_ordering(items)
        frequencies = Hash.new(0)
        first_seen = {}
        items.each_with_index do |item, index|
          label = item.last
          frequencies[label] += 1
          first_seen[label] ||= index
        end
        ordered = frequencies.keys.sort_by { |label| [frequencies[label], first_seen[label]] }
        default = @default_class || ordered.last
        [ordered[0...-1], default]
      end

      def split_items(items)
        shuffled =
          if @seed
            items.shuffle(random: Random.new(@seed))
          else
            items.shuffle
          end
        pivot = (items.size * (1.0 - @prune_ratio)).round
        pivot = [[pivot, 1].max, items.size - 1].min if items.size > 2
        [shuffled[0...pivot].map(&:dup), shuffled[pivot..].map(&:dup)]
      end

      def conditions_to_ruby(conditions)
        conditions.map do |condition|
          label = @data_set.data_labels[condition[:attr_index]]
          case condition[:op]
          when :eq then "#{label} == #{ruby_literal(condition[:value])}"
          when :lte then "#{label} <= #{ruby_literal(condition[:value])}"
          when :gte then "#{label} >= #{ruby_literal(condition[:value])}"
          end
        end.join(' and ')
      end

      def ruby_literal(value)
        value.is_a?(Numeric) ? value.to_s : value.inspect
      end

      def deep_rule(rule)
        { class: rule[:class], conditions: deep_conditions(rule[:conditions]) }
      end

      def deep_conditions(conditions)
        conditions.map(&:dup)
      end
    end
  end
end
