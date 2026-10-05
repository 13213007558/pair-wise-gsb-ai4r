# frozen_string_literal: true

require_relative '../data/data_set'
require_relative '../classifiers/classifier'
module Ai4r
  module Classifiers
    # Ordered decision-list learner with separate grow/prune data, IREP-style
    # pruning, description-length stopping, and RIPPER replacement/revision.
    class RipperClassifier < Classifier
      OPERATORS = %w[== <= >].freeze
      attr_accessor :default_class
      attr_reader :data_set, :rules, :class_frequencies, :class_order, :rule_traces, :phase_splits
      attr_writer :grow_ratio, :random_seed, :rule_cost, :condition_cost, :dl_tolerance, :optimize

      parameters_info(
        grow_ratio: 'Fraction of each stratified split used for growing rules.',
        random_seed: 'Seed used for deterministic grow/prune splitting.',
        rule_cost: 'Description-length cost, in bits, of adding a rule.',
        condition_cost: 'Description-length cost, in bits, of one condition.',
        dl_tolerance: 'Minimum description-length improvement needed to keep a rule.',
        optimize: 'Run RIPPER replacement and revision passes.',
        default_class: 'Class returned when no learned or exported rule matches.'
      )
      def initialize
        super
        @grow_ratio = 2.0 / 3.0
        @random_seed = nil
        @rule_cost = 1.0
        @condition_cost = 1.0
        @dl_tolerance = 0.0
        @optimize = true
        @default_class = nil
        @rules = []
        @rule_traces = []
        @phase_splits = {}
      end

      def build(data_set)
        data_set.check_not_empty
        validate_parameters
        @data_set = data_set
        @examples = data_set.data_items.map(&:dup)
        @numeric_attributes = detect_numeric_attributes
        @class_frequencies = count_classes
        configured_default = @default_class
        @default_class = configured_default.nil? ? most_frequent_class : configured_default
        raise ArgumentError, 'default_class must be a class observed during training' unless @class_frequencies.key?(@default_class)

        @class_order = ordered_classes
        @rules = []
        @rule_traces = []
        @phase_splits = {}
        rng = @random_seed.nil? ? Random.new : Random.new(@random_seed)
        @class_order.each { |klass| learn_class(klass, rng) }
        simplify_rules! if @optimize
        self
      end

      def eval(instance)
        rule = matching_rule(instance, @rules)
        rule ? rule[:class_value] : @default_class
      end

      def get_rules
        lines = ['RULE TABLE (first matching rule wins)']
        @rules.each_with_index do |rule, index|
          lines << "#{index + 1}. IF #{describe_conditions(rule[:conditions])} THEN #{@data_set.category_label} = #{format_value(rule[:class_value])}"
        end
        lines << "DEFAULT #{@data_set.category_label} = #{format_value(@default_class)}"
        lines.join("\n")
      end

      def export_rules
        {
          version: 1,
          semantics: { evaluation: :first_match, default_class: @default_class, class_learning_order: @class_order },
          rules: @rules.map { |rule| deep_dup_rule(rule) }
        }
      end

      def execute_exported_rules(instance, exported = export_rules)
        self.class.execute_rules(exported, instance)
      end

      def self.execute_rules(exported, instance)
        exported[:rules].each do |rule|
          return rule[:class_value] if rule_matches_exported?(rule, instance)
        end
        exported[:semantics][:default_class]
      end

      def self.rule_matches_exported?(rule, instance)
        rule[:conditions].all? { |condition| condition_matches?(condition, instance) }
      end

      def self.condition_matches?(condition, instance)
        value = instance[condition[:attribute_index]]
        case condition[:operator]
        when '==' then value == condition[:value]
        when '<=' then value <= condition[:value]
        when '>' then value > condition[:value]
        else raise ArgumentError, "Unknown operator: #{condition[:operator]}"
        end
      end

      protected

      def validate_parameters
        raise ArgumentError, 'grow_ratio must be between 0 and 1' unless (0...1).cover?(@grow_ratio)
        return unless @rule_cost.negative? || @condition_cost.negative? || @dl_tolerance.negative?

        raise ArgumentError, 'Description-length costs and tolerance must not be negative'
      end

      def learn_class(klass, rng)
        context = @rules.dup
        class_examples = @examples.reject { |example| covered_by?(example, context) }
        grow_data, prune_data = stratified_split(class_examples, rng)
        @phase_splits[klass] = { grow: grow_data, prune: prune_data }
        remaining = class_examples.dup
        initial_rules = []
        loop do
          break unless remaining.any? { |example| example.last == klass }

          rule_context = context + initial_rules
          rule_grow = grow_data.reject { |example| covered_by?(example, rule_context) }
          rule_prune = prune_data.reject { |example| covered_by?(example, rule_context) }
          break if rule_grow.none? { |example| example.last == klass }

          grown = grow_rule(rule_grow, klass, [])
          pruned = prune_rule(grown, rule_prune, klass)
          candidate = { conditions: pruned, class_value: klass }
          before_dl = description_length(rule_context, remaining)
          after_dl = description_length(rule_context + [candidate], remaining)
          accepted = after_dl < before_dl - @dl_tolerance
          @rule_traces << {
            phase: :initial,
            class_value: klass,
            grown_conditions: deep_dup_conditions(grown),
            pruned_conditions: deep_dup_conditions(pruned),
            removed_conditions: conditions_difference(grown, pruned),
            accepted: accepted,
            before_description_length: before_dl,
            after_description_length: after_dl,
            grow_examples: rule_grow.length,
            prune_examples: rule_prune.length
          }
          break unless accepted

          initial_rules << candidate
          remaining.reject! { |example| rule_match?(example, candidate) }
        end
        optimized = optimize_class_rules(klass, context, initial_rules, grow_data, prune_data, class_examples)
        @rules = context + optimized
      end

      def optimize_class_rules(klass, context, class_rules, grow_data, prune_data, class_examples)
        return class_rules unless @optimize

        class_rules.each_with_index do |original, index|
          prior = context + class_rules[0...index]
          replacement_grow = grow_data.reject { |example| covered_by?(example, prior) }
          revision_grow = grow_data.select { |example| rule_match?(example, original) }.reject { |example| covered_by?(example, prior) }
          replacement_conditions = grow_rule(replacement_grow, klass, [])
          revision_conditions = revision_grow.any? { |example| example.last == klass } ? grow_rule(revision_grow, klass, original[:conditions]) : deep_dup_conditions(original[:conditions])
          prune_context = prune_data.reject { |example| covered_by?(example, prior) }
          original_conditions = deep_dup_conditions(original[:conditions])
          candidates = [
            [:kept, original_conditions],
            [:replaced, prune_rule(replacement_conditions, prune_context, klass)],
            [:revised, prune_rule(revision_conditions, prune_context, klass)]
          ]
          selected = choose_candidate(context, prior, class_rules, index, klass, candidates, class_examples)
          class_rules[index] = { conditions: selected[:conditions], class_value: klass }
          @rule_traces << {
            phase: :optimization,
            class_value: klass,
            rule_index: context.length + index,
            action: selected[:action],
            original_conditions: deep_dup_conditions(original_conditions),
            replacement_conditions: deep_dup_conditions(candidates[1][1]),
            revision_conditions: deep_dup_conditions(candidates[2][1]),
            selected_conditions: deep_dup_conditions(selected[:conditions]),
            description_length: selected[:description_length]
          }
        end
        class_rules
      end

      def choose_candidate(context, _prior, class_rules, index, klass, candidates, examples)
        selected = nil
        candidates.each do |action, conditions|
          candidate_rules = class_rules.dup
          candidate_rules[index] = { conditions: conditions, class_value: klass }
          dl = description_length(context + candidate_rules[index..], examples)
          candidate = { action: action, conditions: deep_dup_conditions(conditions), description_length: dl }
          selected = candidate if selected.nil? || dl < selected[:description_length]
        end
        selected
      end

      def simplify_rules!
        changed = true
        while changed
          changed = false
          @rules.each_with_index do |_rule, index|
            candidate = @rules.dup
            candidate.delete_at(index)
            next unless description_length(candidate, @examples) < description_length(@rules, @examples) - @dl_tolerance

            removed = @rules.delete_at(index)
            @rule_traces << { phase: :simplification, action: :removed, class_value: removed[:class_value], conditions: deep_dup_conditions(removed[:conditions]) }
            changed = true
            break
          end
        end
      end

      def grow_rule(data, target, initial_conditions)
        conditions = deep_dup_conditions(initial_conditions)
        used = conditions.map { |condition| condition[:attribute_index] }
        available = attribute_indices - used
        current = data.select { |example| conditions_match?(example, conditions) }
        loop do
          positives = current.count { |example| example.last == target }
          negatives = current.length - positives
          break if positives.zero? || negatives.zero? || available.empty?

          candidates = candidate_conditions(current, target, available)
          best = candidates.max_by { |candidate| candidate[:gain] }
          break if best.nil? || best[:gain] <= 0

          conditions << best[:condition]
          available.delete(best[:condition][:attribute_index])
          current = current.select { |example| condition_match?(example, best[:condition]) }
        end
        conditions
      end

      def candidate_conditions(data, target, available)
        candidates = []
        positive_count = data.count { |example| example.last == target }
        prior_logit = smoothed_logit(positive_count, data.length)
        available.each do |index|
          if @numeric_attributes.include?(index)
            numeric_candidates(data, target, index, prior_logit, candidates)
          else
            nominal_candidates(data, target, index, prior_logit, candidates)
          end
        end
        candidates
      end

      def numeric_candidates(data, target, index, prior_logit, candidates)
        values = data.map { |example| example[index] }.uniq.sort
        thresholds = values.each_cons(2).map { |left, right| (left + right) / 2.0 }
        thresholds.each do |threshold|
          ['<=', '>'].each do |operator|
            condition = { attribute_index: index, operator: operator, value: threshold }
            matching = data.select { |example| condition_match?(example, condition) }
            candidates << { condition: condition, gain: foil_gain(matching, target, prior_logit) }
          end
        end
      end

      def nominal_candidates(data, target, index, prior_logit, candidates)
        data.map { |example| example[index] }.uniq.sort_by(&:to_s).each do |value|
          condition = { attribute_index: index, operator: '==', value: value }
          matching = data.select { |example| condition_match?(example, condition) }
          candidates << { condition: condition, gain: foil_gain(matching, target, prior_logit) }
        end
      end

      def foil_gain(data, target, prior_logit)
        positive = data.count { |example| example.last == target }
        return 0.0 if positive.zero?

        positive * (smoothed_logit(positive, data.length) - prior_logit)
      end

      def smoothed_logit(positive, total)
        probability = (positive + 0.5) / (total + 1.0)
        Math.log(probability / (1.0 - probability))
      end

      def prune_rule(conditions, data, target)
        return deep_dup_conditions(conditions) if data.empty?

        best_conditions = deep_dup_conditions(conditions)
        best_score = rule_score(best_conditions, data, target)
        loop do
          candidate = nil
          best_conditions.each_index do |index|
            prefix = best_conditions[0...index] + best_conditions[(index + 1)..]
            next if prefix.empty?

            candidate_score = rule_score(prefix, data, target)
            if candidate_score > best_score
              best_score = candidate_score
              candidate = prefix
            end
          end
          break if candidate.nil?

          best_conditions = candidate
        end
        best_conditions
      end

      def rule_score(conditions, data, target)
        matching = data.select { |example| conditions_match?(example, conditions) }
        return -Float::INFINITY if matching.empty?

        positive = matching.count { |example| example.last == target }
        negative = matching.length - positive
        (positive - negative).to_f / matching.length
      end

      def description_length(rules, examples)
        theory = rules.sum(0.0) { |rule| @rule_cost + rule[:conditions].length * @condition_cost }
        buckets = Hash.new { |hash, key| hash[key] = { positive: 0, negative: 0 } }
        examples.each do |example|
          rule = matching_rule(example, rules)
          predicted = rule ? rule[:class_value] : @default_class
          bucket = buckets[predicted]
          if example.last == predicted
            bucket[:positive] += 1
          else
            bucket[:negative] += 1
          end
        end
        data_cost = buckets.values.sum { |counts| bernoulli_bits(counts[:positive], counts[:negative]) }
        theory + data_cost
      end

      def bernoulli_bits(positive, negative)
        total = positive + negative
        return 0.0 if total.zero? || positive.zero? || negative.zero?

        probability_positive = positive.to_f / total
        probability_negative = negative.to_f / total
        -(positive * Math.log2(probability_positive) + negative * Math.log2(probability_negative))
      end

      def stratified_split(examples, rng)
        grow = []
        prune = []
        examples.group_by(&:last).sort_by { |klass, _| klass.to_s }.each do |_klass, group|
          shuffled = group.shuffle(random: rng)
          grow_count = (shuffled.length * @grow_ratio).round
          grow_count = [[grow_count, 1].max, shuffled.length - 1].min
          grow.concat(shuffled[0...grow_count])
          prune.concat(shuffled[grow_count..])
        end
        raise ArgumentError, 'grow_ratio must be between 0 and 1' unless (0...1).cover?(@grow_ratio)

        [grow, prune]
      end

      def matching_rule(instance, rules)
        rules.find { |rule| rule_match?(instance, rule) }
      end

      def covered_by?(instance, rules)
        !matching_rule(instance, rules).nil?
      end

      def rule_match?(instance, rule)
        conditions_match?(instance, rule[:conditions])
      end

      def conditions_match?(instance, conditions)
        conditions.all? { |condition| condition_match?(instance, condition) }
      end

      def condition_match?(instance, condition)
        value = instance[condition[:attribute_index]]
        case condition[:operator]
        when '==' then value == condition[:value]
        when '<=' then value <= condition[:value]
        when '>' then value > condition[:value]
        else false
        end
      end

      def attribute_indices
        (0...@examples.first.length - 1).to_a
      end

      def detect_numeric_attributes
        first = @examples.first
        attribute_indices.select { |index| first[index].is_a?(Numeric) }
      end

      def count_classes
        @examples.each_with_object(Hash.new(0)) { |example, counts| counts[example.last] += 1 }
      end

      def most_frequent_class
        @class_frequencies.min_by { |klass, count| [-count, klass.to_s] }
                          .first
      end

      def ordered_classes
        @class_frequencies.reject { |klass, _| klass == @default_class }
                          .sort_by { |klass, count| [count, klass.to_s] }
                          .map(&:first)
      end

      def deep_dup_rule(rule)
        { conditions: deep_dup_conditions(rule[:conditions]), class_value: rule[:class_value] }
      end

      def deep_dup_conditions(conditions)
        conditions.map(&:dup)
      end

      def conditions_difference(grown, pruned)
        grown.reject { |condition| pruned.any? { |remaining| condition_signature(remaining) == condition_signature(condition) } }.map(&:dup)
      end

      def condition_signature(condition)
        [condition[:attribute_index], condition[:operator], condition[:value]]
      end

      def describe_conditions(conditions)
        return 'true' if conditions.empty?

        conditions.map { |condition| describe_condition(condition) }.join(' AND ')
      end

      def describe_condition(condition)
        label = @data_set.data_labels[condition[:attribute_index]]
        "#{label} #{condition[:operator]} #{format_value(condition[:value])}"
      end

      def format_value(value)
        value.is_a?(String) ? "'#{value}'" : value.inspect
      end
    end
  end
end
