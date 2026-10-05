# frozen_string_literal: true

require_relative '../test_helper'
require 'ai4r/classifiers/ripper'
require 'ai4r/data/data_set'

class RipperTest < Minitest::Test
  include Ai4r::Classifiers
  include Ai4r::Data

  LABELS = %w[shape weight grade label].freeze

  # One heavy item is deliberately mislabelled ("square", 31, "C", "ok"):
  # it conflicts with the other heavy items and forces rule growing to add a
  # memorizing condition. The prune split contains the genuine heavy bad item
  # ("square", 32, "B"), which exposes that condition during pruning.
  NOISY_ITEMS = [
    ['square', 30, 'A', 'bad'],
    ['square', 32, 'B', 'bad'],
    ['circle', 8, 'A', 'ok'],
    ['circle', 10, 'B', 'ok'],
    ['circle', 12, 'A', 'ok'],
    ['circle', 9, 'C', 'ok'],
    ['triangle', 5, 'A', 'ok'],
    ['triangle', 7, 'B', 'ok'],
    ['square', 11, 'A', 'ok'],
    ['square', 13, 'C', 'ok'],
    ['triangle', 28, 'B', 'bad'],
    ['square', 31, 'C', 'ok']
  ].freeze

  def setup
    @data_set = DataSet.new(data_items: NOISY_ITEMS.dup.map(&:dup),
                            data_labels: LABELS)
    @classifier = Ripper.new.set_parameters(seed: 8).build(@data_set)
  end

  def test_build_requires_data
    assert_raises(ArgumentError) { Ripper.new.build(DataSet.new) }
  end

  def test_uses_separate_growth_and_pruning_data
    assert_equal 8, @classifier.growth_items.size
    assert_equal 4, @classifier.prune_items.size
    combined = @classifier.growth_items + @classifier.prune_items
    assert_equal @data_set.data_items.size, combined.size
    assert_equal [], combined.map(&:inspect) - @data_set.data_items.map(&:inspect)
  end

  def test_split_is_deterministic_with_seed
    other = Ripper.new.set_parameters(seed: 8)
                  .build(DataSet.new(data_items: NOISY_ITEMS.dup.map(&:dup), data_labels: LABELS))
    assert_equal @classifier.growth_items, other.growth_items
    assert_equal @classifier.prune_items, other.prune_items
  end

  def test_minority_class_is_learned_first_and_majority_is_default
    assert_equal ['bad'], @classifier.class_order
    assert_equal 'ok', @classifier.default_class
  end

  def test_single_class_uses_only_default
    pure = NOISY_ITEMS.reject { |item| item.last == 'bad' }
    classifier = Ripper.new.set_parameters(seed: 1)
                       .build(DataSet.new(data_items: pure.map(&:dup), data_labels: LABELS))
    assert_equal [], classifier.rules
    assert_equal 'ok', classifier.eval(['square', 99, 'Z'])
  end

  def test_numeric_threshold_rule_survives
    assert_equal 1, @classifier.rules.size
    conditions = @classifier.rules.first[:conditions]
    assert_equal [{ attr_index: 1, op: :gte, value: 20.5 }], conditions
    assert_equal 'bad', @classifier.rules.first[:class]
  end

  def test_memorizing_nominal_condition_was_grown_then_removed
    grown = @classifier.trace.find { |event| event[:stage] == :grow }
    refute_nil grown
    assert_equal [{ attr_index: 0, op: :eq, value: 'triangle' }], grown[:removed]
  end

  def test_description_length_stops_rule_growth_for_noise
    # After the clean minority pattern is learned, a lone mislabelled item
    # would require another long rule. The DL comparison must reject it.
    items = [
      ['x', 1, 'rare'],
      ['x', 2, 'rare'],
      ['x', 3, 'rare'],
      ['y', 4, 'common'],
      ['y', 5, 'common'],
      ['y', 6, 'common'],
      ['y', 7, 'common'],
      ['x', 8, 'common']
    ]
    classifier = Ripper.new.set_parameters(seed: 2, optimize: false)
                       .build(DataSet.new(data_items: items, data_labels: %w[color n label]))
    stop = classifier.trace.find { |event| event[:stage] == :stop_dl }
    refute_nil stop
    assert stop[:candidate_dl] > stop[:current_dl]
    assert classifier.rules.size <= 1
  end

  def test_first_matching_rule_wins_and_default_fallback
    assert_equal 'bad', @classifier.eval(['square', 32, 'B'])
    assert_equal 'bad', @classifier.eval(['triangle', 28, 'B'])
    assert_equal 'bad', @classifier.eval(['circle', 25, 'A'])
    assert_equal 'ok', @classifier.eval(['square', 13, 'C'])
    assert_equal 'ok', @classifier.eval(['triangle', 4, 'A'])
  end

  def test_conflicting_pair_is_not_memorized
    # The 31/"ok" item conflicts with the remaining heavy items. The learner
    # does not carve out its label with an extra condition: pruning removes
    # the shape condition, so this noisy item is misclassified as "bad" while
    # genuine heavy items keep their correct prediction.
    assert_equal 'bad', @classifier.eval(['square', 31, 'C'])
    assert_equal 'bad', @classifier.eval(['square', 32, 'B'])
    assert_equal 'bad', @classifier.eval(['square', 21, 'A'])
  end

  def test_exported_rules_are_plain_data
    exported = @classifier.to_rules
    assert_equal 'ok', exported[:default_class]
    assert_equal 'bad', exported[:rules].first[:class]
  end

  def test_exported_data_rules_reproduce_predictions_without_eval
    exported = @classifier.to_rules
    samples = [
      [['square', 32, 'B'], 'bad'],
      [['triangle', 28, 'B'], 'bad'],
      [['circle', 25, 'A'], 'bad'],
      [['square', 31, 'C'], 'bad'],
      [['square', 13, 'C'], 'ok'],
      [['triangle', 4, 'A'], 'ok'],
      [['circle', 8, 'C'], 'ok']
    ]
    samples.each do |attributes, expected|
      assert_equal expected, run_exported_rules(exported, attributes),
                   "exported rules disagree on #{attributes.inspect}"
      assert_equal expected, @classifier.eval(attributes)
    end
  end

  def test_rendered_ruby_chain_uses_first_match_and_else_default
    rendered = @classifier.get_rules
    refute_match(/Kernel|eval/, rendered)
    assert_equal "if weight >= 20.5 then label = \"bad\"\n" \
                 "else label = \"ok\"\nend", rendered

    samples = [
      [32, 'bad'], [21, 'bad'], [20.5, 'bad'], [20, 'ok'], [8, 'ok']
    ]
    samples.each do |weight, expected|
      label = nil
      binding.local_variable_set(:weight, weight)
      eval(rendered)
      assert_equal expected, label
    end
  end

  def test_discrete_equality_and_optimization_multi_class
    items = [
      %w[red small a],
      %w[red small a],
      %w[red large b],
      %w[blue small b],
      %w[blue large b],
      %w[green tiny c],
      %w[green tiny c]
    ]
    labels = %w[color size label]
    classifier = Ripper.new.set_parameters(seed: 3, optimize: true)
                       .build(DataSet.new(data_items: items, data_labels: labels))
    assert_equal 'b', classifier.default_class
    classifier.rules.each do |rule|
      rule[:conditions].each do |condition|
        assert_equal :eq, condition[:op]
      end
    end
    assert_equal classifier.eval(%w[red small]),
                 run_exported_rules(classifier.to_rules, %w[red small])
  end

  def test_optimization_can_be_disabled
    enabled = Ripper.new.set_parameters(seed: 8).build(@data_set)
    disabled = Ripper.new.set_parameters(seed: 8, optimize: false).build(@data_set)
    assert(enabled.trace.any? { |event| event[:stage] == :optimize })
    refute(disabled.trace.any? { |event| event[:stage] == :optimize })
  end

  def test_optimization_revises_phase_one_rule
    # With seed 53 phase 1 prunes the rule to weight >= 21.0, but the prune
    # split contains the genuine heavy item ("triangle", 28) which is already
    # above a lower, better threshold. The optimization pass revises the
    # threshold to 20.5 instead of keeping the slightly too-strict rule.
    classifier = Ripper.new.set_parameters(seed: 53).build(
      DataSet.new(data_items: NOISY_ITEMS.dup.map(&:dup), data_labels: LABELS)
    )
    grow_event = classifier.trace.find { |event| event[:stage] == :grow }
    assert_equal 21.0, grow_event[:rule][:conditions].first[:value]

    optimize_event = classifier.trace.find { |event| event[:stage] == :optimize }
    assert_equal :revision, optimize_event[:kind]
    assert_equal 20.5, classifier.rules.first[:conditions].first[:value]
    assert_equal 'bad', classifier.eval(['triangle', 28, 'B'])
  end

  private

  # Minimal interpreter of to_rules. It walks the ordered rule list and
  # returns on the first match, exactly like Ripper#eval; no Ruby eval is
  # involved in this verification.
  def run_exported_rules(exported, attributes)
    exported[:rules].each do |rule|
      matched = rule[:conditions].all? do |condition|
        value = attributes[condition[:attr_index]]
        case condition[:op]
        when :eq then value == condition[:value]
        when :lte then value <= condition[:value]
        when :gte then value >= condition[:value]
        else false
        end
      end
      return rule[:class] if matched
    end
    exported[:default_class]
  end
end
