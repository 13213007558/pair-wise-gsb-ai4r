# frozen_string_literal: true

require_relative '../test_helper'
require 'ai4r/classifiers/ripper_classifier'

class RipperClassifierTest < Minitest::Test
  include Ai4r::Classifiers
  include Ai4r::Data

  NOISY_EXAMPLES = [
    ['red', 'S', 0, 'F'], ['red', 'M', 1, 'F'], ['red', 'S', 2, 'F'],
    ['red', 'M', 3, 'F'], ['red', 'L', 4, 'F'],
    ['blue', 'L', 10, 'F'], ['blue', 'M', 11, 'F'], ['blue', 'S', 12, 'F'],
    ['blue', 'M', 13, 'F'], ['blue', 'L', 14, 'F'],
    ['red', 'S', 5, 'N'], ['green', 'L', 6, 'N'], ['red', 'M', 7, 'N'],
    ['green', 'S', 8, 'N'], ['blue', 'S', 9, 'N']
  ].freeze

  LABELS = %w[color size weight label].freeze

  def build_classifier(**parameters)
    data = DataSet.new(data_items: NOISY_EXAMPLES, data_labels: LABELS)
    RipperClassifier.new.set_parameters(**{ random_seed: 2, default_class: 'N' }.merge(parameters)).build(data)
  end

  def noisy_dataset
    DataSet.new(data_items: NOISY_EXAMPLES, data_labels: LABELS)
  end

  def test_requires_training_data
    assert_raises(ArgumentError) { RipperClassifier.new.build(DataSet.new) }
  end

  def test_validates_parameters
    assert_raises(ArgumentError) { RipperClassifier.new.set_parameters(random_seed: 2, grow_ratio: 1).build(noisy_dataset) }
    assert_raises(ArgumentError) { RipperClassifier.new.set_parameters(default_class: 'Z').build(noisy_dataset) }
  end

  def test_fixed_class_order_default_and_first_match_semantics
    classifier = build_classifier
    assert_equal(%w[F], classifier.class_order)
    assert_equal('N', classifier.default_class)
    exported = classifier.export_rules
    assert_equal(:first_match, exported[:semantics][:evaluation])
    assert_equal('N', exported[:semantics][:default_class])
  end

  def test_continuous_thresholds_avoid_covering_middle_noise
    classifier = build_classifier
    initial = classifier.rule_traces.select { |item| item[:phase] == :initial }
    operators = initial.map { |item| item[:pruned_conditions].first[:operator] }
    thresholds = initial.map { |item| item[:pruned_conditions].first[:value] }

    assert_equal(['<=', '>'], operators)
    assert_equal([4.5, 9.5], thresholds)
    assert(initial.all? { |item| item[:pruned_conditions].length == 1 })
    assert_equal('N', classifier.eval(['red', 'S', 7]))
  end

  def test_pruning_deletes_discrete_condition_from_grown_rule
    items = []
    5.times { |i| items << ['red', i.even? ? 'S' : 'M', i, 'F'] }
    5.times { |i| items << ['blue', i.even? ? 'M' : 'L', 10 + i, 'F'] }
    5.times { |i| items << [%w[red green][i % 2], i.even? ? 'S' : 'L', 5 + i, 'N'] }
    data = DataSet.new(data_items: items, data_labels: %w[color size weight label])
    classifier = RipperClassifier.new.set_parameters(random_seed: 2).build(data)
    trace = classifier.rule_traces.find { |item| item[:phase] == :initial }

    assert_equal([2, 0], trace[:grown_conditions].map { |condition| condition[:attribute_index] })
    assert_equal([{ attribute_index: 0, operator: '==', value: 'red' }],
                 trace[:removed_conditions])
    assert_equal([{ attribute_index: 2, operator: '>', value: 4.5 }],
                 trace[:pruned_conditions])
    assert_equal('N', classifier.eval(['red', 'S', 6]))
  end

  def test_growth_and_pruning_use_separate_data
    classifier = build_classifier
    traces = classifier.rule_traces.select { |item| item[:phase] == :initial }
    grow = classifier.phase_splits['F'][:grow]
    prune = classifier.phase_splits['F'][:prune]

    assert(traces.all? { |trace| trace[:grow_examples].positive? })
    assert(traces.all? { |trace| trace[:prune_examples].positive? })
    assert_empty(grow & prune)
    assert_equal(NOISY_EXAMPLES.length, grow.length + prune.length)
  end

  def test_description_length_stops_additional_rules
    classifier = build_classifier
    traces = classifier.rule_traces.select { |item| item[:phase] == :initial }

    assert(traces.all? { |trace| trace[:accepted] })
    assert(traces.all? { |trace| trace[:after_description_length] < trace[:before_description_length] })
    assert_equal(2, classifier.rules.length)
  end

  def test_optimization_reports_replacement_revision_and_choice
    classifier = build_classifier
    traces = classifier.rule_traces.select { |item| item[:phase] == :optimization }

    traces.each do |trace|
      assert_includes(%i[kept replaced revised], trace[:action])
      assert(trace.key?(:original_conditions))
      assert(trace.key?(:replacement_conditions))
      assert(trace.key?(:revision_conditions))
    end
    assert_equal(classifier.rules.map { |rule| rule[:conditions] },
                 traces.map { |trace| trace[:selected_conditions] })
  end

  def test_optimization_chooses_lower_description_length_candidate
    data = noisy_dataset
    classifier = RipperClassifier.new.set_parameters(default_class: 'N').build(data)
    first_rule = { conditions: [{ attribute_index: 2, operator: '<=', value: 4.5 }], class_value: 'F' }
    long_condition = { attribute_index: 1, operator: '==', value: 'S' }
    short_condition = { attribute_index: 2, operator: '>', value: 9.5 }
    original = { conditions: [long_condition], class_value: 'F' }
    candidates = [
      [:kept, [long_condition]],
      [:replaced, [short_condition]],
      [:revised, [short_condition]]
    ]

    selected = classifier.send(:choose_candidate, [first_rule], [], [original], 0, 'F', candidates, NOISY_EXAMPLES)
    kept_dl = classifier.send(:description_length, [first_rule, original], NOISY_EXAMPLES)
    replacement_dl = classifier.send(:description_length,
                                     [first_rule, { conditions: [short_condition], class_value: 'F' }],
                                     NOISY_EXAMPLES)
    assert(replacement_dl < kept_dl)

    assert_equal(:replaced, selected[:action])
    assert_equal([short_condition], selected[:conditions])
  end

  def test_prediction_uses_first_matching_rule_then_default
    classifier = build_classifier
    first_rule = { conditions: [{ attribute_index: 0, operator: '==', value: 'red' }], class_value: 'A' }
    second_rule = { conditions: [{ attribute_index: 2, operator: '>', value: 100 }], class_value: 'B' }
    classifier.instance_variable_set(:@rules, [first_rule, second_rule])

    assert_equal('A', classifier.eval(['red', 'S', 200]))
    assert_equal('B', classifier.eval(['blue', 'S', 200]))
    assert_equal('N', classifier.eval(['blue', 'S', 0]))
  end

  def test_exported_rules_execute_without_eval_and_match_classifier
    classifier = build_classifier
    exported = classifier.export_rules
    unseen_examples = [
      ['red', 'S', 5], ['red', 'M', 4], ['blue', 'L', 5],
      ['green', 'S', 20], ['blue', 'M', 10], ['green', 'L', 0]
    ]

    unseen_examples.each do |features|
      assert_equal(classifier.eval(features),
                   RipperClassifier.execute_rules(exported, features))
      assert_equal(classifier.eval(features),
                   classifier.execute_exported_rules(features))
    end
  end

  def test_exported_rules_are_plain_data
    exported = build_classifier.export_rules

    assert_equal(1, exported[:version])
    assert_equal('<=', exported[:rules].first[:conditions].first[:operator])
    condition = exported[:rules].first[:conditions].first
    assert_kind_of(Integer, condition[:attribute_index])
    assert_kind_of(String, condition[:operator])
    assert_kind_of(Float, condition[:value])
  end

  def test_get_rules_is_textual_decision_list
    rules_text = build_classifier.get_rules

    assert_includes(rules_text, 'first matching rule wins')
    assert_includes(rules_text, 'weight <= 4.5')
    assert_includes(rules_text, 'weight > 9.5')
    assert_includes(rules_text, "DEFAULT label = 'N'")
    refute_includes(rules_text, 'eval')
  end
end
