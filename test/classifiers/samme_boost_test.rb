# frozen_string_literal: true

require 'ai4r/classifiers/samme_boost'
require 'ai4r/data/data_set'
require_relative '../test_helper'

class SAMMEBoostTest < Minitest::Test
  include Ai4r::Classifiers
  include Ai4r::Data

  TRUE_LABEL = { 0 => 'red', 1 => 'red', 2 => 'green', 3 => 'green', 4 => 'blue', 5 => 'blue' }.freeze

  class ScriptedWeakLearner
    attr_reader :predictions, :received_weights, :received_labels

    def initialize(predictions)
      @predictions = predictions
      @received_weights = nil
    end

    def build(data_set, sample_weights)
      @received_weights = sample_weights.dup
      @received_labels = data_set.data_items.map(&:last)
      self
    end

    def eval(data)
      @predictions.fetch(data.first)
    end
  end

  class CyclingWeakLearner
    attr_reader :iteration, :wrong_index, :received_weights, :received_labels

    def initialize(iteration)
      @iteration = iteration
      @wrong_index = 0
    end

    def build(data_set, sample_weights)
      @received_weights = sample_weights.dup
      @received_labels = data_set.data_items.map(&:last)
      @wrong_index = sample_weights.each_with_index.max_by(&:first).last
      self
    end

    def eval(data)
      index = data.first
      return TRUE_LABEL.fetch(index) unless index == wrong_index

      labels = %w[red green blue]
      current = labels.index(TRUE_LABEL.fetch(index))
      labels[(current + 1) % labels.length]
    end
  end

  def setup
    @data_set = DataSet.new(
      data_items: TRUE_LABEL.map { |feature, label| [feature, label] },
      data_labels: %w[feature color]
    )
  end

  def test_multiclass_rounds_have_auditable_weights_and_predictions
    boost = build_boost(3, scripted_factory)
    records = boost.rounds

    assert_equal %w[red green blue], boost.classes
    assert_equal({ 'red' => 0, 'green' => 1, 'blue' => 2 }, records.first.class_mapping)
    assert_equal [1.0 / 6] * 6, records.first.sample_weights
    assert_in_delta 0.5, records.first.weighted_error
    assert_in_delta Math.log(2), records.first.learner_weight
    assert_equal [2.0 / 9, 1.0 / 9, 2.0 / 9, 1.0 / 9, 2.0 / 9, 1.0 / 9],
                 records.first.updated_sample_weights
    assert_equal :added, records.first.status

    assert_in_delta 2.0 / 9, records[1].weighted_error
    assert_in_delta Math.log(7), records[1].learner_weight
    expected_second_weights = [14.0 / 51, 1.0 / 51, 14.0 / 51, 1.0 / 51, 14.0 / 51, 7.0 / 51]
    expected_second_weights.each_with_index do |expected, index|
      assert_in_delta expected, records[1].updated_sample_weights[index], 1e-15
    end

    assert_in_delta 0.0, records[2].weighted_error
    assert_equal Float::INFINITY, records[2].learner_weight
    assert_equal :zero_error, records[2].status
    expected_second_weights.each_with_index do |expected, index|
      assert_in_delta expected, records[2].updated_sample_weights[index], 1e-15
    end
    assert_equal 3, records.length

    assert_equal 'green', boost.eval([5], round: 1)
    assert_equal({ 'red' => 0.0, 'green' => Math.log(2), 'blue' => 0.0 },
                 boost.class_scores([5], round: 1))
    assert_equal 'blue', boost.eval([5], round: 2)
    assert_in_delta Math.log(2), boost.class_scores([5], round: 2)['green']
    assert_in_delta Math.log(7), boost.class_scores([5], round: 2)['blue']
    assert_equal 'blue', boost.eval([5])
    assert_equal Float::INFINITY, boost.class_scores([5])['blue']
  end

  def test_weak_learner_factory_receives_sample_weights
    factory = scripted_factory
    build_boost(1, factory)

    learner = factory[:learners].first
    assert_equal [1.0 / 6] * 6, learner.received_weights
    assert_in_delta 1.0, learner.received_weights.sum
    assert_equal %w[red green blue], learner.received_labels.uniq
  end

  def test_same_random_source_records_identical_rounds
    first_factory = scripted_factory
    second_factory = scripted_factory
    first = build_boost(3, first_factory, Random.new(123))
    second = build_boost(3, second_factory, Random.new(123))

    assert_equal round_summary(first), round_summary(second)
    assert_equal first_factory[:draws], second_factory[:draws]
  end

  def test_rejects_unknown_weak_learner_labels
    unknown_predictions = { 0 => :unexpected, 1 => :unexpected, 2 => 'green', 3 => 'green', 4 => 'blue', 5 => 'blue' }
    error = assert_raises(ArgumentError) { build_boost(1, simple_factory([unknown_predictions])) }
    assert_match(/unknown class/, error.message)
  end

  def test_round_must_not_exceed_trained_rounds
    boost = build_boost(1, simple_factory([always_correct_plan]))
    error = assert_raises(ArgumentError) { boost.eval([0], round: 2) }
    assert_equal 'round 2 exceeds 1 trained round(s).', error.message
  end

  def test_stops_on_random_guess_boundary
    predictions = {
      0 => 'green',
      1 => 'green',
      2 => 'red',
      3 => 'green',
      4 => 'red',
      5 => 'blue'
    }
    factory = simple_factory([predictions, always_correct_plan])
    boost = build_boost(2, factory)
    record = boost.rounds.first

    assert_equal 1, boost.rounds.length
    assert_equal :random_guess, record.status
    assert_in_delta 2.0 / 3, record.weighted_error
    assert_equal 0.0, record.learner_weight
    assert_equal [1.0 / 6] * 6, record.updated_sample_weights
    error = assert_raises(RuntimeError) { boost.eval([0]) }
    assert_match(/No weak learner was accepted/, error.message)
  end

  def test_stops_when_weak_learner_is_worse_than_random
    predictions = 0.upto(5).to_h { |index| [index, TRUE_LABEL.fetch(index) == 'red' ? 'green' : 'red'] }
    factory = simple_factory([predictions, always_correct_plan])
    boost = build_boost(2, factory)
    record = boost.rounds.first

    assert_equal :worse_than_random, record.status
    assert_operator record.weighted_error, :>, 2.0 / 3
    assert_nil record.learner_weight
    assert_equal [1.0 / 6] * 6, record.updated_sample_weights
    assert_equal 1, boost.rounds.length
  end

  def test_zero_error_round_gets_infinite_weight
    boost = build_boost(2, simple_factory([always_correct_plan, always_correct_plan]))

    assert_equal 1, boost.rounds.length
    assert_equal :zero_error, boost.rounds.first.status
    assert_equal Float::INFINITY, boost.rounds.first.learner_weight
    assert_equal 'red', boost.eval([1])
  end

  def test_training_requires_multiple_classes
    data_set = DataSet.new(data_items: [[0, 'only'], [1, 'only']], data_labels: %w[x label])
    boost = SAMMEBoost.new.set_parameters(n_estimators: 1, weak_learner_factory: -> { ScriptedWeakLearner.new({}) })

    error = assert_raises(ArgumentError) { boost.build(data_set) }
    assert_equal 'SAMME requires at least two classes.', error.message
  end

  def test_long_run_keeps_a_valid_weight_distribution
    boost = SAMMEBoost.new.set_parameters(
      n_estimators: 200,
      random_seed: 9,
      weak_learner_factory: cycling_factory
    ).build(@data_set)

    assert_equal 200, boost.rounds.length
    boost.rounds.each do |record|
      assert(record.sample_weights.all? { |weight| weight > 0.0 })
      assert_in_delta 1.0, record.sample_weights.sum
      assert(record.updated_sample_weights.all? { |weight| weight > 0.0 })
      assert_in_delta 1.0, record.updated_sample_weights.sum
      assert_operator record.weighted_error, :<, 2.0 / 3
    end
  end

  private

  def build_boost(estimators, factory, random_seed = 7)
    SAMMEBoost.new.set_parameters(
      n_estimators: estimators,
      random_seed: random_seed,
      weak_learner_factory: factory[:callable]
    ).build(@data_set)
  end

  def scripted_factory
    simple_factory([
                     { 0 => 'red', 1 => 'blue', 2 => 'green', 3 => 'red', 4 => 'blue', 5 => 'green' },
                     { 0 => 'red', 1 => 'blue', 2 => 'green', 3 => 'red', 4 => 'blue', 5 => 'blue' },
                     always_correct_plan
                   ])
  end

  def always_correct_plan
    TRUE_LABEL.dup
  end

  def simple_factory(plans)
    learners = []
    draws = []
    queued_plans = plans.dup
    callable = lambda do |rng|
      draws << rng.rand
      learner = ScriptedWeakLearner.new(queued_plans.shift)
      learners << learner
      learner
    end
    { callable: callable, learners: learners, draws: draws }
  end

  def cycling_factory
    @cycling_iteration ||= -1
    lambda {
      @cycling_iteration += 1
      CyclingWeakLearner.new(@cycling_iteration)
    }
  end

  def round_summary(boost)
    boost.rounds.map do |record|
      [
        record.iteration,
        record.classes,
        record.class_mapping,
        record.weighted_error,
        record.learner_weight,
        record.sample_weights,
        record.updated_sample_weights,
        record.status,
        record.learner.predictions
      ]
    end
  end
end
