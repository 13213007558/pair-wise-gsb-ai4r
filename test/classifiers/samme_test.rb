# frozen_string_literal: true

require 'ai4r/classifiers/samme'
require 'ai4r/data/data_set'
require_relative '../test_helper'

# Decision stump that minimizes the weighted training error. When several
# splits tie, the first one wins, unless an RNG is given, in which case the
# split is sampled from the tied candidates.
class WeightedStump
  TIE_TOLERANCE = 1e-12

  def initialize(rng = nil)
    @rng = rng
  end

  def build(data_set, weights)
    items = data_set.data_items
    candidates = []
    (items.first.length - 1).times do |feature|
      values = items.map { |item| item[feature] }.uniq.sort
      values.each_cons(2) do |low, high|
        candidates << score_candidate(feature, (low + high) / 2.0, items, weights)
      end
    end
    best_error = candidates.map { |candidate| candidate[:error] }.min
    tied = candidates.select { |candidate| (candidate[:error] - best_error).abs < TIE_TOLERANCE }
    choose(@rng ? tied.sample(random: @rng) : tied.first)
    self
  end

  def eval(data)
    data[@feature] <= @threshold ? @left_class : @right_class
  end

  private

  def choose(candidate)
    @feature = candidate[:feature]
    @threshold = candidate[:threshold]
    @left_class = candidate[:left_class]
    @right_class = candidate[:right_class]
  end

  def score_candidate(feature, threshold, items, weights)
    left = Hash.new(0.0)
    right = Hash.new(0.0)
    items.each_with_index do |item, index|
      (item[feature] <= threshold ? left : right)[item.last] += weights[index]
    end
    left_class = left.max_by { |_, weight| weight }.first
    right_class = right.max_by { |_, weight| weight }.first
    error = 0.0
    items.each_with_index do |item, index|
      predicted = item[feature] <= threshold ? left_class : right_class
      error += weights[index] if predicted != item.last
    end
    { feature: feature, threshold: threshold, left_class: left_class,
      right_class: right_class, error: error }
  end
end

# Learner that always predicts the same label.
class ConstantLearner
  def initialize(label)
    @label = label
  end

  def build(_data_set, _weights)
    self
  end

  def eval(_data)
    @label
  end
end

# Learner that memorizes the training labels, for a zero error round.
class MemorizingLearner
  def build(data_set, _weights)
    @memory = data_set.data_items.to_h { |item| [item[0...-1], item.last] }
    self
  end

  def eval(data)
    @memory[data]
  end
end

class SammeTest < Minitest::Test
  include Ai4r::Classifiers
  include Ai4r::Data

  LABELS = %w[x kind].freeze
  ITEMS = [[0.0, 'A'], [1.0, 'A'], [2.0, 'B'], [3.0, 'B'],
           [4.0, 'C'], [5.0, 'C'], [6.0, 'A'], [7.0, 'A']].freeze
  LIGHT = 1.0 / 12
  HEAVY = 1.0 / 6
  WEIGHTS_AFTER_ODD_ROUND = [LIGHT, LIGHT, HEAVY, HEAVY, HEAVY, HEAVY, LIGHT, LIGHT].freeze
  WEIGHTS_AFTER_EVEN_ROUND = [HEAVY, HEAVY, LIGHT, LIGHT, LIGHT, LIGHT, HEAVY, HEAVY].freeze

  def build_samme(factory:, n_estimators: 4, seed: nil)
    Samme.new.set_parameters(n_estimators: n_estimators, learning_rate: 1.0,
                             learner_factory: factory, random_seed: seed)
         .build(DataSet.new(data_items: ITEMS, data_labels: LABELS))
  end

  def assert_weights(expected, actual)
    assert_equal expected.length, actual.length
    expected.zip(actual) { |e, a| assert_in_delta e, a, 1e-12 }
  end

  def test_round_records_track_error_alpha_classes_and_weights
    samme = build_samme(factory: -> { WeightedStump.new })
    assert_equal 4, samme.rounds.length
    samme.rounds.each { |round| assert_equal %w[A B C], round.classes }

    assert_in_delta 0.5, samme.rounds[0].error
    assert_in_delta Math.log(2), samme.rounds[0].alpha
    (1..3).each do |index|
      assert_in_delta 1.0 / 3, samme.rounds[index].error
      assert_in_delta 2 * Math.log(2), samme.rounds[index].alpha
    end

    assert_weights WEIGHTS_AFTER_ODD_ROUND, samme.rounds[0].sample_weights
    assert_weights WEIGHTS_AFTER_EVEN_ROUND, samme.rounds[1].sample_weights
    assert_weights WEIGHTS_AFTER_ODD_ROUND, samme.rounds[2].sample_weights
    assert_weights WEIGHTS_AFTER_EVEN_ROUND, samme.rounds[3].sample_weights
  end

  def test_eval_and_scores_honor_round_limit
    samme = build_samme(factory: -> { WeightedStump.new })
    assert_equal 'A', samme.eval([2.0], 1)
    assert_equal 'B', samme.eval([2.0], 2)
    assert_equal 'A', samme.eval([2.0], 3)
    assert_equal 'B', samme.eval([2.0])
    assert_equal 'C', samme.eval([5.0])

    scores = samme.scores([2.0], 2)
    assert_in_delta Math.log(2), scores['A']
    assert_in_delta 2 * Math.log(2), scores['B']
    assert_in_delta 0.0, scores['C']
  end

  def test_zero_error_records_single_max_alpha_round
    samme = build_samme(factory: -> { MemorizingLearner.new }, n_estimators: 10)
    assert_equal 1, samme.rounds.length
    assert_equal 0.0, samme.rounds[0].error
    assert_in_delta Samme::MAX_ALPHA, samme.rounds[0].alpha
    assert_equal 'B', samme.eval([2.0])
  end

  def test_random_guess_bound_keeps_first_learner_with_unit_alpha
    samme = build_samme(factory: -> { ConstantLearner.new('B') })
    assert_equal 1, samme.rounds.length
    assert_in_delta 0.75, samme.rounds[0].error
    assert_in_delta 1.0, samme.rounds[0].alpha
    assert_equal 'B', samme.eval([3.0])
  end

  def test_random_guess_bound_discards_later_rounds
    first = true
    factory = lambda do
      if first
        first = false
        WeightedStump.new
      else
        ConstantLearner.new('A')
      end
    end
    samme = build_samme(factory: factory, n_estimators: 10)
    assert_equal 1, samme.rounds.length
    assert_in_delta 0.5, samme.rounds[0].error
  end

  def test_single_class_trains_no_rounds
    data_set = DataSet.new(data_items: [[0.0, 'A'], [1.0, 'A']], data_labels: LABELS)
    samme = Samme.new.set_parameters(learner_factory: -> { WeightedStump.new }).build(data_set)
    assert_empty samme.rounds
    assert_equal 'A', samme.eval([0.5])
    assert_equal({ 'A' => 0.0 }, samme.scores([0.5]))
  end

  def test_weights_stay_normalized_on_long_runs
    samme = build_samme(factory: -> { WeightedStump.new }, n_estimators: 500)
    assert_equal 500, samme.rounds.length
    samme.rounds.each do |round|
      assert_in_delta 1.0, round.sample_weights.sum, 1e-9
      assert round.sample_weights.all?(&:positive?)
    end
  end

  def test_same_random_seed_reproduces_round_records
    factory = ->(rng) { WeightedStump.new(rng) }
    one = build_samme(factory: factory, n_estimators: 10, seed: 42)
    two = build_samme(factory: factory, n_estimators: 10, seed: 42)
    assert_equal one.rounds.map(&:alpha), two.rounds.map(&:alpha)
    assert_equal one.rounds.map(&:error), two.rounds.map(&:error)
    ITEMS.each do |item|
      assert_equal one.eval(item[0...-1]), two.eval(item[0...-1])
    end
  end

  def test_learner_factory_is_required
    data_set = DataSet.new(data_items: ITEMS, data_labels: LABELS)
    assert_raises(ArgumentError) { Samme.new.build(data_set) }
  end

  def test_get_rules
    samme = build_samme(factory: -> { WeightedStump.new })
    assert_equal 'Samme does not support rule extraction.', samme.get_rules
  end
end
