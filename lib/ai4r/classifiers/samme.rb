# frozen_string_literal: true

# Author::    OpenAI ChatGPT
# License::   MPL 1.1
# Project::   ai4r
#
# SAMME (Stagewise Additive Modeling using a Multiclass Exponential loss)
# implementation of AdaBoost for multi-class problems. Each round trains a
# single multi-class weak learner on weighted samples; the problem is never
# decomposed into one-vs-rest binary tasks.

require_relative '../data/data_set'
require_relative '../classifiers/classifier'

module Ai4r
  module Classifiers
    # SAMME multi-class AdaBoost.
    #
    # Weak learners are supplied through +learner_factory+, a callable invoked
    # once per round as +factory.call+ or +factory.call(rng)+ (when its arity
    # is not zero). The returned learner must respond to
    # +build(data_set, sample_weights)+ and +eval(data)+.
    #
    # Each trained round is recorded as a Round struct holding the learner,
    # its weight (+alpha+), the weighted training +error+, the +classes+
    # mapping (class index => label) and the +sample_weights+ produced by the
    # weight update of that round (input weights of the next one).
    #
    # Explicit edge-case behavior:
    # * Zero weighted error: the round is recorded with +MAX_ALPHA+ and
    #   training stops, since the learner already fits the data perfectly.
    # * Error at or above the random guessing bound (1 - 1/K): training
    #   stops. If no round was recorded yet, the learner is kept as the only
    #   round with alpha 1.0 so the ensemble can still predict.
    # * A single class in the training data: no rounds are trained;
    #   +eval+ always returns that class and +scores+ maps it to 0.0.
    #
    # Sample weights are renormalized to sum 1 after every round, and fall
    # back to a uniform distribution if numerical issues would otherwise
    # produce a zero or non-finite total, so weights never underflow to all
    # zeros on long runs.
    class Samme < Classifier
      # Alpha assigned to a learner with zero weighted error.
      MAX_ALPHA = 10.0

      # Tolerance used when comparing the weighted error against the random
      # guessing bound, so floating point noise does not change the decision.
      RANDOM_GUESS_TOLERANCE = 1e-12

      # Record of a single boosting round.
      Round = Struct.new(:learner, :alpha, :error, :classes, :sample_weights, keyword_init: true)

      parameters_info n_estimators: 'Number of boosting rounds. Default 10.',
                      learning_rate: 'Shrinkage applied to each learner weight. Default 1.0.',
                      learner_factory: 'Callable returning a fresh weak learner per round. Called with an RNG when its arity is not zero.',
                      random_seed: 'Seed for the RNG handed to the learner factory.'

      attr_reader :rounds, :classes

      def initialize
        super()
        @n_estimators = 10
        @learning_rate = 1.0
        @learner_factory = nil
        @random_seed = nil
      end

      def build(data_set)
        data_set.check_not_empty
        raise ArgumentError, 'learner_factory is required' unless @learner_factory

        @classes = data_set.data_items.map(&:last).uniq
        @rounds = []
        return self if @classes.length < 2

        rng = @random_seed ? Random.new(@random_seed) : Random.new
        weights = Array.new(data_set.data_items.length) { 1.0 / data_set.data_items.length }
        @n_estimators.times do
          learner = new_learner(rng)
          learner.build(data_set, weights)
          predictions = data_set.data_items.map { |item| learner.eval(item[0...-1]) }
          error = weighted_error(data_set.data_items, predictions, weights)
          weights = record_round(learner, error, weights, data_set.data_items, predictions)
          break if weights.nil?
        end
        self
      end

      # Predict the class for data using the first +n_rounds+ learners
      # (all of them by default). Ties are broken by class order.
      def eval(data, n_rounds = nil)
        best = @classes.first
        best_score = -Float::INFINITY
        scores(data, n_rounds).each do |label, score|
          next unless score > best_score

          best = label
          best_score = score
        end
        best
      end

      # SAMME scores per class label for data, using the first +n_rounds+
      # learners (all of them by default). The score of a class is the sum of
      # the alphas of the learners voting for it.
      def scores(data, n_rounds = nil)
        result = @classes.to_h { |label| [label, 0.0] }
        used = n_rounds ? @rounds.first(n_rounds) : @rounds
        used.each do |round|
          prediction = round.learner.eval(data)
          result[prediction] += round.alpha if result.key?(prediction)
        end
        result
      end

      def get_rules
        'Samme does not support rule extraction.'
      end

      private

      def new_learner(rng)
        @learner_factory.arity.zero? ? @learner_factory.call : @learner_factory.call(rng)
      end

      def weighted_error(items, predictions, weights)
        error = 0.0
        items.each_with_index do |item, index|
          error += weights[index] if predictions[index] != item.last
        end
        error
      end

      # Records the round and returns the sample weights for the next round,
      # or nil when boosting must stop.
      def record_round(learner, error, weights, items, predictions)
        if error <= 0.0
          add_round(learner, MAX_ALPHA * @learning_rate, 0.0, weights)
          return nil
        end
        if error >= (1.0 - (1.0 / @classes.length)) - RANDOM_GUESS_TOLERANCE
          add_round(learner, 1.0, error, weights) if @rounds.empty?
          return nil
        end
        alpha = @learning_rate * (Math.log((1.0 - error) / error) + Math.log(@classes.length - 1))
        updated = update_weights(weights, items, predictions, alpha)
        add_round(learner, alpha, error, updated)
        updated
      end

      def add_round(learner, alpha, error, sample_weights)
        @rounds << Round.new(learner: learner, alpha: alpha, error: error,
                             classes: @classes.dup, sample_weights: sample_weights.dup)
      end

      def update_weights(weights, items, predictions, alpha)
        factor = Math.exp(alpha)
        updated = weights.each_with_index.map do |weight, index|
          predictions[index] == items[index].last ? weight : weight * factor
        end
        normalize(updated)
      end

      def normalize(weights)
        total = weights.sum
        return Array.new(weights.length) { 1.0 / weights.length } unless total.positive? && total.finite?

        weights.map { |weight| weight / total }
      end
    end
  end
end
