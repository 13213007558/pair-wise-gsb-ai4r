# frozen_string_literal: true

require_relative '../data/data_set'
require_relative 'classifier'

module Ai4r
  module Classifiers
    # SAMME discrete multiclass boosting (Stagewise Additive Modeling using a
    # Multi-class Exponential loss). Each weak learner is trained once on the
    # complete multiclass problem; labels are never split into binary problems.
    class SAMMEBoost < Classifier
      Round = Struct.new(
        :iteration,
        :classes,
        :class_mapping,
        :learner,
        :learner_weight,
        :weighted_error,
        :sample_weights,
        :updated_sample_weights,
        :status
      )

      parameters_info n_estimators: 'Maximum number of boosting rounds. Default 10.',
                      random_seed: 'Integer seed or Random instance passed to the weak learner factory.',
                      weak_learner_factory: 'Callable returning a weak learner that responds to build(data_set, sample_weights) and eval(data).'

      attr_reader :classes, :rounds

      def initialize
        super()
        @n_estimators = 10
        @random_seed = nil
        @weak_learner_factory = nil
        @classes = []
        @rounds = []
      end

      def build(data_set)
        data_set.check_not_empty
        raise ArgumentError, 'A weak learner factory is required.' unless @weak_learner_factory.respond_to?(:call)
        raise ArgumentError, 'n_estimators must be a positive integer.' unless valid_estimator_count?

        labels = data_set.data_items.map(&:last)
        @classes = labels.uniq
        raise ArgumentError, 'SAMME requires at least two classes.' if @classes.length < 2

        class_count = @classes.length
        class_mapping = @classes.each_with_index.to_h
        rng = random_generator
        log_weights = Array.new(labels.length) { -Math.log(labels.length) }
        @rounds = []

        @n_estimators.times do |index|
          sample_weights = weights_from_log_weights(log_weights)
          learner = build_weak_learner(data_set, sample_weights, rng)
          predictions = data_set.data_items.map { |item| learner.eval(item[0...-1]) }
          validate_predictions!(predictions)

          weighted_error = labels.zip(predictions, sample_weights).sum do |label, prediction, weight|
            label == prediction ? 0.0 : weight
          end
          threshold = (class_count - 1).to_f / class_count
          learner_weight = samme_weight(weighted_error, class_count)
          status = round_status(weighted_error, threshold)

          updated_log_weights = log_weights.dup
          update_log_weights!(updated_log_weights, labels, predictions, learner_weight) if status == :added
          updated_weights = status == :added ? weights_from_log_weights(updated_log_weights) : sample_weights.dup

          @rounds << Round.new(
            index + 1,
            @classes.dup,
            class_mapping.dup,
            learner,
            learner_weight,
            weighted_error,
            sample_weights,
            updated_weights,
            status
          )

          break unless status == :added

          log_weights = updated_log_weights
        end

        self
      end

      def eval(data, round: nil)
        class_scores(data, round: round).max_by { |label, score| [score, -@classes.index(label)] }.first
      end

      def class_scores(data, round: nil)
        selected_rounds = accepted_rounds
        if round
          raise ArgumentError, 'round must be a positive integer.' unless valid_round?(round)
          raise ArgumentError, "round #{round} exceeds #{@rounds.length} trained round(s)." if round > @rounds.length

          selected_rounds = selected_rounds.select { |record| record.iteration <= round }
        end

        raise 'No weak learner was accepted during SAMME training.' if selected_rounds.empty?

        scores = @classes.to_h { |label| [label, 0.0] }
        selected_rounds.each do |record|
          prediction = record.learner.eval(data)
          raise ArgumentError, "Weak learner returned unknown class #{prediction.inspect}." unless scores.key?(prediction)

          scores[prediction] += record.learner_weight
        end
        scores
      end

      def accepted_rounds
        @rounds.select { |record| %i[added zero_error].include?(record.status) }
      end

      def get_rules
        'SAMMEBoost does not support rule extraction.'
      end

      private

      def valid_estimator_count?
        @n_estimators.is_a?(Integer) && @n_estimators.positive?
      end

      def valid_round?(round)
        round.is_a?(Integer) && round.positive?
      end

      def random_generator
        return @random_seed.dup if @random_seed.is_a?(Random)

        Random.new(@random_seed || Random.new_seed)
      end

      def build_weak_learner(data_set, sample_weights, rng)
        learner = if @weak_learner_factory.arity == 1
                    @weak_learner_factory.call(rng)
                  else
                    @weak_learner_factory.call
                  end

        raise ArgumentError, 'Weak learners must respond to build and eval.' unless learner.respond_to?(:build) && learner.respond_to?(:eval)

        built_learner = learner.build(data_set, sample_weights)
        raise ArgumentError, 'Weak learner build must return an object that responds to eval.' unless built_learner.respond_to?(:eval)

        built_learner
      end

      def validate_predictions!(predictions)
        unknown = predictions.reject { |prediction| @classes.include?(prediction) }.uniq
        return if unknown.empty?

        raise ArgumentError, "Weak learner returned unknown class(es): #{unknown.map(&:inspect).join(', ')}."
      end

      def samme_weight(weighted_error, class_count)
        return Float::INFINITY if weighted_error.zero?
        # rubocop:disable Lint/FloatComparison
        return 0.0 if weighted_error == ((class_count - 1).to_f / class_count)
        # rubocop:enable Lint/FloatComparison
        return nil if weighted_error > ((class_count - 1).to_f / class_count)

        Math.log((1.0 - weighted_error) / weighted_error) + Math.log(class_count - 1)
      end

      def round_status(weighted_error, threshold)
        return :zero_error if weighted_error.zero?
        return :random_guess if weighted_error == threshold
        return :worse_than_random if weighted_error > threshold

        :added
      end

      def update_log_weights!(log_weights, labels, predictions, learner_weight)
        labels.each_with_index do |label, index|
          next if label == predictions[index]

          log_weights[index] -= learner_weight
        end
      end

      def weights_from_log_weights(log_weights)
        maximum_log_weight = log_weights.max
        unnormalized = log_weights.map do |log_weight|
          Math.exp(log_weight - maximum_log_weight)
        end
        total = unnormalized.sum
        unnormalized.map { |weight| weight / total }
      end
    end

    SAMMEClassifier = SAMMEBoost
  end
end
