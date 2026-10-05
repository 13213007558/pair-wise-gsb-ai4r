# frozen_string_literal: true

require 'ai4r/classifiers/bayesian_probit'
require 'ai4r/data/data_set'
require_relative '../test_helper'

class BayesianProbitTest < Minitest::Test
  include Ai4r::Classifiers
  include Ai4r::Data

  ITEMS = [
    [-3.0, 0],
    [-2.0, 0],
    [1.0, 1],
    [2.0, 1]
  ].freeze

  def setup
    @data_set = DataSet.new(data_items: ITEMS)
    @classifier = BayesianProbit.new.build(@data_set)
  end

  def test_training_retains_mode_and_covariance
    assert @classifier.converged
    assert_equal 2, @classifier.posterior_mode.length
    assert_equal [2, 2], @classifier.posterior_covariance.map(&:length)
    assert_equal 2, @classifier.prior_mean.length
    assert_equal [2, 2], @classifier.prior_precision.map(&:length)
  end

  def test_latent_response_reports_mean_and_variance
    mean, variance = @classifier.latent_linear_response([1.0])
    expected_mean = @classifier.posterior_mode[0] + @classifier.posterior_mode[1]
    covariance = @classifier.posterior_covariance
    expected_variance = covariance[0][0] + (2.0 * covariance[0][1]) + covariance[1][1]

    assert_in_delta expected_mean, mean, 1e-12
    assert_in_delta expected_variance, variance, 1e-12
    assert variance.positive?
  end

  def test_predictive_probability_matches_one_dimensional_integral
    mean, variance = @classifier.latent_linear_response([1.0])
    standard_deviation = Math.sqrt(variance)
    numerical = gaussian_probit_integral(mean, standard_deviation)
    probability = @classifier.positive_class_probability([1.0])
    exact = normal_cdf(mean / Math.sqrt(1.0 + variance))
    mode_only = normal_cdf(mean)

    assert_in_delta numerical, probability, 1e-8
    assert_in_delta exact, probability, 2e-8
    refute_in_delta mode_only, probability, 1e-8
  end

  def test_class_and_probability_map
    assert_equal 1, @classifier.eval([2.0])
    assert_equal 0, @classifier.eval([-2.0])
    probabilities = @classifier.get_probability_map([2.0])

    assert_equal [0, 1], probabilities.keys
    assert_in_delta 1.0, probabilities.values.sum, 1e-12
    assert probabilities.fetch(1) > probabilities.fetch(0)
  end

  def test_repeated_observations_reduce_predictive_uncertainty
    base = BayesianProbit.new.build(@data_set)
    repeated_items = ITEMS.flat_map { |item| Array.new(8, item.dup) }
    repeated = BayesianProbit.new.build(DataSet.new(data_items: repeated_items))

    assert(repeated.warnings.any? { |warning| warning.include?('separated') })
    assert_operator repeated.latent_linear_response([1.0]).last, :<,
                    base.latent_linear_response([1.0]).last
    assert repeated.posterior_covariance[0][0] < base.posterior_covariance[0][0]
  end

  def test_complete_separation_is_explained
    assert(@classifier.warnings.any? { |warning| warning.include?('separated') })
    assert @classifier.optimization_status.include?('converged')
    assert @classifier.posterior_mode.map(&:finite?).all?
    assert @classifier.posterior_covariance.flatten.map(&:finite?).all?
  end

  def test_collinear_features_are_reported_but_still_predictable
    items = [[-2.0, -4.0, 0], [-1.0, -2.0, 0], [1.0, 2.0, 1], [2.0, 4.0, 1]]
    classifier = BayesianProbit.new.build(DataSet.new(data_items: items))

    assert(classifier.warnings.any? { |warning| warning.include?('rank deficient') })
    assert classifier.converged
    assert classifier.get_probability_map([2.0, 4.0]).fetch(1) > 0.5
    assert classifier.posterior_covariance.flatten.map(&:finite?).all?
  end

  def test_configurable_bias_prior_is_separate_from_slope_prior
    classifier = BayesianProbit.new
    classifier.set_parameters(
      slope_prior_mean: 0.5,
      slope_prior_variance: 2.0,
      bias_prior_mean: -0.25,
      bias_prior_variance: 4.0
    )
    classifier.build(@data_set)

    assert_in_delta 0.5, classifier.prior_mean.first, 1e-12
    assert_in_delta(-0.25, classifier.prior_mean.last, 1e-12)
    assert_in_delta 0.5, classifier.prior_precision.first.first, 1e-12
    assert_in_delta 0.25, classifier.prior_precision.last.last, 1e-12
  end

  def test_gaussian_prior_accepts_slope_means_and_covariance_matrix
    classifier = BayesianProbit.new
    classifier.set_parameters(
      slope_prior_mean: [0.1, -0.2],
      slope_prior_variance: [[2.0, 0.5], [0.5, 1.0]],
      bias_prior_variance: 5.0
    )
    items = [[-2.0, -1.0, 0], [-1.0, -0.5, 0], [1.0, 0.5, 1], [2.0, 1.0, 1]]
    classifier.build(DataSet.new(data_items: items))

    assert_equal [0.1, -0.2, 0.0], classifier.prior_mean
    assert_in_delta 4.0 / 7.0, classifier.prior_precision[0][0], 1e-12
    assert_in_delta(-2.0 / 7.0, classifier.prior_precision[0][1], 1e-12)
    assert_in_delta 0.2, classifier.prior_precision.last.last, 1e-12
    assert classifier.converged
  end

  def test_non_finite_prior_mean_fails_before_optimization
    classifier = BayesianProbit.new
    classifier.set_parameters(slope_prior_mean: Float::INFINITY)

    error = assert_raises(ArgumentError) { classifier.build(@data_set) }
    assert_match(/finite/, error.message)
    assert_equal 'not trained', classifier.optimization_status
  end

  def test_non_positive_definite_prior_covariance_fails_clearly
    classifier = BayesianProbit.new
    classifier.set_parameters(slope_prior_variance: [[1.0, 1.0], [1.0, 1.0]])
    items = [[-1.0, -1.0, 0], [0.0, 0.0, 0], [1.0, 1.0, 1], [2.0, 2.0, 1]]

    assert_raises(Ai4r::Classifiers::BayesianProbit::NumericFailure) do
      classifier.build(DataSet.new(data_items: items))
    end
    assert_equal 'not trained', classifier.optimization_status
  end

  def test_extreme_prediction_does_not_return_nan
    negative_probability = @classifier.positive_class_probability([-1.0e100])
    positive_probability = @classifier.positive_class_probability([1.0e100])
    assert negative_probability < 1.0e-4
    assert positive_probability > (1.0 - 1.0e-4)
    mean, variance = @classifier.latent_linear_response([1.0e100])
    assert [mean, variance].map(&:finite?).all?
  end

  def test_non_finite_and_non_numeric_inputs_fail_clearly
    assert_raises(ArgumentError) do
      @classifier.positive_class_probability([Float::NAN])
    end
    assert_raises(ArgumentError) do
      @classifier.positive_class_probability(['large'])
    end
  end

  def test_non_converged_model_reports_status_when_iterations_are_too_low
    classifier = BayesianProbit.new
    classifier.set_parameters(max_iterations: 1)
    classifier.build(@data_set)

    refute classifier.converged
    assert classifier.optimization_status.include?('maximum iterations')
  end

  def test_rules_are_unsupported
    assert_equal 'BayesianProbit does not support rule extraction.',
                 BayesianProbit.new.get_rules
  end

  private

  def normal_cdf(value)
    0.5 * Math.erfc(-value / Math.sqrt(2.0))
  end

  def gaussian_probit_integral(mean, standard_deviation)
    simpson = lambda do |lower, upper, steps|
      width = (upper - lower) / steps
      values = (0..steps).sum(0.0) do |i|
        point = lower + (i * width)
        weight = if i.zero? || i == steps
                   1.0
                 else
                   (i.even? ? 2.0 : 4.0)
                 end
        weight * normal_pdf(point) * normal_cdf(mean + (standard_deviation * point))
      end
      values * width / 3.0
    end
    simpson.call(-8.0, 8.0, 20_000)
  end

  def normal_pdf(value)
    Math.exp(-0.5 * value * value) / Math.sqrt(2.0 * Math::PI)
  end
end
