# frozen_string_literal: true

require 'ai4r/classifiers/bayesian_probit'
require 'ai4r/data/data_set'
require_relative '../test_helper'

class BayesianProbitTest < Minitest::Test
  include Ai4r::Classifiers
  include Ai4r::Data

  DATA_LABELS = %w[score alert].freeze
  DATA_ITEMS = [
    [-2.0, 0],
    [-1.0, 0],
    [-0.5, 0],
    [0.4, 1],
    [1.0, 1],
    [2.0, 1]
  ].freeze

  def setup
    @classifier = build_classifier(DATA_ITEMS)
  end

  def test_build_converges_and_reports_status
    assert @classifier.converged?
    assert_nil @classifier.failure_reason
    assert @classifier.iterations >= 1
  end

  def test_eval
    assert_equal 0, @classifier.eval([-1.5])
    assert_equal 1, @classifier.eval([1.5])
  end

  def test_predict_proba_is_monotone_and_in_unit_interval
    probabilities = [-3.0, -1.0, 0.0, 1.0, 3.0].map { |x| @classifier.predict_proba([x]) }
    probabilities.each do |p|
      assert p.finite?, 'probability must be finite'
      assert p >= 0.0 && p <= 1.0, "probability #{p} out of [0, 1]"
    end
    probabilities.each_cons(2) { |a, b| assert b > a, 'probability should increase with the score' }
  end

  def test_latent_response_returns_mean_and_variance
    mean, variance = @classifier.latent_response([0.8])
    assert mean.finite?
    assert variance.finite?
    assert variance > 0.0, 'posterior latent variance must be positive'
  end

  def test_posterior_mode_and_covariance_shapes
    mode = @classifier.posterior_mode
    covariance = @classifier.posterior_covariance
    assert_equal 2, mode.length # one weight + bias
    assert_equal 2, covariance.length
    covariance.each { |row| assert_equal 2, row.length }
    assert_in_delta covariance[0][1], covariance[1][0], 1e-12
    assert covariance[0][0] > 0.0
    assert covariance[1][1] > 0.0
  end

  def test_predictive_probability_integrates_parameter_uncertainty
    # With parameter uncertainty the predictive probability is pulled towards
    # 0.5 compared to plugging the posterior mode into the probit.
    mean, variance = @classifier.latent_response([3.0])
    assert variance > 0.0
    plug_in = normal_cdf(mean)
    predictive = @classifier.predict_proba([3.0])
    assert predictive < plug_in
    assert predictive > 0.5
  end

  def test_predictive_probability_matches_numerical_integration
    # Reference check: brute force 2D quadrature of Phi(w*x + b) over the
    # bivariate Gaussian posterior of [w, b] must match the closed form
    # predictive probability Phi(a / sqrt(1 + v)).
    x = 0.8
    mode = @classifier.posterior_mode
    covariance = @classifier.posterior_covariance
    mw, mb = mode
    sw = Math.sqrt(covariance[0][0])
    sb = Math.sqrt(covariance[1][1])
    rho = covariance[0][1] / (sw * sb)
    det = 1.0 - (rho * rho)
    grid = 160
    dw = 12.0 * sw / grid
    db = 12.0 * sb / grid
    total = 0.0
    grid.times do |i|
      w = mw - (6.0 * sw) + ((i + 0.5) * dw)
      zw = (w - mw) / sw
      grid.times do |j|
        b = mb - (6.0 * sb) + ((j + 0.5) * db)
        zb = (b - mb) / sb
        exponent = -((zw * zw) - (2.0 * rho * zw * zb) + (zb * zb)) / (2.0 * det)
        density = Math.exp(exponent) / (2.0 * Math::PI * sw * sb * Math.sqrt(det))
        total += normal_cdf((w * x) + b) * density * dw * db
      end
    end
    assert_in_delta @classifier.predict_proba([x]), total, 1e-3
  end

  def test_repeated_observations_shrink_latent_variance
    base = [[-2.0, 0], [-1.0, 0], [1.0, 1], [2.0, 1]]
    variances = [1, 2, 4, 8].map do |copies|
      build_classifier(base * copies, prior_variance: 1.0).latent_response([0.5]).last
    end
    variances.each_cons(2) do |a, b|
      assert b < a, "latent variance should shrink as data accumulates: #{variances.inspect}"
    end
    assert variances.last < variances.first / 2.0,
           'repeating the batch 8 times should reduce variance well below half'
  end

  def test_complete_separation_is_handled
    items = [[-3.0, 0], [-2.0, 0], [-1.0, 0], [1.0, 1], [2.0, 1], [3.0, 1]]
    classifier = build_classifier(items)
    assert classifier.converged?
    high = classifier.predict_proba([5.0])
    low = classifier.predict_proba([-5.0])
    assert high.finite? && high > 0.5 && high <= 1.0
    assert low.finite? && low < 0.5 && low >= 0.0
  end

  def test_collinear_features_are_handled
    items = [[-2.0, -4.0, 0], [-1.0, -2.0, 0], [1.0, 2.0, 1], [2.0, 4.0, 1]]
    classifier = build_classifier(items)
    assert classifier.converged?
    probability = classifier.predict_proba([1.5, 3.0])
    assert probability.finite?
    assert probability > 0.5
    classifier.posterior_covariance.flatten.each { |v| assert v.finite? }
  end

  def test_extreme_inputs_do_not_produce_nan
    [1e6, -1e6, 1e8, -1e8].each do |value|
      probability = @classifier.predict_proba([value])
      assert probability.finite?, "expected finite probability for #{value}"
      assert probability >= 0.0 && probability <= 1.0
      mean, variance = @classifier.latent_response([value])
      assert mean.finite?
      assert variance.finite?
    end
  end

  def test_extreme_training_values_do_not_produce_nan
    items = [[-1.0, 0], [-2.0, 0], [1.0, 1], [1e6, 1]]
    classifier = build_classifier(items)
    probability = classifier.predict_proba([2.0])
    assert probability.finite?
    assert probability >= 0.0 && probability <= 1.0
  end

  def test_invalid_prior_variance_raises
    [-1.0, 0.0, Float::INFINITY, Float::NAN].each do |bad|
      classifier = BayesianProbit.new
      classifier.set_parameters(prior_variance: bad)
      assert_raises(ArgumentError) { classifier.build(data_set(DATA_ITEMS)) }
    end
  end

  def test_bias_prior_is_explicit_and_configurable
    strong = build_classifier(DATA_ITEMS, bias_prior_mean: 5.0, bias_prior_variance: 0.01)
    assert (strong.posterior_mode.last - 5.0).abs < 1.0,
           'a strong bias prior should pull the bias mode towards its mean'
    weak = build_classifier(DATA_ITEMS)
    assert weak.posterior_mode.last.abs < 2.0
  end

  def test_exhausted_iterations_report_failure
    classifier = BayesianProbit.new
    classifier.set_parameters(max_iterations: 0)
    classifier.build(data_set(DATA_ITEMS))
    refute classifier.converged?
    assert_match(/converge/, classifier.failure_reason)
    assert_raises(RuntimeError) { classifier.predict_proba([0.5]) }
  end

  def test_eval_dimension_mismatch
    assert_raises(ArgumentError) { @classifier.eval([0.0, 1.0]) }
  end

  def test_non_finite_input_raises
    assert_raises(ArgumentError) { @classifier.eval([Float::NAN]) }
    assert_raises(ArgumentError) { @classifier.eval([Float::INFINITY]) }
  end

  def test_not_trained_raises
    classifier = BayesianProbit.new
    assert_raises(RuntimeError) { classifier.eval([0.5]) }
  end

  def test_get_rules
    assert_equal 'BayesianProbit does not support rule extraction.', @classifier.get_rules
  end

  private

  def build_classifier(items, parameters = {})
    classifier = BayesianProbit.new
    classifier.set_parameters(parameters) unless parameters.empty?
    classifier.build(data_set(items))
    classifier
  end

  def data_set(items)
    labels = items.first.length == 2 ? DATA_LABELS : %w[x1 x2 alert]
    DataSet.new(data_items: items, data_labels: labels)
  end

  def normal_cdf(z)
    0.5 * (1.0 + Math.erf(z / Math.sqrt(2.0)))
  end
end
