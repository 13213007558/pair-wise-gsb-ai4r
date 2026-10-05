# frozen_string_literal: true

# Author::    OpenAI Assistant
# License::   MPL 1.1
# Project::   ai4r
# Url::       https://github.com/SergioFierens/ai4r
#
# You can redistribute it and/or modify it under the terms of
# the Mozilla Public License version 1.1  as published by the
# Mozilla Foundation at http://www.mozilla.org/MPL/MPL-1.1.txt

require_relative '../data/data_set'
require_relative 'classifier'

module Ai4r
  module Classifiers
    # Bayesian probit classifier with a Gaussian prior on the parameters.
    #
    # The model assumes P(y=1 | x) = Phi(w.x + b), where Phi is the standard
    # normal CDF. Training finds the posterior mode (MAP) with Newton's
    # method and keeps a Laplace (Gaussian) approximation of the posterior:
    # the mode and the inverse of the posterior precision matrix as the
    # covariance approximation.
    #
    # The prior is a Gaussian configured through +prior_mean+ /
    # +prior_variance+ for the feature weights, and through
    # +bias_prior_mean+ / +bias_prior_variance+ for the bias (intercept)
    # term. The bias has its own, explicit prior: it is appended as the last
    # parameter and is regularized independently of the feature weights.
    # Both variances must be positive and finite; a proper prior keeps the
    # posterior precision positive definite even under complete separation
    # or exactly collinear features.
    #
    # Predictions integrate parameter uncertainty instead of plugging the
    # mode back into the probit: for a Gaussian posterior N(m, S) the
    # predictive probability is Phi(a / sqrt(1 + v)) with latent mean
    # a = m.x and latent variance v = x'Sx. Use +latent_response+ to
    # inspect a and v directly.
    #
    # Training reports its outcome through +converged?+, +iterations+ and
    # +failure_reason+; numerical problems (singular precision matrix,
    # failed line search, exhausted iterations) set an explicit
    # +failure_reason+ instead of silently returning NaN.
    #
    # Example:
    #   data = Ai4r::Data::DataSet.new(:data_items => [[0.2, 1], [0.4, 0]])
    #   classifier = BayesianProbit.new.build(data)
    #   classifier.predict_proba([0.3])   # => 0.5..1.0
    #   classifier.latent_response([0.3]) # => [mean, variance]
    class BayesianProbit < Classifier
      attr_reader :posterior_mode, :posterior_covariance, :iterations, :failure_reason

      parameters_info prior_mean: 'Mean of the Gaussian prior on the feature weights.',
                      prior_variance: 'Variance of the Gaussian prior on the feature weights. Must be positive and finite.',
                      bias_prior_mean: 'Mean of the Gaussian prior on the bias (intercept) term. The bias has its own prior, independent of the weights.',
                      bias_prior_variance: 'Variance of the Gaussian prior on the bias term. Must be positive and finite.',
                      max_iterations: 'Maximum number of Newton iterations.',
                      tolerance: 'Convergence tolerance on the gradient infinity norm.'

      LOG_SQRT_2PI = 0.5 * Math.log(2.0 * Math::PI)
      SQRT2 = Math.sqrt(2.0)
      # Below this z, Phi(z) is computed through an asymptotic expansion of
      # the Mills ratio to avoid floating point underflow.
      MILLS_ASYMPTOTIC_THRESHOLD = -30.0

      def initialize
        super()
        @prior_mean = 0.0
        @prior_variance = 10.0
        @bias_prior_mean = 0.0
        @bias_prior_variance = 10.0
        @max_iterations = 100
        @tolerance = 1e-8
        @posterior_mode = nil
        @posterior_covariance = nil
        @iterations = 0
        @failure_reason = nil
        @n_features = 0
      end

      # Train the classifier and keep the Laplace posterior approximation.
      # Check +converged?+ and +failure_reason+ for the optimization status.
      def build(data_set)
        raise 'Error instance must be passed' unless data_set.is_a?(Ai4r::Data::DataSet)

        data_set.check_not_empty
        validate_prior_parameters

        x, y = extract_training_data(data_set)
        @n_features = x.first.length - 1
        prior_mean_vec, prior_precision = build_prior

        @posterior_mode = nil
        @posterior_covariance = nil
        @failure_reason = nil
        @iterations = 0

        theta = prior_mean_vec.dup
        @max_iterations.times do
          @iterations += 1
          gradient, precision = gradient_and_precision(theta, x, y, prior_mean_vec, prior_precision)
          if gradient.map(&:abs).max < @tolerance
            @posterior_mode = theta
            @posterior_covariance = invert_spd(precision)
            unless @posterior_covariance
              @failure_reason = 'posterior precision matrix is not positive definite at the mode'
              @posterior_mode = nil
            end
            return self
          end
          delta = cholesky_solve(precision, gradient)
          unless delta
            @failure_reason = 'posterior precision matrix is not positive definite (check prior variances and feature collinearity)'
            return self
          end
          theta, ok = line_search(theta, delta, x, y, prior_mean_vec, prior_precision)
          unless ok
            @failure_reason = 'line search failed to increase the log posterior (numerical failure)'
            return self
          end
        end
        @failure_reason = "did not converge within #{@max_iterations} iterations"
        self
      end

      # True when training converged and a posterior approximation is held.
      def converged?
        !@posterior_mode.nil? && @failure_reason.nil?
      end

      # Predict the class (0 or 1) using the uncertainty-aware probability.
      def eval(data)
        predict_proba(data) >= 0.5 ? 1 : 0
      end

      # Predictive P(y=1 | x) integrating over the Gaussian posterior of the
      # parameters: Phi(a / sqrt(1 + v)) with latent mean a and variance v.
      def predict_proba(data)
        mean, variance = latent_response(data)
        # Scale by the input magnitude so extreme inputs cannot overflow
        # mean^2 or variance before the ratio is formed.
        scale = [augment(validate_instance(data)).map(&:abs).max, 1.0].max
        scaled_mean = mean / scale
        scaled_variance = variance / (scale * scale)
        normal_cdf(scaled_mean / Math.sqrt((1.0 / (scale * scale)) + scaled_variance))
      end

      # Mean and variance of the latent linear response w.x + b under the
      # Gaussian posterior approximation.
      def latent_response(data)
        ensure_trained!
        x = augment(validate_instance(data))
        mean = dot(@posterior_mode, x)
        variance = quadratic_form(@posterior_covariance, x)
        [mean, variance]
      end

      # BayesianProbit does not support rule extraction.
      def get_rules
        'BayesianProbit does not support rule extraction.'
      end

      private

      def validate_prior_parameters
        { prior_variance: @prior_variance, bias_prior_variance: @bias_prior_variance }.each do |name, value|
          next if value.is_a?(Numeric) && value.positive? && value.finite?

          raise ArgumentError,
                "#{name} must be a positive finite number, got #{value.inspect}; a proper Gaussian prior is required"
        end
      end

      def extract_training_data(data_set)
        x = data_set.data_items.map do |item|
          row = item[0...-1].map { |value| finite_float(value, 'feature') }
          row << 1.0 # bias term
          row
        end
        y = data_set.data_items.map do |item|
          label = item.last
          raise ArgumentError, "class labels must be 0 or 1, got #{label.inspect}" unless [0, 1, 0.0, 1.0].include?(label)

          label.to_i
        end
        [x, y]
      end

      def finite_float(value, role)
        f = Float(value)
        raise ArgumentError, "#{role} values must be finite numbers, got #{value.inspect}" unless f.finite?

        f
      rescue TypeError, ArgumentError => e
        raise ArgumentError, "#{role} values must be finite numbers, got #{value.inspect}" if e.is_a?(TypeError)

        raise
      end

      def validate_instance(data)
        expected = @n_features
        if data.length != expected
          raise ArgumentError,
                "Wrong number of inputs. Expected: #{expected}, received: #{data.length}."
        end

        data.map { |value| finite_float(value, 'input') }
      end

      def augment(row)
        row + [1.0]
      end

      def build_prior
        prior_mean_vec = Array.new(@n_features, @prior_mean.to_f) + [@bias_prior_mean.to_f]
        prior_precision = Array.new(@n_features, 1.0 / @prior_variance) + [1.0 / @bias_prior_variance]
        [prior_mean_vec, prior_precision]
      end

      def gradient_and_precision(theta, x, y, prior_mean_vec, prior_precision)
        dim = theta.length
        gradient = Array.new(dim) { |j| -prior_precision[j] * (theta[j] - prior_mean_vec[j]) }
        precision = Array.new(dim) do |j|
          row = Array.new(dim, 0.0)
          row[j] = prior_precision[j]
          row
        end
        x.each_with_index do |row, i|
          z = dot(theta, row)
          residual = y[i] == 1 ? mills_ratio(z) : -mills_ratio(-z)
          curvature = mills_ratio(z) * mills_ratio(-z)
          dim.times do |j|
            gradient[j] += residual * row[j]
            scaled = curvature * row[j]
            dim.times { |k| precision[j][k] += scaled * row[k] }
          end
        end
        [gradient, precision]
      end

      def log_posterior(theta, x, y, prior_mean_vec, prior_precision)
        value = 0.0
        x.each_with_index do |row, i|
          z = dot(theta, row)
          value += y[i] == 1 ? log_normal_cdf(z) : log_normal_cdf(-z)
        end
        theta.each_with_index do |t, j|
          diff = t - prior_mean_vec[j]
          value -= 0.5 * prior_precision[j] * diff * diff
        end
        value
      end

      def line_search(theta, delta, x, y, prior_mean_vec, prior_precision)
        log_post = log_posterior(theta, x, y, prior_mean_vec, prior_precision)
        step = 1.0
        # Tolerate decreases below floating point noise: near the mode the
        # improvement per Newton step falls under the rounding error of the
        # log posterior itself, and requiring a strict increase never ends.
        noise = 1e-12 * (1.0 + log_post.abs)
        25.times do
          candidate = theta.zip(delta).map { |t, d| t + (step * d) }
          new_log_post = log_posterior(candidate, x, y, prior_mean_vec, prior_precision)
          return [candidate, true] if new_log_post.finite? && new_log_post >= log_post - noise

          step *= 0.5
        end
        [theta, false]
      end

      def ensure_trained!
        raise "Training failed: #{@failure_reason}" if @failure_reason
        raise 'Model not trained' unless @posterior_mode
      end

      def dot(a, b)
        sum = 0.0
        a.each_with_index { |v, i| sum += v * b[i] }
        sum
      end

      def quadratic_form(matrix, x)
        sum = 0.0
        x.each_with_index do |xi, i|
          row_sum = 0.0
          matrix[i].each_with_index { |mij, j| row_sum += mij * x[j] }
          sum += xi * row_sum
        end
        sum
      end

      # Standard normal PDF. Underflows to 0.0 for |z| > ~38, callers that
      # divide by it use the asymptotic Mills ratio instead.
      def normal_pdf(z)
        Math.exp(-0.5 * z * z - LOG_SQRT_2PI)
      end

      # Standard normal CDF. Computed through erfc so that the left tail
      # stays accurate down to z ~ -38 (erf saturates at |z| ~ 8.3 and
      # would return exactly 0.0 or 1.0, producing Inf/NaN downstream).
      def normal_cdf(z)
        0.5 * Math.erfc(-z / SQRT2)
      end

      def log_normal_cdf(z)
        return Math.log(normal_cdf(z)) if z >= MILLS_ASYMPTOTIC_THRESHOLD

        (-0.5 * z * z - LOG_SQRT_2PI) - Math.log(mills_ratio(z))
      end

      # Mills ratio phi(z) / Phi(z), stable for every finite z.
      def mills_ratio(z)
        return normal_pdf(z) / normal_cdf(z) if z >= MILLS_ASYMPTOTIC_THRESHOLD

        # Asymptotic expansion as z -> -infinity:
        # phi(z)/Phi(z) = -z - 1/z + 2/z^3 + O(z^-5)
        -z - (1.0 / z) + (2.0 / (z * z * z))
      end

      # Solve matrix * x = rhs for a symmetric positive definite matrix via
      # Cholesky decomposition. Returns nil when the matrix is not PD.
      def cholesky_solve(matrix, rhs)
        n = matrix.length
        l = Array.new(n) { Array.new(n, 0.0) }
        n.times do |i|
          (i + 1).times do |j|
            sum = matrix[i][j]
            j.times { |k| sum -= l[i][k] * l[j][k] }
            if i == j
              return nil unless sum.positive? && sum.finite?

              l[i][j] = Math.sqrt(sum)
            else
              l[i][j] = sum / l[j][j]
            end
          end
        end
        v = Array.new(n, 0.0)
        n.times do |i|
          sum = rhs[i]
          i.times { |k| sum -= l[i][k] * v[k] }
          v[i] = sum / l[i][i]
        end
        solution = Array.new(n, 0.0)
        (n - 1).downto(0) do |i|
          sum = v[i]
          (i + 1...n).each { |k| sum -= l[k][i] * solution[k] }
          solution[i] = sum / l[i][i]
        end
        solution
      end

      # Invert a symmetric positive definite matrix. Returns nil when the
      # matrix is not positive definite.
      def invert_spd(matrix)
        n = matrix.length
        columns = (0...n).map do |j|
          unit = Array.new(n, 0.0)
          unit[j] = 1.0
          column = cholesky_solve(matrix, unit)
          return nil unless column

          column
        end
        columns.transpose
      end
    end
  end
end
